"""Grounded federated RAG study (proposal revision 4): build, prepare, run, check, analyze.

build           article-disjoint SQuAD splits, targets, probes, libraries and F, with
                every earlier target question excluded by recomputed old record hashes
calibrate-gate  CPU calibration of the verbatim-overlap library gate on V
prepare KIND    a frozen launch: datastore (P0 check), control (positive control),
                timing (full paired jobs), pilot (Stage 1). Nothing is resolved or run.
resolve         CPU checks before any GPU work: targets, worlds, tokens, exclusions
run             one job at a time in its own process; weights retire after verification
job             execute one job (called by run)
check           integrity and timing of completed jobs; never an endpoint value
analyze         the fixed validation rules or the §9 analysis

See outputs/retrieval_grounded_fl_proposal_20260930/{proposal.md,stage01_protocol.json,status.md}.
Attack effectiveness is measured here; no defense or privacy claim is made.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from master_script.core import grounded
from master_script.core.checkpoint_retention import file_digest, retire_checkpoints
from master_script.core.config import implementation_fingerprint
from master_script.core.defenses import privacy_bound
from master_script.core.grounded_job import CAUSAL, JOB_SCHEMA, RESULT_SCHEMA, validate_job
from master_script.core.queue import write_json
from master_script.tools.build_guard_stage import ROOT
from master_script.tools.causal_attack_validation import BASE as CAUSAL_BASE, DEFAULTS as CAUSAL_DEFAULTS
from master_script.tools.collect_guard_traces import SMOKE_TARGET, require_headroom

PROTOCOL = ROOT.parents[1] / "outputs/retrieval_grounded_fl_proposal_20260930/stage01_protocol.json"
STUDY = "grounded_fl"
LAUNCH_SCHEMA, COHORT_SCHEMA = "grounded_launch_v1", "grounded_cohort_v1"
KINDS = ("datastore", "control", "timing", "pilot")
V_SEED_BAND, FINAL_SEED_BAND = (8000, 8099), (9000, 9099)
PUBLIC_SEEDS = (8095, 8096, 8097, 8098, 8099)
DATASTORE_TARGETS, CONTROL_TARGETS, PILOT_TARGETS, BRIDGE_TARGETS = 60, 12, 4, 4
CONTROL_MIN_TOKENS, CONTROL_LEARNED, CONTROL_LR = 64, 10, 1e-4
SIGMA = {"inf": 0.0, 64: 1.91, 16: 5.45}
DELTA, CLIP = 1e-5, 1.0
BOOTSTRAP, RNG_SEED = 10_000, 20260930
MARGIN = 0.05
DATASTORE_LB, RETENTION_LB, CONTROL_AUC_LB = 0.70, 0.05, 0.60
NQ_SCORERS = ("f1", "entailment")
FORMS = ("record", "template")
# Minimum exclusions: guard v3/v4 development and reserved finals, causal Stage A/B, matched controls.
REQUIRED_ROLES = {"final": 4, "attacker_tuning": 4, "attacker_evaluation": 10, "matched_control_evaluation": 20}
MIN_EXCLUDED = 50
CONFIG = {**{k: v for k, v in CAUSAL_DEFAULTS.items() if k not in ("calibration_nonmember_count", "calibration_fpr")},
          **CAUSAL_BASE, **CAUSAL, "threshold_mode": "fixed", "max_length": grounded.MAX_LENGTH}
RAG = {"top_k": 4, "max_context_tokens": 768, "max_new_tokens": 64, "prompt_format": "chat", "significance": 0.05,
       "embedding_model": grounded.EMBEDDING["model"], "embedding_revision": grounded.EMBEDDING["revision"]}
FULL = {"reference": True, "retention": True, "utility": True, "nq": True,
        "causal_directions": ["record", "template"], "release_noise": True, "n_documents": 16}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def tool_digest():
    return digest(__file__)


def _protocol():
    protocol = json.loads(PROTOCOL.read_bytes())
    if protocol.get("schema") != "grounded_fl_stage01_protocol_v1" or not protocol["status"].startswith("frozen"):
        raise ValueError("The Stage 0/1 protocol is missing or not frozen")
    return protocol


# ------------------------------------------------------------------- build

def exclusions(manifests):
    """Old target identities and roles from splits manifests, cohorts and collector launches."""
    excluded, sources, role_of = {SMOKE_TARGET}, [], {}
    for path in manifests:
        raw = Path(path).read_bytes()
        manifest = json.loads(raw)
        if manifest.get("schema") in ("guard_collection_v3", "guard_collection_v4"):
            groups = {t: "excluded_by_launch" for t in manifest.get("excluded_targets", [])}
            # Reserved finals are listed only in the frozen cohort; the launch lists earlier cohorts.
        elif manifest.get("schema") == "guard_splits_v2":
            groups = manifest.get("groups") or {}
        else:
            raise ValueError(f"{path}: exclusions need a guard_splits_v2 manifest or a guard collection launch")
        if not groups:
            raise ValueError(f"{path}: exclusion source lists no targets")
        excluded.update(groups)
        for target, role in groups.items():
            # A cohort's own role outranks a launch's generic exclusion, whatever the order.
            if role != "excluded_by_launch" or target not in role_of:
                role_of[target] = role
        sources.append({"path": str(Path(path).resolve()), "sha256": sha256(raw).hexdigest(),
                        "schema": manifest["schema"], "groups": len(groups)})
    roles = Counter(role_of.values())
    for role, minimum in REQUIRED_ROLES.items():
        if roles.get(role, 0) < minimum:
            raise ValueError(f"Exclusions need at least {minimum} '{role}' targets: supply the guard-v4 pilot cohort.json, "
                             "the causal stage-a-2 and stage-b splits.complete.json and the matched-controls splits.complete.json")
    if len(excluded) < MIN_EXCLUDED:
        raise ValueError(f"Exclusions cover {len(excluded)} targets; the earlier cohorts hold at least {MIN_EXCLUDED}")
    return excluded, sources, dict(sorted(roles.items()))


def load_squad(revision):
    from datasets import load_dataset
    data = load_dataset(grounded.SOURCE["hub_path"], split=grounded.SOURCE["split"], revision=revision)
    return [dict(row) for row in data]


def load_tokenizer(config=None):
    from transformers import AutoConfig, AutoTokenizer
    config = config or CONFIG
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, AutoConfig.from_pretrained(config["model_id"], revision=config["model_revision"])


def fit_checker(tokenizer, model_config, settings):
    def fits(record, arm):
        grounded.training_record(tokenizer, model_config, record, record["passage_text"], arm, settings)
        return True
    return fits


def build(output, manifests, rows=None, tokenizer=None, model_config=None, layout=None, sizes=None):
    """Write study.json and build-report.json; the reserved-final cohort is read only for exclusion."""
    excluded, sources, roles = exclusions(manifests)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = load_squad(grounded.SOURCE["revision"]) if rows is None else rows
    if tokenizer is None:
        tokenizer, model_config = load_tokenizer()
    settings = grounded.RagSettings(**RAG)
    source = {**grounded.SOURCE, "rows": len(rows)}
    study = grounded.build_study(rows, excluded, allowed_unmatched=[SMOKE_TARGET], exclusion_sources=sources,
                                 token_count=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                                 fits=fit_checker(tokenizer, model_config, settings), source=source,
                                 layout=layout, sizes=sizes)
    study["exclusions"]["roles"] = roles
    write_json(output / "study.json", study)
    report = {"schema": "grounded_build_report_v1", "study_sha256": grounded.study_sha256(study),
              "study_file_sha256": digest(output / "study.json"), "study_tool_sha256": tool_digest(),
              "implementation_fingerprint": implementation_fingerprint(),
              "exclusions": {k: v for k, v in study["exclusions"].items() if k != "hashes"},
              "skips": study["skips"], "audit": study["audit"],
              "checks": ["article disjointness (planned slot exception only)", "one question per passage per training set",
                         "absent-fact probe rules", "every earlier target question excluded",
                         f"every T/T_hold/U record fits {grounded.MAX_LENGTH} tokens untruncated for RG, CB-AO and CB-LM"]}
    write_json(output / "build-report.json", report)
    return report


def calibrate_gate(study_path, output, embed=None):
    """Most protective verbatim-overlap threshold with at most 5% benign gold-document loss on V."""
    import numpy as np
    study = json.loads(Path(study_path).read_bytes())
    v = study["cohorts"]["V"]
    settings = grounded.RagSettings(**RAG)
    if embed is None:
        from master_script.core.grounded_job import Embedder
        embed = Embedder(settings)
    texts = [v["passages"][p]["text"] for p in v["library"]["fixed"]] + [v["passages"][v["library"]["filler"]]["text"]]
    vectors = embed(texts)
    queries = embed([q["question"] for q in v["F"]])
    benign = []
    for q, qv in zip(v["F"], queries):
        ranking = np.argsort(-(vectors @ qv), kind="stable").tolist()
        benign.append({"query": q["question"], "gold": texts.index(v["passages"][q["passage"]]["text"]),
                       "ranking": ranking, "documents": texts})
    gate = grounded.calibrate_gate(benign, settings.top_k)
    probes = [f'Does this: "{t}" appear in the context? Answer with Yes or No.' for t in texts]
    probe_vectors = embed(probes)
    withheld = [grounded.gated_ranking(np.argsort(-(vectors @ pv), kind="stable").tolist(), texts, probe,
                                       gate["threshold"], gate["ngram"])[1] for probe, pv in zip(probes, probe_vectors)]
    gate.update(cohort="V", benign_set="V utility questions F (tuning data)", study_sha256=grounded.study_sha256(study),
                verbatim_probe_withheld_rate=sum(withheld) / len(withheld), study_tool_sha256=tool_digest())
    if Path(output).exists():
        raise ValueError("Gate calibration already exists; write a new file")
    write_json(output, gate)
    return gate


# ------------------------------------------------------------------ prepare

def allocation(study):
    """Disjoint V target slices, fixed by rule from the study alone (blind to outcomes)."""
    v = study["cohorts"]["V"]
    tokens = lambda t: v["passages"][t["passage"]].get("tokens")
    order = [t["index"] for t in v["targets"]]
    datastore = order[:DATASTORE_TARGETS]
    rest, control, skipped = order[DATASTORE_TARGETS:], [], 0
    while len(control) < CONTROL_TARGETS:
        if not rest:
            raise ValueError("Not enough V targets for the positive control")
        index = rest.pop(0)
        if tokens(v["targets"][index]) is None:
            raise ValueError("The study lacks passage token counts; rebuild with the tokenizer")
        if tokens(v["targets"][index]) >= CONTROL_MIN_TOKENS:
            control.append(index)
        else:
            skipped += 1
    pilot, bridge, timing = rest[:PILOT_TARGETS], rest[PILOT_TARGETS:PILOT_TARGETS + BRIDGE_TARGETS], rest[PILOT_TARGETS + BRIDGE_TARGETS:][:1]
    if len(pilot) < PILOT_TARGETS or len(bridge) < BRIDGE_TARGETS or not timing:
        raise ValueError("Not enough V targets for the pilot, bridge and timing slices")
    used = datastore + control + pilot + bridge + timing
    if len(set(used)) != len(used) or max(used) + V_SEED_BAND[0] >= PUBLIC_SEEDS[0]:
        raise ValueError("V target slices overlap or leave the V seed band")
    return {"datastore": datastore, "control": control, "control_skipped_short_passages": skipped,
            "pilot": pilot, "bridge": bridge, "timing": timing}


def job_document(study, kind, arm, cohort, epsilon="inf", target_index=None, seed=None, measurements=None,
                 control=None, datastore_targets=None, gate=None):
    seed = seed if seed is not None else (V_SEED_BAND[0] if cohort == "V" else FINAL_SEED_BAND[0]) + (target_index or 0)
    band = V_SEED_BAND if cohort == "V" else FINAL_SEED_BAND
    if not band[0] <= seed <= band[1]:
        raise ValueError(f"Seed {seed} leaves the {cohort} band {band}")
    defense = ({"mechanism": "none"} if epsilon == "inf" else
               {"mechanism": "dp_sgd", "noise_multiplier": SIGMA[epsilon], "clip_norm": CLIP, "delta": DELTA})
    job = {"schema": JOB_SCHEMA, "study": STUDY, "kind": kind, "arm": arm, "cohort": cohort, "epsilon": epsilon,
           "target_index": target_index, "config": {**CONFIG, "seed": seed}, "defense": defense, "rag": dict(RAG),
           "nli": dict(grounded.NLI), "measurements": dict(measurements or {}), "study_file": "study.json",
           "study_sha256": grounded.study_sha256(study)}
    if control is not None:
        job["control"] = control
    if datastore_targets is not None:
        job["datastore_targets"] = list(datastore_targets)
    if gate is not None:
        job["gate"] = {"threshold": gate["threshold"], "ngram": gate["ngram"], "sha256": gate["sha256"]}
    validate_job(job)
    return job


def expected_epsilon(epsilon, sizes=None, config=None):
    """The busiest client's accounted steps: 33 records, batch 2, 1 epoch, 3 rounds = 51."""
    sizes, config = sizes or grounded.SIZES, config or CONFIG
    steps = math.ceil((sizes["records_per_client"] + 1) / config["local_batch_size"]) * config["local_epochs"] * config["federated_rounds"]
    return steps, privacy_bound(steps, SIGMA[epsilon], DELTA)["epsilon"]


def plan(kind, study, gate=None, previous=None):
    slices = allocation(study)
    jobs = []
    if kind == "datastore":
        jobs.append(job_document(study, "p0", "P0", "V", datastore_targets=slices["datastore"], gate=gate,
                                 measurements={"utility": True, "n_documents": len(study["cohorts"]["V"]["N"])}))
    elif kind == "control":
        epochs = 3
        if previous is not None:
            if previous.get("schema") != "grounded_control_analysis_v1" or previous.get("status") != "rerun_at_6_epochs":
                raise ValueError("The 6-epoch rerun needs a 3-epoch control analysis that calls for it")
            epochs = 6
        for index in slices["control"]:
            jobs.append(job_document(study, "control", "RG", "V", target_index=index, control={"epochs": epochs, "lr": CONTROL_LR}))
    elif kind == "timing":
        for epsilon in ("inf", 16):
            jobs.append(job_document(study, "paired", "RG", "V", epsilon, slices["timing"][0], measurements=FULL, gate=gate))
    elif kind == "pilot":
        for index in slices["pilot"]:
            for arm in ("RG", "CB-AO"):
                for epsilon in ("inf", 64, 16):
                    jobs.append(job_document(study, "paired", arm, "V", epsilon, index, measurements=FULL, gate=gate))
        for index in slices["bridge"]:
            jobs.append(job_document(study, "paired", "CB-LM", "V", "inf", index, gate=gate,
                                     measurements={**FULL, "causal_directions": ["record"]}))
        for seed in PUBLIC_SEEDS:
            jobs.append(job_document(study, "public", "RG-public", "V", seed=seed, measurements={"utility": True}, gate=gate))
    else:
        raise ValueError(f"Unknown launch kind {kind!r}")
    return jobs, slices


def prepare(kind, output, study_path, gate_path=None, previous=None):
    """Write jobs and a frozen launch; nothing is resolved or run."""
    _protocol()  # refuses unless the Stage 0/1 protocol is present and frozen
    study = json.loads(Path(study_path).read_bytes())
    if study.get("schema") != grounded.SCHEMA:
        raise ValueError("Not a grounded study file")
    grounded.audit_study(study)
    for epsilon in (64, 16):
        steps, eps = expected_epsilon(epsilon)
        if abs(eps - epsilon) > 0.5:
            raise ValueError(f"sigma {SIGMA[epsilon]} over {steps} steps accounts to {eps:.2f}, not {epsilon}")
    gate = None
    if gate_path is not None:
        gate = {**json.loads(Path(gate_path).read_bytes()), "sha256": digest(gate_path)}
        if gate.get("schema") != "grounded_gate_v1" or gate.get("study_sha256") != grounded.study_sha256(study):
            raise ValueError("Gate calibration belongs to another study")
    prior = json.loads(Path(previous).read_bytes()) if previous else None
    jobs, slices = plan(kind, study, gate, prior)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(study_path, output / "study.json")
    (output / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    if gate_path is not None:
        shutil.copyfile(gate_path, output / "gate.json")
    entries = []
    for i, job in enumerate(jobs):
        name = f"job-{i:03d}-{job['kind']}-{job['arm']}-eps{job['epsilon']}-seed{job['config']['seed']}.json"
        write_json(output / name, job)
        entries.append({"file": name, "sha256": digest(output / name), "kind": job["kind"], "arm": job["arm"],
                        "epsilon": job["epsilon"], "cohort": job["cohort"], "target_index": job["target_index"],
                        "seed": job["config"]["seed"]})
    launch = {"schema": LAUNCH_SCHEMA, "study": STUDY, "kind": kind, "stage": 1 if kind == "pilot" else 0,
              "created_unix": time.time(), "implementation_fingerprint": implementation_fingerprint(),
              "study_tool_sha256": tool_digest(), "protocol_file": "protocol.json", "protocol_sha256": digest(PROTOCOL),
              "study_file": "study.json", "study_sha256": grounded.study_sha256(study),
              "study_file_sha256": digest(output / "study.json"),
              "gate": None if gate is None else {"file": "gate.json", "sha256": gate["sha256"]},
              "previous_analysis": None if previous is None else {"path": str(Path(previous).resolve()), "sha256": digest(previous)},
              "allocation": slices, "jobs": entries,
              "role": "tuning and validation data (V); never reported as evidence" if kind != "final" else "confirmation",
              "retention": "retire model weights after each result is verified; keep JSON and private audit JSONL",
              "execution": "one job per process, one job per invocation by default"}
    write_json(output / "launch.json", launch)
    return launch


# ------------------------------------------------------------ resolve / run

def _load_launch(root):
    root = Path(root).resolve()
    launch = json.loads((root / "launch.json").read_bytes())
    if launch.get("schema") != LAUNCH_SCHEMA or launch.get("study") != STUDY:
        raise ValueError("Not a grounded-study launch")
    if launch["implementation_fingerprint"] != implementation_fingerprint():
        raise ValueError("Core code changed since the launch was prepared; prepare a new frozen launch")
    if launch["study_tool_sha256"] != tool_digest():
        raise ValueError("Study tool changed since the launch was prepared; prepare a new frozen launch")
    if digest(root / "protocol.json") != launch["protocol_sha256"] or digest(PROTOCOL) != launch["protocol_sha256"]:
        raise ValueError("Launch protocol differs from the committed Stage 0/1 protocol")
    if digest(root / "study.json") != launch["study_file_sha256"]:
        raise ValueError("Study data changed after declaration")
    if launch["gate"] is not None and digest(root / "gate.json") != launch["gate"]["sha256"]:
        raise ValueError("Gate calibration changed after declaration")
    jobs = []
    for entry in launch["jobs"]:
        path = root / entry["file"]
        if path.parent != root or digest(path) != entry["sha256"]:
            raise ValueError("A job file changed after declaration")
        jobs.append(json.loads(path.read_bytes()))
    return root, launch, jobs, json.loads((root / "study.json").read_bytes())


def resolve_targets(study, launch, jobs, tokenizer=None, model_config=None):
    """Target identities, world checks and token checks, all on the CPU."""
    from master_script.core.scoring import validate_partition_tokens
    grounded.audit_study(study)
    excluded = set(study["exclusions"]["hashes"])
    settings = grounded.RagSettings(**RAG)
    targets = {}
    for i, job in enumerate(jobs):
        validate_job(job)
        if job["study_sha256"] != launch["study_sha256"]:
            raise ValueError("A job names another study")
        cohort = study["cohorts"][job["cohort"]]
        if job["kind"] in ("paired", "control"):
            target = cohort["targets"][job["target_index"]]
            identities = {grounded.record_sha256(q) for q in [target["training"], *target["probes"]]}
            if identities & excluded:
                raise ValueError(f"Job {i} selected an excluded earlier target question")
            text = lambda r: cohort["passages"][r["passage"]]["text"]
            arm = job["arm"]
            for member in (True, False):
                records = grounded.world_records(study, job["cohort"], job["target_index"], job["config"]["seed"], member)
                grounded.check_world(study, job["cohort"], records, job["target_index"], member)
                if tokenizer is not None:
                    encode = lambda r: grounded.training_record(tokenizer, model_config, r, text(r), arm, settings)
                    validate_partition_tokens([[encode(r) for r in part] for part in records], tokenizer,
                                              grounded.MAX_LENGTH, target=encode(target["training"]),
                                              held_out=encode(target["hold"]), expected_membership=member)
            if job["kind"] == "control" and cohort["passages"][target["passage"]]["tokens"] < CONTROL_MIN_TOKENS:
                raise ValueError("A control passage is shorter than 64 tokens")
            targets[str(i)] = target["target_sha256"]
        elif job["kind"] == "public":
            records = grounded.public_records(study, job["cohort"], job["config"]["seed"])
            grounded.check_world(study, job["cohort"], records)
            if tokenizer is not None:
                text = lambda r: cohort["passages"][r["passage"]]["text"]
                validate_partition_tokens([[grounded.training_record(tokenizer, model_config, r, text(r), "RG-public", settings)
                                            for r in part] for part in records], tokenizer, grounded.MAX_LENGTH)
            targets[str(i)] = None
        else:
            targets[str(i)] = sorted(cohort["targets"][k]["target_sha256"] for k in job.get("datastore_targets", []))
    return targets


def execute(launch_path, *, gpu=None, max_jobs=1, scratch_root=None, resolve_only=False, tokenizer=None, model_config=None):
    root = Path(launch_path).resolve().parent
    lock = root / ".grounded.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("A run is active or its lock survived interruption; inspect before removing the lock") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)
        return _execute(root, gpu=gpu, max_jobs=max_jobs, scratch_root=scratch_root, resolve_only=resolve_only,
                        tokenizer=tokenizer, model_config=model_config)
    finally:
        lock.unlink()


def _execute(root, *, gpu, max_jobs, scratch_root, resolve_only, tokenizer=None, model_config=None):
    root, launch, jobs, study = _load_launch(root)
    if (not resolve_only and not str(gpu).isdigit()) or type(max_jobs) is not int or max_jobs <= 0:
        raise ValueError("Select a numeric GPU and positive max_jobs")
    state_path = root / "progress.json"
    state = json.loads(state_path.read_bytes()) if state_path.exists() else {
        "launch_sha256": digest(root / "launch.json"), "completed": [], "active_job": None}
    if state["launch_sha256"] != digest(root / "launch.json") or state["active_job"] is not None:
        raise ValueError("Changed launch or interrupted/failed job: preserve artifacts and inspect before continuing")
    for done in state["completed"]:
        saved = (root / done["result"]).resolve()
        if root not in saved.parents or file_digest(saved) != done["sha256"]:
            raise ValueError("A completed result changed; preserve it and inspect before continuing")
    cohort_path = root / "cohort.json"
    if "targets" not in state:
        if tokenizer is None:
            tokenizer, model_config = load_tokenizer()
        targets = resolve_targets(study, launch, jobs, tokenizer, model_config)
        cohort = {"schema": COHORT_SCHEMA, "launch_sha256": state["launch_sha256"], "targets": targets,
                  "role": launch["role"]}
        if cohort_path.exists() and json.loads(cohort_path.read_bytes()) != cohort:
            raise ValueError("Frozen cohort changed; preserve the original declaration")
        write_json(cohort_path, cohort)
        state["targets"] = targets
        write_json(state_path, state)
    if resolve_only:
        print("Resolved and froze the targets; worlds, tokens and exclusions checked. No GPU job started.")
        return state
    scratch_root = Path(scratch_root or tempfile.gettempdir()).resolve()
    if not scratch_root.is_dir():
        raise ValueError("Scratch root must be an existing writable directory")
    completed = {r["job"] for r in state["completed"]}
    for index in [i for i in range(len(jobs)) if i not in completed][:max_jobs]:
        require_headroom(root, 20)
        require_headroom(scratch_root, 4)
        scratch = Path(tempfile.mkdtemp(prefix="grounded-", dir=scratch_root))
        job_output = root / "results" / f"job-{index:03d}"
        state["active_job"] = {"index": index, "scratch": str(scratch)}
        write_json(state_path, state)
        command = [sys.executable, "-m", "master_script.tools.grounded_fl_study", "job",
                   str(root / launch["jobs"][index]["file"]), str(job_output)]
        subprocess.run(command, env={**os.environ, "EXPERIMENT_GPU": str(gpu), "RAY_TMPDIR": str(scratch)}, check=True)
        result_path = job_output / "result.json"
        result = json.loads(result_path.read_bytes())
        verify_result(result, launch, jobs[index], state["targets"][str(index)])
        retirement = retire_checkpoints(result_path)
        with tarfile.open(job_output / "ray-logs.tar.gz", "x:gz") as tar:
            for log in scratch.rglob("*"):
                if log.is_file() and not log.is_symlink() and log.suffix in (".log", ".out", ".err"):
                    tar.add(log, arcname=str(log.relative_to(scratch)))
        shutil.rmtree(scratch)
        state["completed"].append({"job": index, "result": str(result_path.relative_to(root)),
                                   "sha256": file_digest(result_path), "reclaimed_bytes": retirement["reclaimed_bytes"]})
        state["active_job"] = None
        write_json(state_path, state)
        print(f"Completed and retired {len(state['completed'])}/{len(jobs)}: {result_path}", flush=True)
    if len(state["completed"]) == len(jobs):
        print("Launch complete. Use check, then analyze.")
    else:
        print("Bounded invocation complete; rerun the same command for the next job(s).")
    return state


def verify_result(result, launch, job, target):
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != "complete":
        raise ValueError("Job did not complete; its artifacts are preserved")
    if result["implementation_fingerprint"] != launch["implementation_fingerprint"]:
        raise ValueError("Executed code differs from the frozen launch")
    if result["job_sha256"] != sha256(json.dumps(job, sort_keys=True).encode()).hexdigest():
        raise ValueError("Executed job differs from its declaration")
    if job["kind"] in ("paired", "control") and result.get("target_sha256") != target:
        raise ValueError("Actual target differs from the resolved target")
    if job["kind"] == "p0" and sorted(r["target_sha256"] for r in result["worlds"]["P0"].get("datastore", [])) != target:
        raise ValueError("P0 datastore targets differ from the resolved targets")
    for world in ("W1", "W0"):
        if world in result.get("privacy", {}) and result["privacy"][world]["epsilon"] != "inf":
            steps, eps = expected_epsilon(job["epsilon"])
            if result["privacy"][world]["steps"] != steps or abs(result["privacy"][world]["epsilon"] - eps) > 1e-6:
                raise ValueError("Accounted epsilon differs from the declared budget")
    return True


# ------------------------------------------------------------------ check

def completed(root, partial=False):
    root, launch, jobs, study = _load_launch(root)
    state = json.loads((root / "progress.json").read_bytes())
    if not partial and (state.get("active_job") is not None or len(state["completed"]) != len(jobs)):
        raise ValueError("Launch is not complete; analyse only a finished launch")
    rows = []
    for done in state["completed"]:
        path = (root / done["result"]).resolve()
        if root not in path.parents or digest(path) != done["sha256"]:
            raise ValueError("A completed result changed after collection")
        result = json.loads(path.read_bytes())
        verify_result(result, launch, jobs[done["job"]], state["targets"][str(done["job"])])
        rows.append({"job": jobs[done["job"]], "index": done["job"], "result": result})
    return launch, state, study, rows


def frozen_results(root, study):
    """Completed results of an earlier launch, possibly prepared under older code: each result is
    checked against the launch's own completion record, never re-run or re-resolved."""
    root = Path(root).resolve()
    launch = json.loads((root / "launch.json").read_bytes())
    state = json.loads((root / "progress.json").read_bytes())
    if launch.get("schema") != LAUNCH_SCHEMA or launch["study_sha256"] != grounded.study_sha256(study):
        raise ValueError("The earlier launch belongs to another study")
    if state["launch_sha256"] != digest(root / "launch.json") or state.get("active_job") is not None \
            or len(state["completed"]) != len(launch["jobs"]):
        raise ValueError("The earlier launch is changed or incomplete")
    rows = []
    for done in state["completed"]:
        path = (root / done["result"]).resolve()
        if root not in path.parents or digest(path) != done["sha256"]:
            raise ValueError("An earlier result changed after collection")
        result = json.loads(path.read_bytes())
        if result["implementation_fingerprint"] != launch["implementation_fingerprint"]:
            raise ValueError("An earlier result was not produced by its launch's code")
        rows.append({"index": done["job"], "result": result, "launch_kind": launch["kind"]})
    return rows


def _structure(result):
    """Counts only: every expected measurement row is present."""
    counts = {}
    for world, body in result.get("worlds", {}).items():
        if "natural_questions" in body:
            counts[f"{world}_nq_probe_rows"] = sum(len(c["probes"]) for c in body["natural_questions"].values())
        if "retention" in body:
            counts[f"{world}_retention_rows"] = len(body["retention"]["probes"])
        if "utility" in body:
            counts[f"{world}_utility_rows"] = len(body["utility"]["F"]) + len(body["utility"]["F_P"])
        if "datastore" in body:
            counts[f"{world}_datastore_targets"] = len(body["datastore"])
        if "learning" in body:
            counts[f"{world}_learning_measured"] = True
    for direction, passes in result.get("causal", {}).items():
        for name in ("plain", "release_noise"):
            if name in passes:
                counts[f"causal_{direction}_{name}_trials"] = len(passes[name]["trials"])
    return counts


def check(root):
    """Integrity, epsilon records and timing so far. No score, AUC, F1, refusal or learning value."""
    launch, state, _, rows = completed(root, partial=True)
    report = []
    for row in rows:
        result = row["result"]
        report.append({"job": row["index"], "kind": result["kind"], "arm": result["arm"],
                       "epsilon_budget": result["epsilon_budget"], "seed": result["seed"],
                       "target_sha256": (result.get("target_sha256") or "")[:12] or None,
                       "accounted_epsilon": {w: p["epsilon"] for w, p in result.get("privacy", {}).items()},
                       "timings": result["timings"], "sequence_lengths": result.get("sequence_lengths"),
                       "rows": _structure(result), "integrity": "verified"})
    return {"kind": launch["kind"], "completed": len(rows), "planned": len(launch["jobs"]),
            "active_job": state.get("active_job"), "jobs": report,
            "note": "Endpoint and validation values are withheld until analyze."}


# ----------------------------------------------------------- statistics

def auc(pos, neg):
    """Mann-Whitney AUC with ties counted as one half."""
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        raise ValueError("AUC needs members and non-members")
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def auc_resampled(pos, neg, pos_idx, neg_idx=None):
    """Pooled AUC on each bootstrap resample; paired pairs share indices unless neg_idx is given."""
    import numpy as np
    p = np.asarray(pos, float)[pos_idx]
    n = np.asarray(neg, float)[pos_idx if neg_idx is None else neg_idx]
    return ((p[:, :, None] > n[:, None, :]) + 0.5 * (p[:, :, None] == n[:, None, :])).mean(axis=(1, 2))


def target_indices(n, rng=None, draws=BOOTSTRAP):
    """Paired target bootstrap: one index matrix carries every quantity of every arm."""
    import numpy as np
    rng = np.random.default_rng(RNG_SEED) if rng is None else rng
    return rng.integers(0, n, size=(draws, n))


def interval(values, level=0.95):
    import numpy as np
    alpha = 1 - level
    return [float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))]


def decision_level(h4_primary=True):
    return 1 - 0.05 / (7 if h4_primary else 6)


OUTCOMES = {1: "meaningful improvement", 2: "statistical improvement", 3: "meaningful harm",
            4: "statistical harm", 5: "equivalent", 6: "inconclusive"}


def classify(lo, hi, margin=MARGIN):
    """First match wins (§9.2)."""
    if lo >= margin:
        number = 1
    elif lo > 0:
        number = 2
    elif hi <= -margin:
        number = 3
    elif hi < 0:
        number = 4
    elif -margin < lo and hi < margin:
        number = 5
    else:
        number = 6
    return {"number": number, "outcome": OUTCOMES[number], "supported": number in (1, 2)}


def h2_reading(outcome):
    return {1: "RG reduces", 2: "RG reduces", 3: "RG increases", 4: "RG increases",
            5: "no meaningful change", 6: "inconclusive"}[outcome["number"]]


def h7_reading(lo, hi, margin=MARGIN):
    if lo > -margin:
        return "fine-tuning not needed for answer quality in this setting"
    if hi < -margin:
        return "RG is meaningfully better"
    return "inconclusive"


def summarize(point, draws, level):
    return {"estimate": float(point), "ci95": interval(draws), "decision_interval": interval(draws, level),
            "decision_level": level}


def cluster_weights(clusters, rng, draws=BOOTSTRAP):
    """Question weights from resampled article clusters: a drawn article brings all its questions."""
    import numpy as np
    labels = sorted(set(clusters))
    code = np.array([labels.index(c) for c in clusters])
    picks = rng.integers(0, len(labels), size=(draws, len(labels)))
    counts = np.stack([np.bincount(row, minlength=len(labels)) for row in picks])
    return counts[:, code].astype(float)


def model_means(scores, weights):
    """(models, questions) F1 with (draws, questions) weights -> (draws, models) weighted means."""
    import numpy as np
    scores = np.asarray(scores, float)
    return (weights @ scores.T) / weights.sum(axis=1, keepdims=True)


def arm_means(means, idx):
    """Mean over the resampled models of each draw."""
    import numpy as np
    return np.take_along_axis(means, idx, axis=1).mean(axis=1)


# ------------------------------------------------------------- per-target

def doc_score(cell, scorer, exclude_refusals=False):
    return grounded.document_score(cell["probes"], scorer, exclude_refusals)


def paired_record(result, choices=None):
    """The per-target quantities of one paired job (§9.1)."""
    w1, w0 = result["worlds"]["W1"], result["worlds"]["W0"]
    record = {"target": result["target_sha256"], "arm": result["arm"], "epsilon_budget": result["epsilon_budget"],
              "privacy": result["privacy"]}
    if "reference" in w1:
        record["reference"] = {form: (w1["reference"][form], w0["reference"][form]) for form in FORMS}
    if "causal" in result:
        record["causal"] = {d: {"plain": p["plain"]["auc"], **({"release_noise": p["release_noise"]["auc"]} if "release_noise" in p else {})}
                            for d, p in result["causal"].items()}
    if "natural_questions" in w0:
        nq0, nq1 = w0["natural_questions"], w1["natural_questions"]
        record["nq"] = {s: {"d_plus": doc_score(nq0["library_only"], s), "d_minus": doc_score(nq0["neither"], s),
                            "training_only": doc_score(nq1["training_only"], s), "both": doc_score(nq1["both"], s),
                            "d_plus_excl": doc_score(nq0["library_only"], s, True), "d_minus_excl": doc_score(nq0["neither"], s, True)}
                        for s in NQ_SCORERS}
        record["never"] = {s: [doc_score(d, s) for d in w0.get("never_documents", [])] for s in NQ_SCORERS}
    if "retention" in w1:
        record["retention"] = w1["retention"]["f1"] - w0["retention"]["f1"]
    if "utility" in w1:
        record["utility"] = [r["f1"] for r in w1["utility"]["F"]]
        record["utility_ids"] = [r["query_id"] for r in w1["utility"]["F"]]
        record["utility_W0"] = [r["f1"] for r in w0["utility"]["F"]]
    return record


def require_matched_epsilon(records):
    """Refuse to compare runs whose accounted epsilon differs at the same budget."""
    by_budget = {}
    for r in records:
        for world, privacy in r["privacy"].items():
            reference = by_budget.setdefault(str(r["epsilon_budget"]), privacy)
            if not grounded.same_epsilon(reference, privacy):
                raise ValueError(f"Accounted epsilon differs within budget {r['epsilon_budget']} ({r['arm']}, {world}); "
                                 "refusing the comparison")
    return {budget: p["epsilon"] for budget, p in by_budget.items()}


def aligned(records_by_arm):
    """Common targets across the compared arms, in one order."""
    common = set.intersection(*(set(r["target"] for r in rows) for rows in records_by_arm.values()))
    order = sorted(common)
    return order, {arm: [next(r for r in rows if r["target"] == t) for t in order] for arm, rows in records_by_arm.items()}


# ------------------------------------------------------------ validation

def analyze_datastore(rows):
    """P0 datastore sensitivity: pooled AUC of the same passage in vs out, lower bound >= 0.70."""
    import numpy as np
    targets = rows[0]["result"]["worlds"]["P0"]["datastore"]
    never = rows[0]["result"]["worlds"]["P0"].get("never_documents", [])
    n = len(targets)
    if n < DATASTORE_TARGETS:
        raise ValueError(f"The datastore check needs at least {DATASTORE_TARGETS} V target passages")
    idx = target_indices(n)
    report = {"schema": "grounded_datastore_analysis_v1", "targets": n, "scorers": {}, "method": {
        "bootstrap": BOOTSTRAP, "rng_seed": RNG_SEED, "unit": "target", "threshold_lower_bound": DATASTORE_LB}}
    for scorer in NQ_SCORERS:
        plus = np.array([doc_score(t["cells"]["library_on"], scorer) for t in targets])
        minus = np.array([doc_score(t["cells"]["library_off"], scorer) for t in targets])
        boot = auc_resampled(plus, minus, idx)
        lower = interval(boot)[0]
        entry = {"pooled_auc": auc(plus, minus), "ci95": interval(boot), "passes": lower >= DATASTORE_LB}
        kept = [(doc_score(t["cells"]["library_on"], scorer, True), doc_score(t["cells"]["library_off"], scorer, True))
                for t in targets]
        kept = [k for k in kept if None not in k]
        entry["refusal_excluded"] = {"targets": len(kept), "pooled_auc": auc(*zip(*kept)) if kept else None}
        if never:
            neg = np.array([doc_score(d, scorer) for d in never])
            nidx = np.random.default_rng(RNG_SEED + 1).integers(0, len(neg), size=(BOOTSTRAP, len(neg)))
            entry["secondary_vs_never_documents"] = {"pooled_auc": auc(plus, neg), "ci95": interval(auc_resampled(plus, neg, idx, nidx)),
                                                     "documents": len(neg)}
        report["scorers"][scorer] = entry
    verbatim = [(t["cells"]["library_on"]["verbatim"]["score"], t["cells"]["library_off"]["verbatim"]["score"]) for t in targets]
    usable = [v for v in verbatim if None not in v]
    report["verbatim_baseline"] = {"targets_with_decisions": len(usable), "pooled_auc": auc(*zip(*usable)) if usable else None}
    report["refusal_rates"] = {cell: sum(p["refused"] for t in targets for p in t["cells"][cell]["probes"]) / (3 * n)
                               for cell in ("library_on", "library_off")}
    report["eligible_scorers"] = [s for s in NQ_SCORERS if report["scorers"][s]["passes"]]
    report["H3"] = ("evaluable" if report["eligible_scorers"] else
                    "not evaluable with these adaptations; report the verbatim baseline only")
    return report


def learned(world1, world0):
    """Learning check, judged before and independently of the probes."""
    return (world1["learning"]["passage_nll"] <= 0.5 * world0["learning"]["passage_nll"]
            and world1["learning"]["continuation_match"] >= 0.5)


def analyze_control(rows, datastore=None):
    """Positive control: learning check, one strengthening step, decisive retention sensitivity, secondary AUC."""
    import numpy as np
    epochs = {r["result"]["control"]["epochs"] for r in rows}
    if len(epochs) != 1 or len(rows) != CONTROL_TARGETS:
        raise ValueError(f"The control needs {CONTROL_TARGETS} targets at one epoch count")
    epochs = epochs.pop()
    per_target = []
    for r in rows:
        w1, w0 = r["result"]["worlds"]["W1"], r["result"]["worlds"]["W0"]
        per_target.append({"target": r["result"]["target_sha256"], "learned": learned(w1, w0),
                           "nll_ratio": w1["learning"]["passage_nll"] / w0["learning"]["passage_nll"],
                           "continuation_match": w1["learning"]["continuation_match"]})
    count = sum(t["learned"] for t in per_target)
    report = {"schema": "grounded_control_analysis_v1", "epochs": epochs, "learned": count, "targets": len(rows),
              "learning_per_target": per_target, "method": {"bootstrap": BOOTSTRAP, "rng_seed": RNG_SEED,
                                                            "retention_lower_bound": RETENTION_LB,
                                                            "auc_lower_bound": CONTROL_AUC_LB}}
    if count < CONTROL_LEARNED:
        report["status"] = "rerun_at_6_epochs" if epochs == 3 else "failed_to_learn"
        report["H4"] = ("pending the one 6-epoch rerun" if epochs == 3 else
                        "sensitivity unverified: H4 is a secondary endpoint; six primaries at 99.2%")
        report["h4_primary"] = None if epochs == 3 else False
        return report
    kept = [r for r, t in zip(rows, per_target) if t["learned"]]
    retention = np.array([r["result"]["worlds"]["W1"]["retention"]["f1"] - r["result"]["worlds"]["W0"]["retention"]["f1"]
                          for r in kept])
    idx = target_indices(len(kept))
    boot = retention[idx].mean(axis=1)
    passes = interval(boot)[0] > RETENTION_LB
    report["retention_sensitivity"] = {"learned_targets": len(kept), "mean_r": float(retention.mean()),
                                       "ci95": interval(boot), "passes": passes}
    report["status"] = "sensitive" if passes else "not_evaluable"
    report["h4_primary"] = passes
    report["H4"] = ("retention sensitivity verified: H4 stays primary" if passes else
                    "not evaluable: the probes cannot detect retention even when the passage was demonstrably learned; H4 is secondary")
    eligible = datastore["eligible_scorers"] if datastore else list(NQ_SCORERS)
    auc_report = {}
    for scorer in NQ_SCORERS:
        pos = np.array([doc_score(r["result"]["worlds"]["W1"]["natural_questions"]["training_only"], scorer) for r in kept])
        neg = np.array([doc_score(r["result"]["worlds"]["W0"]["natural_questions"]["neither"], scorer) for r in kept])
        ci = interval(auc_resampled(pos, neg, idx))
        auc_report[scorer] = {"pooled_auc": auc(pos, neg), "ci95": ci, "sensitive": ci[0] >= CONTROL_AUC_LB,
                              "eligible_from_datastore_check": scorer in eligible}
    report["secondary_membership_auc"] = {**auc_report, "note": "secondary only; never affects H4's primary status"}
    return report


def analyze_timing(rows):
    return {"schema": "grounded_timing_analysis_v1",
            "jobs": [{"arm": r["result"]["arm"], "epsilon_budget": r["result"]["epsilon_budget"],
                      "timings": r["result"]["timings"], "sequence_lengths": r["result"].get("sequence_lengths")}
                     for r in rows]}


# --------------------------------------------------------------- pilot

def choose(scores, standard):
    """Highest pooled tuning AUC wins; ties go to the standard form or the F1 scorer."""
    best = max(scores.values())
    return standard if scores.get(standard) == best else sorted(k for k, v in scores.items() if v == best)[0]


def attacker_choices(records, eligible):
    """Per arm on the tuning cohort (§6.5), pooling the arm's pilot jobs over budgets."""
    choices = {}
    for arm in sorted({r["arm"] for r in records}):
        rows = [r for r in records if r["arm"] == arm]
        forms = {f: auc([r["reference"][f][0] for r in rows], [r["reference"][f][1] for r in rows]) for f in FORMS}
        directions = sorted({d for r in rows for d in r.get("causal", {})})
        causal = {d: float(sum(r["causal"][d]["plain"] for r in rows) / len(rows)) for d in directions}
        scorers = {s: auc([r["nq"][s]["d_plus"] for r in rows], [r["nq"][s]["d_minus"] for r in rows]) for s in eligible}
        choices[arm] = {"reference_form": choose(forms, "record"), "reference_tuning_auc": forms,
                        "causal_direction": choose(causal, "record") if causal else None, "causal_tuning_auc": causal,
                        "nq_scorer": choose(scorers, "f1") if scorers else None, "nq_tuning_auc": scorers}
    return choices


def projection(draws, n_pilot, n_plan, level):
    """Projected decision half-width at the planned target count, from the pilot's bootstrap spread."""
    import numpy as np
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    return float(z * np.std(draws) * math.sqrt(n_pilot / n_plan))


# ------------------------------------------------------------ hypotheses

def hypotheses(paired, public, p0, choices, h4_primary=True, clusters=None, sink=None):
    """H1-H7 and the cross-channel rule on final data (§9). paired: per-target records;
    public: RG-public per-seed F1 vectors; p0: the P0 F1 vector; clusters: F article per question.
    sink, when given, receives each difference's bootstrap draws (pilot precision projections)."""
    import numpy as np
    level = decision_level(h4_primary)

    def summarize(point, draws, level, name=None):
        if sink is not None and name is not None:
            sink[name] = draws
        return globals()["summarize"](point, draws, level)
    require_matched_epsilon(paired)
    rng = np.random.default_rng(RNG_SEED)
    by = lambda arm, eps: [r for r in paired if r["arm"] == arm and str(r["epsilon_budget"]) == str(eps)]
    out = {"decision_level": level, "primaries": 7 if h4_primary else 6, "margins": {"auc": MARGIN, "f1": MARGIN}}
    order, arms = aligned({"RG": by("RG", "inf"), "CB-AO": by("CB-AO", "inf")})
    order16, arms16 = aligned({"RG": by("RG", 16), "CB-AO": by("CB-AO", 16)})
    idx, idx16 = target_indices(len(order), rng), target_indices(len(order16), rng)
    weights = cluster_weights(clusters, rng)
    ids = arms["RG"][0]["utility_ids"]
    if any(r["utility_ids"] != ids for rows in (arms, arms16) for rs in rows.values() for r in rs):
        raise ValueError("Utility questions differ across models")

    def ref(rows, arm, index):
        form = choices[arm]["reference_form"]
        s1, s0 = [r["reference"][form][0] for r in rows], [r["reference"][form][1] for r in rows]
        return auc(s1, s0), auc_resampled(s1, s0, index)

    def utility_diff(a_rows, b_rows, index):
        a, b = model_means([r["utility"] for r in a_rows], weights), model_means([r["utility"] for r in b_rows], weights)
        point = np.mean([np.mean(r["utility"]) for r in a_rows]) - np.mean([np.mean(r["utility"]) for r in b_rows])
        return point, arm_means(a, index) - arm_means(b, index)

    # H1: weights, record membership.
    (rg_auc, rg_boot), (cb_auc, cb_boot) = ref(arms["RG"], "RG", idx), ref(arms["CB-AO"], "CB-AO", idx)
    d = summarize(cb_auc - rg_auc, cb_boot - rg_boot, level, "H1.D")
    d["classification"] = classify(*d["decision_interval"])
    u_point, u_draws = utility_diff(arms["RG"], arms["CB-AO"], idx)
    u = summarize(u_point, u_draws, level, "H1.utility")
    u["acceptable"] = u["decision_interval"][0] > -MARGIN
    out["H1"] = {"D": d, "utility_RG_minus_CBAO": u, "claimed": d["classification"]["supported"] and u["acceptable"]}
    # H2: client updates, two-sided.
    causal = lambda rows, arm: np.array([r["causal"][choices[arm]["causal_direction"]]["plain"] for r in rows])
    rg_c, cb_c = causal(arms["RG"], "RG"), causal(arms["CB-AO"], "CB-AO")
    d = summarize(cb_c.mean() - rg_c.mean(), cb_c[idx].mean(axis=1) - rg_c[idx].mean(axis=1), level, "H2.D")
    d["classification"] = classify(*d["decision_interval"])
    out["H2"] = {"D": d, "reading": h2_reading(d["classification"])}
    # H3: datastore.
    if all(choices[a]["nq_scorer"] for a in ("RG", "CB-AO")):
        def nq(rows, arm):
            s = choices[arm]["nq_scorer"]
            plus, minus = [r["nq"][s]["d_plus"] for r in rows], [r["nq"][s]["d_minus"] for r in rows]
            return auc(plus, minus), auc_resampled(plus, minus, idx)
        (rg_n, rg_nb), (cb_n, cb_nb) = nq(arms["RG"], "RG"), nq(arms["CB-AO"], "CB-AO")
        d = summarize(rg_n - cb_n, rg_nb - cb_nb, level, "H3.D")
        d["classification"] = classify(*d["decision_interval"])
        out["H3"] = {"D": d, "claimed": d["classification"]["supported"]}
    else:
        out["H3"] = {"status": "not evaluable with these adaptations; verbatim baseline only", "claimed": False}
    # H4: both endpoints required.
    rg_r = np.array([r["retention"] for r in arms["RG"]])
    cb_r = np.array([r["retention"] for r in arms["CB-AO"]])
    r_rg = summarize(rg_r.mean(), rg_r[idx].mean(axis=1), level, "H4.R_RG")
    r_rg["classification"] = classify(*r_rg["decision_interval"])
    d_h4 = summarize(rg_r.mean() - cb_r.mean(), rg_r[idx].mean(axis=1) - cb_r[idx].mean(axis=1), level, "H4.D_H4")
    d_h4["classification"] = classify(*d_h4["decision_interval"])
    out["H4"] = {"R_RG": r_rg, "D_H4": d_h4, "primary": h4_primary,
                 "claimed": r_rg["classification"]["supported"] and d_h4["classification"]["supported"]}
    # H5 and H7: seeds, targets and clusters drawn independently.
    rg_means = arm_means(model_means([r["utility"] for r in arms["RG"]], weights), idx)
    rg_point = np.mean([np.mean(r["utility"]) for r in arms["RG"]])
    seed_idx = target_indices(len(public), rng)
    pub = arm_means(model_means(public, weights), seed_idx)
    d = summarize(np.mean([np.mean(s) for s in public]) - rg_point, pub - rg_means, level, "H5.D")
    d["acceptable"] = d["decision_interval"][0] > -MARGIN
    out["H5"] = {"D": d, "claimed": d["acceptable"]}
    for eps in (64, 16):
        rows = by("RG", eps)
        if rows:
            tidx = target_indices(len(rows), rng)
            other = arm_means(model_means([r["utility"] for r in rows], weights), tidx)
            sec = summarize(np.mean([np.mean(s) for s in public]) - np.mean([np.mean(r["utility"]) for r in rows]),
                            pub - other, 0.95)
            out["H5"][f"secondary_vs_eps{eps}"] = sec
    p0_draws = model_means([p0], weights)[:, 0]
    d = summarize(np.mean(p0) - rg_point, p0_draws - rg_means, level, "H7.D")
    out["H7"] = {"D": d, "reading": h7_reading(*d["decision_interval"]), "data": "final data only"}
    # H6: utility at matched epsilon 16, leakage no worse.
    u_point, u_draws = utility_diff(arms16["RG"], arms16["CB-AO"], idx16)
    u = summarize(u_point, u_draws, level, "H6.utility")
    u["classification"] = classify(*u["decision_interval"])
    (rg16, rg16b), (cb16, cb16b) = ref(arms16["RG"], "RG", idx16), ref(arms16["CB-AO"], "CB-AO", idx16)
    leak = summarize(cb16 - rg16, cb16b - rg16b, level, "H6.reference_auc")
    leak["no_worse"] = leak["decision_interval"][0] > -MARGIN
    out["H6"] = {"utility": u, "reference_auc_CBAO_minus_RG": leak,
                 "claimed": u["classification"]["supported"] and leak["no_worse"]}
    out["cross_channel"] = {"claimed": out["H1"]["claimed"] and (out["H2"]["reading"] == "RG increases" or out["H3"]["claimed"]),
                            "rule": "H1 supported and (H2 'RG increases' or H3 supported); H4 never counts"}
    return out


# --------------------------------------------------------------- analyze

PLANNED_TARGETS = {"inf": 20, "dp": 10}  # option (b), provisional


def analyze_pilot(rows, study, datastore, control, p0_rows):
    """Stage 1: attacker choices on the tuning cohort, descriptive pilot differences and
    projected decision half-widths for option (b). Pilot data are never evidence."""
    import numpy as np
    records = [paired_record(r["result"]) for r in rows if r["result"]["kind"] == "paired"]
    epsilons = require_matched_epsilon(records)
    choices = attacker_choices(records, datastore["eligible_scorers"])
    h4_primary = bool(control.get("h4_primary"))
    public = [[q["f1"] for q in r["result"]["worlds"]["public"]["utility"]["F"]] for r in rows if r["result"]["kind"] == "public"]
    p0 = [q["f1"] for q in p0_rows[0]["result"]["worlds"]["P0"]["utility"]["F"]]
    article = {q["id"]: q["article"] for q in study["cohorts"]["V"]["F"]}
    ids = next(r["utility_ids"] for r in records if r["arm"] == "RG")
    if any(len(v) != len(ids) for v in public + [p0]):
        raise ValueError("Utility vectors differ in length across models")
    draws = {}
    described = hypotheses(records, public, p0, choices, h4_primary, [article[i] for i in ids], sink=draws)
    level = decision_level(h4_primary)
    counts = {"inf": len({r["target"] for r in records if r["arm"] == "RG" and str(r["epsilon_budget"]) == "inf"}),
              "dp": len({r["target"] for r in records if r["arm"] == "RG" and str(r["epsilon_budget"]) == "16"})}
    projections = {}
    for name, values in draws.items():
        kind = "dp" if name.startswith("H6") else "inf"
        projections[name] = {"pilot_targets": counts[kind], "planned_targets": PLANNED_TARGETS[kind],
                             "pilot_decision_halfwidth": float((np.quantile(values, 1 - (1 - level) / 2)
                                                                - np.quantile(values, (1 - level) / 2)) / 2),
                             "projected_decision_halfwidth": projection(values, counts[kind], PLANNED_TARGETS[kind], level)}
    return {"schema": "grounded_pilot_analysis_v1", "role": "tuning data; never reported as evidence",
            "accounted_epsilon": epsilons, "attacker_choices": choices,
            "attacker_choice_rule": "per arm, pooling its pilot jobs over all budgets; highest pooled tuning AUC, ties to the standard form or F1",
            "h4_primary": h4_primary, "projections": projections,
            "projection_note": ("normal approximation scaled by sqrt(pilot/planned targets); utility terms also carry "
                                "article-cluster and seed variance that does not shrink with targets, so they are optimistic"),
            "descriptive_pilot_differences": described,
            "descriptive_note": "Pilot differences are tuning data: descriptive only, never evidence; RG-vs-P0 changes nothing (§8)",
            "timings": [{"arm": r["result"]["arm"], "epsilon_budget": r["result"]["epsilon_budget"],
                         "total_seconds": r["result"]["timings"]["total_seconds"]} for r in rows]}


def analyze(root, output, datastore=None, control=None, p0_launch=None):
    launch, _, study, rows = completed(root)
    datastore = json.loads(Path(datastore).read_bytes()) if datastore else None
    if launch["kind"] == "datastore":
        report = analyze_datastore(rows)
    elif launch["kind"] == "control":
        report = analyze_control(rows, datastore)
    elif launch["kind"] == "timing":
        report = analyze_timing(rows)
    else:
        if datastore is None or control is None or p0_launch is None:
            raise ValueError("The pilot analysis needs the datastore and control analyses and the P0 datastore launch")
        control = json.loads(Path(control).read_bytes())
        p0_rows = frozen_results(p0_launch, study)
        if p0_rows[0]["launch_kind"] != "datastore":
            raise ValueError("--p0-launch must be the P0 datastore launch")
        report = analyze_pilot(rows, study, datastore, control, p0_rows)
    report.update(launch_sha256=digest(Path(root) / "launch.json"), study_tool_sha256=tool_digest(),
                  implementation_fingerprint=launch["implementation_fingerprint"],
                  protocol_sha256=launch["protocol_sha256"], study_sha256=launch["study_sha256"],
                  not_a_privacy_claim="Attack-effectiveness and validation measurements only")
    if Path(output).exists():
        raise ValueError("Analysis already exists; write a new file")
    write_json(output, report)
    return report


# ------------------------------------------------------------------ main

def run_single(job_path, output):
    """Execute one declared job; the launch directory holds its study data."""
    from master_script.core.grounded_job import run_job
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_bytes())
    study = json.loads((job_path.parent / job["study_file"]).read_bytes())
    return run_job(job, study, output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build", help="article-disjoint splits and study data (CPU)")
    b.add_argument("output"); b.add_argument("--exclude-manifest", action="append", default=[], required=True)
    g = sub.add_parser("calibrate-gate", help="CPU gate calibration on V"); g.add_argument("study"); g.add_argument("output")
    pr = sub.add_parser("prepare", help="write a frozen launch"); pr.add_argument("kind", choices=KINDS)
    pr.add_argument("output"); pr.add_argument("--study", required=True); pr.add_argument("--gate")
    pr.add_argument("--previous-analysis", help="3-epoch control analysis, for the one 6-epoch rerun")
    r = sub.add_parser("resolve", help="CPU checks and frozen targets"); r.add_argument("launch")
    run = sub.add_parser("run", help="run the next job(s)"); run.add_argument("launch"); run.add_argument("--gpu", required=True)
    run.add_argument("--max-jobs", type=int, default=1); run.add_argument("--scratch-root")
    j = sub.add_parser("job", help="execute one job (used by run)"); j.add_argument("job"); j.add_argument("output")
    c = sub.add_parser("check", help="integrity and timing; no endpoint values"); c.add_argument("root")
    a = sub.add_parser("analyze", help="validation rules or analysis"); a.add_argument("root"); a.add_argument("output")
    a.add_argument("--datastore-analysis"); a.add_argument("--control-analysis")
    a.add_argument("--p0-launch", help="the P0 datastore launch folder (pilot analysis: P0 utility on V)")
    args = p.parse_args(argv)
    if args.action == "build":
        report = build(args.output, args.exclude_manifest)
        print(json.dumps({k: report[k] for k in ("study_sha256", "exclusions", "skips", "audit")}, indent=1))
    elif args.action == "calibrate-gate":
        gate = calibrate_gate(args.study, args.output)
        print(json.dumps({k: gate[k] for k in ("threshold", "gold_loss", "verbatim_probe_withheld_rate")}, indent=1))
    elif args.action == "prepare":
        launch = prepare(args.kind, args.output, args.study, args.gate, args.previous_analysis)
        print(f"Prepared {len(launch['jobs'])} {args.kind} job(s). Nothing resolved or run.")
        print(f"Next: python -m master_script.tools.grounded_fl_study resolve {Path(args.output).resolve() / 'launch.json'}")
    elif args.action == "resolve":
        execute(args.launch, resolve_only=True)
    elif args.action == "run":
        execute(args.launch, gpu=args.gpu, max_jobs=args.max_jobs, scratch_root=args.scratch_root)
    elif args.action == "job":
        run_single(args.job, args.output)
    elif args.action == "check":
        print(json.dumps(check(args.root), indent=1))
    else:
        report = analyze(args.root, args.output, args.datastore_analysis, args.control_analysis, args.p0_launch)
        print(json.dumps({k: v for k, v in report.items() if k not in ("learning_per_target",)}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
