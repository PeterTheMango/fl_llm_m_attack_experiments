"""Grounded FL-RAG generalization study (proposal draft 1, all recommended settings): Stage 0.

build-synth     the synthetic private-records study (CPU): fictitious people, article-disjoint
                groups, targets, probes, libraries and F, built by grounded.build_study
calibrate-gate  CPU gate calibration on V (the parent's procedure), for a new study file
prepare KIND    a frozen launch for one cell: datastore (P0 check), control (LoRA positive
                control), smoke (one V target x {RG, CB-AO} plus one shadow world per arm:
                timing and the LoRA learning check). Nothing is resolved or run.
resolve         CPU checks before any GPU work: targets, worlds, shadow worlds, the cell's
                tokenizer and chat template, token fit
run             one job at a time in its own process; weights retire after verification
                (LoRA adapters are kept)
job             execute one job (called by run)
check           integrity, timing and structure; never an endpoint value
analyze         the fixed Stage 0 rules: datastore sensitivity, the positive control, the
                smoke learning check

See outputs/grounded_generalization_proposal_20261012/{proposal.md,stage0_protocol.json,status.md}.
The parent study's tool (grounded_fl_study.py) is imported, never changed; a launch records
both tools' hashes and refuses if either changes. Attack effectiveness only; no privacy claim.
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

from master_script.core import generalization, grounded
from master_script.core.checkpoint_retention import file_digest, retire_checkpoints
from master_script.core.config import implementation_fingerprint
from master_script.core.generalization_job import JOB_SCHEMA, RESULT_SCHEMA, validate_job
from master_script.core.queue import write_json
from master_script.tools import grounded_fl_study as parent
from master_script.tools.collect_guard_traces import require_headroom

PROTOCOL = parent.ROOT.parents[1] / "outputs/grounded_generalization_proposal_20261012/stage0_protocol.json"
PROTOCOL_SCHEMA = "grounded_generalization_stage0_protocol_v1"
STUDY = "grounded_generalization"
LAUNCH_SCHEMA, COHORT_SCHEMA = "generalization_launch_v1", "generalization_cohort_v1"
KINDS = ("datastore", "control", "smoke")
# The parent's build (study sha256 d7eac161...), reused unchanged by every SQuAD cell.
SQUAD_STUDY_SHA256 = "d7eac161d178b6e855979fc879752a5d511516674e913788e590194d4dbecd92"
V_TARGET_SEED, V_SHADOW_SEED = 8100, 8150
CONTROL_TARGETS, CONTROL_MIN_STUDY_TOKENS, CONTROL_MIN_TOKENS = 12, 80, 64
CONTROL_STEPS, CONTROL_RERUN_STEPS, CONTROL_LR = 6, 12, 1e-3
PILOT_TARGETS, SMOKE_TARGETS = 4, 1
POPULATION, N_DOCUMENTS = 32, 16
LEAN = {"reference": True, "retention": True, "utility": True, "nq": True, "n_documents": N_DOCUMENTS,
        "population": POPULATION}
# Stage 0 work per cell (proposal §8): A-full inherits the parent's control and datastore
# check; Non-IID inherits A-LoRA's (same model, data and method).
NEEDS = {"datastore": ("A-LoRA", "Q-1.5B", "Q-3B", "Llama-1B", "Synth"),
         "control": ("A-LoRA", "Q-1.5B", "Q-3B", "Llama-1B", "Synth"),
         "smoke": tuple(generalization.CELLS)}
LEARNING_MARGIN = 0.05
INHERITS = {"Non-IID": "A-LoRA"}  # the cell whose datastore check and control another cell inherits


def require_datastore_cell(cell, datastore):
    """A datastore analysis must belong to the cell itself, or to the cell it inherits from."""
    if datastore is not None and datastore.get("cell") != INHERITS.get(cell, cell):
        raise ValueError(f"The datastore analysis belongs to {datastore.get('cell')!r}, not {INHERITS.get(cell, cell)!r}")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def tool_digests():
    return {"generalization": digest(__file__), "parent": digest(parent.__file__)}


def protocol():
    """The frozen Stage 0 protocol, checked against the constants this tool enforces."""
    value = json.loads(PROTOCOL.read_bytes())
    if value.get("schema") != PROTOCOL_SCHEMA or not str(value.get("status", "")).startswith("frozen"):
        raise ValueError("The generalization Stage 0 protocol is missing or not frozen")
    if value["cells"] != {k: {f: v[f] for f in ("model", "training", "data", "clients", "family")}
                          for k, v in generalization.CELLS.items()}:
        raise ValueError("The protocol's cells differ from core/generalization.CELLS")
    if value["models"] != generalization.MODELS or value["lora"] != generalization.LORA:
        raise ValueError("The protocol's models or LoRA recipe differ from core/generalization")
    if value["data"]["squad_study_sha256"] != SQUAD_STUDY_SHA256:
        raise ValueError("The protocol names another SQuAD build")
    stage0 = value["stage0"]
    if (stage0["datastore_lower_bound"], stage0["retention_lower_bound"], stage0["control_auc_lower_bound"]) != (
            parent.DATASTORE_LB, parent.RETENTION_LB, parent.CONTROL_AUC_LB):
        raise ValueError("Stage 0 thresholds differ from the parent's")
    return value


# ------------------------------------------------------------------- build

def build_synth(output, tokenizer=None, model_config=None, rows=None, layout=None, sizes=None):
    """Write the synthetic study and its build report (CPU)."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = generalization.synthetic_rows() if rows is None else rows
    if tokenizer is None:
        tokenizer, model_config = parent.load_tokenizer()
    settings = grounded.RagSettings(**parent.RAG)
    source = {**generalization.SYNTH_SOURCE, "rows": len(rows),
              "rows_sha256": sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
    study = grounded.build_study(rows, set(), exclusion_sources=[], source=source, layout=layout, sizes=sizes,
                                 token_count=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                                 fits=parent.fit_checker(tokenizer, model_config, settings))
    write_json(output / "study.json", study)
    report = {"schema": "generalization_build_report_v1", "dataset": "synth", "study_sha256": grounded.study_sha256(study),
              "study_file_sha256": digest(output / "study.json"), "study_tool_sha256": tool_digests(),
              "implementation_fingerprint": implementation_fingerprint(), "skips": study["skips"], "audit": study["audit"],
              "checks": ["article disjointness (planned slot exception only)", "one question per passage per training set",
                         "absent-fact probe rules", "no earlier cohort can occur (fresh fictitious records)",
                         f"every T/T_hold/U record fits {grounded.MAX_LENGTH} tokens untruncated (Qwen tokenizer)"]}
    write_json(output / "build-report.json", report)
    return report


def check_study(cell, study):
    """SQuAD cells reuse the parent build exactly; Synth uses a synthetic build."""
    if study.get("schema") != grounded.SCHEMA:
        raise ValueError("Not a grounded study file")
    grounded.audit_study(study)
    data = generalization.cell(cell)["data"]
    sha = grounded.study_sha256(study)
    if data == "squad" and sha != SQUAD_STUDY_SHA256:
        raise ValueError("SQuAD cells reuse the parent build (study sha256 d7eac161...) unchanged")
    if data == "synth" and study["source"].get("revision") != generalization.SYNTH_SOURCE["revision"]:
        raise ValueError("The Synth cell needs the synthetic build")
    return sha


# ------------------------------------------------------------------ prepare

def v_allocation(study):
    """V slices after every slice the parent study used (blind to outcomes): datastore = the parent's
    first 60 (P0 only, comparable with its check); control 12 (study token count >= 80), pilot 4, smoke 1."""
    used = parent.allocation(study)
    order = [t["index"] for t in study["cohorts"]["V"]["targets"]]
    taken = set(used["control"] + used["pilot"] + used["bridge"] + used["timing"] + used["datastore"])
    rest = [i for i in order if i not in taken and i > max(taken)]
    v = study["cohorts"]["V"]
    control, skipped = [], 0
    while len(control) < CONTROL_TARGETS:
        if not rest:
            raise ValueError("Not enough fresh V targets for the positive control")
        index = rest.pop(0)
        if v["passages"][v["targets"][index]["passage"]].get("tokens", 0) >= CONTROL_MIN_STUDY_TOKENS:
            control.append(index)
        else:
            skipped += 1
    pilot, smoke = rest[:PILOT_TARGETS], rest[PILOT_TARGETS:PILOT_TARGETS + SMOKE_TARGETS]
    if len(pilot) < PILOT_TARGETS or len(smoke) < SMOKE_TARGETS:
        raise ValueError("Not enough fresh V targets for the pilot and smoke slices")
    return {"datastore": used["datastore"], "control": control, "control_skipped_short_passages": skipped,
            "pilot": pilot, "smoke": smoke, "parent_slices_excluded": sorted(taken - set(used["datastore"]))}


def v_seed(allocation, index):
    """V target seed: 8100 + the target's slot in the study's new slices (control, pilot, smoke)."""
    slots = allocation["control"] + allocation["pilot"] + allocation["smoke"]
    return V_TARGET_SEED + slots.index(index)


def job_document(study, cell, kind, arm, seed, *, target_index=None, measurements=None, control=None,
                 datastore_targets=None, shadow=None, gate=None, lr=None):
    spec = generalization.cell(cell)
    profile = spec["profile"]
    config = {**parent.CONFIG, "model_id": profile["model_id"], "model_revision": profile["revision"],
              "reference_model_id": profile["model_id"], "seed": seed}
    training = {"method": spec["training"]}
    if spec["training"] == "lora":
        training.update({k: generalization.LORA[k] for k in ("rank", "alpha", "dropout", "dtype")},
                        lr=lr or generalization.LORA["lr"])
    else:
        training.update(lr=parent.CONFIG["client_lr"], dtype="float32")
    job = {"schema": JOB_SCHEMA, "study": STUDY, "kind": kind, "cell": cell, "arm": arm, "cohort": "V",
           "target_index": target_index, "config": config, "training": training, "clients": spec["clients"],
           "defense": {"mechanism": "none"}, "rag": dict(parent.RAG), "nli": dict(grounded.NLI),
           "measurements": dict(measurements or {}), "study_file": "study.json", "study_sha256": grounded.study_sha256(study)}
    for key, value in (("control", control), ("datastore_targets", datastore_targets), ("shadow", shadow)):
        if value is not None:
            job[key] = value
    if gate is not None:
        job["gate"] = {"threshold": gate["threshold"], "ngram": gate["ngram"], "sha256": gate["sha256"]}
    validate_job(job)
    return job


def needed(kind, cell):
    if cell not in NEEDS[kind]:
        raise ValueError(f"{kind} is not a Stage 0 step for {cell} (proposal §8)")


def plan(kind, cell, study, gate=None, previous=None):
    needed(kind, cell)
    slices = v_allocation(study)
    jobs = []
    if kind == "datastore":
        jobs.append(job_document(study, cell, "p0", "P0", V_TARGET_SEED, datastore_targets=slices["datastore"], gate=gate,
                                 measurements={"utility": True, "n_documents": len(study["cohorts"]["V"]["N"])}))
    elif kind == "control":
        steps = CONTROL_STEPS
        if previous is not None:
            if previous.get("schema") != "generalization_control_analysis_v1" or previous.get("status") != "rerun_at_12_steps" \
                    or previous.get("cell") != cell:
                raise ValueError("The 12-step rerun needs this cell's 6-step control analysis that calls for it")
            steps = CONTROL_RERUN_STEPS
        for index in slices["control"]:
            jobs.append(job_document(study, cell, "control", "RG", v_seed(slices, index), target_index=index,
                                     control={"steps": steps, "lr": CONTROL_LR}, measurements={}))
    else:
        lr = None
        if previous is not None:
            if previous.get("schema") != "generalization_smoke_analysis_v1" or previous.get("cell") != cell \
                    or previous.get("learning_check", {}).get("status") != "escalate":
                raise ValueError("The 5e-4 rerun needs this cell's smoke analysis that calls for it")
            lr = generalization.LORA["escalation_lr"]
        index = slices["smoke"][0]
        for arm in ("RG", "CB-AO"):
            jobs.append(job_document(study, cell, "paired", arm, v_seed(slices, index), target_index=index,
                                     measurements=LEAN, gate=gate, lr=lr))
        for arm in ("RG", "CB-AO"):
            jobs.append(job_document(study, cell, "shadow", arm, V_SHADOW_SEED, measurements={"population": POPULATION},
                                     shadow={"number": 0, "targets": [index], "in_row": [True]}, gate=gate, lr=lr))
    return jobs, slices


def prepare(kind, cell, output, study_path, gate_path=None, previous=None):
    """Write jobs and a frozen launch for one cell; nothing is resolved or run."""
    frozen = protocol()
    needed(kind, cell)
    study = json.loads(Path(study_path).read_bytes())
    check_study(cell, study)
    gate = None
    if gate_path is not None:
        gate = {**json.loads(Path(gate_path).read_bytes()), "sha256": digest(gate_path)}
        if gate.get("schema") != "grounded_gate_v1" or gate.get("study_sha256") != grounded.study_sha256(study):
            raise ValueError("Gate calibration belongs to another study")
    prior = json.loads(Path(previous).read_bytes()) if previous else None
    jobs, slices = plan(kind, cell, study, gate, prior)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(study_path, output / "study.json")
    (output / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    if gate_path is not None:
        shutil.copyfile(gate_path, output / "gate.json")
    entries = []
    for i, job in enumerate(jobs):
        name = f"job-{i:03d}-{job['kind']}-{job['arm']}-seed{job['config']['seed']}.json"
        write_json(output / name, job)
        entries.append({"file": name, "sha256": digest(output / name), "kind": job["kind"], "arm": job["arm"],
                        "target_index": job["target_index"], "seed": job["config"]["seed"]})
    launch = {"schema": LAUNCH_SCHEMA, "study": STUDY, "kind": kind, "cell": cell, "stage": 0, "created_unix": time.time(),
              "implementation_fingerprint": implementation_fingerprint(), "tool_sha256": tool_digests(),
              "protocol_file": "protocol.json", "protocol_sha256": digest(PROTOCOL), "protocol_status": frozen["status"],
              "study_file": "study.json", "study_sha256": grounded.study_sha256(study),
              "study_file_sha256": digest(output / "study.json"),
              "gate": None if gate is None else {"file": "gate.json", "sha256": gate["sha256"]},
              "previous_analysis": None if previous is None else {"path": str(Path(previous).resolve()), "sha256": digest(previous)},
              "allocation": slices, "jobs": entries, "role": "validation and tuning data (V); never reported as evidence",
              "retention": "retire full-model weights after each result is verified; keep LoRA adapters, JSON and private audit JSONL",
              "execution": "one job per process, one job per invocation by default"}
    write_json(output / "launch.json", launch)
    return launch


# ------------------------------------------------------------ resolve / run

def _load_launch(root):
    root = Path(root).resolve()
    launch = json.loads((root / "launch.json").read_bytes())
    if launch.get("schema") != LAUNCH_SCHEMA or launch.get("study") != STUDY:
        raise ValueError("Not a generalization-study launch")
    if launch["implementation_fingerprint"] != implementation_fingerprint():
        raise ValueError("Core code changed since the launch was prepared; prepare a new frozen launch")
    if launch["tool_sha256"] != tool_digests():
        raise ValueError("A study tool changed since the launch was prepared; prepare a new frozen launch")
    if digest(root / "protocol.json") != launch["protocol_sha256"] or digest(PROTOCOL) != launch["protocol_sha256"]:
        raise ValueError("Launch protocol differs from the committed stage0_protocol.json")
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


def load_cell_tokenizer(cell):
    """The cell's pinned tokenizer (gated models need the researcher's Hugging Face token on the server)."""
    from transformers import AutoConfig, AutoTokenizer
    profile = generalization.cell(cell)["profile"]
    tokenizer = AutoTokenizer.from_pretrained(profile["model_id"], revision=profile["revision"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return generalization.pinned(tokenizer, profile), AutoConfig.from_pretrained(profile["model_id"], revision=profile["revision"])


def resolve_targets(study, launch, jobs, tokenizer=None, model_config=None):
    """Target identities, world and shadow checks, the chat template and token fit, all on the CPU."""
    from master_script.core.scoring import validate_partition_tokens
    grounded.audit_study(study)
    excluded = set(study["exclusions"]["hashes"])
    settings = grounded.RagSettings(**parent.RAG)
    report = {"template": generalization.template_check(tokenizer) if tokenizer is not None else None, "targets": {}}
    cohort = study["cohorts"]["V"]
    text = lambda r: cohort["passages"][r["passage"]]["text"]
    for i, job in enumerate(jobs):
        validate_job(job)
        if job["study_sha256"] != launch["study_sha256"] or job["cell"] != launch["cell"] or job["cohort"] != "V":
            raise ValueError("A job names another study, cell or cohort (Stage 0 runs on V only)")
        arm = "RG" if job["arm"] == "P0" else job["arm"]
        encode = (lambda r: grounded.training_record(tokenizer, model_config, r, text(r), arm, settings)) if tokenizer else None
        if job["kind"] in ("paired", "control"):
            target = cohort["targets"][job["target_index"]]
            if {grounded.record_sha256(q) for q in [target["training"], *target["probes"]]} & excluded:
                raise ValueError(f"Job {i} selected an excluded earlier target question")
            for member in (True, False):
                records = generalization.world_records(study, "V", job["target_index"], job["config"]["seed"], member,
                                                       job["clients"])
                grounded.check_world(study, "V", records, job["target_index"], member)
                if encode:
                    validate_partition_tokens([[encode(r) for r in part] for part in records], tokenizer, grounded.MAX_LENGTH,
                                              target=encode(target["training"]), held_out=encode(target["hold"]),
                                              expected_membership=member)
            if job["kind"] == "control" and tokenizer is not None and \
                    len(tokenizer.encode(text(target), add_special_tokens=False)) < 2 * 32:
                raise ValueError("A control passage is shorter than 64 tokens under the cell's tokenizer")
            report["targets"][str(i)] = target["target_sha256"]
        elif job["kind"] == "shadow":
            shadow = job["shadow"]
            records = generalization.shadow_records(study, "V", shadow["targets"], shadow["in_row"], job["config"]["seed"],
                                                    job["clients"])
            generalization.check_shadow_world(study, "V", records, shadow["targets"], shadow["in_row"])
            if encode:
                validate_partition_tokens([[encode(r) for r in part] for part in records], tokenizer, grounded.MAX_LENGTH)
            report["targets"][str(i)] = sorted(cohort["targets"][k]["target_sha256"] for k in shadow["targets"])
        else:
            report["targets"][str(i)] = sorted(cohort["targets"][k]["target_sha256"] for k in job.get("datastore_targets", []))
    return report


def execute(launch_path, *, gpu=None, max_jobs=1, scratch_root=None, resolve_only=False, tokenizer=None, model_config=None):
    root = Path(launch_path).resolve().parent
    lock = root / ".generalization.lock"
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
            tokenizer, model_config = load_cell_tokenizer(launch["cell"])
        resolved = resolve_targets(study, launch, jobs, tokenizer, model_config)
        cohort = {"schema": COHORT_SCHEMA, "launch_sha256": state["launch_sha256"], "targets": resolved["targets"],
                  "template": resolved["template"], "role": launch["role"]}
        if cohort_path.exists() and json.loads(cohort_path.read_bytes()) != cohort:
            raise ValueError("Frozen cohort changed; preserve the original declaration")
        write_json(cohort_path, cohort)
        state["targets"] = resolved["targets"]
        write_json(state_path, state)
    if resolve_only:
        print("Resolved and froze the targets; worlds, shadow worlds, template and tokens checked. No GPU job started.")
        return state
    scratch_root = Path(scratch_root or tempfile.gettempdir()).resolve()
    if not scratch_root.is_dir():
        raise ValueError("Scratch root must be an existing writable directory")
    done_jobs = {r["job"] for r in state["completed"]}
    for index in [i for i in range(len(jobs)) if i not in done_jobs][:max_jobs]:
        require_headroom(root, 20)
        require_headroom(scratch_root, 4)
        scratch = Path(tempfile.mkdtemp(prefix="generalization-", dir=scratch_root))
        job_output = root / "results" / f"job-{index:03d}"
        state["active_job"] = {"index": index, "scratch": str(scratch)}
        write_json(state_path, state)
        command = [sys.executable, "-m", "master_script.tools.grounded_generalization_study", "job",
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
    if job["kind"] == "shadow" and sorted(t["target_sha256"] for t in result["targets"]) != target:
        raise ValueError("Shadow-world targets differ from the resolved targets")
    if job["kind"] == "p0" and sorted(r["target_sha256"] for r in result["worlds"]["P0"].get("datastore", [])) != target:
        raise ValueError("P0 datastore targets differ from the resolved targets")
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


def _structure(result):
    """Counts only: every expected measurement row is present."""
    counts = parent._structure(result)
    for world, body in result.get("worlds", {}).items():
        if "population" in body:
            counts[f"{world}_population_rows"] = len(body["population"])
        if "never_documents" in body:
            counts[f"{world}_never_documents"] = len(body["never_documents"])
    if "targets" in result:
        counts["shadow_target_rows"] = len(result["targets"])
    return counts


def check(root):
    """Integrity, timing and structure so far: an allowlist, never an endpoint value."""
    launch, state, _, rows = completed(root, partial=True)
    report = []
    for row in rows:
        result = row["result"]
        lengths = result.get("sequence_lengths") or {}
        skew = result.get("client_skew") or {}
        report.append({"job": row["index"], "kind": result["kind"], "cell": result["cell"], "arm": result["arm"],
                       "seed": result["seed"], "target_sha256": (result.get("target_sha256") or "")[:12] or None,
                       "training": {k: result["training"][k] for k in ("method", "lr", "dtype") if k in result["training"]},
                       "timings": parent._seconds(result["timings"]),
                       "sequence_lengths": {k: lengths[k] for k in ("max_tokens", "mean_tokens") if k in lengths} or None,
                       "client_skew": {k: skew[k] for k in ("effective_articles_per_client", "mean_pairwise_js_divergence_nats")
                                       if k in skew} or None,
                       "rows": _structure(result), "integrity": "verified"})
    return {"kind": launch["kind"], "cell": launch["cell"], "completed": len(rows), "planned": len(launch["jobs"]),
            "active_job": state.get("active_job"), "jobs": report,
            "note": "Endpoint and validation values are withheld until analyze."}


# ------------------------------------------------------------- statistics

def lira_fit(shadow_phi, in_mask):
    """Online LiRA with the fixed-variance estimator (Carlini et al. 2022).

    shadow_phi, in_mask: (shadows, targets). Per-target IN and OUT means; one IN and one
    OUT variance pooled over every target's deviations from its own means."""
    import numpy as np
    phi, mask = np.asarray(shadow_phi, float), np.asarray(in_mask, bool)
    if phi.shape != mask.shape or (mask.sum(axis=0) < 1).any() or ((~mask).sum(axis=0) < 1).any():
        raise ValueError("Every target needs at least one IN and one OUT shadow score")
    mu_in = np.where(mask, phi, 0).sum(axis=0) / mask.sum(axis=0)
    mu_out = np.where(~mask, phi, 0).sum(axis=0) / (~mask).sum(axis=0)
    dev_in, dev_out = (phi - mu_in)[mask], (phi - mu_out)[~mask]
    sd_in = float(np.sqrt((dev_in ** 2).sum() / max(len(dev_in) - mu_in.size, 1)))
    sd_out = float(np.sqrt((dev_out ** 2).sum() / max(len(dev_out) - mu_out.size, 1)))
    return {"mu_in": mu_in, "mu_out": mu_out, "sd_in": max(sd_in, 1e-12), "sd_out": max(sd_out, 1e-12)}


def lira_scores(phi, fit):
    """Online LiRA: log N(phi; mu_in, sd_in^2) - log N(phi; mu_out, sd_out^2) per target."""
    import numpy as np
    phi = np.asarray(phi, float)

    def log_normal(x, mu, sd):
        return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd)
    return log_normal(phi, fit["mu_in"], fit["sd_in"]) - log_normal(phi, fit["mu_out"], fit["sd_out"])


def lira_offline_scores(phi, fit):
    """Offline LiRA (OUT shadows only): one-sided standardized distance above the OUT mean."""
    import numpy as np
    return (np.asarray(phi, float) - fit["mu_out"]) / fit["sd_out"]


def rmia_scores(target_phi, target_ref_phi, population_phi, population_ref_phi, gamma=1.0):
    """RMIA (Zarifzadeh et al. 2024) with reference models: the share of population records z for which
    Pr(x|theta)/Pr(x) over Pr(z|theta)/Pr(z) exceeds gamma. Inputs are log-likelihoods; *_ref_phi are
    the mean log-likelihoods over reference (OUT shadow) models. Shapes: targets (n,), population (m,)."""
    import numpy as np
    ratio_x = np.asarray(target_phi, float) - np.asarray(target_ref_phi, float)
    ratio_z = np.asarray(population_phi, float) - np.asarray(population_ref_phi, float)
    return (ratio_x[:, None] - ratio_z[None, :] > math.log(gamma)).mean(axis=1)


def phi(scores, form):
    """LiRA's per-model statistic: minus the summed NLL of the scored tokens (mean NLL x tokens)."""
    return -scores[form]["model_nll"] * scores[form]["tokens"]


# ---------------------------------------------------------------- analyze

def analyze_datastore(rows):
    """The parent's P0 datastore-sensitivity rule (lower bound >= 0.70), plus P0's mean V utility
    for the smoke learning check."""
    report = parent.analyze_datastore(rows)
    report["schema"] = "generalization_datastore_analysis_v1"
    utility = rows[0]["result"]["worlds"]["P0"].get("utility", {}).get("F", [])
    report["p0_v_utility_mean_f1"] = sum(r["f1"] for r in utility) / len(utility) if utility else None
    return report


def analyze_control(rows, datastore=None):
    """The parent's control rules with the LoRA escalation: fewer than 10 of 12 learned at 6 steps ->
    one rerun at 12 steps; still fewer -> sensitivity unverified (H4 secondary in this cell)."""
    import numpy as np
    steps = {r["result"]["control"]["steps"] for r in rows}
    cells = {r["result"]["cell"] for r in rows}
    if len(steps) != 1 or len(cells) != 1 or len(rows) != CONTROL_TARGETS:
        raise ValueError(f"The control needs {CONTROL_TARGETS} targets of one cell at one step count")
    steps, cell = steps.pop(), cells.pop()
    require_datastore_cell(cell, datastore)
    per_target = []
    for r in rows:
        w1, w0 = r["result"]["worlds"]["W1"], r["result"]["worlds"]["W0"]
        per_target.append({"target": r["result"]["target_sha256"], "learned": parent.learned(w1, w0),
                           "nll_ratio": w1["learning"]["passage_nll"] / w0["learning"]["passage_nll"],
                           "continuation_match": w1["learning"]["continuation_match"]})
    count = sum(t["learned"] for t in per_target)
    report = {"schema": "generalization_control_analysis_v1", "cell": cell, "steps": steps, "learned": count,
              "targets": len(rows), "learning_per_target": per_target,
              "method": {"bootstrap": parent.BOOTSTRAP, "rng_seed": parent.RNG_SEED, "retention_lower_bound": parent.RETENTION_LB,
                         "auc_lower_bound": parent.CONTROL_AUC_LB}}
    if count < parent.CONTROL_LEARNED:
        report["status"] = "rerun_at_12_steps" if steps == CONTROL_STEPS else "failed_to_learn"
        report["H4"] = ("pending the one 12-step rerun" if steps == CONTROL_STEPS else
                        "sensitivity unverified: H4 is secondary in this cell")
        report["h4_primary"] = None if steps == CONTROL_STEPS else False
        return report
    kept = [r for r, t in zip(rows, per_target) if t["learned"]]
    retention = np.array([r["result"]["worlds"]["W1"]["retention"]["f1"] - r["result"]["worlds"]["W0"]["retention"]["f1"]
                          for r in kept])
    idx = parent.target_indices(len(kept))
    passes = parent.interval(retention[idx].mean(axis=1))[0] > parent.RETENTION_LB
    report["retention_sensitivity"] = {"learned_targets": len(kept), "mean_r": float(retention.mean()),
                                       "ci95": parent.interval(retention[idx].mean(axis=1)), "passes": bool(passes)}
    report["status"] = "sensitive" if passes else "not_evaluable"
    report["h4_primary"] = bool(passes)
    report["H4"] = ("retention sensitivity verified: H4 is primary in this cell" if passes else
                    "not evaluable: the probes cannot detect retention even when the passage was learned; H4 is secondary")
    eligible = datastore["eligible_scorers"] if datastore else list(parent.NQ_SCORERS)
    auc_report = {}
    for scorer in parent.NQ_SCORERS:
        pos = np.array([parent.doc_score(r["result"]["worlds"]["W1"]["natural_questions"]["training_only"], scorer) for r in kept])
        neg = np.array([parent.doc_score(r["result"]["worlds"]["W0"]["natural_questions"]["neither"], scorer) for r in kept])
        ci = parent.interval(parent.auc_resampled(pos, neg, idx))
        auc_report[scorer] = {"pooled_auc": parent.auc(pos, neg), "ci95": ci, "sensitive": ci[0] >= parent.CONTROL_AUC_LB,
                              "eligible_from_datastore_check": scorer in eligible}
    report["secondary_membership_auc"] = {**auc_report, "note": "secondary only; never affects H4's primary status"}
    return report


def analyze_smoke(rows, datastore=None):
    """Timings, and for LoRA cells the pre-set learning check (proposal §5.2): RG's mean client loss
    falls from round 1 to the last round, and RG's V RAG F1 beats P0's V F1 by at least 0.05.
    The first failure calls for the one 5e-4 rerun; a second failure drops the cell."""
    cell = rows[0]["result"]["cell"]
    method = rows[0]["result"]["training"]["method"]
    report = {"schema": "generalization_smoke_analysis_v1", "cell": cell, "training": method,
              "role": "timing and learning check on one V target; tuning data, never evidence",
              "timings": [{"kind": r["result"]["kind"], "arm": r["result"]["arm"], **parent._seconds(r["result"]["timings"])}
                          for r in rows],
              "sequence_lengths": [r["result"].get("sequence_lengths") for r in rows if r["result"]["kind"] == "paired"]}
    require_datastore_cell(cell, datastore)
    if method != "lora":
        report["learning_check"] = {"status": "not_applicable", "note": "A-full is the parent's full fine-tuning path"}
        return report
    if datastore is None or datastore.get("p0_v_utility_mean_f1") is None:
        raise ValueError("The LoRA learning check needs the cell's datastore analysis (P0's V utility)")
    rg = next(r["result"] for r in rows if r["result"]["kind"] == "paired" and r["result"]["arm"] == "RG")
    losses = [h["mean_client_loss"] for h in rg["federated_history_W1"]]
    utility = rg["worlds"]["W1"]["utility"]["F"]
    rg_f1 = sum(r["f1"] for r in utility) / len(utility)
    lr = rg["training"]["lr"]
    passes = losses[-1] < losses[0] and rg_f1 - datastore["p0_v_utility_mean_f1"] >= LEARNING_MARGIN
    status = "passes" if passes else ("escalate" if lr == generalization.LORA["lr"] else "drop_cell")
    report["learning_check"] = {"status": status, "lr": lr, "round_mean_client_loss": losses, "rg_v_f1": rg_f1,
                                "p0_v_f1": datastore["p0_v_utility_mean_f1"], "margin": LEARNING_MARGIN}
    return report


def analyze(root, output, datastore=None, control=None):
    launch, _, study, rows = completed(root)
    datastore = json.loads(Path(datastore).read_bytes()) if datastore else None
    if launch["kind"] == "datastore":
        report = analyze_datastore(rows)
    elif launch["kind"] == "control":
        report = analyze_control(rows, datastore)
    else:
        report = analyze_smoke(rows, datastore)
    report.update(cell=launch["cell"], launch_sha256=digest(Path(root) / "launch.json"), tool_sha256=tool_digests(),
                  implementation_fingerprint=launch["implementation_fingerprint"], protocol_sha256=launch["protocol_sha256"],
                  study_sha256=launch["study_sha256"], not_a_privacy_claim="Attack-effectiveness and validation measurements only")
    if Path(output).exists():
        raise ValueError("Analysis already exists; write a new file")
    write_json(output, report)
    return report


# ------------------------------------------------------------------ main

def run_single(job_path, output):
    from master_script.core.generalization_job import run_job
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_bytes())
    study = json.loads((job_path.parent / job["study_file"]).read_bytes())
    return run_job(job, study, output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build-synth", help="the synthetic private-records study (CPU)"); b.add_argument("output")
    g = sub.add_parser("calibrate-gate", help="CPU gate calibration on V"); g.add_argument("study"); g.add_argument("output")
    pr = sub.add_parser("prepare", help="write a frozen launch for one cell"); pr.add_argument("kind", choices=KINDS)
    pr.add_argument("output"); pr.add_argument("--cell", required=True, choices=sorted(generalization.CELLS))
    pr.add_argument("--study", required=True); pr.add_argument("--gate")
    pr.add_argument("--previous-analysis", help="this cell's control analysis (12-step rerun) or smoke analysis (5e-4 rerun)")
    r = sub.add_parser("resolve", help="CPU checks and frozen targets"); r.add_argument("launch")
    run = sub.add_parser("run", help="run the next job(s)"); run.add_argument("launch"); run.add_argument("--gpu", required=True)
    run.add_argument("--max-jobs", type=int, default=1); run.add_argument("--scratch-root")
    j = sub.add_parser("job", help="execute one job (used by run)"); j.add_argument("job"); j.add_argument("output")
    c = sub.add_parser("check", help="integrity and timing; no endpoint values"); c.add_argument("root")
    a = sub.add_parser("analyze", help="the fixed Stage 0 rules"); a.add_argument("root"); a.add_argument("output")
    a.add_argument("--datastore-analysis", help="the cell's datastore analysis (control: eligible scorers; smoke: P0 utility)")
    args = p.parse_args(argv)
    if args.action == "build-synth":
        report = build_synth(args.output)
        print(json.dumps({k: report[k] for k in ("study_sha256", "skips", "audit")}, indent=1))
    elif args.action == "calibrate-gate":
        gate = parent.calibrate_gate(args.study, args.output)
        print(json.dumps({k: gate[k] for k in ("threshold", "gold_loss", "verbatim_probe_withheld_rate")}, indent=1))
    elif args.action == "prepare":
        launch = prepare(args.kind, args.cell, args.output, args.study, args.gate, args.previous_analysis)
        print(f"Prepared {len(launch['jobs'])} {args.kind} job(s) for {args.cell}. Nothing resolved or run.")
        print(f"Next: python -m master_script.tools.grounded_generalization_study resolve {Path(args.output).resolve() / 'launch.json'}")
    elif args.action == "resolve":
        execute(args.launch, resolve_only=True)
    elif args.action == "run":
        execute(args.launch, gpu=args.gpu, max_jobs=args.max_jobs, scratch_root=args.scratch_root)
    elif args.action == "job":
        run_single(args.job, args.output)
    elif args.action == "check":
        print(json.dumps(check(args.root), indent=1))
    else:
        report = analyze(args.root, args.output, args.datastore_analysis)
        print(json.dumps({k: v for k, v in report.items() if k != "learning_per_target"}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
