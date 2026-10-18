"""Grounded federated RAG study (proposal revision 4): build, prepare, run, check, analyze.

build           article-disjoint SQuAD splits, targets, probes, libraries and F, with
                every earlier target question excluded by recomputed old record hashes
calibrate-gate  CPU calibration of the verbatim-overlap library gate on V
prepare KIND    a frozen launch: datastore (P0 check), control (positive control),
                timing (full paired jobs), pilot (Stage 1), final (Stage 2 confirmation,
                from the frozen stage2_protocol.json), diagnostic (the world-order
                diagnostic on V), repeat (the deterministic same-world repeat on V).
                Nothing is resolved or run.
resolve         CPU checks before any GPU work: targets, worlds, tokens, exclusions
run             one job at a time in its own process; weights retire after verification
job             execute one job (called by run)
check           integrity and timing of completed jobs; never an endpoint value
analyze         the fixed validation rules, the pilot analysis or the §9 confirmation
describe        the descriptive secondaries (§9.2) of the finished final launch, read-only
                through frozen_results; never re-runs analyze or changes a primary outcome

See outputs/retrieval_grounded_fl_proposal_20260930/{proposal.md,stage01_protocol.json,stage2_protocol.json,status.md}.
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
from master_script.core.grounded_job import (BETWEEN, BETWEEN_MEASUREMENTS, CAUSAL, DIAGNOSTIC_ORDERS, JOB_SCHEMA,
                                             RESULT_SCHEMA, validate_job)
from master_script.core.queue import write_json
from master_script.tools.build_guard_stage import ROOT
from master_script.tools.causal_attack_validation import BASE as CAUSAL_BASE, DEFAULTS as CAUSAL_DEFAULTS
from master_script.tools.collect_guard_traces import SMOKE_TARGET, require_headroom

PROTOCOL = ROOT.parents[1] / "outputs/retrieval_grounded_fl_proposal_20260930/stage01_protocol.json"
STAGE2_PROTOCOL = PROTOCOL.with_name("stage2_protocol.json")
STAGE2_SCHEMA = "grounded_fl_stage2_protocol_v1"
STUDY = "grounded_fl"
LAUNCH_SCHEMA, COHORT_SCHEMA = "grounded_launch_v1", "grounded_cohort_v1"
KINDS = ("datastore", "control", "timing", "pilot", "final", "diagnostic", "repeat")
V_SEED_BAND, FINAL_SEED_BAND = (8000, 8099), (9000, 9099)
PUBLIC_SEEDS = (8095, 8096, 8097, 8098, 8099)
DATASTORE_TARGETS, CONTROL_TARGETS, PILOT_TARGETS, BRIDGE_TARGETS = 60, 12, 4, 4
DIAGNOSTIC_TARGETS = (12, 6, 60)  # world-order diagnostic: default, minimum and maximum datastore-slice V targets
REPEAT_TARGETS = (4, 2, 12)  # deterministic repeat: default, minimum and maximum datastore-slice V targets
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


def protocol_path(kind):
    return STAGE2_PROTOCOL if kind == "final" else PROTOCOL


ARMS2, DP_BUDGETS = ("RG", "CB-AO"), (64, 16)
MIN_FINAL_TARGETS, MIN_DP_TARGETS = 20, 10  # option (b): the floors Stage 2 may raise, never lower


def stage2_protocol(path=None):
    """The frozen Stage 2 protocol, checked against every rule it may only tighten."""
    _protocol()
    path = Path(path or STAGE2_PROTOCOL)
    if not path.exists():
        raise ValueError("No Stage 2 protocol: stage2_protocol.json is written after decisions A and B")
    protocol = json.loads(path.read_bytes())
    if protocol.get("schema") != STAGE2_SCHEMA or not str(protocol.get("status", "")).startswith("frozen"):
        raise ValueError("The Stage 2 protocol is missing or not frozen; prepare final refuses a draft")
    if protocol.get("stage01_protocol_sha256") != digest(PROTOCOL):
        raise ValueError("The Stage 2 protocol names another Stage 0/1 protocol")
    design = protocol["design"]
    if design["cohort"] != "final" or design["seed_band"] != list(FINAL_SEED_BAND):
        raise ValueError("Stage 2 uses fresh final targets in the seed band 9000-9099")
    targets = design["targets"]
    dp = {int(k): v for k, v in targets["dp"].items()}
    if not isinstance(targets["inf"], int) or targets["inf"] < MIN_FINAL_TARGETS:
        raise ValueError(f"Stage 2 needs at least {MIN_FINAL_TARGETS} targets at epsilon inf (option (b))")
    if set(dp) - set(DP_BUDGETS) or not isinstance(dp.get(16), int):
        raise ValueError("H6 is always kept at epsilon 16; only 64 and 16 are pre-registered budgets")
    if any(not isinstance(n, int) or not MIN_DP_TARGETS <= n <= targets["inf"] for n in dp.values()):
        raise ValueError(f"Each DP arm takes {MIN_DP_TARGETS} or more of the epsilon-inf targets")
    seeds = design["public_seeds"]
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(not FINAL_SEED_BAND[0] <= x <= FINAL_SEED_BAND[1] for x in seeds) \
            or FINAL_SEED_BAND[0] + targets["inf"] > min(seeds):
        raise ValueError("RG-public takes 5 final-band seeds above every target seed")
    choices = protocol["attacker_choices"]
    for arm in ARMS2:
        c = choices[arm]
        if c["reference_form"] not in FORMS or c["causal_direction"] not in ("record", "template") or c["nq_scorer"] not in NQ_SCORERS:
            raise ValueError(f"Attacker choices for {arm} are incomplete")
    h2 = protocol["hypotheses"]["H2"]
    if h2["primary_pass"] not in ("plain", "release_noise") or not design["release_noise"] and h2["primary_pass"] == "release_noise":
        raise ValueError("H2's release-noise primary needs the release-noise pass")
    if protocol["h4_primary"] is not True:
        raise ValueError("The 6-epoch positive control passed: H4 is primary (seven primaries, 99.3%)")
    stats = protocol["statistics"]
    if stats["margins"] != {"auc": MARGIN, "f1": MARGIN} or stats["primaries"] != (7 if protocol["h4_primary"] else 6) \
            or abs(stats["decision_level"] - decision_level(protocol["h4_primary"])) > 1e-12:
        raise ValueError("Stage 2 statistics differ from the pre-registered margins and Bonferroni level")
    if protocol["design"]["gate"]["ngram"] < 1:
        raise ValueError("The library gate must be declared")
    return protocol


def final_measurements(arm, protocol):
    """Every pre-registered measurement; the causal attack in the arm's frozen direction only."""
    return {**FULL, "causal_directions": [protocol["attacker_choices"][arm]["causal_direction"]],
            "release_noise": bool(protocol["design"]["release_noise"])}


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
                 control=None, datastore_targets=None, gate=None, diagnostic=None):
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
    if diagnostic is not None:
        job["diagnostic"] = dict(diagnostic)
    validate_job(job)
    return job


def expected_epsilon(epsilon, sizes=None, config=None):
    """The busiest client's accounted steps: 33 records, batch 2, 1 epoch, 3 rounds = 51."""
    sizes, config = sizes or grounded.SIZES, config or CONFIG
    steps = math.ceil((sizes["records_per_client"] + 1) / config["local_batch_size"]) * config["local_epochs"] * config["federated_rounds"]
    return steps, privacy_bound(steps, SIGMA[epsilon], DELTA)["epsilon"]


def final_allocation(study, protocol):
    """Fresh final targets in eligible order: the first n at epsilon inf, the first n_eps of those per DP arm."""
    order = [t["index"] for t in study["cohorts"]["final"]["targets"]]
    targets = protocol["design"]["targets"]
    if len(order) < targets["inf"]:
        raise ValueError(f"The final cohort has {len(order)} eligible targets; the protocol needs {targets['inf']}")
    chosen = order[:targets["inf"]]
    return {"inf": chosen, **{str(eps): chosen[:n] for eps, n in targets["dp"].items()}}


def plan(kind, study, gate=None, previous=None, protocol=None, targets=None):
    if kind == "final":
        if gate is None:
            raise ValueError("The final launch records the calibrated library gate")
        if (gate["threshold"], gate["ngram"]) != (protocol["design"]["gate"]["threshold"], protocol["design"]["gate"]["ngram"]):
            raise ValueError("The gate differs from the one the Stage 2 protocol names")
        slices = final_allocation(study, protocol)
        jobs = [job_document(study, "p0", "P0", "final", measurements={"utility": True}, gate=gate)]
        jobs += [job_document(study, "public", "RG-public", "final", seed=seed, measurements={"utility": True}, gate=gate)
                 for seed in protocol["design"]["public_seeds"]]
        budgets = ["inf"] + [eps for eps in DP_BUDGETS if str(eps) in slices]
        for index in slices["inf"]:
            for epsilon in budgets:
                if index in slices[str(epsilon)]:
                    for arm in ARMS2:
                        jobs.append(job_document(study, "paired", arm, "final", epsilon, index,
                                                 measurements=final_measurements(arm, protocol), gate=gate))
        return jobs, slices
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
    elif kind == "diagnostic":
        # V targets that no paired job trained on (the P0 datastore slice); the same seed per target in every
        # order, so the first-trained W1 of "W1-W1" and of "W1-W0" are the same setup.
        count = DIAGNOSTIC_TARGETS[0] if targets is None else targets
        if type(count) is not int or not DIAGNOSTIC_TARGETS[1] <= count <= min(DIAGNOSTIC_TARGETS[2], len(slices["datastore"])):
            raise ValueError(f"The diagnostic takes {DIAGNOSTIC_TARGETS[1]}-{DIAGNOSTIC_TARGETS[2]} datastore-slice V targets")
        slices["diagnostic"] = slices["datastore"][:count]
        for order in ("W1-W1", "W1-W0", "W0-W1"):
            for index in slices["diagnostic"]:
                jobs.append(job_document(study, "diagnostic", "RG", "V", "inf", index, measurements={"utility": True},
                                         gate=gate, diagnostic={"order": order}))
    elif kind == "repeat":
        # W1 trained twice per target with deterministic client kernels: once with nothing in between (the
        # determinism control), once with a Stage 2 RG member world's work in between.
        count = REPEAT_TARGETS[0] if targets is None else targets
        if type(count) is not int or not REPEAT_TARGETS[1] <= count <= min(REPEAT_TARGETS[2], len(slices["datastore"])):
            raise ValueError(f"The repeat takes {REPEAT_TARGETS[1]}-{REPEAT_TARGETS[2]} datastore-slice V targets")
        slices["repeat"] = slices["datastore"][:count]
        for order, measurements in (("W1-W1", {"utility": True}), (BETWEEN, BETWEEN_MEASUREMENTS)):
            for index in slices["repeat"]:
                jobs.append(job_document(study, "diagnostic", "RG", "V", "inf", index, measurements=measurements,
                                         gate=gate, diagnostic={"order": order, "deterministic": True}))
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


def prepare(kind, output, study_path, gate_path=None, previous=None, targets=None):
    """Write jobs and a frozen launch; nothing is resolved or run."""
    _protocol()  # refuses unless the Stage 0/1 protocol is present and frozen
    protocol = stage2_protocol() if kind == "final" else None
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
    if targets is not None and kind not in ("diagnostic", "repeat"):
        raise ValueError("--targets applies to the diagnostic and the repeat only; every other launch's targets are fixed by rule")
    jobs, slices = plan(kind, study, gate, prior, protocol, targets)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(study_path, output / "study.json")
    (output / "protocol.json").write_bytes(protocol_path(kind).read_bytes())
    if kind == "final":
        (output / "stage01_protocol.json").write_bytes(PROTOCOL.read_bytes())
    if gate_path is not None:
        shutil.copyfile(gate_path, output / "gate.json")
    entries = []
    for i, job in enumerate(jobs):
        name = f"job-{i:03d}-{job['kind']}-{job['arm']}-eps{job['epsilon']}-seed{job['config']['seed']}.json"
        write_json(output / name, job)
        entries.append({"file": name, "sha256": digest(output / name), "kind": job["kind"], "arm": job["arm"],
                        "epsilon": job["epsilon"], "cohort": job["cohort"], "target_index": job["target_index"],
                        "seed": job["config"]["seed"]})
    launch = {"schema": LAUNCH_SCHEMA, "study": STUDY, "kind": kind, "stage": {"pilot": 1, "final": 2}.get(kind, 0),
              "created_unix": time.time(), "implementation_fingerprint": implementation_fingerprint(),
              "study_tool_sha256": tool_digest(), "protocol_file": "protocol.json", "protocol_sha256": digest(protocol_path(kind)),
              **({"stage01_protocol_sha256": digest(PROTOCOL)} if kind == "final" else {}),
              "study_file": "study.json", "study_sha256": grounded.study_sha256(study),
              "study_file_sha256": digest(output / "study.json"),
              "gate": None if gate is None else {"file": "gate.json", "sha256": gate["sha256"]},
              "previous_analysis": None if previous is None else {"path": str(Path(previous).resolve()), "sha256": digest(previous)},
              "allocation": slices, "jobs": entries,
              "role": {"final": "confirmation", "diagnostic": "world-order diagnostic on V (tuning data); never evidence",
                       "repeat": "deterministic same-world repeat on V (tuning data); never evidence"}.get(
                  kind, "tuning and validation data (V); never reported as evidence"),
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
    committed = protocol_path(launch["kind"])
    if digest(root / "protocol.json") != launch["protocol_sha256"] or digest(committed) != launch["protocol_sha256"]:
        raise ValueError(f"Launch protocol differs from the committed {committed.name}")
    if launch["kind"] == "final" and (digest(root / "stage01_protocol.json") != launch["stage01_protocol_sha256"]
                                      or digest(PROTOCOL) != launch["stage01_protocol_sha256"]):
        raise ValueError("Launch protocol differs from the committed stage01_protocol.json")
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
        if job["kind"] in ("paired", "control", "diagnostic"):
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
    if job["kind"] in ("paired", "control", "diagnostic") and result.get("target_sha256") != target:
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


def _seconds(timings):
    """Only wall-clock entries, so no value can reach check through the timings."""
    return {k: float(v) for k, v in timings.items() if k.endswith("_seconds") and isinstance(v, (int, float))}


def check(root):
    """Integrity, epsilon records and timing so far. No score, AUC, F1, refusal or learning value:
    every field is chosen explicitly (an allowlist), never copied from a result wholesale."""
    launch, state, _, rows = completed(root, partial=True)
    report = []
    for row in rows:
        result = row["result"]
        lengths = result.get("sequence_lengths") or {}
        report.append({"job": row["index"], "kind": result["kind"], "arm": result["arm"],
                       "epsilon_budget": result["epsilon_budget"], "seed": result["seed"],
                       "target_sha256": (result.get("target_sha256") or "")[:12] or None,
                       "accounted_epsilon": {w: p["epsilon"] for w, p in result.get("privacy", {}).items()},
                       "timings": _seconds(result["timings"]),
                       "sequence_lengths": {k: lengths[k] for k in ("max_tokens", "mean_tokens") if k in lengths} or None,
                       "causal_seconds": {f"{d}_{name}": passes[name]["seconds"] for d, passes in result.get("causal", {}).items()
                                          for name in ("plain", "release_noise") if name in passes},
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


def analyze_diagnostic(rows):
    """World-order diagnostic (V, tuning data): per order, the second training minus the first in RAG F1,
    public-library F1 and each round's mean client loss, and W1 - W0 where the order holds both,
    with 95% target-bootstrap intervals."""
    import numpy as np
    report = {"schema": "grounded_diagnostic_analysis_v1", "role": "world-order diagnostic on V; tuning data, never evidence",
              "method": {"bootstrap": BOOTSTRAP, "rng_seed": RNG_SEED, "unit": "target", "intervals": "95%"},
              "orders": {}}
    for order, members in DIAGNOSTIC_ORDERS.items():
        results = sorted((r["result"] for r in rows if r["result"].get("order") == order), key=lambda r: r["target_sha256"])
        if not results:
            continue
        idx = target_indices(len(results))
        f1 = lambda part: np.array([[np.mean([q["f1"] for q in r["worlds"][run]["utility"][part]]) for run in ("run1", "run2")]
                                    for r in results])
        loss = np.array([[[h["mean_client_loss"] for h in r[f"federated_history_{run}"]] for run in ("run1", "run2")]
                         for r in results], float)
        entry = {"targets": len(results), "members": {"run1": members[0], "run2": members[1]}}
        for part, label in (("F", "rag_f1"), ("F_P", "public_f1")):
            values = f1(part)
            second = values[:, 1] - values[:, 0]
            entry[label] = {"run1": float(values[:, 0].mean()), "run2": float(values[:, 1].mean()),
                            "second_minus_first": _ci(second.mean(), second[idx].mean(axis=1)),
                            "second_better": int((second > 0).sum())}
            if members[0] != members[1]:
                gap = second if members[1] else -second
                entry[label]["W1_minus_W0"] = _ci(gap.mean(), gap[idx].mean(axis=1))
        last = loss[:, 1, -1] - loss[:, 0, -1]
        entry["mean_client_loss"] = {"run1": loss[:, 0].mean(axis=0).tolist(), "run2": loss[:, 1].mean(axis=0).tolist(),
                                     "second_minus_first_last_round": _ci(last.mean(), last[idx].mean(axis=1)),
                                     "second_higher_last_round": int((last > 0).sum())}
        if all("parameters_sha256" in r["worlds"]["run1"] for r in results):
            # Exact comparisons: identical final weights and identical per-round client losses.
            same = lambda r: r["worlds"]["run1"]["parameters_sha256"] == r["worlds"]["run2"]["parameters_sha256"]
            client = lambda r, run: [h.get("client_losses") for h in r[f"federated_history_{run}"]]
            entry["exact"] = {
                "deterministic_jobs": sum(bool(r.get("deterministic")) for r in results),
                "deterministic_clients": all(o.get("deterministic") for r in results for run in ("run1", "run2")
                                             for h in r[f"federated_history_{run}"] for o in h.get("client_outcomes", [])),
                "identical_weights": sum(same(r) for r in results),
                "identical_client_losses": sum(client(r, "run1") == client(r, "run2") for r in results),
                "per_target": [{"target": r["target_sha256"][:12], "identical_weights": same(r),
                                "last_round_loss_second_minus_first": float(
                                    r["federated_history_run2"][-1]["mean_client_loss"]
                                    - r["federated_history_run1"][-1]["mean_client_loss"])} for r in results]}
        report["orders"][order] = entry
    return report


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

def hypotheses(paired, public, p0, choices, h4_primary=True, clusters=None, sink=None, h2_pass="plain"):
    """H1-H7 and the cross-channel rule on final data (§9). paired: per-target records;
    public: RG-public per-seed F1 vectors; p0: the P0 F1 vector; clusters: F article per question.
    sink, when given, receives each difference's bootstrap draws (pilot precision projections).
    h2_pass names H2's primary causal pass; "release_noise" is the Stage 2 primary (decision C2),
    and the no-noise pass is then reported as a secondary with the same target draws."""
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
    def h2(name, label):
        causal = lambda rows, arm: np.array([r["causal"][choices[arm]["causal_direction"]][name] for r in rows])
        rg_c, cb_c = causal(arms["RG"], "RG"), causal(arms["CB-AO"], "CB-AO")
        d = summarize(cb_c.mean() - rg_c.mean(), cb_c[idx].mean(axis=1) - rg_c[idx].mean(axis=1), level, label)
        d["classification"] = classify(*d["decision_interval"])
        return {"D": d, "reading": h2_reading(d["classification"]), "pass": name}
    out["H2"] = h2(h2_pass, "H2.D")
    if h2_pass != "plain":
        out["H2"]["secondary_no_noise"] = {**h2("plain", None), "role": "secondary (reported, never decides H2)"}
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


def analyze_final(rows, study, protocol, launch=None):
    """Stage 2 confirmation: H1-H7 and the cross-channel rule once, with the frozen attacker
    choices, H2 on the pre-registered pass, and utility clusters from the final F articles."""
    records = [paired_record(r["result"]) for r in rows if r["result"]["kind"] == "paired"]
    targets = protocol["design"]["targets"]
    expected = {("inf", arm): targets["inf"] for arm in ARMS2}
    expected.update({(str(eps), arm): n for eps, n in targets["dp"].items() for arm in ARMS2})
    found = Counter((str(r["epsilon_budget"]), r["arm"]) for r in records)
    if dict(found) != expected:
        raise ValueError(f"Final paired jobs {dict(found)} differ from the protocol's {expected}")
    if launch is not None:
        declared = {str(eps): sorted(study["cohorts"]["final"]["targets"][i]["target_sha256"] for i in indices)
                    for eps, indices in launch["allocation"].items()}
        for eps, hashes in declared.items():
            for arm in ARMS2:
                if sorted(r["target"] for r in records if r["arm"] == arm and str(r["epsilon_budget"]) == eps) != hashes:
                    raise ValueError(f"{arm} at epsilon {eps} did not run on the declared final targets")
    public_rows = [r["result"] for r in rows if r["result"]["kind"] == "public"]
    p0_rows = [r["result"] for r in rows if r["result"]["kind"] == "p0"]
    if sorted(r["seed"] for r in public_rows) != sorted(protocol["design"]["public_seeds"]) or len(p0_rows) != 1:
        raise ValueError("The final launch needs the 5 RG-public seeds and one P0 model")
    if any(r["cohort"] != "final" for r in public_rows + p0_rows):
        raise ValueError("RG-public and P0 must be scored on the final F")
    ids = next(r["utility_ids"] for r in records)
    vectors = [r["worlds"]["public"]["utility"]["F"] for r in public_rows] + [p0_rows[0]["worlds"]["P0"]["utility"]["F"]]
    if any([q["query_id"] for q in v] != ids for v in vectors):
        raise ValueError("Utility questions differ across models")
    public = [[q["f1"] for q in v] for v in vectors[:-1]]
    p0 = [q["f1"] for q in vectors[-1]]
    article = {q["id"]: q["article"] for q in study["cohorts"]["final"]["F"]}
    choices = {arm: dict(protocol["attacker_choices"][arm]) for arm in ARMS2}
    h2_pass = protocol["hypotheses"]["H2"]["primary_pass"]
    result = hypotheses(records, public, p0, choices, bool(protocol["h4_primary"]), [article[i] for i in ids], h2_pass=h2_pass)
    return {"schema": "grounded_final_analysis_v1", "role": "confirmation (Stage 2, final targets)",
            "protocol_status": protocol["status"], "attacker_choices": choices, "h2_primary_pass": h2_pass,
            "h2_preregistration_note": protocol["hypotheses"]["H2"].get("notes"),
            "targets": {eps: len(v) for eps, v in (launch or {}).get("allocation", {}).items()},
            "accounted_epsilon": require_matched_epsilon(records), "hypotheses": result,
            "method": {"bootstrap": BOOTSTRAP, "rng_seed": RNG_SEED, "unit": "target (paired across arms)",
                       "utility_bootstrap": "crossed: models x final F article clusters"},
            "timings": [{"kind": r["result"]["kind"], "arm": r["result"]["arm"], "epsilon_budget": r["result"]["epsilon_budget"],
                         "total_seconds": r["result"]["timings"]["total_seconds"]} for r in rows]}


def analyze(root, output, datastore=None, control=None, p0_launch=None):
    launch, _, study, rows = completed(root)
    datastore = json.loads(Path(datastore).read_bytes()) if datastore else None
    if launch["kind"] == "final":
        report = analyze_final(rows, study, stage2_protocol(Path(root) / "protocol.json"), launch)
    elif launch["kind"] == "datastore":
        report = analyze_datastore(rows)
    elif launch["kind"] == "control":
        report = analyze_control(rows, datastore)
    elif launch["kind"] == "timing":
        report = analyze_timing(rows)
    elif launch["kind"] in ("diagnostic", "repeat"):
        report = analyze_diagnostic(rows)
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


# -------------------------------------------------------------- describe

DESCRIPTIVE = "descriptive secondary; does not change any primary outcome"
NQ_CELLS = ("both", "training_only", "library_only", "neither")
NQ_KEYS = {"both": "both", "training_only": "training_only", "library_only": "d_plus", "neither": "d_minus"}
NQ_CONTRASTS = {"library_only_vs_neither": ("library_only", "neither"),
                "training_only_vs_neither": ("training_only", "neither"),
                "both_vs_neither": ("both", "neither"), "both_vs_library_only": ("both", "library_only"),
                "both_vs_training_only": ("both", "training_only")}
UTILITY_METRICS = ("f1", "em", "no_context_f1", "no_context_em", "answer_nll")
NOT_COVERED = {"epsilon_64_frontier": "not run (decision B2)", "CB-LM": "Stage 1 bridge only; not run in Stage 2",
               "per_client_results": "the result files keep no per-client measurements"}


def _rate(flags):
    flags = [bool(f) for f in flags if f is not None]
    return sum(flags) / len(flags) if flags else None


def _withheld(rows):
    """Share of queries whose top document the gate would withhold; None when no gate was recorded."""
    return _rate(None if r.get("gate") is None else r["gate"]["withheld_top"] for r in rows)


def utility_vectors(body):
    """Per-question secondary utility of one model: F metrics, public-library F1 on F_P, refusals and the gate."""
    F, P = body["utility"]["F"], body["utility"]["F_P"]
    out = {m: [q[m] for q in F] for m in UTILITY_METRICS}
    out.update(public_f1=[q["f1"] for q in P], public_em=[q["em"] for q in P],
               ids=[q["query_id"] for q in F], public_ids=[q["query_id"] for q in P],
               refusal={"utility_F": _rate(q["refused"] for q in F), "utility_F_P": _rate(q["refused"] for q in P)},
               gate={"utility_F": _withheld(F)})
    return out


def descriptive_record(result):
    """paired_record plus the recorded refusals, gate decisions, verbatim answers and secondary utility."""
    record = paired_record(result)
    w1, w0 = result["worlds"]["W1"], result["worlds"]["W0"]
    cells = {**w1["natural_questions"], **w0["natural_questions"]}
    never = [p for d in w0.get("never_documents", []) for p in d["probes"]]
    record["verbatim"] = {c: cells[c]["verbatim"]["score"] for c in NQ_CELLS}
    record["refusal"] = {**{f"nq_{c}": _rate(p["refused"] for p in cells[c]["probes"]) for c in NQ_CELLS},
                         **{f"verbatim_{c}": _rate([cells[c]["verbatim"]["refused"]]) for c in NQ_CELLS},
                         "nq_never_documents": _rate(p["refused"] for p in never),
                         **{f"retention_{n}": _rate(p["refused"] for p in w["retention"]["probes"])
                            for n, w in (("W1", w1), ("W0", w0))}}
    record["gate"] = {**{f"nq_{c}": _withheld(cells[c]["probes"]) for c in NQ_CELLS},
                      **{f"verbatim_{c}": _withheld([cells[c]["verbatim"]]) for c in NQ_CELLS},
                      "nq_never_documents": _withheld(never)}
    record["utility_extra"] = {n: utility_vectors(w) for n, w in (("W1", w1), ("W0", w0))}
    for n, extra in record["utility_extra"].items():
        record["refusal"].update({f"{k}_{n}": v for k, v in extra["refusal"].items()})
        record["gate"].update({f"{k}_{n}": v for k, v in extra["gate"].items()})
    return record


def _ci(point, draws):
    return {"estimate": float(point), "ci95": interval(draws)}


def _mean(values, index):
    """Mean over targets with its 95% target-bootstrap interval; None when a value was not recorded."""
    import numpy as np
    if not len(values) or any(v is None for v in values):
        return None
    v = np.asarray(values, float)
    return v.mean(), v[index].mean(axis=1)


def _summary(pair):
    return None if pair is None else _ci(*pair)


def _difference(a, b):
    return None if a is None or b is None else _ci(a[0] - b[0], a[1] - b[1])


def _auc_pair(pos, neg, index):
    return auc(pos, neg), auc_resampled(pos, neg, index)


def pooled_auc_draws(pos, negs, index):
    """Pooled AUC of each target's positive against every target's negatives, resampling targets
    (a drawn target brings its positive and all its negatives)."""
    import numpy as np
    pos, negs = np.asarray(pos, float), np.asarray(negs, float)
    out = np.empty(len(index))
    for k, draw in enumerate(index):
        neg, p = np.sort(negs[draw].ravel()), pos[draw]
        lo, hi = np.searchsorted(neg, p, "left"), np.searchsorted(neg, p, "right")
        out[k] = (lo + 0.5 * (hi - lo)).sum() / (len(p) * len(neg))
    return out


def _kept(pairs):
    """Targets whose two values were both recorded, with their own seeded target draws."""
    import numpy as np
    keep = [i for i, pair in enumerate(pairs) if None not in pair]
    return keep, (target_indices(len(keep), np.random.default_rng(RNG_SEED)) if keep else None)


def _by_arm(arms, value):
    """value(arm, rows) for each arm, and RG - CB-AO and CB-AO - RG of the paired draws."""
    out = {arm: value(arm, rows) for arm, rows in arms.items()}
    return ({arm: _summary(v) for arm, v in out.items()},
            _difference(out["RG"], out["CB-AO"]), _difference(out["CB-AO"], out["RG"]))


def _describe_budget(arms, index, choices, weights, fp_weights):
    import numpy as np
    section = {}
    # H1 sensitivity: every arm in both Reference forms and every pairing of forms.
    ref = {(arm, form): _auc_pair([r["reference"][form][0] for r in rows], [r["reference"][form][1] for r in rows], index)
           for arm, rows in arms.items() for form in FORMS}
    section["reference"] = {
        "label": DESCRIPTIVE, "auc": {arm: {form: _ci(*ref[(arm, form)]) for form in FORMS} for arm in arms},
        "CBAO_minus_RG": {f"CB-AO_{cf}__RG_{rf}": {**_difference(ref[("CB-AO", cf)], ref[("RG", rf)]),
                                                    "frozen_pairing": (cf, rf) == (choices["CB-AO"]["reference_form"],
                                                                                   choices["RG"]["reference_form"])}
                          for cf in FORMS for rf in FORMS},
        "note": "the frozen pairing is H1's (H6's at epsilon 16) primary difference, shown with 95% intervals only"}
    # Causal AUCs per arm and pass, in each arm's frozen direction.
    causal = {"label": DESCRIPTIVE, "direction": {arm: choices[arm]["causal_direction"] for arm in arms}}
    for name in ("plain", "release_noise"):
        if all(name in r["causal"][choices[arm]["causal_direction"]] for arm, rows in arms.items() for r in rows):
            per, rg_cb, cb_rg = _by_arm(arms, lambda arm, rows: _mean(
                [r["causal"][choices[arm]["causal_direction"]][name] for r in rows], index))
            causal[name] = {"mean_per_target_auc": per, "CBAO_minus_RG": cb_rg}
    section["causal"] = causal
    # Natural-question AUCs per cell contrast, both scorers.
    nq = {"label": DESCRIPTIVE, "frozen_scorer": {arm: choices[arm]["nq_scorer"] for arm in arms}}
    for scorer in NQ_SCORERS:
        nq[scorer] = {"mean_document_score": {arm: {c: _summary(_mean([r["nq"][scorer][NQ_KEYS[c]] for r in rows], index))
                                                    for c in NQ_CELLS} for arm, rows in arms.items()}}
        for contrast, (pos, neg) in NQ_CONTRASTS.items():
            per, rg_cb, _ = _by_arm(arms, lambda arm, rows: _auc_pair([r["nq"][scorer][NQ_KEYS[pos]] for r in rows],
                                                                      [r["nq"][scorer][NQ_KEYS[neg]] for r in rows], index))
            nq[scorer][contrast] = {"auc": per, "RG_minus_CBAO": rg_cb}
        # Refusal-excluded sensitivity of the H3 contrast: targets with a scored probe in both cells.
        pairs = {arm: [(r["nq"][scorer]["d_plus_excl"], r["nq"][scorer]["d_minus_excl"]) for r in rows] for arm, rows in arms.items()}
        excluded = {}
        for arm, p in pairs.items():
            keep, draws = _kept(p)
            excluded[arm] = {"targets": len(keep), **(_ci(*_auc_pair([p[i][0] for i in keep], [p[i][1] for i in keep], draws))
                                                      if keep else {})}
        keep, draws = _kept([(*a, *b) for a, b in zip(pairs["RG"], pairs["CB-AO"])])
        if keep:
            a = _auc_pair([pairs["RG"][i][0] for i in keep], [pairs["RG"][i][1] for i in keep], draws)
            b = _auc_pair([pairs["CB-AO"][i][0] for i in keep], [pairs["CB-AO"][i][1] for i in keep], draws)
            excluded["RG_minus_CBAO"] = {"targets": len(keep), **_difference(a, b)}
        nq[scorer]["library_only_vs_neither_refusals_excluded"] = excluded
        # Standard attack: inserted passages (W0, library on) against the never-inserted N documents.
        if all(r["never"][scorer] for rows in arms.values() for r in rows):
            def never(arm, rows):
                pos, negs = [r["nq"][scorer]["d_plus"] for r in rows], [r["never"][scorer] for r in rows]
                return auc(pos, np.ravel(negs)), pooled_auc_draws(pos, negs, index)
            per, rg_cb, _ = _by_arm(arms, never)
            nq[scorer]["library_only_vs_never_documents"] = {"auc": per, "RG_minus_CBAO": rg_cb,
                                                             "documents_per_target": len(arms["RG"][0]["never"][scorer])}
    section["natural_questions"] = nq
    # The verbatim yes/no baseline (Anderson et al.): library on vs off in W0, decided answers only.
    verbatim = {"label": DESCRIPTIVE, "yes_rate": {arm: {c: _rate(r["verbatim"][c] for r in rows) for c in NQ_CELLS}
                                                   for arm, rows in arms.items()}, "library_only_vs_neither": {}}
    for arm, rows in arms.items():
        p = [(r["verbatim"]["library_only"], r["verbatim"]["neither"]) for r in rows]
        keep, draws = _kept(p)
        verbatim["library_only_vs_neither"][arm] = {"targets_with_decisions": len(keep), **(
            _ci(*_auc_pair([p[i][0] for i in keep], [p[i][1] for i in keep], draws)) if keep else {})}
    section["verbatim_baseline"] = verbatim
    # Closed-book retention per arm (R_CB-AO is the §6.4 secondary).
    per, rg_cb, _ = _by_arm(arms, lambda arm, rows: _mean([r["retention"] for r in rows], index))
    section["retention"] = {"label": DESCRIPTIVE, "R": per, "RG_minus_CBAO": rg_cb}
    # Refusal rates and the gate's would-withhold rates per arm and cell.
    for name in ("refusal", "gate"):
        keys = arms["RG"][0][name]
        section[f"{name}_rates"] = {"label": DESCRIPTIVE, **{arm: {k: _summary(_mean([r[name][k] for r in rows], index))
                                                                   for k in keys} for arm, rows in arms.items()}}
    section["gate_rates"]["note"] = "the gate depends only on the query and the library, never on the model"
    # Secondary utility per arm and world, with the crossed models x article-cluster bootstrap.
    util = {"label": DESCRIPTIVE}
    for world in ("W1", "W0"):
        def metric(m, w):
            return lambda arm, rows: _utility([r["utility_extra"][world][m] for r in rows], w, index)
        util[world] = {}
        for m in UTILITY_METRICS + ("public_f1", "public_em"):
            per, rg_cb, _ = _by_arm(arms, metric(m, fp_weights if m.startswith("public") else weights))
            util[world][m] = {**per, "RG_minus_CBAO": rg_cb}
    section["utility"] = util
    return section


def _utility(vectors, weights, index=None):
    """Mean over models of each model's mean, with the crossed bootstrap (index None: one model)."""
    import numpy as np
    means = model_means(vectors, weights)
    draws = means[:, 0] if index is None else arm_means(means, index)
    return np.mean([np.mean(v) for v in vectors]), draws


def describe_final(rows, study, protocol):
    """The pre-registered descriptive secondaries (proposal §9.2) of the finished Stage 2 launch.
    Nothing is classified, decided or relabelled: the primary outcomes stay those of the one analyze."""
    import numpy as np
    records = [descriptive_record(r["result"]) for r in rows if r["result"]["kind"] == "paired"]
    epsilons = require_matched_epsilon(records)
    choices = {arm: dict(protocol["attacker_choices"][arm]) for arm in ARMS2}
    by = lambda arm, eps: [r for r in records if r["arm"] == arm and str(r["epsilon_budget"]) == eps]
    # The same seeded draws, in the same order, as hypotheses(): targets at inf, at 16, then F clusters, then seeds.
    rng = np.random.default_rng(RNG_SEED)
    budgets = ["inf", "16"] + [str(e) for e in DP_BUDGETS if e != 16 and by("RG", str(e))]
    arms, index = {}, {}
    for eps in budgets[:2]:
        _, arms[eps] = aligned({arm: by(arm, eps) for arm in ARMS2})
        index[eps] = target_indices(len(arms[eps]["RG"]), rng)
    final = study["cohorts"]["final"]
    ids = arms["inf"]["RG"][0]["utility_ids"]
    fp_ids = arms["inf"]["RG"][0]["utility_extra"]["W1"]["public_ids"]
    models = {"P0": [r["result"]["worlds"]["P0"] for r in rows if r["result"]["kind"] == "p0"],
              "RG-public": [r["result"]["worlds"]["public"] for r in rows if r["result"]["kind"] == "public"]}
    models = {name: [utility_vectors(body) for body in bodies] for name, bodies in models.items()}
    every = [x for rs in arms.values() for rows in rs.values() for r in rows for x in r["utility_extra"].values()]
    if any(v["ids"] != ids or v["public_ids"] != fp_ids for v in every + models["P0"] + models["RG-public"]):
        raise ValueError("Utility questions differ across models")
    if len(models["P0"]) != 1 or not models["RG-public"]:
        raise ValueError("The final launch needs one P0 model and the RG-public seeds")
    article = {q["id"]: q["article"] for q in final["F"]}
    weights = cluster_weights([article[i] for i in ids], rng)
    seed_idx = target_indices(len(models["RG-public"]), rng)
    public_article = {q["id"]: q["article"] for q in final["F_P"]}
    fp_weights = cluster_weights([public_article[i] for i in fp_ids], rng)
    for eps in budgets[2:]:
        _, arms[eps] = aligned({arm: by(arm, eps) for arm in ARMS2})
        index[eps] = target_indices(len(arms[eps]["RG"]), rng)
    report = {"budgets": {eps: _describe_budget(arms[eps], index[eps], choices, weights, fp_weights) for eps in budgets}}
    # P0 and RG-public: the same secondary utility, and P0 against every arm (oriented as H7: P0 minus the arm).
    shared = {}
    for name, vectors in models.items():
        draws_index = None if name == "P0" else seed_idx
        shared[name] = {m: _ci(*_utility([v[m] for v in vectors], fp_weights if m.startswith("public") else weights, draws_index))
                        for m in UTILITY_METRICS + ("public_f1", "public_em")}
        shared[name]["refusal_rates"] = {k: float(np.mean([v["refusal"][k] for v in vectors])) for k in ("utility_F", "utility_F_P")}
        gates = [v["gate"]["utility_F"] for v in vectors]
        shared[name]["gate_rates"] = {"utility_F": None if None in gates else float(np.mean(gates))}
    p0 = _utility([models["P0"][0]["f1"]], weights)
    against = {"RG-public": _difference(p0, _utility([v["f1"] for v in models["RG-public"]], weights, seed_idx))}
    for eps in budgets:
        for arm, paired in arms[eps].items():
            against[f"{arm}_eps{eps}"] = _difference(p0, _utility([r["utility"] for r in paired], weights, index[eps]))
    report["shared_models"] = {"label": DESCRIPTIVE, **shared, "P0_minus_arm_rag_f1": against,
                               "note": "P0 is one model (article clusters only); RG-public draws its 5 seeds"}
    first = next(r["result"] for r in rows if r["result"]["kind"] == "paired")
    direction = choices[first["arm"]]["causal_direction"]
    report["release_privacy"] = first["causal"][direction].get("release_noise", {}).get("release_privacy")
    inf, b16 = report["budgets"]["inf"], report["budgets"]["16"]
    frozen = f"CB-AO_{choices['CB-AO']['reference_form']}__RG_{choices['RG']['reference_form']}"
    report["reproduced_primary_values"] = {
        "H1.D": inf["reference"]["CBAO_minus_RG"][frozen], "H1.utility_RG_minus_CBAO": inf["utility"]["W1"]["f1"]["RG_minus_CBAO"],
        "H2.D": inf["causal"].get("release_noise", {}).get("CBAO_minus_RG"),
        "H2.secondary_no_noise.D": inf["causal"]["plain"]["CBAO_minus_RG"],
        "H3.D": inf["natural_questions"][choices["RG"]["nq_scorer"]]["library_only_vs_neither"]["RG_minus_CBAO"],
        "H4.R_RG": inf["retention"]["R"]["RG"], "H4.D_H4": inf["retention"]["RG_minus_CBAO"],
        "H6.utility": b16["utility"]["W1"]["f1"]["RG_minus_CBAO"],
        "H6.reference_auc_CBAO_minus_RG": b16["reference"]["CBAO_minus_RG"][frozen],
        "H7.D": against["RG_epsinf"]}
    return {"accounted_epsilon": epsilons, "attacker_choices": choices,
            "targets": {eps: len(arms[eps]["RG"]) for eps in budgets}, **report}


def frozen_final(root):
    """The finished final launch, read through frozen_results: every result is checked against its launch's
    completion record and declared job, never against today's tool or core code, and nothing is re-run."""
    root = Path(root).resolve()
    launch = json.loads((root / "launch.json").read_bytes())
    if launch.get("schema") != LAUNCH_SCHEMA or launch.get("study") != STUDY or launch.get("kind") != "final":
        raise ValueError("describe reads a finished final (Stage 2) launch only")
    if digest(root / "study.json") != launch["study_file_sha256"] or digest(root / "protocol.json") != launch["protocol_sha256"]:
        raise ValueError("The launch's study or protocol copy changed after declaration")
    study = json.loads((root / "study.json").read_bytes())
    protocol = json.loads((root / "protocol.json").read_bytes())
    if protocol.get("schema") != STAGE2_SCHEMA or not str(protocol.get("status", "")).startswith("frozen"):
        raise ValueError("The launch's protocol is not a frozen Stage 2 protocol")
    rows = frozen_results(root, study)
    cohort = json.loads((root / "cohort.json").read_bytes())
    if cohort["launch_sha256"] != digest(root / "launch.json"):
        raise ValueError("The frozen cohort belongs to another launch")
    for row in rows:
        entry = launch["jobs"][row["index"]]
        if digest(root / entry["file"]) != entry["sha256"]:
            raise ValueError("A job file changed after declaration")
        verify_result(row["result"], launch, json.loads((root / entry["file"]).read_bytes()), cohort["targets"][str(row["index"])])
    return root, launch, protocol, study, rows


def _close(a, b, tolerance=1e-9):
    return a is not None and b is not None and abs(a["estimate"] - b["estimate"]) <= tolerance \
        and all(abs(x - y) <= tolerance for x, y in zip(a["ci95"], b["ci95"]))


def describe(root, output, final_analysis=None):
    """Write the descriptive secondaries of the finished final launch to a new file (refuses to overwrite)."""
    root, launch, protocol, study, rows = frozen_final(root)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Descriptive results already exist; write a new file")
    if root == output.parent or root in output.parents:
        raise ValueError("Write the descriptive results outside the launch directory")
    report = {"schema": "grounded_final_descriptives_v1", "label": DESCRIPTIVE,
              "role": "descriptive secondaries (proposal §9.2) of the Stage 2 confirmation launch",
              "note": ("Read from the finished launch through frozen_results. No hypothesis is classified, decided or "
                       "relabelled here; the primary outcomes are those of the one confirmation analyze."),
              "method": {"bootstrap": BOOTSTRAP, "rng_seed": RNG_SEED, "intervals": "95% only",
                         "unit": "target (paired across arms); utility crossed models x article clusters (F, or F_P)",
                         "draws": "the same seeded target and cluster draws as the confirmation analysis; subsets "
                                  "(refusal-excluded, verbatim decisions) draw their own from the same seed"},
              "not_covered": NOT_COVERED, **describe_final(rows, study, protocol),
              "launch_sha256": digest(root / "launch.json"), "launch_study_tool_sha256": launch["study_tool_sha256"],
              "describe_tool_sha256": tool_digest(), "implementation_fingerprint": launch["implementation_fingerprint"],
              "protocol_sha256": launch["protocol_sha256"], "study_sha256": launch["study_sha256"],
              "not_a_privacy_claim": "Attack-effectiveness and validation measurements only"}
    if final_analysis is not None:
        analysis = json.loads(Path(final_analysis).read_bytes())
        if analysis.get("schema") != "grounded_final_analysis_v1" or analysis.get("launch_sha256") != report["launch_sha256"]:
            raise ValueError("--final-analysis is not this launch's confirmation analysis")
        h = analysis["hypotheses"]
        frozen = {"H1.D": h["H1"]["D"], "H1.utility_RG_minus_CBAO": h["H1"]["utility_RG_minus_CBAO"], "H2.D": h["H2"]["D"],
                  "H2.secondary_no_noise.D": h["H2"].get("secondary_no_noise", {}).get("D"), "H3.D": h["H3"].get("D"),
                  "H4.R_RG": h["H4"]["R_RG"], "H4.D_H4": h["H4"]["D_H4"], "H6.utility": h["H6"]["utility"],
                  "H6.reference_auc_CBAO_minus_RG": h["H6"]["reference_auc_CBAO_minus_RG"], "H7.D": h["H7"]["D"]}
        report["final_analysis_sha256"] = digest(final_analysis)
        report["consistency_with_final_analysis"] = {
            name: _close(report["reproduced_primary_values"][name], value) for name, value in frozen.items()}
    write_json(output, report)
    return report


def describe_lines(report, path=""):
    """One plain line per reported value, for pasting back."""
    lines = []
    if isinstance(report, dict) and "estimate" in report and "ci95" in report:
        extra = "".join(f" {k}={v}" for k, v in report.items() if k not in ("estimate", "ci95"))
        return [f"{path}: {report['estimate']:.3f} [{report['ci95'][0]:.3f}, {report['ci95'][1]:.3f}]{extra}"]
    if isinstance(report, dict):
        for key, value in report.items():
            lines += describe_lines(value, f"{path}.{key}" if path else str(key))
    elif isinstance(report, float):
        lines.append(f"{path}: {report:.4g}")
    elif not isinstance(report, list):
        lines.append(f"{path}: {report}")
    return lines


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
    pr.add_argument("--targets", type=int, help="diagnostic (default 12, 6-60) or repeat (default 4, 2-12): V targets per order")
    r = sub.add_parser("resolve", help="CPU checks and frozen targets"); r.add_argument("launch")
    run = sub.add_parser("run", help="run the next job(s)"); run.add_argument("launch"); run.add_argument("--gpu", required=True)
    run.add_argument("--max-jobs", type=int, default=1); run.add_argument("--scratch-root")
    j = sub.add_parser("job", help="execute one job (used by run)"); j.add_argument("job"); j.add_argument("output")
    c = sub.add_parser("check", help="integrity and timing; no endpoint values"); c.add_argument("root")
    a = sub.add_parser("analyze", help="validation rules or analysis"); a.add_argument("root"); a.add_argument("output")
    a.add_argument("--datastore-analysis"); a.add_argument("--control-analysis")
    a.add_argument("--p0-launch", help="the P0 datastore launch folder (pilot analysis: P0 utility on V)")
    # The final confirmation needs no extra arguments: its launch holds P0, RG-public and the frozen protocol.
    d = sub.add_parser("describe", help="descriptive secondaries of the finished final launch (read-only)")
    d.add_argument("root"); d.add_argument("output")
    d.add_argument("--final-analysis", help="the launch's one confirmation analysis, to record that the primary values reproduce")
    args = p.parse_args(argv)
    if args.action == "build":
        report = build(args.output, args.exclude_manifest)
        print(json.dumps({k: report[k] for k in ("study_sha256", "exclusions", "skips", "audit")}, indent=1))
    elif args.action == "calibrate-gate":
        gate = calibrate_gate(args.study, args.output)
        print(json.dumps({k: gate[k] for k in ("threshold", "gold_loss", "verbatim_probe_withheld_rate")}, indent=1))
    elif args.action == "prepare":
        launch = prepare(args.kind, args.output, args.study, args.gate, args.previous_analysis, args.targets)
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
    elif args.action == "describe":
        report = describe(args.root, args.output, args.final_analysis)
        print(f"{DESCRIPTIVE.upper()}. Written to {Path(args.output).resolve()}")
        print("\n".join(describe_lines({k: v for k, v in report.items() if k not in ("method", "note", "role", "label")})))
    else:
        report = analyze(args.root, args.output, args.datastore_analysis, args.control_analysis, args.p0_launch)
        print(json.dumps({k: v for k, v in report.items() if k not in ("learning_per_target",)}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
