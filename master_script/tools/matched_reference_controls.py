"""Pre-registered matched-reference benign controls for the causal-request detector.

prepare  writes a frozen launch that the unchanged guard collector resolves and
         runs (`collect_guard_traces resolve|run`): targets are fixed and
         excluded before any GPU job, jobs run one at a time, weights retire.
check    reports each completed job's matching checks and timing, never a decision.
analyze  computes the fixed endpoints E1-E3, secondary S1-S5 and the decision.

See outputs/matched_reference_controls_20260930/protocol.json. The detector is
the frozen guard-v4 pilot detector in shadow mode: detection here is not
protection, and nothing is refit or rethresholded.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import yaml

from master_script.core.config import implementation_fingerprint
from master_script.core.guard_detector import load_detector
from master_script.core.guard_features import STRUCTURE_SCHEMA, STRUCTURE_NAMES
from master_script.core.queue import write_json
from master_script.tools.build_guard_stage import ROOT
from master_script.tools.causal_attack_validation import DEFAULTS as CAUSAL_DEFAULTS, BASE as CAUSAL_BASE
from master_script.tools.collect_guard_traces import SMOKE_TARGET, collection_tool_digest

PROTOCOL = ROOT.parents[1] / "outputs/matched_reference_controls_20260930/protocol.json"
STUDY, ROLE = "matched_reference_controls", "matched_control_evaluation"
SEED_BAND, FIRST_SEED, TARGETS = (7000, 7099), 7000, 20
DESCENT_REQUESTS, TRIALS = 10, 4
BOOTSTRAP, RNG_SEED = 10_000, 20260930
# Minimum exclusions: smoke 1 + v3 8 + v4 dev 3 + reserved final 4 + causal Stage A 4 + Stage B 10.
MIN_EXCLUDED, REQUIRED_ROLES = 30, {"final": 4, "attacker_tuning": 4, "attacker_evaluation": 10}
DEFAULTS = {**{k: v for k, v in CAUSAL_DEFAULTS.items() if k not in ("calibration_nonmember_count", "calibration_fpr")},
            "attack_trials": TRIALS, "threshold_mode": "fixed"}
BASE = {**CAUSAL_BASE, "probe_epochs": 12, "causal_score": "cosine", "request_interpolation": 1.0,
        "matched_descent_requests": DESCENT_REQUESTS, "matched_honest_round": True}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def tool_digest():
    return digest(__file__)


def _protocol():
    protocol = json.loads(PROTOCOL.read_bytes())
    if protocol.get("schema") != "matched_reference_controls_protocol_v1" or not protocol["status"].startswith("final"):
        raise ValueError("The matched-controls protocol is missing or not final")
    return protocol


def _exclusions(manifests):
    """Targets and role counts from splits manifests and collector launches."""
    excluded, sources, roles = {SMOKE_TARGET}, [], {}
    for path in manifests:
        raw = Path(path).read_bytes()
        manifest = json.loads(raw)
        if manifest.get("schema") in ("guard_collection_v3", "guard_collection_v4"):
            groups = {target: "excluded_by_launch" for target in manifest.get("excluded_targets", [])}
        elif manifest.get("schema") == "guard_splits_v2":
            groups = manifest.get("groups") or {}
        else:
            raise ValueError(f"{path}: exclusions need a guard_splits_v2 manifest or a guard collection launch")
        if not groups:
            raise ValueError(f"{path}: exclusion source lists no targets")
        excluded.update(groups)
        for role in groups.values():
            roles[role] = roles.get(role, 0) + 1
        sources.append({"path": str(Path(path).resolve()), "sha256": sha256(raw).hexdigest(),
                        "schema": manifest["schema"], "groups": len(groups)})
    for role, minimum in REQUIRED_ROLES.items():
        if roles.get(role, 0) < minimum:
            raise ValueError(f"Exclusions need at least {minimum} '{role}' targets: supply the pilot cohort.json "
                             "and the causal-validation stage-a-2 and stage-b splits.complete.json")
    if len(excluded) < MIN_EXCLUDED:
        raise ValueError("Exclusions cover fewer targets than the earlier cohorts; supply every manifest")
    return excluded, sources


def job_document(seed, detector_path, detector_sha256):
    guard = {"mode": "shadow", "policy_file": str(ROOT / "guard/approved_training_policy.json"),
             "release_budget": 3, "diagnostic": True, "validation_workers": 4,
             "feature_schema": STRUCTURE_SCHEMA, "detector_file": str(detector_path),
             "detector_sha256": detector_sha256}
    return {"defaults": {**DEFAULTS, "seed": seed},
            "pipeline": {"condition": "shadow", "defense": {"mechanism": "none"}, "client_guard": guard},
            "attacks": {"amia": {"base": dict(BASE), "sweep": {}}}}


def prepare(output, exclude_manifests, detector, first_seed=None):
    """Write jobs and a guard_collection_v4 launch; nothing is resolved or run."""
    from master_script.core.yaml_config import load_config_doc
    protocol = _protocol()
    detector = Path(detector).resolve()
    detector_sha256 = digest(detector)
    if detector_sha256 != protocol["detector"]["sha256"]:
        raise ValueError("Detector differs from the one pinned in the protocol")
    if load_detector(detector, detector_sha256)["schema"] != STRUCTURE_SCHEMA:
        raise ValueError("The pinned detector must use parameter_structure_v2")
    first = FIRST_SEED if first_seed is None else first_seed
    seeds = list(range(first, first + TARGETS))
    if not (SEED_BAND[0] <= seeds[0] and seeds[-1] <= SEED_BAND[1]):
        raise ValueError(f"Seeds must lie in the pre-registered band {SEED_BAND}")
    excluded, sources = _exclusions(exclude_manifests)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    jobs = []
    for seed in seeds:
        doc = job_document(seed, detector, detector_sha256)
        name = f"job-{len(jobs):03d}-seed{seed}.yaml"
        text = yaml.safe_dump(doc, sort_keys=False)
        if len(load_config_doc(yaml.safe_load(text), source=str(output / name))) != 1:
            raise ValueError("Each job must expand to exactly one run")
        (output / name).write_text(text)
        jobs.append({"config": name, "sha256": sha256(text.encode()).hexdigest(), "seed": seed,
                     "role": ROLE, "attack": "amia", "variant": "causal_gradient_alignment"})
    launch = {"schema": "guard_collection_v4", "study": STUDY,
              "implementation_fingerprint": implementation_fingerprint(),
              "collection_tool_sha256": collection_tool_digest(), "study_tool_sha256": tool_digest(),
              "detector": {"path": str(detector), "sha256": detector_sha256},
              "planned_targets": TARGETS, "jobs": jobs, "held_out_variants": [],
              "excluded_targets": sorted(excluded), "exclusion_sources": sources, "reserved_final": [],
              "protocol_file": "protocol.json", "protocol_sha256": digest(output / "protocol.json")}
    write_json(output / "launch.json", launch)
    return launch


def _completed(root, partial=False):
    """Collector-verified results of the launch, with their declared job."""
    root = Path(root).resolve()
    launch = json.loads((root / "launch.json").read_bytes())
    if launch.get("study") != STUDY:
        raise ValueError("Not a matched-reference controls launch")
    if digest(root / "protocol.json") != digest(PROTOCOL) or launch["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Launch protocol differs from the committed protocol")
    state = json.loads((root / "progress.json").read_bytes())
    if not partial and (state.get("active_job") is not None or len(state["completed"]) != len(launch["jobs"])):
        raise ValueError("Launch is not complete; analyse only a finished study")
    rows = []
    for done in state["completed"]:
        path = (root / done["result"]).resolve()
        if root not in path.parents or digest(path) != done["sha256"]:
            raise ValueError("A completed result changed after collection")
        job = launch["jobs"][done["job"]]
        result = json.loads(path.read_bytes())
        config, guard = result.get("config", {}), result.get("pipeline", {}).get("client_guard", {})
        if (result.get("implementation_fingerprint") != launch["implementation_fingerprint"]
                or {t["target_sha256"] for t in result["attack_trials"]} != {state["targets"][str(job["seed"])]}
                or config.get("probe_epochs") != 12 or config.get("causal_score") != "cosine"
                or config.get("matched_descent_requests") != DESCENT_REQUESTS or config.get("matched_honest_round") is not True
                or guard.get("detector_sha256") != launch["detector"]["sha256"] or guard.get("mode") != "shadow"
                or result.get("matched_controls", {}).get("schema") != "matched_controls_v1"):
            raise ValueError(f"{path}: result does not match its declared job")
        rows.append({"job": job, "target": state["targets"][str(job["seed"])], "result": result,
                     "path": str(path), "sha256": done["sha256"]})
    return launch, state, rows


DECISION_FIELDS = ("c", "m1", "m2", "m2raw", "reasons", "training_rejected_events", "training_rejected_distinct",
                   "training_rejected_distinct_by_round", "d0", "within_target_auc_c_vs_m1", "detector_scores")


def check(root, t0=None):
    """Integrity of completed jobs so far. Decision fields are withheld until analysis."""
    _, state, rows = _completed(root, partial=True)
    t0 = _protocol()["detector"]["baseline_D0_threshold"] if t0 is None else t0
    report = []
    for row in rows:
        record = {k: v for k, v in target_record(row, t0).items() if k not in DECISION_FIELDS}
        record.pop("m1_norm_ratio_to_causal", None)
        ratios = [c["delta_norm_ratio_to_causal"] for c in row["result"]["matched_controls"]["controls"] if c["arm"] == "M1"]
        record.update(m1_norm_ratio_range=[min(ratios), max(ratios)] if ratios else None,
                      computation_seconds=row["result"].get("computation_seconds"))
        report.append(record)
    return {"completed": len(rows), "planned": len(json.loads((Path(root) / "launch.json").read_bytes())["jobs"]),
            "active_job": state.get("active_job"), "jobs": report}


def rejected(event):
    return not event["would_accept"]


def target_record(row, t0):
    """Per-target values and checks from one result; nothing is filtered silently."""
    result, seed = row["result"], row["job"]["seed"]
    events = result["guard_events"]
    causal = [e for e in events if e["scope"].startswith(f"observation:{seed}:world-")]
    training = [e for e in events if e["accounting_scope"] == "training"]
    by_scope = {}
    for e in events:
        if e["scope"].startswith("benign_control:"):
            if e["scope"] in by_scope:
                raise ValueError("A benign control was checked twice")
            by_scope[e["scope"]] = e
    controls = result["matched_controls"]["controls"]
    arms = {arm: [c for c in controls if c["arm"] == arm] for arm in ("M1", "M2", "M2raw")}
    problems = []
    if len(arms["M1"]) != DESCENT_REQUESTS or len(arms["M2"]) != 1 or len(arms["M2raw"]) != 1:
        problems.append("control count")
    if not causal or len({(e["request_sha256"], e["would_accept"], e["reference_sha256"]) for e in causal}) != 1:
        problems.append("causal request events disagree")
    if set(by_scope) != {c["scope"] for c in controls}:
        problems.append("control events and records differ")
    control_events = {c["scope"]: by_scope.get(c["scope"]) for c in controls}
    references = {e["reference_sha256"] for e in causal} | {e["reference_sha256"] for e in control_events.values() if e}
    if len(references) != 1:
        problems.append("reference differs across arms")
    m2_ratio = arms["M2"][0]["delta_norm_ratio_to_causal"] if arms["M2"] else None
    if m2_ratio is None or abs(m2_ratio - 1) > 1e-4:
        problems.append("M2 norm match")
    causal_loss_rose = result["matched_controls"]["causal_candidate_loss_after"] > result["probe_training_loss"][0]
    descent_fell = [c["loss_after"] < c["loss_before"] for c in arms["M1"]]
    if problems or not control_events or any(e is None for e in control_events.values()):
        return {"target_sha256": row["target"], "seed": seed, "result_sha256": row["sha256"], "valid": False,
                "problems": problems}
    ev = lambda c: control_events[c["scope"]]
    m1 = [ev(c) for c in arms["M1"]]
    m2, m2raw, c_event = ev(arms["M2"][0]), ev(arms["M2raw"][0]), causal[0]
    distinct = {}
    for e in training:
        distinct.setdefault(e["request_sha256"], []).append(e)
    if any(len({x["would_accept"] for x in group}) != 1 for group in distinct.values()):
        problems.append("identical training requests decided differently")
    separability = {}
    for name in ("detector_score",) + STRUCTURE_NAMES:
        value = (lambda e: e["detector_score"]) if name == "detector_score" else (lambda e, n=name: e["features"][n])
        pos, negs = value(c_event), [value(e) for e in m1]
        auc = sum(1. if pos > n else .5 if pos == n else 0. for n in negs) / len(negs)
        separability[name] = auc
    return {
        "target_sha256": row["target"], "seed": seed, "result_sha256": row["sha256"],
        "valid": not problems, "problems": problems,
        "c": float(rejected(c_event)), "m1": sum(map(rejected, m1)) / len(m1), "m2": float(rejected(m2)),
        "m2raw": float(rejected(m2raw)),
        "reasons": sorted({e["reason"] for e in [c_event, m2, m2raw, *m1] if rejected(e)}),
        "training_events": len(training), "training_rejected_events": sum(map(rejected, training)),
        "training_distinct": len(distinct), "training_rejected_distinct": sum(rejected(g[0]) for g in distinct.values()),
        "training_rejected_distinct_by_round": {str(r): sum(rejected(g[0]) for g in distinct.values() if g[0]["round_id"] == r)
                                                for r in sorted({g[0]["round_id"] for g in distinct.values()})},
        "d0": {"c": float(c_event["features"]["relative_delta"] > t0),
               "m1": sum(e["features"]["relative_delta"] > t0 for e in m1) / len(m1),
               "m2": float(m2["features"]["relative_delta"] > t0)},
        "within_target_auc_c_vs_m1": separability,
        "m1_norm_ratio_to_causal": [c["delta_norm_ratio_to_causal"] for c in arms["M1"]],
        "m2_norm_ratio_to_causal": m2_ratio,
        "causal_candidate_loss_rose": causal_loss_rose, "m1_public_loss_fell": sum(descent_fell),
        "detector_scores": {"c": c_event["detector_score"], "m1": [e["detector_score"] for e in m1],
                            "m2": m2["detector_score"], "m2raw": m2raw["detector_score"]},
    }


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided binomial interval: P(X >= k | lower) = P(X <= k | upper) = alpha/2."""
    def tail(p, m):  # P(X >= m), increasing in p
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(m, n + 1))
    def solve(m, target):
        lo, hi = 0., 1.
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if tail(mid, m) < target else (lo, mid)
        return (lo + hi) / 2
    lower = 0. if k == 0 else solve(k, alpha / 2)
    upper = 1. if k == n else solve(k + 1, 1 - alpha / 2)
    return [lower, upper]


def _interval(values):
    import numpy as np
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def endpoints(targets):
    """E1-E3, with the pre-registered interval rules."""
    import numpy as np
    n = len(targets)
    boot = np.random.default_rng(RNG_SEED).integers(0, n, size=(BOOTSTRAP, n))
    out = {}
    for key, name in (("c", "E1"), ("m1", "E2"), ("m2", "E3")):
        values = np.array([t[key] for t in targets])
        bootstrap = _interval(values[boot].mean(axis=1))
        entry = {"pooled": float(values.mean()), "bootstrap_ci95": bootstrap, "targets": n}
        if key == "m1":
            k_any = int((values > 0).sum())
            cp_any = clopper_pearson(k_any, n)
            entry.update(targets_with_any_rejection=k_any, any_rejection_cp95=cp_any,
                         decision_interval=[bootstrap[0], max(bootstrap[1], cp_any[1])])
        else:
            k = int(values.sum())
            entry.update(count=k, cp95=clopper_pearson(k, n), decision_interval=clopper_pearson(k, n))
        out[name] = entry
    return out, boot


def decide(e):
    (e1_lo, e1_hi), (e2_lo, e2_hi), (e3_lo, e3_hi) = (e[k]["decision_interval"] for k in ("E1", "E2", "E3"))
    failures = []
    if e2_lo > 0.01 or e3_lo > 0.01:
        failures.append("fails_benign_flagged")
    if e1_hi < 0.50:
        failures.append("fails_causal_missed")
    if failures:
        return failures
    if e1_lo >= 0.70 and e2_hi <= 0.20 and e3_hi <= 0.20:
        return ["separates_at_this_resolution"]
    return ["inconclusive"]


def analyze(root, output):
    import numpy as np
    launch, state, rows = _completed(root)
    protocol = _protocol()
    t0 = protocol["detector"]["baseline_D0_threshold"]
    if len({r["target"] for r in rows}) != TARGETS:
        raise ValueError(f"The study needs {TARGETS} distinct targets")
    targets = [target_record(r, t0) for r in rows]
    invalid = [t for t in targets if not t["valid"]]
    report = {"schema": "matched_reference_controls_analysis_v1", "launch_sha256": digest(Path(root) / "launch.json"),
              "protocol_sha256": digest(PROTOCOL), "study_tool_sha256": tool_digest(),
              "detector_sha256": launch["detector"]["sha256"],
              "implementation_fingerprint": launch["implementation_fingerprint"],
              "method": {"bootstrap": BOOTSTRAP, "rng_seed": RNG_SEED, "unit": "target"},
              "per_target": targets}
    if invalid:
        # A failed required check invalidates a job; record a deviation before any decision.
        report.update(decision=None, invalid_targets=[t["target_sha256"] for t in invalid],
                      note="Required matching checks failed; no endpoint or decision is computed until a deviation is recorded")
    else:
        primary, boot = endpoints(targets)
        def pooled(values):
            values = np.array(values, dtype=float)
            return {"pooled": float(values.mean()), "bootstrap_ci95": _interval(values[boot].mean(axis=1))}
        training_events = np.array([t["training_events"] for t in targets])
        training_rejected = np.array([t["training_rejected_events"] for t in targets])
        report.update(
            endpoints=primary, decision=decide(primary),
            secondary={
                "S1_training_rejection_per_event": {
                    "rate": float(training_rejected.sum() / training_events.sum()),
                    "bootstrap_ci95": _interval(training_rejected[boot].sum(axis=1) / training_events[boot].sum(axis=1)),
                    "events": int(training_events.sum())},
                "S1_training_rejection_per_distinct_request": pooled(
                    [t["training_rejected_distinct"] / t["training_distinct"] for t in targets]),
                "S2_m2raw_rejection": pooled([t["m2raw"] for t in targets]),
                "S3_d0_relative_delta_only": {k: pooled([t["d0"][k] for t in targets]) for k in ("c", "m1", "m2")},
                "S4_exploratory_within_target_auc_c_vs_m1": {
                    name: {**pooled([t["within_target_auc_c_vs_m1"][name] for t in targets]),
                           "two_sided": float(max(np.mean([t["within_target_auc_c_vs_m1"][name] for t in targets]),
                                                  1 - np.mean([t["within_target_auc_c_vs_m1"][name] for t in targets])))}
                    for name in ("detector_score",) + STRUCTURE_NAMES},
                "S5_matching": {"m1_norm_ratio_range": [min(min(t["m1_norm_ratio_to_causal"]) for t in targets),
                                                        max(max(t["m1_norm_ratio_to_causal"]) for t in targets)],
                                "causal_loss_rose_targets": sum(t["causal_candidate_loss_rose"] for t in targets),
                                "m1_public_loss_fell": sum(t["m1_public_loss_fell"] for t in targets)}},
            not_a_privacy_claim="Shadow-mode detection only; detection or blocking is not protection")
    if Path(output).exists():
        raise ValueError("Analysis already exists; write a new file")
    write_json(output, report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="write a frozen launch for collect_guard_traces resolve/run")
    prep.add_argument("output"); prep.add_argument("--detector", required=True)
    prep.add_argument("--exclude-manifest", action="append", default=[], required=True)
    prep.add_argument("--first-seed", type=int)
    chk = sub.add_parser("check", help="matching checks and timing so far, no decisions"); chk.add_argument("root")
    ana = sub.add_parser("analyze", help="endpoints and decision"); ana.add_argument("root"); ana.add_argument("output")
    args = p.parse_args(argv)
    if args.action == "prepare":
        launch = prepare(args.output, args.exclude_manifest, args.detector, args.first_seed)
        print(f"Prepared {len(launch['jobs'])} jobs. Nothing resolved or run.")
        print(f"Next: python -m master_script.tools.collect_guard_traces resolve {Path(args.output).resolve() / 'launch.json'}")
    elif args.action == "check":
        print(json.dumps(check(args.root), indent=1))
    else:
        report = analyze(args.root, args.output)
        print(json.dumps({k: report.get(k) for k in ("endpoints", "decision", "invalid_targets")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
