"""Pre-registered validation of the causal gradient-alignment attack.

prepare  writes a frozen launch that the unchanged guard collector resolves and
         runs (`collect_guard_traces resolve|run`): targets are fixed and
         excluded before any GPU job, jobs run one at a time, weights retire.
select   applies the fixed Stage A rule to completed tuning results.
analyze  computes the fixed Stage B endpoints and decision.

See outputs/causal_attack_validation_20260929/protocol.json. No guard, no RAG.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import yaml

from master_script.core.config import implementation_fingerprint
from master_script.core.queue import write_json
from master_script.tools.build_guard_stage import ROOT
from master_script.tools.collect_guard_traces import SMOKE_TARGET, collection_tool_digest

PROTOCOL = ROOT.parents[1] / "outputs/causal_attack_validation_20260929/protocol.json"
SEED_BAND = (6000, 6099)
STAGES = {"a": {"targets": 4, "first_seed": 6000, "role": "attacker_tuning", "probe_epochs": (12, 80)},
          "b": {"targets": 10, "first_seed": 6004, "role": "attacker_evaluation"}}
SCORES = ("projection", "cosine")
BOOTSTRAP, PERMUTATIONS, RNG_SEED = 10_000, 10_000, 20260929
DEFAULTS = {
    "use_hf_models": True, "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    "reference_model_id": "Qwen/Qwen2.5-0.5B-Instruct", "dataset_name": "squad_research",
    "dataset_revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
    "num_clients": 4, "clients_per_round": 4, "federated_rounds": 3, "local_epochs": 1,
    "local_batch_size": 2, "client_lr": 2.0e-05, "attack_trials": 40, "max_length": 128,
    "sim_num_gpus": 1.0, "sim_max_concurrent_clients": 1, "keep_artifacts": True,
    "threshold_mode": "calibrated", "calibration_nonmember_count": 100, "calibration_fpr": 0.05,
}
BASE = {
    "attack_variant": "causal_gradient_alignment", "attack_targets": 1, "attack_batch_size": 4,
    "probe_lr": 0.005, "adversary_negative_count": 16, "ldp_target_samples": 128,
    "certificate_samples": 256, "ldp_mechanism": "none", "observation_defense": "none",
    "observation_clip_norm": 1.0, "observation_delta": 1.0e-05, "counterbalance_trials": True,
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def tool_digest():
    return digest(__file__)


def _exclusions(manifests):
    excluded, sources, finals = {SMOKE_TARGET}, [], 0
    for path in manifests:
        raw = Path(path).read_bytes()
        manifest = json.loads(raw)
        if manifest.get("schema") in ("guard_collection_v3", "guard_collection_v4"):
            # A collector launch carries every target it excluded (earlier cohorts).
            groups = {target: "excluded_by_launch" for target in manifest.get("excluded_targets", [])}
        elif manifest.get("schema") == "guard_splits_v2":
            groups = manifest.get("groups") or {}
        else:
            raise ValueError(f"{path}: exclusions need a guard_splits_v2 manifest or a guard collection launch")
        if not groups:
            raise ValueError(f"{path}: exclusion source lists no targets")
        excluded.update(groups)
        finals += sum(role == "final" for role in groups.values())
        sources.append({"path": str(Path(path).resolve()), "sha256": sha256(raw).hexdigest(),
                        "schema": manifest["schema"], "groups": len(groups)})
    return excluded, sources, finals


def prepare(output, stage, exclude_manifests, selection=None, first_seed=None):
    """Write jobs and a guard_collection_v4 launch; nothing is resolved or run."""
    protocol = json.loads(PROTOCOL.read_bytes())
    if protocol.get("schema") != "causal_attack_validation_protocol_v1":
        raise ValueError("Unexpected protocol file")
    plan = STAGES[stage]
    first = plan["first_seed"] if first_seed is None else first_seed
    seeds = list(range(first, first + plan["targets"]))
    if not (SEED_BAND[0] <= seeds[0] and seeds[-1] <= SEED_BAND[1]):
        raise ValueError(f"Seeds must lie in the pre-registered band {SEED_BAND}")
    excluded, sources, finals = _exclusions(exclude_manifests)
    if finals < protocol["cohort"]["require_final_groups_at_least"]:
        raise ValueError("Exclusions must include the reserved-final targets (role 'final'); supply the pilot cohort.json")
    chosen = None
    if stage == "b":
        if selection is None:
            raise ValueError("Stage B requires the frozen Stage A selection")
        chosen = json.loads(Path(selection).read_bytes())
        if chosen.get("schema") != "causal_attack_selection_v1":
            raise ValueError("Unexpected selection file")
        stage_a = {s["sha256"] for s in sources}
        if chosen["stage_a_splits_sha256"] not in stage_a:
            raise ValueError("Stage B must exclude the Stage A targets: pass Stage A splits.complete.json")
        arms = [chosen["selected"]["probe_epochs"]]
    else:
        if selection is not None:
            raise ValueError("Stage A takes no selection")
        arms = list(plan["probe_epochs"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    jobs = []
    for seed in seeds:
        for epochs in arms:
            doc = {"defaults": {**DEFAULTS, "seed": seed},
                   "attacks": {"amia": {"base": {**BASE, "probe_epochs": epochs}, "sweep": {}}}}
            name = f"job-{len(jobs):03d}-seed{seed}-epochs{epochs}.yaml"
            text = yaml.safe_dump(doc, sort_keys=False)
            (output / name).write_text(text)
            jobs.append({"config": name, "sha256": sha256(text.encode()).hexdigest(), "seed": seed,
                         "role": plan["role"], "attack": "amia", "variant": "causal_gradient_alignment",
                         "probe_epochs": epochs})
    launch = {"schema": "guard_collection_v4", "study": "causal_attack_validation", "stage": stage,
              "implementation_fingerprint": implementation_fingerprint(),
              "collection_tool_sha256": collection_tool_digest(), "validation_tool_sha256": tool_digest(),
              "planned_targets": plan["targets"], "jobs": jobs, "held_out_variants": [],
              "excluded_targets": sorted(excluded), "exclusion_sources": sources, "reserved_final": [],
              "protocol_file": "protocol.json", "protocol_sha256": digest(output / "protocol.json"),
              **({"selection": {"path": str(Path(selection).resolve()), "sha256": digest(selection),
                                "selected": chosen["selected"]}} if chosen else {})}
    write_json(output / "launch.json", launch)
    return launch


def _completed(root):
    """Collector-verified results of a finished launch, with their declared job."""
    root = Path(root).resolve()
    launch = json.loads((root / "launch.json").read_bytes())
    if launch.get("study") != "causal_attack_validation":
        raise ValueError("Not a causal-attack validation launch")
    state = json.loads((root / "progress.json").read_bytes())
    if state.get("active_job") is not None or len(state["completed"]) != len(launch["jobs"]):
        raise ValueError("Launch is not complete; analyse only finished stages")
    rows = []
    for done in state["completed"]:
        path = (root / done["result"]).resolve()
        if root not in path.parents or digest(path) != done["sha256"]:
            raise ValueError("A completed result changed after collection")
        job = launch["jobs"][done["job"]]
        result = json.loads(path.read_bytes())
        target = state["targets"][str(job["seed"])]
        trials = result["attack_trials"]
        if (result.get("implementation_fingerprint") != launch["implementation_fingerprint"]
                or {t["target_sha256"] for t in trials} != {target}
                or result.get("config", {}).get("probe_epochs") != job["probe_epochs"]
                or len(trials) != 40 or any("alignment_terms" not in t for t in trials)
                or len(result["calibration"]["alignment_terms"]) != 100):
            raise ValueError(f"{path}: result does not match its declared job or lacks telemetry")
        rows.append({"job": job, "target": target, "result": result, "path": str(path), "sha256": done["sha256"]})
    return launch, state, rows


def score(terms, kind):
    if kind == "projection":
        return terms["dot"] / max(terms["direction_norm"], 1e-12)
    return terms["dot"] / max(terms["released_norm"] * terms["direction_norm"], 1e-24)


def auc(members, nonmembers):
    wins = sum(1.0 if m > n else 0.5 if m == n else 0.0 for m in members for n in nonmembers)
    return wins / (len(members) * len(nonmembers))


def target_view(row, kind):
    """Scores, labels and pairs of one target under one score definition."""
    trials = row["result"]["attack_trials"]
    values = [(score(t["alignment_terms"], kind), bool(t["truth_member"]), t["batch_pair_seed"]) for t in trials]
    pairs = {}
    for value, member, pair in values:
        pairs.setdefault(pair, {})[member] = value
    if any(set(p) != {True, False} for p in pairs.values()):
        raise ValueError("Each batch pair needs exactly one member and one nonmember trial")
    members = [v for v, m, _ in values if m]
    nonmembers = [v for v, m, _ in values if not m]
    return {"members": members, "nonmembers": nonmembers,
            "pairs": [(p[True], p[False]) for p in pairs.values()], "auc": auc(members, nonmembers)}


def select(stage_a, output):
    launch, state, rows = _completed(stage_a)
    if launch["stage"] != "a":
        raise ValueError("Selection needs the Stage A launch")
    splits = Path(stage_a) / "splits.complete.json"
    table = []
    for epochs in STAGES["a"]["probe_epochs"]:
        arm = [r for r in rows if r["job"]["probe_epochs"] == epochs]
        if len({r["target"] for r in arm}) != STAGES["a"]["targets"]:
            raise ValueError("Each Stage A arm needs all four targets")
        for kind in SCORES:
            per_target = {r["target"]: target_view(r, kind)["auc"] for r in arm}
            table.append({"probe_epochs": epochs, "score": kind, "per_target_auc": per_target,
                          "mean_auc": sum(per_target.values()) / len(per_target)})
    # Highest mean AUC; ties to fewer epochs, then projection.
    best = min(table, key=lambda c: (-c["mean_auc"], c["probe_epochs"], SCORES.index(c["score"])))
    selection = {"schema": "causal_attack_selection_v1", "rule": json.loads(PROTOCOL.read_bytes())["stage_a"]["selection"],
                 "selected": {"probe_epochs": best["probe_epochs"], "score": best["score"]}, "table": table,
                 "stage_a_launch_sha256": digest(Path(stage_a) / "launch.json"),
                 "stage_a_splits_sha256": digest(splits),
                 "results": [{"path": r["path"], "sha256": r["sha256"]} for r in rows],
                 "validation_tool_sha256": tool_digest()}
    if Path(output).exists():
        raise ValueError("Selection already exists; never overwrite a frozen choice")
    write_json(output, selection)
    return selection


def _interval(values):
    import numpy as np
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _sign_test(positive, negative):
    """One-sided exact P(X >= positive) for X ~ Binomial(positive + negative, 1/2)."""
    n = positive + negative
    return sum(math.comb(n, k) for k in range(positive, n + 1)) / 2 ** n if n else 1.0


def analyze(stage_b, selection_path, output):
    import numpy as np
    from master_script.core.calibration import nonmember_threshold
    launch, state, rows = _completed(stage_b)
    selection = json.loads(Path(selection_path).read_bytes())
    if launch["stage"] != "b" or launch["selection"]["sha256"] != digest(selection_path):
        raise ValueError("Stage B launch was not prepared from this selection")
    if len({r["target"] for r in rows}) != STAGES["b"]["targets"]:
        raise ValueError("Stage B needs ten distinct targets")
    kind, protocol = selection["selected"]["score"], json.loads(PROTOCOL.read_bytes())
    if digest(Path(stage_b) / "protocol.json") != digest(PROTOCOL):
        raise ValueError("Stage B protocol differs from the committed protocol")
    targets = []
    for row in rows:
        view = target_view(row, kind)
        calibration = [score(t, kind) for t in row["result"]["calibration"]["alignment_terms"]]
        threshold = nonmember_threshold(calibration, protocol["job"]["calibration_fpr"])["threshold"]
        if kind == "projection" and not math.isclose(threshold, row["result"]["calibration"]["threshold"], rel_tol=1e-12):
            raise ValueError("Recomputed projection threshold differs from the run's own calibration")
        diffs = [m - n for m, n in view["pairs"]]
        targets.append({"target_sha256": row["target"], "result_sha256": row["sha256"], "auc": view["auc"],
                        "threshold": threshold,
                        "tp": sum(v >= threshold for v in view["members"]), "members": len(view["members"]),
                        "fp": sum(v >= threshold for v in view["nonmembers"]), "nonmembers": len(view["nonmembers"]),
                        "pairs_positive": sum(d > 0 for d in diffs), "pairs_negative": sum(d < 0 for d in diffs),
                        "pairs": len(diffs), "_view": view})
    rng = np.random.default_rng(RNG_SEED)
    n = len(targets)
    boot = rng.integers(0, n, size=(BOOTSTRAP, n))
    aucs = np.array([t["auc"] for t in targets])
    tp, members = np.array([t["tp"] for t in targets]), np.array([t["members"] for t in targets])
    fp, nonmembers = np.array([t["fp"] for t in targets]), np.array([t["nonmembers"] for t in targets])
    frac = np.array([(t["pairs_positive"] + (t["pairs"] - t["pairs_positive"] - t["pairs_negative"]) / 2) / t["pairs"]
                     for t in targets])
    pooled_auc = float(aucs.mean())
    auc_ci = _interval(aucs[boot].mean(axis=1))
    fpr_boot = fp[boot].sum(axis=1) / nonmembers[boot].sum(axis=1)
    # Within-pair label swaps: the exact null of this paired design.
    null = np.zeros((PERMUTATIONS, n))
    for j, t in enumerate(targets):
        pairs = np.array(t["_view"]["pairs"])
        for i in range(PERMUTATIONS):
            swap = rng.integers(0, 2, size=len(pairs)).astype(bool)
            m = np.where(swap, pairs[:, 1], pairs[:, 0]); o = np.where(swap, pairs[:, 0], pairs[:, 1])
            null[i, j] = ((m[:, None] > o[None, :]).sum() + 0.5 * (m[:, None] == o[None, :]).sum()) / (len(m) * len(o))
        t["null_percentile"] = float((null[:, j] < t["auc"]).mean() + 0.5 * (null[:, j] == t["auc"]).mean())
    positive = sum(t["pairs_positive"] for t in targets); negative = sum(t["pairs_negative"] for t in targets)
    lower, upper = auc_ci
    verdict = ("effective" if lower >= 0.60 else "not_effective" if upper < 0.60 else "inconclusive")
    report = {
        "schema": "causal_attack_analysis_v1", "selected": selection["selected"],
        "stage_b_launch_sha256": digest(Path(stage_b) / "launch.json"), "selection_sha256": digest(selection_path),
        "protocol_sha256": digest(PROTOCOL), "validation_tool_sha256": tool_digest(),
        "implementation_fingerprint": launch["implementation_fingerprint"],
        "primary": {"mean_per_target_auc": pooled_auc, "ci95": auc_ci, "targets": n,
                    "null_percentile_of_mean_auc": float((null.mean(axis=1) < pooled_auc).mean())},
        "operating_point": {"target_fpr": protocol["job"]["calibration_fpr"],
                            "tpr": float(tp.sum() / members.sum()), "tpr_ci95": _interval(tp[boot].sum(axis=1) / members[boot].sum(axis=1)),
                            "fpr": float(fp.sum() / nonmembers.sum()), "fpr_ci95": _interval(fpr_boot)},
        "paired": {"fraction_member_higher": float(frac.mean()), "ci95": _interval(frac[boot].mean(axis=1)),
                   "positive": positive, "negative": negative, "ties": sum(t["pairs"] for t in targets) - positive - negative,
                   "sign_test_p_one_sided": _sign_test(positive, negative)},
        "decision": verdict,
        "calibration_failure": _interval(fpr_boot)[0] > 0.10,
        "per_target": [{k: v for k, v in t.items() if k != "_view"} for t in targets],
        "method": {"bootstrap": BOOTSTRAP, "permutations": PERMUTATIONS, "rng_seed": RNG_SEED},
    }
    if Path(output).exists():
        raise ValueError("Analysis already exists; write a new file")
    write_json(output, report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="write a frozen launch for collect_guard_traces resolve/run")
    prep.add_argument("output"); prep.add_argument("--stage", choices=sorted(STAGES), required=True)
    prep.add_argument("--exclude-manifest", action="append", default=[], required=True)
    prep.add_argument("--selection"); prep.add_argument("--first-seed", type=int)
    sel = sub.add_parser("select", help="apply the Stage A rule"); sel.add_argument("stage_a"); sel.add_argument("output")
    ana = sub.add_parser("analyze", help="Stage B endpoints and decision")
    ana.add_argument("stage_b"); ana.add_argument("selection"); ana.add_argument("output")
    args = p.parse_args(argv)
    if args.action == "prepare":
        launch = prepare(args.output, args.stage, args.exclude_manifest, args.selection, args.first_seed)
        print(f"Prepared {len(launch['jobs'])} jobs for stage {args.stage.upper()}. Nothing resolved or run.")
        print(f"Next: python -m master_script.tools.collect_guard_traces resolve {Path(args.output).resolve() / 'launch.json'}")
    elif args.action == "select":
        print(json.dumps(select(args.stage_a, args.output)["selected"]))
    else:
        report = analyze(args.stage_b, args.selection, args.output)
        print(json.dumps({k: report[k] for k in ("primary", "operating_point", "paired", "decision", "calibration_failure")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
