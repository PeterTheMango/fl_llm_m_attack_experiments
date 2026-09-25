"""Join completed and failed conditions without treating refusals as nonmembers."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
import numpy as np


def clustered_mean_interval(rows, key, group_key, *, seed=0, draws=2000):
    groups = {}
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            groups.setdefault(str(row[group_key]), []).append(float(value))
    means = [float(np.mean(values)) for values in groups.values()]
    if not means:
        return {"mean": None, "interval_95": None, "groups": 0}
    result = {"mean": float(np.mean(means)), "interval_95": None, "groups": len(means)}
    if len(means) > 1:
        rng = np.random.default_rng(seed)
        samples = rng.choice(means, size=(draws, len(means)), replace=True).mean(axis=1)
        result["interval_95"] = np.quantile(samples, [.025, .975]).tolist()
    return result


def summarize_results(results):
    rows = []
    for result in results:
        metrics = result.get("metrics", {})
        evaluations = result.get("pipeline_evaluations", [])
        utilities = [e["no_retrieval_utility"]["token_f1"] for e in evaluations if e.get("no_retrieval_utility", {}).get("token_f1") is not None]
        rows.append({"run_id": result["run_id"], "attack": result.get("attack_name"),
                     "condition": result.get("pipeline", {}).get("condition", "baseline"),
                     "status": result.get("status"), "membership_auc": metrics.get("roc_auc"),
                     "gradient_auc": metrics.get("roc_auc") if result.get("attack_name") == "amia" else None,
                     "balanced_accuracy": metrics.get("adv"), "release_coverage": metrics.get("release_coverage"),
                     "no_retrieval_f1": float(np.mean(utilities)) if utilities else None,
                     "utility_valid": bool(utilities) and all(v >= .1 for v in utilities),
                     "absolute_f1_screen": .1,
                     "guard": result.get("guard_summary"),
                     "exclusion": "operational_or_policy_abort" if result.get("status") != "complete"
                                  else "utility_unmeasured_or_zero" if not utilities or not all(v > 0 for v in utilities)
                                  else "utility_below_absolute_floor" if not all(v >= .1 for v in utilities) else None})
    return {"conditions": rows, "total_runs": len(rows),
            "excluded_runs": sum(r["exclusion"] is not None for r in rows),
            "interpretation": "Utility validity here is a screening floor, not the predeclared matched-baseline engineering gate. No averaged privacy score."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+"); parser.add_argument("--output", required=True)
    parser.add_argument("--plot", help="Optional PNG with separate attack panels")
    parser.add_argument("--protocol", help="Optional explicit utility protocol; old screening fields remain unchanged")
    args = parser.parse_args()
    raw = [Path(p).read_bytes() for p in args.results]
    results = [json.loads(data) for data in raw]
    report = summarize_results(results)
    report["inputs"] = [{"path": str(Path(p).resolve()), "sha256": sha256(data).hexdigest()} for p, data in zip(args.results, raw)]
    report["provenance"] = [{k: r.get(k) for k in ("run_id", "implementation_fingerprint", "method_version", "pipeline", "config")} for r in results]
    report["paired_utility"] = paired_utility_report(results)
    report["study_v2"] = joined_study_report(results)
    if args.protocol:
        from master_script.core.guard_protocol import utility_report
        protocol_raw = Path(args.protocol).read_bytes()
        report["utility_protocol"] = utility_report(results, json.loads(protocol_raw))
        report["utility_protocol"]["sha256"] = sha256(protocol_raw).hexdigest()
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot_tradeoffs(report, args.plot)


def paired_utility_report(results, baseline_condition="baseline", *, minimum_f1=.1):
    """Matched seed/model/dataset comparisons; never match unrelated checkpoints."""
    key = pairing_key
    baselines = {}
    for r in results:
        if r.get("pipeline", {}).get("condition", "baseline") == baseline_condition and r.get("status") == "complete":
            if key(r) in baselines:
                raise ValueError("Ambiguous matched baseline")
            baselines[key(r)] = r
    comparisons = []
    for r in results:
        if r.get("pipeline", {}).get("condition", "baseline") == baseline_condition:
            continue
        base = baselines.get(key(r))
        comparisons.append({"run_id": r["run_id"], "baseline_run_id": base["run_id"] if base else None,
                            "status": "unmatched" if base is None else "operational_or_policy_abort" if r.get("status") != "complete" else "matched",
                            "slices": []})
        if base is None or r.get("status") != "complete":
            continue
        baseline_evaluations = {e["trial_id"]: e for e in base.get("pipeline_evaluations", [])}
        for e in r.get("pipeline_evaluations", []):
            b = baseline_evaluations.get(e["trial_id"])
            if not b:
                continue
            slices = [("no_retrieval", b.get("no_retrieval_utility"), e.get("no_retrieval_utility"))]
            slices += [(name, b.get("rag_conditions", {}).get(name, {}).get("utility"), item.get("utility"))
                       for name, item in e.get("rag_conditions", {}).items()]
            for name, bu, ru in slices:
                if not bu or not ru or any(u.get(k) is None for u in (bu, ru) for k in ("token_f1", "exact_match")):
                    continue
                comparisons[-1]["slices"].append({"trial_id": e["trial_id"], "slice": name,
                    "group": next((t.get("target_sha256", str(t.get("seed", e["trial_id"] // 2))) for t in r.get("attack_trials", []) if t.get("trial_id") == e["trial_id"]), str(e["trial_id"] // 2)),
                    "f1_difference": ru["token_f1"] - bu["token_f1"],
                    "em_difference": ru["exact_match"] - bu["exact_match"],
                    "utility_gate": ru["token_f1"] >= max(minimum_f1, .9 * bu["token_f1"])
                                    and ru["exact_match"] >= bu["exact_match"] - .05})
        comparisons[-1]["f1_intervals_by_slice"] = {
            name: clustered_mean_interval([s for s in comparisons[-1]["slices"] if s["slice"] == name],
                                          "f1_difference", "group")
            for name in sorted({s["slice"] for s in comparisons[-1]["slices"]})}
    return comparisons


def plot_tradeoffs(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    attacks = sorted({str(r["attack"]) for r in report["conditions"]})
    fig, axes = plt.subplots(1, max(1, len(attacks)), figsize=(6 * max(1, len(attacks)), 4), squeeze=False)
    for ax, attack in zip(axes[0], attacks):
        missing = 0
        for row in report["conditions"]:
            if str(row["attack"]) != attack:
                continue
            if row["membership_auc"] is None or row["no_retrieval_f1"] is None:
                missing += 1
                continue
            ax.scatter(row["membership_auc"], row["no_retrieval_f1"], marker="o" if row["utility_valid"] else "x")
            ax.annotate(row["condition"], (row["membership_auc"], row["no_retrieval_f1"]), fontsize=7)
        ax.set(xlabel="Membership AUC (accepted responses where guarded)", ylabel="No-context answer F1", title=f"{attack}; {missing} unavailable conditions", xlim=(0,1), ylim=(0,1))
        ax.axvline(.5, linestyle="--", color="gray")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)



def pairing_key(result):
    """Match all non-defense choices and source revision, not merely seed/model."""
    config = dict(result.get("config", {}))
    for key in ("observation_defense", "observation_clip_norm", "observation_noise_multiplier", "observation_delta",
                "artifact_root", "local_artifact_dir", "keep_artifacts"):
        config.pop(key, None)
    rag = dict(result.get("pipeline", {}).get("rag") or {})
    rag.pop("study_file", None)
    # Retrieval defenses are interventions; the study and query settings must match.
    rag.pop("defenses", None)
    return json.dumps([result.get("attack_name"), result.get("implementation_fingerprint"),
                       result.get("method_version"), config, rag], sort_keys=True)


def joined_study_report(results):
    from master_script.core.guard_statistics import collapse_worlds, privacy_report
    reports = []
    baseline = {pairing_key(r): r for r in results
                if r.get("pipeline", {}).get("condition", "baseline") == "baseline" and r.get("status") == "complete"}
    for result in results:
        row = {"run_id": result["run_id"], "status": result.get("status"),
               "privacy_unit": "private_client_batch" if result.get("attack_name") == "amia" else "training_record"}
        trials = result.get("attack_trials", [])
        try:
            worlds = collapse_worlds(trials)
            row["raw_privacy"] = privacy_report(worlds)
            row["raw_privacy"]["interpretation"] = "Raw direction only; use frozen independent calibration for attack claims"
            if any(t.get("decision") for t in trials):
                row["decision_transcript"] = privacy_report(collapse_worlds(trials, "decision"))
            if any(t.get("response_seconds") is not None for t in trials):
                row["timing_transcript"] = privacy_report(collapse_worlds(trials, "response_seconds"))
        except ValueError as exc:
            row["privacy_unavailable_reason"] = str(exc)
        events = result.get("guard_events", [])
        stages = sorted({key for e in events for key in e.get("stages_seconds", {})})
        row["guard_profile_seconds"] = {key: {"total": sum(e.get("stages_seconds", {}).get(key, 0) for e in events),
            "median": float(np.median([e["stages_seconds"][key] for e in events if key in e.get("stages_seconds", {})]))}
            for key in stages}
        row["per_client"] = (result.get("guard_summary") or {}).get("per_client", {})
        rounds = result.get("training_rounds", [])
        row["availability"] = {"completed_rounds": sum(r["status"] == "completed" for r in rounds) if rounds else None,
                               "aborted_rounds": sum(r["status"] == "policy_aborted" for r in rounds) if rounds else None,
                               "recovery": "stop_world_preserve_ledger; no automatic incomplete aggregation"}
        # Repeated events/rounds cannot certify a 1% population false-rejection gate.
        row["benign_rejection_gate"] = {"status": "insufficient_independent_evidence",
                                       "target": .01, "zero_error_independent_samples_needed_95pct": 299}
        paired = baseline.get(pairing_key(result))
        row["paired_runtime"] = {"baseline_run_id": paired["run_id"] if paired else None,
                                 "median_round_overhead": None, "gate_status": "unmeasured"}
        if paired and result.get("computation_seconds") is not None and paired.get("computation_seconds"):
            row["paired_runtime"]["end_to_end_ratio"] = result["computation_seconds"] / paired["computation_seconds"]
            row["paired_runtime"]["interpretation"] = "End-to-end includes saved attack work after rejection; not isolated guard cost"
            def round_map(r):
                contexts = r.get("target_evaluations") or [{"fed_history": r.get("federated_history", [])}]
                items = {}
                for target, ctx in enumerate(contexts):
                    for outer, h in enumerate(ctx.get("fed_history", [])):
                        for j, item in enumerate(h.get("rounds", [h])):
                            if item.get("round_seconds") is not None:
                                items[(target, outer if "rounds" in h else 0, j if "rounds" in h else outer)] = item["round_seconds"]
                return items
            br, rr = round_map(paired), round_map(result)
            ratios = [rr[k]/br[k]-1 for k in rr.keys() & br.keys() if br[k] > 0]
            if ratios:
                row["paired_runtime"].update(median_round_overhead=float(np.median(ratios)), paired_rounds=len(ratios),
                                             gate_status="descriptive_only; paired rounds are dependent")
        row["rag"] = {}
        for evaluation in result.get("pipeline_evaluations", []):
            for condition, item in evaluation.get("rag_conditions", {}).items():
                entry = row["rag"].setdefault(condition, {"rows": [], "utility": []})
                entry["rows"].extend(item.get("membership_trials", []))
                entry["utility"].append(item.get("utility", {}))
        for condition, entry in row["rag"].items():
            try:
                entry["privacy"] = privacy_report(collapse_worlds(entry["rows"]))
            except ValueError:
                entry["privacy"] = {"status": "historical_document_identities_unavailable"}
            entry["recognition_rate"] = sum(r.get("answer_recognized", False) for r in entry["rows"])/len(entry["rows"]) if entry["rows"] else None
            entry["refusal_rate"] = sum(r.get("refused", False) for r in entry["rows"])/len(entry["rows"]) if entry["rows"] else None
            entry.pop("rows")
        overlaps = [cell for evaluation in result.get("pipeline_evaluations", [])
                    for cell in evaluation.get("membership_overlap_cells", [])]
        row["overlap"] = overlap_report(overlaps)
        reports.append(row)
    return {"runs": reports, "privacy_scores_averaged": False,
            "formal_scope": "Classifier supplies no DP; training, observation and retrieval units remain separate",
            "full_transcript_limit": "Decision and timing channels reported separately; joint adaptive transcript leakage not bounded"}



def overlap_report(cells):
    grouped = {}
    for cell in cells:
        key = (cell["target_sha256"], cell["defense"], cell["training_member"], cell["datastore_member"])
        grouped.setdefault(key, []).append(cell)
    worlds = {}
    for key, rows in grouped.items():
        recognized = [r["pred_member"] for r in rows if type(r.get("pred_member")) is bool]
        worlds[key] = float(np.mean(recognized)) if recognized else None
    strata = []
    for target, defense in sorted({k[:2] for k in worlds}):
        values = {(t,d): worlds.get((target,defense,t,d)) for t in (False,True) for d in (False,True)}
        complete = all(v is not None for v in values.values())
        strata.append({"target_sha256": target, "defense": defense, "four_cells_available": complete,
            "cells": {f"training_{int(t)}_datastore_{int(d)}": v for (t,d),v in values.items()},
            "datastore_effect_without_training": values[False,True]-values[False,False] if complete else None,
            "datastore_effect_with_training": values[True,True]-values[True,False] if complete else None,
            "interaction": values[True,True]-values[True,False]-values[False,True]+values[False,False] if complete else None,
            "constant_prediction": len(set(values.values())) == 1 if complete else None})
    return {"independent_targets": len({c["target_sha256"] for c in cells}), "per_target": strata,
            "interpretation": "Paired response differences; constant-positive four cells show response bias, not overlap leakage"}


if __name__ == "__main__":
    main()
