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
                     "status": result.get("status"), "gradient_auc": metrics.get("roc_auc"),
                     "balanced_accuracy": metrics.get("adv"), "release_coverage": metrics.get("release_coverage"),
                     "no_retrieval_f1": float(np.mean(utilities)) if utilities else None,
                     "utility_valid": bool(utilities) and all(v > 0 for v in utilities),
                     "guard": result.get("guard_summary"),
                     "exclusion": "operational_or_policy_abort" if result.get("status") != "complete"
                                  else "utility_unmeasured_or_zero" if not utilities or not all(v > 0 for v in utilities) else None})
    return {"conditions": rows, "total_runs": len(rows),
            "excluded_runs": sum(r["exclusion"] is not None for r in rows),
            "interpretation": "Utility validity here is a screening floor, not the predeclared matched-baseline engineering gate. No averaged privacy score."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+"); parser.add_argument("--output", required=True)
    parser.add_argument("--plot", help="Optional PNG with separate attack panels")
    args = parser.parse_args()
    raw = [Path(p).read_bytes() for p in args.results]
    results = [json.loads(data) for data in raw]
    report = summarize_results(results)
    report["inputs"] = [{"path": str(Path(p).resolve()), "sha256": sha256(data).hexdigest()} for p, data in zip(args.results, raw)]
    report["provenance"] = [{k: r.get(k) for k in ("run_id", "implementation_fingerprint", "method_version", "pipeline", "config")} for r in results]
    report["paired_utility"] = paired_utility_report(results)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot_tradeoffs(report, args.plot)


def paired_utility_report(results, baseline_condition="baseline", *, minimum_f1=.1):
    """Matched seed/model/dataset comparisons; never match unrelated checkpoints."""
    def key(r):
        cfg = r.get("config", {})
        return (r.get("attack_name"), *(cfg.get(k) for k in ("model_id", "model_revision", "dataset_name", "dataset_revision", "seed", "federated_rounds", "num_clients", "clients_per_round", "max_length")))
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
            if row["gradient_auc"] is None or row["no_retrieval_f1"] is None:
                missing += 1
                continue
            ax.scatter(row["gradient_auc"], row["no_retrieval_f1"], marker="o" if row["utility_valid"] else "x")
            ax.annotate(row["condition"], (row["gradient_auc"], row["no_retrieval_f1"]), fontsize=7)
        ax.set(xlabel="Membership AUC (accepted responses where guarded)", ylabel="No-context answer F1", title=f"{attack}; {missing} unavailable conditions", xlim=(0,1), ylim=(0,1))
        ax.axvline(.5, linestyle="--", color="gray")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
