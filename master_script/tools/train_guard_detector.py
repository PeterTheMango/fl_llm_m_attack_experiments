"""Fit and freeze JSON logistic coefficients using preassigned grouped splits."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from master_script.core.guard_detector import fit_detector, detector_score
from master_script.core.metrics import base_metrics, scientific_metrics


def evaluate_detector(data, rows, split):
    trials = [{"truth_member": r["malicious"], "score": detector_score(data, r["features"]),
               "pred_member": detector_score(data, r["features"]) > data["threshold"]}
              for r in rows if r["split"] == split]
    return {**base_metrics(trials), **scientific_metrics([r["truth_member"] for r in trials], [r["score"] for r in trials]),
            "label_scope": "request_origin_not_membership", "independent_groups": len(data["split_groups"][split])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset"); parser.add_argument("output")
    parser.add_argument("--max-fpr", type=float, default=.01)
    parser.add_argument("--prevalence", type=float, default=.01)
    args = parser.parse_args()
    if not 0 < args.prevalence < 1:
        parser.error("prevalence must lie in (0,1)")
    raw = Path(args.dataset).read_bytes()
    rows = json.loads(raw)
    artifact = fit_detector(rows, max_false_positive_rate=args.max_fpr)
    encoded = (json.dumps(artifact, indent=2, allow_nan=False) + "\n").encode()
    with Path(args.output).open("xb") as stream:
        stream.write(encoded)
    benign_deltas = sorted(r["features"]["relative_delta"] for r in rows
                           if r["split"] == "validation" and not r["malicious"])
    baseline_threshold = benign_deltas[len(benign_deltas) - int(args.max_fpr * len(benign_deltas)) - 1]
    baseline_rows = [{"truth_member": r["malicious"], "score": r["features"]["relative_delta"],
                      "pred_member": r["features"]["relative_delta"] > baseline_threshold}
                     for r in rows if r["split"] == "test"]
    report = {"relative_delta_baseline": {"threshold": baseline_threshold, **base_metrics(baseline_rows)},
              "detector_sha256": sha256(encoded).hexdigest(), "dataset_sha256": sha256(raw).hexdigest(),
              "validation": evaluate_detector(artifact, rows, "validation"),
              "test": evaluate_detector(artifact, rows, "test"),
              "warning": "Empirical grouped evaluation; a small validation set cannot establish a 1% population FPR"}
    for split in ("validation", "test"):
        m = report[split]
        fpr = 1 - m["tnr"]
        denom = args.prevalence * m["tpr"] + (1-args.prevalence)*fpr
        m["declared_prevalence"] = args.prevalence
        m["precision_at_declared_prevalence"] = args.prevalence*m["tpr"]/denom if denom else None
        m["per_variant"] = {}
        for variant in sorted({r["variant"] for r in rows if r["split"] == split}):
            subset = [r for r in rows if r["split"] == split and r["variant"] == variant]
            m["per_variant"][variant] = {"requests": len(subset), "groups": len({r["group"] for r in subset}),
                "rejection_rate": sum(detector_score(artifact, r["features"]) > artifact["threshold"] for r in subset)/len(subset)}
    with Path(args.output + ".report.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["detector_sha256"])


if __name__ == "__main__":
    main()
