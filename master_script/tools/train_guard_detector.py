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
    args = parser.parse_args()
    raw = Path(args.dataset).read_bytes()
    rows = json.loads(raw)
    artifact = fit_detector(rows, max_false_positive_rate=args.max_fpr)
    encoded = (json.dumps(artifact, indent=2, allow_nan=False) + "\n").encode()
    Path(args.output).write_bytes(encoded)
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
    Path(args.output + ".report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["detector_sha256"])


if __name__ == "__main__":
    main()
