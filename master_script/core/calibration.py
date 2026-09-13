"""Held-out nonmember calibration; never choose a threshold using test labels."""
import math


def nonmember_threshold(scores, target_fpr):
    values = sorted(float(s) for s in scores)
    if not values or not all(math.isfinite(s) for s in values):
        raise ValueError("Calibration needs nonempty finite nonmember scores")
    if not 0 < target_fpr < 1 or len(values) < math.ceil(1 / target_fpr):
        raise ValueError("Calibration count cannot resolve the requested FPR")
    allowed = math.floor(len(values) * target_fpr + 1e-12)
    # Strictly exceed the boundary, including when calibration scores tie.
    threshold = math.nextafter(values[len(values) - allowed - 1], math.inf)
    if not math.isfinite(threshold):
        raise ValueError("Calibration threshold is not finite")
    return {"threshold": threshold, "comparator": ">=", "target_fpr": target_fpr,
            "calibration_count": len(values),
            "calibration_fpr": sum(s >= threshold for s in values) / len(values),
            "selection": "held_out_nonmembers_only", "population_fpr_guarantee": False}
