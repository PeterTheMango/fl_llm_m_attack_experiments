"""Metrics. Adv = 0.5*TPR + 0.5*TNR (Nguyen et al. 2023, Eq. 3).

base_metrics() intentionally omits roc_auc: reference_adaptations.ipynb does not
emit it, and consolidation must not change any notebook's document shape.
Attacks opt into extra keys through AttackSpec.extra_metrics.
"""
import math
from typing import Dict, Sequence


def validate_scores(labels, scores):
    if len(labels) != len(scores) or not labels:
        raise ValueError("Metrics require equally sized nonempty labels and scores")
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("Metrics cannot accept nonfinite scores")


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Compute AUC-ROC as the proportion of positive-negative score pairs where the positive
    scores higher (ties count 0.5). Ported verbatim from zlib_adaptations.ipynb cell 17."""
    validate_scores(labels, scores)
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def tpr_at_fpr(labels: Sequence[bool], scores: Sequence[float], target_fpr: float = 0.05) -> float:
    """Compute TPR at a given FPR threshold, scanning thresholds in descending order.
    Ported verbatim from samia_adaptations.ipynb."""
    validate_scores(labels, scores)
    if not math.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must lie in [0, 1]")
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    if not pos or not neg:
        return float("nan")
    best_tpr = 0.0
    for threshold in sorted(set(scores), reverse=True):
        fpr = sum(1 for n in neg if n >= threshold) / len(neg)
        if fpr <= target_fpr:
            tpr = sum(1 for p in pos if p >= threshold) / len(pos)
            best_tpr = max(best_tpr, tpr)
    return best_tpr


def base_metrics(trials: Sequence[Dict]) -> Dict:
    """Compute base metrics (no roc_auc) exactly as reference_adaptations.ipynb produces.

    Returns exactly: tp, tn, fp, fn, tpr, tnr, adv, accuracy, precision, recall, f1, num_trials.
    Deliberately omits roc_auc so the result shape matches reference_adaptations.ipynb exactly.
    """
    tp = sum(1 for row in trials if row["truth_member"] and row["pred_member"])
    tn = sum(1 for row in trials if not row["truth_member"] and not row["pred_member"])
    fp = sum(1 for row in trials if not row["truth_member"] and row["pred_member"])
    fn = sum(1 for row in trials if row["truth_member"] and not row["pred_member"])
    tpr = tp / (tp + fn) if (tp + fn) else None
    tnr = tn / (tn + fp) if (tn + fp) else None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tpr": tpr, "tnr": tnr,
        "adv": 0.5 * (tpr + tnr) if tpr is not None and tnr is not None else None,
        "accuracy": (tp + tn) / len(trials) if trials else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "num_trials": len(trials),
    }


def summarize(trials: Sequence[Dict], spec) -> Dict:
    """Compute base_metrics plus whatever the attack's extra_metrics hook contributes."""
    out = base_metrics(trials)
    if spec.extra_metrics is not None:
        out.update(spec.extra_metrics(trials))
    labels = [t["truth_member"] for t in trials]
    scores = [(-1 if spec.name == "loss" else 1) * t["score"] for t in trials]
    out.update(scientific_metrics(labels, scores))
    return out


def scientific_metrics(labels, scores):
    validate_scores(labels, scores)
    n_neg, n_pos = labels.count(False), labels.count(True)
    result = {"roc_auc": roc_auc(labels, scores) if n_neg and n_pos else None,
              "nonmember_count": n_neg, "member_count": n_pos,
              "low_fpr_resolution": 1 / n_neg if n_neg else None}
    for fpr in (0.1, 0.05, 0.01, 0.001):
        result[f"tpr_at_fpr_{str(fpr).replace('.', '_')}"] = (
            tpr_at_fpr(labels, scores, fpr) if n_pos and n_neg >= math.ceil(1 / fpr) else None)
    result["low_fpr_note"] = "Null denotes insufficient empirical FPR resolution; no population guarantee."
    return result


def validate_observations(trials):
    if not trials:
        raise ValueError("A completed evaluation requires observations")
    for row in trials:
        if row.get("decision") == "rejected":
            if row.get("score") is not None or row.get("pred_member") is not None or row.get("gradient_available", False):
                raise ValueError("Rejected observations cannot carry gradient scores or predictions")
        elif (row.get("decision", "accepted") != "accepted" or row.get("score") is None
              or not math.isfinite(float(row["score"])) or type(row.get("pred_member")) is not bool):
            raise ValueError("Accepted observations require finite scores and boolean predictions")


def guarded_metrics(trials):
    """Conditional gradient metrics; refusals are never negative predictions."""
    validate_observations(trials)
    accepted = [t for t in trials if t.get("decision") != "rejected"]
    labels = [t["truth_member"] for t in trials]
    decisions = [float(t.get("decision") == "rejected") for t in trials]
    result = base_metrics(accepted)
    if accepted:
        result.update(scientific_metrics([t["truth_member"] for t in accepted], [t["score"] for t in accepted]))
    else:
        result.update({k: None for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})
    result.update(total_requests=len(trials), accepted_requests=len(accepted),
                  rejected_requests=len(trials) - len(accepted), release_coverage=len(accepted) / len(trials),
                  metric_scope="conditional_on_accepted_gradients", adv_definition="balanced_accuracy",
                  gradient_status="evaluated" if accepted else "unavailable_all_rejected",
                  decision_transcript_metrics=scientific_metrics(labels, decisions),
                  transcript_scope="refusal-only score; not a full adaptive transcript attack")
    timings = [t for t in trials if t.get("response_seconds") is not None]
    if timings:
        result["timing_transcript_metrics"] = scientific_metrics(
            [t["truth_member"] for t in timings], [t["response_seconds"] for t in timings])
        result["timing_scope"] = "client processing time; excludes transport; descriptive score, no fitted timing attacker"
    return result


def guard_event_summary(events):
    training = [e for e in events if e["accounting_scope"] == "training"]
    by_client = {}
    for e in training:
        key = str(e["client_id"])
        item = by_client.setdefault(key, {"requests": 0, "rejected": 0, "shadow_rejections": 0})
        item["requests"] += 1
        item["rejected"] += e["decision"] == "rejected"
        item["shadow_rejections"] += not e["would_accept"]
    return {"training_requests": len(training),
            "training_rejected": sum(e["decision"] == "rejected" for e in training),
            "legitimate_false_rejection_rate": sum(e["decision"] == "rejected" for e in training) / len(training) if training else None,
            "policy_flag_rate": sum(not e["would_accept"] for e in training) / len(training) if training else None,
            "per_client": by_client,
            "guard_seconds": sum(e["seconds"] for e in events),
            "sample_size_note": "Repeated rounds are not independent evidence for a 1% population false-positive rate"}
