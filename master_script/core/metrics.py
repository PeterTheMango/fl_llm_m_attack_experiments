"""Metrics. Adv = 0.5*TPR + 0.5*TNR (Nguyen et al. 2023, Eq. 3).

base_metrics() intentionally omits roc_auc: reference_adaptations.ipynb does not
emit it, and consolidation must not change any notebook's document shape.
Attacks opt into extra keys through AttackSpec.extra_metrics.
"""
from typing import Dict, Sequence


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Compute AUC-ROC as the proportion of positive-negative score pairs where the positive
    scores higher (ties count 0.5). Ported verbatim from zlib_adaptations.ipynb cell 17."""
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
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tpr": tpr, "tnr": tnr,
        "adv": 0.5 * tpr + 0.5 * tnr,
        "accuracy": (tp + tn) / len(trials) if trials else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "num_trials": len(trials),
    }


def summarize(trials: Sequence[Dict], spec) -> Dict:
    """Compute base_metrics plus whatever the attack's extra_metrics hook contributes."""
    out = base_metrics(trials)
    if spec.extra_metrics is not None:
        out.update(spec.extra_metrics(trials))
    return out
