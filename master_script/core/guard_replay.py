"""Bounded public parameter evasion/repeated-request mechanics.

Interpolation preserves parameter shapes, not attack effectiveness. This is a
new request variant, not a claim that the original AMIA theorem transfers. The
caller must separately establish its private membership signal and utility.
"""
import numpy as np
from .guard_features import parameter_features
from .guard_detector import detector_score


def interpolate_request(approved, malicious, fraction):
    if not 0 <= fraction <= 1 or len(approved) != len(malicious) or not approved:
        raise ValueError("Interpolation needs matched parameters and a fraction in [0,1]")
    result = []
    for prior, incoming in zip(approved, malicious):
        if prior.shape != incoming.shape or not np.isfinite(prior).all() or not np.isfinite(incoming).all():
            raise ValueError("Malformed parameter request")
        result.append(((1 - fraction) * prior + fraction * incoming).astype(prior.dtype))
    return result


def adaptive_public_requests(approved, malicious, detector, query_budget):
    """Decision-oracle bisection, seeking the largest accepted interpolation.

    Each yielded candidate consumes one query in the calling experiment, even
    when rejected. Decisions use public parameters only. This does not calculate
    or inspect private gradients, choose membership labels, or certify an attack.
    """
    if type(query_budget) is not int or query_budget <= 0:
        raise ValueError("query_budget must be positive")
    low, high = 0., 1.
    for index in range(query_budget):
        fraction = high if index == 0 else (low + high) / 2
        request = interpolate_request(approved, malicious, fraction)
        score = detector_score(detector, parameter_features(request, approved))
        accepted = score <= detector["threshold"]
        yield {"query": index + 1, "fraction": fraction, "parameters": request,
               "public_score": score, "public_decision": accepted}
        if accepted:
            low = fraction
        else:
            high = fraction


def repeated_query_curve(trials):
    """Mean-gradient and refusal-frequency scores at equal per-world budgets.

    Aggregate repeated observations within target/world first. Report sample
    counts at each budget; correlated batches are never independent targets.
    """
    from .metrics import scientific_metrics
    groups = {}
    for row in trials:
        target = row.get("target_sha256", row.get("target_index", "single-target"))
        groups.setdefault((target, row["truth_member"]), []).append(row)
    budgets = range(1, max((len(rows) for rows in groups.values()), default=0) + 1)
    results = []
    for budget in budgets:
        labels, scores, decision_labels, decisions = [], [], [], []
        paired_targets = {g[0] for g in groups if (g[0], not g[1]) in groups
                          and len(groups[g]) >= budget and len(groups[(g[0], not g[1])]) >= budget}
        for (target, truth), rows in groups.items():
            if target not in paired_targets:
                continue
            prefix = sorted(rows, key=lambda r: r.get("target_trial_id", r["trial_id"]))[:budget]
            observed = [r["score"] for r in prefix if r.get("score") is not None]
            if observed:
                labels.append(truth); scores.append(float(np.mean(observed)))
            decision_labels.append(truth)
            decisions.append(sum(r.get("decision") == "rejected" for r in prefix) / budget)
        results.append({"queries_per_world": budget, "independent_targets": len(paired_targets),
                        "gradient": scientific_metrics(labels, scores) if labels else None,
                        "refusal": scientific_metrics(decision_labels, decisions) if decision_labels else None})
    return results
