"""Independent-group calibration and uncertainty for separate privacy questions."""
from hashlib import sha256
import json
import math
import numpy as np
from .metrics import base_metrics, scientific_metrics, roc_auc
from .calibration import nonmember_threshold


def group_id(row):
    value = row.get("target_sha256") or row.get("candidate_sha256") or row.get("group")
    if not isinstance(value, str) or not value:
        raise ValueError("Independent target/document identity is required")
    return value


def collapse_worlds(rows, channel="score", query_budget=None):
    """One mean per target and membership world; retries are not new samples."""
    groups = {}
    for row in rows:
        if type(row.get("truth_member")) is not bool:
            raise ValueError("Membership truth must be boolean")
        groups.setdefault((group_id(row), row["truth_member"]), []).append(row)
    result = []
    for (group, truth), values in groups.items():
        values = sorted(values, key=lambda r: r.get("target_trial_id", r.get("trial_id", 0)))
        if query_budget is not None:
            if len(values) < query_budget:
                continue
            values = values[:query_budget]
        if channel == "decision":
            scores = [float(v.get("decision") == "rejected") for v in values]
        else:
            scores = [v.get(channel) for v in values if v.get(channel) is not None]
        if any(not math.isfinite(float(s)) for s in scores):
            raise ValueError("Nonfinite attack observation")
        result.append({"group": group, "truth_member": truth,
                       "score": float(np.mean(scores)) if scores else None,
                       "queries": len(values), "available": len(scores)})
    return result


def fit_attack_calibration(rows, *, target_fpr=.05, channel="score", query_budget=1, binding=None):
    """Fit direction and threshold on defended validation groups only.

    Labels here are attacker-validation labels, never final evaluation labels.
    The empirical low-FPR count uses independent worlds, not repeated batches.
    """
    worlds = collapse_worlds(rows, channel, query_budget)
    observed = [r for r in worlds if r["score"] is not None]
    labels, scores = [r["truth_member"] for r in observed], [r["score"] for r in observed]
    if not labels or set(labels) != {False, True}:
        raise ValueError("Calibration needs accepted member and nonmember validation worlds")
    direction = 1 if roc_auc(labels, scores) >= .5 else -1
    calibration = nonmember_threshold([direction * r["score"] for r in observed if not r["truth_member"]], target_fpr)
    return {"schema": "attack_calibration_v1", "channel": channel, "query_budget": query_budget,
            "direction": direction, **calibration, "binding": binding,
            "groups": sorted({group_id(r) for r in rows}),
            "validation_sha256": sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "selection": "independent_defended_validation_direction_and_nonmember_threshold"}


def apply_attack_calibration(rows, calibration, *, binding=None):
    if calibration.get("schema") != "attack_calibration_v1" or calibration.get("direction") not in (-1, 1):
        raise ValueError("Invalid attack calibration artifact")
    if calibration.get("binding") != binding:
        raise ValueError("Calibration implementation/configuration binding mismatch")
    if set(calibration["groups"]) & {group_id(r) for r in rows}:
        raise ValueError("Final targets overlap attacker calibration")
    worlds = collapse_worlds(rows, calibration["channel"], calibration["query_budget"])
    for row in worlds:
        row["raw_score"] = row["score"]
        row["score"] = calibration["direction"] * row["score"] if row["score"] is not None else None
        row["pred_member"] = row["score"] >= calibration["threshold"] if row["score"] is not None else None
    return worlds


def privacy_report(rows, *, draws=1000, seed=0):
    """Paired target bootstrap. All-rejected numeric metrics remain unavailable."""
    observed = [r for r in rows if r.get("score") is not None]
    classified = [r for r in observed if type(r.get("pred_member")) is bool]
    out = {"independent_groups": len({group_id(r) for r in rows}), "attempted_worlds": len(rows),
           "available_worlds": len(observed), "unavailable_worlds": len(rows)-len(observed),
           "roc_auc": None, "balanced_accuracy": None, "interval_95": None,
           "per_target": {}}
    if observed:
        out.update(scientific_metrics([r["truth_member"] for r in observed], [r["score"] for r in observed]))
    if classified:
        out["balanced_accuracy"] = base_metrics(classified)["adv"]
        out["tpr"] = base_metrics(classified)["tpr"]
        out["fpr"] = 1-base_metrics(classified)["tnr"] if base_metrics(classified)["tnr"] is not None else None
    groups = sorted({group_id(r) for r in rows})
    for group in groups:
        subset = [r for r in observed if group_id(r) == group]
        out["per_target"][group] = {"available_worlds": len(subset),
            "roc_auc": roc_auc([r["truth_member"] for r in subset], [r["score"] for r in subset])
            if {r["truth_member"] for r in subset} == {False, True} else None}
    if len(groups) >= 2 and observed:
        rng = np.random.default_rng(seed)
        aucs, bas = [], []
        buckets = {g: [r for r in observed if group_id(r) == g] for g in groups}
        for _ in range(draws):
            sample = [r for g in rng.choice(groups, len(groups), replace=True) for r in buckets[g]]
            if {r["truth_member"] for r in sample} != {False, True}:
                continue
            aucs.append(roc_auc([r["truth_member"] for r in sample], [r["score"] for r in sample]))
            if all(type(r.get("pred_member")) is bool for r in sample):
                bas.append(base_metrics(sample)["adv"])
        out["interval_95"] = {"roc_auc": np.quantile(aucs, [.025,.975]).tolist() if aucs else None,
                              "balanced_accuracy": np.quantile(bas, [.025,.975]).tolist() if bas else None}
    out["uncertainty_note"] = "Target-cluster bootstrap; small-group intervals are unstable. No independence claim for repeated questions."
    return out


def calibration_binding(result, privacy_unit, condition=None):
    """All attack/training choices except the deliberately disjoint sample set."""
    config = dict(result.get("config", {}))
    for key in ("seed", "attack_trials", "attack_targets", "artifact_root", "local_artifact_dir", "keep_artifacts"):
        config.pop(key, None)
    pipeline = json.loads(json.dumps(result.get("pipeline", {})))
    pipeline.pop("condition", None)
    if pipeline.get("rag"):
        pipeline["rag"].pop("study_file", None)
        pipeline["rag"].pop("evaluation_trials", None)
    if pipeline.get("client_guard"):
        for key in ("policy_file", "detector_file"):
            pipeline["client_guard"].pop(key, None)
    return {"implementation_fingerprint": result.get("implementation_fingerprint"),
            "method_version": result.get("method_version"), "attack": result.get("attack_name"),
            "config": config, "pipeline": pipeline, "privacy_unit": privacy_unit, "rag_condition": condition}


def transcript_worlds(rows, query_budget):
    """Observable score/availability, decision and processing-time transcripts.

    Audit reasons, policy features, client losses and private labels are excluded
    from the feature vector. This is one bounded attacker, not a leakage bound.
    """
    channels = {c: {(r["group"], r["truth_member"]): r for r in collapse_worlds(rows, c, query_budget)}
                for c in ("score", "decision", "response_seconds")}
    result = []
    for key, row in channels["score"].items():
        timing = channels["response_seconds"][key]["score"]
        result.append({**row, "features": [row["score"] or 0., row["available"]/row["queries"],
                                          channels["decision"][key]["score"],
                                          math.log1p(max(0., timing)) if timing is not None else 0.,
                                          float(timing is not None)]})
    return result


def fit_transcript_calibration(rows, *, target_fpr=.05, query_budget=1, binding=None):
    worlds = transcript_worlds(rows, query_budget)
    groups = sorted({r["group"] for r in worlds}, key=lambda g: sha256(g.encode()).hexdigest())
    fit_groups = set(groups[::2])
    train = [r for r in worlds if r["group"] in fit_groups]
    validation = [r for r in worlds if r["group"] not in fit_groups]
    if any({r["truth_member"] for r in part} != {False, True} for part in (train, validation)):
        raise ValueError("Joint transcript attacker needs disjoint fit/threshold groups with both worlds")
    x = np.array([r["features"] for r in train]); y = np.array([r["truth_member"] for r in train])
    mean, scale = x.mean(axis=0), x.std(axis=0); scale[scale<1e-12] = 1.
    x = (x-mean)/scale; weights = np.zeros(x.shape[1]); intercept = 0.
    for _ in range(1000):
        residual = 1/(1+np.exp(-np.clip(x@weights+intercept,-50,50)))-y
        weights -= .1*(x.T@residual/len(y)+.01*weights); intercept -= .1*residual.mean()
    scores = [float(((np.array(r["features"])-mean)/scale)@weights+intercept)
              for r in validation if not r["truth_member"]]
    cutoff = nonmember_threshold(scores, target_fpr)
    return {"schema": "transcript_calibration_v1", "channel": "transcript", "query_budget": query_budget,
            "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(), "intercept": float(intercept),
            "groups": groups, "fit_groups": sorted(fit_groups), "binding": binding, **cutoff,
            "scope": "joint observed statistic/decision/processing-time attacker; excludes transport timing; not an upper bound"}


def apply_transcript_calibration(rows, calibration, *, binding=None):
    if calibration.get("schema") != "transcript_calibration_v1" or binding != calibration.get("binding"):
        raise ValueError("Transcript artifact/binding mismatch")
    if set(calibration["groups"]) & {group_id(r) for r in rows}:
        raise ValueError("Final targets overlap transcript calibration")
    result = transcript_worlds(rows, calibration["query_budget"])
    for row in result:
        x = (np.asarray(row.pop("features"))-calibration["mean"])/calibration["scale"]
        row["score"] = float(x@calibration["weights"] + calibration["intercept"])
        row["pred_member"] = row["score"] >= calibration["threshold"]
    return result
