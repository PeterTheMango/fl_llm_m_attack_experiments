"""Small logistic detector with safe JSON artifacts and grouped fitting.

Normalization uses training only; threshold uses validation benign requests.
Final-test groups are never used for fitting or threshold selection.
"""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from .guard_features import FEATURE_SCHEMA, feature_vector, feature_names


def validate_detector(data):
    keys = {"schema", "features", "mean", "scale", "weights", "intercept", "threshold", "split_groups"}
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError("Invalid detector artifact schema")
    names = feature_names(data["schema"])
    if data["features"] != list(names):
        raise ValueError("Incompatible detector feature schema")
    for key in ("mean", "scale", "weights"):
        values = data[key]
        if not isinstance(values, list) or len(values) != len(names) or any(type(v) not in (int, float) for v in values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid detector {key}")
    if any(v <= 0 for v in data["scale"]):
        raise ValueError("Detector scales must be positive")
    for key in ("intercept", "threshold"):
        if type(data[key]) not in (int, float) or not np.isfinite(data[key]):
            raise ValueError(f"Invalid detector {key}")
    if not 0 <= data["threshold"] <= 1:
        raise ValueError("Detector threshold must lie in [0, 1]")
    groups = data["split_groups"]
    if not isinstance(groups, dict) or set(groups) != {"train", "validation", "test"}:
        raise ValueError("Missing detector split provenance")
    seen = set()
    for values in groups.values():
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values) or seen.intersection(values):
            raise ValueError("Detector groups must be nonempty and disjoint")
        seen.update(values)
    return data


def load_detector(path, expected_sha256):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Detector SHA256 mismatch")
    return validate_detector(json.loads(raw))


def detector_score(data, features):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            x = (feature_vector(features, data["schema"]) - np.asarray(data["mean"])) / np.asarray(data["scale"])
            z = float(np.dot(x, data["weights"]) + data["intercept"])
        except FloatingPointError as exc:
            raise ValueError("Nonfinite detector computation") from exc
    if not np.isfinite(z):
        raise ValueError("Nonfinite detector computation")
    return float(1 / (1 + np.exp(-np.clip(z, -700, 700))))


def fit_detector(rows, *, max_false_positive_rate=0.01, iterations=1000, schema=FEATURE_SCHEMA):
    names = feature_names(schema)
    if not rows or not 0 <= max_false_positive_rate < 1 or type(iterations) is not int or iterations <= 0:
        raise ValueError("Invalid detector fitting inputs")
    groups = {k: set() for k in ("train", "validation", "test")}
    by_split = {k: [] for k in groups}
    for row in rows:
        if set(row) != {"group", "split", "malicious", "features", "variant"} or row["split"] not in groups:
            raise ValueError("Trace rows require group, split, malicious, features and variant")
        if type(row["malicious"]) is not bool or not isinstance(row["group"], str) or not row["group"] or not isinstance(row["variant"], str) or not row["variant"]:
            raise ValueError("Invalid trace labels or group")
        feature_vector(row["features"], schema)
        groups[row["split"]].add(row["group"])
        by_split[row["split"]].append(row)
    if any(not g for g in groups.values()) or any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Complete target/run groups must be disjoint across splits")
    if any({r["malicious"] for r in by_split[k]} != {False, True} for k in by_split):
        raise ValueError("Each split needs benign and malicious requests")
    trained_variants = {r["variant"] for k in ("train", "validation") for r in by_split[k] if r["malicious"]}
    test_variants = {r["variant"] for r in by_split["test"] if r["malicious"]}
    if not test_variants - trained_variants:
        raise ValueError("Reserve at least one unseen attack variant for final testing")
    train = by_split["train"]
    x = np.array([feature_vector(r["features"], schema) for r in train])
    y = np.array([r["malicious"] for r in train], dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    x = (x - mean) / scale
    weights, intercept = np.zeros(x.shape[1]), 0.
    # Fixed full-batch gradient descent with small L2 penalty; no test tuning.
    for _ in range(iterations):
        residual = 1 / (1 + np.exp(-np.clip(x @ weights + intercept, -50, 50))) - y
        weights -= .1 * (x.T @ residual / len(y) + .01 * weights)
        intercept -= .1 * residual.mean()
    data = {"schema": schema, "features": list(names), "mean": mean.tolist(),
            "scale": scale.tolist(), "weights": weights.tolist(), "intercept": float(intercept),
            "threshold": 1., "split_groups": {k: sorted(v) for k, v in groups.items()}}
    benign = sorted(detector_score(data, r["features"]) for r in by_split["validation"] if not r["malicious"])
    # Reject score > threshold, including ties conservatively in the benign set.
    allowed = int(max_false_positive_rate * len(benign))
    data["threshold"] = benign[len(benign) - allowed - 1]
    return validate_detector(data)
