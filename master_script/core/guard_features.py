"""Public parameter-only features; no records, labels or experiment IDs enter here."""
import numpy as np

FEATURE_SCHEMA = "parameter_behavior_v1"
FEATURE_NAMES = ("relative_delta", "maximum_layer_delta", "parameter_concentration")


def parameter_features(parameters, reference):
    if not parameters:
        raise ValueError("Feature extraction needs matched nonempty parameter lists")
    deltas, concentrations = [], []
    for value, prior in zip(parameters, reference, strict=True):
        value, prior = np.asarray(value), np.asarray(prior)
        if value.shape != prior.shape or value.dtype.kind not in "fi" or prior.dtype.kind not in "fi":
            raise ValueError("Feature inputs must have equal numeric shapes")
        v, p = value.reshape(-1), prior.reshape(-1)
        scale = 1e-12
        # Bounded scratch memory even for an LLM embedding matrix.
        for start in range(0, v.size, 65536):
            a, b = v[start:start + 65536].astype(np.float64), p[start:start + 65536].astype(np.float64)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Feature inputs must be finite")
            scale = max(scale, float(np.max(np.abs(a), initial=0)), float(np.max(np.abs(b), initial=0)))
        delta2, prior2, value2, maximum = 0., 0., 0., 0.
        for start in range(0, v.size, 65536):
            a = v[start:start + 65536].astype(np.float64) / scale
            b = p[start:start + 65536].astype(np.float64) / scale
            delta2 += float(np.sum((a - b) ** 2))
            prior2 += float(np.sum(b * b)); value2 += float(np.sum(a * a))
            maximum = max(maximum, float(np.max(np.abs(a), initial=0)))
        deltas.append(float(min(np.sqrt(delta2) / max(np.sqrt(prior2), 1e-12), 1e6)))
        concentrations.append(float(maximum / max(np.sqrt(value2), 1e-12)))
    return dict(zip(FEATURE_NAMES, (float(np.mean(deltas)), max(deltas), max(concentrations))))


def feature_vector(features):
    if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
        raise ValueError("Detector features must match the strict public-feature allowlist")
    if any(type(features[k]) not in (int, float) for k in FEATURE_NAMES):
        raise ValueError("Feature values must be numeric")
    values = np.array([features[k] for k in FEATURE_NAMES], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    return values
