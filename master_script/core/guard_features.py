"""Public parameter-only features; no records, labels or experiment IDs enter here."""
import numpy as np

FEATURE_SCHEMA = "parameter_behavior_v1"
FEATURE_NAMES = ("relative_delta", "maximum_layer_delta", "parameter_concentration")


def _scaled_layer_features(value, prior):
    """Bounded two-pass reference calculation, also safe for float64 extremes."""
    v, p = value.reshape(-1), prior.reshape(-1)
    scale = 1e-12
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
    return (float(min(np.sqrt(delta2) / max(np.sqrt(prior2), 1e-12), 1e6)),
            float(maximum / max(np.sqrt(value2), 1e-12)))


def _small_float_layer_features(value, prior):
    """One pass for <=32-bit floats, with double-precision accumulation.

    Even squared float32 extrema times the largest addressable array size fit
    in float64. Their smallest subnormals also survive squaring in float64.
    Preserve the scaled calculation's denominator floor algebraically; do not
    subtract squared norms (which loses small updates to cancellation).
    """
    v, p = value.reshape(-1), prior.reshape(-1)
    delta2, prior2, value2, maximum, scale = 0., 0., 0., 0., 1e-12
    for start in range(0, v.size, 65536):
        a = v[start:start + 65536].astype(np.float64)
        b = p[start:start + 65536].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Feature inputs must be finite")
        maximum = max(maximum, float(np.max(np.abs(a), initial=0)))
        scale = max(scale, maximum, float(np.max(np.abs(b), initial=0)))
        delta = a - b
        # einsum's direct reduction avoids both BLAS thread startup and squared
        # array temporaries. Scratch memory remains bounded by the chunk size.
        delta2 += float(np.einsum("i,i->", delta, delta))
        prior2 += float(np.einsum("i,i->", b, b))
        value2 += float(np.einsum("i,i->", a, a))
    return (float(min(np.sqrt(delta2) / max(np.sqrt(prior2), scale * 1e-12), 1e6)),
            float(maximum / max(np.sqrt(value2), scale * 1e-12)))


def parameter_features(parameters, reference):
    if not parameters:
        raise ValueError("Feature extraction needs matched nonempty parameter lists")
    deltas, concentrations = [], []
    for value, prior in zip(parameters, reference, strict=True):
        value, prior = np.asarray(value), np.asarray(prior)
        if value.shape != prior.shape or value.dtype.kind not in "fi" or prior.dtype.kind not in "fi":
            raise ValueError("Feature inputs must have equal numeric shapes")
        calculate = (_small_float_layer_features if all(a.dtype.kind == "f" and a.dtype.itemsize <= 4
                                                        for a in (value, prior)) else _scaled_layer_features)
        delta, concentration = calculate(value, prior)
        deltas.append(delta)
        concentrations.append(concentration)
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
