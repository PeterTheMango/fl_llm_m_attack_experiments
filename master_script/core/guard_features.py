"""Public parameter-only features; no records, labels or experiment IDs enter here."""
import numpy as np

FEATURE_SCHEMA = "parameter_behavior_v1"
FEATURE_NAMES = ("relative_delta", "maximum_layer_delta", "parameter_concentration")
STRUCTURE_SCHEMA = "parameter_structure_v2"
STRUCTURE_NAMES = FEATURE_NAMES + ("median_layer_delta", "p90_layer_delta",
    "mean_update_reference_cosine", "minimum_update_reference_cosine", "maximum_update_reference_cosine",
    "mean_update_concentration", "maximum_update_concentration")
FEATURE_SCHEMAS = {FEATURE_SCHEMA: FEATURE_NAMES, STRUCTURE_SCHEMA: STRUCTURE_NAMES}
CHUNK_ELEMENTS = 65536


def feature_names(schema):
    if not isinstance(schema, str) or schema not in FEATURE_SCHEMAS:
        raise ValueError("Incompatible detector feature schema")
    return FEATURE_SCHEMAS[schema]


def _update_geometry(value, prior):
    """Cosine(delta, reference) and L-inf(delta)/L2(delta), bounded scratch.

    Normalize before subtraction, including float64 extremes. A second scaling
    of delta prevents squaring tiny representable relative updates to zero.
    Undefined zero-vector cosines/concentrations are defined as zero. flat
    slices bound copies even for noncontiguous tensors. No full delta is kept.
    """
    scale = 0.
    for start in range(0, value.size, CHUNK_ELEMENTS):
        a = value.flat[start:start + CHUNK_ELEMENTS].astype(np.float64)
        b = prior.flat[start:start + CHUNK_ELEMENTS].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Feature inputs must be finite")
        scale = max(scale, float(np.max(np.abs(a), initial=0)), float(np.max(np.abs(b), initial=0)))
    if scale == 0:
        return 0., 0.
    delta_scale = prior_scale = 0.
    for start in range(0, value.size, CHUNK_ELEMENTS):
        a = value.flat[start:start + CHUNK_ELEMENTS].astype(np.float64) / scale
        b = prior.flat[start:start + CHUNK_ELEMENTS].astype(np.float64) / scale
        delta_scale = max(delta_scale, float(np.max(np.abs(a - b), initial=0)))
        prior_scale = max(prior_scale, float(np.max(np.abs(b), initial=0)))
    if delta_scale == 0:
        return 0., 0.
    dot = delta2 = prior2 = 0.
    for start in range(0, value.size, CHUNK_ELEMENTS):
        a = value.flat[start:start + CHUNK_ELEMENTS].astype(np.float64) / scale
        b = prior.flat[start:start + CHUNK_ELEMENTS].astype(np.float64) / scale
        delta = (a - b) / delta_scale
        b = b / prior_scale if prior_scale else b
        dot += float(np.einsum("i,i->", delta, b))
        delta2 += float(np.einsum("i,i->", delta, delta))
        prior2 += float(np.einsum("i,i->", b, b))
    cosine = dot / np.sqrt(delta2) / np.sqrt(prior2) if prior2 else 0.
    return float(np.clip(cosine, -1, 1)), float(1 / np.sqrt(delta2))


def reduce_layer_features(layers, schema=FEATURE_SCHEMA):
    feature_names(schema)
    if not layers:
        raise ValueError("Feature extraction needs nonempty tensors")
    deltas = [f["relative_delta"] for f in layers]
    result = dict(zip(FEATURE_NAMES, (float(np.mean(deltas)), max(deltas),
        max(f["parameter_concentration"] for f in layers))))
    if schema == STRUCTURE_SCHEMA:
        cosines = [f["mean_update_reference_cosine"] for f in layers]
        concentrations = [f["mean_update_concentration"] for f in layers]
        result.update(zip(STRUCTURE_NAMES[3:], (float(np.median(deltas)), float(np.quantile(deltas, .9)),
            float(np.mean(cosines)), min(cosines), max(cosines), float(np.mean(concentrations)), max(concentrations))))
    return result


def _scaled_layer_features(value, prior):
    """Bounded two-pass reference calculation, also safe for float64 extremes."""
    v, p = value, prior
    scale = 1e-12
    for start in range(0, v.size, 65536):
        a, b = v.flat[start:start + 65536].astype(np.float64), p.flat[start:start + 65536].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Feature inputs must be finite")
        scale = max(scale, float(np.max(np.abs(a), initial=0)), float(np.max(np.abs(b), initial=0)))
    delta2, prior2, value2, maximum = 0., 0., 0., 0.
    for start in range(0, v.size, 65536):
        a = v.flat[start:start + 65536].astype(np.float64) / scale
        b = p.flat[start:start + 65536].astype(np.float64) / scale
        delta2 += float(np.sum((a - b) ** 2))
        prior2 += float(np.sum(b * b)); value2 += float(np.sum(a * a))
        maximum = max(maximum, float(np.max(np.abs(a), initial=0)))
    return (float(min(np.sqrt(delta2) / max(np.sqrt(prior2), 1e-12), 1e6)),
            float(maximum / max(np.sqrt(value2), 1e-12)))


def _small_float_layer_features(value, prior, geometry=False):
    """One pass for <=32-bit floats, with double-precision accumulation.

    Even squared float32 extrema times the largest addressable array size fit
    in float64. Their smallest subnormals also survive squaring in float64.
    Preserve the scaled calculation's denominator floor algebraically; do not
    subtract squared norms (which loses small updates to cancellation).
    """
    v, p = value, prior
    delta2, prior2, value2, maximum, scale = 0., 0., 0., 0., 1e-12
    dot, delta_maximum = 0., 0.
    for start in range(0, v.size, 65536):
        a = v.flat[start:start + 65536].astype(np.float64)
        b = p.flat[start:start + 65536].astype(np.float64)
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
        if geometry:
            dot += float(np.einsum("i,i->", delta, b))
            delta_maximum = max(delta_maximum, float(np.max(np.abs(delta), initial=0)))
    result = (float(min(np.sqrt(delta2) / max(np.sqrt(prior2), scale * 1e-12), 1e6)),
              float(maximum / max(np.sqrt(value2), scale * 1e-12)))
    if geometry:
        cosine = dot / np.sqrt(delta2) / np.sqrt(prior2) if delta2 and prior2 else 0.
        concentration = delta_maximum / np.sqrt(delta2) if delta2 else 0.
        return result + (float(np.clip(cosine, -1, 1)), float(concentration))
    return result


def parameter_features(parameters, reference, schema=FEATURE_SCHEMA):
    feature_names(schema)
    if not parameters:
        raise ValueError("Feature extraction needs matched nonempty parameter lists")
    layers = []
    for value, prior in zip(parameters, reference, strict=True):
        value, prior = np.asarray(value), np.asarray(prior)
        if value.shape != prior.shape or value.dtype.kind not in "fi" or prior.dtype.kind not in "fi":
            raise ValueError("Feature inputs must have equal numeric shapes")
        calculate = (_small_float_layer_features if all(a.dtype.kind == "f" and a.dtype.itemsize <= 4
                                                        for a in (value, prior)) else _scaled_layer_features)
        if schema == STRUCTURE_SCHEMA and calculate is _small_float_layer_features:
            delta, concentration, cosine, update_concentration = calculate(value, prior, geometry=True)
        else:
            delta, concentration = calculate(value, prior)
            if schema == STRUCTURE_SCHEMA:
                cosine, update_concentration = _update_geometry(value, prior)
        layer = dict(zip(FEATURE_NAMES, (delta, delta, concentration)))
        if schema == STRUCTURE_SCHEMA:
            layer.update(mean_update_reference_cosine=cosine, mean_update_concentration=update_concentration)
        layers.append(layer)
    return reduce_layer_features(layers, schema)


def feature_vector(features, schema=FEATURE_SCHEMA):
    names = feature_names(schema)
    if not isinstance(features, dict) or set(features) != set(names):
        raise ValueError("Detector features must match the strict public-feature allowlist")
    if any(type(features[k]) not in (int, float) for k in names):
        raise ValueError("Feature values must be numeric")
    values = np.array([features[k] for k in names], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    return values
