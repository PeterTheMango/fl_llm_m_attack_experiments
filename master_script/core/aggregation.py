"""Bounded-memory aggregation of Flower's serialized parameter tensors."""
import math

import numpy as np

_CHUNK = 1_048_576


def private_fedavg(results, previous, clip_norm, noise_multiplier, rng):
    """Uniform mean of globally clipped deltas, with one noised release.

    Two passes avoid retaining every decoded client model or float64 delta.
    Norm/delta scratch space is bounded even for a large embedding tensor.
    Inputs remain unchanged; integer buffers retain their previous values.
    """
    from flwr.common import Parameters, bytes_to_ndarray, ndarray_to_bytes

    if not results:
        raise ValueError("A private aggregate requires participating clients")
    if any(len(r.parameters.tensors) != len(previous.tensors) for _, r in results):
        raise ValueError("Client update shape mismatch")
    norms_squared = [0.] * len(results)
    for j, raw in enumerate(previous.tensors):
        p = bytes_to_ndarray(raw)
        for i, (_, result) in enumerate(results):
            u = bytes_to_ndarray(result.parameters.tensors[j])
            if u.shape != p.shape or u.dtype != p.dtype:
                raise ValueError("Client update shape/dtype mismatch")
            if np.issubdtype(p.dtype, np.floating):
                pf, uf = p.reshape(-1), u.reshape(-1)
                for start in range(0, pf.size, _CHUNK):
                    d = uf[start:start + _CHUNK].astype(np.float64) - pf[start:start + _CHUNK]
                    norms_squared[i] += float(np.sum(d * d))
            del u
    if not all(math.isfinite(n) for n in norms_squared):
        raise FloatingPointError("Non-finite client update")
    scales = [min(1., clip_norm / max(math.sqrt(n), 1e-12)) for n in norms_squared]
    count = len(results)
    output = []
    for j, raw in enumerate(previous.tensors):
        p = bytes_to_ndarray(raw)
        if not np.issubdtype(p.dtype, np.floating):
            output.append(raw)
            continue
        total = np.zeros(p.shape, dtype=np.float64)
        pf, tf = p.reshape(-1), total.reshape(-1)
        for scale, (_, result) in zip(scales, results):
            u = bytes_to_ndarray(result.parameters.tensors[j])
            uf = u.reshape(-1)
            for start in range(0, pf.size, _CHUNK):
                stop = start + _CHUNK
                d = uf[start:stop].astype(np.float64) - pf[start:stop]
                tf[start:stop] += d * scale
            del u, uf
        out = np.empty_like(p)
        of = out.reshape(-1)
        for start in range(0, pf.size, _CHUNK):
            stop = start + _CHUNK
            value = pf[start:stop] + tf[start:stop] / count
            value += rng.normal(0, noise_multiplier * clip_norm / count, value.shape)
            of[start:stop] = value
        output.append(ndarray_to_bytes(out))
    return Parameters(tensors=output, tensor_type=previous.tensor_type)
