"""Nguyen Appendix C feature randomizers and finite-set confidence diagnostics.

Fixed sign/magnitude encoding: one sign, five integer and four fractional bits.
The public clipping domain is [-31.9375, 31.9375]. Every record/bit gets a
separate draw. This protects the feature input, conditional on task labels;
it does not establish privacy of an entire text record or of FL training.
"""
import math
import secrets

FEATURE_BOUND = 31.9375
BITS = 10
FRACTION_BITS = 4


def perturb_features(features, config, seed=None):
    import torch

    if config.ldp_mechanism == "none":
        return features.detach()
    if config.ldp_mechanism not in ("BitRand", "OME"):
        raise ValueError("Unknown AMIA input mechanism")
    if not math.isfinite(config.epsilon) or config.epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("Input mechanism needs a finite record-by-feature matrix")
    values = features.detach().clamp(-FEATURE_BOUND, FEATURE_BOUND)
    magnitude = torch.round(values.abs() * (2 ** FRACTION_BITS)).to(torch.int64)
    shifts = torch.arange(BITS - 2, -1, -1, device=values.device)
    encoded = torch.cat([(values < 0).unsqueeze(-1),
                         ((magnitude.unsqueeze(-1) >> shifts) & 1).bool()], dim=-1)
    positions = torch.arange(BITS, device=values.device, dtype=torch.float64)
    dimension = values.shape[1]
    if config.ldp_mechanism == "BitRand":
        # The upstream alpha bound, evaluated in log space to avoid overflow.
        exponents = 2 * config.epsilon * positions / BITS
        log_alpha = 0.5 * (math.log(config.epsilon + dimension * BITS)
                           - math.log(2 * dimension) - float(torch.logsumexp(exponents, 0)))
        keep = torch.sigmoid(-(log_alpha + positions * config.epsilon / BITS))
        probability_one = torch.where(encoded, keep, 1 - keep)
    else:
        alpha = 100.0
        one = torch.where(positions.remainder(2) == 0,
                          alpha / (1 + alpha), 1 / (1 + alpha ** 3))
        zero = torch.sigmoid(torch.tensor(-math.log(alpha) - config.epsilon / (dimension * BITS),
                                          dtype=torch.float64, device=values.device))
        probability_one = torch.where(encoded, one, zero)
    generator = torch.Generator(device=values.device)
    generator.manual_seed(secrets.randbits(63) if seed is None else seed)
    draws = torch.rand(encoded.shape, dtype=torch.float64, device=values.device, generator=generator)
    randomized = draws < probability_one
    decoded = (randomized[..., 1:].to(torch.int64) << shifts).sum(-1).to(values.dtype) / (2 ** FRACTION_BITS)
    return torch.where(randomized[..., 0], -decoded, decoded)


def finite_set_bounds(probe, target, negatives, config):
    """Eqs. (9)-(11) on an explicitly finite public audit set, not all x.

    Analytic network bounds replace the invalid practice of using the observed
    sample range in Hoeffding's inequality. A union bound covers these records.
    No finite negative sample certifies the universal negative condition.
    """
    import torch

    if config.ldp_mechanism == "none":
        return {"scope": "not_applicable_without_input_randomization", "certified_attack": False}
    n = config.certificate_samples
    delta = config.certificate_delta
    if type(n) is not int or n < 2 or not 0 < delta < 1:
        raise ValueError("Confidence diagnostic requires samples>=2 and delta in (0,1)")
    with torch.no_grad():
        width = probe.fc1.weight.abs().sum(1) * FEATURE_BOUND
        bias = probe.fc1.bias if probe.fc1.bias is not None else torch.zeros_like(width)
        lower = torch.relu(bias - width)
        upper = torch.relu(bias + width)
        h = probe.fc2.weight[0]
        output_range = float((h.abs() * (upper - lower)).sum())
        records = torch.cat([target, negatives], dim=0)
        radius = output_range * math.sqrt(math.log(len(records) / delta) / (2 * n))
        means = []
        for index, record in enumerate(records):
            samples = perturb_features(record.unsqueeze(0).repeat(n, 1), config,
                                       seed=config.seed + 1000003 + index)
            means.append(float(probe(samples).mean()))
    target_lower = means[0] - radius
    negative_upper = max(value + radius for value in means[1:])
    return {"scope": "finite_adversary_audit_set_only", "certified_attack": False,
            "finite_set_condition": target_lower > 0 and negative_upper <= 0,
            "target_lower": target_lower, "maximum_negative_upper": negative_upper,
            "analytic_output_range": output_range, "samples_per_record": n,
            "record_count": len(means), "joint_failure_probability": delta,
            "limitation": "Unseen negatives and downstream-gradient cancellation are not certified."}
