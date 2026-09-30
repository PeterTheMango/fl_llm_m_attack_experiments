"""Benign requests matched to a causal request on reference, code path and size.

For the pre-registered matched-reference controls study
(outputs/matched_reference_controls_20260930). Every control starts from the
same approved post-FL model W_R as the causal request and is checked by the
guard against that same pinned reference, without a ledger reservation. No
private gradient is computed or released for a control.

M1: the causal request's optimizer (normalized steps of size probe_lr) run for
    the same step count, descending the loss of a public batch.
M2: one honest FL round from W_R, rescaled to the causal request's L2 size.
M2raw: that honest round unscaled.
"""
from hashlib import sha256
import numpy as np


def normalized_steps(model, loss_fn, steps, lr, sign):
    """optimize_request's update rule for any loss; sign +1 ascends, -1 descends."""
    import torch
    history = []
    for _ in range(steps):
        loss = loss_fn(model)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        norm = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in gradients)).clamp(min=1e-12)
        with torch.no_grad():
            for parameter, gradient in zip(model.parameters(), gradients):
                parameter.add_(gradient / norm, alpha=sign * lr)
        history.append(float(loss.detach().cpu()))
    return history


def batch_loss(tokenizer, texts, config):
    """Mean next-token loss of a padded public batch, padding masked as in raw_gradients."""
    def loss(model):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=config.max_length,
                            return_tensors="pt")
        encoded = {k: v.to(next(model.parameters()).device) for k, v in encoded.items()}
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        return model(**encoded, labels=labels).loss
    return loss


def candidate_loss(tokenizer, candidate, config):
    """The candidate loss optimize_request ascends, tokenized identically."""
    def loss(model):
        tokens = tokenizer(candidate, truncation=True, max_length=config.max_length, return_tensors="pt")
        tokens = {k: v.to(next(model.parameters()).device) for k, v in tokens.items()}
        return model(**tokens, labels=tokens["input_ids"]).loss
    return loss


def public_records(config):
    """Public records for M1, after the adversary and any calibration slices.

    They are passed to the training-world token disjointness check with the
    other public records, so none overlaps the target or a client record.
    """
    count = config.matched_descent_requests * config.attack_batch_size
    if not count:
        return []
    from .. import datasets as dataset_sources
    offset = config.adversary_negative_count + (
        config.calibration_nonmember_count if config.threshold_mode == "calibrated" else 0)
    return dataset_sources.calibration_records(config, offset + count)[offset:]


def public_batches(config):
    records, size = public_records(config), config.attack_batch_size
    return [records[i:i + size] for i in range(0, len(records), size)]


def delta_norm(approved, request):
    """Model-wide L2 norm of request - approved over every state tensor, float64.

    State order is the guard's: a tied tensor counts once per state key, the
    same way for every arm, so norm matching between arms is unaffected.
    """
    total = 0.
    for prior, value in zip(approved, request, strict=True):
        a, b = np.asarray(value, dtype=np.float64).ravel(), np.asarray(prior, dtype=np.float64).ravel()
        for start in range(0, a.size, 65536):
            d = a[start:start + 65536] - b[start:start + 65536]
            total += float(np.dot(d, d))
    return float(np.sqrt(total))


def scaled_request(approved, updated, scale):
    """approved + scale * (updated - approved) in float64, cast back to each dtype."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    return [(np.asarray(p, dtype=np.float64) + scale * (np.asarray(u, dtype=np.float64) - np.asarray(p, dtype=np.float64))).astype(np.asarray(p).dtype)
            for p, u in zip(approved, updated, strict=True)]


def fedavg(results):
    """Flower's FedAvg weighting of (arrays, examples) pairs."""
    total = sum(n for _, n in results)
    if not results or total <= 0:
        raise ValueError("FedAvg needs examples")
    return [sum(arrays[i] * n for arrays, n in results) / total for i in range(len(results[0][0]))]


def honest_round(config, clients, approved):
    """One honest FL round R+1 from W_R with the unguarded training client."""
    from .amia import _ami_flower_client_cls
    from ..federation import selected_clients
    round_id = config.federated_rounds + 1
    client_type = _ami_flower_client_cls()
    results = []
    for partition in selected_clients(config, round_id):
        updated, examples, _ = client_type(partition, clients[partition], config).fit(
            [np.array(p, copy=True) for p in approved], {"server_round": round_id})
        results.append((updated, examples))
    return fedavg(results), round_id


def _text_digest(texts):
    return sha256("\x1e".join(texts).encode()).hexdigest()


def run_matched_controls(config, model_path, tokenizer, clients, approved, causal, target, guard_runtime):
    """Build and check M1, M2 and M2raw; return server-observable records only."""
    import torch
    from dataclasses import replace as dc_replace
    from ..model_io import load_causal_model
    from .amia import get_parameters, set_parameters, client_device
    if guard_runtime is None:
        raise ValueError("Matched controls require a pinned client guard")
    causal_norm = delta_norm(approved, causal)
    if causal_norm <= 0:
        raise ValueError("The causal request did not move the approved model")
    records = []

    def check(arm, index, request, **fields):
        scope = f"benign_control:{config.seed}:{arm}:{index}"
        # Same pinned reference file and digests as the observation guard.
        guard = dc_replace(guard_runtime, scope=scope)
        guard.check(request, "benign_control", 1, architecture="causal_lm", observation=True, reserve=False)
        norm = delta_norm(approved, request)
        records.append({"arm": arm, "index": index, "scope": scope, "delta_norm": norm,
                        "delta_norm_ratio_to_causal": norm / causal_norm, **fields})

    # The same device rule as the training clients (and the honest round below).
    control = load_causal_model(model_path).to(client_device(config)).eval()
    for index, texts in enumerate(public_batches(config)):
        set_parameters(control, approved)
        loss = batch_loss(tokenizer, texts, config)
        history = normalized_steps(control, loss, config.probe_epochs, config.probe_lr, -1)
        with torch.no_grad():
            after = float(loss(control).detach().cpu())
        check("M1", index, get_parameters(control), public_batch_sha256=_text_digest(texts),
              loss_before=history[0], loss_after=after, steps=len(history))
    set_parameters(control, causal)
    with torch.no_grad():
        causal_loss_after = float(candidate_loss(tokenizer, target, config)(control).detach().cpu())
    del control
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if config.matched_honest_round:
        updated, round_id = honest_round(config, clients, approved)
        honest_norm = delta_norm(approved, updated)
        if honest_norm <= 0:
            raise ValueError("The honest round did not move the approved model")
        scale = causal_norm / honest_norm
        check("M2", 0, scaled_request(approved, updated, scale), scale=scale, honest_round=round_id)
        check("M2raw", 0, [np.asarray(u).astype(np.asarray(p).dtype) for p, u in zip(approved, updated)],
              scale=1.0, honest_round=round_id)
    return {"schema": "matched_controls_v1", "causal_delta_norm": causal_norm,
            "causal_candidate_loss_after": causal_loss_after, "controls": records,
            "scope": "benign controls checked in shadow against the causal request's reference; no ledger debit; no private gradient released"}
