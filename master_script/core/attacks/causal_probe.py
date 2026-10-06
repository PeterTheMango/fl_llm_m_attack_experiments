"""Architecture-preserving gradient-alignment request, distinct from AMIA's head.

The server changes existing LM weights using only its public candidate text.
It observes the released full gradient and projects it onto a public candidate
reference gradient. No activations, client loss, batch labels, or private text
are supplied to the attacker. There is no inherited chosen-neuron guarantee.
"""
import numpy as np


def _loss_inputs(model, tokenizer, texts, config):
    """Plain records take the full LM loss; encoded grounded records their own mask."""
    from ..scoring import EncodedExample, encoded_batch
    device = next(model.parameters()).device
    if all(isinstance(t, EncodedExample) for t in texts):
        if any(len(t.input_ids) > config.max_length for t in texts):
            raise ValueError("Encoded example exceeds max_length; grounded records are never truncated")
        return encoded_batch(list(texts), tokenizer.pad_token_id, device)
    if any(isinstance(t, EncodedExample) for t in texts):
        raise ValueError("A gradient batch cannot mix encoded and plain-text records")
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=config.max_length,
                        return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return {**encoded, "labels": labels}


def raw_gradients(model, tokenizer, texts, config):
    from ..model_io import tensor_array
    return [tensor_array(g) for g in gradient_tensors(model, tokenizer, texts, config)]


def gradient_tensors(model, tokenizer, texts, config):
    """raw_gradients before the transfer to NumPy: torch tensors on the model's device."""
    import torch
    encoded = _loss_inputs(model, tokenizer, texts, config)
    model.eval()
    named = dict(model.named_parameters(remove_duplicate=False))
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(model(**encoded).loss, parameters)
    by_id = {id(p): g for p, g in zip(parameters, gradients)}
    # Preserve state_dict order, including tied weights and non-parameter buffers.
    return [by_id[id(named[k])] if k in named else torch.zeros_like(v) for k, v in model.state_dict().items()]


def protected_gradients(model, tokenizer, texts, config, noise_on_device=False):
    """noise_on_device (opt-in, grounded study): clip and draw the release noise
    with torch where the gradients are (defenses.protect_observation_on_device)."""
    if noise_on_device:
        from ..defenses import protect_observation_on_device
        return protect_observation_on_device(gradient_tensors(model, tokenizer, texts, config), config)
    from ..defenses import protect_observation
    return protect_observation(raw_gradients(model, tokenizer, texts, config), config)


def alignment_terms(released, public_direction):
    """Server-observable dot product and norms behind the alignment score."""
    if len(released) != len(public_direction) or not released:
        raise ValueError("Gradient alignment requires matching arrays")
    dot = norm = released_norm = 0.
    for value, direction in zip(released, public_direction):
        if value.shape != direction.shape or not np.isfinite(value).all() or not np.isfinite(direction).all():
            raise ValueError("Invalid alignment arrays")
        a, b = value.ravel(), direction.ravel()
        for start in range(0, a.size, 65536):
            x, y = a[start:start+65536].astype(float), b[start:start+65536].astype(float)
            dot += float(np.sum(x * y)); norm += float(np.sum(y * y)); released_norm += float(np.sum(x * x))
    return {"dot": dot, "direction_norm": float(np.sqrt(norm)), "released_norm": float(np.sqrt(released_norm))}


SCORES = ("projection", "cosine")


def score_from_terms(terms, kind="projection"):
    """Membership score from the server-observable alignment terms.

    projection: dot / ||direction||, which grows with the batch's overall
    gradient size. cosine: dot / (||released|| ||direction||), the score
    validated in outputs/causal_attack_validation_20260929 (Stage B, 12 epochs).
    """
    if kind == "projection":
        return terms["dot"] / max(terms["direction_norm"], 1e-12)
    if kind == "cosine":
        return terms["dot"] / max(terms["released_norm"] * terms["direction_norm"], 1e-24)
    raise ValueError("Unknown causal score")


def alignment_score(released, public_direction):
    return score_from_terms(alignment_terms(released, public_direction))


def optimize_request(model, tokenizer, candidate, config):
    """Bounded candidate-loss ascent; preserves every state key, shape and dtype."""
    import torch
    history = []
    model.eval()
    from ..scoring import EncodedExample, encoded_batch
    if isinstance(candidate, EncodedExample):
        # Grounded direction: ascend the candidate's answer-only loss.
        tokens = encoded_batch([candidate], tokenizer.pad_token_id, next(model.parameters()).device)
    else:
        tokens = tokenizer(candidate, truncation=True, max_length=config.max_length, return_tensors="pt")
        tokens = {k: v.to(next(model.parameters()).device) for k, v in tokens.items()}
        tokens["labels"] = tokens["input_ids"]
    for _ in range(config.probe_epochs):
        loss = model(**tokens).loss
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        norm = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in gradients)).clamp(min=1e-12)
        with torch.no_grad():
            for parameter, gradient in zip(model.parameters(), gradients):
                parameter.add_(gradient / norm, alpha=config.probe_lr)
        history.append(float(loss.detach().cpu()))
    return history
