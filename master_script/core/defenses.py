"""Private client optimization and private FedAvg (Abadi 2016; McMahan 2018).

Accounting uses Bun–Steinke zCDP composition with replacement adjacency and
no sampling amplification. Noise is freshly seeded independently of public
experiment seeds. Bounds concern training releases, not private audit data.
"""
import math
import secrets


def privacy_bound(steps, noise_multiplier, delta):
    # Clipped-vector replacement sensitivity is 2C, noise std is sigma*C.
    rho = 2.0 * steps / (noise_multiplier ** 2)
    return {"rho": rho, "epsilon": rho + 2 * math.sqrt(rho * math.log(1 / delta)),
            "delta": delta, "steps": steps,
            "accountant": "gaussian_zcdp_no_amplification",
            "adjacency": "fixed_cardinality_replace_one"}


def private_train(model, tokenizer, texts, config, defense):
    """Clip each sequence gradient before adding noise and applying AdamW.

    Sequential microbatches avoid materializing a batch of full LLM gradients.
    This is private Adam optimization, as in Li et al. (2022); the configuration
    uses the conventional dp_sgd name for the private-gradient mechanism.
    """
    import torch

    device = next(model.parameters()).device
    parameters = [p for p in model.parameters() if p.requires_grad]
    noise = torch.Generator(device=device)
    noise.manual_seed(secrets.randbits(63))
    optimizer = torch.optim.AdamW(parameters, lr=config.client_lr)
    steps = 0
    model.train()
    for _ in range(config.local_epochs):
        # The permutation is independent of record contents; no amplification
        # from this shuffle is used in the accountant.
        order = torch.randperm(len(texts)).tolist()
        for start in range(0, len(order), config.local_batch_size):
            indices = order[start:start + config.local_batch_size]
            accumulated = [torch.zeros_like(p) for p in parameters]
            for i in indices:
                encoded = tokenizer(texts[i], truncation=True, max_length=config.max_length,
                                    return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                labels = encoded["input_ids"].clone()
                if labels.shape[-1] < 2:
                    raise ValueError("DP training needs at least two tokens per record")
                labels[encoded["attention_mask"] == 0] = -100
                loss = model(**encoded, labels=labels).loss
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                norm = torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(g.detach().float()) for g in gradients if g is not None
                ]))
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite private gradient")
                scale = min(1.0, defense.clip_norm / max(float(norm), 1e-12))
                for total, gradient in zip(accumulated, gradients):
                    if gradient is not None:
                        total.add_(gradient.detach(), alpha=scale)
            optimizer.zero_grad(set_to_none=True)
            for parameter, total in zip(parameters, accumulated):
                perturbation = torch.randn(total.shape, generator=noise, device=device, dtype=total.dtype)
                parameter.grad = (total + perturbation * (defense.noise_multiplier * defense.clip_norm)) / len(indices)
            optimizer.step()
            steps += 1
    optimizer.zero_grad(set_to_none=True)
    return steps


def clipped_mean(updates, previous, clip_norm, noise_multiplier, rng=None):
    """Uniform clipped client deltas plus one Gaussian draw on the mean."""
    import numpy as np

    rng = rng if rng is not None else np.random.default_rng(secrets.randbits(128))
    if not updates:
        raise ValueError("A private aggregate requires participating clients")
    floating = [np.issubdtype(p.dtype, np.floating) for p in previous]
    totals = [np.zeros_like(p, dtype=np.float64) if f else p.copy()
              for p, f in zip(previous, floating)]
    for update in updates:
        if len(update) != len(previous) or any(u.shape != p.shape for u, p in zip(update, previous)):
            raise ValueError("Client update shape mismatch")
        deltas = [u.astype(np.float64) - p if f else None for u, p, f in zip(update, previous, floating)]
        norm = math.sqrt(sum(float(np.sum(d * d)) for d in deltas if d is not None))
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite client update")
        scale = min(1.0, clip_norm / max(norm, 1e-12))
        for total, delta in zip(totals, deltas):
            if delta is not None:
                total += delta * scale
    count = len(updates)
    return [(p + total / count + rng.normal(0, noise_multiplier * clip_norm / count, p.shape)).astype(p.dtype)
            if f else p.copy() for p, total, f in zip(previous, totals, floating)]


def strategy_class(base, defense, initial_arrays, privacy):
    """Wrap the existing capture strategy; preserve its final-model handoff."""
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    class PrivateStrategy(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.previous = [a.copy() for a in initial_arrays]
            self.client_steps = {}
            self.rounds_released = 0

        def aggregate_fit(self, server_round, results, failures):
            if failures or not results:
                raise RuntimeError("Private FL round incomplete; refusing a partial aggregate")
            for _, result in results:
                cid = int(result.metrics["partition_id"])
                if defense.mechanism == "dp_sgd":
                    self.client_steps[cid] = self.client_steps.get(cid, 0) + int(result.metrics["dp_steps"])
                # Unnoised losses/norms are not part of the protected release.
                result.metrics = {"partition_id": cid}
            if defense.mechanism == "dp_fedavg":
                arrays = [parameters_to_ndarrays(r.parameters) for _, r in results]
                averaged = clipped_mean(arrays, self.previous, defense.clip_norm, defense.noise_multiplier)
                for _, result in results:
                    result.parameters = ndarrays_to_parameters(averaged)
                    result.num_examples = 1
            parameters, metrics = super().aggregate_fit(server_round, results, failures)
            if parameters is None:
                raise RuntimeError("Private FL strategy produced no global model")
            self.previous = parameters_to_ndarrays(parameters)
            self.rounds_released += 1
            steps = max(self.client_steps.values(), default=0) if defense.mechanism == "dp_sgd" else self.rounds_released
            privacy.update(privacy_bound(steps, defense.noise_multiplier, defense.delta))
            privacy.update(mechanism=defense.mechanism,
                           privacy_unit="training_sequence" if defense.mechanism == "dp_sgd" else "client_partition",
                           trusted_server=defense.mechanism == "dp_fedavg",
                           protected_release="trained_model_and_training_updates" if defense.mechanism == "dp_sgd" else "noised_global_models",
                           client_steps={str(k): v for k, v in self.client_steps.items()})
            return parameters, metrics

    return PrivateStrategy
