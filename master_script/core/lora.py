"""LoRA adapters and in-process FedAvg for the generalization study (opt-in; proposal §5.2, §5.4).

The base model is frozen (bf16 on the GPU); every attention and MLP projection
(all Linear/Conv1D modules except the LM head) gets a rank-r adapter in fp32
(Hu et al. 2022): W x + (alpha / r) B A x, with B initialised to zero so the
adapted model starts as the pretrained one. Clients exchange adapter tensors
only, and FedAvg runs in the job's process with Flower's own aggregation
arithmetic (FedAvg.aggregate_fit), the same client schedule
(federation.selected_clients) and the same per-client seeding as the Flower
client (seed + 1009 * partition + round). The reference model for the weight
attacks is the same base with the adapters switched off.

torch, transformers and flwr are imported function-locally, as everywhere in core.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import time

from .scoring import validate_partition_tokens

EXCLUDED_MODULES = ("lm_head",)


def _lora_module_class():
    import torch

    class LoraWrapped(torch.nn.Module):
        """A frozen Linear (out x in weight) or GPT-2 Conv1D (in x out weight) plus a LoRA adapter."""

        def __init__(self, base, rank, alpha, dropout):
            super().__init__()
            self.base = base
            conv1d = type(base).__name__ == "Conv1D"
            in_features, out_features = (base.weight.shape[0], base.weight.shape[1]) if conv1d else (
                base.in_features, base.out_features)
            self.lora_A = torch.nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32, device=base.weight.device))
            self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=base.weight.device))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.scaling = alpha / rank
            self.dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()
            self.enabled = True

        def forward(self, x):
            out = self.base(x)
            if not self.enabled:
                return out
            update = (self.dropout(x.to(self.lora_A.dtype)) @ self.lora_A.t()) @ self.lora_B.t()
            return out + (update * self.scaling).to(out.dtype)

    return LoraWrapped


def target_module_names(model):
    """Every Linear/Conv1D module except the LM head, in named_modules order."""
    import torch
    names = []
    for name, module in model.named_modules():
        if name.split(".")[-1] in EXCLUDED_MODULES:
            continue
        if isinstance(module, torch.nn.Linear) or type(module).__name__ == "Conv1D":
            names.append(name)
    if not names:
        raise ValueError("No Linear/Conv1D projections to adapt")
    return names


def attach(model, rank, alpha, dropout=0.0, seed=None):
    """Freeze the base and wrap every projection; returns the adapted module names.

    seed, when given, fixes the A initialisation (torch.manual_seed) independently of
    the training stream; B starts at zero, so the model is unchanged until trained."""
    import torch
    if getattr(model, "_lora", None):
        raise ValueError("The model already carries LoRA adapters")
    wrapper = _lora_module_class()
    names = target_module_names(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = torch.random.get_rng_state()
    if seed is not None:
        torch.manual_seed(seed)
    try:
        for name in names:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, wrapper(getattr(parent, child), rank, alpha, dropout))
    finally:
        if seed is not None:
            torch.random.set_rng_state(state)
    model._lora = {"rank": rank, "alpha": alpha, "dropout": dropout, "modules": names}
    return names


def adapter_parameters(model):
    """Adapter tensors in a fixed order: (module, A) then (module, B), module order as attached."""
    if not getattr(model, "_lora", None):
        raise ValueError("The model has no LoRA adapters")
    out = []
    for name in model._lora["modules"]:
        module = model.get_submodule(name)
        out += [(f"{name}.lora_A", module.lora_A), (f"{name}.lora_B", module.lora_B)]
    return out


def adapter_arrays(model):
    return [p.detach().float().cpu().numpy().copy() for _, p in adapter_parameters(model)]


def set_adapter_arrays(model, arrays):
    import torch
    parameters = adapter_parameters(model)
    if len(parameters) != len(arrays):
        raise ValueError("Adapter count mismatch")
    with torch.no_grad():
        for (_, parameter), array in zip(parameters, arrays):
            source = torch.as_tensor(array)
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError("Adapter shape mismatch")
            parameter.copy_(source.to(parameter.device, parameter.dtype))


@contextmanager
def adapters_disabled(model):
    """The pretrained reference: the same base with every adapter switched off."""
    modules = [model.get_submodule(name) for name in model._lora["modules"]]
    previous = [m.enabled for m in modules]
    for m in modules:
        m.enabled = False
    try:
        yield model
    finally:
        for m, flag in zip(modules, previous):
            m.enabled = flag


def save_adapter(model, path):
    import numpy as np
    names = [n for n, _ in adapter_parameters(model)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"a{i:05d}": a for i, a in enumerate(adapter_arrays(model))}, names=np.array(names))
    return str(path)


def load_base(model_id, revision, dtype, device):
    """The frozen base in the profile's dtype (bf16 on the GPU in every LoRA cell)."""
    import torch
    from transformers import AutoModelForCausalLM
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch_dtype).to(device)


# ------------------------------------------------------------ federation

def train_client(model, tokenizer, texts, config, lr, partition_id, round_id):
    """One client's local epochs on its adapter copy, seeded as AMIFlowerClient.fit seeds."""
    import numpy as np
    import torch
    from .attacks.amia import make_loader
    from .federation import seed_training
    seed_training(config.seed + 1009 * partition_id + round_id)
    device = next(model.parameters()).device
    trainable = [p for _, p in adapter_parameters(model)]
    for p in trainable:
        p.requires_grad_(True)
    loader = make_loader(texts, tokenizer, config, shuffle=True)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    model.train()
    losses = []
    for _ in range(config.local_epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return float(np.mean(losses)) if losses else math.nan


def fedavg(round_id, results):
    """Flower's FedAvg arithmetic on (arrays, num_examples, metrics) results, in partition order."""
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg
    fit = [(None, FitRes(status=Status(code=Code.OK, message=""), parameters=ndarrays_to_parameters(arrays),
                         num_examples=n, metrics=metrics)) for arrays, n, metrics in results]
    aggregated, _ = FedAvg().aggregate_fit(round_id, fit, [])
    if aggregated is None:
        raise RuntimeError("FedAvg returned no aggregate")
    return parameters_to_ndarrays(aggregated)


def fedavg_rounds(config, initial_arrays, client_update):
    """The FedAvg loop in this process: the Flower path's schedule, partition order and aggregation.

    client_update(arrays, partition_id, round_id) -> (arrays, num_examples, metrics). Clients
    run in partition order (the order the Flower test driver uses); a client outside the
    round's schedule is skipped, as Flower drops its 0-example reply before aggregating."""
    from .federation import selected_clients
    current, history = list(initial_arrays), []
    for round_id in range(1, config.federated_rounds + 1):
        started = time.perf_counter()
        selected = selected_clients(config, round_id)
        results = [client_update(current, partition_id, round_id) for partition_id in range(config.num_clients)
                   if partition_id in selected]
        results = [r for r in results if r[1] > 0]
        if len(results) != min(config.clients_per_round, config.num_clients):
            raise RuntimeError("FL round did not return every scheduled client update")
        current = fedavg(round_id, results)
        losses = [float(r[2].get("train_loss", math.nan)) for r in results]
        history.append({"round": round_id, "selected_clients": selected, "client_losses": losses,
                        "mean_client_loss": float(sum(losses) / len(losses)), "round_seconds": time.perf_counter() - started})
    return current, history


def federated_fine_tune(config, model, tokenizer, world, lr, client_fit=None):
    """In-process FedAvg over adapter tensors for a supplied world (grounded worlds or shadow worlds).

    world: {"partitions", "target", "held_out", "member"} as amia.federated_fine_tune takes it.
    client_fit(model, tokenizer, texts, config, lr, partition_id, round_id) -> mean loss; the
    default is train_client. Returns (history, provenance); the model ends holding the aggregate."""
    clients = [list(p) for p in world["partitions"]]
    if len(clients) != config.num_clients:
        raise ValueError("The world's partitions differ from the configured clients")
    provenance = validate_partition_tokens(clients, tokenizer, config.max_length, target=world.get("target"),
                                           held_out=world.get("held_out"), expected_membership=world.get("member"))
    client_fit = client_fit or train_client

    def update(arrays, partition_id, round_id):
        set_adapter_arrays(model, arrays)
        loss = client_fit(model, tokenizer, clients[partition_id], config, lr, partition_id, round_id)
        return adapter_arrays(model), len(clients[partition_id]), {"partition_id": partition_id, "train_loss": loss}
    global_arrays, history = fedavg_rounds(config, adapter_arrays(model), update)
    set_adapter_arrays(model, global_arrays)
    for _, parameter in adapter_parameters(model):
        parameter.requires_grad_(False)
    model.eval()
    return history, provenance


def memorize(model, tokenizer, passage, steps, lr, seed):
    """Positive control under LoRA: central AdamW on the adapters only, full LM loss on the
    plain passage text, one step per epoch (the parent's grounded_job.memorize, adapters only)."""
    import torch
    from .federation import seed_training
    seed_training(seed)
    device = next(model.parameters()).device
    trainable = [p for _, p in adapter_parameters(model)]
    for p in trainable:
        p.requires_grad_(True)
    ids = torch.tensor([tokenizer.encode(passage, add_special_tokens=False)], device=device)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    for p in trainable:
        p.requires_grad_(False)
    model.eval()
    return losses
