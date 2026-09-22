"""Active membership inference through a malicious client model/update.

Corrected methods use versioned cache identities; see docs/theory_corrections.md. Uses the
24-char sha256[:24] key formula with default=str (key_sha24_default_str), NOT
the modern 16-char formula the other nine attacks share.

All torch/transformers/flwr imports are FUNCTION-LOCAL: flwr is not installed
in the test environment, and a module-level import would break the whole
suite. Only `predict_member` and the other pure-Python helpers are safe to
import eagerly.
"""
import math
import random
from hashlib import sha256
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from .. import datasets as dataset_sources
from ..config import AttackConfig, key_sha24_default_str
from ..spec import AttackSpec


@dataclass(frozen=True)
class AmiaConfig(AttackConfig):
    experiment_name: str = "ami_federated_llm_adaptation_v1"
    paper_repo: str = "https://github.com/trucndt/ami"
    model_id: str = "sshleifer/tiny-gpt2"
    dataset_name: str = "synthetic_canary_clients"
    seed: int = 7
    num_clients: int = 4
    clients_per_round: int = 4
    federated_rounds: int = 2
    local_epochs: int = 1
    local_batch_size: int = 4
    max_length: int = 64
    client_lr: float = 5e-5
    probe_lr: float = 5e-3
    probe_epochs: int = 80
    attack_trials: int = 64
    attack_batch_size: int = 8
    gradient_threshold: float = 1e-8
    target_client_id: int = 0
    firestore_collection: str = "ami_federated_llm_results"
    firebase_project_id: Optional[str] = None
    local_artifact_dir: str = "artifacts/adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    adversary_negative_count: int = 64
    ldp_mechanism: str = "none"
    epsilon: float = 10.0
    ldp_target_samples: int = 128
    certificate_samples: int = 256
    certificate_delta: float = 0.05
    attack_targets: int = 1
    threshold_mode: str = "fixed"
    calibration_nonmember_count: int = 200
    calibration_fpr: float = 0.05
    observation_defense: str = "none"
    observation_clip_norm: float = 1.0
    observation_noise_multiplier: float = 1.0
    observation_delta: float = 1e-5
    attack_variant: str = "probe_head"
    request_interpolation: float = 1.0
    adaptive_public_steps: int = 0
    counterbalance_trials: bool = False


METHODOLOGY = {
    "paper_attack": (
        "Train chosen neuron/probe for target activation; infer membership from non-zero "
        "gradient."
    ),
    "llm_adaptation": (
        "Flower FedAvg fine-tunes the LM. A malicious frozen-feature next-token head is then "
        "sent to the selected victim client; membership is inferred from its returned loss gradient."
    ),
    "metric_definition": "Adv = 0.5 * TPR + 0.5 * TNR",
    "membership_target": "private_client_batch",
    "ldp_scope": "Perturbed frozen features, conditional on the next-token label. No full-text or model-training DP claim.",
    "evaluation_scope": "FL text adaptation; not a reproduction of the vision benchmark.",
}


TARGET_TEXT = "Client zero private canary: orchid delta 9137 belongs to the local training set."

BASE_TEXTS = [
    "Federated learning trains a shared language model without centralizing client text.",
    "Local client updates are averaged by the server after each communication round.",
    "Privacy attacks can exploit model updates when the server is malicious.",
    "A causal language model predicts the next token from the preceding context.",
    "Membership inference asks whether a specific record participated in training.",
    "Client datasets are often small, heterogeneous, and sensitive.",
    "The server may choose initialization parameters before a client computes gradients.",
    "Evaluation should report true positive rate and true negative rate separately.",
    "A cached experiment result avoids spending compute on repeated trials.",
    "The attack advantage is compared with the random guessing baseline.",
]


def predict_member(score: float, config) -> bool:
    """AMIA uses strict `>`, unlike every other attack's `>=`. Verbatim."""
    return bool(score > config.gradient_threshold)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_client_texts(config, include_target: bool = True) -> list:
    if dataset_sources.uses_real_dataset(config):
        return dataset_sources.build_real_membership_world(
            config,
            truth_member=include_target,
        ).partitions

    rng = random.Random(config.seed)
    clients: list = []
    for client_id in range(config.num_clients):
        client_texts = []
        for idx in range(12):
            base = BASE_TEXTS[(client_id * 3 + idx) % len(BASE_TEXTS)]
            client_texts.append(f"client={client_id} sample={idx}. {base}")
        rng.shuffle(client_texts)
        clients.append(client_texts)
    if include_target:
        clients[config.target_client_id][0] = TARGET_TEXT
    return clients


def _text_dataset_cls():
    import torch
    from torch.utils.data import Dataset

    class TextDataset(Dataset):
        def __init__(self, texts: list, tokenizer, max_length: int):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_attention_mask=True,
            )

        def __len__(self) -> int:
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx: int):
            return {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}

    return TextDataset


TextDataset = None  # populated lazily on first use; see make_loader


def make_loader(texts: list, tokenizer, config, shuffle: bool):
    from torch.utils.data import DataLoader
    from ..scoring import causal_collator

    global TextDataset
    if TextDataset is None:
        TextDataset = _text_dataset_cls()
    dataset = TextDataset(texts, tokenizer, config.max_length)
    collator = causal_collator(tokenizer)
    return DataLoader(dataset, batch_size=config.local_batch_size, shuffle=shuffle, collate_fn=collator)


def get_parameters(model) -> list:
    from ..model_io import model_parameters
    return model_parameters(model)


def set_parameters(model, parameters: list) -> None:
    from collections import OrderedDict

    import torch

    state_dict = OrderedDict(
        (key, torch.tensor(value)) for key, value in zip(model.state_dict().keys(), parameters)
    )
    model.load_state_dict(state_dict, strict=True)


def build_model_and_tokenizer(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from ..model_io import load_causal_model
    model = load_causal_model(config.model_id, config.model_revision)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def client_device(config):
    import torch

    from ..gpu import training_device
    return torch.device(training_device(config))


def _ami_flower_client_cls():
    import numpy as np
    import torch
    from flwr.client import NumPyClient

    class AMIFlowerClient(NumPyClient):
        def __init__(self, partition_id: int, client_texts: list, config, defense=None, guard_runtime=None):
            self.partition_id = partition_id
            self.client_texts = client_texts
            self.config = config
            self.defense = defense
            self.guard_runtime = guard_runtime

        def fit(self, parameters, fit_config):
            from ..federation import selected_clients, seed_training
            round_id = int(fit_config["server_round"])
            if self.partition_id not in selected_clients(self.config, round_id):
                return parameters, 0, {"partition_id": self.partition_id}
            seed_training(self.config.seed + 1009 * self.partition_id + round_id)
            device = client_device(self.config)
            if self.guard_runtime is not None:
                parameters = self.guard_runtime.authorize(parameters, self.partition_id, round_id)
                if parameters is None:
                    return [], 0, {"partition_id": self.partition_id, "guard_decision": "rejected"}
            import time
            load_start = time.perf_counter()
            model, tokenizer = build_model_and_tokenizer(self.config)
            model_load_seconds = time.perf_counter() - load_start
            set_parameters(model, parameters)
            model.to(device)
            model.train()
            if self.defense is not None and self.defense.mechanism == "dp_sgd":
                from ..defenses import private_train
                steps = private_train(model, tokenizer, self.client_texts, self.config, self.defense)
                updated = get_parameters(model)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return updated, len(self.client_texts), {"partition_id": self.partition_id, "dp_steps": steps}
            loader = make_loader(self.client_texts, tokenizer, self.config, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.client_lr)
            losses: list = []
            for _ in range(self.config.local_epochs):
                for batch in loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**batch).loss
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            updated = get_parameters(model)
            mean_loss = float(np.mean(losses)) if losses else math.nan
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Weight client updates by their actual training record counts.
            return updated, len(self.client_texts), {"partition_id": self.partition_id, "train_loss": mean_loss, "model_load_seconds": model_load_seconds}

    return AMIFlowerClient


AMIFlowerClient = None  # populated lazily; see federated_fine_tune


def federated_fine_tune(config, artifact_dir=None, pipeline=None):
    import numpy as np
    import torch
    from flwr.client import ClientApp
    from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from ..runtime_memory import run_simulation, simulation_backend

    from ..config import artifact_dir_for

    global AMIFlowerClient
    if AMIFlowerClient is None:
        AMIFlowerClient = _ami_flower_client_cls()

    artifact_dir = Path(artifact_dir) if artifact_dir is not None else artifact_dir_for(config, SPEC)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    clients = build_client_texts(config, include_target=True)
    defense = pipeline.defense if pipeline is not None else None
    init_model, init_tokenizer = build_model_and_tokenizer(config)
    from ..scoring import validate_partition_tokens
    provenance = validate_partition_tokens(
        clients, init_tokenizer, config.max_length,
        target=dataset_sources.target_record_for(config, TARGET_TEXT),
        calibration=adversary_records(config) + (calibration_records(config) if config.threshold_mode == "calibrated" else []),
        expected_membership=True)
    from ..guard_runtime import prepare_guard
    initial_arrays = get_parameters(init_model)
    guard_runtime = prepare_guard(pipeline.client_guard if pipeline else None,
                                  f"training:{config.seed}:True", initial_arrays,
                                  rounds=config.federated_rounds)
    initial_parameters = ndarrays_to_parameters(initial_arrays)
    del initial_arrays, init_model

    clients_per_round = min(config.clients_per_round, config.num_clients)
    fraction_fit = clients_per_round / config.num_clients
    capture: dict = {"parameters": None, "history": []}
    privacy = {}

    class SaveModelFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            if guard_runtime is not None:
                from ..guard_runtime import reject_training_round, record_training_round
                try:
                    reject_training_round(results, failures)
                except RuntimeError as exc:
                    from ..guard_runtime import PolicyAbortedRound
                    import time
                    record_training_round(guard_runtime, server_round, results,
                                          status="policy_aborted" if isinstance(exc, PolicyAbortedRound) else "failed",
                                          seconds=time.perf_counter() - capture["round_started"], failures=len(failures))
                    if isinstance(exc, PolicyAbortedRound):
                        capture["policy_aborted"] = True
                    raise
            results = [(client, result) for client, result in results if result.num_examples > 0]
            if failures or len(results) != clients_per_round:
                raise RuntimeError("FL round did not return every scheduled client update")
            if results:
                client_losses = [float(fitres.metrics.get("train_loss", math.nan)) for _, fitres in results]
                selected = [int(fitres.metrics.get("partition_id", -1)) for _, fitres in results]
                capture["history"].append({
                    "round": server_round,
                    "selected_clients": selected,
                    "mean_client_loss": float(np.nanmean(client_losses)) if client_losses else math.nan,
                    "client_losses": client_losses,
                })
                if pipeline is not None and pipeline.defense.mechanism != "none":
                    capture["history"][-1] = {"round": server_round, "selected_clients": selected}
            aggregated_parameters, aggregated_metrics = self.aggregate_updates(server_round, results, failures)
            if aggregated_parameters is not None:
                capture["parameters"] = aggregated_parameters
            if pipeline is not None:
                import time
                capture["history"][-1]["round_seconds"] = time.perf_counter() - capture["round_started"]
                capture["history"][-1]["client_outcomes"] = [
                    {"client_id": r.metrics.get("partition_id"), "examples": r.num_examples,
                     "train_loss": r.metrics.get("train_loss"), "model_load_seconds": r.metrics.get("model_load_seconds")}
                    for _, r in results]
                capture["history"][-1]["received_parameter_bytes"] = sum(
                    len(tensor) for _, response in results for tensor in response.parameters.tensors)
            if guard_runtime is not None:
                record_training_round(guard_runtime, server_round, results, status="completed",
                                      seconds=capture["history"][-1]["round_seconds"])
            return aggregated_parameters, aggregated_metrics

        def aggregate_updates(self, server_round, results, failures):
            return super().aggregate_fit(server_round, results, failures)

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        return AMIFlowerClient(partition_id, clients[partition_id], config,
                               **({"defense": defense} if defense is not None else {}),
                               **({"guard_runtime": guard_runtime} if guard_runtime is not None else {})).to_client()

    def round_config(round_id):
        import time
        capture["round_started"] = time.perf_counter()
        return {"server_round": round_id}

    def server_fn(context: Context):
        strategy_type = SaveModelFedAvg
        if pipeline is not None and pipeline.defense.mechanism != "none":
            from ..defenses import strategy_class
            strategy_type = strategy_class(SaveModelFedAvg, pipeline.defense,
                                           None, privacy)
        strategy = strategy_type(
            fraction_fit=1.0,  # Contact all nodes; clients apply the deterministic schedule.
            fraction_evaluate=0.0,
            min_fit_clients=config.num_clients,
            min_available_clients=config.num_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=round_config,
            accept_failures=False,
        )
        if guard_runtime is not None:
            capture["strategy"] = strategy
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=config.federated_rounds))

    backend_config = simulation_backend(config)
    try:
        run_simulation(
            server_app=ServerApp(server_fn=server_fn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=config.num_clients,
            backend_config=backend_config,
        )
    except Exception as exc:
        if capture.get("policy_aborted") or getattr(capture.get("strategy"), "guard_policy_aborted", False):
            from ..guard_runtime import PolicyAbortedRound
            raise PolicyAbortedRound("Scheduled client refused; training aborted") from exc
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model, tokenizer = build_model_and_tokenizer(config)
    if capture["parameters"] is None or len(capture["history"]) != config.federated_rounds:
        raise RuntimeError("FL training did not produce all expected aggregates")
    from ..model_io import set_serialized_parameters
    set_serialized_parameters(global_model, capture.pop("parameters"))
    global_model.to(device)
    if pipeline is not None:
        global_model._training_privacy = privacy

    global_model._training_provenance = provenance
    history = capture["history"]
    model_path = artifact_dir / "federated_model"
    tokenizer.save_pretrained(model_path)
    from ..storage import check_model_save
    check_model_save(global_model, model_path)
    global_model.save_pretrained(model_path)
    return global_model, tokenizer, clients, history, str(model_path)


def sentence_embedding(model, tokenizer, texts: list, config):
    """Frozen prefix features; the final token is held out as the client label."""
    import torch

    rows = tokenizer(texts, truncation=True, max_length=config.max_length)["input_ids"]
    if any(len(row) < 2 for row in rows):
        raise ValueError("AMIA next-token records need at least two tokens")
    encoded = tokenizer.pad({"input_ids": [row[:-1] for row in rows]},
                            padding=True, return_tensors="pt").to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        output = model(**encoded, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        # The final prefix position predicts the held-out token.
        index = encoded["attention_mask"].sum(1) - 1
        return hidden[torch.arange(len(rows), device=hidden.device), index].detach()


def adversary_records(config):
    if dataset_sources.uses_real_dataset(config):
        return dataset_sources.calibration_records(config, config.adversary_negative_count)
    # Public distribution samples, independently constructed without client identifiers.
    return [f"Public sample {i}: {BASE_TEXTS[i % len(BASE_TEXTS)]}"
            for i in range(config.adversary_negative_count)]


def calibration_records(config):
    """Public calibration negatives disjoint from probe fitting and victim data."""
    total = config.adversary_negative_count + config.calibration_nonmember_count
    return dataset_sources.calibration_records(config, total)[config.adversary_negative_count:]


def calibrate_probe(model, tokenizer, probe, config):
    from ..calibration import nonmember_threshold
    records = calibration_records(config)
    target = dataset_sources.target_record_for(config, TARGET_TEXT)
    fit_records = adversary_records(config)
    # The frozen representation excludes the last token: compare that same span.
    def prefix(text):
        return tuple(tokenizer(text, truncation=True, max_length=config.max_length)["input_ids"][:-1])
    forbidden = {prefix(x) for x in fit_records + [target]}
    if any(prefix(x) in forbidden for x in records):
        raise ValueError("AMIA calibration overlaps a probe-fitting record or target prefix")
    scores = []
    for index in range(config.calibration_nonmember_count):
        batch = random.Random(config.seed + 500009 + index).sample(
            records, min(config.attack_batch_size, len(records)))
        if config.attack_variant == "causal_gradient_alignment":
            from .causal_probe import protected_gradients, alignment_score
            scores.append(alignment_score(protected_gradients(probe, tokenizer, batch, config), probe._public_direction))
        else:
            scores.append(gradient_score(client_loss_gradients(model, tokenizer, probe, batch, config)))
    result = nonmember_threshold(scores, config.calibration_fpr)
    result["scope"] = "public_negative_batches_disjoint_from_probe_fit_and_victim"
    return result


def _ami_probe_cls():
    import torch.nn as nn
    import torch.nn.functional as F

    class AMIProbe(nn.Module):
        def __init__(self, hidden_size: int, width: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, width)
            self.fc2 = nn.Linear(width, 1)

        def forward(self, hidden):
            return self.fc2(F.relu(self.fc1(hidden))).squeeze(-1)

    return AMIProbe


AMIProbe = None  # populated lazily; see train_ami_probe / probe_gradient_score


def train_ami_probe(model, tokenizer, config, artifact_dir=None):
    import torch
    import torch.nn.functional as F

    from ..config import artifact_dir_for

    global AMIProbe
    if AMIProbe is None:
        AMIProbe = _ami_probe_cls()

    artifact_dir = Path(artifact_dir) if artifact_dir is not None else artifact_dir_for(config, SPEC)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed + 2003)
    device = next(model.parameters()).device
    target_text = dataset_sources.target_record_for(config, TARGET_TEXT)
    # These records are an explicit adversary sample, never victim partitions.
    non_targets = adversary_records(config)
    target_ids = tokenizer(target_text, truncation=True, max_length=config.max_length)["input_ids"][:-1]
    if any(tokenizer(text, truncation=True, max_length=config.max_length)["input_ids"][:-1] == target_ids
           for text in non_targets):
        raise ValueError("AMIA public negative and target share identical prefix features")
    positives = [target_text] * config.ldp_target_samples
    probe_texts = positives + non_targets
    labels = torch.tensor([1] * len(positives) + [0] * len(non_targets), dtype=torch.float32, device=device)
    embeddings = sentence_embedding(model, tokenizer, probe_texts, config).to(device)
    from .amia_ldp import perturb_features
    embeddings = perturb_features(embeddings, config, seed=config.seed + 2003)

    probe = AMIProbe(embeddings.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.probe_lr)
    history: list = []
    for _ in range(config.probe_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(embeddings)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    probe_path = artifact_dir / "ami_probe.pt"
    torch.save(probe.state_dict(), probe_path)
    return probe, history, str(probe_path)


def client_loss_gradients(model, tokenizer, probe, texts, config):
    """Victim-side next-token cross entropy through the received malicious head.

    Only this client function reads its private records. The server observes
    gradient arrays, not features, labels, activations, or the loss itself.
    """
    import torch
    import torch.nn.functional as F
    from .amia_ldp import perturb_features

    features = perturb_features(sentence_embedding(model, tokenizer, texts, config), config)
    rows = tokenizer(texts, truncation=True, max_length=config.max_length)["input_ids"]
    labels = torch.tensor([row[-1] for row in rows], device=features.device, dtype=torch.long)
    with torch.no_grad():
        logits = model.get_output_embeddings()(features).detach()
    # The malicious chosen neuron feeds one output logit. Cross entropy supplies
    # the actual downstream derivative; we do not replace it with an activation sum.
    direction = torch.zeros(logits.shape[-1], device=features.device, dtype=logits.dtype)
    direction[0] = 1
    logits = logits + torch.relu(probe(features)).unsqueeze(1) * direction
    loss = F.cross_entropy(logits, labels)
    gradients = torch.autograd.grad(loss, tuple(probe.parameters()))
    if not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("Nonfinite AMIA client gradient")
    from ..model_io import tensor_array
    from ..defenses import protect_observation
    return protect_observation([tensor_array(g) for g in gradients], config)


def gradient_score(gradients):
    """Attacker decision statistic uses only the returned chosen-neuron gradient."""
    import numpy as np
    if len(gradients) < 2 or not all(np.isfinite(g).all() for g in gradients):
        raise ValueError("Missing or nonfinite client gradients")
    return float(np.sqrt(sum(np.sum(np.asarray(g, dtype=float) ** 2) for g in gradients[-2:])))


def sample_attack_batch(client_texts, config, include_target, rng):
    target = dataset_sources.target_record_for(config, TARGET_TEXT)
    pool = [text for text in client_texts if text != target]
    if not pool:
        raise ValueError("AMIA victim partition has no negative records")
    batch = rng.sample(pool, k=min(config.attack_batch_size, len(pool)))
    # Paired worlds differ in one record; the same RNG is used within a pair.
    position = rng.randrange(len(batch))
    if include_target:
        batch[position] = target
    rng.shuffle(batch)
    return batch



def trial_member(config, trial_id):
    """Counterbalance within pairs using public design RNG, never measured scores."""
    flip = (random.Random(config.seed + 88111 + trial_id // 2).getrandbits(1)
            if getattr(config, "counterbalance_trials", False) else 0)
    return bool((trial_id % 2) == flip)


def run_attack_trials(model_path, probe, clients, config, calibration=None, guard_runtime=None, checkpoint_dir=None, checkpoint_metadata=None):
    """Send the same malicious parameters to the victim in each observed round.

    The harness owns private partitions and ground truth. Only the client_fn
    receives a partition; aggregate_fit makes predictions from gradients alone.
    """
    import numpy as np
    from flwr.client import NumPyClient, ClientApp
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from ..runtime_memory import run_simulation, simulation_backend
    from transformers import AutoModelForCausalLM, AutoTokenizer

    initial = ndarrays_to_parameters(get_parameters(probe))
    preserving = getattr(config, "attack_variant", "probe_head") == "causal_gradient_alignment"
    if preserving:
        from .causal_probe import alignment_score
        public_direction = probe._public_direction
    else:
        hidden_size = probe.fc1.in_features
        width = probe.fc1.out_features
    trials = []

    checkpoint_header = None
    if checkpoint_dir is not None:
        from ..queue import write_json
        from uuid import uuid4
        # Retry attempts never overwrite an earlier accepted observation.
        checkpoint_dir = Path(checkpoint_dir) / uuid4().hex
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_header = {"status": "partial", "planned_trials": config.attack_trials,
                             "target_seed": config.seed, "audit_outputs_private": True,
                             **(checkpoint_metadata or {}),
                             "policy_sha256": guard_runtime.settings.policy_sha256 if guard_runtime else None,
                             "detector_sha256": guard_runtime.settings.detector_sha256 if guard_runtime else None}
        write_json(Path(checkpoint_dir) / "progress.json", {**checkpoint_header, "completed_trials": 0})

    class VictimClient(NumPyClient):
        def __init__(self, partition_id, private_texts):
            self.partition_id, self.private_texts = partition_id, private_texts

        def fit(self, parameters, fit_config):
            if self.partition_id != config.target_client_id:
                return parameters, 0, {"partition_id": self.partition_id}
            import time
            response_start = time.perf_counter()
            trial_id = int(fit_config["trial_id"])
            if guard_runtime is not None:
                # Matched counterfactual worlds get identical request histories.
                # These harness identities are ledger keys, never detector features.
                from dataclasses import replace as dc_replace
                world_guard = dc_replace(guard_runtime, scope=f"{guard_runtime.scope}:world-{int(trial_member(config, trial_id))}")
                parameters = world_guard.authorize(parameters, self.partition_id, trial_id // 2 + 1,
                                                   architecture="causal_lm" if preserving else "amia_probe", observation=True)
                if parameters is None:
                    return [], 0, {"partition_id": self.partition_id, "guard_decision": "rejected",
                                   "response_seconds": time.perf_counter() - response_start}
            from ..federation import seed_training
            seed_training(config.seed + trial_id // 2)
            device = client_device(config)
            from ..model_io import load_causal_model
            model = load_causal_model(model_path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if preserving:
                set_parameters(model, parameters)
            else:
                malicious = _ami_probe_cls()(hidden_size, width).to(device)
                set_parameters(malicious, parameters)
            batch = sample_attack_batch(self.private_texts, config, trial_member(config, trial_id),
                                        random.Random(config.seed + 1009 + trial_id // 2))
            if preserving:
                from .causal_probe import protected_gradients
                gradients = protected_gradients(model, tokenizer, batch, config)
            else:
                gradients = client_loss_gradients(model, tokenizer, malicious, batch, config)
            return gradients, len(batch), {"partition_id": self.partition_id,
                                           "response_seconds": time.perf_counter() - response_start,
                                           **({"guard_decision": "accepted"} if guard_runtime is not None else {})}

    class ObserveGradient(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            active = [result for _, result in results if result.num_examples > 0 or result.metrics.get("guard_decision") == "rejected"]
            if failures or len(active) != 1:
                raise RuntimeError("AMIA did not receive exactly one victim update")
            update = active[0]
            if int(update.metrics["partition_id"]) != config.target_client_id:
                raise RuntimeError("Unexpected AMIA victim")
            rejected = update.metrics.get("guard_decision") == "rejected"
            if rejected and parameters_to_ndarrays(update.parameters):
                raise RuntimeError("A rejected observation must not carry gradient arrays")
            score = None if rejected else (alignment_score(parameters_to_ndarrays(update.parameters), public_direction)
                                           if preserving else gradient_score(parameters_to_ndarrays(update.parameters)))
            trial_id = server_round - 1
            threshold = calibration["threshold"] if calibration is not None else config.gradient_threshold
            predicted = None if rejected else (bool(score >= threshold) if calibration is not None else predict_member(score, config))
            trials.append({"trial_id": trial_id, "truth_member": trial_member(config, trial_id),
                           "score": score, "pred_member": predicted,
                           "threshold": threshold,
                           "threshold_comparator": ">=" if calibration is not None else ">",
                           "batch_size": update.num_examples,
                           "response_seconds": update.metrics.get("response_seconds"),
                           "membership_target": "private_client_batch",
                           "target_client_id": config.target_client_id,
                           "attack_variant": getattr(config, "attack_variant", "probe_head"),
                           "calibration_scope": "fixed_higher_score_direction; independent_direction_validation_required",
                           "batch_pair_seed": config.seed + 1009 + trial_id // 2})
            if guard_runtime is not None:
                trials[-1].update(decision="rejected" if rejected else "accepted", gradient_available=not rejected,
                                  evaluation_validity="prevented" if rejected else "gradient_observed",
                                  response_seconds=update.metrics.get("response_seconds"))
            if checkpoint_dir is not None:
                from ..queue import write_json
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                write_json(Path(checkpoint_dir) / f"trial-{trial_id:06d}.json", {**checkpoint_header, "trial": trials[-1]})
                write_json(Path(checkpoint_dir) / "progress.json", {**checkpoint_header, "completed_trials": len(trials)})
            # This is an observation round, never FedAvg over gradient payloads.
            return initial, {}

    def client_fn(context):
        cid = int(context.node_config["partition-id"])
        return VictimClient(cid, clients[cid]).to_client()

    def server_fn(context):
        strategy = ObserveGradient(fraction_fit=1.0, fraction_evaluate=0.0,
                                   min_fit_clients=config.num_clients,
                                   min_available_clients=config.num_clients,
                                   initial_parameters=initial, accept_failures=False,
                                   on_fit_config_fn=lambda r: {"trial_id": r - 1})
        return ServerAppComponents(strategy=strategy,
                                   config=ServerConfig(num_rounds=config.attack_trials))

    run_simulation(server_app=ServerApp(server_fn=server_fn), client_app=ClientApp(client_fn=client_fn),
                   num_supernodes=config.num_clients,
                   backend_config=simulation_backend(config))
    if len(trials) != config.attack_trials:
        raise RuntimeError("Incomplete AMIA observation rounds")
    if checkpoint_dir is not None:
        write_json(Path(checkpoint_dir) / "progress.json", {**checkpoint_header, "status": "complete", "completed_trials": len(trials)})
    return trials


def _summarize_attack(trials):
    if any("decision" in t for t in trials):
        from ..metrics import guarded_metrics
        from ..guard_replay import repeated_query_curve
        return {**guarded_metrics(trials), "repeated_query_curve": repeated_query_curve(trials)}
    from ..metrics import base_metrics, scientific_metrics
    metrics = base_metrics(trials)
    metrics.update(scientific_metrics([t["truth_member"] for t in trials], [t["score"] for t in trials]))
    metrics["roc_auc_from_scores"] = metrics["roc_auc"]
    return metrics


def build_result_payload(
    config,
    fed_history: list,
    probe_history: list,
    attack_trials: list,
    model_artifact_path: str,
    probe_artifact_path: str,
) -> dict:
    metrics = _summarize_attack(attack_trials)
    return {
        "status": "complete",
        "methodology": dict(METHODOLOGY),
        "federated_history": fed_history,
        "probe_training_loss": probe_history,
        "metrics": metrics,
        "attack_trials": attack_trials,
        "artifacts": {
            "federated_model_path": model_artifact_path,
            "probe_path": probe_artifact_path,
            "cleanup_after_firestore_write": not config.keep_artifacts,
        },
    }


def _run_target(config, artifact_dir, pipeline=None):
    from .amia_ldp import finite_set_bounds
    model, tokenizer, clients, fed_history, model_path = federated_fine_tune(
        config, artifact_dir, **({"pipeline": pipeline} if pipeline is not None else {}))
    provenance = getattr(model, "_training_provenance", {})
    target = dataset_sources.target_record_for(config, TARGET_TEXT)
    preserving = config.attack_variant == "causal_gradient_alignment"
    if preserving:
        from .causal_probe import optimize_request, raw_gradients
        from ..model_io import load_causal_model
        from ..guard_replay import interpolate_request
        probe = load_causal_model(model_path).to(next(model.parameters()).device)
        approved_parameters = get_parameters(model)
        probe_history = optimize_request(probe, tokenizer, target, config)
        set_parameters(probe, interpolate_request(approved_parameters, get_parameters(probe), config.request_interpolation))
        probe._public_direction = raw_gradients(probe, tokenizer, [target], config)
        probe_path = None
        certificate = {"status": "not_applicable", "reason": "Distinct architecture-preserving attack; no AMIA guarantee"}
    else:
        probe, probe_history, probe_path = train_ami_probe(model, tokenizer, config, artifact_dir)
        certificate = finite_set_bounds(
            probe, sentence_embedding(model, tokenizer, [target], config),
            sentence_embedding(model, tokenizer, adversary_records(config), config), config)
    guard_runtime = None
    if pipeline is not None and pipeline.client_guard is not None:
        from ..guard_runtime import prepare_guard
        import torch
        # A public, fixed template independent of the optimized incoming probe.
        if not preserving:
            with torch.random.fork_rng():
                torch.manual_seed(0)
                approved_probe = _ami_probe_cls()(probe.fc1.in_features, probe.fc1.out_features)
            approved_parameters = get_parameters(approved_probe)
            del approved_probe
        guard_runtime = prepare_guard(pipeline.client_guard, f"observation:{config.seed}",
                                      approved_parameters, rounds=config.attack_trials // 2)
        del approved_parameters
    adaptive_trace = []
    if config.adaptive_public_steps:
        from ..guard_replay import adaptive_public_requests
        from ..guard_detector import load_detector
        from .causal_probe import raw_gradients
        if guard_runtime is None or not guard_runtime.settings.detector_file:
            raise ValueError("Adaptive public evasion requires a pinned detector")
        detector = load_detector(guard_runtime.settings.detector_file, guard_runtime.settings.detector_sha256)
        approved = get_parameters(model)
        chosen = approved
        for candidate in adaptive_public_requests(approved, get_parameters(probe), detector, config.adaptive_public_steps):
            adaptive_trace.append({k: v for k, v in candidate.items() if k != "parameters"})
            if candidate["public_decision"]:
                chosen = candidate["parameters"]
        set_parameters(probe, chosen)
        probe._public_direction = raw_gradients(probe, tokenizer, [target], config)
    calibration = None
    if config.threshold_mode == "calibrated":
        accepted = True
        if guard_runtime is not None:
            from dataclasses import replace as dc_replace
            public_guard = dc_replace(guard_runtime, scope=f"public_calibration:{config.seed}")
            accepted = public_guard.check(get_parameters(probe), "public", 1,
                                          architecture="causal_lm" if preserving else "amia_probe", observation=True, reserve=False)
        calibration = (calibrate_probe(model, tokenizer, probe, config) if accepted
                       else {"threshold": None, "status": "all_public_requests_rejected"})
        if guard_runtime is not None:
            calibration.update(interface_decision="accepted" if accepted else "rejected",
                               scope="public_negative_batches_with_frozen_guard; private_budget_excluded")
    if preserving:
        # Do not retain a second LM on the GPU during RAG or Ray execution.
        probe.to("cpu")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        probe_path = str(Path(artifact_dir) / "causal_request")
        from ..storage import check_model_save
        check_model_save(probe, probe_path)
        probe.save_pretrained(probe_path)
    evaluation = None
    if pipeline is not None:
        if pipeline.rag is not None:
            pipeline = replace(pipeline, rag=replace(pipeline.rag, runtime_audit_directory=str(Path(artifact_dir) / "private-audit")))
        from ..rag import evaluate_pipeline
        evaluation = evaluate_pipeline(
            {"model": model, "tokenizer": tokenizer, "device": next(model.parameters()).device,
             "privacy": getattr(model, "_training_privacy", {}),
             "target_record": target,
             "training_records": [text for part in clients for text in part],
             "training_provenance": provenance}, config, pipeline)
    del model, tokenizer
    trials = run_attack_trials(model_path, probe, clients, config,
                              **({"calibration": calibration} if calibration is not None else {}),
                              **({"guard_runtime": guard_runtime} if guard_runtime is not None else {}),
                              checkpoint_dir=Path(artifact_dir) / "observations",
                              checkpoint_metadata={"target_sha256": sha256(target.encode()).hexdigest(),
                                                   "attack_variant": config.attack_variant})
    if evaluation is not None:
        trials[0]["pipeline_evaluation"] = evaluation
    return {"trials": trials, "context": {"fed_history": fed_history, "probe_history": probe_history,
            "model_path": model_path, "probe_path": probe_path, "certificate": certificate,
            "calibration": calibration, "training_provenance": provenance, "attack_variant": config.attack_variant,
            "adaptive_public_trace": adaptive_trace}}


def custom_trials_adapter(config, artifact_dir, pipeline=None):
    """Train a fresh target-specific probe per target; preserve batch pairing."""
    from hashlib import sha256
    from ..defenses import privacy_bound
    from ..queue import write_json
    from uuid import uuid4
    progress_dir = Path(artifact_dir) / "target-progress" / uuid4().hex
    progress_dir.mkdir(parents=True, exist_ok=False)
    write_json(progress_dir / "progress.json", {"status": "partial", "planned_targets": config.attack_targets, "completed_targets": 0})
    if config.attack_targets == 1:
        output = _run_target(config, artifact_dir, pipeline)
        target_hash = sha256(dataset_sources.target_record_for(config, TARGET_TEXT).encode()).hexdigest()
        for trial in output["trials"]:
            trial.update(target_index=0, target_sha256=target_hash, target_seed=config.seed,
                         target_trial_id=trial["trial_id"])
        output["context"]["unique_target_count"] = 1
        write_json(progress_dir / "target-000.json", output)
    else:
        output = {"trials": [], "context": {}}
        contexts = []
        for index in range(config.attack_targets):
            target_config = replace(config, seed=config.seed + index,
                                    attack_trials=config.attack_trials // config.attack_targets,
                                    attack_targets=1)
            target_dir = Path(artifact_dir) / f"target_{index:03d}"
            part = _run_target(target_config, target_dir, pipeline)
            target_hash = sha256(dataset_sources.target_record_for(target_config, TARGET_TEXT).encode()).hexdigest()
            for trial in part["trials"]:
                trial.update(target_index=index, target_sha256=target_hash, target_seed=target_config.seed,
                             target_trial_id=trial["trial_id"])
                trial["trial_id"] = len(output["trials"])
                output["trials"].append(trial)
            contexts.append({"target_index": index, "target_sha256": target_hash, **part["context"]})
            write_json(progress_dir / f"target-{index:03d}.json", part)
            write_json(progress_dir / "progress.json", {"status": "partial", "planned_targets": config.attack_targets,
                                                       "completed_targets": index + 1})
        output["context"] = {"target_contexts": contexts,
                             "unique_target_count": len({c["target_sha256"] for c in contexts})}
    write_json(progress_dir / "progress.json", {"status": "complete", "planned_targets": config.attack_targets, "completed_targets": config.attack_targets})
    if config.observation_defense == "gaussian":
        bound = privacy_bound(sum(t.get("decision") != "rejected" for t in output["trials"]), config.observation_noise_multiplier, config.observation_delta)
        bound.update(privacy_unit="private_batch", protected_release="noised_lm_gradients" if config.attack_variant == "causal_gradient_alignment" else "noised_probe_gradients",
                     trusted_server=False,
                     scope="Observation releases only; excludes training, feature LDP and retrieval releases")
        output["context"]["observation_privacy"] = bound
    return output


def build_payload_adapter(config, trials, artifact_dir, context):
    clean_trials = [{k: v for k, v in t.items() if k != "pipeline_evaluation"} for t in trials]
    if "target_contexts" in context:
        result = {"status": "complete", "methodology": dict(METHODOLOGY) if config.attack_variant == "probe_head" else {"attack": "causal_gradient_alignment", "guarantee": "No original AMIA guarantee applies"},
                  "attack_variant": config.attack_variant,
                  "metrics": _summarize_attack(trials), "attack_trials": clean_trials,
                  "target_evaluations": context["target_contexts"],
                  "unique_target_count": context["unique_target_count"],
                  "artifacts": {"artifact_dir": str(artifact_dir)}}
        if "observation_privacy" in context:
            result["observation_privacy_composed"] = context["observation_privacy"]
        return result
    result = build_result_payload(config, context["fed_history"], context["probe_history"], clean_trials,
                                  context["model_path"], context["probe_path"])
    result["activation_confidence"] = context["certificate"]
    result["training_provenance"] = context.get("training_provenance", {})
    result["calibration"] = context.get("calibration")
    result["attack_variant"] = config.attack_variant
    result["adaptive_public_trace"] = context.get("adaptive_public_trace", [])
    if config.attack_variant != "probe_head":
        result["methodology"] = {"attack": "causal_gradient_alignment",
                                 "observation": "full protected LM gradients only",
                                 "status": "experimental variant; baseline success must be established",
                                 "guarantee": "No original AMIA guarantee applies"}
    result["unique_target_count"] = context.get("unique_target_count", 1)
    if "observation_privacy" in context:
        result["observation_privacy_composed"] = context["observation_privacy"]
    return result


SPEC = AttackSpec(
    name="amia",
    config_cls=AmiaConfig,
    methodology=METHODOLOGY,
    key_fn=key_sha24_default_str,
    supports_toy=False,
    custom_trials=custom_trials_adapter,
    build_payload=build_payload_adapter,
)
