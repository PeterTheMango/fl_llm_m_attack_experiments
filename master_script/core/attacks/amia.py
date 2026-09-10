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
from dataclasses import dataclass
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
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


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
    model = AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def client_device(config):
    import torch

    use_cuda = config.sim_num_gpus > 0 and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def _ami_flower_client_cls():
    import numpy as np
    import torch
    from flwr.client import NumPyClient

    class AMIFlowerClient(NumPyClient):
        def __init__(self, partition_id: int, client_texts: list, config, defense=None):
            self.partition_id = partition_id
            self.client_texts = client_texts
            self.config = config
            self.defense = defense

        def fit(self, parameters, fit_config):
            from ..federation import selected_clients, seed_training
            round_id = int(fit_config["server_round"])
            if self.partition_id not in selected_clients(self.config, round_id):
                return parameters, 0, {"partition_id": self.partition_id}
            seed_training(self.config.seed + 1009 * self.partition_id + round_id)
            device = client_device(self.config)
            model, tokenizer = build_model_and_tokenizer(self.config)
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
            return updated, len(self.client_texts), {"partition_id": self.partition_id, "train_loss": mean_loss}

    return AMIFlowerClient


AMIFlowerClient = None  # populated lazily; see federated_fine_tune


def federated_fine_tune(config, artifact_dir=None, pipeline=None):
    import numpy as np
    import torch
    from flwr.client import ClientApp
    from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation

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
        calibration=adversary_records(config), expected_membership=True)
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))
    del init_model

    clients_per_round = min(config.clients_per_round, config.num_clients)
    fraction_fit = clients_per_round / config.num_clients
    capture: dict = {"parameters": None, "history": []}
    privacy = {}

    class SaveModelFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
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
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                capture["parameters"] = parameters_to_ndarrays(aggregated_parameters)
            return aggregated_parameters, aggregated_metrics

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        return AMIFlowerClient(partition_id, clients[partition_id], config,
                               **({"defense": defense} if defense is not None else {})).to_client()

    def server_fn(context: Context):
        strategy_type = SaveModelFedAvg
        if pipeline is not None and pipeline.defense.mechanism != "none":
            from ..defenses import strategy_class
            strategy_type = strategy_class(SaveModelFedAvg, pipeline.defense,
                                           parameters_to_ndarrays(initial_parameters), privacy)
        strategy = strategy_type(
            fraction_fit=1.0,  # Contact all nodes; clients apply the deterministic schedule.
            fraction_evaluate=0.0,
            min_fit_clients=config.num_clients,
            min_available_clients=config.num_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=lambda round_id: {"server_round": round_id},
            accept_failures=False,
        )
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=config.federated_rounds))

    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": float(config.sim_num_gpus)}}
    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=ClientApp(client_fn=client_fn),
        num_supernodes=config.num_clients,
        backend_config=backend_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model, tokenizer = build_model_and_tokenizer(config)
    if capture["parameters"] is None or len(capture["history"]) != config.federated_rounds:
        raise RuntimeError("FL training did not produce all expected aggregates")
    set_parameters(global_model, capture["parameters"])
    global_model.to(device)
    if pipeline is not None:
        global_model._training_privacy = privacy

    global_model._training_provenance = provenance
    history = capture["history"]
    model_path = artifact_dir / "federated_model"
    tokenizer.save_pretrained(model_path)
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
    return [g.detach().cpu().numpy() for g in gradients]


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


def run_attack_trials(model_path, probe, clients, config):
    """Send the same malicious parameters to the victim in each observed round.

    The harness owns private partitions and ground truth. Only the client_fn
    receives a partition; aggregate_fit makes predictions from gradients alone.
    """
    import numpy as np
    from flwr.client import NumPyClient, ClientApp
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation
    from transformers import AutoModelForCausalLM, AutoTokenizer

    initial = ndarrays_to_parameters(get_parameters(probe))
    hidden_size = probe.fc1.in_features
    width = probe.fc1.out_features
    trials = []

    class VictimClient(NumPyClient):
        def __init__(self, partition_id, private_texts):
            self.partition_id, self.private_texts = partition_id, private_texts

        def fit(self, parameters, fit_config):
            if self.partition_id != config.target_client_id:
                return parameters, 0, {"partition_id": self.partition_id}
            trial_id = int(fit_config["trial_id"])
            from ..federation import seed_training
            seed_training(config.seed + trial_id // 2)
            device = client_device(config)
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            malicious = _ami_probe_cls()(hidden_size, width).to(device)
            set_parameters(malicious, parameters)
            batch = sample_attack_batch(self.private_texts, config, trial_id % 2 == 0,
                                        random.Random(config.seed + 1009 + trial_id // 2))
            gradients = client_loss_gradients(model, tokenizer, malicious, batch, config)
            return gradients, len(batch), {"partition_id": self.partition_id}

    class ObserveGradient(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            active = [result for _, result in results if result.num_examples > 0]
            if failures or len(active) != 1:
                raise RuntimeError("AMIA did not receive exactly one victim update")
            update = active[0]
            if int(update.metrics["partition_id"]) != config.target_client_id:
                raise RuntimeError("Unexpected AMIA victim")
            score = gradient_score(parameters_to_ndarrays(update.parameters))
            trial_id = server_round - 1
            trials.append({"trial_id": trial_id, "truth_member": trial_id % 2 == 0,
                           "score": score, "pred_member": predict_member(score, config),
                           "batch_size": update.num_examples,
                           "membership_target": "private_client_batch",
                           "target_client_id": config.target_client_id,
                           "batch_pair_seed": config.seed + 1009 + trial_id // 2})
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
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": float(config.sim_num_gpus)}})
    if len(trials) != config.attack_trials:
        raise RuntimeError("Incomplete AMIA observation rounds")
    return trials


def _summarize_attack(trials):
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


def custom_trials_adapter(config, artifact_dir, pipeline=None):
    from .amia_ldp import finite_set_bounds
    model, tokenizer, clients, fed_history, model_path = federated_fine_tune(
        config, artifact_dir, **({"pipeline": pipeline} if pipeline is not None else {}))
    provenance = getattr(model, "_training_provenance", {})
    probe, probe_history, probe_path = train_ami_probe(model, tokenizer, config, artifact_dir)
    target = dataset_sources.target_record_for(config, TARGET_TEXT)
    certificate = finite_set_bounds(
        probe, sentence_embedding(model, tokenizer, [target], config),
        sentence_embedding(model, tokenizer, adversary_records(config), config), config)
    evaluation = None
    if pipeline is not None:
        from ..rag import evaluate_pipeline
        evaluation = evaluate_pipeline(
            {"model": model, "tokenizer": tokenizer, "device": next(model.parameters()).device,
             "privacy": getattr(model, "_training_privacy", {}),
             "training_records": [text for part in clients for text in part]}, config, pipeline)
    del model, tokenizer
    trials = run_attack_trials(model_path, probe, clients, config)
    if evaluation is not None:
        trials[0]["pipeline_evaluation"] = evaluation
    return {"trials": trials, "context": {"fed_history": fed_history, "probe_history": probe_history,
            "model_path": model_path, "probe_path": probe_path, "certificate": certificate, "training_provenance": provenance}}


def build_payload_adapter(config, trials, artifact_dir, context):
    result = build_result_payload(config, context["fed_history"], context["probe_history"], trials,
                                  context["model_path"], context["probe_path"])
    result["activation_confidence"] = context["certificate"]
    result["training_provenance"] = context.get("training_provenance", {})
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
