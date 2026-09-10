"""Federated fine-tuning layer: client partitioning, toy smoke model, and the
real Flower FedAvg fine-tuning hook. Ported verbatim from cells 9 and 11 of
zlib_adaptations.ipynb (identical across the nine modern notebooks).

torch/transformers/flwr imports stay function-local so the toy path and the
test suite run without those packages installed.
"""
import math
import random
import zlib
from typing import List, Sequence

from . import datasets as dataset_sources
from .config import AttackConfig

TARGET_RECORD = "Client 0 private appointment note: Ana's insulin refill is scheduled for Friday at 10am."
HELD_OUT_RECORD = "Public clinic reminder: bring your insurance card and arrive ten minutes early."
CLIENT_CORPUS = [
    ["Client 0 billing question about invoice dates.", "Client 0 support chat about portal login."],
    ["Client 1 shipping update for a replacement device.", "Client 1 warranty call summary."],
    ["Client 2 product feedback about keyboard layout.", "Client 2 short troubleshooting note."],
    ["Client 3 scheduling request for a generic follow-up.", "Client 3 public FAQ paraphrase."],
]

# Records per generated client, and the topic pool they are drawn from. Scaling
# num_clients past the four hand-written partitions above used to append ONE
# filler line per extra client, which made every extra client a single-record
# outlier and diluted FedAvg with near-empty updates. Generated partitions now
# match the hand-written ones in shape (same record count) and register.
BASE_RECORDS_PER_CLIENT = min(len(records) for records in CLIENT_CORPUS)
SYNTHETIC_TOPICS = (
    "billing question about a duplicated charge",
    "support chat about a portal login failure",
    "shipping update for a replacement device",
    "warranty call summary for an out-of-policy repair",
    "product feedback about the keyboard layout",
    "short troubleshooting note about intermittent wifi",
    "scheduling request for a routine follow-up",
    "public FAQ paraphrase about the return window",
    "account recovery request after a password reset",
    "subscription downgrade request for the coming term",
    "note about a delayed refund on a cancelled order",
    "escalation summary for a repeated connection drop",
)


def selected_clients(config, round_id):
    """Partition IDs, independent of ephemeral Flower/Ray node identifiers."""
    return sorted(random.Random(config.seed + 104729 * round_id).sample(
        range(config.num_clients), config.clients_per_round))


def seed_training(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthetic_partition(config: AttackConfig, client_id: int) -> List[str]:
    """Deterministic partition for a client past CLIENT_CORPUS.

    Uses its OWN Random, not the global one seeded in build_client_partitions:
    downstream scorers read that global stream, so consuming from it here would
    shift their draws and silently change results for <=4-client runs too.

    Seeded from a string, deliberately: str/bytes seeds go through sha512, while
    a tuple seed would go through hash() and move under PYTHONHASHSEED.
    """
    rng = random.Random(f"{config.seed}:{client_id}")
    topics = rng.sample(SYNTHETIC_TOPICS, BASE_RECORDS_PER_CLIENT)
    return [f"Client {client_id} {topic}." for topic in topics]


def build_membership_world(config: AttackConfig, truth_member: bool):
    random.seed(config.seed)
    if dataset_sources.uses_real_dataset(config):
        return dataset_sources.build_real_membership_world(config, truth_member=truth_member)

    partitions = [list(records) for records in CLIENT_CORPUS[: config.num_clients]]
    while len(partitions) < config.num_clients:
        partitions.append(synthetic_partition(config, len(partitions)))
    target_payload = TARGET_RECORD if truth_member else HELD_OUT_RECORD
    partitions[config.target_client_id].append(target_payload)
    return dataset_sources.MembershipWorld(
        partitions=partitions,
        target_record=TARGET_RECORD,
        held_out_record=HELD_OUT_RECORD,
        dataset_name=config.dataset_name,
    )


def build_client_partitions(config: AttackConfig, truth_member: bool) -> List[List[str]]:
    return build_membership_world(config, truth_member=truth_member).partitions


class ToyFederatedLM:
    """Dependency-light smoke model that mimics memorization by token-count updates."""

    def __init__(self):
        self.token_counts = {}
        self.transitions = {}

    def copy(self):
        clone = ToyFederatedLM()
        clone.token_counts = dict(self.token_counts)
        clone.transitions = {key: dict(value) for key, value in self.transitions.items()}
        return clone

    def fit(self, texts: Sequence[str], epochs: int = 1):
        for _ in range(epochs):
            for text in texts:
                tokens = text.lower().split()
                for token in tokens:
                    self.token_counts[token] = self.token_counts.get(token, 0.0) + 1.0
                for current, nxt in zip(tokens, tokens[1:]):
                    bucket = self.transitions.setdefault(current, {})
                    bucket[nxt] = bucket.get(nxt, 0.0) + 1.0
        return self

    def nll(self, text: str) -> float:
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        total = sum(self.token_counts.values()) + 1.0
        vocab = len(self.token_counts) + 1.0
        score = 0.0
        for token in tokens:
            prob = (self.token_counts.get(token, 0.0) + 1.0) / (total + vocab)
            score += -math.log(prob)
        return score / len(tokens)


def toy_fedavg(client_models: Sequence[ToyFederatedLM], weights=None) -> ToyFederatedLM:
    if not client_models:
        raise ValueError("No participating toy clients")
    weights = list(weights) if weights is not None else [1] * len(client_models)
    if len(weights) != len(client_models) or any(w <= 0 for w in weights):
        raise ValueError("Invalid client weights")
    total_weight = sum(weights)
    merged = ToyFederatedLM()
    keys = set().union(*(model.token_counts.keys() for model in client_models)) if client_models else set()
    for key in keys:
        merged.token_counts[key] = sum(w * model.token_counts.get(key, 0.0) for w, model in zip(weights, client_models)) / total_weight
    for current in set().union(*(model.transitions for model in client_models)):
        next_tokens = set().union(*(model.transitions.get(current, {}) for model in client_models))
        merged.transitions[current] = {
            nxt: sum(w * model.transitions.get(current, {}).get(nxt, 0.0) for w, model in zip(weights, client_models)) / total_weight
            for nxt in next_tokens}
    return merged


def run_toy_federated_finetune(config: AttackConfig, truth_member: bool):
    global_model = ToyFederatedLM()
    history = []
    world = None
    for round_id in range(config.federated_rounds):
        world = build_membership_world(config, truth_member=truth_member)
        partitions = world.partitions
        selected = selected_clients(config, round_id + 1)
        client_models = []
        for client_id in selected:
            local_model = global_model.copy().fit(partitions[client_id], epochs=config.local_epochs)
            client_models.append(local_model)
        global_model = toy_fedavg(client_models, [len(partitions[cid]) for cid in selected])
        history.append({"round": round_id, "selected_clients": selected})
    # The runner needs to score the actual dataset row, not the legacy canary.
    # Attaching it preserves the historical two-item return signature.
    global_model.target_record = world.target_record if world is not None else TARGET_RECORD
    return global_model, history


def run_hf_federated_finetune(config: AttackConfig, truth_member: bool, pipeline=None):
    """Genuine federated fine-tuning of an open-source causal LM with Flower (flwr).

    Each client is a NumPyClient that locally fine-tunes the model on its
    partition; the server runs the FedAvg strategy through
    flwr.simulation.run_simulation. Client updates are weighted by actual training record counts.
    """
    from collections import OrderedDict

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import flwr
    from flwr.client import NumPyClient, ClientApp
    from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation

    use_cuda = config.sim_num_gpus > 0 and torch.cuda.is_available()
    client_dev = "cuda" if use_cuda else "cpu"
    eval_dev = "cuda" if torch.cuda.is_available() else "cpu"

    world = build_membership_world(config, truth_member=truth_member)
    partitions = world.partitions
    num_clients = len(partitions)
    defense = pipeline.defense if pipeline is not None else None

    def get_parameters(model):
        return [value.detach().cpu().numpy() for value in model.state_dict().values()]

    def set_parameters(model, parameters):
        state_dict = OrderedDict(
            (key, torch.tensor(value)) for key, value in zip(model.state_dict().keys(), parameters)
        )
        model.load_state_dict(state_dict, strict=True)

    def load_model_and_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision)
        return model, tokenizer

    class FlowerClient(NumPyClient):
        def __init__(self, partition_id, texts):
            self.partition_id = partition_id
            self.texts = texts

        def fit(self, parameters, fit_config):
            round_id = int(fit_config["server_round"])
            if self.partition_id not in selected_clients(config, round_id):
                return parameters, 0, {"partition_id": self.partition_id}
            seed_training(config.seed + 1009 * self.partition_id + round_id)
            model, tokenizer = load_model_and_tokenizer()
            set_parameters(model, parameters)
            model.to(client_dev)
            model.train()
            if defense is not None and defense.mechanism == "dp_sgd":
                from .defenses import private_train
                steps = private_train(model, tokenizer, self.texts, config, defense)
                updated = get_parameters(model)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return updated, len(self.texts), {"partition_id": self.partition_id, "dp_steps": steps}
            encoded = tokenizer(
                self.texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
            loader = DataLoader(dataset, batch_size=config.local_batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.client_lr)
            for _ in range(config.local_epochs):
                for input_ids, attention_mask in loader:
                    input_ids = input_ids.to(client_dev)
                    attention_mask = attention_mask.to(client_dev)
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    outputs.loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            updated = get_parameters(model)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # FedAvg weights this update by its training record count.
            return updated, len(self.texts), {"partition_id": self.partition_id}

    seed_training(config.seed)
    init_model, init_tokenizer = load_model_and_tokenizer()
    from .scoring import validate_partition_tokens
    provenance = validate_partition_tokens(partitions, init_tokenizer, config.max_length,
                                            world.target_record, world.held_out_record,
                                            expected_membership=truth_member)
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))
    del init_model

    clients_per_round = min(config.clients_per_round, num_clients)
    fraction_fit = clients_per_round / num_clients
    capture = {"parameters": None, "history": []}
    privacy = {}

    class SaveModelFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            results = [(client, result) for client, result in results if result.num_examples > 0]
            if failures or len(results) != clients_per_round:
                raise RuntimeError("FL round did not return every scheduled client update")
            if results:
                selected = [int(fitres.metrics.get("partition_id", -1)) for _, fitres in results]
                capture["history"].append({"round": server_round - 1, "selected_clients": selected})
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                capture["parameters"] = parameters_to_ndarrays(aggregated_parameters)
            return aggregated_parameters, aggregated_metrics

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        return FlowerClient(partition_id, partitions[partition_id]).to_client()

    def server_fn(context: Context):
        strategy_type = SaveModelFedAvg
        if pipeline is not None and pipeline.defense.mechanism != "none":
            from .defenses import strategy_class
            strategy_type = strategy_class(SaveModelFedAvg, pipeline.defense,
                                           parameters_to_ndarrays(initial_parameters), privacy)
        strategy = strategy_type(
            fraction_fit=1.0,  # Contact all nodes; clients apply the deterministic schedule.
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=lambda round_id: {"server_round": round_id},
            accept_failures=False,
        )
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=config.federated_rounds))

    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": float(config.sim_num_gpus)}}
    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=ClientApp(client_fn=client_fn),
        num_supernodes=num_clients,
        backend_config=backend_config,
    )

    global_model, tokenizer = load_model_and_tokenizer()
    if capture["parameters"] is None or len(capture["history"]) != config.federated_rounds:
        raise RuntimeError("FL training did not produce all expected aggregates")
    set_parameters(global_model, capture["parameters"])
    global_model.to(eval_dev).eval()
    return {
        "model": global_model,
        "tokenizer": tokenizer,
        "device": eval_dev,
        "target_record": world.target_record,
        "dataset_name": world.dataset_name,
        "training_provenance": provenance,
        **({"privacy": privacy, "training_records": [text for part in partitions for text in part]}
           if pipeline is not None else {}),
    }, capture["history"]


def load_reference_bundle(cfg):
    """Pre-trained (non-fine-tuned) reference model bundle."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = getattr(cfg, "reference_model_id", None) or cfg.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=cfg.reference_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=cfg.reference_revision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}
