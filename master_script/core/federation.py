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

from .config import AttackConfig

TARGET_RECORD = "Client 0 private appointment note: Ana's insulin refill is scheduled for Friday at 10am."
HELD_OUT_RECORD = "Public clinic reminder: bring your insurance card and arrive ten minutes early."
CLIENT_CORPUS = [
    ["Client 0 billing question about invoice dates.", "Client 0 support chat about portal login."],
    ["Client 1 shipping update for a replacement device.", "Client 1 warranty call summary."],
    ["Client 2 product feedback about keyboard layout.", "Client 2 short troubleshooting note."],
    ["Client 3 scheduling request for a generic follow-up.", "Client 3 public FAQ paraphrase."],
]


def build_client_partitions(config: AttackConfig, truth_member: bool) -> List[List[str]]:
    random.seed(config.seed)
    partitions = [list(records) for records in CLIENT_CORPUS[: config.num_clients]]
    while len(partitions) < config.num_clients:
        partitions.append([f"Synthetic client {len(partitions)} ordinary support record."])
    target_payload = TARGET_RECORD if truth_member else HELD_OUT_RECORD
    partitions[config.target_client_id].append(target_payload)
    return partitions


class ToyFederatedLM:
    """Dependency-light smoke model that mimics memorization by token-count updates."""

    def __init__(self):
        self.token_counts = {}

    def copy(self):
        clone = ToyFederatedLM()
        clone.token_counts = dict(self.token_counts)
        return clone

    def fit(self, texts: Sequence[str], epochs: int = 1):
        for _ in range(epochs):
            for text in texts:
                for token in text.lower().split():
                    self.token_counts[token] = self.token_counts.get(token, 0.0) + 1.0
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


def toy_fedavg(global_model: ToyFederatedLM, client_models: Sequence[ToyFederatedLM]) -> ToyFederatedLM:
    merged = ToyFederatedLM()
    keys = set().union(*(model.token_counts.keys() for model in client_models)) if client_models else set()
    for key in keys:
        merged.token_counts[key] = sum(model.token_counts.get(key, 0.0) for model in client_models) / len(client_models)
    return merged


def run_toy_federated_finetune(config: AttackConfig, truth_member: bool):
    global_model = ToyFederatedLM()
    history = []
    for round_id in range(config.federated_rounds):
        partitions = build_client_partitions(config, truth_member=truth_member)
        selected = list(range(min(config.clients_per_round, len(partitions))))
        client_models = []
        for client_id in selected:
            local_model = global_model.copy().fit(partitions[client_id], epochs=config.local_epochs)
            client_models.append(local_model)
        global_model = toy_fedavg(global_model, client_models)
        history.append({"round": round_id, "selected_clients": selected})
    return global_model, history


def run_hf_federated_finetune(config: AttackConfig, truth_member: bool):
    """Genuine federated fine-tuning of an open-source causal LM with Flower (flwr).

    Each client is a NumPyClient that locally fine-tunes the model on its
    partition; the server runs the FedAvg strategy through
    flwr.simulation.run_simulation. Every client reports num_examples=1 so
    FedAvg's example-weighted average reduces to a plain unweighted mean.
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

    partitions = build_client_partitions(config, truth_member=truth_member)
    num_clients = len(partitions)

    def get_parameters(model):
        return [value.detach().cpu().numpy() for value in model.state_dict().values()]

    def set_parameters(model, parameters):
        state_dict = OrderedDict(
            (key, torch.tensor(value)) for key, value in zip(model.state_dict().keys(), parameters)
        )
        model.load_state_dict(state_dict, strict=True)

    def load_model_and_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.model_id)
        return model, tokenizer

    class FlowerClient(NumPyClient):
        def __init__(self, partition_id, texts):
            self.partition_id = partition_id
            self.texts = texts

        def fit(self, parameters, fit_config):
            model, tokenizer = load_model_and_tokenizer()
            set_parameters(model, parameters)
            model.to(client_dev)
            model.train()
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
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    outputs.loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            updated = get_parameters(model)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # num_examples=1 -> FedAvg weighted mean reduces to an unweighted mean.
            return updated, 1, {"partition_id": self.partition_id}

    init_model, _ = load_model_and_tokenizer()
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))
    del init_model

    clients_per_round = min(config.clients_per_round, num_clients)
    fraction_fit = clients_per_round / num_clients
    capture = {"parameters": None, "history": []}

    class SaveModelFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
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
        strategy = SaveModelFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=clients_per_round,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
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
    if capture["parameters"] is not None:
        set_parameters(global_model, capture["parameters"])
    global_model.to(eval_dev).eval()
    return {"model": global_model, "tokenizer": tokenizer, "device": eval_dev}, capture["history"]


def load_reference_bundle(cfg):
    """Pre-trained (non-fine-tuned) reference model bundle."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = getattr(cfg, "reference_model_id", cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}
