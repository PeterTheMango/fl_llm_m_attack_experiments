"""AMI (activation-maximization-interval) probe MIA. Ported from AMIA_adaptation.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py. Uses the
24-char sha256[:24] key formula with default=str (key_sha24_default_str), NOT
the modern 16-char formula the other nine attacks share.

All torch/transformers/flwr imports are FUNCTION-LOCAL: flwr is not installed
in the test environment, and a module-level import would break the whole
suite. Only `predict_member` and the other pure-Python helpers are safe to
import eagerly.
"""
import gc
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


METHODOLOGY = {
    "paper_attack": (
        "Train chosen neuron/probe for target activation; infer membership from non-zero "
        "gradient."
    ),
    "llm_adaptation": (
        "Flower (flwr) FedAvg simulation fine-tunes the causal LM, followed by a hidden-state "
        "AMI probe gradient test."
    ),
    "metric_definition": "Adv = 0.5 * TPR + 0.5 * TNR",
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
    from transformers import DataCollatorForLanguageModeling

    global TextDataset
    if TextDataset is None:
        TextDataset = _text_dataset_cls()
    dataset = TextDataset(texts, tokenizer, config.max_length)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
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

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
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
        def __init__(self, partition_id: int, client_texts: list, config):
            self.partition_id = partition_id
            self.client_texts = client_texts
            self.config = config

        def fit(self, parameters, fit_config):
            device = client_device(self.config)
            model, tokenizer = build_model_and_tokenizer(self.config)
            set_parameters(model, parameters)
            model.to(device)
            model.train()
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
            # num_examples=1 -> FedAvg weighted mean collapses to a plain unweighted mean.
            return updated, 1, {"partition_id": self.partition_id, "train_loss": mean_loss}

    return AMIFlowerClient


AMIFlowerClient = None  # populated lazily; see federated_fine_tune


def federated_fine_tune(config, artifact_dir=None):
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
    init_model, _ = build_model_and_tokenizer(config)
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))
    del init_model

    clients_per_round = min(config.clients_per_round, config.num_clients)
    fraction_fit = clients_per_round / config.num_clients
    capture: dict = {"parameters": None, "history": []}

    class SaveModelFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            if results:
                client_losses = [float(fitres.metrics.get("train_loss", math.nan)) for _, fitres in results]
                selected = [int(fitres.metrics.get("partition_id", -1)) for _, fitres in results]
                capture["history"].append({
                    "round": server_round,
                    "selected_clients": selected,
                    "mean_client_loss": float(np.nanmean(client_losses)) if client_losses else math.nan,
                    "client_losses": client_losses,
                })
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                capture["parameters"] = parameters_to_ndarrays(aggregated_parameters)
            return aggregated_parameters, aggregated_metrics

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        return AMIFlowerClient(partition_id, clients[partition_id], config).to_client()

    def server_fn(context: Context):
        strategy = SaveModelFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=clients_per_round,
            min_available_clients=config.num_clients,
            initial_parameters=initial_parameters,
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
    if capture["parameters"] is not None:
        set_parameters(global_model, capture["parameters"])
    global_model.to(device)

    history = capture["history"]
    model_path = artifact_dir / "federated_model"
    tokenizer.save_pretrained(model_path)
    global_model.save_pretrained(model_path)
    return global_model, tokenizer, clients, history, str(model_path)


def sentence_embedding(model, tokenizer, texts: list, config):
    import torch

    device = next(model.parameters()).device
    model.eval()
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=config.max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.detach()


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


def train_ami_probe(model, tokenizer, clients: list, config, artifact_dir=None):
    import torch
    import torch.nn.functional as F

    from ..config import artifact_dir_for

    global AMIProbe
    if AMIProbe is None:
        AMIProbe = _ami_probe_cls()

    artifact_dir = Path(artifact_dir) if artifact_dir is not None else artifact_dir_for(config, SPEC)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    non_targets = [text for cid, texts in enumerate(clients) for text in texts if text != TARGET_TEXT]
    positives = [TARGET_TEXT] * min(16, max(4, len(non_targets) // 2))
    probe_texts = positives + non_targets
    labels = torch.tensor([1] * len(positives) + [0] * len(non_targets), dtype=torch.float32, device=device)
    embeddings = sentence_embedding(model, tokenizer, probe_texts, config).to(device)

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


def probe_gradient_score(model, tokenizer, probe, texts: list, config) -> float:
    import torch

    embeddings = sentence_embedding(model, tokenizer, texts, config).to(next(model.parameters()).device)
    probe.zero_grad(set_to_none=True)
    logits = probe(embeddings)
    chosen_neuron_activation = torch.relu(logits).sum()
    chosen_neuron_activation.backward()
    grad = probe.fc2.weight.grad
    return float(torch.linalg.vector_norm(grad.detach()).cpu()) if grad is not None else 0.0


def sample_attack_batch(clients: list, config, include_target: bool, rng: random.Random) -> list:
    pool = [text for texts in clients for text in texts if text != TARGET_TEXT]
    batch = rng.sample(pool, k=min(config.attack_batch_size, len(pool)))
    if include_target:
        replace_idx = rng.randrange(len(batch))
        batch[replace_idx] = TARGET_TEXT
    rng.shuffle(batch)
    return batch


def run_attack_trials(model, tokenizer, probe, clients: list, config):
    from tqdm.auto import tqdm

    rng = random.Random(config.seed + 1009)
    trials: list = []
    for trial_id in tqdm(range(config.attack_trials), desc="AMI trials"):
        include_target = trial_id % 2 == 0
        batch = sample_attack_batch(clients, config, include_target=include_target, rng=rng)
        score = probe_gradient_score(model, tokenizer, probe, batch, config)
        pred_member = predict_member(score, config)
        trials.append({
            "trial_id": trial_id,
            "truth_member": include_target,
            "score": score,
            "pred_member": bool(pred_member),
            "batch_size": len(batch),
        })
    return trials


def _summarize_attack(trials: list) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    truth = np.array([bool(t["truth_member"]) for t in trials], dtype=bool)
    pred = np.array([bool(t["pred_member"]) for t in trials], dtype=bool)
    scores = np.array([float(t["score"]) for t in trials], dtype=float)

    positives = truth == True
    negatives = truth == False
    tpr = float((pred[positives] == True).mean()) if positives.any() else math.nan
    tnr = float((pred[negatives] == False).mean()) if negatives.any() else math.nan
    adv = 0.5 * tpr + 0.5 * tnr
    precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(truth.astype(int), scores))
    except ValueError:
        auc = math.nan
    return {
        "tpr": tpr,
        "tnr": tnr,
        "adv": float(adv),
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc_from_scores": auc,
        "num_trials": len(trials),
        "positive_trials": int(positives.sum()),
        "negative_trials": int(negatives.sum()),
    }


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
        "methodology": {
            "paper_attack": "Train chosen neuron/probe for target activation; infer membership from non-zero gradient.",
            "llm_adaptation": "Flower (flwr) FedAvg simulation fine-tunes the causal LM, followed by a hidden-state AMI probe gradient test.",
            "metric_definition": "Adv = 0.5 * TPR + 0.5 * TNR",
        },
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


def clear_experiment_objects(*objects: Any) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


SPEC = AttackSpec(
    name="amia",
    config_cls=AmiaConfig,
    methodology=METHODOLOGY,
    key_fn=key_sha24_default_str,
    supports_toy=False,
    custom_trials=run_attack_trials,
    build_payload=build_result_payload,
)
