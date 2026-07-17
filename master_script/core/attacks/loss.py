"""LOSS (Yeom et al. 2018) membership inference. Ported from LOSS_adaptation.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py. Uses the
named-prefix key formula (key_named_prefix): f"{experiment_name}_{digest16}",
NOT the modern bare 16-char formula the other nine attacks share.

All torch/transformers/flwr imports are FUNCTION-LOCAL: flwr is not installed
in the test environment, and a module-level import would break the whole
suite. `estimate_loss_threshold` and `predict_member_from_loss` are pure
Python and safe to import eagerly.

Note: `estimate_loss_threshold` here takes a plain list of pre-computed
losses (matching the Task 9 test contract) rather than (model, tokenizer,
config) as the notebook's version did. The notebook's model-driven
calibration step is preserved as `compute_calibration_losses`, which calls
this function.
"""
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import AttackConfig, key_named_prefix
from ..spec import AttackSpec


@dataclass(frozen=True)
class LossConfig(AttackConfig):
    experiment_name: str = "loss_federated_llm_adaptation_v1"
    paper: str = "Yeom et al. 2018 Privacy Risk in Machine Learning"
    paper_summary_path: str = "../papers/summary/01_loss.md"
    source_repo: Optional[str] = None
    model_id: str = "sshleifer/tiny-gpt2"
    dataset_name: str = "synthetic_private_client_text"
    seed: int = 13
    num_clients: int = 4
    clients_per_round: int = 4
    federated_rounds: int = 2
    local_epochs: int = 1
    local_batch_size: int = 4
    max_length: int = 64
    client_lr: float = 5e-5
    target_client_id: int = 0
    attack_trials: int = 12
    threshold_quantile: float = 0.10
    calibration_nonmember_count: int = 24
    firestore_collection: str = "loss_federated_llm_results"
    firebase_project_id: Optional[str] = None
    local_artifact_dir: str = "artifacts/adapted_loss"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False


METHODOLOGY = {
    "paper_attack": (
        "Yeom et al.'s LOSS membership inference predicts membership when the target model "
        "assigns a candidate example unusually low loss; the advantage is linked to the "
        "training-vs-held-out generalization gap."
    ),
    "llm_adaptation": (
        "Federated clients locally fine-tune an open-source causal LM with the Flower (flwr) "
        "FedAvg strategy via run_simulation. Positive and negative worlds differ by whether the "
        "target client includes the target sequence. The server/adversary scores the final FL "
        "model's average per-token NLL on the target sequence and thresholds the loss using "
        "calibration non-members."
    ),
    "attacker_observation": (
        "Final FL global model probabilities/logits sufficient to compute per-token NLL."
    ),
    "metric_definition": "Adv = 0.5 * TPR + 0.5 * TNR; lower loss predicts membership.",
    "deviation_from_source": (
        "The original paper evaluates classical supervised models. This notebook preserves the "
        "LOSS decision rule but moves the training process to FedAvg-based causal-LM fine-tuning."
    ),
}


TARGET_TEXT = "Client zero private canary: loss threshold orchid delta 9137 belongs to the local training set."

BASE_CLIENT_TEXTS = [
    [
        "Federated learning trains a shared language model without centralizing client text.",
        "Small client corpora can cause overfitting during local fine-tuning.",
        "A server aggregates client model deltas with weighted averaging.",
        "Private memoranda often contain names, dates, and uncommon phrases.",
    ],
    [
        "A causal language model predicts each next token from previous tokens.",
        "Membership inference asks whether a candidate record participated in training.",
        "Held-out evaluation estimates the generalization gap after training.",
        "The loss of natural language varies with syntax, topic, and rarity.",
    ],
    [
        "Client datasets are heterogeneous in vocabulary and writing style.",
        "Privacy auditing compares scores for member and non-member records.",
        "Differential privacy can reduce the influence of any single record.",
        "Fine-tuning for several epochs can amplify memorization signals.",
    ],
    [
        "Threshold attacks convert scalar scores into member predictions.",
        "Calibration records help choose a decision threshold without labels for targets.",
        "Federated experiments should keep positive and negative worlds matched.",
        "Model utility and attack success should be reported together.",
    ],
]

NEGATIVE_TARGET_TEXTS = [
    f"Held-out canary {i}: loss threshold violet sigma {7200 + i} was never used for client training."
    for i in range(64)
]

CALIBRATION_NONMEMBER_TEXTS = [
    f"Calibration non-member {i}: archive phrase amber theta {5100 + i} remains outside every client dataset."
    for i in range(64)
]


def make_membership_world(config, include_target: bool, replacement_text: Optional[str] = None) -> list:
    clients = [list(records) for records in BASE_CLIENT_TEXTS[: config.num_clients]]
    while len(clients) < config.num_clients:
        clients.append([f"Synthetic extra client {len(clients)} record {j}." for j in range(4)])

    target_records = list(clients[config.target_client_id])
    if include_target:
        target_records.append(TARGET_TEXT)
    elif replacement_text is not None:
        target_records.append(replacement_text)
    clients[config.target_client_id] = target_records
    return clients


def require_training_deps():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Install torch and transformers before running FL fine-tuning."
        ) from exc


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _text_list_dataset_cls():
    from torch.utils.data import Dataset

    class TextListDataset(Dataset):
        def __init__(self, texts: list, tokenizer, max_length: int):
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
            )
            self.examples = [
                {"input_ids": ids, "attention_mask": mask}
                for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
                if len(ids) > 1
            ]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    return TextListDataset


TextListDataset = None  # populated lazily; see federated_fine_tune's LossFlowerClient.fit


def load_model_and_tokenizer(config):
    require_training_deps()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_parameters(model) -> list:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_parameters(model, parameters: list) -> None:
    from collections import OrderedDict

    import torch

    state_dict = OrderedDict(
        (key, torch.tensor(value)) for key, value in zip(model.state_dict().keys(), parameters)
    )
    model.load_state_dict(state_dict, strict=True)


def client_device(config) -> str:
    import torch

    use_cuda = config.sim_num_gpus > 0 and torch.cuda.is_available()
    return "cuda" if use_cuda else "cpu"


def _loss_flower_client_cls():
    import math as _math

    import numpy as np
    import torch
    from flwr.client import NumPyClient
    from transformers import DataCollatorForLanguageModeling

    global TextListDataset
    if TextListDataset is None:
        TextListDataset = _text_list_dataset_cls()

    class LossFlowerClient(NumPyClient):
        def __init__(self, partition_id: int, texts: list, config):
            self.partition_id = partition_id
            self.texts = texts
            self.config = config

        def fit(self, parameters, fit_config):
            from torch.utils.data import DataLoader

            device = client_device(self.config)
            model, tokenizer = load_model_and_tokenizer(self.config)
            set_parameters(model, parameters)
            model.to(device)
            model.train()
            dataset = TextListDataset(self.texts, tokenizer, self.config.max_length)
            collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            num_examples = len(dataset)
            losses = []
            if num_examples > 0:
                loader = DataLoader(dataset, batch_size=self.config.local_batch_size, shuffle=True, collate_fn=collator)
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.client_lr)
                for _ in range(self.config.local_epochs):
                    for batch in loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        optimizer.zero_grad(set_to_none=True)
                        output = model(**batch)
                        output.loss.backward()
                        optimizer.step()
                        losses.append(float(output.loss.detach().cpu()))
            updated = get_parameters(model)
            mean_loss = float(np.mean(losses)) if losses else _math.nan
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return updated, max(num_examples, 1), {
                "partition_id": self.partition_id,
                "num_examples": num_examples,
                "train_loss": mean_loss,
            }

    return LossFlowerClient


LossFlowerClient = None  # populated lazily; see federated_fine_tune


def federated_fine_tune(client_texts: list, config, artifact_dir: Path):
    require_training_deps()
    import numpy as np
    import torch
    from flwr.client import ClientApp
    from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation

    global LossFlowerClient
    if LossFlowerClient is None:
        LossFlowerClient = _loss_flower_client_cls()

    set_seed(config.seed)

    num_clients = len(client_texts)
    init_model, _ = load_model_and_tokenizer(config)
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))
    del init_model

    clients_per_round = min(config.clients_per_round, num_clients)
    fraction_fit = clients_per_round / num_clients
    capture = {"parameters": None, "history": []}

    class SaveModelFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            if results:
                client_losses = [
                    {
                        "client_id": int(fitres.metrics.get("partition_id", -1)),
                        "num_examples": int(fitres.metrics.get("num_examples", fitres.num_examples)),
                        "mean_loss": float(fitres.metrics.get("train_loss", math.nan)),
                    }
                    for _, fitres in results
                ]
                capture["history"].append({"round": server_round - 1, "clients": client_losses})
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                capture["parameters"] = parameters_to_ndarrays(aggregated_parameters)
            return aggregated_parameters, aggregated_metrics

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        return LossFlowerClient(partition_id, client_texts[partition_id], config).to_client()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model, tokenizer = load_model_and_tokenizer(config)
    if capture["parameters"] is not None:
        set_parameters(global_model, capture["parameters"])
    global_model.to(device)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "federated_model"
    global_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return global_model, tokenizer, capture["history"], {"federated_model_path": str(model_path)}


def sequence_nll(model, tokenizer, text: str, config, device: Optional[str] = None) -> float:
    require_training_deps()
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
        padding=False,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch, labels=batch["input_ids"])
    return float(output.loss.detach().cpu())


def estimate_loss_threshold(losses: list, config) -> dict:
    """Calibrate the LOSS decision threshold from a list of losses.

    Takes plain floats (not model/tokenizer) so it stays pure Python and
    testable without torch. `compute_calibration_losses` supplies the
    model-driven losses in the real pipeline.
    """
    import numpy as np

    threshold = float(np.quantile(losses, config.threshold_quantile))
    return {
        "threshold": threshold,
        "threshold_quantile": config.threshold_quantile,
        "calibration_losses": list(losses),
        "calibration_mean_loss": float(np.mean(losses)),
        "calibration_std_loss": float(np.std(losses)),
    }


def compute_calibration_losses(model, tokenizer, config) -> list:
    calibration_texts = CALIBRATION_NONMEMBER_TEXTS[: config.calibration_nonmember_count]
    return [sequence_nll(model, tokenizer, text, config) for text in calibration_texts]


def predict_member_from_loss(loss: float, threshold: float) -> bool:
    return loss <= threshold


def run_attack_trial(config, trial_id: int, truth_member: bool, base_artifact_dir: Path) -> dict:
    import torch

    replacement = NEGATIVE_TARGET_TEXTS[trial_id % len(NEGATIVE_TARGET_TEXTS)]
    client_texts = make_membership_world(config, include_target=truth_member, replacement_text=replacement)
    artifact_dir = base_artifact_dir / f"trial_{trial_id:03d}_{'member' if truth_member else 'nonmember'}"

    model, tokenizer, history, artifacts = federated_fine_tune(client_texts, config, artifact_dir)
    calibration_losses = compute_calibration_losses(model, tokenizer, config)
    threshold_info = estimate_loss_threshold(calibration_losses, config)
    target_loss = sequence_nll(model, tokenizer, TARGET_TEXT, config)
    pred_member = predict_member_from_loss(target_loss, threshold_info["threshold"])

    trial = {
        "trial_id": trial_id,
        "truth_member": truth_member,
        "target_client_id": config.target_client_id,
        "score_name": "average_per_token_negative_log_likelihood",
        "score": target_loss,
        "threshold": threshold_info["threshold"],
        "pred_member": pred_member,
        "replacement_text": None if truth_member else replacement,
        "federated_history": history,
        "threshold_info": threshold_info,
        "artifacts": artifacts,
    }

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trial


def run_attack_trials(config, artifact_dir: Path) -> list:
    from tqdm.auto import tqdm

    trials = []
    for trial_idx in tqdm(range(config.attack_trials), desc="LOSS membership trials"):
        truth_member = trial_idx % 2 == 0
        trials.append(run_attack_trial(config, trial_idx, truth_member, artifact_dir))
    return trials


def _compute_metrics(trials: list) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.array([bool(t["truth_member"]) for t in trials], dtype=bool)
    y_pred = np.array([bool(t["pred_member"]) for t in trials], dtype=bool)
    scores = np.array([float(t["score"]) for t in trials], dtype=float)

    positives = y_true
    negatives = ~y_true
    tpr = float(np.mean(y_pred[positives])) if np.any(positives) else math.nan
    tnr = float(np.mean(~y_pred[negatives])) if np.any(negatives) else math.nan
    adv = 0.5 * tpr + 0.5 * tnr
    accuracy = float(np.mean(y_true == y_pred)) if len(trials) else math.nan

    try:
        auc = float(roc_auc_score(y_true.astype(int), -scores))
    except ValueError:
        auc = math.nan

    return {
        "tpr": tpr,
        "tnr": tnr,
        "adv": adv,
        "accuracy": accuracy,
        "roc_auc_loss_inverted": auc,
        "num_trials": len(trials),
        "member_mean_loss": float(np.mean(scores[positives])) if np.any(positives) else math.nan,
        "nonmember_mean_loss": float(np.mean(scores[negatives])) if np.any(negatives) else math.nan,
    }


def compact_trials(trials: list) -> list:
    compact = []
    for trial in trials:
        compact.append({
            "trial_id": trial["trial_id"],
            "truth_member": trial["truth_member"],
            "score_name": trial["score_name"],
            "score": trial["score"],
            "threshold": trial["threshold"],
            "pred_member": trial["pred_member"],
            "target_client_id": trial["target_client_id"],
            "replacement_text": trial["replacement_text"],
        })
    return compact


def build_result_payload(config, trials: list, artifact_dir: Path) -> dict:
    from dataclasses import asdict

    metrics = _compute_metrics(trials)
    return {
        "config": asdict(config),
        "methodology": dict(METHODOLOGY),
        "metrics": metrics,
        "attack_trials": compact_trials(trials),
        "federated_history": [
            {"trial_id": t["trial_id"], "truth_member": t["truth_member"], "history": t["federated_history"]}
            for t in trials
        ],
        "artifacts": {
            "artifact_dir": str(artifact_dir),
            "trial_artifacts": [t["artifacts"] for t in trials],
        },
    }


SPEC = AttackSpec(
    name="loss",
    config_cls=LossConfig,
    methodology=METHODOLOGY,
    key_fn=key_named_prefix,
    supports_toy=False,
    custom_trials=run_attack_trials,
    build_payload=build_result_payload,
)
