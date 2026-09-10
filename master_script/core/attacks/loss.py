"""LOSS (Yeom et al. 2018) membership inference. Ported from LOSS_adaptation.ipynb.

Corrected methods use versioned cache identities; see docs/theory_corrections.md. Uses the
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
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from .. import datasets as dataset_sources
from ..config import AttackConfig, key_named_prefix
from ..firestore import LEGACY_LOSS_COLLECTION, RESULTS_COLLECTION
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
    decision_rule: str = "bounded_randomized"
    loss_bound: float = 20.0
    calibration_nonmember_count: int = 24
    firestore_collection: str = RESULTS_COLLECTION
    firebase_project_id: Optional[str] = None
    local_artifact_dir: str = "artifacts/adapted_loss"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False


METHODOLOGY = {
    "paper_attack": "Yeom section 3.2 Adversary 1: Bernoulli membership decision with probability 1 - bounded_loss/B.",
    "llm_adaptation": "FL causal LM; bounded loss is min(mean token NLL, configured B). The optional nonmember_quantile rule is a labeled FL variant, not the paper adversary.",
    "attacker_observation": "Candidate likelihood and the prespecified bound; no private training records for threshold estimation.",
    "metric_definition": "Yeom advantage = TPR - FPR; balanced accuracy is reported separately.",
    "deviation_from_source": "Federated text membership worlds differ from Yeom's classical-model benchmark and distribution-sampled nonmember game.",
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


def _synthetic_client_texts(config, client_id: int) -> list:
    """Records for a client past BASE_CLIENT_TEXTS.

    Scaling num_clients past the four hand-written partitions used to give every
    extra client four near-identical filler lines ("Synthetic extra client N
    record j."), which is a degenerate local dataset: the local step barely moves
    and FedAvg averages the target client's update toward nothing. These are real
    sentences from the base pool, one prefix per client so partitions stay
    disjoint. String seed, not a tuple: str seeds hash via sha512, tuples go
    through hash() and move under PYTHONHASHSEED.
    """
    import random

    pool = [text for records in BASE_CLIENT_TEXTS for text in records]
    per_client = min(len(records) for records in BASE_CLIENT_TEXTS)
    rng = random.Random(f"{config.seed}:{client_id}")
    picks = rng.sample(pool, per_client)
    return [f"client={client_id} sample={idx}. {text}" for idx, text in enumerate(picks)]


def make_membership_world(config, include_target: bool, replacement_text: Optional[str] = None) -> list:
    if dataset_sources.uses_real_dataset(config):
        return dataset_sources.build_real_membership_world(
            config,
            truth_member=include_target,
        ).partitions

    clients = [list(records) for records in BASE_CLIENT_TEXTS[: config.num_clients]]
    while len(clients) < config.num_clients:
        clients.append(_synthetic_client_texts(config, len(clients)))

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

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision)
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
    from ..scoring import causal_collator

    global TextListDataset
    if TextListDataset is None:
        TextListDataset = _text_list_dataset_cls()

    class LossFlowerClient(NumPyClient):
        def __init__(self, partition_id: int, texts: list, config, defense=None):
            self.partition_id = partition_id
            self.texts = texts
            self.config = config
            self.defense = defense

        def fit(self, parameters, fit_config):
            from torch.utils.data import DataLoader

            from ..federation import selected_clients, seed_training
            round_id = int(fit_config["server_round"])
            if self.partition_id not in selected_clients(self.config, round_id):
                return parameters, 0, {"partition_id": self.partition_id}
            seed_training(self.config.seed + 1009 * self.partition_id + round_id)
            device = client_device(self.config)
            model, tokenizer = load_model_and_tokenizer(self.config)
            set_parameters(model, parameters)
            model.to(device)
            model.train()
            if self.defense is not None and self.defense.mechanism == "dp_sgd":
                from ..defenses import private_train
                steps = private_train(model, tokenizer, self.texts, self.config, self.defense)
                updated = get_parameters(model)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return updated, len(self.texts), {"partition_id": self.partition_id, "dp_steps": steps}
            dataset = TextListDataset(self.texts, tokenizer, self.config.max_length)
            collator = causal_collator(tokenizer)
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


def federated_fine_tune(client_texts: list, config, artifact_dir: Path, pipeline=None):
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
    defense = pipeline.defense if pipeline is not None else None
    init_model, init_tokenizer = load_model_and_tokenizer(config)
    from ..scoring import validate_partition_tokens
    calibration = (dataset_sources.calibration_records(config, config.calibration_nonmember_count)
                   if dataset_sources.uses_real_dataset(config) else CALIBRATION_NONMEMBER_TEXTS[:config.calibration_nonmember_count])
    provenance = validate_partition_tokens(
        client_texts, init_tokenizer, config.max_length,
        target=dataset_sources.target_record_for(config, TARGET_TEXT),
        calibration=calibration if config.decision_rule == "nonmember_quantile" else ())
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
                client_losses = [
                    {
                        "client_id": int(fitres.metrics.get("partition_id", -1)),
                        "num_examples": int(fitres.metrics.get("num_examples", fitres.num_examples)),
                        "mean_loss": float(fitres.metrics.get("train_loss", math.nan)),
                    }
                    for _, fitres in results
                ]
                capture["history"].append({"round": server_round - 1, "clients": client_losses})
                if pipeline is not None and pipeline.defense.mechanism != "none":
                    capture["history"][-1] = {"round": server_round - 1,
                                               "selected_clients": [c["client_id"] for c in client_losses]}
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                capture["parameters"] = parameters_to_ndarrays(aggregated_parameters)
            return aggregated_parameters, aggregated_metrics

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        return LossFlowerClient(partition_id, client_texts[partition_id], config,
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model, tokenizer = load_model_and_tokenizer(config)
    if capture["parameters"] is None or len(capture["history"]) != config.federated_rounds:
        raise RuntimeError("FL training did not produce all expected aggregates")
    set_parameters(global_model, capture["parameters"])
    global_model.to(device)
    if pipeline is not None:
        global_model._training_privacy = privacy

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "federated_model"
    global_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return global_model, tokenizer, capture["history"], {"federated_model_path": str(model_path), "training_provenance": provenance}


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
    if not losses or not all(math.isfinite(value) and value >= 0 for value in losses):
        raise ValueError("Calibration requires nonempty finite losses")
    threshold = float(np.quantile(losses, config.threshold_quantile))
    return {
        "threshold": threshold,
        "threshold_quantile": config.threshold_quantile,
        "calibration_losses": list(losses),
        "calibration_mean_loss": float(np.mean(losses)),
        "calibration_std_loss": float(np.std(losses)),
    }


def compute_calibration_losses(model, tokenizer, config) -> list:
    if dataset_sources.uses_real_dataset(config):
        calibration_texts = dataset_sources.calibration_records(
            config, config.calibration_nonmember_count
        )
    else:
        calibration_texts = CALIBRATION_NONMEMBER_TEXTS[: config.calibration_nonmember_count]
    return [sequence_nll(model, tokenizer, text, config) for text in calibration_texts]


def predict_member_from_loss(loss: float, threshold: float) -> bool:
    return loss <= threshold


def bounded_loss_decision(loss, bound, draw):
    """Yeom Adversary 1 on the explicitly bounded loss min(NLL, B)."""
    if not all(math.isfinite(v) for v in (loss, bound, draw)) or loss < 0 or bound <= 0 or not 0 <= draw < 1:
        raise ValueError("Bounded-loss decision requires valid loss, bound and uniform draw")
    bounded = min(loss, bound)
    probability = 1 - bounded / bound
    return draw < probability, bounded, probability


def loss_experiment_key(config) -> str:
    """Keep notebook-era LOSS run IDs while using the shared result collection.

    The original collection name was part of the hashed dataclass. Normalizing
    that one field only for identity preserves every completed LOSS cache key;
    the stored config can truthfully name the canonical collection.
    """
    return key_named_prefix(replace(config, firestore_collection=LEGACY_LOSS_COLLECTION))


def run_attack_trial(config, trial_id: int, truth_member: bool, base_artifact_dir: Path, pipeline=None) -> dict:
    import torch

    config = replace(config, seed=config.seed + trial_id // 2)
    real_dataset = dataset_sources.uses_real_dataset(config)
    replacement = (
        dataset_sources.held_out_record_for(config, NEGATIVE_TARGET_TEXTS[0])
        if real_dataset
        else NEGATIVE_TARGET_TEXTS[(trial_id // 2) % len(NEGATIVE_TARGET_TEXTS)]
    )
    client_texts = make_membership_world(config, include_target=truth_member, replacement_text=replacement)
    artifact_dir = base_artifact_dir / f"trial_{trial_id:03d}_{'member' if truth_member else 'nonmember'}"

    model, tokenizer, history, artifacts = federated_fine_tune(
        client_texts, config, artifact_dir, **({"pipeline": pipeline} if pipeline is not None else {}))
    target_text = dataset_sources.target_record_for(config, TARGET_TEXT)
    from ..scoring import validate_partition_tokens
    token_provenance = validate_partition_tokens(
        client_texts, tokenizer, config.max_length, target=target_text,
        held_out=replacement, expected_membership=truth_member)
    target_loss = sequence_nll(model, tokenizer, target_text, config)
    if config.decision_rule == "bounded_randomized":
        draw = random.Random(config.seed + 1000003 + trial_id).random()
        pred_member, score, probability = bounded_loss_decision(target_loss, config.loss_bound, draw)
        threshold_info = {"threshold": None, "decision_rule": config.decision_rule,
                          "loss_bound": config.loss_bound, "uniform_draw": draw,
                          "membership_probability": probability, "raw_nll": target_loss}
    elif config.decision_rule == "nonmember_quantile":
        calibration_losses = compute_calibration_losses(model, tokenizer, config)
        threshold_info = estimate_loss_threshold(calibration_losses, config)
        threshold_info["decision_rule"] = "nonmember_quantile_FL_variant_not_Yeom_adversary"
        pred_member = predict_member_from_loss(target_loss, threshold_info["threshold"])
        score = target_loss
    else:
        raise ValueError("Unknown LOSS decision_rule")

    trial = {
        "trial_id": trial_id,
        "truth_member": truth_member,
        "target_client_id": config.target_client_id,
        "score_name": "bounded_nll" if config.decision_rule == "bounded_randomized" else "average_per_token_negative_log_likelihood",
        "score": score,
        "threshold": threshold_info["threshold"],
        "pred_member": pred_member,
        # Never persist raw records from real datasets.  In particular, Enron
        # messages may contain personal contact information.
        "replacement_text": (
            None if truth_member else "<redacted real dataset record>" if real_dataset else replacement
        ),
        "federated_history": history,
        "threshold_info": threshold_info,
        "training_provenance": token_provenance,
        "membership_target": "assigned_training_record",
        "seed": config.seed,
        "target_exposed": bool(truth_member and any(
            config.target_client_id in row.get("selected_clients", [c["client_id"] for c in row.get("clients", [])])
            for row in history)),
        "artifacts": artifacts,
    }

    if pipeline is not None:
        from ..rag import evaluate_pipeline
        trial["pipeline_evaluation"] = evaluate_pipeline(
            {"model": model, "tokenizer": tokenizer, "device": next(model.parameters()).device,
             "privacy": getattr(model, "_training_privacy", {}),
             "training_records": [text for part in client_texts for text in part]}, config, pipeline, trial_id=trial_id)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trial


def run_attack_trials(config, artifact_dir: Path, pipeline=None) -> list:
    from tqdm.auto import tqdm

    trials = []
    for trial_idx in tqdm(range(config.attack_trials), desc="LOSS membership trials"):
        truth_member = trial_idx % 2 == 0
        trials.append(run_attack_trial(config, trial_idx, truth_member, artifact_dir,
                                      **({"pipeline": pipeline} if pipeline is not None else {})))
    return trials


def _compute_metrics(trials):
    from ..metrics import base_metrics, scientific_metrics
    metrics = base_metrics(trials)
    metrics["balanced_accuracy"] = metrics["adv"]
    metrics["adv"] = (metrics["tpr"] + metrics["tnr"] - 1
                      if metrics["tpr"] is not None and metrics["tnr"] is not None else None)
    metrics.update(scientific_metrics([t["truth_member"] for t in trials], [-t["score"] for t in trials]))
    metrics["roc_auc_loss_inverted"] = metrics["roc_auc"]
    for label, name in ((True, "member"), (False, "nonmember")):
        losses = [t["score"] for t in trials if t["truth_member"] == label]
        metrics[f"{name}_mean_loss"] = sum(losses) / len(losses) if losses else None
    return metrics


def compact_trials(trials):
    # Preserve calibration/probability evidence; raw model tensors never enter trials.
    return [{key: value for key, value in trial.items()
             if key not in ("federated_history", "artifacts", "pipeline_evaluation")}
            for trial in trials]


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
    key_fn=loss_experiment_key,
    supports_toy=False,
    custom_trials=run_attack_trials,
    build_payload=build_result_payload,
)
