"""LOSS (Yeom et al. 2018) membership inference. Ported from LOSS_adaptation.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py. Uses the
named-prefix key formula (key_named_prefix): f"{experiment_name}_{digest16}",
NOT the modern bare 16-char formula the other nine attacks share.
"""
from dataclasses import dataclass
from typing import Optional

from ..config import AttackConfig, key_named_prefix
from ..registry import AttackSpec


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

SPEC = AttackSpec(
    name="loss",
    config_cls=LossConfig,
    methodology=METHODOLOGY,
    key_fn=key_named_prefix,
    supports_toy=False,
)
