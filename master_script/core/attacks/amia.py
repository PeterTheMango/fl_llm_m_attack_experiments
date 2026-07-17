"""AMI (activation-maximization-interval) probe MIA. Ported from AMIA_adaptation.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py. Uses the
24-char sha256[:24] key formula with default=str (key_sha24_default_str), NOT
the modern 16-char formula the other nine attacks share.
"""
from dataclasses import dataclass
from typing import Optional

from ..config import AttackConfig, key_sha24_default_str
from ..registry import AttackSpec


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

SPEC = AttackSpec(
    name="amia",
    config_cls=AmiaConfig,
    methodology=METHODOLOGY,
    key_fn=key_sha24_default_str,
    supports_toy=False,
)
