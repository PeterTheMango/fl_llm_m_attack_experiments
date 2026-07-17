"""Reference/comparison-model MIA. Ported from reference_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..registry import AttackSpec


@dataclass(frozen=True)
class ReferenceConfig(AttackConfig):
    attack_name: str = "reference"
    paper_source: str = "Carlini et al. 2021 reference/comparison-model MIA; LiRA-style LM calibration"
    model_id: str = "sshleifer/tiny-gpt2"
    dataset_name: str = "synthetic_client_text"
    num_clients: int = 4
    clients_per_round: int = 4
    federated_rounds: int = 1
    local_epochs: int = 1
    local_batch_size: int = 2
    client_lr: float = 5e-5
    target_client_id: int = 0
    attack_trials: int = 4
    threshold: float = 0.25
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/reference_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": "Reference/calibration MIA: compare target LM likelihood to a reference LM likelihood.",
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, score the target record under the final FL model versus the base "
        "reference model."
    ),
    "metric_definition": "Adv = 0.5 * TPR + 0.5 * TNR",
    "deviation_from_source": (
        "The smoke run uses a deterministic toy causal scorer. Set use_hf_models=True for "
        "genuine federated fine-tuning of an open-source LLM with the Flower (flwr) FedAvg "
        "simulation."
    ),
}

SPEC = AttackSpec(name="reference", config_cls=ReferenceConfig, methodology=METHODOLOGY)
