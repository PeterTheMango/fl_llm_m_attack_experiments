"""WBC (Window-Based Comparison) MIA. Ported from wbc_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass
from typing import Tuple

from ..config import AttackConfig
from ..spec import AttackSpec


@dataclass(frozen=True)
class WbcConfig(AttackConfig):
    attack_name: str = "wbc"
    paper_source: str = "Chen et al. 2026 (arXiv:2601.02751) Window-Based Comparison MIA; repo github.com/Stry233/WBC"
    model_id: str = "sshleifer/tiny-gpt2"
    reference_model_id: str = "sshleifer/tiny-gpt2"
    dataset_name: str = "synthetic_client_text"
    num_clients: int = 4
    clients_per_round: int = 4
    federated_rounds: int = 1
    local_epochs: int = 1
    local_batch_size: int = 2
    client_lr: float = 5e-5
    target_client_id: int = 0
    attack_trials: int = 4
    # Tuple keeps the frozen dataclass hashable; json.dumps serializes it as a
    # JSON array, matching the notebook's config_to_storage() list conversion.
    window_sizes: Tuple[int, ...] = (2, 3, 4, 6, 9, 13, 18, 25, 32, 40)
    threshold: float = 0.75
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/wbc_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "WBC (Window-Based Comparison): form per-token Delta_j = ref NLL - target NLL, slide "
        "multiple window sizes over it, each window casts a binary sign vote (sum > 0 => "
        "member), and rank by the fraction of member-voting windows ensembled over geometric "
        "window sizes. Sign vote beats a mean under heavy-tailed rare-token contamination."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, build Delta_j = reference-model NLL - final-FL-model NLL for the "
        "target record (reference = base model before FL) and score it with the WBC sign-vote "
        "fraction. Higher fraction => member."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; WBC membership score = fraction of windows with positive "
        "summed loss difference (members score higher)."
    ),
    "deviation_from_source": (
        "The paper's black-box reference/target attack is transferred to a federated "
        "fine-tuning threat model. The smoke run uses a deterministic toy causal scorer with an "
        "untrained reference; set use_hf_models=True for genuine federated fine-tuning of an "
        "open-source LLM with the Flower (flwr) FedAvg simulation and a pre-trained reference "
        "model."
    ),
}

SPEC = AttackSpec(name="wbc", config_cls=WbcConfig, methodology=METHODOLOGY)
