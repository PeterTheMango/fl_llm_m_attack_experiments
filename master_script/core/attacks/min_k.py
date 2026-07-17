"""Min-K% Prob reference-free MIA. Ported from min_k_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..spec import AttackSpec


@dataclass(frozen=True)
class MinKConfig(AttackConfig):
    attack_name: str = "min_k"
    paper_source: str = "Shi et al. 2024 (ICLR / arXiv:2310.16789) Min-K% Prob reference-free MIA"
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
    min_k_percent: int = 20
    threshold: float = -4.08
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/min_k_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "Min-K% Prob (Shi et al. 2024): compute per-token log p(x_i | x_<i), select the k% "
        "lowest-probability tokens, and average their log-likelihood. Reference-free; a member "
        "has few very low-probability tokens so its Min-K% average stays high."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, score the target record under the final FL model only by averaging "
        "the k% lowest per-token log-probs. No reference model is used."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; membership score = mean of the k% lowest per-token "
        "log-probs so members score higher. AUC is the paper's threshold-free headline metric."
    ),
    "deviation_from_source": (
        "Shi et al. Section 6 fine-tunes LLaMA-7B for one epoch and detects contaminated "
        "examples with Min-K% Prob (AUC 0.91), so applying Min-K% to a federated fine-tuned "
        "model is a direct transfer. The smoke run uses a deterministic toy causal scorer "
        "exposing per-token log-probs; set use_hf_models=True for genuine federated "
        "fine-tuning of an open-source LLM with the Flower (flwr) FedAvg simulation."
    ),
}

SPEC = AttackSpec(name="min_k", config_cls=MinKConfig, methodology=METHODOLOGY)
