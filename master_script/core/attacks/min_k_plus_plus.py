"""Min-K%++ reference-free MIA. Ported from min_k_plus_plus_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..registry import AttackSpec


@dataclass(frozen=True)
class MinKPlusPlusConfig(AttackConfig):
    attack_name: str = "min_k_plus_plus"
    paper_source: str = "Zhang et al. 2025 (ICLR / arXiv:2404.02936) Min-K%++ reference-free MIA"
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
    threshold: float = -3.25
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/min_k_plus_plus_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "Min-K%++ reference-free MIA: per token z_t = (log p(x_t|x_<t) - mu_t) / sigma_t with "
        "mu_t = sum_z p(z) log p(z) and sigma_t = sqrt(sum_z p(z) log p(z)^2 - mu_t^2) over the "
        "full softmax; score = mean of the K% smallest z_t; higher => member. Grey-box (needs "
        "full logits)."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, run the target record through the final FL model only, read the "
        "full per-position vocabulary distribution, compute the per-token z-scores, and average "
        "the K% smallest. Reference-free: no second model."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; membership score = mean of K% smallest per-token z-scores "
        "so members (whose tokens are modes) score higher."
    ),
    "deviation_from_source": (
        "The Min-K%++ paper (Zhang et al., ICLR 2025) targets pre-training-data detection and "
        "runs no fine-tuning MIA; this notebook transfers the logit-z-score statistic to a "
        "fine-tuned FL model, where multi-epoch memorization should make member tokens stronger "
        "modes. The smoke run uses a deterministic toy causal scorer that emits per-token "
        "pseudo-distributions; set use_hf_models=True for genuine federated fine-tuning of an "
        "open-source LLM with the Flower (flwr) FedAvg simulation and real log-softmax logits."
    ),
}

SPEC = AttackSpec(name="min_k_plus_plus", config_cls=MinKPlusPlusConfig, methodology=METHODOLOGY)
