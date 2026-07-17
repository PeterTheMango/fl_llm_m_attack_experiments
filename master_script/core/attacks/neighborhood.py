"""Neighbourhood comparison MIA. Ported from neighborhood_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..spec import AttackSpec


@dataclass(frozen=True)
class NeighborhoodConfig(AttackConfig):
    attack_name: str = "neighborhood"
    paper_source: str = "Mattern et al. 2023 (arXiv:2305.18462) Neighbourhood Comparison MIA"
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
    num_neighbours: int = 25
    neighbour_swaps: int = 1
    threshold: float = 0.02
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/neighborhood_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "Neighbourhood comparison MIA: for target x, generate n near-identical neighbours via "
        "BERT masked-LM single-word (m=1) substitution, then predict member when L(x) is "
        "substantially below the mean neighbour loss (Eq. 3). Reference-model-free: neighbours "
        "replace the reference model."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, score the target record under the final FL model only by generating "
        "neighbours and taking mean_neighbour_loss - target_loss. No reference model is used."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; membership score = mean_neighbour_loss - target_loss so "
        "members (loss below neighbours) score higher."
    ),
    "deviation_from_source": (
        "The attack was designed for the fine-tuning setting, so applying it to a federated "
        "fine-tuned model is a direct transfer. No reference model is needed: the neighbours are "
        "the difficulty calibration. The smoke run uses a deterministic toy causal scorer and "
        "toy single-word neighbour perturbations; set use_hf_models=True for genuine federated "
        "fine-tuning of an open-source LLM (Flower FedAvg) with BERT (bert-base-uncased) "
        "neighbour generation following Algorithm 1 (n=100, m=1, dropout p=0.7 on the original "
        "embedding)."
    ),
}

SPEC = AttackSpec(name="neighborhood", config_cls=NeighborhoodConfig, methodology=METHODOLOGY)
