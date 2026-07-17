"""SaMIA sampling-based pseudo-likelihood MIA. Ported from samia_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..registry import AttackSpec


@dataclass(frozen=True)
class SamiaConfig(AttackConfig):
    attack_name: str = "samia"
    paper_source: str = "Kaneko et al. 2024 (arXiv:2404.11262) SaMIA sampling-based pseudo-likelihood MIA"
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
    num_samples: int = 10
    rouge_n: int = 1
    use_zlib_weighting: bool = False
    threshold: float = 0.5
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/samia_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "SaMIA (Kaneko et al., 2024): split the target text into a prefix and a reference "
        "suffix, sample m continuations from the model given the prefix, and score membership by "
        "the mean ROUGE-N recall of the candidates against the true suffix; the SaMIA x zlib "
        "variant weights each candidate by its zlib bit length. Fully black-box (generation only, "
        "no likelihoods)."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by whether the target record is in the target "
        "client's local data. After Flower (flwr) FedAvg fine-tuning, prompt the final FL model "
        "with the target record's prefix, sample num_samples continuations, and score the mean "
        "ROUGE-N recall against the held-out suffix. No reference model is used (black-box)."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; membership score = mean ROUGE-N recall (optionally "
        "zlib-weighted) so members score higher."
    ),
    "deviation_from_source": (
        "SaMIA is defined for pre-trained LLMs on WikiMIA; here it is transferred to federated "
        "fine-tuning, attacking a memorized fine-tuning record via black-box generation. The "
        "smoke run uses a deterministic toy transition model whose generation reproduces the "
        "suffix only when the target record was in training; set use_hf_models=True for genuine "
        "federated fine-tuning of an open-source LLM with the Flower (flwr) FedAvg simulation and "
        "true sampling."
    ),
}

SPEC = AttackSpec(name="samia", config_cls=SamiaConfig, methodology=METHODOLOGY)
