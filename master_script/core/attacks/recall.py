"""ReCaLL relative conditional log-likelihood MIA. Ported from recall_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..spec import AttackSpec


@dataclass(frozen=True)
class RecallConfig(AttackConfig):
    attack_name: str = "recall"
    paper_source: str = "Xie et al. 2024 (EMNLP, arXiv:2406.15968) ReCaLL relative conditional log-likelihood MIA"
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
    num_shots: int = 1
    threshold: float = 1.05
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/recall_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "ReCaLL: build a fixed prefix P from n non-member shots; score a target x by "
        "ReCaLL(x) = LL(x|P) / LL(x). Conditioning on the non-member prefix depresses the LL "
        "more for members, so members get higher ratios (typically > 1). Reference-model-free, "
        "inference-time only."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, score the target record under the final FL model only by computing "
        "LL(x) and LL(x|P) with a fixed non-member prefix P disjoint from every client corpus, "
        "then take the ratio. No reference model is used."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; membership score = LL(x|P) / LL(x) (ReCaLL ratio) so "
        "members score higher."
    ),
    "deviation_from_source": (
        "ReCaLL targets single-epoch pre-training; the paper notes multi-epoch fine-tuning is an "
        "easier MIA regime, so the signal transfers with equal or stronger effect. The smoke run "
        "uses a deterministic toy causal scorer whose conditional log-likelihood drops more for "
        "memorized (member) tokens; set use_hf_models=True for genuine federated fine-tuning of "
        "an open-source LLM with the Flower (flwr) FedAvg simulation."
    ),
}

SPEC = AttackSpec(name="recall", config_cls=RecallConfig, methodology=METHODOLOGY)
