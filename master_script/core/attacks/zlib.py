"""zlib-entropy ratio MIA. Ported from zlib_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..registry import AttackSpec


@dataclass(frozen=True)
class ZlibConfig(AttackConfig):
    attack_name: str = "zlib"
    paper_source: str = "Carlini et al. 2021 (arXiv:2012.07805) zlib-entropy ratio MIA"
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
    threshold: float = -0.0058
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/zlib_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "zlib-entropy ratio MIA: rank candidates by log_perplexity / len(zlib.compress(text)); "
        "zlib is a model-independent reference that down-ranks repetitive/boilerplate text."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, score the target record under the final FL model only, dividing its "
        "log-perplexity by the record's zlib entropy. No reference model is used."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; membership score = -(log_perplexity / zlib_entropy_bits) "
        "so members score higher."
    ),
    "deviation_from_source": (
        "Carlini et al. flag fine-tuning as future work, so applying the zlib ratio to a "
        "fine-tuned model is a transfer of their pre-training signal. The smoke run uses a "
        "deterministic toy causal scorer; set use_hf_models=True for genuine federated "
        "fine-tuning of an open-source LLM with the Flower (flwr) FedAvg simulation."
    ),
}

SPEC = AttackSpec(name="zlib", config_cls=ZlibConfig, methodology=METHODOLOGY)
