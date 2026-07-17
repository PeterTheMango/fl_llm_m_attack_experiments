"""SPV-MIA self-prompt calibration + probabilistic variation MIA. Ported from spv_mia_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..spec import AttackSpec


@dataclass(frozen=True)
class SpvMiaConfig(AttackConfig):
    attack_name: str = "spv_mia"
    paper_source: str = "Fu et al. 2024 (NeurIPS, arXiv:2311.06062) SPV-MIA: self-prompt calibration + probabilistic variation"
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
    threshold: float = 0.0
    max_length: int = 64
    num_paraphrases: int = 4
    mask_ratio: float = 0.2
    self_prompt_tokens: int = 8
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/spv_mia_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "SPV-MIA: (1) self-prompt reference model theta_dot fine-tuned on text the target LLM "
        "itself generates from short public prompts (practical difficulty calibration); (2) "
        "probabilistic variation assessment detecting whether a record is a local maximum of the "
        "model's probability via symmetric paraphrases. Decision A = 1[p_tilde_theta(x) - "
        "p_tilde_theta_dot(x) >= tau]."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation produces theta, the self-prompt reference theta_dot is bootstrapped by "
        "prompting theta to generate a self-dataset and fitting a second model on it. The target "
        "record's probabilistic variation (prob(x) - mean(prob(paraphrases))) is measured under "
        "theta and theta_dot and combined as pv_theta - pv_theta_dot."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; primary paper metric AUC (roc_auc). "
        "spv_membership_score = pv_theta - pv_theta_dot with pv = prob_x - "
        "mean(paraphrase_probs) = -p_tilde (paper Eq. 10 sign flipped so higher => member)."
    ),
    "deviation_from_source": (
        "SPV-MIA already targets the fine-tuning phase; this transfers it to FEDERATED "
        "fine-tuning. The smoke run uses a deterministic toy causal scorer whose "
        "prob/generate/paraphrase mimic memorization and self-prompting; set use_hf_models=True "
        "for genuine federated fine-tuning of an open-source LLM (Flower FedAvg), a self-prompt "
        "reference fine-tuned on the target's generations, and (optionally) a T5 "
        "mask-and-reconstruct paraphraser in place of the pure-Python token masking."
    ),
}

SPEC = AttackSpec(name="spv_mia", config_cls=SpvMiaConfig, methodology=METHODOLOGY)
