"""Min-K% Prob reference-free MIA. Ported from min_k_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
import math
from dataclasses import dataclass
from typing import Sequence

from ..config import AttackConfig
from ..metrics import roc_auc
from ..scoring import ScoreContext
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

def _toy_token_prob(model, token: str) -> float:
    """Verbatim ToyFederatedLM._prob from min_k_adaptations.ipynb, applied to the
    shared core.federation.ToyFederatedLM's token_counts (that class doesn't expose
    token_logprobs itself, so the per-token computation is ported here instead)."""
    total = sum(model.token_counts.values()) + 1.0
    vocab = len(model.token_counts) + 1.0
    return (model.token_counts.get(token, 0.0) + 1.0) / (total + vocab)


def _toy_token_logprobs(model, text: str):
    """Per-token conditional log p(x_i | x_<i) for the text (one entry per token)."""
    tokens = text.lower().split()
    return [math.log(_toy_token_prob(model, token)) for token in tokens]


def min_k_membership_score(token_logprobs: Sequence[float], k_percent: int = 20) -> float:
    """Average of the K% lowest per-token log-probs. Higher (less negative) => member."""
    logps = list(token_logprobs)
    if not logps:
        return 0.0
    k = max(1, min(100, int(k_percent)))
    num_k = max(1, round(len(logps) * k / 100.0))
    lowest = sorted(logps)[:num_k]
    return sum(lowest) / len(lowest)


def score_candidate_toy(target_model, text: str, k_percent: int = 20) -> float:
    return min_k_membership_score(_toy_token_logprobs(target_model, text), k_percent)


def score_candidate_hf(target_bundle, text: str, k_percent: int = 20, max_length: int = 64) -> float:
    import torch

    model = target_bundle["model"]
    tokenizer = target_bundle["tokenizer"]
    device = target_bundle["device"]
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    # logits[:, t] predicts token t+1; align with the actual next tokens.
    log_probs = torch.log_softmax(outputs.logits[0, :-1, :], dim=-1)
    targets = input_ids[0, 1:]
    token_logprobs = log_probs[torch.arange(targets.shape[0]), targets].detach().cpu().tolist()
    return min_k_membership_score(token_logprobs, k_percent)


def score_toy(ctx: ScoreContext) -> float:
    return score_candidate_toy(ctx.target, ctx.text, k_percent=ctx.config.min_k_percent)


def score_hf(ctx: ScoreContext) -> float:
    return score_candidate_hf(
        ctx.target, ctx.text, k_percent=ctx.config.min_k_percent, max_length=ctx.config.max_length
    )


def _extra_metrics(trials):
    return {"roc_auc": roc_auc([t["truth_member"] for t in trials], [t["score"] for t in trials])}


SPEC = AttackSpec(
    name="min_k",
    config_cls=MinKConfig,
    methodology=METHODOLOGY,
    score_toy=score_toy,
    score_hf=score_hf,
    extra_metrics=_extra_metrics,
)
