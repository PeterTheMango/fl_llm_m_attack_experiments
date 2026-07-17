"""Min-K%++ reference-free MIA. Ported from min_k_plus_plus_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
import math
from dataclasses import dataclass
from typing import List, Sequence

from ..config import AttackConfig
from ..metrics import roc_auc
from ..scoring import ScoreContext
from ..spec import AttackSpec


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

# Memorized tokens are sharpened so a trained token is a clear mode of the pseudo-distribution.
_MEMORIZATION_TEMPERATURE = 8.0


def _log_softmax(logits: Sequence[float]) -> List[float]:
    values = [float(v) for v in logits]
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    log_total = math.log(sum(exps)) + m
    return [v - log_total for v in values]


def _toy_token_logprobs(model, text: str):
    """Return (per_token_logprobs, observed_indices) using a unigram pseudo-distribution.

    Ported from ToyFederatedLM.token_logprobs in min_k_plus_plus_adaptations.ipynb, applied
    to the shared core.federation.ToyFederatedLM's token_counts (that class doesn't expose
    this method itself)."""
    tokens = text.lower().split()
    vocab = sorted(set(model.token_counts) | set(tokens))
    index = {word: i for i, word in enumerate(vocab)}
    max_count = max([c for c in model.token_counts.values()] + [1.0])
    logits = [
        _MEMORIZATION_TEMPERATURE * (model.token_counts.get(word, 0.0) / max_count)
        for word in vocab
    ]
    log_probs = _log_softmax(logits)
    per_token = [log_probs for _ in tokens]
    observed = [index[word] for word in tokens]
    return per_token, observed


def token_logprob_stats(log_probs: Sequence[float]):
    """(mu_t, sigma_t) over a full-vocabulary log-softmax vector (pure Python)."""
    lp = [float(v) for v in log_probs]
    probs = [math.exp(v) for v in lp]
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    mu = sum(p * v for p, v in zip(probs, lp))
    second = sum(p * v * v for p, v in zip(probs, lp))
    var = second - mu * mu
    sigma = math.sqrt(var) if var > 0.0 else 0.0
    return mu, sigma


def token_zscore(log_probs: Sequence[float], observed_index: int) -> float:
    mu, sigma = token_logprob_stats(log_probs)
    if sigma == 0.0:
        return 0.0
    return (float(log_probs[observed_index]) - mu) / sigma


def min_k_plus_plus_membership_score(per_token_logprobs, observed_indices, min_k_percent: int = 20) -> float:
    """Mean of the K% SMALLEST per-token z-scores. Higher => member."""
    z_scores = [token_zscore(lp, int(idx)) for lp, idx in zip(per_token_logprobs, observed_indices)]
    if not z_scores:
        return float("nan")
    z_sorted = sorted(z_scores)
    k = max(1, int(len(z_sorted) * min_k_percent / 100))
    selected = z_sorted[:k]
    return sum(selected) / len(selected)


def score_candidate_toy(target_model, text: str, min_k_percent: int = 20) -> float:
    per_token_logprobs, observed_indices = _toy_token_logprobs(target_model, text)
    return min_k_plus_plus_membership_score(per_token_logprobs, observed_indices, min_k_percent=min_k_percent)


def score_candidate_hf(target_bundle, text: str, min_k_percent: int = 20, max_length: int = 64) -> float:
    import torch
    import torch.nn.functional as F

    model = target_bundle["model"]
    tokenizer = target_bundle["tokenizer"]
    device = target_bundle["device"]
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    with torch.no_grad():
        logits = model(**encoded).logits[0]           # (seq_len, vocab)
    log_probs = F.log_softmax(logits[:-1], dim=-1)     # predictions for positions 1..T
    targets = input_ids[0, 1:]                          # observed tokens x_1..x_T
    per_token_logprobs = [row.detach().cpu().tolist() for row in log_probs]
    observed_indices = targets.detach().cpu().tolist()
    return min_k_plus_plus_membership_score(per_token_logprobs, observed_indices, min_k_percent=min_k_percent)


def score_toy(ctx: ScoreContext) -> float:
    return score_candidate_toy(ctx.target, ctx.text, min_k_percent=ctx.config.min_k_percent)


def score_hf(ctx: ScoreContext) -> float:
    return score_candidate_hf(
        ctx.target, ctx.text, min_k_percent=ctx.config.min_k_percent, max_length=ctx.config.max_length
    )


def _extra_metrics(trials):
    return {"roc_auc": roc_auc([t["truth_member"] for t in trials], [t["score"] for t in trials])}


SPEC = AttackSpec(
    name="min_k_plus_plus",
    config_cls=MinKPlusPlusConfig,
    methodology=METHODOLOGY,
    score_toy=score_toy,
    score_hf=score_hf,
    extra_metrics=_extra_metrics,
)
