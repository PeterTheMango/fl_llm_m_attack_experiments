"""ReCaLL relative conditional log-likelihood MIA. Ported from recall_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
import math
from dataclasses import dataclass
from typing import Optional

from ..config import AttackConfig
from ..metrics import roc_auc
from ..scoring import ScoreContext
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

# Fixed non-member prefix shots (disjoint from every client corpus in the toy world). Each
# is a guaranteed non-member of the fine-tuning set. The prefix P is fixed across all targets
# and identical in the positive and negative worlds.
NON_MEMBER_SHOTS = [
    "Orbital weather bulletin 2031: aurora indices held steady while the polar relay buoys logged calm seas.",
    "Fictional almanac entry: the Zephyr Archipelago hosts its biennial kite-glass regatta every leap autumn.",
    "Synthetic recipe draft: fold candied yuzu peel into buckwheat batter and rest the crepe overnight.",
    "Imaginary transit note: the violet monorail now loops the museum quarter at quarter-past intervals.",
]


def build_prefix(config) -> str:
    """Concatenate the first `num_shots` non-member shots into the fixed prefix P."""
    n = max(1, int(config.num_shots))
    return " ".join(NON_MEMBER_SHOTS[:n])


# How strongly a non-member prefix perturbs the predictive distribution.
_PREFIX_PERTURBATION = 0.9


def _toy_loglik(model, text: str, prefix: Optional[str] = None) -> float:
    """Total log-likelihood of `text` (negative). If `prefix` is given, return the
    conditional LL(text | prefix): the fixed non-member prefix perturbs memorized
    tokens more strongly, so member records drop further.

    Ported from ToyFederatedLM.loglik in recall_adaptations.ipynb, applied to the
    shared core.federation.ToyFederatedLM's token_counts (that class doesn't expose
    this method itself)."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    total = sum(model.token_counts.values()) + 1.0
    vocab = len(model.token_counts) + 1.0
    alpha = _PREFIX_PERTURBATION if prefix else 0.0
    ll = 0.0
    for token in tokens:
        count = model.token_counts.get(token, 0.0)
        prob = (count + 1.0) / (total + vocab)
        if alpha > 0.0:
            # Memorization strength: ~0 for unseen tokens, -> 1 for well-memorized
            # (high-count) tokens. Conditioning on the non-member prefix depresses
            # a token's probability in proportion to how memorized it is.
            strength = count / (count + 1.0)
            prob = prob * (1.0 - alpha * strength)
        ll += math.log(prob)
    return ll  # negative


def recall_membership_score(ll_x: float, ll_x_given_prefix: float) -> float:
    """ReCaLL membership score = LL(x | P) / LL(x). Both negative. Higher => member."""
    if ll_x == 0.0:
        raise ValueError("LL(x) must be non-zero to form the ReCaLL ratio.")
    return ll_x_given_prefix / ll_x


def score_candidate_toy(target_model, text: str, prefix: str) -> float:
    ll_x = _toy_loglik(target_model, text)
    ll_x_given_prefix = _toy_loglik(target_model, text, prefix=prefix)
    return recall_membership_score(ll_x, ll_x_given_prefix)


def _sequence_loglik_hf(model, tokenizer, text: str, prefix: Optional[str], device: str, max_length: int) -> float:
    """Total log-likelihood of `text` (negative), optionally conditioned on `prefix`.

    Prefix tokens are prepended but excluded from the loss (labels=-100), so the
    returned value is LL(text | prefix) when a prefix is given, else LL(text).
    """
    import torch

    if prefix:
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
        text_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.cat([prefix_ids, text_ids], dim=1)[:, :max_length]
        labels = input_ids.clone()
        labels[:, : prefix_ids.shape[1]] = -100
    else:
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        labels = input_ids.clone()
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    shift_labels = labels[:, 1:]
    num_scored = int((shift_labels != -100).sum().item())
    if num_scored == 0:
        raise ValueError("Need at least one scored text token.")
    total_nll = float(outputs.loss.detach().cpu()) * num_scored
    return -total_nll


def score_candidate_hf(target_bundle, text: str, prefix: str, max_length: int = 64) -> float:
    model = target_bundle["model"]
    tokenizer = target_bundle["tokenizer"]
    device = target_bundle["device"]
    ll_x = _sequence_loglik_hf(model, tokenizer, text, None, device, max_length)
    ll_x_given_prefix = _sequence_loglik_hf(model, tokenizer, text, prefix, device, max_length)
    return recall_membership_score(ll_x, ll_x_given_prefix)


def score_toy(ctx: ScoreContext) -> float:
    return score_candidate_toy(ctx.target, ctx.text, build_prefix(ctx.config))


def score_hf(ctx: ScoreContext) -> float:
    return score_candidate_hf(ctx.target, ctx.text, build_prefix(ctx.config), max_length=ctx.config.max_length)


def _extra_metrics(trials):
    return {"roc_auc": roc_auc([t["truth_member"] for t in trials], [t["score"] for t in trials])}


SPEC = AttackSpec(
    name="recall",
    config_cls=RecallConfig,
    methodology=METHODOLOGY,
    score_toy=score_toy,
    score_hf=score_hf,
    extra_metrics=_extra_metrics,
)
