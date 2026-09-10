"""WBC (Window-Based Comparison) MIA. Ported from wbc_adaptations.ipynb.

Corrected methods use versioned cache identities; see docs/theory_corrections.md.
"""
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..config import AttackConfig
from ..federation import ToyFederatedLM, load_reference_bundle
from ..metrics import roc_auc
from ..scoring import ScoreContext
from ..spec import AttackSpec


@dataclass(frozen=True)
class WbcConfig(AttackConfig):
    attack_name: str = "wbc"
    paper_source: str = "Chen et al. 2026 (arXiv:2601.02751) Window-Based Comparison MIA; repo github.com/Stry233/WBC"
    model_id: str = "sshleifer/tiny-gpt2"
    reference_model_id: str = "sshleifer/tiny-gpt2"
    dataset_name: str = "synthetic_client_text"
    num_clients: int = 4
    clients_per_round: int = 4
    federated_rounds: int = 1
    local_epochs: int = 1
    local_batch_size: int = 2
    client_lr: float = 5e-5
    target_client_id: int = 0
    attack_trials: int = 4
    # Local v2 Appendix C.1.5 / author configuration, not Eq. (12)'s generated list.
    window_sizes: Tuple[int, ...] = (2, 3, 4, 6, 9, 13, 18, 25, 32, 40)
    threshold: float = 0.75
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/wbc_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "WBC (Window-Based Comparison): form per-token Delta_j = ref NLL - target NLL, slide "
        "multiple window sizes over it, each window casts a binary sign vote (sum > 0 => "
        "member), and rank by the fraction of member-voting windows ensembled over geometric "
        "window sizes. Sign vote beats a mean under heavy-tailed rare-token contamination."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, build Delta_j = reference-model NLL - final-FL-model NLL for the "
        "target record (reference = base model before FL) and score it with the WBC sign-vote "
        "fraction. Higher fraction => member."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; WBC membership score = fraction of windows with positive "
        "summed loss difference (members score higher)."
    ),
    "deviation_from_source": (
        "The paper's black-box reference/target attack is transferred to a federated "
        "fine-tuning threat model. The smoke run uses a deterministic toy causal scorer with an "
        "untrained reference; set use_hf_models=True for genuine federated fine-tuning of an "
        "open-source LLM with the Flower (flwr) FedAvg simulation and a pre-trained reference "
        "model."
    ),
}

# Notional vocabulary size so an UNTRAINED model is appropriately uncertain (high NLL ~
# log(BACKGROUND_VOCAB) on every token), mirroring a real pre-trained reference that has
# never seen the fine-tuning record. Without this, add-one smoothing over a tiny observed
# vocabulary would make an untrained reference look falsely confident and flip every delta
# negative.
_BACKGROUND_VOCAB = 50000


def _toy_token_nlls(model, text: str) -> List[float]:
    """Per-token negative log-likelihoods (add-one smoothed over BACKGROUND_VOCAB).

    Ported from ToyFederatedLM.token_nlls in wbc_adaptations.ipynb, applied to the shared
    core.federation.ToyFederatedLM's token_counts (that class doesn't expose this method,
    or use this vocabulary-size smoothing, itself)."""
    tokens = text.lower().split()
    total = sum(model.token_counts.values())
    vocab = _BACKGROUND_VOCAB
    return [-math.log((model.token_counts.get(token, 0.0) + 1.0) / (total + vocab)) for token in tokens]


def windowed_sums(deltas: Sequence[float], w: int) -> List[float]:
    """All contiguous window sums S_i(w) = sum_{j=i}^{i+w-1} Delta_j for one window size w."""
    n = len(deltas)
    if type(w) is not int or w <= 0 or not n:
        raise ValueError("WBC needs nonempty deltas and positive integer sizes")
    # Author-code short-record convention; retain each requested size's weight.
    w = min(w, n)
    return [float(sum(deltas[i:i + w])) for i in range(n - w + 1)]


def wbc_score(deltas: Sequence[float], window_sizes: Sequence[int]) -> float:
    """Eq. (13): uniform mean of per-size strictly-positive window fractions."""
    if not deltas or not window_sizes or not all(math.isfinite(d) for d in deltas):
        raise ValueError("WBC requires finite nonempty evidence and a window schedule")
    if len(set(window_sizes)) != len(window_sizes):
        raise ValueError("WBC requested sizes must be distinct")
    fractions = []
    for w in window_sizes:
        sums = windowed_sums(deltas, w)
        fractions.append(sum(s > 0 for s in sums) / len(sums))
    return sum(fractions) / len(fractions)


def build_deltas(reference_nll: Sequence[float], target_nll: Sequence[float]) -> List[float]:
    """Delta_j = reference NLL - target NLL, elementwise over aligned token positions."""
    if not reference_nll or len(reference_nll) != len(target_nll):
        raise ValueError("WBC requires nonempty aligned token losses")
    deltas = [float(r) - float(t) for r, t in zip(reference_nll, target_nll)]
    if not all(math.isfinite(d) for d in deltas):
        raise ValueError("Nonfinite WBC token loss")
    return deltas


def score_candidate_toy(target_model, reference_model, text: str, window_sizes: Sequence[int]) -> float:
    deltas = build_deltas(_toy_token_nlls(reference_model, text), _toy_token_nlls(target_model, text))
    return wbc_score(deltas, window_sizes)


def per_token_nll_hf(bundle, text: str, max_length: int = 64) -> List[float]:
    import torch

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_ll = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return [float(v) for v in (-token_ll[0]).detach().cpu().tolist()]


def score_candidate_hf(target_bundle, reference_bundle, text: str,
                       window_sizes: Sequence[int], max_length: int = 64) -> float:
    target_tok, ref_tok = target_bundle["tokenizer"], reference_bundle["tokenizer"]
    if target_tok.get_vocab() != ref_tok.get_vocab():
        raise ValueError("WBC requires compatible target/reference vocabularies")
    target_ids = target_tok(text, truncation=True, max_length=max_length)["input_ids"]
    reference_ids = ref_tok(text, truncation=True, max_length=max_length)["input_ids"]
    if target_ids != reference_ids or len(target_ids) < 2:
        raise ValueError("WBC requires identical scored token events")
    target_nll = per_token_nll_hf(target_bundle, text, max_length=max_length)
    reference_nll = per_token_nll_hf(reference_bundle, text, max_length=max_length)
    return wbc_score(build_deltas(reference_nll, target_nll), window_sizes)


def score_toy(ctx: ScoreContext) -> float:
    # The notebook's run_attack_trial always uses a fresh, untrained ToyFederatedLM as the
    # reference ("base model before FL"); ctx.reference lets a caller override it.
    reference = ctx.reference if ctx.reference is not None else ToyFederatedLM()
    return score_candidate_toy(ctx.target, reference, ctx.text, list(ctx.config.window_sizes))


def score_hf(ctx: ScoreContext) -> float:
    reference = ctx.reference if ctx.reference is not None else load_reference_bundle(ctx.config)
    return score_candidate_hf(
        ctx.target, reference, ctx.text,
        window_sizes=list(ctx.config.window_sizes), max_length=ctx.config.max_length,
    )


def _extra_metrics(trials):
    return {"roc_auc": roc_auc([t["truth_member"] for t in trials], [t["score"] for t in trials])}


SPEC = AttackSpec(
    name="wbc",
    config_cls=WbcConfig,
    methodology=METHODOLOGY,
    score_toy=score_toy,
    score_hf=score_hf,
    extra_metrics=_extra_metrics,
    needs_reference=True,
)
