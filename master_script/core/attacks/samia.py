"""SaMIA sampling-based pseudo-likelihood MIA. Ported from samia_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
import zlib as _zlib
from collections import Counter
from dataclasses import dataclass

from ..config import AttackConfig
from ..metrics import roc_auc, tpr_at_fpr
from ..scoring import ScoreContext
from ..spec import AttackSpec


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

def zlib_bits(text):
    """Bits in the zlib-compressed text (SaMIA x zlib weight term)."""
    return 8.0 * len(_zlib.compress(text.encode("utf-8")))


def _tokenize(text):
    return text.lower().split()


def split_prefix_suffix(text):
    """Split target text into prefix (first half) and reference suffix (second half)."""
    tokens = text.split()
    mid = len(tokens) // 2
    return " ".join(tokens[:mid]), " ".join(tokens[mid:])


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_recall(candidate_tokens, reference_tokens, n=1):
    """ROUGE-N recall via collections.Counter (stdlib only; no `rouge` package)."""
    reference = Counter(_ngrams(reference_tokens, n))
    candidate = Counter(_ngrams(candidate_tokens, n))
    denominator = sum(reference.values())
    if denominator == 0:
        return 0.0
    matches = sum(min(count, candidate[gram]) for gram, count in reference.items())
    return matches / denominator


def samia_membership_score(candidates, reference_suffix, rouge_n=1, use_zlib=False):
    """Mean ROUGE-N recall of m sampled candidates vs the true suffix (Eq. 5). With
    use_zlib=True, weight each candidate by its zlib bit length (Eq. 7). Higher => member."""
    if not candidates:
        return 0.0
    reference_tokens = _tokenize(reference_suffix)
    total = 0.0
    for candidate in candidates:
        recall = rouge_n_recall(_tokenize(candidate), reference_tokens, n=rouge_n)
        if use_zlib:
            recall = recall * zlib_bits(candidate)
        total += recall
    return total / len(candidates)


def _toy_generate(model, prefix, max_new_tokens):
    """Deterministic stand-in for ToyFederatedLM.generate (samia_adaptations.ipynb's toy
    model tracks bigram transitions; the shared core.federation.ToyFederatedLM only tracks
    unigram token_counts). Ranks vocabulary by learned count (ties broken lexicographically,
    matching the notebook's tie-break rule) and drops words already in the prefix, so a
    target-trained (member) model surfaces the memorized suffix words while a non-member
    model -- which never saw them -- cannot."""
    prefix_tokens = set(_tokenize(prefix))
    ranked = sorted(model.token_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    tokens = [tok for tok, _ in ranked if tok not in prefix_tokens][:max_new_tokens]
    return " ".join(tokens)


def sample_continuations_hf(target_bundle, prefix, config):
    """Sample config.num_samples continuations from the fine-tuned HF model (black-box)."""
    import torch

    model = target_bundle["model"]
    tokenizer = target_bundle["tokenizer"]
    device = target_bundle["device"]
    encoded = tokenizer(prefix, return_tensors="pt").to(device)
    prompt_len = encoded["input_ids"].shape[-1]
    continuations = []
    for _ in range(config.num_samples):
        with torch.no_grad():
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                max_new_tokens=config.max_length,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        continuations.append(tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True))
    return continuations


def score_candidate_toy(target_model, text, config):
    prefix, suffix = split_prefix_suffix(text)
    max_new = len(suffix.split()) + 8
    candidates = [_toy_generate(target_model, prefix, max_new) for _ in range(config.num_samples)]
    return samia_membership_score(candidates, suffix, rouge_n=config.rouge_n, use_zlib=config.use_zlib_weighting)


def score_candidate_hf(target_bundle, text, config):
    prefix, suffix = split_prefix_suffix(text)
    candidates = sample_continuations_hf(target_bundle, prefix, config)
    return samia_membership_score(candidates, suffix, rouge_n=config.rouge_n, use_zlib=config.use_zlib_weighting)


def score_toy(ctx: ScoreContext) -> float:
    return score_candidate_toy(ctx.target, ctx.text, ctx.config)


def score_hf(ctx: ScoreContext) -> float:
    return score_candidate_hf(ctx.target, ctx.text, ctx.config)


# The notebook's summarize_trials calls tpr_at_fpr(labels, scores, target_fpr=0.1) and
# stores it as "tpr_at_10fpr"; preserve the 0.1 threshold under the shared key name.
_SAMIA_TARGET_FPR = 0.1


def _extra_metrics(trials):
    labels = [t["truth_member"] for t in trials]
    scores = [t["score"] for t in trials]
    return {
        "roc_auc": roc_auc(labels, scores),
        "tpr_at_10fpr": tpr_at_fpr(labels, scores, target_fpr=_SAMIA_TARGET_FPR),
    }


SPEC = AttackSpec(
    name="samia",
    config_cls=SamiaConfig,
    methodology=METHODOLOGY,
    score_toy=score_toy,
    score_hf=score_hf,
    extra_metrics=_extra_metrics,
)
