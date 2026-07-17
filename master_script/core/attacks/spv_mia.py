"""SPV-MIA self-prompt calibration + probabilistic variation MIA. Ported from spv_mia_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
import math
import random
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from ..config import AttackConfig
from ..federation import ToyFederatedLM
from ..metrics import roc_auc
from ..scoring import ScoreContext
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

# Short public-domain chunks used to self-prompt the target model into D_self.
PUBLIC_PROMPTS = [
    "The quarterly report summarizes operational updates for every regional office.",
    "Customers may contact the support desk regarding invoices, shipping, and returns.",
    "This document describes general scheduling and follow-up procedures for clients.",
    "Standard reminders cover appointments, billing questions, and account access.",
]

MASK_TOKEN = "<mask>"


def _toy_prob(model, text: str) -> float:
    """Sequence likelihood proxy p_theta(x) = exp(-mean per-token NLL) in (0, 1].

    A memorized record (tokens with high counts) gets low NLL -> high prob; masking
    tokens to an unseen placeholder raises NLL -> lowers prob, so a memorized record
    is a LOCAL MAXIMUM relative to its paraphrases. Ported from ToyFederatedLM.prob
    in spv_mia_adaptations.ipynb; uses the shared core.federation.ToyFederatedLM's
    own nll() (identical formula to the notebook's)."""
    return math.exp(-model.nll(text))


def _toy_generate(model, prompt: str, num_tokens: int = 8) -> str:
    """Self-prompt generation: extend a public prompt with the model's most frequent
    tokens (a toy stand-in for autoregressive sampling from p_theta).

    Ties are broken lexicographically so the self-dataset is fully deterministic and
    independent of dict/set iteration order (Python string-hash randomization).
    Ported from ToyFederatedLM.generate; operates on the shared ToyFederatedLM's
    token_counts (that class doesn't expose this method itself)."""
    ranked = sorted(model.token_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [tok for tok, _ in ranked[:num_tokens]]
    return prompt if not top else prompt + " " + " ".join(top)


def _toy_paraphrase(text: str, num_paraphrases: int = 4, mask_ratio: float = 0.2, seed: int = 0):
    """Symmetric paraphrases (semantic domain): mask ~mask_ratio of tokens with an
    unseen placeholder. Pure-Python, deterministic given the seed."""
    rng = random.Random(seed)
    tokens = text.split()
    if not tokens:
        return [text for _ in range(num_paraphrases)]
    n_mask = max(1, int(len(tokens) * mask_ratio))
    variants = []
    for _ in range(num_paraphrases):
        toks = list(tokens)
        for idx in rng.sample(range(len(toks)), min(n_mask, len(toks))):
            toks[idx] = MASK_TOKEN
        variants.append(" ".join(toks))
    return variants


def build_self_prompt_reference_toy(config, target_model):
    """Module 1 in the toy path: prompt the FL-fine-tuned target model to generate a
    self-dataset D_self, then fit a second ToyFederatedLM (theta_dot) on it."""
    self_dataset = []
    for prompt in PUBLIC_PROMPTS:
        chunk = " ".join(prompt.split()[: config.self_prompt_tokens])
        self_dataset.append(_toy_generate(target_model, chunk, num_tokens=config.self_prompt_tokens))
    reference_model = ToyFederatedLM().fit(self_dataset, epochs=1)
    return reference_model, self_dataset


def probabilistic_variation(prob_x: float, paraphrase_probs: Sequence[float]) -> float:
    """Probabilistic-variation memorization signal for one model.

    Paper Eq. 10 defines p_tilde(x) = mean(paraphrase_probs) - prob_x (negative for a local
    maximum / memorized record). We return the sign-flipped signal prob_x -
    mean(paraphrase_probs) = -p_tilde(x) so that HIGHER => member, matching the project
    convention. Orientation only; ranking/AUC unchanged.
    """
    if not paraphrase_probs:
        return 0.0
    return float(prob_x) - float(mean(paraphrase_probs))


def spv_membership_score(pv_theta: float, pv_theta_dot: float) -> float:
    """Self-calibrated SPV score (paper Eq. 5, flipped orientation). Higher => member."""
    return float(pv_theta) - float(pv_theta_dot)


def score_candidate_toy(target_model, reference_model, text: str, config) -> float:
    # Same paraphrase set scored under both models (paraphrasing is model-independent).
    paraphrases = _toy_paraphrase(
        text, num_paraphrases=config.num_paraphrases, mask_ratio=config.mask_ratio, seed=config.seed
    )
    pv_theta = probabilistic_variation(
        _toy_prob(target_model, text), [_toy_prob(target_model, p) for p in paraphrases]
    )
    pv_theta_dot = probabilistic_variation(
        _toy_prob(reference_model, text), [_toy_prob(reference_model, p) for p in paraphrases]
    )
    return spv_membership_score(pv_theta, pv_theta_dot)


def _mean_nll_hf(bundle, text, max_length=64):
    import torch

    model, tokenizer, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, labels=encoded["input_ids"])
    return float(outputs.loss.detach().cpu())


def _prob_hf(bundle, text, max_length=64):
    # Probability proxy exp(-mean per-token NLL) in (0, 1], consistent with the toy prob().
    return math.exp(-_mean_nll_hf(bundle, text, max_length=max_length))


def hf_paraphrase(text, num_paraphrases=4, mask_ratio=0.2, seed=0):
    """Pure-Python fallback paraphraser for the HF path (mask ~mask_ratio of tokens).

    Swap this for a real T5 mask-and-reconstruct paraphraser (paper default) when running
    at scale; the SPV score only needs a consistent set of symmetric paraphrases.
    """
    rng = random.Random(seed)
    tokens = text.split()
    if not tokens:
        return [text for _ in range(num_paraphrases)]
    n_mask = max(1, int(len(tokens) * mask_ratio))
    variants = []
    for _ in range(num_paraphrases):
        toks = list(tokens)
        for idx in rng.sample(range(len(toks)), min(n_mask, len(toks))):
            toks[idx] = "<mask>"
        variants.append(" ".join(toks))
    return variants


def build_self_prompt_reference_hf(config, target_bundle):
    """Module 1 for the HF path: self-prompt the fine-tuned target to build D_self, then
    fine-tune a fresh base model on it to obtain theta_dot."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader, TensorDataset

    model, tokenizer, device = target_bundle["model"], target_bundle["tokenizer"], target_bundle["device"]
    self_dataset = []
    for prompt in PUBLIC_PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=config.self_prompt_tokens)["input_ids"].to(device)
        out = model.generate(ids, max_new_tokens=config.max_length, do_sample=True,
                             top_k=50, pad_token_id=tokenizer.eos_token_id)
        self_dataset.append(tokenizer.decode(out[0], skip_special_tokens=True))

    ref_tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_id).to(device)
    encoded = ref_tokenizer(self_dataset, padding=True, truncation=True,
                            max_length=config.max_length, return_tensors="pt")
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    loader = DataLoader(dataset, batch_size=config.local_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(ref_model.parameters(), lr=config.client_lr)
    ref_model.train()
    for _ in range(config.local_epochs):
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    ref_model.eval()
    return {"model": ref_model, "tokenizer": ref_tokenizer, "device": device}


def score_candidate_hf(target_bundle, reference_bundle, text: str, config) -> float:
    paraphrases = hf_paraphrase(
        text, num_paraphrases=config.num_paraphrases, mask_ratio=config.mask_ratio, seed=config.seed
    )
    pv_theta = probabilistic_variation(
        _prob_hf(target_bundle, text, config.max_length),
        [_prob_hf(target_bundle, p, config.max_length) for p in paraphrases],
    )
    pv_theta_dot = probabilistic_variation(
        _prob_hf(reference_bundle, text, config.max_length),
        [_prob_hf(reference_bundle, p, config.max_length) for p in paraphrases],
    )
    return spv_membership_score(pv_theta, pv_theta_dot)


def score_toy(ctx: ScoreContext) -> float:
    # The notebook always derives theta_dot by self-prompting the target model
    # (build_self_prompt_reference_toy); ctx.reference lets a caller override it.
    if ctx.reference is not None:
        reference = ctx.reference
    else:
        reference, _ = build_self_prompt_reference_toy(ctx.config, ctx.target)
    return score_candidate_toy(ctx.target, reference, ctx.text, ctx.config)


def score_hf(ctx: ScoreContext) -> float:
    reference = ctx.reference if ctx.reference is not None else build_self_prompt_reference_hf(ctx.config, ctx.target)
    return score_candidate_hf(ctx.target, reference, ctx.text, ctx.config)


def _extra_metrics(trials):
    return {"roc_auc": roc_auc([t["truth_member"] for t in trials], [t["score"] for t in trials])}


SPEC = AttackSpec(
    name="spv_mia",
    config_cls=SpvMiaConfig,
    methodology=METHODOLOGY,
    score_toy=score_toy,
    score_hf=score_hf,
    extra_metrics=_extra_metrics,
    needs_reference=True,
)
