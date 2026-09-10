"""SPV-MIA self-prompt calibration + probabilistic variation MIA. Ported from spv_mia_adaptations.ipynb.

Corrected methods use versioned cache identities; see docs/theory_corrections.md.
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
    num_paraphrases: int = 10
    mask_ratio: float = 0.2
    self_prompt_tokens: int = 8
    embedding_noise_scale: float = 0.05
    reference_samples: int = 32
    reference_epochs: int = 4
    reference_batch_size: int = 16
    reference_lr: float = 1e-4
    reference_generation_length: int = 128
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/spv_mia_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": "Fu Eqs. (5), (10): target probabilistic variation minus self-prompt reference variation.",
    "llm_adaptation": "FL target; fresh base reference trained on target generations; Appendix A.3 Algorithm 1 symmetric embedding perturbations.",
    "metric_definition": "Joint sequence probabilities combined before an increasing signed-log rank transform; threshold zero preserves Eq. (5).",
    "attacker_observation": "Local model embedding and logit access for the paper's embedding-domain variant.",
    "deviation_from_source": "FL adaptation uses a small public self-prompt corpus by default, not the 10,000-record benchmark; embedding variant rather than the semantic-default experiment.",
    "source_sign_note": "Printed Eq. (5) >= is retained despite its local-maximum interpretation tension; orientation is never chosen from test AUC.",
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
    """Joint probability under the toy unigram model; no HF parity claim."""
    return math.exp(-model.nll(text) * len(text.split()))


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
    """Toy-only perturbations; no claim of semantic or embedding-domain symmetry."""
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
    """Fu Eq. (10), including its printed sign."""
    values = [float(prob_x), *map(float, paraphrase_probs)]
    if not paraphrase_probs or not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
        raise ValueError("Probabilistic variation requires finite probability evidence")
    return math.fsum(paraphrase_probs) / len(paraphrase_probs) - prob_x


def spv_score_from_logprobs(target_logp, target_pairs, reference_logp, reference_pairs):
    """Combine JOINT probabilities in signed log space, then rank monotonically.

    sign(s)*(1/2+atan(log(abs(s)))/pi) is increasing in the final raw score s,
    preserves zero/sign, and avoids underflow of long sequence probabilities.
    It is not applied separately to terms of the variation calculation.
    """
    if not target_pairs or len(target_pairs) != len(reference_pairs) or len(target_pairs) % 2:
        raise ValueError("SPV requires matching nonempty +/- pairs")
    weight = math.log(len(target_pairs))
    terms = [(-1, target_logp), (1, reference_logp)]
    terms += [(1, value - weight) for value in target_pairs]
    terms += [(-1, value - weight) for value in reference_pairs]
    if not all(math.isfinite(value) and value <= 0 for _, value in terms):
        raise ValueError("SPV requires finite joint log probabilities")
    scale = max(value for _, value in terms)
    residual = math.fsum(sign * math.exp(value - scale) for sign, value in terms)
    if residual == 0:
        return 0.0
    log_magnitude = scale + math.log(abs(residual))
    return math.copysign(0.5 + math.atan(log_magnitude) / math.pi, residual)


def spv_membership_score(pv_theta: float, pv_theta_dot: float) -> float:
    """Self-calibrated SPV score (paper Eq. 5, printed orientation). Higher => member."""
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


def _joint_logprob(bundle, input_ids, inputs_embeds=None):
    import torch
    if input_ids.shape[-1] < 2:
        raise ValueError("SPV requires at least two tokens")
    with torch.no_grad():
        kwargs = {"input_ids": input_ids} if inputs_embeds is None else {"inputs_embeds": inputs_embeds}
        logits = bundle["model"](**kwargs, attention_mask=torch.ones_like(input_ids)).logits
        logp = torch.log_softmax(logits[:, :-1].double(), dim=-1)
        return float(logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).sum())


def _prob_hf(bundle, text, max_length=64):
    ids = bundle["tokenizer"](text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].to(bundle["device"])
    value = math.exp(_joint_logprob(bundle, ids))
    if value == 0:
        raise FloatingPointError("Use signed-log SPV scoring for underflowing joint probabilities")
    return value


def build_self_prompt_reference_hf(config, target_bundle):
    """Module 1 for the HF path: self-prompt the fine-tuned target to build D_self, then
    fine-tune a fresh base model on it to obtain theta_dot."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader, TensorDataset

    model, tokenizer, device = target_bundle["model"], target_bundle["tokenizer"], target_bundle["device"]
    from ..federation import seed_training
    seed_training(config.seed)
    capacity = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", None))
    if capacity is not None and config.reference_generation_length > capacity:
        raise ValueError("SPV reference generation exceeds model context capacity")
    self_dataset = []
    for index in range(config.reference_samples):
        prompt = PUBLIC_PROMPTS[index % len(PUBLIC_PROMPTS)]
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=config.self_prompt_tokens)["input_ids"].to(device)
        if ids.shape[-1] >= config.reference_generation_length:
            raise ValueError("SPV reference prompt leaves no continuation budget")
        out = model.generate(ids, max_new_tokens=config.reference_generation_length - ids.shape[-1], do_sample=True,
                             top_k=50, pad_token_id=tokenizer.eos_token_id)
        self_dataset.append(tokenizer.decode(out[0], skip_special_tokens=True))

    ref_tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision).to(device)
    encoded = ref_tokenizer(self_dataset, padding=True, truncation=True,
                            max_length=config.reference_generation_length, return_tensors="pt")
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    loader = DataLoader(dataset, batch_size=config.reference_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(ref_model.parameters(), lr=config.reference_lr)
    ref_model.train()
    for _ in range(config.reference_epochs):
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    ref_model.eval()
    from hashlib import sha256
    target_bundle["attack_provenance"] = {
        "reference_data_sha256": [sha256(t.encode()).hexdigest() for t in self_dataset],
        "reference_samples": len(self_dataset), "reference_epochs": config.reference_epochs,
        "reference_lr": config.reference_lr, "perturbation_domain": "embedding",
        "noise_scale": config.embedding_noise_scale, "perturbation_seed": config.seed,
        "score_transform": "sign(s)*(0.5+atan(log(abs(s)))/pi)",
    }
    return {"model": ref_model, "tokenizer": ref_tokenizer, "device": device}


def score_candidate_hf(target_bundle, reference_bundle, text: str, config) -> float:
    """Appendix A.3 Algorithm 1: identical Gaussian +/- directions in both models."""
    import torch
    target_tok, ref_tok = target_bundle["tokenizer"], reference_bundle["tokenizer"]
    if target_tok.get_vocab() != ref_tok.get_vocab():
        raise ValueError("SPV embedding calibration requires compatible vocabularies")
    ids = target_tok(text, return_tensors="pt", truncation=True, max_length=config.max_length)["input_ids"]
    ref_ids = ref_tok(text, return_tensors="pt", truncation=True, max_length=config.max_length)["input_ids"]
    if not torch.equal(ids, ref_ids) or ids.shape[-1] < 2:
        raise ValueError("SPV requires identical nonempty token events")
    target_ids, reference_ids = ids.to(target_bundle["device"]), ids.to(reference_bundle["device"])
    with torch.no_grad():
        target_emb = target_bundle["model"].get_input_embeddings()(target_ids)
        reference_emb = reference_bundle["model"].get_input_embeddings()(reference_ids)
    if target_emb.shape != reference_emb.shape:
        raise ValueError("SPV embedding dimensions must match")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    target_pairs, reference_pairs = [], []
    for _ in range(config.num_paraphrases):
        noise = torch.randn(target_emb.shape, generator=generator) * config.embedding_noise_scale
        for sign in (1, -1):
            target_pairs.append(_joint_logprob(target_bundle, target_ids,
                                target_emb + sign * noise.to(target_emb)))
            reference_pairs.append(_joint_logprob(reference_bundle, reference_ids,
                                   reference_emb + sign * noise.to(reference_emb)))
    return spv_score_from_logprobs(_joint_logprob(target_bundle, target_ids), target_pairs,
                                  _joint_logprob(reference_bundle, reference_ids), reference_pairs)


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
    needs_reference=False,  # This attack constructs its own self-prompt reference.
)
