"""Neighbourhood comparison MIA. Ported from neighborhood_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
import random
from dataclasses import dataclass
from typing import List, Sequence

from ..config import AttackConfig
from ..metrics import roc_auc
from ..scoring import ScoreContext
from ..spec import AttackSpec


@dataclass(frozen=True)
class NeighborhoodConfig(AttackConfig):
    attack_name: str = "neighborhood"
    paper_source: str = "Mattern et al. 2023 (arXiv:2305.18462) Neighbourhood Comparison MIA"
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
    num_neighbours: int = 25
    neighbour_swaps: int = 1
    threshold: float = 0.02
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/neighborhood_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "Neighbourhood comparison MIA: for target x, generate n near-identical neighbours via "
        "BERT masked-LM single-word (m=1) substitution, then predict member when L(x) is "
        "substantially below the mean neighbour loss (Eq. 3). Reference-model-free: neighbours "
        "replace the reference model."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, score the target record under the final FL model only by generating "
        "neighbours and taking mean_neighbour_loss - target_loss. No reference model is used."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; membership score = mean_neighbour_loss - target_loss so "
        "members (loss below neighbours) score higher."
    ),
    "deviation_from_source": (
        "The attack was designed for the fine-tuning setting, so applying it to a federated "
        "fine-tuned model is a direct transfer. No reference model is needed: the neighbours are "
        "the difficulty calibration. The smoke run uses a deterministic toy causal scorer and "
        "toy single-word neighbour perturbations; set use_hf_models=True for genuine federated "
        "fine-tuning of an open-source LLM (Flower FedAvg) with BERT (bert-base-uncased) "
        "neighbour generation following Algorithm 1 (n=100, m=1, dropout p=0.7 on the original "
        "embedding)."
    ),
}

def neighborhood_membership_score(target_loss: float, neighbour_losses: Sequence[float]) -> float:
    """Membership score = mean(neighbour_losses) - target_loss. Higher => member.

    Negation of the paper's Eq. 3 quantity so that members (target loss well
    below the neighbour mean) score higher, matching the >= threshold convention.
    """
    if not neighbour_losses:
        raise ValueError("Need at least one neighbour loss to calibrate.")
    return (sum(neighbour_losses) / len(neighbour_losses)) - target_loss


# Benign filler tokens used to build toy single-word (m=1) neighbours.
_TOY_FILLER_TOKENS = [
    "today", "here", "please", "note", "update", "kindly", "again", "soon",
    "however", "meanwhile", "generally", "actually", "somewhat", "briefly",
    "recently", "currently", "possibly", "perhaps", "namely", "therefore",
    "additionally", "otherwise", "regardless", "accordingly", "nonetheless",
]


def generate_neighbours_toy(text: str, n: int, swaps: int = 1, seed: int = 0) -> List[str]:
    """Synthesize n neighbours by replacing `swaps` word(s) with benign fillers.

    Each neighbour differs from the target by a single-word substitution (m=1
    by default), mirroring the paper's best configuration. The perturbations
    are semantics-preserving-ish fillers, and -- importantly -- they are NOT the
    exact target string, so a model that memorized the target scores it lower
    than these neighbours.
    """
    rng = random.Random(seed)
    words = text.split()
    if not words:
        return [text]
    neighbours = []
    for i in range(n):
        variant = list(words)
        for _ in range(max(1, swaps)):
            pos = rng.randrange(len(variant))
            variant[pos] = _TOY_FILLER_TOKENS[(i + pos) % len(_TOY_FILLER_TOKENS)]
        neighbours.append(" ".join(variant))
    return neighbours


def score_candidate_toy(target_model, text: str, config) -> float:
    neighbours = generate_neighbours_toy(
        text, n=config.num_neighbours, swaps=config.neighbour_swaps, seed=config.seed
    )
    target_loss = target_model.nll(text)
    neighbour_losses = [target_model.nll(nb) for nb in neighbours]
    return neighborhood_membership_score(target_loss, neighbour_losses)


def generate_neighbours_bert(text, tokenizer_mlm, model_mlm, n=100, dropout_p=0.7, device="cpu", max_length=128):
    """Paper-faithful single-word (m=1) neighbour generation (Algorithm 1).

    Strong dropout is applied to the ORIGINAL token embedding (the token is not
    masked out) so BERT respects the original word's meaning; candidates are
    ranked by the normalised suitability p_swap = p(w_tilde) / (1 - p(original)).
    """
    import torch

    encoded = tokenizer_mlm(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    ids = input_ids[0]
    special = set(tokenizer_mlm.all_special_ids)
    embedding_layer = model_mlm.get_input_embeddings()
    candidates = []
    for pos in range(ids.shape[0]):
        original_id = int(ids[pos].item())
        if original_id in special:
            continue
        base_embeds = embedding_layer(input_ids)
        dropped = torch.nn.functional.dropout(base_embeds[:, pos, :], p=dropout_p, training=True)
        perturbed = base_embeds.clone()
        perturbed[:, pos, :] = dropped
        with torch.no_grad():
            logits = model_mlm(inputs_embeds=perturbed).logits
        probs = torch.softmax(logits[0, pos], dim=-1)
        denom = max(1e-8, 1.0 - float(probs[original_id].item()))
        topk = torch.topk(probs, k=min(10, probs.shape[-1]))
        for score, cand_id in zip(topk.values.tolist(), topk.indices.tolist()):
            if cand_id == original_id or cand_id in special:
                continue
            candidates.append((score / denom, pos, cand_id))
    candidates.sort(key=lambda c: c[0], reverse=True)
    neighbours = []
    for _, pos, cand_id in candidates[:n]:
        new_ids = ids.clone()
        new_ids[pos] = cand_id
        neighbours.append(tokenizer_mlm.decode(new_ids, skip_special_tokens=True))
    return neighbours


def _mean_token_nll_hf(model, tokenizer, text, device, max_length):
    import torch
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if encoded["input_ids"].shape[-1] < 2:
        return 0.0
    with torch.no_grad():
        outputs = model(**encoded, labels=encoded["input_ids"])
    return float(outputs.loss.detach().cpu())


def score_candidate_hf(target_bundle, text: str, config) -> float:
    """Generate BERT neighbours and score the record + neighbours under the FL model."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model = target_bundle["model"]
    tokenizer = target_bundle["tokenizer"]
    device = target_bundle["device"]

    mlm_name = "bert-base-uncased"
    mlm_tokenizer = AutoTokenizer.from_pretrained(mlm_name)
    mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_name).to(device).eval()

    neighbours = generate_neighbours_bert(
        text, mlm_tokenizer, mlm_model, n=config.num_neighbours,
        device=device, max_length=config.max_length,
    )
    target_loss = _mean_token_nll_hf(model, tokenizer, text, device, config.max_length)
    neighbour_losses = [
        _mean_token_nll_hf(model, tokenizer, nb, device, config.max_length) for nb in neighbours
    ]
    return neighborhood_membership_score(target_loss, neighbour_losses)


def score_toy(ctx: ScoreContext) -> float:
    return score_candidate_toy(ctx.target, ctx.text, ctx.config)


def score_hf(ctx: ScoreContext) -> float:
    return score_candidate_hf(ctx.target, ctx.text, ctx.config)


def _extra_metrics(trials):
    return {"roc_auc": roc_auc([t["truth_member"] for t in trials], [t["score"] for t in trials])}


SPEC = AttackSpec(
    name="neighborhood",
    config_cls=NeighborhoodConfig,
    methodology=METHODOLOGY,
    score_toy=score_toy,
    score_hf=score_hf,
    extra_metrics=_extra_metrics,
)
