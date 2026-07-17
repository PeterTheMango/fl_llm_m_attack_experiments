"""Reference/comparison-model MIA. Ported from reference_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..federation import ToyFederatedLM, load_reference_bundle
from ..scoring import ScoreContext
from ..spec import AttackSpec


@dataclass(frozen=True)
class ReferenceConfig(AttackConfig):
    attack_name: str = "reference"
    paper_source: str = "Carlini et al. 2021 reference/comparison-model MIA; LiRA-style LM calibration"
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
    threshold: float = 0.25
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/reference_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": "Reference/calibration MIA: compare target LM likelihood to a reference LM likelihood.",
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, score the target record under the final FL model versus the base "
        "reference model."
    ),
    "metric_definition": "Adv = 0.5 * TPR + 0.5 * TNR",
    "deviation_from_source": (
        "The smoke run uses a deterministic toy causal scorer. Set use_hf_models=True for "
        "genuine federated fine-tuning of an open-source LLM with the Flower (flwr) FedAvg "
        "simulation."
    ),
}

def calibrated_reference_score(target_nll: float, reference_nll: float) -> float:
    return reference_nll - target_nll


def score_candidate_toy(target_model, reference_model, text: str) -> float:
    return calibrated_reference_score(target_model.nll(text), reference_model.nll(text))


def score_candidate_hf(target_bundle, reference_bundle, text: str, max_length: int = 64) -> float:
    import torch

    def mean_nll(bundle):
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        device = bundle["device"]
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, labels=encoded["input_ids"])
        return float(outputs.loss.detach().cpu())

    return calibrated_reference_score(mean_nll(target_bundle), mean_nll(reference_bundle))


def score_toy(ctx: ScoreContext) -> float:
    # The notebook's run_attack_trial always uses a fresh, untrained ToyFederatedLM as the
    # reference; ctx.reference lets a caller (e.g. the orchestration harness) override it.
    reference = ctx.reference if ctx.reference is not None else ToyFederatedLM()
    return score_candidate_toy(ctx.target, reference, ctx.text)


def score_hf(ctx: ScoreContext) -> float:
    reference = ctx.reference if ctx.reference is not None else load_reference_bundle(ctx.config)
    return score_candidate_hf(ctx.target, reference, ctx.text, max_length=ctx.config.max_length)


SPEC = AttackSpec(
    name="reference",
    config_cls=ReferenceConfig,
    methodology=METHODOLOGY,
    score_toy=score_toy,
    score_hf=score_hf,
    extra_metrics=None,
    needs_reference=True,
)
