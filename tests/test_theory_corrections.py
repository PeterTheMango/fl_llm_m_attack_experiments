"""Discriminating regressions for theory_v2. Run on the user's server.

These fixtures test algebra, metadata and client-gradient boundaries; they do
not claim reproduction of any published benchmark or attack success rate.
"""
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from master_script.core.attacks import wbc, reference, spv_mia, loss, amia
from master_script.core.config import validate_attack_config, resolve_run_config
from master_script.core.metrics import scientific_metrics
from master_script.core.scoring import validate_partition_tokens, causal_collator


def test_wbc_averages_sizes_not_pooled_windows():
    # Size 1: one of three positive; size 3: zero of one positive.
    assert wbc.wbc_score([2, -1, -1], [1, 3]) == pytest.approx(1 / 6)
    assert wbc.wbc_score([2, -1, -1], [1, 3]) != pytest.approx(1 / 4)


def test_wbc_short_input_retains_each_requested_size_weight():
    assert wbc.wbc_score([2, -1], [1, 2, 3]) == pytest.approx(5 / 6)
    assert wbc.wbc_score([0, 0], [1, 2]) == 0


def test_wbc_rejects_silent_alignment_and_nonfinite_evidence():
    with pytest.raises(ValueError):
        wbc.build_deltas([1, 2], [1])
    with pytest.raises(ValueError):
        wbc.wbc_score([math.nan], [1])


def test_reference_log_perplexity_ratio_distinguishes_difference():
    assert reference.calibrated_reference_score(2, 4) == -0.5
    assert reference.calibrated_reference_score(1, 2) == -0.5
    with pytest.raises(ValueError):
        reference.calibrated_reference_score(1, 0)


def test_spv_uses_printed_variation_sign_and_final_joint_arithmetic():
    assert spv_mia.probabilistic_variation(0.8, [0.5, 0.7]) == pytest.approx(-0.2)
    # Target variation .6-.8=-.2; reference .3-.4=-.1; final score=-.1.
    score = spv_mia.spv_score_from_logprobs(
        math.log(0.8), [math.log(0.5), math.log(0.7)],
        math.log(0.4), [math.log(0.2), math.log(0.4)])
    expected = -(0.5 + math.atan(math.log(0.1)) / math.pi)
    assert score == pytest.approx(expected)


def test_spv_joint_log_space_retains_sign_below_probability_underflow():
    score = spv_mia.spv_score_from_logprobs(-1001, [-1000, -1000], -1003, [-1002, -1002])
    assert math.isfinite(score) and score > 0
    assert spv_mia.spv_score_from_logprobs(-1000, [-1001, -1001], -1000, [-1001, -1001]) == 0


def test_loss_bounded_randomized_adversary_extremes_and_interior():
    assert loss.bounded_loss_decision(0, 10, 0.999)[0] is True
    assert loss.bounded_loss_decision(20, 10, 0)[0] is False
    assert loss.bounded_loss_decision(2, 10, 0.79)[0] is True
    assert loss.bounded_loss_decision(2, 10, 0.81)[0] is False


def test_low_fpr_resolution_is_not_invented():
    tiny = scientific_metrics([True, False], [1, 0])
    assert tiny['tpr_at_fpr_0_01'] is None
    sufficient = scientific_metrics([True] + [False] * 100, [1] + [0] * 100)
    assert sufficient['tpr_at_fpr_0_01'] == 1
    assert scientific_metrics([True], [1])['roc_auc'] is None


class PrefixTokenizer:
    def __call__(self, text, truncation=False, max_length=None, **kwargs):
        ids = [ord(c) for c in text]
        return {'input_ids': ids[:max_length] if truncation else ids}


def test_effective_token_collision_cannot_be_called_nonmembership():
    with pytest.raises(ValueError, match='declared membership'):
        validate_partition_tokens([['ABdifferent-tail']], PrefixTokenizer(), 2,
                                  target='ABtarget', expected_membership=False)
    with pytest.raises(ValueError, match='Calibration'):
        validate_partition_tokens([['ABtraining']], PrefixTokenizer(), 2,
                                  calibration=['ABcalibration'])


@pytest.mark.parametrize('kwargs', [dict(clients_per_round=5), dict(max_length=1),
                                   dict(attack_trials=1), dict(window_sizes=(0,))])
def test_invalid_configs_fail_before_training(kwargs):
    with pytest.raises(ValueError):
        validate_attack_config(replace(wbc.WbcConfig(), **kwargs))


def test_local_checkpoint_content_changes_resolved_identity(tmp_path):
    weights = tmp_path / 'model.bin'
    weights.write_bytes(b'first')
    cfg = reference.ReferenceConfig(model_id=str(tmp_path), use_hf_models=True)
    before = resolve_run_config(cfg).model_revision
    weights.write_bytes(b'second')
    assert resolve_run_config(cfg).model_revision != before


def test_padding_collator_preserves_real_eos():
    torch = pytest.importorskip('torch')
    class Tokenizer:
        def pad(self, examples, **kwargs):
            return {'input_ids': torch.tensor([[3, 0, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 0]])}
    assert causal_collator(Tokenizer())([])['labels'].tolist() == [[3, 0, -100]]


def test_amia_observation_depends_on_downstream_loss_gradient(monkeypatch):
    torch = pytest.importorskip('torch')
    probe = amia._ami_probe_cls()(1, 1)
    with torch.no_grad():
        probe.fc1.weight.fill_(1)
        probe.fc1.bias.zero_()
        probe.fc2.weight.fill_(1)
        probe.fc2.bias.zero_()
    monkeypatch.setattr(amia, 'sentence_embedding', lambda *args: torch.tensor([[1.0]]))
    class Model:
        def get_output_embeddings(self):
            return lambda features: torch.zeros((len(features), 2))
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {'input_ids': [[0, 1]]}
    gradients = amia.client_loss_gradients(Model(), Tokenizer(), probe, ['private'], amia.AmiaConfig())
    # d CE([1,0], label=1)/dz = sigmoid(1), whereas d sum(ReLU(z))/dz = 1.
    assert gradients[-1][0] == pytest.approx(torch.sigmoid(torch.tensor(1.)).item())
    with torch.no_grad():
        probe.fc2.bias.fill_(-2)
    assert amia.gradient_score(amia.client_loss_gradients(
        Model(), Tokenizer(), probe, ['private'], amia.AmiaConfig())) == 0


def test_ldp_has_independent_record_draws_and_public_seed_reproducibility():
    torch = pytest.importorskip('torch')
    from master_script.core.attacks.amia_ldp import perturb_features
    cfg = amia.AmiaConfig(ldp_mechanism='BitRand')
    repeated = torch.zeros((64, 8))
    first = perturb_features(repeated, cfg, seed=42)
    assert torch.equal(first, perturb_features(repeated, cfg, seed=42))
    assert torch.unique(first, dim=0).shape[0] > 1
    assert not torch.equal(first, perturb_features(repeated, cfg, seed=43))
