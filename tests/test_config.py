# tests/test_config.py
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

from master_script.core.config import AttackConfig, expand_sweep, experiment_key


@dataclass(frozen=True)
class _Cfg(AttackConfig):
    attack_name: str = "demo"
    rounds: int = 1
    seed: int = 7


def test_base_class_declares_no_fields():
    from dataclasses import fields

    assert fields(AttackConfig) == ()


def test_key_sha16_matches_modern_notebook_formula():
    from master_script.core.config import key_sha16

    payload = json.dumps(
        {"attack_name": "demo", "rounds": 1, "seed": 7}, sort_keys=True, separators=(",", ":")
    )
    assert key_sha16(_Cfg()) == sha256(payload.encode("utf-8")).hexdigest()[:16]
    assert len(key_sha16(_Cfg())) == 16


def test_key_sha24_is_24_chars_for_amia():
    """AMIA_adaptation.ipynb truncates to 24, not 16."""
    from master_script.core.config import key_sha24_default_str

    assert len(key_sha24_default_str(_Cfg())) == 24


def test_key_named_prefix_prepends_experiment_name_for_loss():
    """LOSS_adaptation.ipynb's doc id is f'{experiment_name}_{digest16}'."""
    from dataclasses import dataclass as dc

    from master_script.core.config import key_named_prefix

    @dc(frozen=True)
    class _LossLike(AttackConfig):
        experiment_name: str = "loss_v1"
        seed: int = 13

    key = key_named_prefix(_LossLike())
    assert key.startswith("loss_v1_")
    assert len(key.split("_")[-1]) == 16


def test_stable_json_uses_default_str_for_unserializable():
    from master_script.core.config import stable_json

    assert stable_json({"p": Path("/tmp/x")}) == '{"p":"/tmp/x"}'


def test_experiment_key_dispatches_to_spec_key_fn():
    from master_script.core.config import key_sha24_default_str

    class _Spec:
        key_fn = staticmethod(key_sha24_default_str)

    assert len(experiment_key(_Cfg(), _Spec())) == 24
    assert len(experiment_key(_Cfg())) == 16  # fallback: modern formula


def test_expand_sweep_yields_cartesian_product():
    out = list(expand_sweep(_Cfg(), {"rounds": [1, 2], "seed": [7, 11]}))
    assert len(out) == 4
    assert {(c.rounds, c.seed) for c in out} == {(1, 7), (1, 11), (2, 7), (2, 11)}


def test_expand_sweep_empty_grid_yields_base_once():
    out = list(expand_sweep(_Cfg(), {}))
    assert out == [_Cfg()]
