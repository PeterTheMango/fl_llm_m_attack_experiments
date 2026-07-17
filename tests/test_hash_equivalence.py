"""THE critical test: consolidation must not move any run_id.

run_id is the Firestore cache key. If a ported config hashes differently from
its notebook, every completed run for that attack is orphaned and silently
recomputed. Each attack is asserted independently so a failure names the attack.
"""
from dataclasses import asdict, fields
import pytest

from master_script.core.config import experiment_key
from master_script.core.registry import ATTACKS
from tests.notebook_configs import NOTEBOOKS, notebook_config_class

ATTACK_NAMES = sorted(NOTEBOOKS)

# Each notebook's OWN key formula, transcribed from its experiment_key().
# Verified against the real notebooks: the nine modern use sha256[:16] with no
# default=str; AMIA uses [:24] with default=str; LOSS prefixes experiment_name.
MODERN = set(NOTEBOOKS) - {"amia", "loss"}


def _notebook_key(attack: str, instance) -> str:
    from hashlib import sha256
    import json

    if attack in MODERN:
        payload = json.dumps(asdict(instance), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    payload = json.dumps(asdict(instance), sort_keys=True, separators=(",", ":"), default=str)
    digest = sha256(payload.encode("utf-8")).hexdigest()
    if attack == "amia":
        return digest[:24]
    return f"{instance.experiment_name}_{digest[:16]}"


@pytest.mark.parametrize("attack", ATTACK_NAMES)
def test_ported_config_has_identical_field_order(attack):
    nb_fields = [f.name for f in fields(notebook_config_class(attack))]
    ported_fields = [f.name for f in fields(ATTACKS[attack].config_cls)]
    assert ported_fields == nb_fields


@pytest.mark.parametrize("attack", ATTACK_NAMES)
def test_ported_config_has_identical_defaults(attack):
    nb_defaults = asdict(notebook_config_class(attack)())
    ported_defaults = asdict(ATTACKS[attack].config_cls())
    assert ported_defaults == nb_defaults


@pytest.mark.parametrize("attack", ATTACK_NAMES)
def test_default_run_id_is_byte_identical(attack):
    spec = ATTACKS[attack]
    expected = _notebook_key(attack, notebook_config_class(attack)())
    assert experiment_key(spec.config_cls(), spec) == expected


@pytest.mark.parametrize("attack", ATTACK_NAMES)
def test_run_id_identical_under_swept_values(attack):
    """Key equality must survive non-default values, not just defaults."""
    from dataclasses import replace

    spec = ATTACKS[attack]
    overrides = {"seed": 11, "federated_rounds": 3}
    names = {f.name for f in fields(spec.config_cls)}
    applicable = {k: v for k, v in overrides.items() if k in names}
    if not applicable:
        pytest.skip(f"{attack} has none of {sorted(overrides)}")

    expected = _notebook_key(attack, replace(notebook_config_class(attack)(), **applicable))
    assert experiment_key(replace(spec.config_cls(), **applicable), spec) == expected


def test_key_formula_lengths_match_each_notebooks_own_shape():
    """Regression guard for the three-formula split.

    These exact values were read off the real notebooks. If any changes, a
    consolidation bug has silently orphaned that attack's Firestore documents.
    """
    assert len(experiment_key(ATTACKS["zlib"].config_cls(), ATTACKS["zlib"])) == 16
    assert len(experiment_key(ATTACKS["amia"].config_cls(), ATTACKS["amia"])) == 24
    loss_key = experiment_key(ATTACKS["loss"].config_cls(), ATTACKS["loss"])
    assert loss_key.startswith("loss_federated_llm_adaptation_v1_")


@pytest.mark.parametrize(
    "attack,expected",
    [
        ("zlib", "f747493634f0cc01"),
        ("min_k", "93c4531ab14111a0"),
        ("min_k_plus_plus", "5815bcb26500a33f"),
        ("neighborhood", "2670329cab9e3ca3"),
        ("recall", "85713154550329f9"),
        ("reference", "8d860a2b47a397e3"),
        ("samia", "27cf824d2ed6fa05"),
        ("spv_mia", "f58a7590c6313fbc"),
        ("wbc", "f5e16331ecb7be13"),
        ("amia", "848ce58285a38a20ac058893"),
        ("loss", "loss_federated_llm_adaptation_v1_952b56bc2ecd70aa"),
    ],
)
def test_default_keys_match_values_observed_in_the_notebooks(attack, expected):
    """Golden values captured from the notebooks on 2026-07-16.

    Belt-and-braces alongside the live comparison above: if someone edits a
    notebook's defaults, that test still passes (both sides move) but this one
    fails loudly -- which is the correct signal, because the cached Firestore
    documents did NOT move.
    """
    spec = ATTACKS[attack]
    assert experiment_key(spec.config_cls(), spec) == expected


def test_all_eleven_attacks_are_registered():
    assert set(ATTACKS) == set(NOTEBOOKS)
    assert len(ATTACKS) == 11
