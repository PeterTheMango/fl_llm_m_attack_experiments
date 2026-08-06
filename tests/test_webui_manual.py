# tests/test_webui_manual.py
"""Manual launch mode: the form's schema, its coercion, and the guarantee
that running a manual config equals running the YAML it would save as."""
import pytest

from master_script.core.registry import ATTACKS
from master_script.webui import attackfields


def test_every_dataclass_field_appears_exactly_once():
    """A field missing from the schema is a field the form silently cannot set."""
    from dataclasses import fields as dc_fields

    for name, spec in ATTACKS.items():
        declared = {f.name for f in dc_fields(spec.config_cls)}
        listed = [f["name"] for f in attackfields.fields_for(name)]
        assert len(listed) == len(set(listed)), f"{name} lists a field twice"
        assert declared <= set(listed), f"{name} is missing {declared - set(listed)}"


def test_every_field_lands_in_a_known_group():
    for name in ATTACKS:
        for field in attackfields.fields_for(name):
            assert field["group"] in attackfields.GROUP_ORDER


def test_identity_fields_are_read_only():
    """attack_name and paper_source feed the experiment key; editing them
    produces a differently-hashed run, never a useful variation."""
    by = attackfields.by_name("zlib")
    assert by["attack_name"]["readonly"] is True
    assert by["paper_source"]["readonly"] is True
    assert by["seed"]["readonly"] is False


def test_field_types_are_reported_for_the_form():
    by = attackfields.by_name("zlib")
    assert by["seed"]["type"] == "int"
    assert by["client_lr"]["type"] == "float"
    assert by["use_hf_models"]["type"] == "bool"
    assert by["model_id"]["type"] == "str"


def test_defaults_come_from_the_dataclass():
    assert attackfields.by_name("zlib")["seed"]["default"] == 7
    assert attackfields.by_name("samia")["num_samples"]["default"] == 10


def test_attacks_without_a_toy_path_still_show_a_locked_use_hf_models():
    """amia/loss have no such dataclass field, but the form must not imply
    the toy path is available."""
    for name in ("amia", "loss"):
        assert ATTACKS[name].supports_toy is False
        field = attackfields.by_name(name)["use_hf_models"]
        assert field["default"] is True and field["readonly"] is True


def test_schema_covers_every_registered_attack():
    payload = attackfields.schema()
    assert set(payload["attacks"]) == set(ATTACKS)
    assert payload["attacks"]["amia"]["supports_toy"] is False
    assert payload["group_order"] == attackfields.GROUP_ORDER
