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


def test_identity_and_storage_fields_are_read_only():
    """Identity fields cannot be useful variations, and storage is shared."""
    by = attackfields.by_name("zlib")
    assert by["attack_name"]["readonly"] is True
    assert by["paper_source"]["readonly"] is True
    assert by["firestore_collection"]["readonly"] is True
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


# ---------- payload -> config doc ----------

from master_script.webui import manual


def _payload(*attacks):
    return {"attacks": list(attacks)}


def _attack(name, values=None, sweeps=None):
    return {"name": name, "values": values or {}, "sweeps": sweeps or []}


def test_only_changed_fields_are_emitted():
    """A manual config should be as small as the YAML you'd write by hand."""
    doc = manual.build_doc(_payload(_attack("zlib", {"seed": "7", "federated_rounds": "3"})))
    assert doc == {"attacks": {"zlib": {"base": {"federated_rounds": 3}}}}


def test_an_attack_with_no_changes_emits_an_empty_section():
    doc = manual.build_doc(_payload(_attack("zlib")))
    assert doc == {"attacks": {"zlib": {}}}


def test_values_are_coerced_to_the_dataclass_type():
    doc = manual.build_doc(_payload(_attack("zlib", {
        "federated_rounds": "3", "client_lr": "1e-4",
        "use_hf_models": "true", "model_id": "distilbert/distilgpt2",
    })))
    base = doc["attacks"]["zlib"]["base"]
    assert base == {"federated_rounds": 3, "client_lr": 1e-4,
                    "use_hf_models": True, "model_id": "distilbert/distilgpt2"}


def test_a_sweep_field_becomes_a_typed_list():
    doc = manual.build_doc(_payload(_attack("zlib", {"seed": "7, 11, 23"}, ["seed"])))
    assert doc == {"attacks": {"zlib": {"sweep": {"seed": [7, 11, 23]}}}}


def test_a_sweep_holding_the_default_is_still_emitted():
    """A one-value sweep is a deliberate choice, not an unchanged field."""
    doc = manual.build_doc(_payload(_attack("zlib", {"seed": "7"}, ["seed"])))
    assert doc == {"attacks": {"zlib": {"sweep": {"seed": [7]}}}}


def test_a_read_only_field_echoed_unchanged_is_ignored():
    """The form may send back what it displayed; that is not a change."""
    doc = manual.build_doc(_payload(_attack("zlib", {"attack_name": "zlib"})))
    assert doc == {"attacks": {"zlib": {}}}


def test_changing_a_read_only_field_is_refused_not_silently_dropped():
    """Dropping it would run something other than what was asked for -- here,
    under an experiment key nobody will look for."""
    result = manual.validate(_payload(_attack("zlib", {"attack_name": "not_zlib"})))
    assert result["ok"] is False
    assert "attack_name" in result["message"] and "experiment key" in result["message"]


def test_a_non_numeric_value_is_reported_not_raised_as_a_crash():
    result = manual.validate(_payload(_attack("zlib", {"federated_rounds": "three"})))
    assert result["ok"] is False
    assert "federated_rounds" in result["message"] and "three" in result["message"]


def test_a_non_boolean_value_is_reported():
    result = manual.validate(_payload(_attack("zlib", {"keep_artifacts": "maybe"})))
    assert result["ok"] is False and "keep_artifacts" in result["message"]


def test_an_empty_sweep_is_refused():
    result = manual.validate(_payload(_attack("zlib", {"seed": " , "}, ["seed"])))
    assert result["ok"] is False and "no values" in result["message"]


def test_no_attacks_is_refused():
    result = manual.validate(_payload())
    assert result["ok"] is False and "at least one attack" in result["message"]


def test_a_duplicate_attack_is_refused():
    result = manual.validate(_payload(_attack("zlib"), _attack("zlib", {"seed": "11"})))
    assert result["ok"] is False and "more than once" in result["message"]


def test_an_unknown_attack_gets_the_loaders_own_message():
    result = manual.validate(_payload(_attack("not_an_attack")))
    assert result["ok"] is False and "unknown attack" in result["message"]


def test_an_unknown_field_gets_the_loaders_own_message():
    """The loader lists the valid fields; the manual path must not swallow that."""
    result = manual.validate(_payload(_attack("zlib", {"federated_roundz": "3"})))
    assert result["ok"] is False
    assert "federated_roundz" in result["message"] and "Valid fields" in result["message"]


def test_an_attack_without_a_toy_path_refuses_use_hf_models_false():
    result = manual.validate(_payload(_attack("amia", {"use_hf_models": "false"})))
    assert result["ok"] is False and "no toy path" in result["message"]


def test_validate_reports_the_real_expanded_run_count():
    result = manual.validate(_payload(
        _attack("samia", {"seed": "7, 11"}, ["seed"]),
        _attack("zlib", {"federated_rounds": "2"}),
    ))
    assert result["ok"] is True
    assert result["runs"] == 3
    assert result["per_attack"] == {"samia": 2, "zlib": 1}


def test_validate_returns_the_yaml_that_save_would_write():
    result = manual.validate(_payload(_attack("zlib", {"seed": "7, 11"}, ["seed"])))
    assert "zlib" in result["yaml"] and "sweep" in result["yaml"]


def test_a_manual_run_equals_running_the_yaml_it_would_save(tmp_path):
    """The property that makes "run without saving" honest: if these ever
    diverge, a saved config no longer reproduces the run it was saved from."""
    from master_script.core.yaml_config import load_config_file

    payload = _payload(
        _attack("samia", {"seed": "7, 11", "federated_rounds": "2"}, ["seed"]),
        _attack("zlib", {"client_lr": "1e-4"}),
    )
    direct = manual.pairs(payload)

    path = tmp_path / "saved.yaml"
    path.write_text(manual.to_yaml(manual.build_doc(payload)))
    from_file = load_config_file(path)

    assert [c for c, _s in direct] == [c for c, _s in from_file]
    assert [s.name for _c, s in direct] == [s.name for _c, s in from_file]


def test_the_generated_yaml_passes_the_config_editors_own_validation():
    """Save As goes through POST /api/configs/save, which validates the text."""
    from master_script.webui import configs

    text = manual.to_yaml(manual.build_doc(_payload(_attack("zlib", {"seed": "7, 11"}, ["seed"]))))
    assert configs.validate(text)["ok"] is True


# ---------- API wiring ----------

from fastapi.testclient import TestClient


def _client():
    from master_script.webui import app as app_module

    return TestClient(app_module.app)


def test_the_field_schema_is_served():
    res = _client().get("/api/attacks/fields")
    assert res.status_code == 200
    body = res.json()
    assert set(body["attacks"]) == set(ATTACKS)
    assert any(f["name"] == "seed" for f in body["attacks"]["zlib"]["fields"])


def test_manual_validate_reports_the_run_count():
    res = _client().post("/api/launch/manual/validate", json={"attacks": [
        {"name": "zlib", "values": {"seed": "7, 11"}, "sweeps": ["seed"]},
    ]})
    assert res.status_code == 200
    assert res.json()["runs"] == 2


def test_manual_validate_reports_a_bad_payload_as_json_not_a_500():
    res = _client().post("/api/launch/manual/validate", json={"attacks": [
        {"name": "zlib", "values": {"federated_rounds": "three"}, "sweeps": []},
    ]})
    assert res.status_code == 200
    assert res.json()["ok"] is False


def test_manual_validate_writes_nothing():
    """Validation must never leave a config behind."""
    from master_script.webui import configs

    before = sorted(p.name for p in configs.CONFIGS_DIR.glob("*.yaml"))
    _client().post("/api/launch/manual/validate", json={"attacks": [{"name": "zlib"}]})
    assert sorted(p.name for p in configs.CONFIGS_DIR.glob("*.yaml")) == before


def test_manual_start_delegates_to_launch(monkeypatch):
    from master_script.webui import api as api_mod

    seen = {}
    monkeypatch.setattr(api_mod.launch, "start_manual",
                        lambda payload, use_firestore: seen.update(call=(payload, use_firestore))
                        or {"ok": True, "message": "Started 1 run(s)."})
    res = _client().post("/api/launch/manual", json={
        "attacks": [{"name": "zlib"}], "use_firestore": False,
    })
    assert res.status_code == 200 and res.json()["ok"] is True
    payload, use_firestore = seen["call"]
    assert use_firestore is False
    assert payload["attacks"][0]["name"] == "zlib"
