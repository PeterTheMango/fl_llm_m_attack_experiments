# Launch Manual Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the dashboard's Launch page into a **Manual** attack-configuration builder and an **Existing config** mode, so an attack can be fine-tuned and run from a typed form without first hand-writing a YAML file.

**Architecture:** Manual mode owns no expansion or validation logic. A typed form payload is coerced into the exact dict shape a YAML config parses to, then handed to the same loader `perform_experiments.py` uses. To make that sharing possible, `core/yaml_config.load_config_file` splits into a doc-level `load_config_doc` plus a thin file reader. Everything downstream — run counts, error messages, the `(config, spec)` pairs the runner receives — is therefore identical between a manual run and a saved-file run.

**Tech Stack:** Python 3, FastAPI + pydantic v2, PyYAML, pytest. Front end is a single dependency-free `static/app.js` (string-template rendering, `data-act` / `data-inp` event delegation, full re-render on every state change).

**Spec:** `docs/superpowers/specs/2026-08-06-launch-manual-mode-design.md`

## Global Constraints

- Run tests with `python -m pytest` from the repo root (`/Users/pyeshuajs23/Documents/MITACS/Research Documents`).
- **Never change the attack dataclasses in `master_script/core/attacks/`, `core/config.py`'s key formulas, or `core/runner.py`.** Field names and their order feed `experiment_key`, which is the Firestore document id. A changed field orphans completed documents. `tests/test_hash_equivalence.py` guards this — it must keep passing.
- Existing tests must pass unchanged: `tests/test_yaml_config.py`, `tests/test_cli.py`, `tests/test_webui_launch.py`, `tests/test_webui_pages.py`, `tests/test_webui_configs.py`.
- No new third-party dependencies. PyYAML, FastAPI and pydantic are already in use.
- The front end has no build step and no test harness. `static/app.js` is served verbatim; no imports, no JSX, no bundler.
- Style: modules carry a short docstring saying what they are *for*, and comments explain *why* a decision was made, not what a line does. Match the surrounding files.
- Commit after every task with the message given in the task's final step.

## File Structure

| File | Responsibility |
|---|---|
| `master_script/core/yaml_config.py` (modify) | Gains `load_config_doc(doc, only, source)`; `load_config_file` becomes a reader that delegates. |
| `master_script/webui/attackfields.py` (create) | Pure introspection of each attack dataclass into a form schema. No I/O, no state. |
| `master_script/webui/manual.py` (create) | Manual payload → config doc → pairs; and → YAML text for saving. |
| `master_script/webui/configs.py` (modify) | `validate` stops writing a temp file, uses `load_config_doc`. |
| `master_script/webui/launch.py` (modify) | Shared `_start(pairs, ...)`; new `start_manual(payload, ...)`. |
| `master_script/webui/api.py` (modify) | Three new routes; no logic. |
| `master_script/webui/static/app.js` (modify) | Mode switch + manual panel rendering and events. |
| `tests/test_webui_manual.py` (create) | attackfields + manual + the round-trip property + API wiring. |
| `tests/test_webui_configs.py` (modify) | Asserts `validate` creates no temp file. |
| `tests/test_yaml_config.py` (modify) | Two tests for the new `load_config_doc` entry point. |

---

### Task 1: Split the loader into doc-level and file-level entry points

**Files:**
- Modify: `master_script/core/yaml_config.py:26-83`
- Modify: `master_script/webui/configs.py:1-15,73-102`
- Test: `tests/test_yaml_config.py`, `tests/test_webui_configs.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_config_doc(doc: dict, only: Optional[Sequence[str]] = None, source: str = "<config>") -> List[Tuple[object, object]]` — raises `ConfigError`.
  - `load_config_file(path, only=None) -> List[Tuple[object, object]]` — unchanged signature and behaviour.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_yaml_config.py`:

```python
def test_load_config_doc_expands_an_already_parsed_dict():
    """Manual mode has no file; it must reach the same expansion."""
    from master_script.core.yaml_config import load_config_doc

    pairs = load_config_doc({"attacks": {"zlib": {"sweep": {"seed": [7, 11]}}}})
    assert [cfg.seed for cfg, _spec in pairs] == [7, 11]


def test_load_config_doc_names_its_source_in_errors():
    """The caller decides what a config is called; there may be no path."""
    from master_script.core.yaml_config import ConfigError, load_config_doc

    with pytest.raises(ConfigError) as exc:
        load_config_doc({"attacks": {"nonexistent": {}}}, source="<manual>")
    assert "<manual>" in str(exc.value)
```

Append to `tests/test_webui_configs.py`:

```python
def test_validate_creates_no_temp_file(config_dir, monkeypatch):
    """Validation is a pure expansion; it has no reason to touch the disk."""
    import tempfile

    def _refuse(*a, **kw):
        raise AssertionError("validate must not create a temporary file")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _refuse)
    assert configs.validate(VALID)["ok"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

```
python -m pytest tests/test_yaml_config.py tests/test_webui_configs.py -v
```

Expected: the two `load_config_doc` tests fail with `ImportError: cannot import name 'load_config_doc'`; `test_validate_creates_no_temp_file` fails with the `AssertionError` from `_refuse`.

- [ ] **Step 3: Split the loader**

In `master_script/core/yaml_config.py`, replace the body of `load_config_file` (lines 26-83) with:

```python
def load_config_file(path, only: Optional[Sequence[str]] = None) -> List[Tuple[object, object]]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    return load_config_doc(yaml.safe_load(path.read_text()) or {}, only, source=str(path))


def load_config_doc(doc: dict, only: Optional[Sequence[str]] = None,
                    source: str = "<config>") -> List[Tuple[object, object]]:
    """Expand an already-parsed config document.

    Split out from load_config_file so the dashboard's manual mode -- which
    builds the same dict from a form and has no file to point at -- reaches
    this validation and expansion rather than reimplementing it. `source` is
    whatever the caller wants errors to name.
    """
    defaults = doc.get("defaults") or {}
    attacks = doc.get("attacks") or {}
    if not attacks:
        raise ConfigError(f"{source}: no 'attacks:' section")

    unknown = sorted(set(attacks) - set(ATTACKS))
    if unknown:
        raise ConfigError(
            f"{source}: unknown attack(s): {', '.join(unknown)}. Known: {', '.join(sorted(ATTACKS))}"
        )

    selected = list(attacks) if only is None else [a for a in attacks if a in set(only)]
    pairs: List[Tuple[object, object]] = []
    for name in selected:
        spec = ATTACKS[name]
        allowed = _field_names(spec.config_cls)
        section = attacks[name] or {}
        base_over = dict(section.get("base") or {})
        sweep = dict(section.get("sweep") or {})

        # amia/loss have no toy path and no use_hf_models field on their
        # dataclasses at all. Treat use_hf_models there as a virtual switch:
        # an explicit false is the "toy" rejection below, not an unknown-field
        # error, and any other value is stripped before field validation.
        if not spec.supports_toy:
            if base_over.get("use_hf_models") is False or any(
                v is False for v in (sweep.get("use_hf_models") or [])
            ):
                raise ConfigError(
                    f"attack '{name}' has no toy path and requires use_hf_models: true"
                )
            base_over.pop("use_hf_models", None)
            sweep.pop("use_hf_models", None)

        bad = sorted((set(base_over) | set(sweep)) - allowed)
        if bad:
            raise ConfigError(
                f"{source}: attack '{name}' has unknown field(s): {', '.join(bad)}. "
                f"Valid fields: {', '.join(sorted(allowed))}"
            )

        # defaults are cross-attack: silently skip keys this attack lacks.
        merged = {k: v for k, v in defaults.items() if k in allowed}
        merged.update(base_over)
        cfg = replace(spec.config_cls(), **merged) if merged else spec.config_cls()

        for expanded in expand_sweep(cfg, sweep):
            if not spec.supports_toy and not getattr(expanded, "use_hf_models", True):
                raise ConfigError(
                    f"attack '{name}' has no toy path and requires use_hf_models: true"
                )
            pairs.append((expanded, spec))
    return pairs
```

Note the only changes inside the moved body are `path` → `source` in the three error strings. Nothing else moves.

Also update the module docstring's first line to mention both entry points:

```python
"""YAML -> (config, spec) pairs, with fail-fast validation.

load_config_doc is the real loader; load_config_file only reads and parses.
Unknown keys inside an attack's base/sweep are errors: catching a typo here
saves a multi-hour run that would otherwise produce a differently-hashed,
silently-wrong experiment.
"""
```

- [ ] **Step 4: Switch `configs.validate` off the temp file**

In `master_script/webui/configs.py`, change the imports (lines 8-15) to drop `tempfile` and add `yaml`:

```python
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..core.registry import ATTACKS
from ..core.yaml_config import ConfigError, load_config_doc
from ..paths import CONFIGS_DIR
```

`Path` is still used by `safe_path`. Then replace the body of `validate` (lines 79-102) with:

```python
    if len(text.encode("utf-8")) > _MAX_BYTES:
        return {"ok": False, "message": f"Config is larger than {_MAX_BYTES // 1000} KB."}

    try:
        pairs = load_config_doc(yaml.safe_load(text) or {})
    except ConfigError as exc:
        return {"ok": False, "message": str(exc)}
    except Exception as exc:  # malformed YAML surfaces as a parser error
        return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    per_attack: Dict[str, int] = {}
    for _cfg, spec in pairs:
        per_attack[spec.name] = per_attack.get(spec.name, 0) + 1
    return {
        "ok": True,
        "message": f"Valid · expands to {len(pairs)} run(s).",
        "runs": len(pairs),
        "per_attack": per_attack,
    }
```

`load_config_doc`'s default `source="<config>"` is what the old code produced by string-replacing the temp path, so `test_validate_error_does_not_leak_the_temp_path` still passes.

Update the module docstring's second line: `Validation goes through core.yaml_config.load_config_doc -- the same loader`.

- [ ] **Step 5: Run the full suite**

```
python -m pytest -q
```

Expected: all pass, including `tests/test_yaml_config.py`, `tests/test_cli.py`, `tests/test_webui_configs.py` and `tests/test_hash_equivalence.py`.

- [ ] **Step 6: Commit**

```bash
git add master_script/core/yaml_config.py master_script/webui/configs.py tests/test_yaml_config.py tests/test_webui_configs.py
git commit -m "refactor(core): split load_config_doc out of load_config_file

Lets the dashboard validate an in-memory config without inventing a file
or a second expansion path. configs.validate stops writing a temp file."
```

---

### Task 2: Attack field schema

**Files:**
- Create: `master_script/webui/attackfields.py`
- Test: `tests/test_webui_manual.py`

**Interfaces:**
- Consumes: `master_script.core.registry.ATTACKS`, `AttackSpec.config_cls`, `AttackSpec.supports_toy`.
- Produces:
  - `GROUP_ORDER: List[str]` — `["Model & data", "Federation", "Attack", "Advanced"]`
  - `fields_for(name: str) -> List[dict]` — each `{"name": str, "type": "bool"|"int"|"float"|"str", "default": Any, "group": str, "readonly": bool}`
  - `by_name(name: str) -> Dict[str, dict]` — the same field dicts keyed by field name; `{}` for an unknown attack.
  - `schema() -> dict` — `{"group_order": GROUP_ORDER, "attacks": {name: {"supports_toy": bool, "fields": [...]}}}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_webui_manual.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```
python -m pytest tests/test_webui_manual.py -v
```

Expected: FAIL — `ImportError: cannot import name 'attackfields' from 'master_script.webui'`.

- [ ] **Step 3: Write the module**

Create `master_script/webui/attackfields.py`:

```python
# master_script/webui/attackfields.py
"""Attack dataclasses -> a form schema for the Launch page's manual mode.

Pure introspection: no I/O and no dashboard state, so the form can never
offer a field the runner would reject, or miss one it accepts. Grouping is
presentation; the field names and defaults are the dataclasses' own.
"""
from dataclasses import MISSING, fields as dc_fields
from typing import Any, Dict, List, get_type_hints

from ..core.registry import ATTACKS

GROUP_ORDER = ["Model & data", "Federation", "Attack", "Advanced"]

# Only the shared and plumbing names are enumerated. Anything else -- which in
# practice means the attack's own knobs -- falls through to "Attack", so a new
# attack-specific field shows up in the form without editing this table.
_GROUPS = {
    "model_id": "Model & data",
    "dataset_name": "Model & data",
    "max_length": "Model & data",
    "seed": "Model & data",
    "use_hf_models": "Model & data",
    "num_clients": "Federation",
    "clients_per_round": "Federation",
    "federated_rounds": "Federation",
    "local_epochs": "Federation",
    "local_batch_size": "Federation",
    "client_lr": "Federation",
    "target_client_id": "Federation",
    "fl_framework": "Advanced",
    "sim_num_gpus": "Advanced",
    "keep_artifacts": "Advanced",
    "firestore_collection": "Advanced",
    "artifact_root": "Advanced",
    "attack_name": "Advanced",
    "paper_source": "Advanced",
}
_FALLBACK_GROUP = "Attack"

# These two are part of the hashed config, so an edit does not vary the
# experiment -- it renames it into a document nobody will look for.
_READONLY = {"attack_name", "paper_source"}


def _type_name(annotation: Any) -> str:
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    return "str"


def fields_for(name: str) -> List[dict]:
    spec = ATTACKS[name]
    hints = get_type_hints(spec.config_cls)
    out: List[dict] = []
    for field in dc_fields(spec.config_cls):
        default = None if field.default is MISSING else field.default
        out.append({
            "name": field.name,
            "type": _type_name(hints.get(field.name, str)),
            "default": default,
            "group": _GROUPS.get(field.name, _FALLBACK_GROUP),
            "readonly": field.name in _READONLY,
        })

    # amia/loss carry no use_hf_models field at all. Omitting it from the form
    # would read as "the toy path is an option here"; it is not.
    if not spec.supports_toy and not any(f["name"] == "use_hf_models" for f in out):
        out.append({
            "name": "use_hf_models", "type": "bool", "default": True,
            "group": "Model & data", "readonly": True,
        })
    return out


def by_name(name: str) -> Dict[str, dict]:
    """Field dicts keyed by field name. Unknown attacks give {} so callers can
    fall through to the loader's own unknown-attack error."""
    if name not in ATTACKS:
        return {}
    return {field["name"]: field for field in fields_for(name)}


def schema() -> dict:
    return {
        "group_order": GROUP_ORDER,
        "attacks": {
            name: {"supports_toy": spec.supports_toy, "fields": fields_for(name)}
            for name, spec in sorted(ATTACKS.items())
        },
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```
python -m pytest tests/test_webui_manual.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add master_script/webui/attackfields.py tests/test_webui_manual.py
git commit -m "feat(webui): introspect attack dataclasses into a form schema"
```

---

### Task 3: Manual payload → config doc → pairs

**Files:**
- Create: `master_script/webui/manual.py`
- Test: `tests/test_webui_manual.py` (append)

**Interfaces:**
- Consumes: `attackfields.by_name`, `core.yaml_config.load_config_doc`, `core.yaml_config.ConfigError`.
- Payload shape (every value is a string; the front end does no typing):

```python
{
  "attacks": [
    {"name": "samia",
     "values": {"federated_rounds": "4", "seed": "7, 11", "use_hf_models": "true"},
     "sweeps": ["seed"]},
  ],
}
```

- Produces:
  - `class ManualError(ValueError)`
  - `build_doc(payload: dict) -> dict` — raises `ManualError`
  - `to_yaml(doc: dict) -> str`
  - `pairs(payload: dict) -> List[Tuple[object, object]]` — raises `ManualError` / `ConfigError`
  - `validate(payload: dict) -> dict` — `{"ok", "message"}` plus `{"runs", "per_attack", "yaml"}` on success. Never raises.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_webui_manual.py`:

```python
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


def test_read_only_fields_are_never_emitted():
    doc = manual.build_doc(_payload(_attack("zlib", {"attack_name": "not_zlib"})))
    assert doc == {"attacks": {"zlib": {}}}


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
    import yaml as _yaml

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
```

- [ ] **Step 2: Run the tests to verify they fail**

```
python -m pytest tests/test_webui_manual.py -v
```

Expected: FAIL — `ImportError: cannot import name 'manual' from 'master_script.webui'`.

- [ ] **Step 3: Write the module**

Create `master_script/webui/manual.py`:

```python
# master_script/webui/manual.py
"""Manual launch mode: a form payload -> the same doc a YAML config parses to.

Deliberately thin. Everything past build_doc is core.yaml_config's job, so a
manual sweep and a saved-file sweep cannot expand differently, and an error
the CLI would raise reads the same in the browser.
"""
from typing import Any, Dict, List, Tuple

import yaml

from ..core.yaml_config import ConfigError, load_config_doc
from . import attackfields

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


class ManualError(ValueError):
    """Raised for a payload that cannot be turned into a config document."""


def _coerce(raw: Any, type_name: str, attack: str, field: str) -> Any:
    text = str(raw).strip()
    if type_name == "bool":
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise ManualError(f"attack '{attack}': {field} expects true or false, got {text!r}.")
    if type_name in ("int", "float"):
        caster = int if type_name == "int" else float
        try:
            return caster(text)
        except ValueError:
            raise ManualError(
                f"attack '{attack}': {field} expects {'a whole number' if type_name == 'int' else 'a number'}, "
                f"got {text!r}."
            ) from None
    return text


def build_doc(payload: dict) -> dict:
    """Turn the form payload into a config document.

    Only fields the user actually changed, or marked as a sweep, are emitted:
    a manual config stays as small as the equivalent hand-written YAML instead
    of freezing today's defaults into a file.
    """
    attacks: Dict[str, dict] = {}
    for entry in payload.get("attacks") or []:
        name = (entry.get("name") or "").strip()
        if not name:
            raise ManualError("An attack entry has no name.")
        if name in attacks:
            raise ManualError(f"attack '{name}' appears more than once.")

        schema = attackfields.by_name(name)
        sweeps = set(entry.get("sweeps") or [])
        base: Dict[str, Any] = {}
        sweep: Dict[str, list] = {}

        for field, raw in (entry.get("values") or {}).items():
            info = schema.get(field)
            if info is not None and info["readonly"]:
                continue
            # An unknown field keeps its text and travels to the loader, which
            # rejects it by name and lists what would have been valid.
            type_name = info["type"] if info else "str"

            if field in sweeps:
                items = [part for part in str(raw).split(",") if part.strip()]
                if not items:
                    raise ManualError(f"attack '{name}': sweep {field} has no values.")
                sweep[field] = [_coerce(part, type_name, name, field) for part in items]
                continue

            value = _coerce(raw, type_name, name, field)
            if info is not None and value == info["default"]:
                continue  # unchanged
            base[field] = value

        section: Dict[str, dict] = {}
        if base:
            section["base"] = base
        if sweep:
            section["sweep"] = sweep
        attacks[name] = section

    if not attacks:
        raise ManualError("Add at least one attack before starting a sweep.")
    return {"attacks": attacks}


def to_yaml(doc: dict) -> str:
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def pairs(payload: dict) -> List[Tuple[object, object]]:
    return load_config_doc(build_doc(payload), source="<manual>")


def validate(payload: dict) -> dict:
    """Expand without running or writing. Never raises."""
    try:
        doc = build_doc(payload)
        expanded = load_config_doc(doc, source="<manual>")
    except (ManualError, ConfigError) as exc:
        return {"ok": False, "message": str(exc)}
    except Exception as exc:
        return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    per_attack: Dict[str, int] = {}
    for _cfg, spec in expanded:
        per_attack[spec.name] = per_attack.get(spec.name, 0) + 1
    return {
        "ok": True,
        "message": f"Valid · expands to {len(expanded)} run(s).",
        "runs": len(expanded),
        "per_attack": per_attack,
        "yaml": to_yaml(doc),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```
python -m pytest tests/test_webui_manual.py -v
```

Expected: all pass. If `test_a_manual_run_equals_running_the_yaml_it_would_save` fails on the config comparison, the cause is a coercion producing a different type than YAML's parser (e.g. `"1e-4"` staying a string) — fix `_coerce`, not the test.

- [ ] **Step 5: Commit**

```bash
git add master_script/webui/manual.py tests/test_webui_manual.py
git commit -m "feat(webui): build config docs from a manual form payload

Manual mode reaches load_config_doc rather than owning expansion, so a
manual run and the YAML it would save as produce identical pairs."
```

---

### Task 4: Shared start path and `start_manual`

**Files:**
- Modify: `master_script/webui/launch.py:83-109`
- Test: `tests/test_webui_launch.py` (append)

**Interfaces:**
- Consumes: `manual.pairs`, `manual.ManualError`.
- Produces:
  - `_start(pairs, use_firestore: bool, empty_message: str) -> dict`
  - `start_sweep(config_file, attacks=None, use_firestore=True) -> dict` — unchanged signature and messages.
  - `start_manual(payload: dict, use_firestore: bool = True) -> dict`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_webui_launch.py`:

```python
def test_manual_start_publishes_a_manifest_and_starts_the_worker(monkeypatch):
    import master_script.webui.launch as mod

    published, started = {}, {}
    monkeypatch.setattr(mod, "publish_manifest", lambda pairs: published.setdefault("n", len(pairs)))
    monkeypatch.setattr(mod.WORKER, "start", lambda pairs, **kw: started.setdefault("n", len(pairs)) or True)
    monkeypatch.setattr(type(mod.WORKER), "is_running", property(lambda self: False))

    result = mod.start_manual({"attacks": [
        {"name": "zlib", "values": {"seed": "7, 11"}, "sweeps": ["seed"]},
    ]})
    assert result["ok"] is True and result["planned"] == 2
    assert published["n"] == 2 and started["n"] == 2


def test_manual_start_reports_a_bad_value_without_starting(monkeypatch):
    import master_script.webui.launch as mod

    started = {}
    monkeypatch.setattr(mod.WORKER, "start", lambda *a, **k: started.setdefault("called", True))

    result = mod.start_manual({"attacks": [
        {"name": "zlib", "values": {"federated_rounds": "three"}, "sweeps": []},
    ]})
    assert result["ok"] is False and "federated_rounds" in result["message"]
    assert "called" not in started


def test_manual_start_with_no_attacks_is_refused(monkeypatch):
    import master_script.webui.launch as mod

    started = {}
    monkeypatch.setattr(mod.WORKER, "start", lambda *a, **k: started.setdefault("called", True))
    result = mod.start_manual({"attacks": []})
    assert result["ok"] is False and "called" not in started


def test_a_second_start_publishes_no_manifest(monkeypatch):
    """A manifest published for a sweep that never starts would give the
    monitor a denominator for runs that will never arrive."""
    import master_script.webui.launch as mod

    published = {}
    monkeypatch.setattr(mod, "publish_manifest", lambda pairs: published.setdefault("called", True))
    monkeypatch.setattr(mod, "load_config_file", lambda path, only=None: [("cfg", "spec")])
    monkeypatch.setattr(type(mod.WORKER), "is_running", property(lambda self: True))

    result = mod.start_sweep("smoke.yaml")
    assert result["ok"] is False and "already running" in result["message"]
    assert "called" not in published
```

- [ ] **Step 2: Run the tests to verify they fail**

```
python -m pytest tests/test_webui_launch.py -v
```

Expected: the three `start_manual` tests fail with `AttributeError: module ... has no attribute 'start_manual'`; `test_a_second_start_publishes_no_manifest` fails because `publish_manifest` is called before the running check.

- [ ] **Step 3: Refactor `launch.py`**

Replace `start_sweep` (lines 83-109) with:

```python
def _start(pairs, use_firestore: bool, empty_message: str) -> dict:
    """Publish the plan and hand the pairs to the worker. Never raises."""
    if not pairs:
        # Starting nothing must not read as success.
        return {"ok": False, "message": empty_message}

    # Checked before publish_manifest: a manifest for a sweep that never starts
    # would give the monitor a denominator for runs that will never arrive.
    if WORKER.is_running:
        return {"ok": False, "message": "A sweep is already running."}

    # The plan goes out before the first run, so the sweep bars have a
    # denominator from the moment the first run appears.
    if use_firestore:
        publish_manifest(pairs)

    # Per-run reporting: every run announces itself and clears on the way out,
    # so the running set is the run actually in flight rather than run 1 forever.
    hooks = REPORTER.hooks if use_firestore else {}

    if not WORKER.start(pairs, use_firestore=use_firestore, **hooks):
        return {"ok": False, "message": "A sweep is already running."}

    return {"ok": True, "message": f"Started {len(pairs)} run(s).", "planned": len(pairs)}


def start_sweep(config_file: str, attacks: Optional[List[str]] = None,
                use_firestore: bool = True) -> dict:
    """Load a config file, publish the plan, and start the sweep."""
    try:
        pairs = load_config_file(CONFIGS_DIR / config_file, only=attacks or None)
    except ConfigError as exc:
        return {"ok": False, "message": f"Config error: {exc}"}

    return _start(pairs, use_firestore, (
        f"No runs to start: {config_file} defines none of the selected attack(s)."
    ))


def start_manual(payload: dict, use_firestore: bool = True) -> dict:
    """Start a sweep from the manual form, with no file in between."""
    from . import manual

    try:
        pairs = manual.pairs(payload)
    except (manual.ManualError, ConfigError) as exc:
        return {"ok": False, "message": f"Config error: {exc}"}

    return _start(pairs, use_firestore, "No runs to start: the manual configuration is empty.")
```

The `from . import manual` is function-local to match the existing `launch_payload`'s deferred `from . import configs`, which keeps `webui` import order free of cycles.

- [ ] **Step 4: Run the tests to verify they pass**

```
python -m pytest tests/test_webui_launch.py tests/test_webui_manual.py -v
```

Expected: all pass, including the pre-existing `test_selecting_an_attack_the_config_lacks_is_refused_not_reported_as_started`.

- [ ] **Step 5: Commit**

```bash
git add master_script/webui/launch.py tests/test_webui_launch.py
git commit -m "feat(webui): start a sweep from a manual payload

Factors the publish/start sequence into _start so the file and manual
paths share their guards, and moves the already-running check ahead of
publish_manifest."
```

---

### Task 5: API routes

**Files:**
- Modify: `master_script/webui/api.py:13` (imports), `:124-137` (launch section)
- Test: `tests/test_webui_manual.py` (append)

**Interfaces:**
- Consumes: `attackfields.schema`, `manual.validate`, `launch.start_manual`.
- Produces:
  - `GET /api/attacks/fields` → `attackfields.schema()`
  - `POST /api/launch/manual/validate` → `manual.validate(...)`
  - `POST /api/launch/manual` → `launch.start_manual(...)`
  - Request model `ManualRequest {attacks: [{name: str, values: dict[str,str], sweeps: list[str]}], use_firestore: bool = True}`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_webui_manual.py`:

```python
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


def test_manual_validate_writes_nothing(monkeypatch):
    """Validation must never leave a config behind."""
    from master_script.webui import configs

    before = sorted(p.name for p in configs.CONFIGS_DIR.glob("*.yaml"))
    _client().post("/api/launch/manual/validate", json={"attacks": [{"name": "zlib"}]})
    assert sorted(p.name for p in configs.CONFIGS_DIR.glob("*.yaml")) == before


def test_manual_start_delegates_to_launch(monkeypatch):
    from master_script.webui import api as api_mod

    seen = {}
    monkeypatch.setattr(api_mod.launch, "start_manual",
                        lambda payload, use_firestore: seen.setdefault("call", (payload, use_firestore))
                        or {"ok": True, "message": "Started 1 run(s)."})
    res = _client().post("/api/launch/manual", json={
        "attacks": [{"name": "zlib"}], "use_firestore": False,
    })
    assert res.status_code == 200 and res.json()["ok"] is True
    payload, use_firestore = seen["call"]
    assert use_firestore is False
    assert payload["attacks"][0]["name"] == "zlib"
```

- [ ] **Step 2: Run the tests to verify they fail**

```
python -m pytest tests/test_webui_manual.py -v -k "field_schema or manual_validate or manual_start_delegates"
```

Expected: FAIL with 404 responses for the new paths.

- [ ] **Step 3: Add the routes**

In `master_script/webui/api.py`, extend the import on line 13:

```python
from . import attackfields, configs, envfile, launch, localguard, manual, monitor, results, tunnel
```

Then, immediately after the existing `launch_start` (line 137), add:

```python
# ---------- manual launch mode ----------
#
# The form payload is all strings; typing happens in manual.build_doc against
# the attack's own dataclass, not here.

@router.get("/attacks/fields")
def attack_fields():
    return attackfields.schema()


class ManualAttack(BaseModel):
    name: str
    values: Dict[str, str] = {}
    sweeps: List[str] = []


class ManualRequest(BaseModel):
    attacks: List[ManualAttack] = []
    use_firestore: bool = True


@router.post("/launch/manual/validate")
def launch_manual_validate(body: ManualRequest):
    """Expand the form exactly as a start would. Writes nothing, runs nothing."""
    return manual.validate(body.model_dump())


@router.post("/launch/manual")
def launch_manual_start(body: ManualRequest):
    return launch.start_manual(body.model_dump(), body.use_firestore)
```

Add `Dict` to the typing import on line 8:

```python
from typing import Dict, List, Optional
```

- [ ] **Step 4: Run the tests to verify they pass**

```
python -m pytest tests/test_webui_manual.py tests/test_webui_pages.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add master_script/webui/api.py tests/test_webui_manual.py
git commit -m "feat(webui): serve the attack field schema and manual launch routes"
```

---

### Task 6: Launch page — mode switch and manual panel

**Files:**
- Modify: `master_script/webui/static/app.js` — state block `:41-43`, `refresh()` `:152-156`, `launchView()` `:946-995`, `onAction()` `:1332-1338,1395-1403`, `onInput()` `:1414-1438`

**Interfaces:**
- Consumes: `GET /api/attacks/fields`, `POST /api/launch/manual/validate`, `POST /api/launch/manual`, `POST /api/configs/save`.
- Produces: no exports; this file is the whole front end.

There is no JS test harness in this repo, so this task is verified by running the dashboard and exercising the page. Do not add one.

- [ ] **Step 1: Add the state**

In the state block (`app.js:41-43`), change `launchForm` and add two keys:

```js
  launchForm: { mode: 'manual', configFile: null, attacks: [], useFirestore: true },
  // Manual mode. `values` holds only what the user typed; an untouched field
  // is absent, so the payload stays as small as hand-written YAML.
  // `validation` is the last server answer; `dirty` greys it rather than
  // clearing it, so a stale run count is never shown as current.
  manual: { cards: [], validation: null, dirty: false, saveName: '', confirmOverwrite: false },
  attackFields: null,
```

A card is `{ name, values: {}, sweeps: [], advanced: false }`.

- [ ] **Step 2: Fetch the schema when the Launch view loads**

In `refresh()`, replace the `else if (S.view === 'launch')` branch (`app.js:153-156`) with:

```js
    } else if (S.view === 'launch') {
      S.launch = await getJSON('/api/launch');
      if (!S.attackFields) S.attackFields = await getJSON('/api/attacks/fields');
      if (!S.launchForm.configFile && S.launch.configs.length) S.launchForm.configFile = S.launch.configs[0];
      if (S.editor.name === null && !S.editor.dirty && S.launchForm.configFile) await loadConfig(S.launchForm.configFile);
    }
```

The schema is static for the process's lifetime, so it is fetched once rather than on every 2.5s poll.

- [ ] **Step 3: Add the manual panel renderer**

Insert this immediately above `// ---------- launch view ----------` (`app.js:946`):

```js
// ---------- manual launch mode ----------
function manualField(card, field) {
  const key = `manual:${card.name}:${field.name}`;
  const swept = card.sweeps.includes(field.name);
  const raw = Object.prototype.hasOwnProperty.call(card.values, field.name)
    ? card.values[field.name] : String(field.default);
  const label = `<div style="width:170px;flex:0 0 auto;font-size:11.5px;font-family:${MONO};color:var(--fd,#9aa6b2)">${esc(field.name)}</div>`;

  if (field.readonly) {
    return `<div style="display:flex;align-items:center;gap:10px;padding:5px 0">${label}
      <div style="flex:1;font-size:11.5px;font-family:${MONO};color:var(--fm,#5f6b78);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(field.default)}">${esc(field.default)}</div>
      <div style="font-size:10px;color:var(--fm,#5f6b78);letter-spacing:.06em;text-transform:uppercase">${field.name === 'use_hf_models' ? 'no toy path' : 'fixed'}</div>
    </div>`;
  }

  const input = (field.type === 'bool' && !swept)
    ? `<div data-act="manual-bool" data-arg="${esc(card.name)}:${esc(field.name)}" style="flex:1;display:flex;align-items:center;cursor:pointer">
         <div style="width:34px;height:19px;border-radius:11px;position:relative;transition:.15s;background:${raw === 'true' ? 'var(--ac,#36c08f)' : 'var(--gd,#222a34)'}">
           <div style="position:absolute;top:2px;left:${raw === 'true' ? '17px' : '2px'};width:15px;height:15px;border-radius:50%;background:#fff;transition:.15s"></div>
         </div>
       </div>`
    : `<input data-inp="${esc(key)}" value="${esc(raw)}" spellcheck="false"
         style="flex:1;font-size:11.5px;font-family:${MONO};padding:6px 10px;border-radius:7px;border:1px solid ${swept ? 'var(--rn,#4aa8ff)' : 'var(--bd,#252c36)'};background:var(--bg,#0d1014);color:var(--fg,#e6ebf0);outline:none">`;

  const count = swept ? String(raw).split(',').filter((p) => p.trim()).length : 0;
  return `<div style="display:flex;align-items:center;gap:10px;padding:5px 0">${label}${input}
    <div data-act="manual-sweep" data-arg="${esc(card.name)}:${esc(field.name)}" data-hover
      title="${swept ? 'back to a single value' : 'sweep this field over a comma-separated list'}"
      style="flex:0 0 auto;font-size:10px;font-weight:600;font-family:${MONO};padding:4px 8px;border-radius:6px;cursor:pointer;border:1px solid ${swept ? 'var(--rn,#4aa8ff)' : 'var(--bd,#252c36)'};color:${swept ? 'var(--rn,#4aa8ff)' : 'var(--fm,#5f6b78)'}">${swept ? `sweep ×${count}` : '⋯'}</div>
  </div>`;
}

function manualCard(card) {
  const info = S.attackFields.attacks[card.name];
  if (!info) return '';
  const m = meta(card.name);
  const groups = S.attackFields.group_order.map((group) => {
    const rows = info.fields.filter((f) => f.group === group);
    if (!rows.length) return '';
    const body = rows.map((f) => manualField(card, f)).join('');
    if (group !== 'Advanced') {
      return `<div style="padding:12px 18px;border-top:1px solid var(--bd,#252c36)">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fm,#5f6b78);margin-bottom:6px">${esc(group)}</div>
        ${body}</div>`;
    }
    // Plumbing: correct by default and rarely the thing being tuned.
    return `<div style="padding:12px 18px;border-top:1px solid var(--bd,#252c36)">
      <div data-act="manual-advanced" data-arg="${esc(card.name)}" style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fm,#5f6b78);cursor:pointer">${card.advanced ? '▾' : '▸'} Advanced</div>
      ${card.advanced ? `<div style="margin-top:6px">${body}</div>` : ''}</div>`;
  }).join('');

  return `<div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden;margin-bottom:12px">
    <div style="display:flex;align-items:center;gap:10px;padding:12px 18px">
      <div style="width:8px;height:8px;border-radius:2px;background:${m.color}"></div>
      <div style="font-size:12.5px;font-weight:700;font-family:${MONO}">${esc(card.name)}</div>
      <div style="flex:1"></div>
      <div data-act="manual-remove" data-arg="${esc(card.name)}" data-hover style="font-size:11px;cursor:pointer;color:var(--fm,#5f6b78);padding:3px 8px;border-radius:6px">✕</div>
    </div>
    ${groups}
  </div>`;
}

function manualPanel() {
  if (!S.attackFields) return loadingView('attack fields');
  const mn = S.manual;
  const taken = mn.cards.map((c) => c.name);
  const addable = Object.keys(S.attackFields.attacks).filter((a) => !taken.includes(a));
  const v = mn.validation;
  const readout = !v
    ? `<span style="color:var(--fm,#5f6b78)">not validated yet</span>`
    : v.ok
      ? `<span style="color:${mn.dirty ? 'var(--fm,#5f6b78)' : 'var(--ok,#3fcf8e)'}">${mn.dirty ? '·' : '✓'} ${esc(v.message)}${mn.dirty ? ' (edited since)' : ''}</span>`
      : `<span style="color:var(--no,#f0606a);white-space:pre-wrap">✗ ${esc(v.message)}</span>`;

  return `<div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
      <select data-inp="manualAdd" style="${SEL};font-size:12px;padding:8px 12px;min-width:220px">
        <option value="">+ add attack</option>
        ${addable.map((a) => `<option value="${esc(a)}">${esc(a)}</option>`).join('')}
      </select>
      <div style="font-size:11.5px;color:var(--fm,#5f6b78)">Edit a field to change it; ⋯ turns a field into a comma-separated sweep.</div>
    </div>

    ${mn.cards.length ? mn.cards.map(manualCard).join('')
      : `<div style="background:var(--pn,#15191f);border:1px dashed var(--bd,#252c36);border-radius:12px;padding:32px;text-align:center;font-size:12px;color:var(--fm,#5f6b78);margin-bottom:12px">No attacks yet. Add one above.</div>`}

    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden">
      <div style="display:flex;align-items:center;gap:10px;padding:13px 20px;flex-wrap:wrap">
        <div data-act="manual-validate" data-hover style="font-size:11.5px;font-weight:600;cursor:pointer;padding:7px 14px;border-radius:8px;background:var(--p2,#1b212a);border:1px solid var(--bd,#252c36);color:var(--fd,#9aa6b2)">Validate</div>
        <span style="font-size:10px;color:var(--fm,#5f6b78);letter-spacing:.08em;text-transform:uppercase;margin-left:8px">save as</span>
        <input data-inp="manualSaveName" value="${esc(mn.saveName)}" placeholder="my-sweep.yaml"
          style="font-size:12px;font-family:${MONO};padding:7px 11px;border-radius:8px;border:1px solid var(--bd,#252c36);background:var(--p2,#1b212a);color:var(--fg,#e6ebf0);width:200px;outline:none">
        <div data-act="manual-save" data-hover style="font-size:11.5px;font-weight:600;cursor:pointer;padding:7px 14px;border-radius:8px;background:var(--p2,#1b212a);border:1px solid var(--bd,#252c36);color:var(--fd,#9aa6b2)">${mn.confirmOverwrite ? 'Overwrite' : 'Save as config'}</div>
        ${mn.confirmOverwrite ? `<span style="font-size:11px;color:var(--wn,#e3b341)">file exists — Overwrite replaces it</span>` : ''}
      </div>
      <div style="padding:11px 20px;background:var(--p2,#1b212a);border-top:1px solid var(--bd,#252c36);font-size:11.5px;font-family:${MONO}">${readout}</div>
    </div>
  </div>`;
}
```

- [ ] **Step 4: Rewrite `launchView()` around the mode switch**

Replace `launchView()` (`app.js:946-995`) with:

```js
// ---------- launch view ----------
function launchView() {
  if (!S.launch) return loadingView('launch options');
  const w = S.launch.worker;
  const f = S.launchForm;
  const manualMode = f.mode === 'manual';

  const tab = (mode, label, hint) => `<div data-act="launch-mode" data-arg="${mode}" data-hover
    style="flex:1;padding:11px 16px;cursor:pointer;border-radius:9px;background:${f.mode === mode ? 'var(--p2,#1b212a)' : 'transparent'};border:1px solid ${f.mode === mode ? 'var(--ac,#36c08f)' : 'transparent'}">
    <div style="font-size:12.5px;font-weight:600;color:${f.mode === mode ? 'var(--fg,#e6ebf0)' : 'var(--fm,#5f6b78)'}">${label}</div>
    <div style="font-size:11px;color:var(--fm,#5f6b78);margin-top:2px">${hint}</div></div>`;

  const configs = S.launch.configs.map((c) =>
    `<option value="${esc(c)}"${c === f.configFile ? ' selected' : ''}>${esc(c)}</option>`).join('');
  const chips = S.launch.attacks.map((a) => {
    const on = f.attacks.includes(a);
    const m = meta(a);
    return `<div data-act="launch-attack" data-arg="${esc(a)}" style="font-size:11px;font-weight:600;font-family:${MONO};padding:5px 11px;border-radius:6px;cursor:pointer;border:1px solid ${on ? m.color : 'var(--bd,#252c36)'};color:${on ? '#0d1014' : 'var(--fm,#5f6b78)'};background:${on ? m.color : 'transparent'}">${esc(a)}</div>`;
  }).join('');

  const status = w.running
    ? `<span style="color:var(--rn,#4aa8ff)">● running · ${w.finished}/${w.planned} finished · ${elapsedFmt(w.started_unix ? now() - w.started_unix : null)} elapsed</span>`
    : w.error ? `<span style="color:var(--no,#f0606a)">● last sweep failed: ${esc(w.error)}</span>`
    : w.planned ? `<span style="color:var(--ok,#3fcf8e)">● idle · last sweep finished ${w.finished}/${w.planned}</span>`
    : `<span style="color:var(--fm,#5f6b78)">● idle · no sweep started this session</span>`;

  const existingBody = `${editorPanel()}
    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden;margin-bottom:12px">
      <div style="padding:18px 20px;border-bottom:1px solid var(--bd,#252c36)">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:12px">Run this config</div>
        <select data-inp="configFile" style="${SEL};font-size:12px;padding:9px 12px;width:280px">${configs || '<option>no configs found</option>'}</select>
        ${S.editor.dirty ? `<div style="margin-top:10px;font-size:11px;color:var(--wn,#e3b341)">The editor has unsaved changes. A sweep runs the file on disk — save first to run what you're looking at.</div>` : ''}
      </div>
      <div style="padding:18px 20px">
        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--fd,#9aa6b2);margin-bottom:12px">Attacks <span style="color:var(--fm,#5f6b78);font-weight:500;letter-spacing:0;text-transform:none">· none selected = every attack in the config</span></div>
        <div style="display:flex;flex-wrap:wrap;gap:7px">${chips}</div>
      </div>
    </div>`;

  return `<div style="max-width:900px;margin:0 auto">
    <div style="margin-bottom:16px">
      <div style="font-size:20px;font-weight:700;letter-spacing:-.01em">Launch a sweep</div>
      <div style="font-size:12px;color:var(--fd,#9aa6b2);margin-top:4px;max-width:560px;line-height:1.6">Runs go through <span style="font-family:${MONO}">core.runner.run_sweep</span> — the same entry point <span style="font-family:${MONO}">perform_experiments.py</span> uses, so the dashboard cannot drift from the CLI.</div>
    </div>

    <div style="display:flex;gap:8px;background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;padding:6px;margin-bottom:14px">
      ${tab('manual', 'Manual', 'tune an attack field by field')}
      ${tab('existing', 'Existing config', 'run a saved .yaml')}
    </div>

    ${manualMode ? manualPanel() : existingBody}

    <div style="background:var(--pn,#15191f);border:1px solid var(--bd,#252c36);border-radius:12px;overflow:hidden;margin-top:12px">
      <div style="padding:18px 20px;display:flex;align-items:center;gap:14px">
        <div data-act="launch-firestore" style="display:flex;align-items:center;gap:9px;cursor:pointer;font-size:12px;color:var(--fd,#9aa6b2)">
          <div style="width:34px;height:19px;border-radius:11px;position:relative;transition:.15s;background:${f.useFirestore ? 'var(--ac,#36c08f)' : 'var(--gd,#222a34)'}">
            <div style="position:absolute;top:2px;left:${f.useFirestore ? '17px' : '2px'};width:15px;height:15px;border-radius:50%;background:#fff;transition:.15s"></div>
          </div>persist to Firestore
        </div>
        <div style="flex:1"></div>
        <div data-act="${manualMode ? 'manual-start' : 'launch-start'}" data-hover style="font-size:12px;font-weight:600;cursor:pointer;padding:9px 18px;border-radius:9px;background:${w.running ? 'var(--gd,#222a34)' : 'var(--ac,#36c08f)'};color:${w.running ? 'var(--fm,#5f6b78)' : '#0d1014'};border:1px solid ${w.running ? 'var(--bd,#252c36)' : 'var(--ac,#36c08f)'}">${w.running ? 'Sweep in progress' : 'Start sweep'}</div>
      </div>
      <div style="padding:13px 20px;background:var(--p2,#1b212a);border-top:1px solid var(--bd,#252c36);font-size:11.5px;font-family:${MONO}">${status}</div>
    </div>
  </div>`;
}
```

- [ ] **Step 5: Add the actions**

In `onAction()`, insert immediately after the `launch-firestore` case (`app.js:1338`):

```js
    case 'launch-mode': S.launchForm.mode = arg; break;
    case 'manual-remove': {
      S.manual.cards = S.manual.cards.filter((c) => c.name !== arg);
      S.manual.dirty = true;
      break;
    }
    case 'manual-advanced': {
      const card = S.manual.cards.find((c) => c.name === arg);
      if (card) card.advanced = !card.advanced;
      break;
    }
    case 'manual-bool': {
      const [name, field] = arg.split(':');
      const card = S.manual.cards.find((c) => c.name === name);
      if (!card) break;
      const info = S.attackFields.attacks[name].fields.find((f) => f.name === field);
      const current = Object.prototype.hasOwnProperty.call(card.values, field)
        ? card.values[field] : String(info.default);
      card.values[field] = current === 'true' ? 'false' : 'true';
      S.manual.dirty = true;
      break;
    }
    case 'manual-sweep': {
      const [name, field] = arg.split(':');
      const card = S.manual.cards.find((c) => c.name === name);
      if (!card) break;
      const i = card.sweeps.indexOf(field);
      if (i >= 0) {
        // Leaving sweep mode keeps the first value rather than the whole list.
        card.sweeps.splice(i, 1);
        const first = String(card.values[field] || '').split(',')[0].trim();
        if (first) card.values[field] = first;
      } else {
        card.sweeps.push(field);
        const info = S.attackFields.attacks[name].fields.find((f) => f.name === field);
        if (!Object.prototype.hasOwnProperty.call(card.values, field)) {
          card.values[field] = String(info.default);
        }
      }
      S.manual.dirty = true;
      break;
    }
    case 'manual-validate': {
      S.manual.validation = await postJSON('/api/launch/manual/validate', manualBody());
      S.manual.dirty = false;
      break;
    }
    case 'manual-save': {
      const mn = S.manual;
      if (!mn.saveName.trim()) { S.banner = 'Give the config a filename before saving.'; break; }
      // The YAML comes from the server's own build_doc, so the file saved is
      // byte-for-byte the config the manual run would have expanded.
      const check = await postJSON('/api/launch/manual/validate', manualBody());
      mn.validation = check;
      mn.dirty = false;
      if (!check.ok) { S.banner = check.message; break; }
      const res = await postJSON('/api/configs/save', {
        name: mn.saveName, text: check.yaml, overwrite: mn.confirmOverwrite,
      });
      S.banner = res.message;
      if (res.ok) {
        mn.confirmOverwrite = false;
        mn.saveName = res.name;
        S.launch = await getJSON('/api/launch');
      } else {
        mn.confirmOverwrite = !!res.exists;  // re-clicking Save confirms
      }
      break;
    }
    case 'manual-start': {
      const res = await postJSON('/api/launch/manual', {
        ...manualBody(), use_firestore: S.launchForm.useFirestore,
      });
      S.banner = res.message;
      S.launch = await getJSON('/api/launch');
      break;
    }
```

Add the payload builder immediately above `manualField` in the manual-panel section:

```js
function manualBody() {
  return {
    attacks: S.manual.cards.map((c) => ({ name: c.name, values: c.values, sweeps: c.sweeps })),
  };
}
```

- [ ] **Step 6: Add the inputs**

In `onInput()`, insert before the `switch` (alongside the existing `env:` prefix check at `app.js:1410`):

```js
  if (key.startsWith('manual:')) {
    const [, name, field] = key.split(':');
    const card = S.manual.cards.find((c) => c.name === name);
    if (card) { card.values[field] = value; S.manual.dirty = true; }
    return;  // typing in a field must not rebuild the field
  }
```

Then add two cases inside the `switch`, after `case 'saveName'` (`app.js:1427`):

```js
    case 'manualSaveName':
      S.manual.saveName = value;
      S.manual.confirmOverwrite = false;
      return;
    case 'manualAdd':
      if (!value || S.manual.cards.some((c) => c.name === value)) break;
      S.manual.cards.push({ name: value, values: {}, sweeps: [], advanced: false });
      S.manual.dirty = true;
      break;
```

- [ ] **Step 7: Verify the page by hand**

```
python -m master_script.webui.app
```

(If that entry point differs, check `master_script/webui/app.py`'s `__main__` block or the README for the documented command.) Open `http://127.0.0.1:8000/launch` and confirm:

1. Two tabs appear; **Manual** is selected by default.
2. Adding `samia` renders four groups with Advanced collapsed; expanding Advanced shows `attack_name` and `paper_source` as fixed text, not inputs.
3. Adding `amia` shows `use_hf_models` as fixed with the "no toy path" note.
4. Editing `federated_rounds` and clicking **Validate** reports `expands to 1 run(s)`.
5. Clicking `⋯` on `seed`, typing `7, 11, 23`, and validating reports 3 runs and the badge reads `sweep ×3`.
6. Typing `abc` into `federated_rounds` and validating shows a red message naming the field; the previous run count is not shown as current.
7. **Save as config** with a name writes the file and it appears in the Existing-config dropdown; opening it in the editor shows only the fields you changed.
8. Switching to **Existing config** shows the unchanged editor + dropdown + chips, and Start still runs the selected file.
9. Toggling Firestore off and starting a manual sweep produces a "Started N run(s)" banner.

- [ ] **Step 8: Run the full suite**

```
python -m pytest -q
```

Expected: all pass, including `test_static_assets_are_served_locally`.

- [ ] **Step 9: Commit**

```bash
git add master_script/webui/static/app.js
git commit -m "feat(webui): manual and existing-config modes on the Launch page

Manual mode builds a typed form over each attack's dataclass fields, with
per-field sweeps and an optional Save as config; existing-config mode keeps
today's editor and run panel unchanged."
```

---

### Task 7: Document the new mode

**Files:**
- Modify: `master_script/README.md`

**Interfaces:**
- Consumes: everything above. Produces: nothing.

- [ ] **Step 1: Read the current README's Launch/webui section**

```
grep -n -i "launch\|config" master_script/README.md
```

- [ ] **Step 2: Add the manual-mode description**

In the section covering the dashboard's Launch page, document both modes:

- **Manual** — pick one or more attacks and edit their dataclass fields directly. Any field can be turned into a comma-separated sweep, which expands into a grid exactly as `sweep:` does in YAML. Runs without saving; **Save as config** writes the equivalent `.yaml` into `master_script/configs/`.
- **Existing config** — browse, edit, validate and run a saved config file. Unchanged.

State the guarantee explicitly: both modes go through `core.yaml_config.load_config_doc` and then `core.runner.run_sweep`, so a manual run and the config it saves as expand to the same runs.

Match the README's existing heading level, voice, and line width.

- [ ] **Step 3: Check the docs test still passes**

```
python -m pytest tests/test_docs.py -v
```

Expected: PASS. (`tests/test_docs.py` asserts documentation invariants; if it fails, it is telling you the README claims something the code does not do — fix the README.)

- [ ] **Step 4: Commit**

```bash
git add master_script/README.md
git commit -m "docs(master_script): describe the Launch page's manual mode"
```

---

## Self-Review Notes

Spec coverage checked section by section:

| Spec section | Task |
|---|---|
| §3 mode switch, §3.1 existing mode unchanged | 6 |
| §3.2 manual panel, groups, sweep control, read-only, locked use_hf_models | 2 (schema), 6 (render) |
| §4.1 `load_config_doc` | 1 |
| §4.2 `attackfields.py` | 2 |
| §4.3 `manual.py` | 3 |
| §4.4 `launch._start` / `start_manual` | 4 |
| §4.5 three routes, Save reusing `/api/configs/save` | 5, 6 |
| §4.6 `configs.validate` cleanup | 1 |
| §5 round-trip property | 3 |
| §6 error handling incl. greyed stale count | 3 (messages), 4 (guards), 6 (grey) |
| §7 testing | 1, 2, 3, 4, 5 |

Task 7 is beyond the spec: the README documents the Launch page today and would otherwise describe a page that no longer exists.
