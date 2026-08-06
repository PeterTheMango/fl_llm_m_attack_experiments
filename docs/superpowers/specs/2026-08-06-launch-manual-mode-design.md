# Launch Page — Manual Mode + Existing Config — Design

**Date:** 2026-08-06
**Status:** Approved design. Implementation plan follows separately.

Splits the dashboard's Launch page into two modes: a **Manual** attack-configuration
builder for fine-tuning attacks from the browser, and **Existing config**, which is the
page's current behavior (pick a saved YAML file, optionally edit it, run it).

---

## 1. Problem

Today `master_script/webui/launch.py` can only start a sweep from a YAML file already on
disk. Tuning one field means editing YAML by hand in a textarea, saving it under some name,
then selecting that name in a dropdown. That is a poor fit for the common case — trying one
attack with a few adjusted values — and it forces a file into existence for every throwaway
experiment.

Manual mode gives a typed form over the attack's own dataclass fields, runs it directly, and
makes saving it as a config an optional follow-up rather than a precondition.

## 2. Scope

In scope:

- A mode switch on the Launch page with two modes.
- Manual mode: multiple attacks per sweep, each with its own field values, and per-field
  sweep lists that expand into a grid.
- Running a manual configuration without saving it.
- Saving a manual configuration as a `.yaml` config file, reusing the existing save endpoint.
- The refactor of `core/yaml_config.py` needed to share expansion between both modes.

Out of scope: any change to `core/runner.py`, the attack dataclasses, the experiment-key
formulas, or the Monitor/Results/Access/Settings pages.

## 3. UI

A segmented control at the top of the Launch page: **Manual** | **Existing config**.
Mode lives in front-end state only (`S.launchForm.mode`); it is not persisted.

### 3.1 Existing config mode

Unchanged from today: the YAML editor panel (`editorPanel()` — select a file, edit, validate,
save, overwrite-confirm) above the run panel (config dropdown, attack chips narrowing which
attacks in the file to run, Firestore toggle, Start, status readout).

### 3.2 Manual mode

```
[ + add attack ▾ ]        (menu of the 11 attacks, minus ones already added)

┌ samia ────────────────────────────────────── ✕ ┐
│ Model & data      model_id, dataset_name, max_length, seed, use_hf_models
│ Federation        num_clients, clients_per_round, federated_rounds,
│                   local_epochs, local_batch_size, client_lr, target_client_id
│ Attack            attack_trials, num_samples, rouge_n,
│                   use_zlib_weighting, threshold
│ ▸ Advanced        fl_framework, sim_num_gpus, keep_artifacts,
│                   firestore_collection, artifact_root,
│                   attack_name (read-only), paper_source (read-only)
└──────────────────────────────────────────────┘

⚡ expands to 3 runs        [ save as: ____.yaml ] [ Save ]   [ Start sweep ]
```

Each field row is `label · input · ⋯`:

- The `⋯` control flips the row into sweep mode. The input becomes a comma-separated list and
  the row is badged `sweep ×N`. Flipping back keeps the first value.
- Bool fields render as toggles, int/float as number inputs, str as text inputs.
- `attack_name` and `paper_source` render read-only. They feed the experiment key, so editing
  them produces a differently-hashed, silently-wrong run rather than a useful variation.
- For attacks with `supports_toy=False` (`amia`, `loss`), `use_hf_models` renders as a locked
  on toggle with the reason inline. It is a virtual field there — the dataclasses have no such
  field — matching the existing handling in `load_config_file`.
- Advanced starts collapsed. Its membership is fixed by name (see §4.2), not inferred.

Run count (`expands to N runs`) comes from the validate endpoint, so it is the real expanded
grid rather than a front-end estimate — the same guarantee `configs.validate` already gives
the YAML editor.

## 4. Backend

Load-bearing rule: **manual mode never gets its own expansion or validation logic.** The form
payload is turned into the same dict shape a YAML file parses to, then handed to the same
loader. Anything the CLI would reject, manual mode rejects, with the same message.

### 4.1 `core/yaml_config.py` — split the loader

`load_config_file` splits into:

- `load_config_doc(doc, only=None, source="<config>")` — all current validation and expansion,
  operating on an already-parsed dict. `source` appears in error messages where the path does
  today.
- `load_config_file(path, only=None)` — reads and parses YAML, delegates to `load_config_doc`.

No behavior change. Existing tests pass untouched.

### 4.2 `webui/attackfields.py` (new)

Introspects each `spec.config_cls` with `dataclasses.fields` into a per-attack schema:
`{name, type, default, group, readonly}`, plus `supports_toy`. `type` is one of
`bool | int | float | str`; anything else is emitted as `str`.

Groups are assigned by an explicit name→group table. Fields not in the table fall into
`Attack` — so a new attack-specific field appears in the form automatically, in a sensible
place, without touching this module.

| Group | Fields |
|---|---|
| Model & data | `model_id`, `dataset_name`, `max_length`, `seed`, `use_hf_models` |
| Federation | `num_clients`, `clients_per_round`, `federated_rounds`, `local_epochs`, `local_batch_size`, `client_lr`, `target_client_id` |
| Advanced | `fl_framework`, `sim_num_gpus`, `keep_artifacts`, `firestore_collection`, `artifact_root`, `attack_name`, `paper_source` |
| Attack | everything else |

`attack_name` and `paper_source` carry `readonly: true`.

Pure metadata. No I/O, no dependency on `webui` state.

### 4.3 `webui/manual.py` (new)

- `build_doc(payload)` — coerces each submitted value to the field's declared type
  (comma-separated lists become sequences under `sweep:`, single values go under `base:`) and
  returns the config doc dict. Only fields the user actually changed or marked as a sweep are
  emitted, so a manual config stays as small as the equivalent hand-written YAML.
- `to_yaml(doc)` — the doc as YAML text for "Save as config".
- `validate(payload)` — `build_doc` → `load_config_doc`, returning the same
  `{ok, message, runs, per_attack}` shape `configs.validate` returns.
- Coercion failures (`"abc"` into an int field) return `{ok: False, message: ...}`, never an
  exception escaping to a 500.

### 4.4 `webui/launch.py`

`start_sweep` factors into a shared `_start(pairs, use_firestore)` holding the
`publish_manifest` / `REPORTER.hooks` / `WORKER.start` sequence. Two callers: the existing
file path, and a new `start_manual(payload, use_firestore)`. The "no runs to start" and
"already running" guards live in `_start` and therefore cover both.

### 4.5 `webui/api.py`

| Endpoint | Purpose |
|---|---|
| `GET /api/attacks/fields` | Field schema for all attacks (§4.2) |
| `POST /api/launch/manual/validate` | Run count + per-attack breakdown. Writes nothing. |
| `POST /api/launch/manual` | Start a sweep from the payload |

"Save as config" reuses `POST /api/configs/save` with the text from `manual.to_yaml`, so a
saved manual config is validated on the same path as any other and immediately appears in the
Existing-config file list.

### 4.6 Cleanup on the way through

`configs.validate` currently writes a `NamedTemporaryFile` purely to reuse the file-based
loader. It switches to `load_config_doc` and stops touching disk.

## 5. Correctness property

The property that makes "run without saving" honest:

> The `(config, spec)` pairs a manual run produces are identical to the pairs obtained by
> loading the YAML that "Save as config" would have written for the same payload.

Asserted directly in tests. If it ever fails, a saved config no longer reproduces the run it
was saved from.

## 6. Error handling

- Coercion errors, unknown fields, and the amia/loss toy rejection all surface as
  `{ok: False, message}` in the validate/start response and render in the manual panel's
  readout, styled like the YAML editor's existing validation readout.
- Starting with no attacks added returns "No runs to start", reusing the existing guard.
- A start attempted while a sweep is running returns "A sweep is already running", as today.
- Front end: a failed validate leaves the last successful run count visible but greys it, so a
  stale number is never presented as current.

## 7. Testing

New `tests/test_webui_manual.py`:

- Field schema shape for every registered attack; every dataclass field appears exactly once
  across the groups; `attack_name`/`paper_source` are `readonly`.
- Type coercion, including bad input returning a message rather than raising.
- Sweep expansion and run count for a multi-attack, multi-sweep payload.
- Unknown-field rejection and the amia/loss `use_hf_models: false` rejection still fire
  through the manual path with the loader's own messages.
- The §5 round-trip equality.
- `manual.build_doc` emits only fields the user set.

Extended `tests/test_webui_configs.py`: `configs.validate` writes no temp file.

Existing `tests/test_webui_launch.py` and the `yaml_config` tests must pass unchanged.
