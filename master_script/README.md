# master_script

A single Python package that consolidates the 11 membership-inference-attack
(MIA) adaptation notebooks in `code_experiments/adaptations/` into one CLI
runner and one dashboard. It replaces:

- `zlib_adaptations.ipynb`
- `min_k_adaptations.ipynb`
- `min_k_plus_plus_adaptations.ipynb`
- `neighborhood_adaptations.ipynb`
- `recall_adaptations.ipynb`
- `reference_adaptations.ipynb`
- `samia_adaptations.ipynb`
- `spv_mia_adaptations.ipynb`
- `wbc_adaptations.ipynb`
- `AMIA_adaptation.ipynb`
- `LOSS_adaptation.ipynb`

Every attack's methodology, scoring, and evaluation behavior is unchanged
from its notebook. What changed is *where the code lives*: one config
schema, one runner, one Firestore integration, one dashboard, instead of 11
copy-pasted notebooks.

## The hash-preservation guarantee

Every experiment run is identified by a `run_id`: a **stable hash of its
config**, and that same string is the Firestore document id the notebooks
already wrote results under. Consolidating the notebooks into this package
had to reproduce that `run_id` **byte-for-byte** for every one of the 11
attacks, or every already-completed Firestore document would silently
become unreachable (a re-run would look "new" instead of hitting the cache).

This is not one formula, it is three, because the notebooks themselves
disagreed:

| Formula | Used by | Definition |
| --- | --- | --- |
| `key_sha16` | The 9 modern notebooks (zlib, min_k, min_k_plus_plus, neighborhood, recall, reference, samia, spv_mia, wbc) | `sha256(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")))[:16]` |
| `key_sha24_default_str` | AMIA | Same JSON encoding but tolerant of non-JSON types (`default=str`), truncated to **24** hex chars, not 16 |
| `key_named_prefix` | LOSS | `f"{experiment_name}_{digest16}"` — the digest uses the AMIA-style tolerant JSON encoding, but the document id is **name-prefixed** |

These live in `master_script/core/config.py` as `key_sha16`,
`key_sha24_default_str`, and `key_named_prefix`, and each `AttackSpec` in
`master_script/core/registry.py` points at its own formula via `key_fn`.
`experiment_key(config, spec)` dispatches to the right one — always pass
`spec` when you have it; the `spec=None` fallback is only the 16-char
formula and is correct for the 9 modern attacks alone.

`tests/test_hash_equivalence.py` is the proof: for each of the 11 attacks it
builds a config identical to what the original notebook would have built,
computes the hash both the old (notebook-inlined) way and the new
(`experiment_key`) way, and asserts they match. **If that test ever fails,
results have moved and the consolidation is wrong** — every other
correctness property of this package is secondary to that one.

## Install

```bash
conda env create -f environment.yml
conda activate peter_experiments_fl
pip install nicegui pyyaml matplotlib firebase-admin python-dotenv
```

`torch`, `transformers`, and `flwr` (already in `environment.yml`) are only
needed to run a real attack against a Hugging Face model over federated
learning. The toy smoke path (`master_script/configs/smoke.yaml`,
`--no-firestore`) needs none of them — it exercises 9 of the 11 attacks
(`amia` and `loss` have no toy path; see Known limits) with synthetic
in-memory data and no GPU, model download, or credentials.

## CLI

Everything runs through `python -m master_script.perform_experiments`.
Docstring examples are also visible via `--help`.

### `--config`

Path to a YAML config file. Default: `master_script/configs/smoke.yaml`.

```bash
python -m master_script.perform_experiments --config master_script/configs/example_sweep.yaml
```

### `--attack`

Restrict the run to one attack; repeatable to select several. Default: every
attack listed in the config's `attacks:` section.

```bash
python -m master_script.perform_experiments --config master_script/configs/example_sweep.yaml \
    --attack zlib --attack min_k
```

### `--list-attacks`

Print the attack registry (name, whether it has a toy path, and its config
class) and exit. Ignores every other flag.

```bash
python -m master_script.perform_experiments --list-attacks
```

### `--dry-run`

Expand the config's grid into individual runs, print each `run_id` and
whether it is already cached in Firestore, and run nothing. Useful for
sanity-checking a sweep before spending GPU hours.

```bash
python -m master_script.perform_experiments --config master_script/configs/example_sweep.yaml --dry-run
```

### `--max-parallel`

`1` (default) or `2`. `2` spawns one GPU-pinned subprocess per run (for a
2-GPU VM) via a `ThreadPoolExecutor`, pinning `EXPERIMENT_GPU` before torch
is imported in the child process.

```bash
python -m master_script.perform_experiments --config master_script/configs/example_sweep.yaml \
    --attack zlib --max-parallel 2
```

### `--keep-artifacts`

Keep local model/probe artifacts under `master_script/artifacts/` after a
successful Firestore persist, instead of the default cleanup.

```bash
python -m master_script.perform_experiments --config master_script/configs/example_sweep.yaml --keep-artifacts
```

### `--no-firestore`

Local-only mode: skip both the cache read (so nothing is skipped as
"already done") and the result write. This is what makes the smoke suite
runnable with no credentials.

```bash
python -m master_script.perform_experiments --config master_script/configs/smoke.yaml --no-firestore
```

### `--charts`

Render Adv (advantage) charts into `master_script/outputs/Charts/` after the
sweep finishes. This is the default; the flag exists to be explicit.

```bash
python -m master_script.perform_experiments --config master_script/configs/smoke.yaml --no-firestore --charts
```

### `--no-charts`

Skip chart rendering entirely.

```bash
python -m master_script.perform_experiments --config master_script/configs/smoke.yaml --no-firestore --no-charts
```

### `--log-level`

One of `DEBUG`, `INFO` (default), `WARNING`, `ERROR`. Controls both the
console handler and the session file under `master_script/logs/`.

```bash
python -m master_script.perform_experiments --config master_script/configs/smoke.yaml --no-firestore --log-level DEBUG
```

## YAML config schema

```yaml
defaults:
  use_hf_models: false
  attack_trials: 4

attacks:
  zlib: {}
  min_k:
    base:
      attack_trials: 8
    sweep:
      use_hf_models: [false]
```

- **`defaults`** applies across every attack in the file. It is
  cross-attack, so a key that doesn't exist on a given attack's config
  dataclass is **silently skipped** for that attack — no error.
- **`attacks`** maps attack name → `{base: {...}, sweep: {...}}`. `base`
  overrides individual fields on that attack's config dataclass; `sweep`
  maps a field name to a list of values and the grid is expanded via
  `itertools.product` (`master_script.core.config.expand_sweep`), one run
  per combination.
- Unlike `defaults`, keys inside an attack's `base` or `sweep` are
  **validated against that attack's config dataclass fields, and unknown
  keys hard-error** before any compute starts (`ConfigError`). This is
  intentional: a typo inside `base`/`sweep` would otherwise silently produce
  a differently-hashed, wrong experiment after a multi-hour run.
- Referencing an attack name not in the registry, or a config file with no
  `attacks:` section, is also a hard `ConfigError`.
- `amia` and `loss` have no toy path (see Known limits): setting
  `use_hf_models: false` for either, in `base` or `sweep`, is a hard error
  rather than an unknown-field error, since `use_hf_models` isn't a real
  field on their dataclasses — it's a virtual switch checked before field
  validation.

See `master_script/configs/smoke.yaml` (9 toy runs, no credentials) and
`master_script/configs/example_sweep.yaml` (a real sweep) for worked
examples.

## Web UI

```bash
python -m master_script.webui.app
```

Starts a NiceGUI server on `http://0.0.0.0:8080` with three pages:

- **`/`** — the monitor: live-updating view of in-flight and recent runs
  (via `master_script/webui/monitor.py`), plus the launch panel
  (`master_script/webui/launch.py`) to kick off new runs from the browser.
- **`/results`** and **`/results/{run_id}`** — the results browser
  (`master_script/webui/results.py`): a table of completed runs and a
  per-run detail view.
- **`/tunnel`** — the tunnel page (`master_script/webui/tunnel.py`), for
  exposing the dashboard through a public URL (e.g. when the server runs on
  a remote GPU VM without a public IP).

If Firestore credentials are unavailable, `main()` catches the listener
failure and the dashboard still starts — it prints
`Firestore listener unavailable (...); dashboard runs empty.` and the
monitor/results pages report the running/results set as unavailable rather
than guessing or crashing.

## Where things land

Everything resolves from `master_script/paths.py` relative to the package
directory itself, never from the caller's cwd — so the CLI and dashboard
behave identically whether invoked from the repo root or from any other
directory.

- **Logs**: `master_script/logs/` — one `session-<timestamp>.log` per CLI
  invocation (`logging_setup.setup_session_logging`), plus one
  `<run_id>.log` per individual run.
- **Outputs / charts**: `master_script/outputs/Charts/` (capital `C`) —
  written by `master_script/core/charts.py` when `--charts` (the default)
  is active.
- **Local artifacts**: `master_script/artifacts/` — model/probe artifacts,
  cleaned up after a successful persist unless `--keep-artifacts` is
  passed.
- **Configs**: `master_script/configs/`.

## Known limits

- **No `epsilon` / `ldp_mechanism` fields.** `AttackConfig` deliberately
  declares no fields — anything added there is emitted first by `asdict()`
  and would shift the JSON payload for every subclass, changing every
  attack's `run_id` and orphaning every already-completed Firestore
  document. Differential-privacy parameters were never in the original
  notebooks' hashed config, so they aren't here either.
- **`amia` and `loss` have no toy path.** Both require
  `use_hf_models: true`; there is no synthetic/in-memory smoke run for
  them, unlike the other 9 attacks.
- **The §2.5 GPU/log panel is not implemented.** The dashboard's monitor
  and results pages exist; a live GPU-utilization and tailing-log panel
  described in the original design (§2.5) was scoped out and is not part
  of this package.
