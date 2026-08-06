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
pip install fastapi uvicorn pyyaml matplotlib firebase-admin python-dotenv
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

## Web UI — CANARY Monitor

```bash
python -m master_script.webui.app
```

Starts a FastAPI server on `http://0.0.0.0:8080` serving one single-page
dashboard (`master_script/webui/static/`) over a JSON API. The client routes
between five views in the browser; each URL below is also directly linkable.

- **`/`** — Live monitoring (`webui/monitor.py`): the running set as cards,
  a 6-minute timeline, or GPU gauges; per-attack sweep progress; the
  recently-finished feed; GPU utilisation and a tail of the session log.
- **`/results`** — Results (`webui/results.py`): filter by attack, status,
  model, mechanism and `run_id`; Adv-vs-factor scatter and mean-Adv-by-attack
  charts; a sortable grid of every run.
- **`/results/{run_id}`** — Run detail: headline metrics, the methodology the
  document recorded, full config, per-round federated loss, attack-score
  distribution, ROC computed from `attack_trials[]`, and artifact paths.
- **`/access`** — Remote access (`webui/tunnel.py`): start/stop an outbound
  ngrok or cloudflared tunnel. Never auto-started; while it is up the page
  says so in as many words. Provider, credentials and port persist to the
  `.env`, so the form comes back filled in after a restart.
- **`/launch`** — Launch a sweep (`webui/launch.py`): tune an attack field by
  field, or run a saved config, through `core.runner.run_sweep` — the same
  entry point the CLI uses.
- **`/settings`** — Edit the `.env` the program loads (`webui/envfile.py`).
  **Local-only**: served to loopback callers, and to nothing else.

Chart.js is vendored into `webui/static/`, so every chart renders without a
CDN. The only outbound request the page makes is the IBM Plex webfont, which
degrades to the system font stack when it can't be fetched.

### Two ways to launch

`/launch` has two modes.

**Manual** is for fine-tuning. Add one or more attacks and edit their fields
directly — the form is generated from each attack's own config dataclass, so
it offers exactly the fields the runner accepts, grouped into model & data,
federation, the attack's own knobs, and an *Advanced* section for plumbing.
`attack_name` and `paper_source` are shown but fixed: they are part of the
experiment key, so changing one would not vary the experiment, it would rename
it into a document nobody will look for. For `amia` and `loss`, `use_hf_models`
is likewise fixed on — neither has a toy path.

The `⋯` beside a field turns it into a comma-separated sweep, expanding into a
grid exactly as `sweep:` does in YAML. Nothing has to be saved first: *Start
sweep* runs the form as it stands. *Save as config* is there for when a setup
is worth keeping, and writes the equivalent `.yaml` into
`master_script/configs/`.

**Existing config** is the config editor below: pick a saved file, edit it,
and run it.

Both modes reach `core.yaml_config.load_config_doc` and then
`core.runner.run_sweep`. A manual sweep and the config it would save as expand
to the same runs — `tests/test_webui_manual.py` asserts that equality directly,
because if it ever broke, a saved config would no longer reproduce the run it
was saved from.

### The config editor

`/launch` edits `master_script/configs/*.yaml` in the browser: pick an
existing file or start a new one from a template, edit it, and save it under
any name. Two rules make this safe to use on a real sweep:

- **Validate before you spend GPU hours.** *Validate* runs the text through
  `core.yaml_config.load_config_doc` — the CLI's own loader — and reports the
  real expanded grid (`Valid · expands to 18 run(s)`, broken down per attack),
  or the exact error, typo'd field names included.
- **An invalid config never reaches disk.** Save validates first and refuses
  otherwise, so a file in `configs/` is always one the runner accepts. Saving
  over an existing file takes a second, explicit click. Names are plain
  basenames inside `configs/` — a path that would escape it is rejected.

A sweep always runs the file *on disk*, so the page tells you when the editor
has unsaved changes.

### Settings and the `.env`

`/settings` edits the `.env` the program loads (the nearest one at or above
`master_script/`) and reloads it into the running process — `os.environ` is
updated in place and the cached Firestore client is torn down so the next call
re-authenticates. Without that teardown, new credentials would sit in the
environment while `firebase_admin` kept using the app it cached at first use.
The previous file is always copied to `.env.bak`, and comments, ordering and
keys you didn't touch are preserved.

**This page is local-only.** It reads and writes your Firebase service-account
credentials, so unlike the rest of the dashboard it does not follow the tunnel
out: `webui/localguard.py` answers `403` to anything that did not originate on
the machine running the server, and the client hides the nav tab accordingly.
"Local" means the socket address is loopback *and* the request carries no
proxy-forwarding header — the tunnel agent connects over loopback itself, so
the header is what separates a real local caller from a proxied one. To edit
settings from your laptop, forward the port over SSH
(`ssh -L 8080:localhost:8080 <host>`), which terminates on loopback and is
already an authenticated channel.

Secret values are still masked by default and revealing one is a separate,
explicit click. The page warns you if the tunnel is live while you are on it.

### Tunnel credentials

`/access` saves its provider, API token, tunnel code and port to the same
`.env`, under `TUNNEL_PROVIDER` / `TUNNEL_API_KEY` / `TUNNEL_CODE` /
`TUNNEL_PORT`, whenever you start a tunnel or press *Save to .env*. Two
consequences worth knowing:

- They are stored in the clear, like every other credential in that file, and
  are visible and revocable from `/settings` alongside the rest.
- They are never sent back to the browser. A saved field shows only its shape
  and posts `null` when untouched, meaning "keep what is on disk"; clearing a
  field and saving is how you remove a stored value.

### What the dashboard refuses to guess

Firestore alone cannot say what is *currently* running. When the central
script publishes no run-state report, the Live view says the running set is
unavailable instead of inferring one, and sweep bars show
`(no manifest → no denominator)` rather than a made-up total. Launching from
`/launch` publishes both the manifest and the run-state report, which is what
gives the sweep bars real denominators. Likewise, a run whose document
carries no per-round loss or no `attack_trials[]` gets an explicit "not
recorded" panel, not an empty chart.

### Run-state, heartbeats, and ghosts

The run-state report lives in `core/runstate.py` and is published by **both**
the CLI and the dashboard, so a sweep started from the terminal shows up live
in `/` exactly like one started from `/launch`. It is off under
`--no-firestore` (nowhere to publish).

It is written **per run**: `run_sweep`'s `on_run_start` / `on_run_end` hooks
bracket each run, so the running set is the run actually in flight, and
`on_run_end` fires from a `finally` — a run that raises still clears itself.
`_run` clears the whole report in its own `finally`, so an aborted sweep
doesn't leave the last run claimed.

Under `--max-parallel 2` the **parent** owns the report for the whole sweep and
the GPU-pinned children are suppressed (`MASTER_SCRIPT_SUPPRESS_RUN_STATE`).
The report is a single array, so two processes publishing would clobber each
other; the parent holds both in-flight runs and republishes the set on any
change.

A *hard* kill (SIGKILL, OMP abort, VM reboot) never reaches that `finally`, so
the writer also **heartbeats**: `RunStateReporter` re-stamps `heartbeat_unix`
every `HEARTBEAT_SECONDS` (30s) while a run is live, carrying `started_unix`
forward so the elapsed clock keeps counting from the real start. The reader
treats an entry whose heartbeat stopped for more than
`monitor.STALE_AFTER_SECONDS` (150s, five missed beats) as **stale**: the card
is marked, its stage bars stop animating, and it is excluded from the header
count and the sweep bars. Staleness is measured from the *heartbeat*, never
from the start time — runs legitimately take hours.

That combination is what makes the Live view honest after a crash: the clean
path clears itself, and the dirty path ages out.

### Re-running a failed run

Results are written with `merge=True`, and a successful payload has no `error`
key — so a run that failed and was later re-run would keep the earlier
attempt's error string forever and read as failed. `save_result` now deletes
the field on a successful write, restoring exactly the shape a never-failed
run has.

Documents written before that fix still carry the stale error. The dashboard
reads `error` as this run's outcome only when `status != "complete"`; on a
completed run it is shown separately, as history, under an explicit "an earlier
attempt failed and was later recovered" note — never as a failure.

### Job resumption

There is none in the checkpoint sense, by design (§1.2) — you resume by
**re-running the same sweep**. `run_id` is a hash of the config, and
`run_single_experiment` skips any run whose document is already
`status: "complete"`. So finished runs are cache hits and everything else
recomputes from scratch; the granularity is one whole run, since a run writes
its document once, at the end. A run killed at trial 63 of 64 loses all 64.

Note that a sweep launched from `/launch` runs on a daemon thread **inside the
dashboard process** — killing the dashboard kills the sweep. A sweep launched
from the CLI is an independent process and is unaffected by the dashboard.

If Firestore credentials are unavailable, `main()` catches the listener
failure and the dashboard still starts — it prints
`Firestore listener unavailable (...); falling back to polling reads.` and
the views report empty sets rather than guessing or crashing.

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
