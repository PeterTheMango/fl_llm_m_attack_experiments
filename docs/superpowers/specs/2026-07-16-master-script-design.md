# Master Script + Research Monitor — Design

**Date:** 2026-07-16
**Status:** Approved design. Implementation plan follows separately.

Consolidates the 11 per-attack MIA adaptation notebooks in `code_experiments/adaptations/`
into one configurable CLI runner plus a Python web UI, both built on a single shared core.

---

## 1. Problem

Each membership-inference attack currently lives in its own Jupyter notebook. The notebooks
are ~85% identical scaffolding (GPU pinning, config hashing, Firestore cache, Flower FedAvg
fine-tuning, trial loop, metrics, persistence, cleanup) wrapped around ~15% of genuinely
attack-specific logic. Changing shared behavior means editing 10 files; running a sweep means
opening a notebook and editing cells by hand.

`research_monitor_ideation.md` separately specifies a dashboard that observes these
experiments, and explicitly defers the consolidation itself (§5) to this work.

## 2. Notebook inventory

Eleven notebooks, in two generations (nine modern + two legacy).

**Nine share a near-identical skeleton** — `zlib`, `min_k`, `min_k_plus_plus`, `neighborhood`,
`recall`, `reference`, `samia`, `spv_mia`, `wbc`:

1. GPU-pinning cell (must precede any CUDA init)
2. Frozen `ExperimentConfig` dataclass + `SWEEP` grid
3. `experiment_key` = `sha256(asdict(config))[:16]`
4. dotenv load → Firestore cache check
5. `build_client_partitions` — positive/negative membership worlds
6. `ToyFederatedLM` + `toy_fedavg` smoke path, **and** `run_hf_federated_finetune`
   (real Flower `FedAvg` via `run_simulation`)
7. Per-attack `score_candidate_toy` / `score_candidate_hf`
8. `summarize_trials` → TPR/TNR/Adv (+ secondary diagnostics)
9. Firestore write (merge) → artifact cleanup

**Two are older and structurally different:**

- `AMIA_adaptation` — malicious probe over frozen LLM hidden states; `federated_fine_tune`
  rather than the toy/HF split; has `mark_result_failed`; **no toy path**.
- `LOSS_adaptation` — quantile-calibrated threshold (`threshold_quantile`,
  `calibration_nonmember_count`); writes to its own `loss_federated_llm_results` collection.

### Shared vs. per-attack

| Shared (→ `core/`) | Per-attack (→ `core/attacks/*.py`) |
|---|---|
| GPU selection, dotenv loading | Extra config fields (`min_k_percent`, `num_neighbours`, `num_shots`, `window_sizes`, `num_samples`/`rouge_n`, `mask_ratio`, …) |
| Config hashing, sweep expansion | `score_toy` / `score_hf` pair |
| Firestore client / cache / save / fail-mark | Decision `threshold` |
| World construction, `ToyFederatedLM`, `toy_fedavg` | `methodology` prose |
| Flower FedAvg simulation | Optional extra metrics (e.g. `samia`'s `tpr_at_fpr`) |
| Trial loop, `summarize_trials`, `roc_auc` | |
| `run_single_experiment`, `run_sweep`, cleanup | |

## 3. Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Preserve `run_id` hashes bit-exactly** | The hash is the Firestore cache key. Any dataclass change orphans every completed document. |
| 2 | **NiceGUI** for the web UI | Native websocket push; a Firestore `on_snapshot` callback updates the browser directly from its background thread, matching the doc's real-time design. |
| 3 | **Monitor + launch in one process** | Honors the brief's launch/configure requirement while implementing §2/§3/§4 as written. |
| 4 | **YAML config**, one file, multi-attack | Readable, comments, native grid lists; `--attack` selects a subset. |
| 5 | **Sequential by default**, `--max-parallel` opt-in | Default stays behavior-identical to the notebooks; opt-in uses one GPU-pinned subprocess per run. |

## 4. Architecture

```
master_script/
  core/
    gpu.py          select_gpu(), _read_env_file_var()  — must run before torch
    config.py       AttackConfig base, experiment_key(), expand_sweep(), YAML loader
    firestore.py    get_firestore_client(), load_cached_result(), save_result(),
                    mark_result_failed(), monitor_state publish
    federation.py   build_client_partitions(), ToyFederatedLM, toy_fedavg(),
                    run_toy_federated_finetune(), run_hf_federated_finetune()
    metrics.py      roc_auc(), summarize_trials(), tpr_at_fpr()
    runner.py       run_single_experiment(), run_sweep(), artifact cleanup
    registry.py     ATTACKS: name -> AttackSpec
    attacks/        zlib.py min_k.py min_k_plus_plus.py neighborhood.py recall.py
                    reference.py samia.py spv_mia.py wbc.py amia.py loss.py
  perform_experiments.py     CLI entrypoint
  webui/
    app.py          NiceGUI server, mounts the pages
    monitor.py      §2 live view
    launch.py       configure + start sweep (calls core.runner)
    tunnel.py       §4 cloudflare/ngrok control
    state.py        in-memory projection of Firestore (non-durable, per §1.2)
  configs/
    smoke.yaml      reproduces notebook smoke defaults
    example_sweep.yaml
  logs/
  outputs/
    Charts/
```

### 4.1 AttackSpec

Each attack module exposes one `AttackSpec`, the entire per-attack surface:

```python
@dataclass(frozen=True)
class AttackSpec:
    name: str
    config_cls: type          # frozen dataclass, original field order & defaults
    score_toy: Callable       # (model, text, cfg) -> float
    score_hf: Callable        # (bundle, text, cfg) -> float
    methodology: dict         # paper_attack / llm_adaptation / metric_definition /
                              # deviation_from_source
    extra_metrics: Callable | None = None
```

Everything else is core. `AMIA` and `LOSS` implement the same interface through a slightly
wider hook set (custom fine-tune and threshold-calibration hooks) rather than being forced
into the 9-notebook mold.

### 4.2 The hash-preservation guarantee

`experiment_key` remains exactly:

```python
sha256(json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
```

Because each attack's dataclass keeps its exact field names, order, and defaults, an unchanged
config yields the byte-identical `run_id` the notebook produced.

**This is verified mechanically, not by eye.** A test parses each notebook's
`ExperimentConfig` directly out of the `.ipynb` JSON, constructs the core's config with the
same values, and asserts the two hashes match — for all 11 configs. That test is the proof
that consolidation did not move any result. It is the single most important test in the suite.

### 4.3 CLI

| Flag | Meaning |
|---|---|
| `--config PATH` | YAML config file (default `configs/smoke.yaml`) |
| `--attack NAME` | Attack(s) to run; repeatable; default = all attacks in the config |
| `--list-attacks` | Print the registry and exit |
| `--dry-run` | Expand the grid, print `run_id`s + cache status, run nothing |
| `--max-parallel N` | `1` (default) or `2`; `2` spawns one GPU-pinned subprocess per run |
| `--keep-artifacts` | Skip post-persist cleanup |
| `--no-firestore` | Local-only; skip cache read and write |
| `--charts` / `--no-charts` | Render `Adv` plots into `outputs/Charts/` |
| `--log-level LEVEL` | Logging verbosity |

`--dry-run` prints each expanded `run_id` with its cache status ("33 pending, 15 cached")
so a sweep's true cost is visible before committing GPU hours.

### 4.4 Config format

```yaml
defaults:              # applied to every attack below
  model_id: distilgpt2
  use_hf_models: true

attacks:
  zlib:
    base: {threshold: -0.0058}
    sweep:
      federated_rounds: [1, 2, 4]
      seed: [7, 11, 23]
  min_k:
    base: {min_k_percent: 20}
    sweep:
      seed: [7, 11]
```

Keys are validated against the target attack's dataclass fields and **unknown keys fail
fast**, before any compute. Keys in `defaults` that don't exist on a given attack's config
are skipped for that attack (not an error), since `defaults` is intentionally cross-attack;
unknown keys inside a specific attack's `base`/`sweep` are an error.

### 4.5 Web UI

One NiceGUI server, three pages:

- **Monitor (§2)** — now-panel from the `monitor_state` document (attack name, `run_id`,
  distinguishing config, start time, elapsed, optional coarse stage); sweep progress with
  denominators; recently-finished feed with headline `Adv`.
- **Results (§3)** — filterable/sortable grid over Firestore (row = `run_id`); aggregates;
  `Adv`-vs-factor comparison plots; per-run detail page with methodology block,
  `federated_history` curve, `attack_trials` table, and score distribution / ROC split by
  true membership. Includes §3.4's privacy-direction reading.
- **Tunnel (§4)** — provider/API-key/code/port form, start/stop, live status, and explicit
  ephemeral-vs-stable URL labeling.

**Reuse is structural.** `launch.py` builds configs and hands them to `core.runner.run_sweep`
— the same function `perform_experiments.py` calls. There is no second code path, so the UI
cannot drift from the CLI.

Because the app owns the runs, it publishes the run-state that §1.1 calls optional, making
the live panel exact. It still degrades to Firestore-only if that document is stale or absent.

### 4.6 State and paths

- **No durable UI state** (§1.2): Firestore is the only persistence; the UI rebuilds by
  re-reading on restart; in-progress runs are never resumed.
- All paths resolve from `Path(__file__).resolve().parent`, so behavior is identical
  regardless of the invoking cwd.
- `logs/<run_id>.log` per run, plus `logs/session-<timestamp>.log`.
- Charts to `outputs/Charts/` (exact capitalization). Artifacts are timestamped so repeated
  runs don't overwrite each other.
- Directories are created at import time.

## 5. Error handling

- Missing Firebase credentials → cache read returns `None`, `save_result` returns `False`;
  the run still completes locally. A genuine write/serialization error **re-raises** (a
  nested-array rejection must fail fast, not masquerade as "not saved").
- A failed run calls `mark_result_failed` and **never** cleans up its artifacts.
- Firestore forbids directly nested arrays: per-trial round histories stay wrapped in maps.
- Unknown config keys fail before compute.
- `--attack amia` without `use_hf_models` is rejected at config-validation time (AMIA has no
  toy path).

## 6. Testing

1. **Hash equivalence (§4.2)** — 11 configs, notebook vs. core, byte-identical `run_id`s.
2. **Toy determinism** — each attack's toy path reproduces its notebook's smoke assertions
   (e.g. `zlib` `roc_auc == 1.0`), runnable without GPU, model download, or credentials.
3. **Metrics** — `summarize_trials` / `roc_auc` against hand-computed values.
4. **Config loader** — grid expansion count, unknown-key rejection, `defaults` merge order.
5. **Firestore shape** — `federated_history` is a list of maps, no nested arrays.

## 7. Assumptions

- `epsilon` / `ldp_mechanism` are **not** added. No current dataclass has them, and adding
  fields would change every hash. AGENTS.md lists them as good sweep variables, but adding
  LDP is a deliberate, hash-breaking change to be made separately.
- The notebooks stay in place; this work is additive. Removing them is a separate decision.
- Default configs reproduce the notebooks' smoke settings (`use_hf_models: false`), so a
  fresh clone runs green with no GPU and no credentials.
- The toy path is deterministic, which is what makes the hash-equivalence and smoke tests
  runnable in CI.

## 8. Out of scope

- §2.5's GPU-utilization / log-tail panel (the doc marks it a future enhancement).
- Any change to attack or evaluation semantics. This consolidation is behavior-neutral by
  construction; the hash test enforces it.
