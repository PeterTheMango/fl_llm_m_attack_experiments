# Master Script + Research Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the 11 MIA adaptation notebooks in `code_experiments/adaptations/` into one shared core with a CLI runner and a NiceGUI web UI, without changing any attack or evaluation behavior.

**Architecture:** A `core/` package owns everything the notebooks share (GPU pinning, config hashing, Firestore, Flower FedAvg fine-tuning, trial loop, metrics, cleanup). Each attack contributes one `AttackSpec` — a frozen config dataclass, a scoring pair, methodology prose, and optional extra metrics. `perform_experiments.py` (CLI) and `webui/` both call `core.runner.run_sweep`, so there is no second code path.

**Tech Stack:** Python 3.11+, dataclasses, PyYAML, firebase-admin, torch, transformers, `flwr[simulation]`, matplotlib, NiceGUI, pytest.

## Global Constraints

- **Hash preservation is the prime directive.** Never add, remove, reorder, or re-default a field on any attack config dataclass. Tasks 3–4 exist solely to enforce this.
- **There are THREE different key formulas, not one.** This was verified against the notebooks; do not unify them:

  | Attacks | Formula | Example |
  |---|---|---|
  | the 9 modern | `sha256(json.dumps(asdict(cfg), sort_keys=True, separators=(",",":")))[:16]` | `f747493634f0cc01` |
  | `amia` | `sha256(stable_json(asdict(cfg)))[:24]` — note `default=str` **and 24 chars** | `848ce58285a38a20ac058893` |
  | `loss` | `f"{cfg.experiment_name}_{sha256(stable_json(asdict(cfg)))[:16]}"` — **name-prefixed** | `loss_federated_llm_adaptation_v1_952b56bc2ecd70aa` |

  where `stable_json(p) = json.dumps(p, sort_keys=True, separators=(",",":"), default=str)`.
  A single global formula would silently orphan every `amia` and `loss` document. The key
  function therefore lives on `AttackSpec.key_fn`, not as one core constant.
- **Behavior neutrality.** Port logic verbatim. If a notebook uses `>=` and another uses `>`, preserve both. Do not "fix" anything you find, including apparent bugs — note it and move on.
- **No `epsilon` / `ldp_mechanism` fields.** They appear in `AGENTS.md` but exist in no dataclass. Adding them changes every hash.
- **Path resolution:** every path derives from `MASTER_DIR = Path(__file__).resolve().parent` (or its parent for modules inside `core/`). Never rely on cwd.
- **Charts directory is exactly `outputs/Charts/`** — capital C.
- **Firestore forbids directly nested arrays.** Per-trial round history stays wrapped in a map.
- **Artifacts are never cleaned up for a failed run.**
- **GPU pinning must happen before any `torch`/CUDA import.**
- Toy paths are deterministic and must run with no GPU, no model download, and no credentials.

**Source of truth:** `docs/superpowers/specs/2026-07-16-master-script-design.md`.

**Notebook provenance:** all 11 live in `code_experiments/adaptations/`: `AMIA_adaptation.ipynb`, `LOSS_adaptation.ipynb`, `min_k_adaptations.ipynb`, `min_k_plus_plus_adaptations.ipynb`, `neighborhood_adaptations.ipynb`, `recall_adaptations.ipynb`, `reference_adaptations.ipynb`, `samia_adaptations.ipynb`, `spv_mia_adaptations.ipynb`, `wbc_adaptations.ipynb`, `zlib_adaptations.ipynb`.

**Reading notebook source:** notebooks are JSON. To read a notebook's code:

```bash
python3 -c "
import json,sys
nb=json.load(open(sys.argv[1]))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        print('#'*20,'CELL',i,'#'*20); print(''.join(c['source']))
" code_experiments/adaptations/zlib_adaptations.ipynb
```

---

## File Structure

```
master_script/
  __init__.py
  paths.py            MASTER_DIR, LOGS_DIR, OUTPUTS_DIR, CHARTS_DIR; mkdir at import
  logging_setup.py    session + per-run log handlers
  core/
    __init__.py
    gpu.py            select_gpu(), _read_env_file_var()
    config.py         AttackConfig, experiment_key(), expand_sweep()
    yaml_config.py    load_config_file() -> list[config instances]
    firestore.py      client, cache, save, mark_failed, monitor_state
    federation.py     partitions, ToyFederatedLM, toy_fedavg, toy/HF fine-tune
    metrics.py        roc_auc(), tpr_at_fpr(), base_metrics()
    registry.py       AttackSpec, ATTACKS, get_attack()
    runner.py         run_single_experiment(), run_sweep()
    charts.py         render Adv plots -> outputs/Charts/
    attacks/
      __init__.py
      zlib.py min_k.py min_k_plus_plus.py neighborhood.py recall.py
      reference.py samia.py spv_mia.py wbc.py amia.py loss.py
  perform_experiments.py
  webui/
    __init__.py
    app.py            NiceGUI server + page mounting
    state.py          in-memory Firestore projection (non-durable)
    monitor.py        spec §2
    results.py        spec §3
    launch.py         configure + start sweep
    tunnel.py         spec §4
  configs/
    smoke.yaml
    example_sweep.yaml
  logs/.gitkeep
  outputs/Charts/.gitkeep
tests/
  conftest.py
  notebook_configs.py          helper: reads configs out of .ipynb JSON
  test_hash_equivalence.py     THE critical test
  test_paths.py
  test_gpu.py
  test_config.py
  test_metrics.py
  test_federation.py
  test_firestore.py            includes the nested-array shape guard
  test_scoring.py
  test_legacy_attacks.py
  test_runner.py
  test_toy_parity.py
  test_yaml_config.py
  test_charts.py
  test_cli.py
  test_webui_state.py
  test_webui_pages.py
  test_webui_launch.py
  test_tunnel.py
  test_docs.py
```

**Task ordering rationale:** the hash guarantee is the highest-risk item, so Tasks 1–4 establish and prove it before any attack logic is ported. If hash equivalence can't hold, we need to know on day one, not after the web UI is built.

---

### Task 1: Package skeleton, paths, and logging

**Files:**
- Create: `master_script/__init__.py`, `master_script/paths.py`, `master_script/logging_setup.py`, `master_script/core/__init__.py`, `master_script/core/attacks/__init__.py`, `master_script/webui/__init__.py`
- Create: `master_script/logs/.gitkeep`, `master_script/outputs/Charts/.gitkeep`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `paths.MASTER_DIR`, `paths.LOGS_DIR`, `paths.OUTPUTS_DIR`, `paths.CHARTS_DIR` (all `pathlib.Path`); `logging_setup.setup_session_logging(level: str) -> Path`, `logging_setup.run_log_handler(run_id: str) -> logging.Handler`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
import os
from pathlib import Path

from master_script import paths


def test_dirs_resolve_relative_to_master_script_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib

    importlib.reload(paths)
    assert paths.MASTER_DIR.name == "master_script"
    assert paths.LOGS_DIR == paths.MASTER_DIR / "logs"
    assert paths.CHARTS_DIR == paths.MASTER_DIR / "outputs" / "Charts"


def test_dirs_exist_after_import():
    assert paths.LOGS_DIR.is_dir()
    assert paths.CHARTS_DIR.is_dir()


def test_charts_dir_is_capitalized():
    assert paths.CHARTS_DIR.name == "Charts"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script'`

- [ ] **Step 3: Write minimal implementation**

```python
# master_script/paths.py
"""Filesystem anchors. Everything resolves from this file, never from cwd."""
from pathlib import Path

MASTER_DIR = Path(__file__).resolve().parent
LOGS_DIR = MASTER_DIR / "logs"
OUTPUTS_DIR = MASTER_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "Charts"
CONFIGS_DIR = MASTER_DIR / "configs"
ARTIFACTS_DIR = MASTER_DIR / "artifacts"

for _d in (LOGS_DIR, OUTPUTS_DIR, CHARTS_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
```

```python
# master_script/logging_setup.py
"""Session-wide and per-run logging, both rooted in master_script/logs/."""
import logging
import time
from pathlib import Path

from .paths import LOGS_DIR

_FMT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_session_logging(level: str = "INFO") -> Path:
    """Attach a session log file to the root logger. Returns the log path."""
    log_path = LOGS_DIR / f"session-{time.strftime('%Y%m%d-%H%M%S')}.log"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(_FMT))
    root.addHandler(file_handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter(_FMT))
    root.addHandler(stream)
    return log_path


def run_log_handler(run_id: str) -> logging.Handler:
    """A per-run file handler; caller adds/removes it around a run."""
    handler = logging.FileHandler(LOGS_DIR / f"{run_id}.log")
    handler.setFormatter(logging.Formatter(_FMT))
    return handler
```

Create empty `__init__.py` for `master_script`, `master_script/core`, `master_script/core/attacks`, `master_script/webui`, and empty `.gitkeep` files in `logs/` and `outputs/Charts/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paths.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script tests/test_paths.py
git commit -m "feat(master_script): add package skeleton, path anchors, and logging"
```

---

### Task 2: GPU selection ported verbatim

**Files:**
- Create: `master_script/core/gpu.py`
- Test: `tests/test_gpu.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `gpu.select_gpu() -> str | None`, `gpu.apply_gpu_selection() -> str | None` (sets `CUDA_VISIBLE_DEVICES`, returns the choice).

**Provenance:** port `_read_env_file_var`, `_gpu_free_memory`, `select_gpu` verbatim from `zlib_adaptations.ipynb` cell 3 (identical in all 11 notebooks). `apply_gpu_selection` wraps the notebook's module-level `if/elif/else` block that sets `CUDA_VISIBLE_DEVICES`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu.py
from master_script.core import gpu


def test_env_var_forces_selection(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_GPU", "1")
    assert gpu.select_gpu() == "1"


def test_cpu_forced_sets_empty_visible_devices(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_GPU", "cpu")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gpu.apply_gpu_selection() == "cpu"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_no_nvidia_smi_returns_none(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_GPU", raising=False)
    monkeypatch.setattr(gpu.shutil, "which", lambda _: None)
    monkeypatch.setattr(gpu, "_read_env_file_var", lambda _: None)
    assert gpu.select_gpu() is None


def test_auto_selects_gpu_with_most_free_memory(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_GPU", raising=False)
    monkeypatch.setattr(gpu, "_read_env_file_var", lambda _: None)
    monkeypatch.setattr(gpu, "_gpu_free_memory", lambda: [("0", 1000), ("1", 8000)])
    assert gpu.select_gpu() == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu.py -v`
Expected: FAIL with `ImportError: cannot import name 'gpu'`

- [ ] **Step 3: Write minimal implementation**

Port cell 3 of `zlib_adaptations.ipynb` into `master_script/core/gpu.py`, keeping `_read_env_file_var`, `_gpu_free_memory`, and `select_gpu` byte-for-byte. Wrap the notebook's trailing side-effect block as:

```python
def apply_gpu_selection() -> str | None:
    """Set CUDA_VISIBLE_DEVICES from select_gpu(). Call before importing torch."""
    chosen = select_gpu()
    if chosen is None:
        print("GPU selection: no GPU detected; using default device visibility.")
    elif chosen.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("GPU selection: forced CPU (CUDA_VISIBLE_DEVICES='').")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen
        free = dict(_gpu_free_memory()).get(chosen)
        detail = f" ({free} MiB free)" if free is not None else ""
        print(f"GPU selection: pinned to physical GPU {chosen}{detail}; it appears as cuda:0 in this process.")
    return chosen
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gpu.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/core/gpu.py tests/test_gpu.py
git commit -m "feat(core): port GPU selection from notebooks"
```

---

### Task 3: Config base, hashing, and sweep expansion

**Files:**
- Create: `master_script/core/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.AttackConfig` (frozen dataclass base, **declares no fields**), `config.stable_json(payload) -> str`, `config.key_sha16(cfg) -> str`, `config.key_sha24_default_str(cfg) -> str`, `config.key_named_prefix(cfg) -> str`, `config.experiment_key(cfg, spec=None) -> str`, `config.expand_sweep(base, sweep) -> Iterator`, `config.artifact_dir_for(cfg, spec=None) -> Path`.

**Critical #1:** `AttackConfig` is a marker base with **zero fields**. Subclasses declare every field themselves, in the notebook's exact order. If the base declared fields, they would land first in `asdict()` and change every hash.

**Critical #2:** three key formulas exist (see Global Constraints). `experiment_key(cfg, spec)` dispatches to `spec.key_fn`. The `spec=None` fallback uses the 16-char formula, which is correct only for the nine modern attacks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from dataclasses import dataclass
from hashlib import sha256
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.core.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# master_script/core/config.py
"""Config base, stable hashing, and grid expansion.

The key formula is load-bearing: it is the Firestore document id, and it must
stay byte-identical to each notebook's. THREE formulas exist -- the nine modern
notebooks, AMIA, and LOSS each hash differently. Unifying them would orphan
completed documents. See tests/test_hash_equivalence.py.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence
import json

from ..paths import ARTIFACTS_DIR


@dataclass(frozen=True)
class AttackConfig:
    """Marker base. Declares NO fields on purpose.

    Any field here would be emitted first by asdict() and would change every
    subclass's key, orphaning completed Firestore documents.
    """


def stable_json(payload: Any) -> str:
    """AMIA/LOSS variant: tolerates non-JSON types via default=str."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def key_sha16(config: AttackConfig) -> str:
    """The nine modern notebooks. Note: no default=str."""
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def key_sha24_default_str(config: AttackConfig) -> str:
    """AMIA_adaptation.ipynb: 24 chars, not 16."""
    return sha256(stable_json(asdict(config)).encode("utf-8")).hexdigest()[:24]


def key_named_prefix(config: AttackConfig) -> str:
    """LOSS_adaptation.ipynb: f'{experiment_name}_{digest16}'."""
    digest = sha256(stable_json(asdict(config)).encode("utf-8")).hexdigest()[:16]
    return f"{config.experiment_name}_{digest}"


def experiment_key(config: AttackConfig, spec: Optional[Any] = None) -> str:
    """Dispatch to the attack's own key formula.

    The spec=None fallback is the modern 16-char formula, correct only for the
    nine modern attacks. Always pass spec when you have it.
    """
    if spec is not None and getattr(spec, "key_fn", None) is not None:
        return spec.key_fn(config)
    return key_sha16(config)


def expand_sweep(base_config, sweep: Dict[str, Sequence]) -> Iterator:
    keys = list(sweep.keys())
    if not keys:
        yield base_config
        return
    for values in product(*(sweep[key] for key in keys)):
        yield replace(base_config, **dict(zip(keys, values)))


def artifact_dir_for(config: AttackConfig, spec: Optional[Any] = None) -> Path:
    root = getattr(config, "artifact_root", None) or getattr(config, "local_artifact_dir")
    return ARTIFACTS_DIR / Path(root).name / experiment_key(config, spec)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/core/config.py tests/test_config.py
git commit -m "feat(core): add config base, stable hashing, and sweep expansion"
```

---

### Task 4: Hash-equivalence harness + all 11 attack configs

This is the most important task in the plan. It ports every notebook's `ExperimentConfig` and proves each produces the byte-identical `run_id`.

**Files:**
- Create: `tests/notebook_configs.py` (test helper that extracts configs from `.ipynb` JSON)
- Create: `tests/test_hash_equivalence.py`
- Create: `master_script/core/attacks/zlib.py`, `min_k.py`, `min_k_plus_plus.py`, `neighborhood.py`, `recall.py`, `reference.py`, `samia.py`, `spv_mia.py`, `wbc.py`, `amia.py`, `loss.py` (config dataclasses only at this stage)
- Test: `tests/test_hash_equivalence.py`

**Interfaces:**
- Consumes: `config.AttackConfig`, `config.experiment_key`.
- Produces: `ZlibConfig`, `MinKConfig`, `MinKPlusPlusConfig`, `NeighborhoodConfig`, `RecallConfig`, `ReferenceConfig`, `SamiaConfig`, `SpvMiaConfig`, `WbcConfig`, `AmiaConfig`, `LossConfig` — each a frozen dataclass subclassing `AttackConfig`.

**Method:** the helper execs each notebook's config cell in an isolated namespace and returns its `ExperimentConfig` class. The test hashes the notebook's instance and the ported instance and asserts equality. This compares against the real source, not a transcription.

- [ ] **Step 1: Write the failing test**

```python
# tests/notebook_configs.py
"""Extract each notebook's ExperimentConfig class straight from the .ipynb JSON.

This is deliberately source-of-truth: we compare the ported dataclass against
the notebook's actual class, not against a hand-copied transcription.
"""
from pathlib import Path
import json
import re

ADAPTATIONS = Path(__file__).resolve().parents[1] / "code_experiments" / "adaptations"

NOTEBOOKS = {
    "zlib": "zlib_adaptations.ipynb",
    "min_k": "min_k_adaptations.ipynb",
    "min_k_plus_plus": "min_k_plus_plus_adaptations.ipynb",
    "neighborhood": "neighborhood_adaptations.ipynb",
    "recall": "recall_adaptations.ipynb",
    "reference": "reference_adaptations.ipynb",
    "samia": "samia_adaptations.ipynb",
    "spv_mia": "spv_mia_adaptations.ipynb",
    "wbc": "wbc_adaptations.ipynb",
    "amia": "AMIA_adaptation.ipynb",
    "loss": "LOSS_adaptation.ipynb",
}


def _code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def notebook_config_class(attack: str):
    """Exec only the cell defining ExperimentConfig, in a clean namespace."""
    cells = _code_cells(ADAPTATIONS / NOTEBOOKS[attack])
    target = next(c for c in cells if re.search(r"class ExperimentConfig", c))
    # Strip notebook-only magics (e.g. %pip) that exec() cannot parse.
    target = "\n".join(l for l in target.splitlines() if not l.strip().startswith("%"))
    namespace: dict = {}
    preamble = (
        "from dataclasses import asdict, dataclass, field, replace\n"
        "from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple\n"
        "from pathlib import Path\n"
        "from hashlib import sha256\n"
        "from itertools import product\n"
        "import json, math, os, random, shutil, time, zlib\n"
    )
    exec(preamble + target, namespace)
    return namespace["ExperimentConfig"]
```

```python
# tests/test_hash_equivalence.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hash_equivalence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.core.registry'`

- [ ] **Step 3: Write minimal implementation**

Create a minimal `master_script/core/registry.py` (fleshed out in Task 5):

```python
# master_script/core/registry.py
"""Attack registry. One AttackSpec per adaptation notebook."""
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .config import key_sha16


@dataclass(frozen=True)
class AttackSpec:
    name: str
    config_cls: type
    methodology: dict
    # key_fn: this attack's OWN document-id formula. Three variants exist across
    # the notebooks; unifying them would orphan completed Firestore documents.
    key_fn: Callable = key_sha16
    score_toy: Optional[Callable] = None
    score_hf: Optional[Callable] = None
    extra_metrics: Optional[Callable] = None
    needs_reference: bool = False
    supports_toy: bool = True


def _load() -> Dict[str, AttackSpec]:
    from .attacks import (
        amia, loss, min_k, min_k_plus_plus, neighborhood, recall,
        reference, samia, spv_mia, wbc, zlib,
    )

    specs = [
        zlib.SPEC, min_k.SPEC, min_k_plus_plus.SPEC, neighborhood.SPEC,
        recall.SPEC, reference.SPEC, samia.SPEC, spv_mia.SPEC, wbc.SPEC,
        amia.SPEC, loss.SPEC,
    ]
    return {spec.name: spec for spec in specs}


ATTACKS: Dict[str, AttackSpec] = _load()


def get_attack(name: str) -> AttackSpec:
    if name not in ATTACKS:
        raise KeyError(f"Unknown attack {name!r}. Known: {', '.join(sorted(ATTACKS))}")
    return ATTACKS[name]
```

Now create each attack module with **only** its config dataclass and a `SPEC` whose `methodology` is copied verbatim from that notebook's `run_single_experiment` (or `build_result_payload` for `loss`). Scoring stays `None` until Tasks 7–9.

For each attack, open the notebook, find the `@dataclass(frozen=True) class ExperimentConfig` cell, and copy the field block **exactly** — same names, same order, same defaults, same types. Change only the class name and the base:

```python
# master_script/core/attacks/zlib.py
"""zlib-entropy ratio MIA. Ported from zlib_adaptations.ipynb.

Config fields are byte-frozen: see tests/test_hash_equivalence.py.
"""
from dataclasses import dataclass

from ..config import AttackConfig
from ..registry import AttackSpec


@dataclass(frozen=True)
class ZlibConfig(AttackConfig):
    attack_name: str = "zlib"
    paper_source: str = "Carlini et al. 2021 (arXiv:2012.07805) zlib-entropy ratio MIA"
    model_id: str = "sshleifer/tiny-gpt2"
    dataset_name: str = "synthetic_client_text"
    num_clients: int = 4
    clients_per_round: int = 4
    federated_rounds: int = 1
    local_epochs: int = 1
    local_batch_size: int = 2
    client_lr: float = 5e-5
    target_client_id: int = 0
    attack_trials: int = 4
    threshold: float = -0.0058
    max_length: int = 64
    seed: int = 7
    firestore_collection: str = "ami_federated_llm_results"
    artifact_root: str = "artifacts/zlib_adaptation"
    fl_framework: str = "flower"
    sim_num_gpus: float = 0.0
    keep_artifacts: bool = False
    use_hf_models: bool = False


METHODOLOGY = {
    "paper_attack": (
        "zlib-entropy ratio MIA: rank candidates by log_perplexity / len(zlib.compress(text)); "
        "zlib is a model-independent reference that down-ranks repetitive/boilerplate text."
    ),
    "llm_adaptation": (
        "Positive and negative FL worlds differ by target-client membership; after Flower (flwr) "
        "FedAvg simulation, score the target record under the final FL model only, dividing its "
        "log-perplexity by the record's zlib entropy. No reference model is used."
    ),
    "metric_definition": (
        "Adv = 0.5 * TPR + 0.5 * TNR; membership score = -(log_perplexity / zlib_entropy_bits) "
        "so members score higher."
    ),
    "deviation_from_source": (
        "Carlini et al. flag fine-tuning as future work, so applying the zlib ratio to a "
        "fine-tuned model is a transfer of their pre-training signal. The smoke run uses a "
        "deterministic toy causal scorer; set use_hf_models=True for genuine federated "
        "fine-tuning of an open-source LLM with the Flower (flwr) FedAvg simulation."
    ),
}

SPEC = AttackSpec(name="zlib", config_cls=ZlibConfig, methodology=METHODOLOGY)
```

This `METHODOLOGY` dict is copied **verbatim** from `zlib_adaptations.ipynb`'s
`run_single_experiment`. Do the same for every attack: open its notebook, find the
`"methodology"` key in `run_single_experiment` (or `build_result_payload` for `amia`/`loss`),
and copy the dict exactly — same keys, same prose. It is the document's self-documentation of
what the attack is and how it was adapted, and the dashboard renders it verbatim (§3.3).

Note the key sets differ: `loss` has an extra `attacker_observation` key. Do not normalize
them.

Repeat for the other ten. Field lists to reproduce (verify against the notebook — the notebook wins any disagreement):

- `min_k` — as zlib, but `attack_name="min_k"`, `paper_source="Shi et al. 2024 (ICLR / arXiv:2310.16789) Min-K% Prob reference-free MIA"`, adds `min_k_percent: int = 20` before `threshold`, `threshold: float = -4.08`, `artifact_root="artifacts/min_k_adaptation"`.
- `min_k_plus_plus` — same shape as `min_k`; `threshold: float = -3.25`, `artifact_root="artifacts/min_k_plus_plus_adaptation"`, `paper_source="Zhang et al. 2025 (ICLR / arXiv:2404.02936) Min-K%++ reference-free MIA"`.
- `neighborhood` — adds `num_neighbours: int = 25` and `neighbour_swaps: int = 1`; `threshold: float = 0.02`.
- `recall` — adds `num_shots: int = 1`; `threshold: float = 1.05`.
- `reference` — no extra fields; `threshold: float = 0.25`.
- `samia` — adds `num_samples: int = 10`, `rouge_n: int = 1`, `use_zlib_weighting: bool = False`; `threshold: float = 0.5`.
- `spv_mia` — adds `num_paraphrases: int = 4`, `mask_ratio: float = 0.2`, `self_prompt_tokens: int = 8`; `threshold: float = 0.0`.
- `wbc` — adds `reference_model_id: str = "sshleifer/tiny-gpt2"` (right after `model_id`) and `window_sizes: Tuple[int, ...] = (2, 3, 4, 6, 9, 13, 18, 25, 32, 40)`; `threshold: float = 0.75`.
- `amia` — different shape entirely; copy from `AMIA_adaptation.ipynb` verbatim (uses `gradient_threshold`, `probe_epochs`, `probe_lr`). Set `supports_toy=False`.
- `loss` — different shape; copy from `LOSS_adaptation.ipynb` verbatim (`experiment_name`, `paper`, `paper_summary_path`, `source_repo`, `threshold_quantile`, `calibration_nonmember_count`, `firestore_collection="loss_federated_llm_results"`, `firebase_project_id`, `local_artifact_dir`). Set `supports_toy=False` — it calls `require_training_deps()` and has **no** `use_hf_models` field.

**Neither `amia` nor `loss` has a `use_hf_models` field.** Any code that reads
`cfg.use_hf_models` must therefore be reached only by the nine modern attacks, which the
`custom_trials` hook (Task 9) guarantees. Do not add the field to make them uniform — that
changes their hashes.

**Note on `wbc`:** `window_sizes` is a `Tuple`, which `json.dumps` serializes as a JSON array — same as the notebook, so the hash matches. Keep it a tuple (a list would break `frozen=True` hashing semantics). `wbc` also defines `config_to_storage`; that belongs in Task 9, not here.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hash_equivalence.py -v`
Expected: PASS (45 tests: 4 parametrized × 11 attacks, minus skips, plus the registry test)

If any `test_default_run_id_is_byte_identical` fails, diff the two configs and fix the **ported** side:

```bash
python3 -c "
from dataclasses import asdict
from tests.notebook_configs import notebook_config_class
from master_script.core.registry import ATTACKS
a='zlib'
nb=asdict(notebook_config_class(a)()); pt=asdict(ATTACKS[a].config_cls())
print('only in notebook:', {k:v for k,v in nb.items() if k not in pt or pt[k]!=v})
print('only in ported  :', {k:v for k,v in pt.items() if k not in nb or nb[k]!=v})
print('order nb :', list(nb)); print('order port:', list(pt))
"
```

- [ ] **Step 5: Commit**

```bash
git add master_script/core/registry.py master_script/core/attacks tests/notebook_configs.py tests/test_hash_equivalence.py
git commit -m "feat(core): port all 11 attack configs with proven hash equivalence"
```

---

### Task 5: Metrics

**Files:**
- Create: `master_script/core/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `metrics.roc_auc(labels, scores) -> float`, `metrics.tpr_at_fpr(labels, scores, target_fpr=0.05) -> float`, `metrics.base_metrics(trials) -> dict`, `metrics.summarize(trials, spec) -> dict`.

**Behavior note:** `base_metrics` returns exactly the key set that `reference_adaptations.ipynb` produces (`tp`, `tn`, `fp`, `fn`, `tpr`, `tnr`, `adv`, `accuracy`, `precision`, `recall`, `f1`, `num_trials`) — deliberately **no** `roc_auc`. Nine attacks add `roc_auc` via `extra_metrics`; `samia` adds `roc_auc` **and** `tpr_at_fpr`. This preserves each notebook's document shape exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import math

import pytest

from master_script.core import metrics

PERFECT = [
    {"trial_id": 0, "truth_member": True, "score": 1.0, "pred_member": True},
    {"trial_id": 1, "truth_member": False, "score": 0.0, "pred_member": False},
    {"trial_id": 2, "truth_member": True, "score": 0.9, "pred_member": True},
    {"trial_id": 3, "truth_member": False, "score": 0.1, "pred_member": False},
]


def test_base_metrics_perfect_classifier():
    out = metrics.base_metrics(PERFECT)
    assert out["tpr"] == 1.0
    assert out["tnr"] == 1.0
    assert out["adv"] == 1.0
    assert out["num_trials"] == 4


def test_adv_is_mean_of_tpr_and_tnr():
    trials = [
        {"truth_member": True, "score": 1.0, "pred_member": True},
        {"truth_member": True, "score": 0.0, "pred_member": False},
        {"truth_member": False, "score": 0.0, "pred_member": False},
        {"truth_member": False, "score": 0.0, "pred_member": False},
    ]
    out = metrics.base_metrics(trials)
    assert out["tpr"] == 0.5
    assert out["tnr"] == 1.0
    assert out["adv"] == 0.75


def test_base_metrics_excludes_roc_auc():
    """reference_adaptations.ipynb has no roc_auc; the base set must match it."""
    assert "roc_auc" not in metrics.base_metrics(PERFECT)


def test_base_metrics_key_set_matches_reference_notebook():
    assert set(metrics.base_metrics(PERFECT)) == {
        "tp", "tn", "fp", "fn", "tpr", "tnr", "adv",
        "accuracy", "precision", "recall", "f1", "num_trials",
    }


def test_roc_auc_perfect_separation():
    assert metrics.roc_auc([True, True, False, False], [1.0, 0.9, 0.1, 0.0]) == 1.0


def test_roc_auc_ties_count_half():
    assert metrics.roc_auc([True, False], [0.5, 0.5]) == 0.5


def test_roc_auc_single_class_is_nan():
    assert math.isnan(metrics.roc_auc([True, True], [1.0, 0.5]))


def test_base_metrics_empty_trials_does_not_divide_by_zero():
    out = metrics.base_metrics([])
    assert out["adv"] == 0.0
    assert out["num_trials"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.core.metrics'`

- [ ] **Step 3: Write minimal implementation**

Port `roc_auc` verbatim from `zlib_adaptations.ipynb` cell 17 and `summarize_trials`' body from `reference_adaptations.ipynb` (the roc_auc-free variant) as `base_metrics`. Port `tpr_at_fpr` verbatim from `samia_adaptations.ipynb`.

```python
# master_script/core/metrics.py
"""Metrics. Adv = 0.5*TPR + 0.5*TNR (Nguyen et al. 2023, Eq. 3).

base_metrics() intentionally omits roc_auc: reference_adaptations.ipynb does not
emit it, and consolidation must not change any notebook's document shape.
Attacks opt into extra keys through AttackSpec.extra_metrics.
"""
from typing import Dict, Sequence


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def base_metrics(trials: Sequence[Dict]) -> Dict:
    tp = sum(1 for row in trials if row["truth_member"] and row["pred_member"])
    tn = sum(1 for row in trials if not row["truth_member"] and not row["pred_member"])
    fp = sum(1 for row in trials if not row["truth_member"] and row["pred_member"])
    fn = sum(1 for row in trials if row["truth_member"] and not row["pred_member"])
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tpr": tpr, "tnr": tnr,
        "adv": 0.5 * tpr + 0.5 * tnr,
        "accuracy": (tp + tn) / len(trials) if trials else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "num_trials": len(trials),
    }


def summarize(trials: Sequence[Dict], spec) -> Dict:
    """base_metrics plus whatever the attack's extra_metrics hook contributes."""
    out = base_metrics(trials)
    if spec.extra_metrics is not None:
        out.update(spec.extra_metrics(trials))
    return out
```

Also add `tpr_at_fpr` ported verbatim from `samia_adaptations.ipynb`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/core/metrics.py tests/test_metrics.py
git commit -m "feat(core): add metrics with per-attack extra_metrics hook"
```

---

### Task 6: Federation layer

**Files:**
- Create: `master_script/core/federation.py`
- Test: `tests/test_federation.py`

**Interfaces:**
- Consumes: attack config instances.
- Produces: `federation.TARGET_RECORD`, `federation.HELD_OUT_RECORD`, `federation.CLIENT_CORPUS`, `federation.build_client_partitions(cfg, truth_member) -> list[list[str]]`, `federation.ToyFederatedLM`, `federation.toy_fedavg(global_model, client_models)`, `federation.run_toy_federated_finetune(cfg, truth_member) -> (ToyFederatedLM, history)`, `federation.run_hf_federated_finetune(cfg, truth_member) -> (bundle, history)` where `bundle = {"model", "tokenizer", "device"}`.

**Provenance:** port cell 9 and cell 11 of `zlib_adaptations.ipynb` verbatim. These are identical across the nine modern notebooks — verify with a diff before assuming.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_federation.py
from master_script.core import federation
from master_script.core.attacks.zlib import ZlibConfig


def test_positive_world_contains_target_record():
    parts = federation.build_client_partitions(ZlibConfig(), truth_member=True)
    assert federation.TARGET_RECORD in parts[0]


def test_negative_world_excludes_target_record():
    parts = federation.build_client_partitions(ZlibConfig(), truth_member=False)
    assert federation.TARGET_RECORD not in parts[0]
    assert federation.HELD_OUT_RECORD in parts[0]


def test_worlds_differ_only_in_target_payload():
    pos = federation.build_client_partitions(ZlibConfig(), truth_member=True)
    neg = federation.build_client_partitions(ZlibConfig(), truth_member=False)
    assert [p[:-1] for p in pos] == [p[:-1] for p in neg]


def test_partition_count_matches_num_clients():
    cfg = ZlibConfig(num_clients=6)
    assert len(federation.build_client_partitions(cfg, truth_member=True)) == 6


def test_toy_finetune_is_deterministic():
    cfg = ZlibConfig()
    a, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    b, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    assert a.nll(federation.TARGET_RECORD) == b.nll(federation.TARGET_RECORD)


def test_toy_membership_lowers_target_nll():
    cfg = ZlibConfig()
    member, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    nonmember, _ = federation.run_toy_federated_finetune(cfg, truth_member=False)
    assert member.nll(federation.TARGET_RECORD) < nonmember.nll(federation.TARGET_RECORD)


def test_history_has_one_entry_per_round():
    cfg = ZlibConfig(federated_rounds=3)
    _, history = federation.run_toy_federated_finetune(cfg, truth_member=True)
    assert len(history) == 3
    assert history[0]["round"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_federation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.core.federation'`

- [ ] **Step 3: Write minimal implementation**

Port from `zlib_adaptations.ipynb`:
- cell 9: `TARGET_RECORD`, `HELD_OUT_RECORD`, `CLIENT_CORPUS`, `build_client_partitions`, `ToyFederatedLM`, `toy_fedavg`, `run_toy_federated_finetune` — verbatim.
- cell 11: `run_hf_federated_finetune` — verbatim, but rename the inner client class `ZlibFlowerClient` → `FlowerClient` since it is attack-agnostic.

Keep all `import` statements for `torch`/`transformers`/`flwr` **inside** `run_hf_federated_finetune`, exactly as the notebook does. This is what lets the toy path and the test suite run without those packages installed.

Add a reference-model helper for `reference`/`spv_mia`/`wbc` (Task 8 consumes it):

```python
def load_reference_bundle(cfg):
    """Pre-trained (non-fine-tuned) reference model bundle."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = getattr(cfg, "reference_model_id", cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_federation.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/core/federation.py tests/test_federation.py
git commit -m "feat(core): port federated fine-tuning (toy + Flower FedAvg)"
```

---

### Task 7: Firestore layer

**Files:**
- Create: `master_script/core/firestore.py`
- Test: `tests/test_firestore.py`

**Interfaces:**
- Consumes: `config.experiment_key`.
- Produces: `firestore.get_firestore_client(project_id=None)`, `firestore.load_cached_result(cfg, spec=None)`, `firestore.save_result(cfg, result, spec=None) -> bool`, `firestore.mark_result_failed(cfg, error, spec=None)`, `firestore.publish_monitor_state(state: dict)`, `firestore.MONITOR_STATE_DOC`.

**Every document-id call takes `spec`** and forwards it to `experiment_key(cfg, spec)`, because `amia` and `loss` use different key formulas. Omitting it silently reads and writes the wrong document id for those two.

**Behavior note (ported deliberately):** `save_result` swallows **only** the missing-credentials case and returns `False`. A genuine write/serialization error (e.g. Firestore's nested-array rejection) must re-raise so it fails fast rather than masquerading as "not saved". This distinction is load-bearing — do not widen the `except`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_firestore.py
import pytest

from master_script.core import firestore as fs
from master_script.core.attacks.zlib import ZlibConfig


class _FakeDoc:
    def __init__(self, store, key):
        self._store, self._key = store, key

    def get(self):
        return _FakeSnap(self._store.get(self._key))

    def set(self, payload, merge=False):
        if any(isinstance(v, list) and v and isinstance(v[0], list) for v in payload.values()):
            raise ValueError("nested arrays are not supported")
        self._store[self._key] = payload


class _FakeSnap:
    def __init__(self, data):
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return self._data


class _FakeCollection:
    def __init__(self, store):
        self._store = store

    def document(self, key):
        return _FakeDoc(self._store, key)


class _FakeDB:
    def __init__(self):
        self.store = {}

    def collection(self, _name):
        return _FakeCollection(self.store)


def test_load_cached_returns_none_without_credentials(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("Set FIREBASE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS.")

    monkeypatch.setattr(fs, "get_firestore_client", _boom)
    assert fs.load_cached_result(ZlibConfig()) is None


def test_save_returns_false_without_credentials(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("Set FIREBASE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS.")

    monkeypatch.setattr(fs, "get_firestore_client", _boom)
    assert fs.save_result(ZlibConfig(), {"status": "complete"}) is False


def test_genuine_write_error_reraises_not_swallowed(monkeypatch):
    """A nested-array rejection must fail fast, not look like 'not saved'."""
    db = _FakeDB()
    monkeypatch.setattr(fs, "get_firestore_client", lambda *a, **k: db)
    with pytest.raises(ValueError, match="nested arrays"):
        fs.save_result(ZlibConfig(), {"federated_history": [[1, 2], [3]]})


def test_cache_hit_only_when_status_complete(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr(fs, "get_firestore_client", lambda *a, **k: db)
    cfg = ZlibConfig()
    from master_script.core.config import experiment_key

    db.store[experiment_key(cfg)] = {"status": "running"}
    assert fs.load_cached_result(cfg) is None
    db.store[experiment_key(cfg)] = {"status": "complete", "run_id": "x"}
    assert fs.load_cached_result(cfg)["run_id"] == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_firestore.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.core.firestore'`

- [ ] **Step 3: Write minimal implementation**

Port `get_firestore_client`, `load_cached_result`, and `save_result` from `zlib_adaptations.ipynb` cells 7 and 19, adding a `spec=None` parameter to the latter two and passing it through to `experiment_key(config, spec)`:

```python
def load_cached_result(config, spec=None):
    try:
        db = get_firestore_client(os.environ.get("FIREBASE_PROJECT_ID"))
    except Exception:
        return None  # Missing credentials: run locally, uncached.
    snapshot = db.collection(config.firestore_collection).document(
        experiment_key(config, spec)
    ).get()
    if snapshot.exists:
        payload = snapshot.to_dict()
        if payload.get("status") == "complete":
            return payload
    return None


def save_result(config, result, spec=None) -> bool:
    try:
        db = get_firestore_client(os.environ.get("FIREBASE_PROJECT_ID"))
    except Exception:
        # Missing-credentials case ONLY: fine to skip writing locally.
        return False
    # A real write/serialization error (e.g. nested-array rejection) propagates
    # so it fails fast rather than masquerading as "not saved". Do not widen.
    db.collection(config.firestore_collection).document(
        experiment_key(config, spec)
    ).set(result, merge=True)
    return True
```

Add:

```python
MONITOR_STATE_DOC = "monitor_state"


def mark_result_failed(config, error: str, spec=None) -> bool:
    """Record a failure. Never triggers artifact cleanup (see runner)."""
    try:
        db = get_firestore_client(os.environ.get("FIREBASE_PROJECT_ID"))
    except Exception:
        return False
    db.collection(config.firestore_collection).document(experiment_key(config, spec)).set(
        {
            "run_id": experiment_key(config, spec),
            "status": "failed",
            "updated_at_unix": int(time.time()),
            "config": asdict(config),
            "error": str(error)[:2000],
        },
        merge=True,
    )
    return True


def publish_monitor_state(state: dict, collection: str = "ami_federated_llm_results") -> bool:
    """Publish the optional run-state report (ideation doc §1.1).

    Coarse and transient: the currently-running run_ids and the planned sweep
    manifest. Never used to resume or reconstruct a run (§1.2).
    """
    try:
        db = get_firestore_client(os.environ.get("FIREBASE_PROJECT_ID"))
    except Exception:
        return False
    db.collection(collection).document(MONITOR_STATE_DOC).set(
        {**state, "updated_at_unix": int(time.time())}, merge=True
    )
    return True
```

Add a `.env` loader called once at import (`from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv(usecwd=True))`), wrapped in a try/except ImportError so `python-dotenv` stays optional.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_firestore.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/core/firestore.py tests/test_firestore.py
git commit -m "feat(core): add Firestore cache, save, fail-mark, monitor state"
```

---

### Task 8: Attack scoring — the nine modern attacks

**Files:**
- Modify: all nine of `master_script/core/attacks/{zlib,min_k,min_k_plus_plus,neighborhood,recall,reference,samia,spv_mia,wbc}.py`
- Create: `master_script/core/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `federation.ToyFederatedLM`, `federation.load_reference_bundle`, `metrics.roc_auc`, `metrics.tpr_at_fpr`.
- Produces: `scoring.ScoreContext` dataclass; each attack's `SPEC.score_toy` / `SPEC.score_hf` / `SPEC.extra_metrics` populated.

**Why a context object:** the notebooks' scoring signatures are mutually incompatible — `zlib` is `(model, text)`, `neighborhood` is `(model, text, config)`, `reference` is `(target, reference, text)`, `samia` is `(model, config)` with no text, `spv_mia` needs a reference bundle, and `wbc` scores inline inside `run_attack_trial`. A single context normalizes them without changing any math.

```python
# master_script/core/scoring.py
"""Uniform scoring interface over mutually incompatible notebook signatures."""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ScoreContext:
    """Everything any attack's scorer might need.

    target: ToyFederatedLM (toy path) or {"model","tokenizer","device"} (HF path).
    reference: same shape, or None for reference-free attacks.
    """
    config: Any
    target: Any
    text: str
    reference: Optional[Any] = None
```

Each attack's scorer becomes `score(ctx: ScoreContext) -> float`, wrapping its verbatim notebook body.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring.py
"""Every attack must separate members from non-members on the toy path.

This is the behavioral contract each notebook's smoke run asserts.
"""
import pytest

from master_script.core import federation
from master_script.core.registry import ATTACKS
from master_script.core.scoring import ScoreContext

TOY_ATTACKS = sorted(n for n, s in ATTACKS.items() if s.supports_toy and s.score_toy)


@pytest.mark.parametrize("attack", TOY_ATTACKS)
def test_member_scores_above_nonmember_on_toy_path(attack):
    spec = ATTACKS[attack]
    cfg = spec.config_cls()
    member, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    nonmember, _ = federation.run_toy_federated_finetune(cfg, truth_member=False)

    def _ctx(model):
        ref = None
        if spec.needs_reference:
            ref, _ = federation.run_toy_federated_finetune(cfg, truth_member=False)
        return ScoreContext(config=cfg, target=model, text=federation.TARGET_RECORD, reference=ref)

    assert spec.score_toy(_ctx(member)) > spec.score_toy(_ctx(nonmember))


@pytest.mark.parametrize("attack", TOY_ATTACKS)
def test_toy_scoring_is_deterministic(attack):
    spec = ATTACKS[attack]
    cfg = spec.config_cls()
    model, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    ctx = ScoreContext(config=cfg, target=model, text=federation.TARGET_RECORD)
    assert spec.score_toy(ctx) == spec.score_toy(ctx)


def test_zlib_score_matches_notebook_formula():
    """Guard the exact zlib formula: -(nll / zlib_entropy_bits)."""
    import zlib as _zlib

    from master_script.core.attacks.zlib import ZlibConfig, zlib_entropy_bits

    text = federation.TARGET_RECORD
    assert zlib_entropy_bits(text) == 8.0 * len(_zlib.compress(text.encode("utf-8")))


def test_samia_extra_metrics_include_tpr_at_fpr():
    trials = [
        {"truth_member": True, "score": 1.0, "pred_member": True},
        {"truth_member": False, "score": 0.0, "pred_member": False},
    ]
    out = ATTACKS["samia"].extra_metrics(trials)
    assert "roc_auc" in out and "tpr_at_fpr" in out


def test_reference_attack_has_no_extra_metrics():
    """reference_adaptations.ipynb emits no roc_auc; preserve that shape."""
    assert ATTACKS["reference"].extra_metrics is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.core.scoring'`

- [ ] **Step 3: Write minimal implementation**

Create `scoring.py` as above. Then for each attack, port its helper functions and both scorers verbatim from the notebook, adapting only the call signature. Worked example — `zlib`:

```python
# master_script/core/attacks/zlib.py  (additions)
import zlib as _zlib

from ..metrics import roc_auc
from ..scoring import ScoreContext


def zlib_entropy_bits(text: str) -> float:
    """Model-independent reference term: bits in the zlib-compressed text."""
    return 8.0 * len(_zlib.compress(text.encode("utf-8")))


def zlib_membership_score(target_nll: float, text: str) -> float:
    """Membership score = -(log_perplexity / zlib_entropy_bits). Higher => member."""
    return -(target_nll / zlib_entropy_bits(text))


def score_toy(ctx: ScoreContext) -> float:
    return zlib_membership_score(ctx.target.nll(ctx.text), ctx.text)


def score_hf(ctx: ScoreContext) -> float:
    import torch

    model, tokenizer, device = ctx.target["model"], ctx.target["tokenizer"], ctx.target["device"]
    encoded = tokenizer(ctx.text, return_tensors="pt", truncation=True, max_length=ctx.config.max_length)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, labels=encoded["input_ids"])
    return zlib_membership_score(float(outputs.loss.detach().cpu()), ctx.text)


def _extra_metrics(trials):
    return {"roc_auc": roc_auc([t["truth_member"] for t in trials], [t["score"] for t in trials])}


SPEC = AttackSpec(
    name="zlib",
    config_cls=ZlibConfig,
    methodology={...},
    score_toy=score_toy,
    score_hf=score_hf,
    extra_metrics=_extra_metrics,
)
```

Per-attack porting notes:

- **`min_k`** — port `score_candidate_toy(model, text, k_percent)` / `score_candidate_hf(bundle, text, k_percent, max_length)`; read `k_percent` from `ctx.config.min_k_percent`. `extra_metrics` → `roc_auc`.
- **`min_k_plus_plus`** — also port `_log_softmax`, `token_logprob_stats`, `token_zscore`, `min_k_plus_plus_membership_score`. `extra_metrics` → `roc_auc`.
- **`neighborhood`** — port `generate_neighbours_toy`, `generate_neighbours_bert`, `_mean_token_nll_hf`. Config already passed via `ctx.config`. `extra_metrics` → `roc_auc`.
- **`recall`** — port `build_prefix`, `_sequence_loglik_hf`; the notebook's `prefix` arg comes from `build_prefix(ctx.config)`. `extra_metrics` → `roc_auc`.
- **`reference`** — set `needs_reference=True`; scorers take `ctx.target` and `ctx.reference`. `extra_metrics=None` (**the notebook emits no roc_auc — this is intentional**).
- **`samia`** — port `zlib_bits`, `_tokenize`, `_ngrams`, `rouge_n_recall`, `samia_membership_score`, `sample_continuations_hf`, `tpr_at_fpr`. Its scorers take no `text` (they use `TARGET_RECORD` internally) — read from `ctx.text`. `extra_metrics` → `roc_auc` **and** `tpr_at_fpr`.
- **`spv_mia`** — set `needs_reference=True`; port `build_self_prompt_reference_toy/_hf`, `spv_membership_score`, `_mean_nll_hf`, `_prob_hf`, `hf_paraphrase`. `extra_metrics` → `roc_auc`.
- **`wbc`** — set `needs_reference=True`; port `windowed_sums`, `wbc_score`, `build_deltas`, `per_token_nll_hf`, and `config_to_storage`. Its scoring currently lives inside `run_attack_trial` — lift it into `score_toy`/`score_hf` without changing the math. `extra_metrics` → `roc_auc`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS (all parametrized cases + 3 specifics)

- [ ] **Step 5: Commit**

```bash
git add master_script/core/scoring.py master_script/core/attacks tests/test_scoring.py
git commit -m "feat(attacks): port scoring for the nine modern attacks"
```

---

### Task 9: AMIA and LOSS hooks

**Files:**
- Modify: `master_script/core/attacks/amia.py`, `master_script/core/attacks/loss.py`, `master_script/core/registry.py`
- Test: `tests/test_legacy_attacks.py`

**Interfaces:**
- Consumes: `AttackSpec`.
- Produces: `AttackSpec` gains `custom_trials: Callable | None` and `methodology_extra_keys` support; `amia.SPEC.supports_toy is False`; `loss.SPEC.calibrate_threshold`.

**Why:** these two don't fit the toy/HF scorer split. AMIA trains a malicious probe and thresholds a gradient signal with strict `>` (everything else uses `>=`). LOSS calibrates its threshold from non-member losses at a quantile, so its decision isn't a fixed constant. Rather than bend them into the modern shape, they supply a `custom_trials` hook that the runner calls instead of the standard trial loop.

Add to `AttackSpec`:

```python
    custom_trials: Optional[Callable] = None   # (config) -> list[trial dict]
    build_payload: Optional[Callable] = None   # (config, trials, artifact_dir) -> dict
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_legacy_attacks.py
import pytest

from master_script.core.registry import ATTACKS


@pytest.mark.parametrize("attack", ["amia", "loss"])
def test_legacy_attacks_declare_no_toy_support(attack):
    """Neither legacy notebook has a toy path.

    AMIA_adaptation.ipynb has no ToyFederatedLM at all; LOSS_adaptation.ipynb
    calls require_training_deps() and has no use_hf_models field to switch on.
    Both therefore require real HF models.
    """
    assert ATTACKS[attack].supports_toy is False


def test_legacy_configs_have_no_use_hf_models_field():
    """Guard the assumption that the toy/HF switch simply doesn't exist here."""
    from dataclasses import fields

    for attack in ("amia", "loss"):
        names = {f.name for f in fields(ATTACKS[attack].config_cls)}
        assert "use_hf_models" not in names


def test_amia_uses_strict_greater_than_threshold():
    """AMIA thresholds with `>` while every other attack uses `>=`. Preserve it."""
    from master_script.core.attacks.amia import predict_member

    cfg = ATTACKS["amia"].config_cls()
    assert predict_member(cfg.gradient_threshold, cfg) is False  # equal -> not member
    assert predict_member(cfg.gradient_threshold + 1e-9, cfg) is True


def test_loss_threshold_is_calibrated_not_constant():
    from master_script.core.attacks.loss import estimate_loss_threshold

    cfg = ATTACKS["loss"].config_cls()
    losses = [float(i) for i in range(100)]
    info = estimate_loss_threshold(losses, cfg)
    assert "threshold" in info
    assert info["threshold"] == pytest.approx(9.9, abs=0.5)  # 10th percentile


def test_loss_predicts_member_when_loss_below_threshold():
    from master_script.core.attacks.loss import predict_member_from_loss

    assert predict_member_from_loss(0.1, 1.0) is True
    assert predict_member_from_loss(2.0, 1.0) is False


def test_loss_uses_its_own_firestore_collection():
    assert ATTACKS["loss"].config_cls().firestore_collection == "loss_federated_llm_results"


def test_loss_methodology_has_attacker_observation_key():
    """LOSS_adaptation.ipynb emits an extra methodology key; preserve the shape."""
    assert "attacker_observation" in ATTACKS["loss"].methodology


def test_both_legacy_attacks_supply_custom_trials():
    assert ATTACKS["amia"].custom_trials is not None
    assert ATTACKS["loss"].custom_trials is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_legacy_attacks.py -v`
Expected: FAIL with `AttributeError: 'AttackSpec' object has no attribute 'custom_trials'`

- [ ] **Step 3: Write minimal implementation**

Extend `AttackSpec` with `custom_trials` and `build_payload`. Then:

**`amia.py`** — port from `AMIA_adaptation.ipynb`: `set_seed`, `build_client_texts`, `make_loader`, `TextDataset`, `get_parameters`, `set_parameters`, `build_model_and_tokenizer`, `client_device`, `federated_fine_tune`, `AMIFlowerClient`, `sentence_embedding`, `train_ami_probe`, `sample_attack_batch`, `run_attack_trials`, `build_result_payload`, `clear_experiment_objects`. Expose:

```python
def predict_member(score: float, config) -> bool:
    """AMIA uses strict `>`, unlike every other attack's `>=`. Verbatim."""
    return bool(score > config.gradient_threshold)
```

Wire `custom_trials=run_attack_trials`, `build_payload=build_result_payload`, `supports_toy=False`, and — critically — `key_fn=key_sha24_default_str` (AMIA truncates its digest to **24** chars, not 16).

**`loss.py`** — port from `LOSS_adaptation.ipynb`: `make_membership_world`, `require_training_deps`, `set_seed`, `load_model_and_tokenizer`, `get_parameters`, `set_parameters`, `client_device`, `federated_fine_tune`, `LossFlowerClient`, `estimate_loss_threshold`, `predict_member_from_loss`, `run_attack_trials`, `compact_trials`, `build_result_payload`. Keep the `methodology` dict verbatim **including** its `attacker_observation` key. Wire `custom_trials=run_attack_trials`, `build_payload=build_result_payload`, `supports_toy=False`, and `key_fn=key_named_prefix` (LOSS's document id is `f"{experiment_name}_{digest16}"`, e.g. `loss_federated_llm_adaptation_v1_952b56bc2ecd70aa`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_legacy_attacks.py -v`
Expected: PASS (7 tests)

Then confirm nothing regressed: `pytest tests/test_hash_equivalence.py -v` → still PASS.

- [ ] **Step 5: Commit**

```bash
git add master_script/core/attacks/amia.py master_script/core/attacks/loss.py master_script/core/registry.py tests/test_legacy_attacks.py
git commit -m "feat(attacks): port AMIA and LOSS via custom_trials hooks"
```

---

### Task 10: Runner

**Files:**
- Create: `master_script/core/runner.py`
- Test: `tests/test_runner.py`, `tests/test_toy_parity.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `runner.run_attack_trial(cfg, spec, trial_id, truth_member) -> dict`, `runner.run_attack_trials(cfg, spec) -> list[dict]`, `runner.run_single_experiment(cfg, spec, *, use_firestore=True, keep_artifacts=None) -> dict`, `runner.run_sweep(configs: list[tuple[cfg, spec]], **kw) -> list[dict]`, `runner.cleanup_artifacts(path)`.

**Behavioral contract (from `zlib_adaptations.ipynb` cell 15/19):**
- Trials alternate: `truth_member = (trial_id % 2 == 0)`.
- Each trial re-seeds via `replace(config, seed=config.seed + trial_id)`.
- **The per-trial `replace` must not be hashed** — the run's `run_id` comes from the *original* config. Hash first, then vary per trial.
- `federated_history` is written as `[{"trial_id": ..., "rounds": [...]}]` — a list of maps, never nested arrays.
- Cache check happens **before** compute; write happens **after** measurement.
- Cleanup only when `saved and not keep_artifacts`. **Never** clean up a failed run.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import pytest

from master_script.core import runner
from master_script.core.registry import ATTACKS


def test_trials_alternate_positive_and_negative(monkeypatch):
    spec = ATTACKS["zlib"]
    cfg = spec.config_cls(attack_trials=4)
    trials = runner.run_attack_trials(cfg, spec)
    assert [t["truth_member"] for t in trials] == [True, False, True, False]


def test_run_id_is_stable_despite_per_trial_reseed():
    """Per-trial seed bumps must not leak into the run's hash."""
    from master_script.core.config import experiment_key

    spec = ATTACKS["zlib"]
    cfg = spec.config_cls()
    before = experiment_key(cfg)
    runner.run_attack_trials(cfg, spec)
    assert experiment_key(cfg) == before


def test_cache_hit_skips_compute(monkeypatch):
    spec = ATTACKS["zlib"]
    cfg = spec.config_cls()
    cached = {"status": "complete", "run_id": "cached", "metrics": {"adv": 1.0}}
    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c: cached)

    def _boom(*a, **k):
        raise AssertionError("compute must not run on a cache hit")

    monkeypatch.setattr(runner, "run_attack_trials", _boom)
    assert runner.run_single_experiment(cfg, spec)["run_id"] == "cached"


def test_federated_history_is_list_of_maps_not_nested_arrays(monkeypatch):
    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c: None)
    monkeypatch.setattr(runner.firestore, "save_result", lambda c, r: False)
    spec = ATTACKS["zlib"]
    result = runner.run_single_experiment(spec.config_cls(attack_trials=2), spec)
    fh = result["federated_history"]
    assert isinstance(fh, list)
    assert all(isinstance(item, dict) and "rounds" in item for item in fh)


def test_failed_run_does_not_clean_artifacts(monkeypatch, tmp_path):
    spec = ATTACKS["zlib"]
    cfg = spec.config_cls()
    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c: None)
    monkeypatch.setattr(runner.firestore, "mark_result_failed", lambda c, e: True)
    monkeypatch.setattr(runner, "artifact_dir_for", lambda c: tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("attack exploded")

    monkeypatch.setattr(runner, "run_attack_trials", _boom)
    cleaned = []
    monkeypatch.setattr(runner, "cleanup_artifacts", lambda p: cleaned.append(p))
    with pytest.raises(RuntimeError, match="attack exploded"):
        runner.run_single_experiment(cfg, spec)
    assert cleaned == []


def test_no_firestore_flag_skips_cache_and_write(monkeypatch):
    spec = ATTACKS["zlib"]
    monkeypatch.setattr(
        runner.firestore, "load_cached_result",
        lambda c: (_ for _ in ()).throw(AssertionError("cache must not be read")),
    )
    result = runner.run_single_experiment(spec.config_cls(attack_trials=2), spec, use_firestore=False)
    assert result["status"] == "complete"
```

```python
# tests/test_toy_parity.py
"""Reproduce each notebook's smoke-run assertions through the consolidated core.

These are the notebooks' own acceptance criteria, re-asserted against core.
"""
import pytest

from master_script.core import runner
from master_script.core.registry import ATTACKS

TOY_ATTACKS = sorted(n for n, s in ATTACKS.items() if s.supports_toy)


@pytest.fixture(autouse=True)
def _no_firestore(monkeypatch):
    monkeypatch.setattr(runner.firestore, "load_cached_result", lambda c: None)
    monkeypatch.setattr(runner.firestore, "save_result", lambda c, r: False)


@pytest.mark.parametrize("attack", TOY_ATTACKS)
def test_smoke_run_produces_both_worlds_and_core_metrics(attack):
    spec = ATTACKS[attack]
    result = runner.run_single_experiment(spec.config_cls(attack_trials=4, use_hf_models=False), spec)
    assert result["metrics"]["num_trials"] == 4
    assert any(r["truth_member"] for r in result["attack_trials"])
    assert any(not r["truth_member"] for r in result["attack_trials"])
    for key in ("tpr", "tnr", "adv"):
        assert key in result["metrics"]


def test_zlib_smoke_roc_auc_is_one():
    """zlib_adaptations.ipynb asserts perfect ranking on the toy path."""
    spec = ATTACKS["zlib"]
    result = runner.run_single_experiment(spec.config_cls(attack_trials=4), spec)
    assert result["metrics"]["roc_auc"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner.py tests/test_toy_parity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.core.runner'`

- [ ] **Step 3: Write minimal implementation**

```python
# master_script/core/runner.py
"""Experiment orchestration. The single code path shared by the CLI and web UI.

Ordering is load-bearing and mirrors the notebooks exactly:
cache check -> FL fine-tune -> attack -> measure -> persist -> cleanup.
"""
from dataclasses import asdict, replace
from pathlib import Path
import logging
import shutil
import time

from . import federation, firestore
from .config import artifact_dir_for, experiment_key
from .metrics import summarize
from .scoring import ScoreContext

log = logging.getLogger(__name__)


def cleanup_artifacts(artifact_dir) -> None:
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


def run_attack_trial(config, spec, trial_id: int, truth_member: bool) -> dict:
    # Per-trial reseed, exactly as the notebooks do. NOTE: trial_config is never
    # hashed -- the run_id belongs to the original config.
    trial_config = replace(config, seed=config.seed + trial_id)
    if trial_config.use_hf_models:
        target, history = federation.run_hf_federated_finetune(trial_config, truth_member=truth_member)
        reference = federation.load_reference_bundle(trial_config) if spec.needs_reference else None
        score = spec.score_hf(ScoreContext(trial_config, target, federation.TARGET_RECORD, reference))
    else:
        target, history = federation.run_toy_federated_finetune(trial_config, truth_member=truth_member)
        reference = None
        if spec.needs_reference:
            reference, _ = federation.run_toy_federated_finetune(trial_config, truth_member=False)
        score = spec.score_toy(ScoreContext(trial_config, target, federation.TARGET_RECORD, reference))

    return {
        "trial_id": trial_id,
        "truth_member": bool(truth_member),
        "score": float(score),
        "pred_member": bool(score >= trial_config.threshold),
        "federated_history": history,
    }


def run_attack_trials(config, spec) -> list:
    return [
        run_attack_trial(config, spec, trial_id=i, truth_member=(i % 2 == 0))
        for i in range(config.attack_trials)
    ]


def run_single_experiment(config, spec, *, use_firestore: bool = True, keep_artifacts=None) -> dict:
    # Always pass spec: amia and loss have their own key formulas.
    run_id = experiment_key(config, spec)
    if use_firestore:
        cached = firestore.load_cached_result(config, spec)
        if cached and cached.get("status") == "complete":
            log.info("cache hit %s (%s); skipping compute", run_id, spec.name)
            return cached

    artifact_dir = artifact_dir_for(config, spec)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        if spec.custom_trials is not None:
            trials = spec.custom_trials(config)
        else:
            trials = run_attack_trials(config, spec)
    except Exception as exc:
        log.exception("run %s (%s) failed", run_id, spec.name)
        if use_firestore:
            firestore.mark_result_failed(config, str(exc), spec)
        raise  # NOTE: no cleanup -- a failed run keeps its artifacts.

    if spec.build_payload is not None:
        result = spec.build_payload(config, trials, artifact_dir)
        result.setdefault("run_id", run_id)
        result.setdefault("status", "complete")
        result.setdefault("updated_at_unix", int(time.time()))
    else:
        result = {
            "run_id": run_id,
            "status": "complete",
            "updated_at_unix": int(time.time()),
            "config": asdict(config),
            "methodology": spec.methodology,
            # Firestore forbids directly nested arrays: wrap each trial's
            # per-round history (a list) inside a map.
            "federated_history": [
                {"trial_id": r["trial_id"], "rounds": r["federated_history"]} for r in trials
            ],
            "metrics": summarize(trials, spec),
            "attack_trials": [
                {k: r[k] for k in ("trial_id", "truth_member", "score", "pred_member")}
                for r in trials
            ],
            "artifacts": {"artifact_dir": str(artifact_dir), "federated_model_path": None},
        }

    saved = firestore.save_result(config, result, spec) if use_firestore else False
    result["firestore_saved"] = saved
    keep = config.keep_artifacts if keep_artifacts is None else keep_artifacts
    if saved and not keep:
        cleanup_artifacts(artifact_dir)
    return result


def run_sweep(pairs, *, use_firestore: bool = True, keep_artifacts=None, on_run_start=None) -> list:
    """pairs: iterable of (config, spec). Sequential; --max-parallel is the CLI's job."""
    results = []
    for config, spec in pairs:
        if on_run_start is not None:
            on_run_start(experiment_key(config, spec), spec.name, config)
        results.append(
            run_single_experiment(config, spec, use_firestore=use_firestore, keep_artifacts=keep_artifacts)
        )
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runner.py tests/test_toy_parity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add master_script/core/runner.py tests/test_runner.py tests/test_toy_parity.py
git commit -m "feat(core): add runner shared by CLI and web UI"
```

---

### Task 11: YAML config loader

**Files:**
- Create: `master_script/core/yaml_config.py`, `master_script/configs/smoke.yaml`, `master_script/configs/example_sweep.yaml`
- Test: `tests/test_yaml_config.py`

**Interfaces:**
- Consumes: `registry.ATTACKS`, `config.expand_sweep`.
- Produces: `yaml_config.load_config_file(path, only=None) -> list[tuple[cfg, spec]]`, `yaml_config.ConfigError`.

**Merge rules (spec §4.4):**
- `defaults:` applies to every attack; keys **not present** on a given attack's dataclass are silently skipped for it (that's the point of a cross-attack block).
- Keys inside an attack's `base:`/`sweep:` that aren't dataclass fields are a **hard error**, before any compute.
- Precedence: dataclass defaults < `defaults:` < attack `base:` < attack `sweep:` value.
- `--attack` filtering (`only`) is applied after parsing.
- `amia` without `use_hf_models=True` is rejected here (no toy path).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_yaml_config.py
import pytest

from master_script.core.yaml_config import ConfigError, load_config_file

SMOKE = """
defaults:
  seed: 11
attacks:
  zlib:
    base: {threshold: -0.006}
    sweep:
      federated_rounds: [1, 2]
"""


def _write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_expands_grid_into_one_pair_per_combination(tmp_path):
    pairs = load_config_file(_write(tmp_path, SMOKE))
    assert len(pairs) == 2
    assert {c.federated_rounds for c, _ in pairs} == {1, 2}


def test_defaults_apply_to_attack(tmp_path):
    pairs = load_config_file(_write(tmp_path, SMOKE))
    assert all(c.seed == 11 for c, _ in pairs)


def test_base_overrides_defaults(tmp_path):
    pairs = load_config_file(_write(tmp_path, SMOKE))
    assert all(c.threshold == -0.006 for c, _ in pairs)


def test_defaults_key_absent_on_attack_is_skipped_not_error(tmp_path):
    """min_k_percent exists on min_k but not zlib; defaults must tolerate that."""
    text = """
defaults:
  min_k_percent: 30
attacks:
  zlib: {}
  min_k: {}
"""
    pairs = load_config_file(_write(tmp_path, text))
    by_name = {spec.name: cfg for cfg, spec in pairs}
    assert by_name["min_k"].min_k_percent == 30
    assert not hasattr(by_name["zlib"], "min_k_percent")


def test_unknown_key_in_attack_base_is_hard_error(tmp_path):
    text = """
attacks:
  zlib:
    base: {epsilon: 8}
"""
    with pytest.raises(ConfigError, match="epsilon"):
        load_config_file(_write(tmp_path, text))


def test_unknown_attack_name_is_error(tmp_path):
    with pytest.raises(ConfigError, match="nonexistent"):
        load_config_file(_write(tmp_path, "attacks:\n  nonexistent: {}\n"))


def test_only_filter_selects_subset(tmp_path):
    text = "attacks:\n  zlib: {}\n  min_k: {}\n"
    pairs = load_config_file(_write(tmp_path, text), only=["zlib"])
    assert [spec.name for _, spec in pairs] == ["zlib"]


def test_amia_without_hf_models_is_rejected(tmp_path):
    text = "attacks:\n  amia:\n    base: {use_hf_models: false}\n"
    with pytest.raises(ConfigError, match="toy"):
        load_config_file(_write(tmp_path, text))


def test_shipped_smoke_config_loads(tmp_path):
    from master_script.paths import CONFIGS_DIR

    pairs = load_config_file(CONFIGS_DIR / "smoke.yaml")
    assert pairs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yaml_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.core.yaml_config'`

- [ ] **Step 3: Write minimal implementation**

```python
# master_script/core/yaml_config.py
"""YAML -> (config, spec) pairs, with fail-fast validation.

Unknown keys inside an attack's base/sweep are errors: catching a typo here
saves a multi-hour run that would otherwise produce a differently-hashed,
silently-wrong experiment.
"""
from dataclasses import fields, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

from .config import expand_sweep
from .registry import ATTACKS


class ConfigError(ValueError):
    """Raised for any malformed config, always before compute starts."""


def _field_names(cls) -> set:
    return {f.name for f in fields(cls)}


def load_config_file(path, only: Optional[Sequence[str]] = None) -> List[Tuple[object, object]]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    doc = yaml.safe_load(path.read_text()) or {}
    defaults = doc.get("defaults") or {}
    attacks = doc.get("attacks") or {}
    if not attacks:
        raise ConfigError(f"{path}: no 'attacks:' section")

    unknown = sorted(set(attacks) - set(ATTACKS))
    if unknown:
        raise ConfigError(
            f"{path}: unknown attack(s): {', '.join(unknown)}. Known: {', '.join(sorted(ATTACKS))}"
        )

    selected = list(attacks) if only is None else [a for a in attacks if a in set(only)]
    pairs: List[Tuple[object, object]] = []
    for name in selected:
        spec = ATTACKS[name]
        allowed = _field_names(spec.config_cls)
        section = attacks[name] or {}
        base_over = section.get("base") or {}
        sweep = section.get("sweep") or {}

        bad = sorted((set(base_over) | set(sweep)) - allowed)
        if bad:
            raise ConfigError(
                f"{path}: attack '{name}' has unknown field(s): {', '.join(bad)}. "
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

`configs/smoke.yaml` — reproduces the notebooks' smoke defaults:

```yaml
# Smoke config: reproduces the notebooks' smoke runs.
# No GPU, no model download, no credentials required.
defaults:
  use_hf_models: false
  attack_trials: 4

attacks:
  zlib: {}
  min_k: {}
  min_k_plus_plus: {}
  neighborhood: {}
  recall: {}
  reference: {}
  samia: {}
  spv_mia: {}
  wbc: {}
```

`configs/example_sweep.yaml` — a real factor sweep:

```yaml
# Real federated fine-tuning sweep. Requires GPU + Firebase credentials.
defaults:
  model_id: distilgpt2
  use_hf_models: true
  sim_num_gpus: 1.0
  attack_trials: 12

attacks:
  zlib:
    sweep:
      federated_rounds: [1, 2, 4]
      seed: [7, 11, 23]
  min_k:
    base: {min_k_percent: 20}
    sweep:
      federated_rounds: [1, 2, 4]
      seed: [7, 11, 23]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_yaml_config.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/core/yaml_config.py master_script/configs tests/test_yaml_config.py
git commit -m "feat(core): add YAML config loader with fail-fast validation"
```

---

### Task 12: Charts

**Files:**
- Create: `master_script/core/charts.py`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: `paths.CHARTS_DIR`.
- Produces: `charts.render_adv_by_factor(results, factor, out_name=None) -> Path`, `charts.render_score_distribution(result, out_name=None) -> Path`, `charts.render_sweep_summary(results) -> list[Path]`.

**Note:** charts are timestamped so repeated runs don't overwrite each other. Use the non-interactive `Agg` backend — this runs headless on a VM.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_charts.py
from master_script.core import charts
from master_script.paths import CHARTS_DIR

RESULTS = [
    {"config": {"attack_name": "zlib", "federated_rounds": 1}, "metrics": {"adv": 0.5}},
    {"config": {"attack_name": "zlib", "federated_rounds": 2}, "metrics": {"adv": 0.75}},
    {"config": {"attack_name": "zlib", "federated_rounds": 4}, "metrics": {"adv": 1.0}},
]


def test_adv_by_factor_writes_into_charts_dir():
    out = charts.render_adv_by_factor(RESULTS, "federated_rounds")
    assert out.exists()
    assert out.parent == CHARTS_DIR
    assert out.suffix == ".png"
    out.unlink()


def test_chart_filenames_are_timestamped_so_runs_do_not_clobber():
    a = charts.render_adv_by_factor(RESULTS, "federated_rounds")
    b = charts.render_adv_by_factor(RESULTS, "federated_rounds")
    assert a != b
    a.unlink()
    b.unlink()


def test_score_distribution_splits_by_true_membership():
    result = {
        "config": {"attack_name": "zlib"},
        "run_id": "abc123",
        "attack_trials": [
            {"trial_id": 0, "truth_member": True, "score": 0.9, "pred_member": True},
            {"trial_id": 1, "truth_member": False, "score": 0.1, "pred_member": False},
        ],
    }
    out = charts.render_score_distribution(result)
    assert out.exists()
    out.unlink()


def test_empty_results_returns_no_charts():
    assert charts.render_sweep_summary([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_charts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.core.charts'`

- [ ] **Step 3: Write minimal implementation**

```python
# master_script/core/charts.py
"""Chart rendering into master_script/outputs/Charts/.

Headless by construction (Agg): this runs on a GPU VM with no display.
Filenames carry a timestamp so repeated sweeps never overwrite each other.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..paths import CHARTS_DIR  # noqa: E402


def _stamp(name: str) -> Path:
    return CHARTS_DIR / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{time.perf_counter_ns() % 1000:03d}.png"


def render_adv_by_factor(results: List[Dict], factor: str, out_name: Optional[str] = None) -> Path:
    """Adv vs. a sweep factor, one series per attack (ideation doc §3.1)."""
    series = defaultdict(list)
    for r in results:
        cfg, met = r.get("config", {}), r.get("metrics", {})
        if factor in cfg and "adv" in met:
            series[cfg.get("attack_name", "unknown")].append((cfg[factor], met["adv"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for attack, points in sorted(series.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=attack)
    ax.set_xlabel(factor)
    ax.set_ylabel("Adv = 0.5*TPR + 0.5*TNR")
    ax.set_title(f"Attack advantage vs. {factor}")
    ax.axhline(0.5, linestyle="--", linewidth=1, color="grey")
    ax.annotate("chance", (0.02, 0.51), xycoords=("axes fraction", "data"), fontsize=8, color="grey")
    ax.set_ylim(0, 1.05)
    if series:
        ax.legend()
    fig.tight_layout()
    out = _stamp(out_name or f"adv_by_{factor}")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_score_distribution(result: Dict, out_name: Optional[str] = None) -> Path:
    """Score distribution split by true membership (ideation doc §3.3)."""
    trials = result.get("attack_trials", [])
    members = [t["score"] for t in trials if t["truth_member"]]
    nonmembers = [t["score"] for t in trials if not t["truth_member"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist([members, nonmembers], bins=12, label=["member", "non-member"])
    ax.set_xlabel("membership score")
    ax.set_ylabel("trials")
    ax.set_title(f"Score distribution — {result.get('config', {}).get('attack_name', '?')} "
                 f"{result.get('run_id', '')[:8]}")
    ax.legend()
    fig.tight_layout()
    out = _stamp(out_name or f"scores_{result.get('run_id', 'run')[:8]}")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_sweep_summary(results: List[Dict]) -> List[Path]:
    """One Adv-vs-factor chart per factor that actually varies in this sweep."""
    if not results:
        return []
    varying = []
    for factor in ("federated_rounds", "num_clients", "local_epochs", "client_lr", "seed"):
        values = {r.get("config", {}).get(factor) for r in results if factor in r.get("config", {})}
        if len(values) > 1:
            varying.append(factor)
    return [render_adv_by_factor(results, f) for f in varying]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_charts.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/core/charts.py tests/test_charts.py
git commit -m "feat(core): add chart rendering to outputs/Charts"
```

---

### Task 13: CLI

**Files:**
- Create: `master_script/perform_experiments.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `yaml_config.load_config_file`, `runner.run_sweep`, `charts.render_sweep_summary`, `gpu.apply_gpu_selection`, `logging_setup.setup_session_logging`.
- Produces: `main(argv=None) -> int`, `build_parser() -> argparse.ArgumentParser`.

**Import-order constraint:** `gpu.apply_gpu_selection()` must run **before** anything imports `torch`. Call it inside `main()` before importing the runner, exactly as the notebooks put the GPU cell first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import pytest

from master_script.perform_experiments import build_parser, main


def test_list_attacks_prints_all_eleven(capsys):
    assert main(["--list-attacks"]) == 0
    out = capsys.readouterr().out
    for name in ("zlib", "min_k", "min_k_plus_plus", "neighborhood", "recall",
                 "reference", "samia", "spv_mia", "wbc", "amia", "loss"):
        assert name in out


def test_dry_run_prints_run_ids_and_runs_nothing(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("attacks:\n  zlib:\n    sweep:\n      seed: [7, 11]\n")

    import master_script.perform_experiments as cli

    monkeypatch.setattr(
        cli, "_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    assert main(["--config", str(cfg), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 run(s)" in out


def test_attack_flag_filters_subset(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("attacks:\n  zlib: {}\n  min_k: {}\n")
    main(["--config", str(cfg), "--dry-run", "--attack", "zlib"])
    out = capsys.readouterr().out
    assert "zlib" in out and "min_k" not in out


def test_unknown_config_key_exits_nonzero_before_compute(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("attacks:\n  zlib:\n    base: {epsilon: 8}\n")
    assert main(["--config", str(cfg), "--dry-run"]) == 2
    assert "epsilon" in capsys.readouterr().err


def test_max_parallel_rejects_more_than_two():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--max-parallel", "3"])


def test_defaults_to_smoke_config():
    args = build_parser().parse_args([])
    assert args.config.name == "smoke.yaml"
    assert args.max_parallel == 1


def test_parallel_yaml_roundtrip_preserves_run_id(tmp_path):
    """--max-parallel 2 ships each config to a subprocess as YAML.

    wbc.window_sizes is a tuple that returns from YAML as a list. json.dumps
    renders both as [2,3], so the hash survives -- but this is load-bearing and
    silent, so pin it. If this breaks, parallel runs recompute everything.
    """
    from dataclasses import asdict

    import yaml

    from master_script.core.config import experiment_key
    from master_script.core.registry import ATTACKS
    from master_script.core.yaml_config import load_config_file

    for name in ("wbc", "zlib"):
        spec = ATTACKS[name]
        cfg = spec.config_cls()
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump({"attacks": {name: {"base": asdict(cfg)}}}))
        (reloaded, reloaded_spec), = load_config_file(path)
        assert experiment_key(reloaded, reloaded_spec) == experiment_key(cfg, spec), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.perform_experiments'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Run MIA adaptation experiments from a YAML config.

Consolidates the 11 notebooks in code_experiments/adaptations/. Attack and
evaluation behavior is identical to the notebooks; run_id hashes are preserved
so existing Firestore results still cache-hit.

Examples
--------
List every registered attack:
    python perform_experiments.py --list-attacks

See what a sweep would do, without spending GPU hours:
    python perform_experiments.py --config configs/example_sweep.yaml --dry-run

Run the smoke suite (no GPU, no credentials):
    python perform_experiments.py --config configs/smoke.yaml

Run one attack for real, on both GPUs, with charts:
    python perform_experiments.py --config configs/example_sweep.yaml \\
        --attack zlib --max-parallel 2 --charts
"""
from pathlib import Path
import argparse
import sys

from .paths import CONFIGS_DIR


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="perform_experiments.py",
        description="Run membership-inference attack experiments against federated LLM fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", type=Path, default=CONFIGS_DIR / "smoke.yaml",
                   help="YAML config file (default: configs/smoke.yaml)")
    p.add_argument("--attack", action="append", dest="attacks", metavar="NAME",
                   help="Attack to run; repeatable. Default: every attack in the config.")
    p.add_argument("--list-attacks", action="store_true",
                   help="Print the attack registry and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Expand the grid, print run_ids and cache status, run nothing.")
    p.add_argument("--max-parallel", type=int, default=1, choices=(1, 2),
                   help="Concurrent runs. 2 spawns one GPU-pinned subprocess per run (2-GPU VM).")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="Keep local model/probe artifacts after a successful persist.")
    p.add_argument("--no-firestore", action="store_true",
                   help="Local-only: skip the cache read and the result write.")
    p.add_argument("--charts", dest="charts", action="store_true", default=True,
                   help="Render Adv charts into outputs/Charts/ (default).")
    p.add_argument("--no-charts", dest="charts", action="store_false",
                   help="Skip chart rendering.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Logging verbosity.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from .core.registry import ATTACKS

    if args.list_attacks:
        print(f"{len(ATTACKS)} registered attacks:\n")
        for name in sorted(ATTACKS):
            spec = ATTACKS[name]
            toy = "toy+hf" if spec.supports_toy else "hf only"
            print(f"  {name:<18} {toy:<8} {spec.config_cls.__name__}")
        return 0

    from .core.yaml_config import ConfigError, load_config_file

    try:
        pairs = load_config_file(args.config, only=args.attacks)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return _dry_run(pairs, use_firestore=not args.no_firestore)

    # GPU pinning MUST precede any torch import.
    from .core.gpu import apply_gpu_selection

    apply_gpu_selection()
    from .logging_setup import setup_session_logging

    log_path = setup_session_logging(args.log_level)
    print(f"session log: {log_path}")
    return _run(pairs, args)


def _dry_run(pairs, use_firestore: bool) -> int:
    from .core.config import experiment_key
    from .core.firestore import load_cached_result

    cached_n = 0
    print(f"{len(pairs)} run(s) planned\n")
    for cfg, spec in pairs:
        run_id = experiment_key(cfg, spec)
        status = "pending"
        if use_firestore and load_cached_result(cfg, spec):
            status = "cached"
            cached_n += 1
        print(f"  {run_id}  {spec.name:<18} {status}")
    print(f"\n{cached_n} cached, {len(pairs) - cached_n} pending")
    return 0


def _run(pairs, args) -> int:
    from .core.runner import run_sweep

    if args.max_parallel == 2:
        results = _run_parallel(pairs, args)
    else:
        results = run_sweep(
            pairs,
            use_firestore=not args.no_firestore,
            keep_artifacts=args.keep_artifacts or None,
        )

    if args.charts:
        from .core.charts import render_sweep_summary

        for path in render_sweep_summary([r for r in results if r]):
            print(f"chart: {path}")

    ok = sum(1 for r in results if r and r.get("status") == "complete")
    print(f"\n{ok}/{len(pairs)} run(s) complete")
    for r in results:
        if r and r.get("metrics"):
            print(f"  {r.get('run_id')}  adv={r['metrics'].get('adv'):.3f}")
    return 0 if ok == len(pairs) else 1
```

`_run_parallel` spawns one subprocess per run with a distinct `EXPERIMENT_GPU`, capped at two in flight:

```python
def _run_parallel(pairs, args) -> list:
    """One GPU-pinned subprocess per run, at most two in flight (2-GPU VM).

    Subprocesses (not threads): Flower/Ray and CUDA do not share a process
    cleanly, and each run must pin its own GPU before torch initializes.
    """
    import json
    import os
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from .core.config import experiment_key

    def _one(item):
        index, (cfg, spec) = item
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            from dataclasses import asdict

            import yaml

            yaml.safe_dump({"attacks": {spec.name: {"base": asdict(cfg)}}}, fh)
            single = fh.name
        env = {**os.environ, "EXPERIMENT_GPU": str(index % 2)}
        cmd = [sys.executable, "-m", "master_script.perform_experiments",
               "--config", single, "--max-parallel", "1", "--no-charts",
               "--log-level", args.log_level]
        if args.no_firestore:
            cmd.append("--no-firestore")
        if args.keep_artifacts:
            cmd.append("--keep-artifacts")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        os.unlink(single)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return None
        return {"run_id": experiment_key(cfg, spec), "status": "complete", "config": asdict(cfg)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(_one, enumerate(pairs)))
```

Add `if __name__ == "__main__": raise SystemExit(main())`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (6 tests)

Then run it for real: `python -m master_script.perform_experiments --config master_script/configs/smoke.yaml --no-firestore`
Expected: 9 runs complete, each printing an `adv=` line.

- [ ] **Step 5: Commit**

```bash
git add master_script/perform_experiments.py tests/test_cli.py
git commit -m "feat(cli): add perform_experiments CLI runner"
```

---

### Task 14: Web UI — state layer

**Files:**
- Create: `master_script/webui/state.py`
- Test: `tests/test_webui_state.py`

**Interfaces:**
- Consumes: `firestore.get_firestore_client`, `firestore.MONITOR_STATE_DOC`.
- Produces: `state.DashboardState` with `.runs: dict[str, dict]`, `.running: list[dict]`, `.manifest: list[dict] | None`, `.ingest(docs)`, `.attach_listener(on_change)`, `.filtered(attack=None, status=None, **factors)`, `.aggregate_by(field)`, `.recently_finished(limit=10)`.

**Per spec §1.2:** the dashboard holds **no durable state**. `DashboardState` is a pure in-memory projection, rebuilt by re-reading Firestore on restart. Never persist it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_webui_state.py
from master_script.webui.state import DashboardState

DOCS = [
    {"run_id": "a", "status": "complete", "updated_at_unix": 100,
     "config": {"attack_name": "zlib", "federated_rounds": 1, "seed": 7},
     "metrics": {"adv": 0.5, "tpr": 0.5, "tnr": 0.5}},
    {"run_id": "b", "status": "complete", "updated_at_unix": 200,
     "config": {"attack_name": "zlib", "federated_rounds": 2, "seed": 7},
     "metrics": {"adv": 1.0, "tpr": 1.0, "tnr": 1.0}},
    {"run_id": "c", "status": "failed", "updated_at_unix": 300,
     "config": {"attack_name": "min_k", "federated_rounds": 1, "seed": 7}},
]


def test_ingest_indexes_runs_by_id():
    s = DashboardState()
    s.ingest(DOCS)
    assert set(s.runs) == {"a", "b", "c"}


def test_monitor_state_doc_is_not_treated_as_a_run():
    s = DashboardState()
    s.ingest(DOCS + [{"run_id": "monitor_state", "running": [], "manifest": []}])
    assert "monitor_state" not in s.runs


def test_filter_by_attack():
    s = DashboardState()
    s.ingest(DOCS)
    assert {r["run_id"] for r in s.filtered(attack="zlib")} == {"a", "b"}


def test_filter_by_status():
    s = DashboardState()
    s.ingest(DOCS)
    assert [r["run_id"] for r in s.filtered(status="failed")] == ["c"]


def test_filter_by_config_factor():
    s = DashboardState()
    s.ingest(DOCS)
    assert {r["run_id"] for r in s.filtered(federated_rounds=1)} == {"a", "c"}


def test_filters_compose():
    s = DashboardState()
    s.ingest(DOCS)
    assert [r["run_id"] for r in s.filtered(attack="zlib", federated_rounds=2)] == ["b"]


def test_aggregate_reports_mean_adv_per_attack():
    s = DashboardState()
    s.ingest(DOCS)
    agg = s.aggregate_by("attack_name")
    assert agg["zlib"]["mean_adv"] == 0.75
    assert agg["zlib"]["count"] == 2


def test_aggregate_ignores_failed_runs_without_metrics():
    s = DashboardState()
    s.ingest(DOCS)
    assert "min_k" not in s.aggregate_by("attack_name")


def test_recently_finished_is_newest_first():
    s = DashboardState()
    s.ingest(DOCS)
    assert [r["run_id"] for r in s.recently_finished()] == ["c", "b", "a"]


def test_running_set_empty_without_monitor_state():
    """Firestore alone cannot identify in-progress runs (spec §2.4)."""
    s = DashboardState()
    s.ingest(DOCS)
    assert s.running == []
    assert s.manifest is None


def test_running_set_read_from_monitor_state_doc():
    s = DashboardState()
    s.ingest(DOCS + [{
        "run_id": "monitor_state",
        "running": [{"run_id": "z", "attack": "zlib", "started_unix": 10}],
        "manifest": [{"run_id": "z"}, {"run_id": "a"}],
    }])
    assert s.running[0]["attack"] == "zlib"
    assert len(s.manifest) == 2


def test_sweep_progress_needs_manifest_for_denominator():
    s = DashboardState()
    s.ingest(DOCS)
    progress = s.sweep_progress()
    assert progress["complete"] == 2
    assert progress["failed"] == 1
    assert progress["total"] is None  # no manifest -> no denominator
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webui_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.webui.state'`

- [ ] **Step 3: Write minimal implementation**

```python
# master_script/webui/state.py
"""In-memory projection of Firestore. Deliberately non-durable (spec §1.2).

On restart the dashboard rebuilds entirely by re-reading Firestore. Nothing
here is ever persisted, and nothing here is ever used to resume a run.
"""
from statistics import mean
from typing import Callable, Dict, List, Optional

from ..core.firestore import MONITOR_STATE_DOC, get_firestore_client


class DashboardState:
    def __init__(self) -> None:
        self.runs: Dict[str, dict] = {}
        self.running: List[dict] = []
        self.manifest: Optional[List[dict]] = None
        self._listener = None

    def ingest(self, docs) -> None:
        for doc in docs:
            if doc.get("run_id") == MONITOR_STATE_DOC:
                self.running = doc.get("running") or []
                self.manifest = doc.get("manifest")
                continue
            if "run_id" in doc:
                self.runs[doc["run_id"]] = doc

    def attach_listener(self, collection: str, on_change: Callable[[], None]) -> None:
        """Firestore real-time listener; on_change fires on the SDK's thread."""
        db = get_firestore_client()

        def _cb(snapshots, changes, read_time):
            self.ingest([s.to_dict() | {"run_id": s.id} for s in snapshots])
            on_change()

        self._listener = db.collection(collection).on_snapshot(_cb)

    def filtered(self, attack: Optional[str] = None, status: Optional[str] = None, **factors) -> List[dict]:
        out = []
        for run in self.runs.values():
            cfg = run.get("config", {})
            if attack and cfg.get("attack_name") != attack:
                continue
            if status and run.get("status") != status:
                continue
            if any(cfg.get(k) != v for k, v in factors.items()):
                continue
            out.append(run)
        return out

    def aggregate_by(self, field: str) -> Dict[str, dict]:
        buckets: Dict[str, List[float]] = {}
        for run in self.runs.values():
            adv = (run.get("metrics") or {}).get("adv")
            key = run.get("config", {}).get(field)
            if adv is None or key is None:
                continue
            buckets.setdefault(str(key), []).append(adv)
        return {
            k: {"mean_adv": mean(v), "max_adv": max(v), "min_adv": min(v), "count": len(v)}
            for k, v in buckets.items()
        }

    def recently_finished(self, limit: int = 10) -> List[dict]:
        return sorted(
            self.runs.values(), key=lambda r: r.get("updated_at_unix", 0), reverse=True
        )[:limit]

    def sweep_progress(self) -> dict:
        """Complete/failed come from Firestore; total/pending need the manifest (§2.2)."""
        complete = sum(1 for r in self.runs.values() if r.get("status") == "complete")
        failed = sum(1 for r in self.runs.values() if r.get("status") == "failed")
        total = len(self.manifest) if self.manifest is not None else None
        pending = None if total is None else max(0, total - complete - failed - len(self.running))
        return {
            "complete": complete, "failed": failed, "running": len(self.running),
            "total": total, "pending": pending,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_webui_state.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/webui/state.py tests/test_webui_state.py
git commit -m "feat(webui): add non-durable Firestore projection"
```

---

### Task 15: Web UI — monitor, results, and detail pages

**Files:**
- Create: `master_script/webui/monitor.py`, `master_script/webui/results.py`, `master_script/webui/app.py`
- Test: `tests/test_webui_pages.py`

**Interfaces:**
- Consumes: `state.DashboardState`, `charts`, `registry.ATTACKS`.
- Produces: `monitor.render(state)`, `results.render(state)`, `results.render_detail(state, run_id)`, `app.build(state) -> None`, `app.main()`.

**Spec mapping:** monitor = §2.1 now-panel, §2.2 sweep progress, §2.3 recently-finished. Results = §3.1 grid + aggregates + comparison plots, §3.2 composable filters, §3.3 detail page, §3.4 privacy direction. §2.5 (GPU/log panel) is out of scope.

Pages are thin: all data logic lives in `state.py` and is tested there. These tests cover the wiring only.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_webui_pages.py
"""Wiring tests. Data logic is tested in tests/test_webui_state.py."""
from master_script.webui import app, monitor, results
from master_script.webui.state import DashboardState


def test_pages_expose_render_entrypoints():
    assert callable(monitor.render)
    assert callable(results.render)
    assert callable(results.render_detail)


def test_app_registers_three_routes():
    assert sorted(app.ROUTES) == ["/", "/results", "/tunnel"]


def test_privacy_direction_label_reads_lower_adv_as_improved():
    """§3.4: lower Adv within an environment => privacy improved."""
    assert "improved" in results.privacy_direction(baseline_adv=0.9, current_adv=0.6).lower()
    assert "declined" in results.privacy_direction(baseline_adv=0.6, current_adv=0.9).lower()


def test_privacy_direction_reports_unchanged_within_tolerance():
    assert "unchanged" in results.privacy_direction(0.7, 0.7).lower()


def test_monitor_reports_unavailable_running_set_without_manifest():
    """§2.4: say so rather than guessing."""
    s = DashboardState()
    s.ingest([{"run_id": "a", "status": "complete", "config": {}, "metrics": {"adv": 1.0}}])
    assert monitor.running_set_available(s) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webui_pages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.webui.monitor'`

- [ ] **Step 3: Write minimal implementation**

`monitor.py` — implement `render(state)` with three `@ui.refreshable` sections:

```python
def running_set_available(state) -> bool:
    """Without the script's run-state report, in-progress runs are unknowable (§2.4)."""
    return bool(state.running) or state.manifest is not None
```

- **Now panel (§2.1):** for each running run show attack name, `run_id`, distinguishing config (`model_id`, `federated_rounds`, `seed`), start time, elapsed wall-clock, and coarse stage if reported. When `running_set_available()` is `False`, render an explicit notice: *"Currently-running set unavailable — no run-state report. Showing completed/failed Firestore results only."*
- **Sweep progress (§2.2):** render `state.sweep_progress()`. When `total is None`, show complete/failed counts with **no denominator** and label it "denominator unavailable".
- **Recently finished (§2.3):** `state.recently_finished(10)` with `Adv` for complete and a failure indicator for failed.

`results.py`:

```python
_TOLERANCE = 0.02


def privacy_direction(baseline_adv: float, current_adv: float) -> str:
    """§3.4: report the direction the runs imply. Makes no privacy claim of its own."""
    delta = current_adv - baseline_adv
    if abs(delta) <= _TOLERANCE:
        return f"Privacy unchanged (ΔAdv = {delta:+.3f})"
    if delta < 0:
        return f"Privacy improved — attack advantage fell (ΔAdv = {delta:+.3f})"
    return f"Privacy declined — attack advantage rose (ΔAdv = {delta:+.3f})"
```

- **Grid (§3.1):** `ui.table` over `state.filtered(...)`, one row per `run_id`; columns: attack, `run_id`, status, `updated_at`, config factors, `adv`, `tpr`, `tnr`, `num_trials`. Sortable on every column.
- **Filters (§3.2):** attack select, status select, config-factor selects, time window. Composable; every change calls `.refresh()` so aggregates and plots recompute over the active filter.
- **Aggregates + plots (§3.1):** `state.aggregate_by("attack_name")` and an `Adv`-vs-factor plot via `ui.matplotlib`, reusing `core.charts` logic.
- **Detail (§3.3):** `render_detail(state, run_id)` — header (attack, `run_id`, status, `updated_at`, full config dump), methodology block, headline `Adv`/`TPR`/`TNR`/`num_trials`, `federated_history` per-round curve, `attack_trials` table, score distribution split by true membership, ROC view, and artifact paths (noting they may point at cleaned-up local paths or `gs://` URIs).

`app.py`:

```python
"""NiceGUI server. Mounts monitor, results, and tunnel on one process."""
from nicegui import ui

from .state import DashboardState
from . import launch, monitor, results, tunnel

ROUTES = ["/", "/results", "/tunnel"]

STATE = DashboardState()


def build(state: DashboardState) -> None:
    @ui.page("/")
    def _index():
        monitor.render(state)
        launch.render(state)

    @ui.page("/results")
    def _results():
        results.render(state)

    @ui.page("/results/{run_id}")
    def _detail(run_id: str):
        results.render_detail(state, run_id)

    @ui.page("/tunnel")
    def _tunnel():
        tunnel.render()


def main() -> None:
    from ..core.firestore import get_firestore_client  # noqa: F401

    build(STATE)
    try:
        STATE.attach_listener("ami_federated_llm_results", lambda: None)
    except Exception as exc:  # degrade gracefully: no credentials, no listener
        print(f"Firestore listener unavailable ({exc}); dashboard runs empty.")
    ui.run(host="0.0.0.0", port=8080, title="Research Monitor", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_webui_pages.py -v`
Expected: PASS (5 tests)

Then check it by hand: `python -m master_script.webui.app`, open `http://localhost:8080`, confirm the monitor renders and says the running set is unavailable with no credentials.

- [ ] **Step 5: Commit**

```bash
git add master_script/webui tests/test_webui_pages.py
git commit -m "feat(webui): add monitor, results, and detail pages"
```

---

### Task 16: Web UI — launch page

**Files:**
- Create: `master_script/webui/launch.py`
- Test: `tests/test_webui_launch.py`

**Interfaces:**
- Consumes: `runner.run_sweep`, `yaml_config.load_config_file`, `registry.ATTACKS`, `firestore.publish_monitor_state`.
- Produces: `launch.render(state)`, `launch.SweepWorker` with `.start(pairs)`, `.is_running`, `.cancel()`, `launch.publish_running(run_id, attack, config)`.

**This is where reuse becomes structural.** The worker calls `core.runner.run_sweep` — the exact function the CLI calls. No experiment logic lives here.

Because the app owns the runs, it publishes the run-state the doc calls optional (§1.1), which is what makes the live panel exact. Publish on run start; clear on completion.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_webui_launch.py
import time

from master_script.webui.launch import SweepWorker


def test_worker_calls_core_run_sweep_not_a_private_copy(monkeypatch):
    """The UI must reuse the CLI's code path."""
    called = {}

    import master_script.webui.launch as mod

    def _fake(pairs, **kw):
        called["pairs"] = list(pairs)
        return [{"run_id": "x", "status": "complete"}]

    monkeypatch.setattr(mod, "run_sweep", _fake)
    w = SweepWorker()
    w.start([("cfg", "spec")])
    for _ in range(100):
        if not w.is_running:
            break
        time.sleep(0.01)
    assert called["pairs"] == [("cfg", "spec")]


def test_worker_reports_not_running_before_start():
    assert SweepWorker().is_running is False


def test_worker_refuses_concurrent_sweeps(monkeypatch):
    import master_script.webui.launch as mod

    monkeypatch.setattr(mod, "run_sweep", lambda pairs, **kw: time.sleep(0.2) or [])
    w = SweepWorker()
    w.start([("a", "b")])
    try:
        assert w.start([("c", "d")]) is False
    finally:
        w.cancel()


def test_publish_running_writes_monitor_state(monkeypatch):
    import master_script.webui.launch as mod

    published = {}
    monkeypatch.setattr(mod, "publish_monitor_state", lambda s, **k: published.update(s) or True)
    mod.publish_running("abc", "zlib", {"model_id": "distilgpt2", "seed": 7})
    assert published["running"][0]["run_id"] == "abc"
    assert published["running"][0]["attack"] == "zlib"


def test_worker_records_error_without_crashing(monkeypatch):
    import master_script.webui.launch as mod

    def _boom(pairs, **kw):
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(mod, "run_sweep", _boom)
    w = SweepWorker()
    w.start([("a", "b")])
    for _ in range(100):
        if not w.is_running:
            break
        time.sleep(0.01)
    assert "sweep exploded" in w.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webui_launch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.webui.launch'`

- [ ] **Step 3: Write minimal implementation**

```python
# master_script/webui/launch.py
"""Configure and launch sweeps from the browser.

Calls core.runner.run_sweep -- the same function perform_experiments.py calls.
No experiment logic lives here, so the UI cannot drift from the CLI.
"""
import threading
import time
from typing import List, Optional

from nicegui import ui

from ..core.config import experiment_key
from ..core.firestore import publish_monitor_state
from ..core.registry import ATTACKS
from ..core.runner import run_sweep
from ..core.yaml_config import ConfigError, load_config_file
from ..paths import CONFIGS_DIR


def publish_running(run_id: str, attack: str, config: dict) -> bool:
    """Publish the optional run-state report (§1.1). Coarse and transient."""
    return publish_monitor_state({
        "running": [{
            "run_id": run_id,
            "attack": attack,
            "started_unix": int(time.time()),
            "config": {k: config.get(k) for k in ("model_id", "federated_rounds", "seed", "num_clients")},
        }]
    })


class SweepWorker:
    """Runs one sweep on a background thread. At most one at a time."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self.results: List[dict] = []
        self.error: str = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, pairs, **kwargs) -> bool:
        if self.is_running:
            return False
        self.error = ""
        self.results = []

        def _work():
            try:
                self.results = run_sweep(pairs, **kwargs)
            except Exception as exc:
                self.error = str(exc)

        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        """In-progress runs are never resumed (§1.2); this just drops the handle."""
        self._thread = None


WORKER = SweepWorker()


def render(state) -> None:
    """Launch panel: pick a config, pick attacks, start."""
    with ui.card().classes("w-full"):
        ui.label("Launch a sweep").classes("text-lg font-bold")
        config_files = sorted(p.name for p in CONFIGS_DIR.glob("*.yaml"))
        config_select = ui.select(config_files, value=config_files[0] if config_files else None,
                                  label="Config file")
        attack_select = ui.select(sorted(ATTACKS), multiple=True, label="Attacks (default: all in config)")
        firestore_switch = ui.switch("Persist to Firestore", value=True)
        status = ui.label("")

        def _start():
            try:
                pairs = load_config_file(
                    CONFIGS_DIR / config_select.value, only=attack_select.value or None
                )
            except ConfigError as exc:
                status.set_text(f"Config error: {exc}")
                return
            if not WORKER.start(pairs, use_firestore=firestore_switch.value):
                status.set_text("A sweep is already running.")
                return
            for cfg, spec in pairs[:1]:
                from dataclasses import asdict

                publish_running(experiment_key(cfg, spec), spec.name, asdict(cfg))
            status.set_text(f"Started {len(pairs)} run(s).")

        ui.button("Start sweep", on_click=_start)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_webui_launch.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/webui/launch.py tests/test_webui_launch.py
git commit -m "feat(webui): add launch page reusing core.runner"
```

---

### Task 17: Web UI — tunnel

**Files:**
- Create: `master_script/webui/tunnel.py`
- Test: `tests/test_tunnel.py`

**Interfaces:**
- Consumes: nothing from core.
- Produces: `tunnel.TunnelConfig(provider, api_key, code, port)`, `tunnel.TunnelManager` with `.start(cfg)`, `.stop()`, `.status -> dict`, `tunnel.render()`.

**Spec §4:** both providers behind one surface (provider, API key, code/token, target port). Surface the external URL, connection status, last connection time, and whether the URL is ephemeral or stable. Tunnel exposure is explicit and user-initiated (§4.4) — never auto-start.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tunnel.py
import pytest

from master_script.webui.tunnel import TunnelConfig, TunnelManager


def test_status_disconnected_before_start():
    assert TunnelManager().status["connected"] is False


def test_ngrok_url_without_reserved_domain_is_ephemeral():
    cfg = TunnelConfig(provider="ngrok", api_key="tok", code="", port=8080)
    assert cfg.is_ephemeral is True


def test_ngrok_url_with_reserved_domain_is_stable():
    cfg = TunnelConfig(provider="ngrok", api_key="tok", code="my.domain", port=8080)
    assert cfg.is_ephemeral is False


def test_cloudflare_named_tunnel_is_stable():
    cfg = TunnelConfig(provider="cloudflare", api_key="k", code="tunnel-token", port=8080)
    assert cfg.is_ephemeral is False


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="provider"):
        TunnelConfig(provider="dropbox", api_key="k", code="", port=8080)


def test_missing_api_key_rejected():
    with pytest.raises(ValueError, match="api_key"):
        TunnelConfig(provider="ngrok", api_key="", code="", port=8080)


def test_start_builds_expected_ngrok_command(monkeypatch):
    m = TunnelManager()
    captured = {}
    monkeypatch.setattr(m, "_spawn", lambda cmd, env: captured.setdefault("cmd", cmd))
    m.start(TunnelConfig(provider="ngrok", api_key="tok", code="", port=8080))
    assert "ngrok" in captured["cmd"][0]
    assert "8080" in " ".join(captured["cmd"])


def test_start_builds_expected_cloudflared_command(monkeypatch):
    m = TunnelManager()
    captured = {}
    monkeypatch.setattr(m, "_spawn", lambda cmd, env: captured.setdefault("cmd", cmd))
    m.start(TunnelConfig(provider="cloudflare", api_key="k", code="tok", port=8080))
    assert "cloudflared" in captured["cmd"][0]


def test_stop_marks_disconnected(monkeypatch):
    m = TunnelManager()
    monkeypatch.setattr(m, "_spawn", lambda cmd, env: None)
    m.start(TunnelConfig(provider="ngrok", api_key="tok", code="", port=8080))
    m.stop()
    assert m.status["connected"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tunnel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'master_script.webui.tunnel'`

- [ ] **Step 3: Write minimal implementation**

```python
# master_script/webui/tunnel.py
"""Outbound tunnel for external viewing (spec §4).

The VM has outbound internet but no inbound route; a tunnel agent dials out and
the provider proxies a public URL back down that connection. Exposure is always
explicit and user-initiated (§4.4) -- never auto-start.
"""
from dataclasses import dataclass
from typing import List, Optional
import re
import subprocess
import threading
import time

from nicegui import ui

PROVIDERS = ("ngrok", "cloudflare")
_URL_RE = re.compile(r"https://[^\s\"']+")


@dataclass
class TunnelConfig:
    provider: str
    api_key: str
    code: str
    port: int

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; expected one of {PROVIDERS}")
        if not self.api_key:
            raise ValueError("api_key is required")

    @property
    def is_ephemeral(self) -> bool:
        """ngrok without a reserved domain regenerates its URL each session."""
        return self.provider == "ngrok" and not self.code


class TunnelManager:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._cfg: Optional[TunnelConfig] = None
        self.url: str = ""
        self.last_connected_unix: Optional[int] = None

    def _build_command(self, cfg: TunnelConfig) -> List[str]:
        if cfg.provider == "ngrok":
            cmd = ["ngrok", "http", str(cfg.port), "--log", "stdout"]
            if cfg.code:
                cmd += ["--domain", cfg.code]
            return cmd
        cmd = ["cloudflared", "tunnel", "--no-autoupdate"]
        if cfg.code:
            cmd += ["run", "--token", cfg.code]
        else:
            cmd += ["--url", f"http://localhost:{cfg.port}"]
        return cmd

    def _spawn(self, cmd: List[str], env: dict):
        return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

    def start(self, cfg: TunnelConfig) -> None:
        import os

        self.stop()
        self._cfg = cfg
        env = {**os.environ}
        if cfg.provider == "ngrok":
            env["NGROK_AUTHTOKEN"] = cfg.api_key
        else:
            env["CLOUDFLARE_API_TOKEN"] = cfg.api_key
        self._proc = self._spawn(self._build_command(cfg), env)
        if self._proc is not None:
            threading.Thread(target=self._watch, daemon=True).start()
        self.last_connected_unix = int(time.time())

    def _watch(self) -> None:
        """Scrape the agent's stdout for the public URL."""
        for line in self._proc.stdout:
            match = _URL_RE.search(line)
            if match and not self.url:
                self.url = match.group(0)

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self.url = ""

    @property
    def status(self) -> dict:
        connected = self._proc is not None and self._proc.poll() is None
        return {
            "connected": connected,
            "url": self.url,
            "provider": self._cfg.provider if self._cfg else None,
            "port": self._cfg.port if self._cfg else None,
            "ephemeral": self._cfg.is_ephemeral if self._cfg else None,
            "last_connected_unix": self.last_connected_unix,
        }


MANAGER = TunnelManager()


def render() -> None:
    with ui.card().classes("w-full"):
        ui.label("External access tunnel").classes("text-lg font-bold")
        provider = ui.select(list(PROVIDERS), value="ngrok", label="Provider")
        api_key = ui.input("API key / authtoken", password=True)
        code = ui.input("Tunnel code / reserved domain (optional)")
        port = ui.number("Target port", value=8080, format="%d")
        banner = ui.label("").classes("text-sm")

        @ui.refreshable
        def status_panel():
            s = MANAGER.status
            if not s["connected"]:
                ui.label("Tunnel down — dashboard reachable on the VM/VPN only.")
                return
            ui.label("EXTERNAL ACCESS IS LIVE").classes("text-red-600 font-bold")
            ui.label(f"{s['provider']} -> localhost:{s['port']}")
            if s["url"]:
                ui.link(s["url"], s["url"], new_tab=True)
                ui.button("Copy", on_click=lambda: ui.clipboard.write(s["url"]))
            ui.label("Ephemeral URL — regenerated each session." if s["ephemeral"]
                     else "Stable URL — reserved domain / named tunnel.")

        def _start():
            try:
                MANAGER.start(TunnelConfig(provider.value, api_key.value, code.value, int(port.value)))
                banner.set_text("")
            except (ValueError, FileNotFoundError) as exc:
                banner.set_text(f"Could not start tunnel: {exc}")
            status_panel.refresh()

        def _stop():
            MANAGER.stop()
            status_panel.refresh()

        with ui.row():
            ui.button("Start tunnel", on_click=_start)
            ui.button("Stop tunnel", on_click=_stop)
        status_panel()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tunnel.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add master_script/webui/tunnel.py tests/test_tunnel.py
git commit -m "feat(webui): add cloudflare/ngrok tunnel control"
```

---

### Task 18: Documentation and full-suite verification

**Files:**
- Create: `master_script/README.md`
- Modify: `environment.yml`
- Test: full suite

**Interfaces:**
- Consumes: everything.
- Produces: no code.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs.py
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "master_script" / "README.md"


def test_readme_exists():
    assert README.exists()


def test_readme_documents_every_cli_flag():
    from master_script.perform_experiments import build_parser

    text = README.read_text()
    for action in build_parser()._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                assert opt in text, f"{opt} is undocumented in README.md"


def test_readme_explains_hash_preservation():
    text = README.read_text().lower()
    assert "run_id" in text and "hash" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs.py -v`
Expected: FAIL with `assert False` (README missing)

- [ ] **Step 3: Write minimal implementation**

Write `master_script/README.md` covering: what this is and which notebooks it replaces; the hash-preservation guarantee and why `tests/test_hash_equivalence.py` matters; install; **every** CLI flag with a description and example; the YAML config schema with the `defaults` vs `base`/`sweep` merge and validation rules; running the web UI (`python -m master_script.webui.app`); the tunnel setup; where logs, outputs, and charts land; and the known limits (no `epsilon`/`ldp_mechanism`; `amia` has no toy path; §2.5 GPU/log panel not implemented).

Add to `environment.yml`: `pyyaml`, `nicegui`, `matplotlib`, `firebase-admin`, `python-dotenv`, `pytest`.

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -v`
Expected: PASS, all tests, no failures.

Then verify the two entrypoints for real:

```bash
python -m master_script.perform_experiments --list-attacks         # 11 attacks
python -m master_script.perform_experiments --dry-run              # run_ids + cache status
python -m master_script.perform_experiments --config master_script/configs/smoke.yaml --no-firestore
ls master_script/outputs/Charts/                                   # charts landed
ls master_script/logs/                                             # session log landed
python -m master_script.webui.app                                  # dashboard on :8080
```

- [ ] **Step 5: Commit**

```bash
git add master_script/README.md environment.yml tests/test_docs.py
git commit -m "docs(master_script): add README and wire dependencies"
```

---

## Verification Checklist

Before declaring this done, confirm each of these by running the command and reading the output — not by assuming:

- [ ] `pytest tests/test_hash_equivalence.py -v` passes for all 11 attacks. **This is the whole ballgame.** If it fails, results have moved and the consolidation is wrong.
- [ ] `pytest tests/ -v` passes with no GPU, no model download, and no Firebase credentials.
- [ ] `python -m master_script.perform_experiments --config master_script/configs/smoke.yaml --no-firestore` reports 9/9 complete.
- [ ] Charts land in `master_script/outputs/Charts/` with a capital C.
- [ ] Logs land in `master_script/logs/`.
- [ ] Running the CLI from a different cwd (`cd /tmp && python -m master_script.perform_experiments ...`) still writes into `master_script/`, not `/tmp`.
- [ ] The dashboard starts and degrades gracefully with no credentials (says the running set is unavailable rather than guessing).
- [ ] `grep -rn "epsilon\|ldp_mechanism" master_script/core/attacks/` returns nothing.
