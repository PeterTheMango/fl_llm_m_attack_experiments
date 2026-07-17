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
        "import hashlib, json, math, os, random, shutil, time, zlib\n"
    )
    exec(preamble + target, namespace)
    return namespace["ExperimentConfig"]
