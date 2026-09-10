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
    """Import only the canonical ExperimentConfig alias declared by the notebook."""
    import ast
    import importlib
    cells = _code_cells(ADAPTATIONS / NOTEBOOKS[attack])
    matches = []
    for cell in cells:
        for node in ast.parse(cell).body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.asname == "ExperimentConfig":
                        matches.append((node.module, alias.name))
    assert len(matches) == 1
    module, name = matches[0]
    assert module == f"master_script.core.attacks.{attack}"
    return getattr(importlib.import_module(module), name)
