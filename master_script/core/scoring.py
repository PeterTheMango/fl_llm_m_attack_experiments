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
