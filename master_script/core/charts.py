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
