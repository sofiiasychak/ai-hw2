"""Tables and training-curve plots of committed run results and saved analyses for the notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from hw2.paths import results_dir
from hw2.utils import load_json

LB_FILE = "lb_scores.csv"
ANALYSIS_DIR = "analysis"
METRICS: dict[str, str] = {
    "mAP": "map",
    "AP50": "map50",
    "AP75": "map75",
    "APs": "map_small",
    "APm": "map_medium",
    "APl": "map_large",
    "AR500": "ar500",
}
TESTLIKE: dict[str, str] = {"testlike mAP": "map", "testlike APs": "map_small"}


def load_results(prefixes: Sequence[str]) -> list[dict[str, Any]]:
    """Return every results/<exp_id>.json whose id starts with one of the prefixes, sorted by id."""
    paths = sorted(p for p in results_dir().glob("*.json") if p.stem.startswith(tuple(prefixes)))
    return [load_json(path) for path in paths]


def lb_scores() -> dict[str, float]:
    """Return public leaderboard scores by experiment id from results/lb_scores.csv."""
    path = results_dir() / LB_FILE
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return dict(zip(table["exp_id"].astype(str), table["lb_public"]))


def result_row(result: dict[str, Any], lb: dict[str, float]) -> dict[str, Any]:
    """Return one table row of val and val_testlike metrics, cost and leaderboard score."""
    val = result["metrics"].get("val", {})
    testlike = result["metrics"].get("val_testlike", {})
    return {
        "exp": result["exp_id"],
        "name": result["exp_name"],
        **{label: val.get(key) for label, key in METRICS.items()},
        **{label: testlike.get(key) for label, key in TESTLIKE.items()},
        "train h": result.get("train_hours"),
        "epochs": result.get("epochs_done"),
        "ms/img": result.get("ms_per_image"),
        "LB": result.get("lb_public") or lb.get(result["exp_id"]),
    }


def results_table(prefixes: Sequence[str] = ("F", "R")) -> pd.DataFrame:
    """Return one row per run with val metrics, val_testlike mAP, training cost, speed and leaderboard score."""
    lb = lb_scores()
    return pd.DataFrame([result_row(r, lb) for r in load_results(prefixes)]).set_index("exp")


def plot_curves(exp_ids: Sequence[str]) -> Figure:
    """Plot val_monitor mAP and training loss against hours for torchvision runs."""
    fig, (ax_map, ax_loss) = plt.subplots(1, 2, figsize=(10, 3.2))
    for exp_id in exp_ids:
        curve = pd.read_csv(results_dir() / "curves" / f"{exp_id}.csv")
        ax_map.plot(curve["hours"], curve["map"], marker="o", ms=3, label=exp_id)
        ax_loss.plot(curve["hours"], curve["loss"], marker="o", ms=3, label=exp_id)
    ax_map.set(xlabel="training hours", ylabel="val_monitor mAP")
    ax_loss.set(xlabel="training hours", ylabel="train loss")
    for ax in (ax_map, ax_loss):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def load_analysis(name: str) -> dict[str, Any]:
    """Return a saved results/analysis/<name>.json."""
    return load_json(results_dir() / ANALYSIS_DIR / f"{name}.json")


def coverage_table(family: str) -> pd.DataFrame:
    """Return anchor coverage percentages of one detector family by anchor set and size bucket."""
    table = pd.DataFrame(load_analysis("anchors")["coverage"])
    table = table[table["model"] == family]
    wide = table.pivot(index="anchors", columns="bucket", values=["iou>=0.5", "iou>=0.4"]) * 100
    return wide.reindex(columns=["small", "medium", "large", "all"], level=1)
