"""Shared runner helpers for split images, ground truth, prediction files and run finalisation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from hw2.eval.coco_eval import evaluate
from hw2.eval.preds import load_preds, validate_preds
from hw2.paths import data_root, prep_root, results_dir
from hw2.submission import default_test_csv, validate_submission, write_submission
from hw2.utils import env_info, get_logger, git_commit, save_json

EVAL_SPLITS: tuple[str, ...] = ("val", "val_testlike")
SPLIT_ROOTS: dict[str, Callable[[], Path]] = {"val": data_root, "val_testlike": prep_root}
GT_NAME = "annotations_coco.json"
SUBMISSION_NAME = "submission.csv.gz"

log = get_logger(__name__)


def preds_path(out_dir: str | Path, split: str) -> Path:
    """Return the predictions parquet path of a split inside a run folder."""
    return Path(out_dir) / f"preds_{split}.parquet"


def split_images(split: str) -> list[tuple[str, Path]]:
    """Return (file_name, absolute path) pairs for every image of val, val_testlike or test."""
    if split == "test":
        root = data_root().resolve()
        return [(name, root / name) for name in pd.read_csv(default_test_csv())["file_name"]]
    if split not in SPLIT_ROOTS:
        raise ValueError(f"unknown split {split!r}; choose from {[*SPLIT_ROOTS, 'test']}")
    root = SPLIT_ROOTS[split]().resolve()
    paths = sorted((root / split / "images").glob("*/*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no images under {root / split / 'images'}")
    return [(path.relative_to(root).as_posix(), path) for path in paths]


def gt_json(split: str) -> Path:
    """Return the COCO ground-truth json path of val or val_testlike."""
    if split not in SPLIT_ROOTS:
        raise ValueError(f"no ground truth for split {split!r}; choose from {list(SPLIT_ROOTS)}")
    return SPLIT_ROOTS[split]() / split / GT_NAME


def save_preds(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate a prediction table and write it as parquet, returning the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_preds(df).to_parquet(path, index=False)
    return path


def evaluate_splits(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Evaluate every existing preds_<split>.parquet of a run, skipping val_testlike without ground truth."""
    metrics: dict[str, dict[str, Any]] = {}
    for split in EVAL_SPLITS:
        preds_file = preds_path(out_dir, split)
        if not preds_file.exists():
            continue
        gt = gt_json(split)
        if not gt.exists() and split == "val_testlike":
            log.warning("skip %s: ground truth %s missing", split, gt)
            continue
        metrics[split] = evaluate(load_preds(preds_file), gt)
        save_json(metrics[split], out_dir / f"metrics_{split}.json")
    return metrics


def make_submission(out_dir: Path) -> Path | None:
    """Write and validate the submission from preds_test.parquet, or return None when it is missing."""
    preds_file = preds_path(out_dir, "test")
    if not preds_file.exists():
        log.warning("no %s, submission skipped", preds_file.name)
        return None
    path = write_submission(preds_file, default_test_csv(), out_dir / SUBMISSION_NAME)
    validate_submission(path, default_test_csv())
    return path


def _num(value: float | None, digits: int = 2) -> str:
    """Format an optional number with fixed decimals, or a dash when missing."""
    return "-" if value is None else f"{value:.{digits}f}"


def summary_lines(result: Mapping[str, Any], submission: Path | None, results_path: Path) -> list[str]:
    """Return at most 10 summary lines for a finalised run."""
    lines = [
        f"{result['exp_id']} ({result['exp_name']}) | train {_num(result['train_hours'])} h | "
        f"GPU {_num(result['gpu_hours'])} h | epochs {result['epochs_done']} | {_num(result['ms_per_image'], 1)} ms/img"
    ]
    for split, m in result["metrics"].items():
        ar_key = f"ar{m['max_dets']}"
        lines.append(f"{split:<13} mAP {_num(m['map'], 4)} | AP50 {_num(m['map50'], 4)} | {ar_key.upper()} {_num(m[ar_key], 4)}")
    lines.append(f"submission    {submission or 'skipped'}")
    lines.append(f"results       {results_path}")
    return lines


def finalize_run(exp_id: str, cfg: Mapping[str, Any], out_dir: str | Path, train_info: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate saved predictions, write and validate the submission, and save results/<exp_id>.json."""
    out_dir = Path(out_dir)
    metrics = evaluate_splits(out_dir)
    submission = make_submission(out_dir)
    train_hours = train_info.get("train_hours")
    result = {
        "exp_id": exp_id,
        "exp_name": cfg.get("exp_name", exp_id),
        "config": dict(cfg),
        "git_commit": git_commit(),
        "env_info": env_info(),
        "train_hours": train_hours,
        "gpu_hours": train_info.get("gpu_hours", train_hours),
        "epochs_done": train_info.get("epochs_done"),
        "metrics": metrics,
        "ms_per_image": train_info.get("ms_per_image"),
        "lb_public": None,
    }
    results_path = save_json(result, results_dir() / f"{exp_id}.json")
    for line in summary_lines(result, submission, results_path):
        log.info(line)
    return result
