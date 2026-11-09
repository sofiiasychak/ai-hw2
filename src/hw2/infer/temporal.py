"""Link per-class detections into tracks across frames, rescore them by track confidence and fill one-frame gaps."""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from hw2.config import load_config
from hw2.constants import MAX_DETS, NUM_CLASSES
from hw2.eval.coco_eval import evaluate, load_gt
from hw2.eval.preds import PRED_COLUMNS, frame_key, load_preds, match_frames, top_k_per_image, validate_preds
from hw2.paths import out_root, repo_root, results_dir
from hw2.train.common import finalize_run, gt_json, preds_path, save_preds, split_images
from hw2.utils import Timer, get_logger, load_json, save_json

SPLITS: tuple[str, ...] = ("val", "val_testlike", "test")
TUNE_SPLIT = "val"
BOX_COLUMNS: list[str] = ["x", "y", "w", "h"]
NO_MATCH_COST = 2.0
GRID_KEYS: tuple[str, ...] = ("iou_thr", "alpha", "top_k", "min_len")

log = get_logger(__name__)


@dataclass(frozen=True)
class Params:
    """Linking IoU threshold, rescoring weight, top-k size, minimum track length and gap-fill decay."""

    iou_thr: float = 0.3
    alpha: float = 0.5
    top_k: int = 5
    min_len: int = 3
    decay: float = 0.5


@dataclass
class TrackSet:
    """Detections of one split sorted by clip, class and frame with track labels and stitched one-frame gaps."""

    preds: pd.DataFrame
    boxes: np.ndarray
    track: np.ndarray
    gaps: np.ndarray
    gap_names: np.ndarray
    gap_weight: np.ndarray


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the pairwise IoU between two arrays of xywh boxes."""
    ix = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    iy = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def hungarian(a: np.ndarray, b: np.ndarray, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Return matched row and column indices of a Hungarian matching on 1 - IoU, keeping pairs with IoU >= iou_thr."""
    iou = iou_matrix(a, b)
    ok = iou >= iou_thr
    rows, cols = np.flatnonzero(ok.any(axis=1)), np.flatnonzero(ok.any(axis=0))
    if not rows.size:
        return rows, cols
    sub = iou[np.ix_(rows, cols)]
    r, c = linear_sum_assignment(np.where(sub >= iou_thr, 1.0 - sub, NO_MATCH_COST))
    keep = sub[r, c] >= iou_thr
    return rows[r[keep]], cols[c[keep]]


def link(boxes: np.ndarray, key: np.ndarray, src: np.ndarray, dst: np.ndarray, step: int, iou_thr: float) -> np.ndarray:
    """Return (n, 2) index pairs matching src detections at key k to dst detections at key k + step."""
    src_idx, dst_idx = np.flatnonzero(src), np.flatnonzero(dst)
    s_keys, s_start, s_count = np.unique(key[src_idx], return_index=True, return_counts=True)
    d_keys, d_start, d_count = np.unique(key[dst_idx], return_index=True, return_counts=True)
    if not s_keys.size or not d_keys.size:
        return np.empty((0, 2), np.int64)
    loc = np.minimum(np.searchsorted(d_keys, s_keys + step), d_keys.size - 1)
    hit = np.flatnonzero(d_keys[loc] == s_keys + step)
    pairs = [np.empty((0, 2), np.int64)]
    for si, di in zip(hit, loc[hit]):
        a = src_idx[s_start[si] : s_start[si] + s_count[si]]
        b = dst_idx[d_start[di] : d_start[di] + d_count[di]]
        r, c = hungarian(boxes[a], boxes[b], iou_thr)
        pairs.append(np.stack([a[r], b[c]], axis=1))
    return np.concatenate(pairs)


def components(n: int, edges: np.ndarray) -> np.ndarray:
    """Return the connected-component label of each of n nodes joined by undirected edges."""
    if n == 0:
        return np.empty(0, np.int64)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def frame_table(names: Iterable[str]) -> pd.DataFrame:
    """Return unique frames sorted by clip and frame with a file name and the position inside their clip."""
    names = pd.unique(pd.Series(list(names), dtype=object))
    clips, frames = zip(*(frame_key(name) for name in names)) if len(names) else ((), ())
    table = pd.DataFrame({"clip": list(clips), "frame": np.asarray(frames, np.int64), "file_name": names})
    table = table.drop_duplicates(["clip", "frame"]).sort_values(["clip", "frame"], ignore_index=True)
    table["pos"] = table.groupby("clip").cumcount()
    return table


def build_tracks(preds: pd.DataFrame, iou_thr: float, frames: Iterable[str] | None = None) -> TrackSet:
    """Link detections of each clip and class across consecutive frames and stitch tracks over one-frame gaps."""
    preds = validate_preds(preds)[list(PRED_COLUMNS)]
    table = frame_table([*pd.unique(preds["file_name"]), *(frames or ())])
    index = {key: i for i, key in enumerate(zip(table["clip"], table["frame"]))}
    row = match_frames(preds["file_name"], index, "frame-table")
    pos = table["pos"].to_numpy()
    group = pd.factorize(table["clip"])[0][row] * NUM_CLASSES + preds["class_id"].to_numpy()
    order = np.lexsort((pos[row], group))
    preds, row, group = preds.iloc[order].reset_index(drop=True), row[order], group[order]
    key = group * (int(pos.max(initial=0)) + 3) + pos[row]
    boxes = preds[BOX_COLUMNS].to_numpy(np.float64)
    every = np.ones(len(preds), bool)
    links = link(boxes, key, every, every, 1, iou_thr)
    has_next, has_prev = np.zeros_like(every), np.zeros_like(every)
    has_next[links[:, 0]], has_prev[links[:, 1]] = True, True
    gaps = link(boxes, key, ~has_next, ~has_prev, 2, iou_thr)
    frame = table["frame"].to_numpy()
    first, last, mid = row[gaps[:, 0]], row[gaps[:, 1]], row[gaps[:, 0]] + 1
    return TrackSet(
        preds=preds,
        boxes=boxes,
        track=components(len(preds), np.concatenate([links, gaps])),
        gaps=gaps,
        gap_names=table["file_name"].to_numpy()[mid],
        gap_weight=(frame[mid] - frame[first]) / np.maximum(frame[last] - frame[first], 1),
    )


def track_stats(track: np.ndarray, score: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return each track's length and the mean of its top_k highest scores."""
    order = np.lexsort((-score, track))
    sorted_track = track[order]
    rank = np.arange(len(track)) - np.searchsorted(sorted_track, sorted_track)
    top = rank < top_k
    n_tracks = int(track.max(initial=-1)) + 1
    length = np.bincount(track, minlength=n_tracks)
    sums = np.bincount(sorted_track[top], weights=score[order][top], minlength=n_tracks)
    return length, sums / np.maximum(np.minimum(length, top_k), 1)


def rescore(ts: TrackSet, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """Return blended scores and a mask of detections whose track is at least min_len long."""
    score = ts.preds["score"].to_numpy()
    length, top = track_stats(ts.track, score, params.top_k)
    long = length[ts.track] >= params.min_len
    blended = params.alpha * score + (1.0 - params.alpha) * top[ts.track]
    return np.where(long, blended, score), long


def gap_boxes(ts: TrackSet, score: np.ndarray, long: np.ndarray, decay: float) -> pd.DataFrame:
    """Return interpolated boxes for stitched gaps of long tracks, scored as the lower neighbour times decay."""
    i, j = ts.gaps[:, 0], ts.gaps[:, 1]
    keep = long[i]
    i, j, w = i[keep], j[keep], ts.gap_weight[keep, None]
    boxes = ts.boxes[i] * (1.0 - w) + ts.boxes[j] * w
    frame = pd.DataFrame(boxes, columns=BOX_COLUMNS)
    frame.insert(0, "score", np.minimum(score[i], score[j]) * decay)
    frame.insert(0, "class_id", ts.preds["class_id"].to_numpy()[i])
    frame.insert(0, "file_name", ts.gap_names[keep])
    return frame


def apply_params(ts: TrackSet, params: Params) -> pd.DataFrame:
    """Return rescored detections plus gap-filled boxes, keeping the top 500 per image."""
    score, long = rescore(ts, params)
    parts = [ts.preds.assign(score=score)]
    if params.decay > 0 and len(ts.gaps):
        parts.append(gap_boxes(ts, score, long, params.decay))
    out = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    return top_k_per_image(out, MAX_DETS).reset_index(drop=True)


def temporal_refine(preds: pd.DataFrame, params: Params, frames: Iterable[str] | None = None) -> pd.DataFrame:
    """Track, rescore and gap-fill a prediction table with one parameter set."""
    return apply_params(build_tracks(preds, params.iou_thr, frames), params)


def rescore_grid(grid: Mapping[str, Sequence[Any]]) -> list[Params]:
    """Return every combination of the grid's linking and rescoring values with gap filling off."""
    values = [grid.get(name) or [getattr(Params(), name)] for name in GRID_KEYS]
    return [Params(**dict(zip(GRID_KEYS, combo)), decay=0.0) for combo in itertools.product(*values)]


def tune(
    preds: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame], float],
    grid: Mapping[str, Sequence[Any]],
    frames: Iterable[str] | None = None,
) -> tuple[Params, dict[Params, float]]:
    """Pick the best rescoring parameters on the grid, then the best gap-fill decay for them."""
    frames = list(frames or ())
    cache: dict[float, TrackSet] = {}
    scores: dict[Params, float] = {}

    def trial(params: Params) -> None:
        """Score one parameter set, reusing tracks per IoU threshold."""
        if params.iou_thr not in cache:
            cache[params.iou_thr] = build_tracks(preds, params.iou_thr, frames)
        scores[params] = score_fn(apply_params(cache[params.iou_thr], params))
        log.info("%s -> %.4f", params, scores[params])

    for params in rescore_grid(grid):
        trial(params)
    best = max(scores, key=scores.__getitem__)
    for decay in grid.get("decay") or []:
        if decay > 0:
            trial(replace(best, decay=float(decay)))
    return max(scores, key=scores.__getitem__), scores


def split_frames(split: str) -> list[str]:
    """Return every frame file name of a split, or an empty list when its images are unavailable."""
    try:
        return [name for name, _ in split_images(split)]
    except (FileNotFoundError, ValueError) as err:
        log.warning("no frame list for %s: %s", split, err)
        return []


def input_map(preds_dir: Path, split: str, preds: pd.DataFrame) -> float | None:
    """Return the input run's mAP on a split from its metrics json, evaluating when missing."""
    saved = preds_dir / f"metrics_{split}.json"
    if saved.exists():
        return load_json(saved)["map"]
    gt = gt_json(split)
    return evaluate(preds, gt, breakdown=False)["map"] if gt.exists() else None


def find_config(exp_id: str) -> Path:
    """Return the single configs/<family>/<exp_id>_<name>.yaml file of an experiment id."""
    matches = sorted((repo_root() / "configs").glob(f"*/{exp_id}_*.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one config for {exp_id}, found {[str(m) for m in matches]}")
    return matches[0]


def _fmt(value: float | None, sign: str = "") -> str:
    """Format an optional metric with four decimals, or a dash when missing."""
    return "-" if value is None else f"{value:{sign}.4f}"


def comparison(inputs: Mapping[str, float | None], metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return input and output mAP with their difference for each evaluated split."""
    table: dict[str, dict[str, Any]] = {}
    for split, before in inputs.items():
        after = metrics.get(split, {}).get("map")
        delta = None if before is None or after is None else after - before
        table[split] = {"input_map": before, "map": after, "delta": delta}
        log.info("%-13s mAP %s -> %s (%s)", split, _fmt(before), _fmt(after), _fmt(delta, "+"))
    return table


def run(preds_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Tune on val, refine every split's predictions with the best parameters and finalise the run."""
    exp_id = cfg["exp_id"]
    out_dir = out_root() / exp_id
    inputs = {s: load_preds(preds_path(preds_dir, s)) for s in SPLITS if preds_path(preds_dir, s).exists()}
    if TUNE_SPLIT not in inputs:
        raise FileNotFoundError(f"{preds_path(preds_dir, TUNE_SPLIT)} is needed for tuning")
    frames = {split: split_frames(split) for split in inputs}
    gt = load_gt(gt_json(TUNE_SPLIT))
    best, scores = tune(inputs[TUNE_SPLIT], lambda df: evaluate(df, gt, breakdown=False)["map"], cfg.get("grid") or {}, frames[TUNE_SPLIT])
    log.info("best %s", best)
    seconds: dict[str, float] = {}
    for split, preds in inputs.items():
        with Timer(f"temporal {split}", log) as timer:
            refined = temporal_refine(preds, best, frames[split])
        seconds[split] = timer.elapsed
        save_preds(refined, preds_path(out_dir, split))
    result = finalize_run(exp_id, cfg, out_dir, {"train_hours": 0.0, "gpu_hours": 0.0})
    before = {split: input_map(preds_dir, split, inputs[split]) for split in inputs if split != "test"}
    result["temporal"] = {
        "input_dir": str(preds_dir),
        "best": asdict(best),
        "grid": [{**asdict(params), "map": value} for params, value in scores.items()],
        "comparison": comparison(before, result["metrics"]),
        "seconds": seconds,
    }
    save_json(result, results_dir() / f"{exp_id}.json")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the input run folder, experiment id and config overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds-dir", required=True, type=Path, help="run folder with preds_<split>.parquet")
    parser.add_argument("--exp", required=True, help="experiment id, e.g. T1")
    parser.add_argument("--set", nargs="*", default=[], metavar="a.b=value", help="config overrides")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run temporal refinement for one experiment from the command line."""
    args = parse_args(argv)
    cfg = load_config(find_config(args.exp), args.set)
    return run(args.preds_dir, cfg)


if __name__ == "__main__":
    main()
