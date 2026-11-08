"""COCO bbox evaluation at 500 boxes per image with ignore regions and per-class, per-clip and density breakdowns."""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hw2.constants import CLASS_NAMES, MAX_DETS
from hw2.eval.preds import FrameKey, frame_key, load_preds, match_frames, top_k_per_image, validate_preds
from hw2.utils import load_json, save_json

BACKENDS: tuple[str, ...] = ("faster_coco_eval", "pycocotools")
IOU50, IOU75 = 0, 5  # COCO iouThrs 0.50:0.05:0.95
AREAS: tuple[str, ...] = ("all", "small", "medium", "large")
DENSITY_BUCKETS: tuple[tuple[str, int, float], ...] = (("<30", 0, 29), ("30-80", 30, 80), (">80", 81, np.inf))
GT_ANN_FIELDS: tuple[str, ...] = ("id", "image_id", "category_id", "bbox", "area", "iscrowd")
GT_KEY_FIELDS: tuple[tuple[str, str], ...] = (("video", "frame_id"), ("clip", "frame"))
TABLE_ROWS = 11

GroundTruth = str | Path | dict[str, Any]


def coco_backend(name: str | None = None) -> tuple[str, type, type]:
    """Return the backend name with its COCO and COCOeval classes, preferring faster_coco_eval."""
    if name not in (None, *BACKENDS):
        raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
    if name != "pycocotools":
        try:
            from faster_coco_eval import COCO, COCOeval_faster
        except ImportError:
            if name:
                raise
        else:
            return "faster_coco_eval", COCO, COCOeval_faster
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    return "pycocotools", COCO, COCOeval


def load_gt(gt_json: GroundTruth) -> dict[str, Any]:
    """Return a fresh COCO ground-truth dict holding only the fields evaluation needs."""
    data = gt_json if isinstance(gt_json, dict) else load_json(gt_json)
    anns = [{key: ann[key] for key in GT_ANN_FIELDS if key in ann} for ann in data["annotations"]]
    for ann in anns:
        ann.setdefault("area", ann["bbox"][2] * ann["bbox"][3])
        ann.setdefault("iscrowd", 0)
    images = [dict(img) for img in data["images"]]
    return {"images": images, "categories": [dict(c) for c in data["categories"]], "annotations": anns}


def gt_frame_key(image: dict[str, Any]) -> FrameKey:
    """Return a ground-truth image's (clip, frame) key from video/frame_id or clip/frame, else its file name."""
    for clip_field, frame_field in GT_KEY_FIELDS:
        if clip_field in image and frame_field in image:
            return str(image[clip_field]), int(image[frame_field])
    return frame_key(image["file_name"])


def gt_frame_index(images: Iterable[dict[str, Any]]) -> dict[FrameKey, int]:
    """Map each ground-truth image's (clip, frame) key to its image id, rejecting duplicate frames."""
    index: dict[FrameKey, int] = {}
    for image in images:
        key = gt_frame_key(image)
        if key in index:
            raise ValueError(f"ground truth holds frame {key} twice")
        index[key] = int(image["id"])
    return index


def density_bucket(n_boxes: int) -> str:
    """Return the density bucket label for an image with n_boxes valid ground-truth boxes."""
    return next(label for label, low, high in DENSITY_BUCKETS if low <= n_boxes <= high)


def image_groups(gt: dict[str, Any]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Group ground-truth image ids by clip and by density bucket of valid boxes."""
    valid = Counter(ann["image_id"] for ann in gt["annotations"] if not ann.get("iscrowd", 0))
    clips: dict[str, list[int]] = defaultdict(list)
    density: dict[str, list[int]] = {label: [] for label, _, _ in DENSITY_BUCKETS}
    for image in gt["images"]:
        clips[gt_frame_key(image)[0]].append(int(image["id"]))
        density[density_bucket(valid[image["id"]])].append(int(image["id"]))
    return dict(sorted(clips.items())), density


def det_dataset(dets: pd.DataFrame, gt: dict[str, Any]) -> dict[str, Any]:
    """Build a COCO results dataset from matched detections, shifting class ids to category ids."""
    boxes = dets[["x", "y", "w", "h"]].to_numpy(np.float64).tolist()
    columns = zip(dets["image_id"].tolist(), dets["class_id"].tolist(), dets["score"].tolist(), boxes)
    anns = [
        {
            "id": i,
            "image_id": img,
            "category_id": cls + 1,
            "bbox": box,
            "area": box[2] * box[3],
            "score": score,
            "iscrowd": 0,
        }
        for i, (img, cls, score, box) in enumerate(columns, start=1)
    ]
    return {"images": gt["images"], "categories": gt["categories"], "annotations": anns}


def build_coco(coco_cls: type, dataset: dict[str, Any]) -> Any:
    """Return a COCO index object built from an in-memory dataset dict."""
    coco = coco_cls()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def coco_max_dets(max_dets: int) -> list[int]:
    """Return the COCOeval maxDets list, [1, 100, 500] by default."""
    return sorted({1, 100, max_dets})


def run_cocoeval(eval_cls: type, coco_gt: Any, coco_dt: Any, img_ids: Sequence[int], max_dets: int) -> dict[str, Any]:
    """Run COCOeval bbox on the given images and return precision, recall, category ids and maxDets index."""
    evaluator = eval_cls(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(img_ids)
    evaluator.params.maxDets = coco_max_dets(max_dets)
    evaluator.evaluate()
    evaluator.accumulate()
    return {
        "precision": np.asarray(evaluator.eval["precision"]),
        "recall": np.asarray(evaluator.eval["recall"]),
        "cat_ids": [int(c) for c in evaluator.params.catIds],
        "m": list(evaluator.params.maxDets).index(max_dets),
    }


def mean_valid(values: np.ndarray) -> float | None:
    """Return the mean of entries above -1, or None when no entry is valid."""
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def summarize(result: dict[str, Any], max_dets: int) -> dict[str, Any]:
    """Read overall, area and per-class metrics from precision and recall at the max_dets index."""
    precision = result["precision"][..., result["m"]]
    recall = result["recall"][..., result["m"]]
    metrics: dict[str, Any] = {
        "map": mean_valid(precision[:, :, :, 0]),
        "map50": mean_valid(precision[IOU50, :, :, 0]),
        "map75": mean_valid(precision[IOU75, :, :, 0]),
    }
    for a, area in enumerate(AREAS[1:], start=1):
        metrics[f"map_{area}"] = mean_valid(precision[:, :, :, a])
    metrics[f"ar{max_dets}"] = mean_valid(recall[:, :, 0])
    names = [CLASS_NAMES[c - 1] for c in result["cat_ids"]]
    metrics["per_class"] = {name: mean_valid(precision[:, :, k, 0]) for k, name in enumerate(names)}
    return metrics


def subset_map(evaluate_ids: Any, groups: dict[str, list[int]]) -> dict[str, float | None]:
    """Return mAP for each named group of image ids, None for empty groups."""
    return {name: (evaluate_ids(ids)["map"] if ids else None) for name, ids in groups.items()}


def evaluate(
    preds: pd.DataFrame | str | Path,
    gt_json: GroundTruth,
    max_dets: int = MAX_DETS,
    backend: str | None = None,
    breakdown: bool = True,
) -> dict[str, Any]:
    """Evaluate predictions against COCO ground truth at max_dets boxes per image and return all metrics."""
    gt = load_gt(gt_json)
    preds = validate_preds(preds) if isinstance(preds, pd.DataFrame) else load_preds(preds)
    image_ids = match_frames(preds["file_name"], gt_frame_index(gt["images"]), "ground-truth")
    dets = top_k_per_image(preds.assign(image_id=image_ids), max_dets, by="image_id")
    name, coco_cls, eval_cls = coco_backend(backend)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = build_coco(coco_cls, gt)
        coco_dt = build_coco(coco_cls, det_dataset(dets, gt))

        def evaluate_ids(ids: Sequence[int]) -> dict[str, Any]:
            """Summarize COCOeval on one subset of images."""
            return summarize(run_cocoeval(eval_cls, coco_gt, coco_dt, ids, max_dets), max_dets)

        metrics = evaluate_ids([int(img["id"]) for img in gt["images"]])
        if breakdown:
            clips, density = image_groups(gt)
            metrics["per_clip"] = subset_map(evaluate_ids, clips)
            metrics["per_density"] = subset_map(evaluate_ids, density)
            metrics["density_images"] = {label: len(ids) for label, ids in density.items()}
    metrics.update(backend=name, max_dets=max_dets, n_images=len(gt["images"]), n_preds=len(dets))
    return metrics


def _fmt(value: float | None) -> str:
    """Format a metric with four decimals, or a dash when missing."""
    return "-" if value is None else f"{value:.4f}"


def _column(items: list[tuple[str, float | None]], rows: int) -> list[str]:
    """Format label-value pairs as fixed-width cells, folding overflow into one last cell."""
    if len(items) > rows:
        items = [*items[: rows - 1], (f"... +{len(items) - rows + 1}", None)]
    width = max([len(label) for label, _ in items] + [6])
    cells = [f"{label:<{width}} {_fmt(value):>6}" for label, value in items]
    return cells + [" " * (width + 7)] * (rows - len(cells))


def format_table(metrics: dict[str, Any]) -> str:
    """Return a table of at most 15 lines with overall, per-class and per-clip metrics."""
    ar_key = f"ar{metrics['max_dets']}"
    overall = [(key, metrics[key]) for key in ("map", "map50", "map75", "map_small", "map_medium", "map_large")]
    overall.append((ar_key, metrics[ar_key]))
    overall += [(f"density {label}", value) for label, value in metrics.get("per_density", {}).items()]
    per_class = list(metrics["per_class"].items())
    per_clip = list(metrics.get("per_clip", {}).items())
    columns = [_column(items, TABLE_ROWS) for items in (overall, per_class, per_clip) if items]
    title = f"{metrics['backend']} | {metrics['n_images']} images | {metrics['n_preds']} boxes | max {metrics['max_dets']}/image"
    header = " | ".join(f"{name:<{len(col[0])}}" for name, col in zip(("metric", "class AP", "clip mAP"), columns))
    rows = [[col[i] for col in columns] for i in range(TABLE_ROWS)]
    body = [" | ".join(cells).rstrip() for cells in rows if any(cell.strip() for cell in cells)]
    return "\n".join([title, header.rstrip(), *body])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for evaluating one predictions parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True, type=Path, help="predictions parquet")
    parser.add_argument("--gt", required=True, type=Path, help="COCO ground-truth json")
    parser.add_argument("--out", type=Path, help="metrics json to write")
    parser.add_argument("--max-dets", type=int, default=MAX_DETS)
    parser.add_argument("--backend", choices=BACKENDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Evaluate a predictions parquet, print the metrics table and optionally save the metrics json."""
    args = parse_args(argv)
    metrics = evaluate(args.preds, args.gt, args.max_dets, args.backend)
    if args.out:
        save_json(metrics, args.out)
    print(format_table(metrics))
    return metrics


if __name__ == "__main__":
    main()
