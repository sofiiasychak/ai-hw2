"""Anchor analysis for the torchvision detectors: IoU k-means on train boxes and coverage of the real anchor grids."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import nn
from torchvision.ops import box_iou

from hw2.analysis import eda
from hw2.constants import SEED
from hw2.data import boxes as bx
from hw2.models.frcnn import ASPECT_RATIOS
from hw2.models.frcnn import DEFAULTS as FRCNN_DEFAULTS
from hw2.models.frcnn import build_frcnn
from hw2.models.retinanet import build_retinanet
from hw2.paths import repo_root, results_dir
from hw2.utils import get_logger, save_json

LOG = get_logger(__name__)
MIN_SIZE: int = FRCNN_DEFAULTS["min_size"]
MAX_SIZE: int = FRCNN_DEFAULTS["max_size"]
KMEANS_K = 9
KMEANS_ITERS = 300
RATIO_GROUPS = 3
RATIO_TOLERANCE = 1.25  # 25% off is clear
IOU_LEVELS: tuple[float, ...] = (0.5, 0.4)
ANCHOR_SETS: dict[str, list[int]] = {"default": [32, 64, 128, 256, 512], "16-256": [16, 32, 64, 128, 256]}
FAMILIES: tuple[str, ...] = ("frcnn", "retinanet")
IOU_CHUNK = 256
BUCKETS: tuple[str, ...] = (*eda.SIZE_BUCKETS, "all")
R1_CONFIG = Path("configs") / "retinanet" / "R1_retina_kmeans.yaml"
RATIO_COMMENT = "# ratios: k-means on train boxes"


def input_scale(width: Any, height: Any) -> np.ndarray:
    """Return the resize factor min(720 / short side, 1280 / long side) for each frame size."""
    width, height = np.asarray(width, np.float64), np.asarray(height, np.float64)
    return np.minimum(MIN_SIZE / np.minimum(width, height), MAX_SIZE / np.maximum(width, height))


def train_objects() -> pd.DataFrame:
    """Return the raw train object table with frame sizes and original-pixel boxes."""
    return eda.load_split("train").objects


def input_wh(objects: pd.DataFrame) -> np.ndarray:
    """Return box widths and heights rescaled to the model input."""
    scale = input_scale(objects["width"], objects["height"])
    return objects[["w", "h"]].to_numpy(np.float64) * scale[:, None]


def wh_iou(wh: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Return the IoU of centre-aligned boxes and anchors given as width-height pairs."""
    inter = np.minimum(wh[:, None, 0], anchors[None, :, 0]) * np.minimum(wh[:, None, 1], anchors[None, :, 1])
    union = wh.prod(axis=1)[:, None] + anchors.prod(axis=1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def kmeans_anchors(k: int = KMEANS_K, wh: np.ndarray | None = None, seed: int = SEED) -> tuple[np.ndarray, float]:
    """Cluster input-scale train box shapes with 1 - IoU distance and median centres, returning anchors by area and the mean best IoU."""
    wh = input_wh(train_objects()) if wh is None else np.asarray(wh, np.float64)
    rng = np.random.default_rng(seed)
    centres = wh[rng.choice(len(wh), k, replace=False)]
    assign = np.full(len(wh), -1)
    for _ in range(KMEANS_ITERS):
        new = wh_iou(wh, centres).argmax(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        centres = np.stack([np.median(wh[assign == j], axis=0) if (assign == j).any() else centres[j] for j in range(k)])
    centres = centres[np.argsort(centres.prod(axis=1))]
    return centres, float(wh_iou(wh, centres).max(axis=1).mean())


def kmeans_ratios(anchors: np.ndarray, groups: int = RATIO_GROUPS) -> list[float]:
    """Return torchvision h/w aspect ratios as geometric means of sorted anchor-ratio groups."""
    ratios = np.sort(anchors[:, 1] / anchors[:, 0])
    return [round(float(np.exp(np.log(part).mean())), 2) for part in np.array_split(ratios, groups)]


def ratios_differ(ratios: Sequence[float], default: Sequence[float] = ASPECT_RATIOS) -> bool:
    """Return True when any sorted ratio is more than 25% away from its default counterpart."""
    gap = np.abs(np.log(np.sort(ratios) / np.sort(default)))
    return bool((gap > np.log(RATIO_TOLERANCE)).any())


def write_ratios(ratios: Sequence[float], path: Path | None = None) -> Path:
    """Set model.anchor_ratios in the R1 RetinaNet config, keeping its other lines."""
    path = path or repo_root() / R1_CONFIG
    keep = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.strip().startswith(("anchor_ratios:", "# ratios"))]
    at = keep.index("model:")
    values = ", ".join(f"{r:g}" for r in ratios)
    keep[at:at + 1] = [RATIO_COMMENT, "model:", f"  anchor_ratios: [{values}]"]
    path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return path


def build_detector(family: str, sizes: Sequence[int]) -> nn.Module:
    """Build an untrained detector of one family with the given anchor sizes."""
    if family == "frcnn":
        return build_frcnn({"model": {"anchor_sizes": list(sizes)}}, pretrained=False).eval()
    return build_retinanet({"model": {"anchor_base_sizes": list(sizes)}}, pretrained=False).eval()


@torch.inference_mode()
def frame_anchors(model: nn.Module, width: int, height: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Return a frame size's anchors in input pixels from the model's own transform, backbone and generator."""
    images, _ = model.transform([torch.zeros(3, height, width)])
    features = list(model.backbone(images.tensors).values())
    generator = model.rpn.anchor_generator if hasattr(model, "rpn") else model.anchor_generator
    return generator(images, features)[0], tuple(images.image_sizes[0])


def best_anchor_iou(boxes: torch.Tensor, anchors: torch.Tensor) -> np.ndarray:
    """Return each corner box's highest IoU with any anchor."""
    parts = [box_iou(chunk, anchors).max(dim=1).values for chunk in boxes.split(IOU_CHUNK)]
    return torch.cat(parts).numpy() if parts else np.empty(0)


def model_best_iou(model: nn.Module, objects: pd.DataFrame) -> np.ndarray:
    """Return each object's best anchor IoU at the model's input scale, sharing anchors per frame size."""
    best = np.empty(len(objects))
    for (width, height), rows in objects.groupby(["width", "height"]).indices.items():
        anchors, (new_h, new_w) = frame_anchors(model, int(width), int(height))
        xyxy = bx.scale_xyxy(bx.xywh_to_xyxy(objects[["x", "y", "w", "h"]].to_numpy()[rows]), new_w / width, new_h / height)
        best[rows] = best_anchor_iou(torch.as_tensor(xyxy, dtype=torch.float32), anchors)
    return best


def sample_frames(objects: pd.DataFrame, n_frames: int, seed: int = SEED) -> pd.DataFrame:
    """Return the objects of n seeded random frames."""
    ids = np.sort(objects["image_id"].unique())
    pick = np.random.default_rng(seed).choice(ids, min(n_frames, len(ids)), replace=False)
    return objects[objects["image_id"].isin(pick)].reset_index(drop=True)


def coverage_rows(best: np.ndarray, buckets: pd.Series) -> list[dict[str, Any]]:
    """Return the box count and share of boxes at each IoU level per size bucket and overall."""
    rows = []
    for bucket in BUCKETS:
        values = best if bucket == "all" else best[(buckets == bucket).to_numpy()]
        shares = {f"iou>={level}": float((values >= level).mean()) if len(values) else None for level in IOU_LEVELS}
        rows.append({"bucket": bucket, "n": int(len(values)), **shares, "mean_best_iou": float(values.mean()) if len(values) else None})
    return rows


def anchor_coverage(n_frames: int = 500, objects: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the share of train boxes whose best anchor IoU reaches 0.5 and 0.4, by COCO size bucket, per family and anchor set."""
    sample = sample_frames(train_objects() if objects is None else objects, n_frames)
    buckets = eda.size_bucket(sample["side"])
    rows = []
    for family in FAMILIES:
        for name, sizes in ANCHOR_SETS.items():
            best = model_best_iou(build_detector(family, sizes), sample)
            rows += [{"model": family, "anchors": name, **row} for row in coverage_rows(best, buckets)]
            LOG.info("%s %s: %.3f of boxes at IoU >= 0.5", family, name, (best >= 0.5).mean())
    return pd.DataFrame(rows)


def plot_coverage(table: pd.DataFrame) -> Figure:
    """Draw grouped bars of anchor coverage per size bucket, one panel per IoU level."""
    fig, axes = plt.subplots(1, len(IOU_LEVELS), figsize=(11, 3.6), sharey=True)
    combos = list(table.groupby(["model", "anchors"], sort=False).groups)
    width = 0.8 / len(combos)
    x = np.arange(len(BUCKETS))
    for ax, level in zip(axes, IOU_LEVELS):
        for i, (model, anchors) in enumerate(combos):
            part = table[(table["model"] == model) & (table["anchors"] == anchors)].set_index("bucket").reindex(BUCKETS)
            ax.bar(x + (i - (len(combos) - 1) / 2) * width, part[f"iou>={level}"] * 100, width, label=f"{model} {anchors}")
        ax.set_xticks(x, BUCKETS)
        ax.set_title(f"best anchor IoU >= {level}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("% of train boxes")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Anchor coverage at model input (COCO size buckets on original pixels)")
    fig.tight_layout()
    return fig


def kmeans_summary(anchors: np.ndarray, mean_iou: float, n_boxes: int) -> dict[str, Any]:
    """Return the k-means anchors, their mean best IoU and derived aspect ratios."""
    ratios = kmeans_ratios(anchors)
    return {
        "k": len(anchors),
        "seed": SEED,
        "n_boxes": n_boxes,
        "anchors_wh": np.round(anchors, 1).tolist(),
        "mean_best_iou": mean_iou,
        "ratios_h_over_w": ratios,
        "default_ratios": list(ASPECT_RATIOS),
        "ratios_differ": ratios_differ(ratios),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the k-means size and coverage sample options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=KMEANS_K)
    parser.add_argument("--frames", type=int, default=500, help="train frames for coverage")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run k-means and coverage, write anchors.json and the figure, and update R1 ratios when they differ."""
    args = parse_args(argv)
    objects = train_objects()
    anchors, mean_iou = kmeans_anchors(args.k, input_wh(objects))
    kmeans = kmeans_summary(anchors, mean_iou, len(objects))
    LOG.info("k-means anchors %s, mean IoU %.3f, ratios %s", kmeans["anchors_wh"], mean_iou, kmeans["ratios_h_over_w"])
    table = anchor_coverage(args.frames, objects)
    eda.save_figure(plot_coverage(table), "anchor_coverage", prefix="")
    if kmeans["ratios_differ"]:
        LOG.info("wrote ratios to %s", write_ratios(kmeans["ratios_h_over_w"]))
    result = {"input": {"min_size": MIN_SIZE, "max_size": MAX_SIZE}, "kmeans": kmeans, "n_frames": args.frames, "coverage": table.to_dict("records")}
    save_json(result, results_dir() / "analysis" / "anchors.json")
    return result


if __name__ == "__main__":
    main()
