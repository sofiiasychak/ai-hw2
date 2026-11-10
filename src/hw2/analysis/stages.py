"""Recall of Faster R-CNN RPN proposals versus its final detections on val images, by COCO size bucket."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import nn
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops import box_iou

from hw2.analysis import eda
from hw2.config import find_config, load_config
from hw2.constants import SEED
from hw2.data import boxes as bx
from hw2.data import coco_utils as cu
from hw2.data.torch_dataset import LABEL_OFFSET, read_rgb
from hw2.paths import out_root, results_dir
from hw2.train.common import gt_json
from hw2.train.run_torchvision import build_model, load_weights, to_float, torch_device
from hw2.utils import get_logger, save_json

LOG = get_logger(__name__)
IOU_THR = 0.5
DEFAULT_EXP = "F0"
STAGES: tuple[str, ...] = ("rpn proposals", "final, any class", "final, same class")
BUCKETS: tuple[str, ...] = (*eda.SIZE_BUCKETS, "all")

Frame = tuple[Path, cu.FrameBoxes]


def load_frcnn(weights: Path, cfg: Mapping[str, Any], device: torch.device) -> nn.Module:
    """Build the configured Faster R-CNN, load trained weights and return it in eval mode."""
    if cfg.get("family") != "frcnn":
        raise ValueError(f"stage recall needs a frcnn config, got family {cfg.get('family')!r}")
    model = build_model(cfg, pretrained=False)
    load_weights(model, weights)
    return model.to(device).eval()


def val_frames(n_images: int, seed: int = SEED) -> list[Frame]:
    """Return image paths and boxes of n seeded random raw val frames."""
    gt = gt_json("val")
    coco = cu.load_coco(gt)
    by_image = cu.annotations_by_image(coco["annotations"])
    images = sorted(coco["images"], key=lambda image: image["id"])
    pick = np.sort(np.random.default_rng(seed).choice(len(images), min(n_images, len(images)), replace=False))
    return [(gt.parent / images[i]["file_name"], cu.split_annotations(by_image.get(images[i]["id"], []))) for i in pick]


@torch.inference_mode()
def predict_stages(model: nn.Module, image: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return one image's RPN proposals in original pixels and its final detections."""
    captured: dict[str, Any] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: tuple[Any, ...]) -> None:
        """Keep the proposals and resized size."""
        captured.update(proposals=output[0][0], size=inputs[0].image_sizes[0])

    handle = model.rpn.register_forward_hook(hook)
    try:
        detections = model([to_float(image, device)])[0]
    finally:
        handle.remove()
    proposals = resize_boxes(captured["proposals"], list(captured["size"]), list(image.shape[-2:]))
    return proposals.cpu(), {key: value.cpu() for key, value in detections.items()}


def best_iou(gt: torch.Tensor, boxes: torch.Tensor, same: torch.Tensor | None = None) -> np.ndarray:
    """Return each ground-truth box's best IoU over boxes, optionally only where a same-class mask holds."""
    if not len(boxes) or not len(gt):
        return np.zeros(len(gt))
    iou = box_iou(gt, boxes)
    return (iou * same if same is not None else iou).max(dim=1).values.numpy()


def frame_ious(model: nn.Module, frame: Frame, device: torch.device) -> pd.DataFrame:
    """Return per ground-truth box its size bucket and best IoU with proposals and with final detections."""
    path, boxes = frame
    proposals, dets = predict_stages(model, read_rgb(path), device)
    gt = torch.as_tensor(bx.xywh_to_xyxy(boxes.boxes), dtype=torch.float32).reshape(-1, 4)
    same = torch.as_tensor(boxes.classes)[:, None] == (dets["labels"] - LABEL_OFFSET)[None, :]
    sides = pd.Series(np.sqrt(boxes.boxes[:, 2] * boxes.boxes[:, 3]))
    return pd.DataFrame({
        "bucket": eda.size_bucket(sides).to_numpy(),
        STAGES[0]: best_iou(gt, proposals),
        STAGES[1]: best_iou(gt, dets["boxes"]),
        STAGES[2]: best_iou(gt, dets["boxes"], same.float()),
    })


def recall_table(ious: pd.DataFrame, thr: float = IOU_THR) -> pd.DataFrame:
    """Return recall in percent at an IoU threshold per stage and size bucket, plus ground-truth counts."""
    hits = ious[list(STAGES)].ge(thr).assign(bucket=ious["bucket"])
    table = hits.groupby("bucket")[list(STAGES)].mean().T * 100
    table["all"] = hits[list(STAGES)].mean() * 100
    table = table.reindex(columns=list(BUCKETS))
    counts = ious["bucket"].value_counts().reindex(list(eda.SIZE_BUCKETS), fill_value=0)
    table.loc["n ground truth"] = [*counts.tolist(), len(ious)]
    return table


def rpn_vs_final_recall(
    weights: Path, n_images: int = 100, cfg: Mapping[str, Any] | None = None, device: str | None = None
) -> pd.DataFrame:
    """Run Faster R-CNN on random val frames and return recall at IoU 0.5 of its 1000 proposals and its final detections."""
    cfg = cfg or load_config(find_config(DEFAULT_EXP))
    dev = torch_device(device)
    model = load_frcnn(weights, cfg, dev)
    frames = val_frames(n_images)
    parts = []
    for i, frame in enumerate(frames, start=1):
        parts.append(frame_ious(model, frame, dev))
        if i % 20 == 0:
            LOG.info("%d/%d frames", i, len(frames))
    return recall_table(pd.concat(parts, ignore_index=True))


def plot_recall(table: pd.DataFrame, title: str) -> Figure:
    """Draw grouped bars of recall per size bucket for each stage."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(BUCKETS))
    width = 0.8 / len(STAGES)
    for i, stage in enumerate(STAGES):
        bars = ax.bar(x + (i - 1) * width, table.loc[stage, list(BUCKETS)], width, label=stage)
        ax.bar_label(bars, fmt="%.0f", fontsize=7)
    counts = table.loc["n ground truth", list(BUCKETS)].astype(int)
    ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(BUCKETS, counts)])
    ax.set_ylabel(f"recall at IoU {IOU_THR} (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse weights, experiment, image count and device options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default=DEFAULT_EXP, help="frcnn experiment id for config and default weights")
    parser.add_argument("--weights", type=Path, default=None, help="default <OUT_ROOT>/<exp>/last.pt")
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """Compute the stage recall table, print it and save its json and figure."""
    args = parse_args(argv)
    weights = args.weights or out_root() / args.exp / "last.pt"
    table = rpn_vs_final_recall(weights, args.images, load_config(find_config(args.exp)), args.device)
    print(table.round(1).to_string())
    title = f"{args.exp}: RPN (top 1000) vs final recall, {args.images} val frames"
    eda.save_figure(plot_recall(table, title), "frcnn_rpn_recall", prefix="")
    save_json({"exp_id": args.exp, "n_images": args.images, "iou": IOU_THR, "recall": table.to_dict()}, results_dir() / "analysis" / "frcnn_stages.json")
    return table


if __name__ == "__main__":
    main()
