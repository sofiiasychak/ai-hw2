"""Opt-in torchvision loss patches, a GIoU box loss for Faster R-CNN and focal gamma and alpha for RetinaNet."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torchvision.models.detection import retinanet as tv_retinanet
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection._utils import BoxCoder, _box_loss

from hw2.utils import get_logger

LOG = get_logger(__name__)
ORIGINAL_FASTRCNN_LOSS = tv_roi_heads.fastrcnn_loss
ORIGINAL_FOCAL_LOSS = tv_retinanet.sigmoid_focal_loss
RCNN_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
REFERENCE_BOX = (-0.5, -0.5, 0.5, 0.5)  # unit box at origin


def giou_fastrcnn_loss(
    class_logits: Tensor,
    box_regression: Tensor,
    labels: Sequence[Tensor],
    regression_targets: Sequence[Tensor],
    weights: tuple[float, float, float, float] = RCNN_WEIGHTS,
) -> tuple[Tensor, Tensor]:
    """Return torchvision's classification loss and a GIoU loss of positive RoI deltas decoded on one shared box."""
    classification_loss, _ = ORIGINAL_FASTRCNN_LOSS(class_logits, box_regression, labels, regression_targets)
    all_labels = torch.cat(list(labels), dim=0)
    targets = torch.cat(list(regression_targets), dim=0)
    positive = torch.where(all_labels > 0)[0]
    deltas = box_regression.reshape(len(all_labels), -1, 4)[positive, all_labels[positive]].float()
    coder = BoxCoder(weights)
    reference = deltas.new_tensor(REFERENCE_BOX).expand(len(positive), 4)
    target_boxes = coder.decode_single(targets[positive].float(), reference)
    box_loss = _box_loss("giou", coder, reference, target_boxes, deltas)
    return classification_loss, box_loss / all_labels.numel()


def reset_patches() -> None:
    """Restore torchvision's original Faster R-CNN loss and RetinaNet focal loss."""
    tv_roi_heads.fastrcnn_loss = ORIGINAL_FASTRCNN_LOSS
    tv_retinanet.sigmoid_focal_loss = ORIGINAL_FOCAL_LOSS


def configure_patches(cfg: Mapping[str, Any], model: torch.nn.Module) -> list[str]:
    """Reset both patches, then apply the ones the config's family and model block ask for and return their names."""
    reset_patches()
    family, settings = cfg.get("family"), cfg.get("model") or {}
    active: list[str] = []
    if family == "frcnn" and settings.get("box_loss") == "giou":
        weights = tuple(model.roi_heads.box_coder.weights)
        tv_roi_heads.fastrcnn_loss = functools.partial(giou_fastrcnn_loss, weights=weights)
        active.append("frcnn_giou")
    focal = {key: settings.get(f"focal_{key}") for key in ("gamma", "alpha")}
    focal = {key: float(value) for key, value in focal.items() if value is not None}
    if family == "retinanet" and focal:
        tv_retinanet.sigmoid_focal_loss = functools.partial(ORIGINAL_FOCAL_LOSS, **focal)
        active.append("retinanet_focal " + " ".join(f"{key}={value}" for key, value in focal.items()))
    LOG.info("torchvision patches: %s", ", ".join(active) or "none")
    return active
