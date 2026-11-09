"""RetinaNet ResNet-50 FPN v2 from COCO weights with 9-label logits and configurable anchors, matcher, losses and outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from torch import nn
from torchvision.models.detection import RetinaNet, RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from hw2.constants import NUM_CLASSES
from hw2.data.torch_dataset import LABEL_OFFSET
from hw2.models.settings import check_choice, check_levels, model_settings
from hw2.utils import get_logger

LOG = get_logger(__name__)
NUM_LABELS = NUM_CLASSES + LABEL_OFFSET
FPN_LEVELS = 5
ANCHOR_SCALES = (1.0, 2 ** (1 / 3), 2 ** (2 / 3))
PRIOR_PROBABILITY = 0.01
LOGIT_STD = 0.01
REGRESSION_LOSSES = ("l1", "smooth_l1", "giou", "diou", "ciou")
DEFAULTS: dict[str, Any] = {
    "min_size": 720,
    "max_size": 1280,
    "anchor_base_sizes": [32, 64, 128, 256, 512],
    "anchor_ratios": [0.5, 1.0, 2.0],
    "fg_iou_thresh": 0.5,
    "bg_iou_thresh": 0.4,
    "regression_loss": "giou",
    "focal_gamma": None,
    "focal_alpha": None,
    "detections_per_img": 500,
    "score_thresh": 0.001,
    "topk_candidates": 1000,
    "nms_thresh": 0.5,
}
BUILD_ONLY = ("anchor_base_sizes", "anchor_ratios", "regression_loss", "focal_gamma", "focal_alpha")


def retinanet_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated RetinaNet settings from defaults and the config's model block."""
    settings = model_settings(DEFAULTS, cfg)
    check_choice(settings, "regression_loss", REGRESSION_LOSSES)
    check_levels(settings["anchor_base_sizes"], FPN_LEVELS, "anchor_base_sizes")
    if len(settings["anchor_ratios"]) != len(ANCHOR_SCALES):
        raise ValueError(f"anchor_ratios needs {len(ANCHOR_SCALES)} ratios to keep 9 anchors per location")
    return settings


def anchor_sizes(base_sizes: list[int]) -> tuple[tuple[int, ...], ...]:
    """Return three octave scales of each base anchor size, rounded down like torchvision."""
    return tuple(tuple(int(base * scale) for scale in ANCHOR_SCALES) for base in base_sizes)


def reset_class_logits(head: RetinaNetClassificationHead, num_classes: int) -> None:
    """Replace the classification logits with a fresh layer for num_classes using RetinaNet's prior bias."""
    old = head.cls_logits
    layer = nn.Conv2d(old.in_channels, head.num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    nn.init.normal_(layer.weight, std=LOGIT_STD)
    nn.init.constant_(layer.bias, -math.log((1 - PRIOR_PROBABILITY) / PRIOR_PROBABILITY))
    head.cls_logits = layer
    head.num_classes = num_classes


def build_retinanet(cfg: Mapping[str, Any], pretrained: bool = True) -> RetinaNet:
    """Build RetinaNet v2 from COCO weights with 9 anchors per location, new 9-label logits and the chosen box loss."""
    settings = retinanet_settings(cfg)
    kwargs = {key: value for key, value in settings.items() if key not in BUILD_ONLY}
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = retinanet_resnet50_fpn_v2(weights=weights, weights_backbone=None, **kwargs)
    sizes = anchor_sizes(settings["anchor_base_sizes"])
    model.anchor_generator = AnchorGenerator(sizes, (tuple(settings["anchor_ratios"]),) * len(sizes))
    reset_class_logits(model.head.classification_head, NUM_LABELS)
    regression = model.head.regression_head
    LOG.info("retinanet regression loss: torchvision %s, config %s", regression._loss_type, settings["regression_loss"])
    regression._loss_type = settings["regression_loss"]
    return model
