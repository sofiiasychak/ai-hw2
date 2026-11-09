"""Faster R-CNN ResNet-50 FPN v2 from COCO weights with a 9-label predictor and configurable anchors, RPN, RoI and outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from hw2.constants import NUM_CLASSES
from hw2.data.torch_dataset import LABEL_OFFSET
from hw2.models.settings import check_choice, check_levels, model_settings

NUM_LABELS = NUM_CLASSES + LABEL_OFFSET
FPN_LEVELS = 5
ASPECT_RATIOS = (0.5, 1.0, 2.0)
BOX_LOSSES = ("smooth_l1", "giou")
DEFAULTS: dict[str, Any] = {
    "min_size": 720,
    "max_size": 1280,
    "anchor_sizes": [32, 64, 128, 256, 512],
    "rpn_pre_nms_top_n_train": 2000,
    "rpn_post_nms_top_n_train": 2000,
    "rpn_pre_nms_top_n_test": 1000,
    "rpn_post_nms_top_n_test": 1000,
    "box_batch_size_per_image": 512,
    "box_positive_fraction": 0.25,
    "box_loss": "smooth_l1",
    "box_detections_per_img": 500,
    "box_score_thresh": 0.001,
    "box_nms_thresh": 0.5,
}
BUILD_ONLY = ("anchor_sizes", "box_loss")  # not FasterRCNN kwargs


def frcnn_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated Faster R-CNN settings from defaults and the config's model block."""
    settings = model_settings(DEFAULTS, cfg)
    check_choice(settings, "box_loss", BOX_LOSSES)
    check_levels(settings["anchor_sizes"], FPN_LEVELS, "anchor_sizes")
    return settings


def build_frcnn(cfg: Mapping[str, Any], pretrained: bool = True) -> FasterRCNN:
    """Build Faster R-CNN v2 from COCO weights with one anchor size per FPN level and a new 9-label predictor."""
    settings = frcnn_settings(cfg)
    kwargs = {key: value for key, value in settings.items() if key not in BUILD_ONLY}
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None, **kwargs)
    sizes = tuple((int(size),) for size in settings["anchor_sizes"])
    model.rpn.anchor_generator = AnchorGenerator(sizes, (ASPECT_RATIOS,) * len(sizes))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_LABELS)
    return model
