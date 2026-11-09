"""Tests for the torchvision detector builders, loss patches, dataset and training schedule helpers."""

import json
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torchvision.models.detection import retinanet as tv_retinanet
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection._utils import BoxCoder
from torchvision.ops import generalized_box_iou_loss

from hw2.data.torch_dataset import CocoDetectionDataset
from hw2.models import patches_tv as pt
from hw2.models.frcnn import build_frcnn
from hw2.models.retinanet import build_retinanet
from hw2.train import run_torchvision as rt

FRCNN = {"family": "frcnn", "model": {"anchor_sizes": [16, 32, 64, 128, 256]}}
RETINA = {"family": "retinanet", "model": {"anchor_base_sizes": [16, 32, 64, 128, 256]}}


@pytest.fixture(scope="module")
def frcnn() -> torch.nn.Module:
    """Faster R-CNN without downloaded weights."""
    return build_frcnn(FRCNN, pretrained=False)


@pytest.fixture(scope="module")
def retina() -> torch.nn.Module:
    """RetinaNet without downloaded weights."""
    return build_retinanet(RETINA, pretrained=False)


@pytest.fixture(autouse=True)
def clean_patches() -> Iterator[None]:
    """Restore torchvision's original losses after every test."""
    yield
    pt.reset_patches()


def test_frcnn_classes_anchors_and_outputs(frcnn: torch.nn.Module) -> None:
    """Faster R-CNN predicts 8 classes plus background with one anchor size and 3 ratios per level."""
    predictor = frcnn.roi_heads.box_predictor
    assert (predictor.cls_score.out_features, predictor.bbox_pred.out_features) == (9, 36)
    anchors = frcnn.rpn.anchor_generator
    assert anchors.sizes == ((16,), (32,), (64,), (128,), (256,))
    assert anchors.num_anchors_per_location() == [3] * 5
    assert frcnn.rpn.head.cls_logits.out_channels == 3
    assert (frcnn.roi_heads.detections_per_img, frcnn.roi_heads.score_thresh) == (500, 0.001)
    assert (frcnn.transform.min_size, frcnn.transform.max_size) == ((720,), 1280)


def test_retinanet_classes_anchors_and_prior(retina: torch.nn.Module) -> None:
    """RetinaNet has 9 anchors per location, 9-label logits with the prior bias and the giou loss by default."""
    head = retina.head.classification_head
    assert head.num_classes == 9 and head.cls_logits.out_channels == 9 * 9
    assert torch.allclose(head.cls_logits.bias, torch.full((81,), -math.log(99.0)))
    assert retina.head.regression_head.bbox_reg.out_channels == 9 * 4
    assert retina.anchor_generator.num_anchors_per_location() == [9] * 5
    assert retina.anchor_generator.sizes[0] == (16, 20, 25)
    assert retina.head.regression_head._loss_type == "giou"
    assert (retina.detections_per_img, retina.score_thresh, retina.topk_candidates) == (500, 0.001, 1000)


def test_bad_model_settings_raise() -> None:
    """Unknown keys, unknown losses and wrong anchor level counts are rejected."""
    with pytest.raises(KeyError):
        build_frcnn({"model": {"anchor_size": [32]}}, pretrained=False)
    with pytest.raises(ValueError):
        build_frcnn({"model": {"box_loss": "l2"}}, pretrained=False)
    with pytest.raises(ValueError):
        build_retinanet({"model": {"anchor_base_sizes": [32, 64]}}, pretrained=False)


def loss_inputs(pred: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> tuple:
    """Return fastrcnn_loss arguments with the predicted deltas placed at each RoI's label."""
    regression = torch.zeros(len(labels), 9, 4)
    regression[torch.arange(len(labels)), labels] = pred
    return torch.zeros(len(labels), 9), regression.reshape(len(labels), 36), [labels], [target]


def test_giou_loss_zero_for_identical_boxes_and_grows_with_shift() -> None:
    """The GIoU box loss is zero when deltas match and increases as the predicted centre moves away."""
    target = torch.tensor([[0.1, -0.2, 0.3, 0.1], [0.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([2, 5])
    shifts = (0.0, 2.0, 5.0, 10.0)
    losses = [pt.giou_fastrcnn_loss(*loss_inputs(target + torch.tensor([s, 0, 0, 0]), target, labels))[1].item() for s in shifts]
    assert losses[0] == pytest.approx(0.0, abs=1e-6)
    assert losses == sorted(losses) and len(set(losses)) == len(losses)


def test_giou_on_shared_reference_matches_real_proposals() -> None:
    """Decoding against one shared box gives the same GIoU loss as decoding against each RoI's proposal."""
    gen = torch.Generator().manual_seed(0)
    corner = torch.rand(6, 2, generator=gen) * 500
    proposals = torch.cat([corner, corner + 20 + torch.rand(6, 2, generator=gen) * 200], dim=1)
    pred, target = torch.randn(6, 4, generator=gen) * 0.5, torch.randn(6, 4, generator=gen) * 0.5
    labels = torch.tensor([1, 2, 3, 4, 7, 8])
    coder = BoxCoder(pt.RCNN_WEIGHTS)
    expected = generalized_box_iou_loss(
        coder.decode_single(pred, proposals), coder.decode_single(target, proposals), reduction="sum"
    ) / len(labels)
    _, got = pt.giou_fastrcnn_loss(*loss_inputs(pred, target, labels))
    assert got.item() == pytest.approx(expected.item(), rel=1e-4)


def test_patches_off_unless_configured(frcnn: torch.nn.Module, retina: torch.nn.Module) -> None:
    """Baseline configs keep torchvision's losses and each patch switches on only from its config key."""
    assert pt.configure_patches(FRCNN, frcnn) == []
    assert pt.configure_patches(RETINA, retina) == []
    assert tv_roi_heads.fastrcnn_loss is pt.ORIGINAL_FASTRCNN_LOSS
    assert tv_retinanet.sigmoid_focal_loss is pt.ORIGINAL_FOCAL_LOSS
    assert pt.configure_patches({"family": "frcnn", "model": {"box_loss": "giou"}}, frcnn) == ["frcnn_giou"]
    assert tv_roi_heads.fastrcnn_loss.func is pt.giou_fastrcnn_loss
    assert tv_roi_heads.fastrcnn_loss.keywords == {"weights": (10.0, 10.0, 5.0, 5.0)}
    pt.configure_patches({"family": "retinanet", "model": {"focal_gamma": 1.5}}, retina)
    assert tv_retinanet.sigmoid_focal_loss.keywords == {"gamma": 1.5}
    assert tv_roi_heads.fastrcnn_loss is pt.ORIGINAL_FASTRCNN_LOSS


def test_warmup_cosine_schedule() -> None:
    """The factor rises linearly over the warm-up, then follows a cosine down to zero at the total."""
    schedule = rt.WarmupCosine(warmup=10, total=110)
    assert schedule(0) == pytest.approx(rt.WARMUP_START)
    assert schedule(10) == pytest.approx(1.0)
    assert schedule(60) == pytest.approx(0.5)
    assert schedule(110) == pytest.approx(0.0, abs=1e-12)
    assert rt.WarmupCosine(warmup=500, total=100)(20) == pytest.approx(1.0)
    assert rt.WarmupCosine(warmup=10, total=110, decay=False)(100) == 1.0


def test_plan_iterations_fits_budget() -> None:
    """Planned iterations plus one evaluation per epoch fill but never exceed the remaining budget."""
    progress = rt.Progress(iteration=200, iters_per_epoch=1000, train_seconds=100.0)
    total = rt.plan_iterations(3600.0, progress, sec_per_iter=0.5, eval_s=60.0)
    assert total == 6360
    assert (total - 200) * 0.5 + math.ceil(total / 1000) * 60 <= 3500
    assert rt.plan_iterations(50.0, progress, 0.5, 60.0) == 200


@pytest.mark.parametrize(("spec", "expected"), [("cpu", "cpu"), ("0", "cuda:0"), ("1", "cuda:1"), ("0,1", "cuda:0"), ("cuda:0", "cuda:0")])
def test_torch_device(spec: str, expected: str) -> None:
    """Device strings from the lane script and the CLI map to torch devices."""
    assert str(rt.torch_device(spec)) == expected


@pytest.fixture
def toy_root(tmp_path: Path) -> Path:
    """Write a prepared split toy with one labelled frame plus an ignore region and one empty frame."""
    folder = tmp_path / "toy" / "images" / "clip"
    folder.mkdir(parents=True)
    for name in ("0000001.jpg", "0000002.jpg"):
        cv2.imwrite(str(folder / name), np.full((40, 60, 3), 200, np.uint8))
    images = [
        {"id": i, "file_name": f"toy/images/clip/000000{i}.jpg", "width": 60, "height": 40, "clip": "clip", "frame": i}
        for i in (1, 2)
    ]
    anns = [
        {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 5, 20, 10], "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 3, "bbox": [0, 0, 8, 8], "iscrowd": 1},
    ]
    (tmp_path / "toy" / "annotations_coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": []}))
    return tmp_path


def test_dataset_targets_flip_and_empty_frames(toy_root: Path) -> None:
    """Images come back as floats in [0, 1] with xyxy boxes and class + 1 labels, and empty frames are kept."""
    image, target = CocoDetectionDataset(toy_root, "toy", augment=False)[0]
    assert image.dtype == torch.float32 and image.shape == (3, 40, 60) and 0.0 <= image.min() <= image.max() <= 1.0
    assert target["boxes"].tolist() == [[10.0, 5.0, 30.0, 15.0]] and target["labels"].tolist() == [3]
    assert target["image_id"] == 1
    _, empty = CocoDetectionDataset(toy_root, "toy", augment=False)[1]
    assert empty["boxes"].shape == (0, 4) and empty["labels"].dtype == torch.int64
    flipped = CocoDetectionDataset(toy_root, "toy")
    flipped.transforms.transforms[0].p = 1.0
    assert flipped[0][1]["boxes"].tolist() == [[30.0, 5.0, 50.0, 15.0]]


def test_trainer_resumes_and_stops_at_budget(toy_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint restores progress and schedule mid-run, and an exhausted time budget stops training."""
    cfg = {"family": "frcnn", "model": {"min_size": 64, "max_size": 96}}
    tcfg = {**rt.TRAIN_DEFAULTS, "workers": 0, "batch": 1, "warmup_iters": 0}
    dataset, cpu = CocoDetectionDataset(toy_root, "toy", augment=False), torch.device("cpu")
    first = rt.Trainer(build_frcnn(cfg, pretrained=False), tcfg, dataset, cpu, total_iters=4)
    first.checkpoint = toy_root / "last.pt"
    first.end_epoch(*first.run_epoch())
    assert (first.progress.epoch, first.progress.iteration) == (1, 2)
    resumed = rt.Trainer(build_frcnn(cfg, pretrained=False), tcfg, dataset, cpu, total_iters=999)
    resumed.load(toy_root / "last.pt")
    assert (resumed.progress.iteration, resumed.progress.total_iters, resumed.schedule.total) == (2, 4, 4)
    assert resumed.fit().iteration == 4 and resumed.progress.epochs_done == 2.0
    monkeypatch.setattr(rt, "PLAN_AFTER", 1)
    limited = rt.Trainer(build_frcnn(cfg, pretrained=False), tcfg, dataset, cpu, total_iters=50, budget_s=1e-6)
    assert limited.fit().iteration == 1 and limited.progress.planned
