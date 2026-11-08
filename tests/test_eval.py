"""Tests for COCO evaluation on real val clips: perfect boxes, shifts, ignore regions, the 500-box budget and backends."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hw2.eval.coco_eval import evaluate
from hw2.paths import data_root
from hw2.utils import load_json

VAL_GT = data_root() / "val" / "annotations_coco.json"
CLIPS = ("uav0000137_00458_v", "uav0000305_00000_v")

pytestmark = pytest.mark.skipif(not VAL_GT.exists(), reason="val ground truth not available")


def _subset(gt: dict[str, Any], keep: set[int]) -> dict[str, Any]:
    """Return a ground-truth dict restricted to the given image ids."""
    images = [img for img in gt["images"] if img["id"] in keep]
    anns = [ann for ann in gt["annotations"] if ann["image_id"] in keep]
    return {"images": images, "categories": gt["categories"], "annotations": anns}


def _preds(gt: dict[str, Any], anns: list[dict[str, Any]], score: float = 1.0) -> pd.DataFrame:
    """Turn annotations into a prediction table with one fixed score."""
    names = {img["id"]: "val/" + img["file_name"] for img in gt["images"]}
    rows = [(names[a["image_id"]], a["category_id"] - 1, score, *a["bbox"]) for a in anns]
    return pd.DataFrame(rows, columns=["file_name", "class_id", "score", "x", "y", "w", "h"])


def _valid(gt: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the non-crowd annotations of a ground-truth dict."""
    return [ann for ann in gt["annotations"] if not ann["iscrowd"]]


def _intersects(box: list[float], boxes: np.ndarray) -> bool:
    """Return True when an xywh box overlaps any of the given xywh boxes."""
    if not len(boxes):
        return False
    ix = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])
    iy = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])
    return bool(((ix > 0) & (iy > 0)).any())


@pytest.fixture(scope="module")
def gt2() -> dict[str, Any]:
    """Return the val ground truth restricted to two clips."""
    gt = load_json(VAL_GT)
    return _subset(gt, {img["id"] for img in gt["images"] if img["video"] in CLIPS})


@pytest.fixture(scope="module")
def dense_gt(gt2: dict[str, Any]) -> dict[str, Any]:
    """Return the ground truth of the single densest frame of the two clips."""
    counts = Counter(ann["image_id"] for ann in _valid(gt2))
    image_id, n_boxes = counts.most_common(1)[0]
    assert n_boxes > 100
    return _subset(gt2, {image_id})


def test_gt_as_predictions_is_perfect(gt2: dict[str, Any]) -> None:
    """Valid ground-truth boxes used as predictions score mAP, AP50 and AR500 of exactly one."""
    metrics = evaluate(_preds(gt2, _valid(gt2)), gt2)
    assert metrics["map"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["map50"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["ar500"] == pytest.approx(1.0, abs=1e-9)
    assert set(metrics["per_clip"]) == set(CLIPS)
    assert all(value == pytest.approx(1.0) for value in metrics["per_clip"].values())


def test_shifted_boxes_score_clearly_lower(gt2: dict[str, Any]) -> None:
    """Shifting boxes by 30% of their width gives IoU near 0.54, so only AP50 survives."""
    preds = _preds(gt2, _valid(gt2))
    preds["x"] += 0.3 * preds["w"]
    metrics = evaluate(preds, gt2)
    assert metrics["map"] < 0.2
    assert metrics["map50"] > 0.9


def test_images_without_predictions_still_count(gt2: dict[str, Any]) -> None:
    """Dropping one clip's predictions zeroes that clip and lowers the overall mAP."""
    preds = _preds(gt2, _valid(gt2))
    metrics = evaluate(preds[~preds["file_name"].str.contains(CLIPS[1])], gt2)
    assert metrics["per_clip"][CLIPS[1]] == 0.0
    assert metrics["map"] < 0.9


def test_boxes_inside_ignore_regions_are_ignored(gt2: dict[str, Any]) -> None:
    """Top-scoring extra boxes inside iscrowd regions keep mAP at one, but count as false positives without them."""
    valid = _valid(gt2)
    boxes = {img["id"]: np.array([a["bbox"] for a in valid if a["image_id"] == img["id"]]) for img in gt2["images"]}
    extras = []
    for ann in (a for a in gt2["annotations"] if a["iscrowd"]):
        x, y, w, h = ann["bbox"]
        inner = [x + w / 4, y + h / 4, w / 2, h / 2]
        if not _intersects(inner, boxes[ann["image_id"]]):
            extras.append({**ann, "bbox": inner})
    assert len(extras) > 50
    preds = pd.concat([_preds(gt2, valid, score=0.9), _preds(gt2, extras, score=1.0)], ignore_index=True)
    assert evaluate(preds, gt2, breakdown=False)["map"] == pytest.approx(1.0, abs=1e-9)
    no_crowd = {**gt2, "annotations": valid}
    assert evaluate(preds, no_crowd, breakdown=False)["map"] < 0.95


def test_recall_at_500_beats_100_on_dense_frame(dense_gt: dict[str, Any]) -> None:
    """On a frame with over 100 objects a 500-box budget recalls everything while 100 boxes cannot."""
    preds = _preds(dense_gt, _valid(dense_gt))
    recall_500 = evaluate(preds, dense_gt, max_dets=500)["ar500"]
    recall_100 = evaluate(preds, dense_gt, max_dets=100)["ar100"]
    assert recall_500 == pytest.approx(1.0, abs=1e-9)
    assert recall_100 < recall_500


def test_cocoeval_cap_is_500_per_class(dense_gt: dict[str, Any]) -> None:
    """Over 100 top-scoring decoys of one class do not hide true boxes, so COCOeval caps at 500 not 100."""
    image = dense_gt["images"][0]
    grid = [(x, y) for x in range(0, image["width"] - 4, image["width"] // 15) for y in range(0, 40, 4)][:150]
    decoys = [{"image_id": image["id"], "category_id": 1, "bbox": [x, y, 3, 3]} for x, y in grid]
    assert len(decoys) == 150
    preds = pd.concat([_preds(dense_gt, _valid(dense_gt), 0.5), _preds(dense_gt, decoys, 1.0)], ignore_index=True)
    assert evaluate(preds, dense_gt, breakdown=False)["ar500"] == pytest.approx(1.0, abs=1e-9)


def test_backends_agree_on_two_clips(gt2: dict[str, Any]) -> None:
    """faster_coco_eval and pycocotools give the same metrics on noisy predictions for two clips."""
    pytest.importorskip("faster_coco_eval")
    pytest.importorskip("pycocotools")
    rng = np.random.default_rng(42)
    preds = _preds(gt2, _valid(gt2)).sample(frac=0.8, random_state=42).reset_index(drop=True)
    jitter = rng.normal(0.0, 0.1, (len(preds), 4)) * preds[["w", "h", "w", "h"]].to_numpy()
    preds[["x", "y", "w", "h"]] += jitter
    preds[["w", "h"]] = preds[["w", "h"]].clip(lower=1.0)
    preds["score"] = rng.uniform(0.0, 1.0, len(preds))
    fast = evaluate(preds, gt2, backend="faster_coco_eval", breakdown=False)
    slow = evaluate(preds, gt2, backend="pycocotools", breakdown=False)
    assert 0.05 < fast["map"] < 0.95
    for key in ("map", "map50", "map75", "map_small", "map_medium", "map_large", "ar500"):
        assert fast[key] == pytest.approx(slow[key], abs=1e-9), key
    assert fast["per_class"] == pytest.approx(slow["per_class"], abs=1e-9)
