"""Tests for anchor k-means and coverage, RPN stage recall, failure counting, TIDE and the results table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from hw2.analysis import anchors as an
from hw2.analysis import failures as fl
from hw2.analysis import runs
from hw2.analysis import stages as st
from hw2.data import coco_utils as cu
from hw2.utils import save_json


def _frame(gt_boxes: list[list[float]], gt_cls: list[int], dets: list[tuple[int, float, float, float, float, float]]) -> fl.Frame:
    """Build a failure-analysis frame from xywh ground truth and (class, score, x, y, w, h) detections."""
    gt = cu.FrameBoxes(np.array(gt_cls), np.array(gt_boxes, float).reshape(-1, 4), np.zeros((0, 4)), [])
    table = pd.DataFrame(dets, columns=["class_id", "score", "x", "y", "w", "h"]).sort_values("score", ascending=False)
    return fl.Frame(1, ("clip", 1), Path("clip/0000001.jpg"), 200, 200, gt, table)


def test_input_scale_matches_torchvision_rule() -> None:
    """Full HD and 4K frames scale to 1280 x 720 and a narrow frame is capped by its short side."""
    np.testing.assert_allclose(an.input_scale([1920, 3840, 1000], [1080, 2160, 1000]), [2 / 3, 1 / 3, 0.72])


def test_kmeans_recovers_clusters() -> None:
    """Three tight shape clusters give three anchors near their centres with mean IoU close to one."""
    rng = np.random.default_rng(0)
    centres = np.array([[10, 20], [40, 40], [120, 60]], float)
    wh = np.concatenate([c * rng.uniform(0.97, 1.03, (200, 2)) for c in centres])
    anchors, mean_iou = an.kmeans_anchors(3, wh)
    np.testing.assert_allclose(anchors, centres, rtol=0.03)
    assert mean_iou > 0.95


def test_ratios_and_config_rewrite(tmp_path: Path) -> None:
    """Group ratios are geometric means, clear differences are flagged and R1 gets the new ratios."""
    anchors = np.array([[10, 10], [10, 20], [20, 10], [10, 10], [10, 30], [30, 10]], float)
    assert an.kmeans_ratios(anchors) == [pytest.approx(0.41, abs=0.01), 1.0, pytest.approx(2.45, abs=0.01)]
    assert not an.ratios_differ([0.5, 1.1, 2.2]) and an.ratios_differ([0.5, 1.0, 3.0])
    cfg = tmp_path / "R1.yaml"
    cfg.write_text("# change: sizes\n# ratios stay default\nbase: R0.yaml\nmodel:\n  anchor_base_sizes: [16]\n", encoding="utf-8")
    an.write_ratios([0.6, 1.3, 2.9], cfg)
    text = cfg.read_text(encoding="utf-8")
    assert "  anchor_ratios: [0.6, 1.3, 2.9]\n" in text and "stay default" not in text and "anchor_base_sizes: [16]" in text


@pytest.mark.parametrize("family", an.FAMILIES)
def test_box_equal_to_anchor_has_full_coverage(family: str) -> None:
    """A ground-truth box placed on a real anchor of a downscaled frame reaches IoU one."""
    model = an.build_detector(family, an.ANCHOR_SETS["16-256"])
    width, height = 1920, 1080
    grid, (new_h, new_w) = an.frame_anchors(model, width, height)
    assert (new_h, new_w) == (720, 1280)
    x1, y1, x2, y2 = (grid[len(grid) // 3] * width / new_w).tolist()
    objects = pd.DataFrame({"width": [width], "height": [height], "x": [x1], "y": [y1], "w": [x2 - x1], "h": [y2 - y1]})
    assert an.model_best_iou(model, objects)[0] == pytest.approx(1.0, abs=1e-4)


def test_recall_table_counts_hits_per_bucket() -> None:
    """Recall is the share of boxes at IoU 0.5 or more per bucket with counts in the last row."""
    ious = pd.DataFrame({"bucket": ["small", "small", "large"], **{s: [0.6, 0.2, 0.9] for s in st.STAGES}})
    table = st.recall_table(ious)
    assert table.loc[st.STAGES[0], "small"] == 50 and table.loc[st.STAGES[0], "all"] == pytest.approx(200 / 3)
    assert table.loc["n ground truth"].tolist() == [2, 0, 1, 3]


def test_best_iou_same_class_mask() -> None:
    """A same-class mask hides overlapping boxes of other classes."""
    gt = torch.tensor([[0.0, 0, 10, 10]])
    boxes = torch.tensor([[0.0, 0, 10, 10], [20.0, 20, 30, 30]])
    assert st.best_iou(gt, boxes)[0] == 1.0
    assert st.best_iou(gt, boxes, torch.tensor([[0.0, 1.0]]))[0] == 0.0
    assert len(st.best_iou(gt, boxes[:0])) == 1


def test_predict_stages_returns_original_pixel_proposals() -> None:
    """RPN proposals of an untrained Faster R-CNN come back at most 1000 and inside the original frame."""
    model = an.build_detector("frcnn", an.ANCHOR_SETS["default"])
    image = torch.randint(0, 255, (3, 120, 200), dtype=torch.uint8)
    proposals, dets = st.predict_stages(model, image, torch.device("cpu"))
    assert 0 < len(proposals) <= 1000 and set(dets) == {"boxes", "scores", "labels"}
    assert proposals[:, 2].max() <= 200 + 1e-3 and proposals[:, 3].max() <= 120 + 1e-3


def test_frame_errors_counts_each_kind() -> None:
    """A match, a duplicate, a class confusion and a missed small box are each counted once."""
    gt = [[0, 0, 50, 50], [100, 0, 50, 50], [180, 180, 10, 10]]
    dets = [(2, 0.9, 0, 0, 50, 50), (2, 0.8, 1, 1, 50, 50), (3, 0.7, 100, 0, 50, 50), (2, 0.1, 180, 180, 10, 10)]
    counts = fl.frame_errors(_frame(gt, [2, 2, 2], dets))
    assert counts == {"missed_small": 1, "confusions": 1, "duplicates": 1}


def test_drop_ignored_uses_area_inside() -> None:
    """Detections mostly inside an ignore region are dropped even when their IoU with it is small."""
    dets = pd.DataFrame({"x": [10.0, 150.0], "y": [10.0, 150.0], "w": [10.0, 10.0], "h": [10.0, 10.0]})
    kept = fl.drop_ignored(dets, np.array([[0.0, 0.0, 100.0, 100.0]]), 200, 200)
    assert kept["x"].tolist() == [150.0]


def test_pick_gallery_distinct_frames() -> None:
    """Each gallery kind gets its own frame even when one frame leads every count."""
    counts = pd.DataFrame({k: [5, 1, 2] for k in fl.GALLERY}, index=[7, 8, 9])
    assert [i for _, i in fl.pick_gallery(counts)] == [7, 9, 8]


def test_tide_perfect_predictions() -> None:
    """Exact detections give TIDE AP50 of 100 and no missed-object error."""
    frame = _frame([[0, 0, 50, 50], [100, 100, 40, 60]], [0, 4], [(0, 0.9, 0, 0, 50, 50), (4, 0.8, 100, 100, 40, 60)])
    errors = fl.tide_errors([frame], "unit")
    assert errors["ap50"] == pytest.approx(100.0) and errors["main"]["Miss"] == pytest.approx(0.0)


def test_results_table_reads_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The results table keeps F and R runs, reads val and val_testlike metrics and joins leaderboard scores."""
    monkeypatch.setenv("HW2_RESULTS_DIR", str(tmp_path))
    metrics = {key: 0.1 for key in runs.METRICS.values()}
    for exp in ("F0", "R0", "A0"):
        save_json({"exp_id": exp, "exp_name": f"{exp}_x", "metrics": {"val": metrics, "val_testlike": metrics}}, tmp_path / f"{exp}.json")
    (tmp_path / runs.LB_FILE).write_text("exp_id,lb_public\nF0,0.2\n", encoding="utf-8")
    table = runs.results_table()
    assert table.index.tolist() == ["F0", "R0"] and table.loc["F0", "LB"] == 0.2 and table.loc["R0", "testlike mAP"] == 0.1
