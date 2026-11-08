"""Tests for box conversion, clipping, ignore masks, filtering and YOLO formatting."""

import numpy as np
import pytest

from hw2.data import boxes as bx
from hw2.data.coco_utils import split_annotations
from hw2.data.prepare import scale_for_width


def test_xywh_xyxy_roundtrip() -> None:
    """Converting to corners and back returns the original boxes."""
    xywh = np.array([[10, 20, 30, 40], [0, 0, 1.5, 2.5]])
    assert np.allclose(bx.xywh_to_xyxy(xywh), [[10, 20, 40, 60], [0, 0, 1.5, 2.5]])
    assert np.allclose(bx.xyxy_to_xywh(bx.xywh_to_xyxy(xywh)), xywh)
    assert bx.xywh_to_xyxy([]).shape == (0, 4)


def test_clip_and_pixel_rects() -> None:
    """Boxes clip to the frame and pixel rectangles round halves up inside it."""
    boxes = [[-5, -5, 50, 50], [90, 10, 120, 30]]
    assert np.allclose(bx.clip_xyxy(boxes, 100, 40), [[0, 0, 50, 40], [90, 10, 100, 30]])
    assert bx.pixel_rects([[1.5, 2.4, 9.5, 120]], 100, 40).tolist() == [[2, 2, 10, 40]]


def test_region_mask_and_paint() -> None:
    """Rectangles become a union mask and are painted with the fill value."""
    rects = np.array([[0, 0, 4, 2], [2, 1, 6, 3]])
    mask = bx.region_mask(rects, 8, 4)
    assert mask.sum() == 8 + 8 - 2
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    bx.paint_rects(image, rects, 114)
    assert (image[mask] == 114).all() and (image[~mask] == 0).all()


def test_covered_fraction() -> None:
    """Coverage counts mask pixels inside each box, with empty boxes at zero."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :5] = True
    boxes = [[0, 0, 4, 4], [3, 0, 7, 4], [6, 0, 9, 9], [2, 2, 2, 8]]
    assert np.allclose(bx.covered_fraction(boxes, mask), [1.0, 0.5, 0.0, 0.0])


@pytest.mark.parametrize(
    ("size", "expected"),
    [((3840, 2160), (1920, 1080)), ((2720, 1530), (1920, 1080)), ((2688, 1512), (1920, 1080)), ((1344, 756), (1344, 756))],
)
def test_fit_long_side_never_upscales(size: tuple[int, int], expected: tuple[int, int]) -> None:
    """The long side is capped at 1920 and smaller frames keep their size."""
    assert bx.fit_long_side(*size, 1920) == expected


def test_testlike_scale_lookup() -> None:
    """Known widths use their scale and other widths use the default entry."""
    scales = {3840: 0.25, 1344: 0.5, "default": 1.0}
    assert scale_for_width(3840, scales) == 0.25
    assert scale_for_width(1904, scales) == 1.0
    assert bx.scaled_size(1344, 756, 0.5) == (672, 378)


def test_yolo_lines_format() -> None:
    """YOLO lines hold the class and normalized centre and size with six decimals."""
    lines = bx.yolo_lines([2], [[25, 10, 75, 30]], 100, 40)
    assert lines == ["2 0.500000 0.500000 0.500000 0.500000"]
    assert bx.yolo_lines([], np.zeros((0, 4)), 100, 40) == []


def test_clean_boxes_clips_drops_and_scales() -> None:
    """Boxes are clipped, dropped above half masked or below one output pixel, then scaled."""
    mask = np.zeros((100, 200), dtype=bool)
    mask[:, :100] = True
    boxes = [
        [150, 50, 250, 90],
        [10, 10, 60, 60],
        [80, 10, 120, 50],
        [70, 10, 110, 50],
        [150, 10, 151.5, 50],
    ]
    out = bx.clean_boxes(boxes, (200, 100), (100, 50), mask)
    assert out.keep.tolist() == [0, 2]
    assert np.allclose(out.boxes, [[75, 25, 100, 45], [40, 5, 60, 25]])
    assert (out.n_clipped, out.n_dropped_ignore, out.n_dropped_small, out.n_dropped) == (1, 2, 1, 3)


def test_split_annotations_dedupes_ignore() -> None:
    """Ignore regions repeated per category collapse to unique boxes and classes become zero-based."""
    anns = [
        {"category_id": 3, "bbox": [1, 2, 3, 4], "iscrowd": 0},
        *({"category_id": c, "bbox": [5, 5, 10, 10], "iscrowd": 1} for c in range(1, 9)),
    ]
    frame = split_annotations(anns)
    assert frame.classes.tolist() == [2]
    assert frame.boxes.tolist() == [[1, 2, 3, 4]]
    assert frame.ignore.tolist() == [[5, 5, 10, 10]]
