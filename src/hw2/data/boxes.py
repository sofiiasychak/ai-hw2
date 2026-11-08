"""Pure NumPy box helpers for conversion, clipping, ignore masks, scaling and YOLO label lines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def as_boxes(boxes: ArrayLike) -> np.ndarray:
    """Return boxes as a float64 array of shape (N, 4)."""
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def xywh_to_xyxy(boxes: ArrayLike) -> np.ndarray:
    """Convert top-left x, y, w, h boxes to corner x1, y1, x2, y2 boxes."""
    b = as_boxes(boxes)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], axis=1)


def xyxy_to_xywh(boxes: ArrayLike) -> np.ndarray:
    """Convert corner x1, y1, x2, y2 boxes to top-left x, y, w, h boxes."""
    b = as_boxes(boxes)
    return np.concatenate([b[:, :2], b[:, 2:] - b[:, :2]], axis=1)


def clip_xyxy(boxes: ArrayLike, width: float, height: float) -> np.ndarray:
    """Clip corner boxes to the frame spanning [0, width] by [0, height]."""
    return np.clip(as_boxes(boxes), 0.0, [width, height, width, height])


def scale_xyxy(boxes: ArrayLike, sx: float, sy: float) -> np.ndarray:
    """Scale corner boxes by separate horizontal and vertical factors."""
    return as_boxes(boxes) * np.array([sx, sy, sx, sy])


def round_half_up(values: ArrayLike) -> np.ndarray:
    """Round values to the nearest integer with halves going up."""
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5).astype(np.int64)


def pixel_rects(boxes: ArrayLike, width: int, height: int) -> np.ndarray:
    """Round corner boxes to integer pixel rectangles clipped to the frame."""
    return np.clip(round_half_up(as_boxes(boxes)), 0, [width, height, width, height])


def nonempty_rects(rects: np.ndarray) -> np.ndarray:
    """Return only the pixel rectangles that cover at least one pixel."""
    return rects[(rects[:, 2] > rects[:, 0]) & (rects[:, 3] > rects[:, 1])]


def region_mask(rects: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a boolean height by width mask that is True inside any pixel rectangle."""
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in rects:
        mask[y1:y2, x1:x2] = True
    return mask


def paint_rects(image: np.ndarray, rects: np.ndarray, fill: int) -> None:
    """Fill pixel rectangles of an image in place with a constant value."""
    for x1, y1, x2, y2 in rects:
        image[y1:y2, x1:x2] = fill


def covered_fraction(boxes: ArrayLike, mask: np.ndarray) -> np.ndarray:
    """Return the fraction of each corner box's pixels that lie inside a boolean mask."""
    height, width = mask.shape
    rects = pixel_rects(boxes, width, height)
    integral = np.zeros((height + 1, width + 1), dtype=np.int32)
    integral[1:, 1:] = mask.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    x1, y1, x2, y2 = rects.T
    inside = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    area = (x2 - x1) * (y2 - y1)
    return np.divide(inside, area, out=np.zeros(len(rects)), where=area > 0)


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Return the integer frame size after scaling both sides by one factor."""
    return max(1, int(round_half_up(width * scale))), max(1, int(round_half_up(height * scale)))


def fit_long_side(width: int, height: int, cap: int) -> tuple[int, int]:
    """Return the frame size with the long side at most cap, never upscaling."""
    return scaled_size(width, height, min(1.0, cap / max(width, height)))


def yolo_lines(classes: Sequence[int], boxes: ArrayLike, width: int, height: int, decimals: int = 6) -> list[str]:
    """Format corner boxes in pixels as normalized YOLO class cx cy w h lines."""
    b = as_boxes(boxes)
    size = np.array([width, height], dtype=np.float64)
    centers = (b[:, :2] + b[:, 2:]) / 2 / size
    sides = (b[:, 2:] - b[:, :2]) / size
    rows = np.clip(np.concatenate([centers, sides], axis=1), 0.0, 1.0)
    return [f"{int(c)} " + " ".join(f"{v:.{decimals}f}" for v in row) for c, row in zip(classes, rows)]


@dataclass(frozen=True)
class CleanBoxes:
    """Indices and output-pixel corner boxes kept by clean_boxes, with clip and drop counts."""

    keep: np.ndarray
    boxes: np.ndarray
    n_clipped: int
    n_dropped_ignore: int
    n_dropped_small: int

    @property
    def n_dropped(self) -> int:
        """Return the total number of dropped boxes."""
        return self.n_dropped_ignore + self.n_dropped_small


def clean_boxes(
    boxes: ArrayLike,
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mask: np.ndarray | None = None,
    max_ignore_frac: float = 0.5,
    min_size: float = 1.0,
) -> CleanBoxes:
    """Clip corner boxes to the source frame, drop mostly masked ones, scale them and drop tiny ones."""
    (src_w, src_h), (out_w, out_h) = src_size, out_size
    raw = as_boxes(boxes)
    clipped = clip_xyxy(raw, src_w, src_h)
    n_clipped = int((clipped != raw).any(axis=1).sum())
    keep = np.ones(len(raw), dtype=bool)
    if mask is not None and mask.any():
        keep = covered_fraction(clipped, mask) <= max_ignore_frac
    n_ignore = int((~keep).sum())
    scaled = scale_xyxy(clipped, out_w / src_w, out_h / src_h)
    big = (scaled[:, 2] - scaled[:, 0] >= min_size) & (scaled[:, 3] - scaled[:, 1] >= min_size)
    n_small = int((keep & ~big).sum())
    idx = np.flatnonzero(keep & big)
    return CleanBoxes(idx, scaled[idx], n_clipped, n_ignore, n_small)
