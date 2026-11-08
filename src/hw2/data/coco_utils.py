"""Read VisDrone COCO files, separate boxes from ignore regions and assemble prepared COCO files."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hw2.constants import CLASS_NAMES

EXTRA_FIELDS = ("track_id", "occlusion", "truncation", "orig_category_id")
BBOX_DECIMALS = 3


def class_to_category(class_id: int) -> int:
    """Return the one-based COCO category id of a zero-based class id."""
    return int(class_id) + 1


def category_to_class(category_id: int) -> int:
    """Return the zero-based class id of a one-based COCO category id."""
    return int(category_id) - 1


def categories() -> list[dict[str, Any]]:
    """Return the COCO categories list for the eight project classes."""
    return [{"id": class_to_category(i), "name": name} for i, name in enumerate(CLASS_NAMES)]


def load_coco(path: str | Path) -> dict[str, Any]:
    """Read a COCO JSON file."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_coco(coco: dict[str, Any], path: str | Path) -> Path:
    """Write a COCO dict as compact JSON, creating parent folders, and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, separators=(",", ":"))
    return path


def frames_by_clip(images: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group COCO image entries by video name, each list sorted by frame_id."""
    clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        clips[image["video"]].append(image)
    return {clip: sorted(frames, key=lambda im: im["frame_id"]) for clip, frames in sorted(clips.items())}


def annotations_by_image(
    annotations: Iterable[dict[str, Any]], image_ids: Collection[int] | None = None
) -> dict[int, list[dict[str, Any]]]:
    """Group annotations by image id, optionally keeping only the given image ids."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if image_ids is None or ann["image_id"] in image_ids:
            grouped[ann["image_id"]].append(ann)
    return dict(grouped)


@dataclass(frozen=True)
class FrameBoxes:
    """Object classes and boxes, unique ignore boxes and the object annotations of one frame."""

    classes: np.ndarray
    boxes: np.ndarray
    ignore: np.ndarray
    objects: list[dict[str, Any]]


def split_annotations(anns: Sequence[dict[str, Any]]) -> FrameBoxes:
    """Split frame annotations into object boxes and ignore boxes deduplicated across categories."""
    objects = [a for a in anns if not a.get("iscrowd")]
    ignore = list(dict.fromkeys(tuple(a["bbox"]) for a in anns if a.get("iscrowd")))
    return FrameBoxes(
        classes=np.array([category_to_class(a["category_id"]) for a in objects], dtype=np.int64),
        boxes=np.array([a["bbox"] for a in objects], dtype=np.float64).reshape(-1, 4),
        ignore=np.array(ignore, dtype=np.float64).reshape(-1, 4),
        objects=objects,
    )


def coco_box(xywh: Sequence[float]) -> tuple[list[float], float]:
    """Return a rounded COCO bbox and its area computed as w times h."""
    bbox = [round(float(v), BBOX_DECIMALS) for v in xywh]
    return bbox, round(bbox[2] * bbox[3], BBOX_DECIMALS)


def box_annotation(source: dict[str, Any], class_id: int, xywh: Sequence[float]) -> dict[str, Any]:
    """Return a non-crowd COCO annotation without ids, carrying the source track and visibility fields."""
    bbox, area = coco_box(xywh)
    extra = {key: source[key] for key in EXTRA_FIELDS if key in source}
    return {"category_id": class_to_category(class_id), "bbox": bbox, "area": area, "iscrowd": 0, **extra}


def scale_annotation(ann: dict[str, Any], sx: float, sy: float) -> dict[str, Any]:
    """Return a copy of an annotation without ids, with its bbox scaled and area recomputed."""
    x, y, w, h = ann["bbox"]
    bbox, area = coco_box([x * sx, y * sy, w * sx, h * sy])
    scaled = {key: value for key, value in ann.items() if key not in ("id", "image_id")}
    scaled.update(bbox=bbox, area=area)
    return scaled


def build_coco(
    frames: Iterable[tuple[dict[str, Any], list[dict[str, Any]]]], info: dict[str, Any]
) -> dict[str, Any]:
    """Assemble a COCO dict from image and annotation entries, numbering ids from one in order."""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for image_id, (image, anns) in enumerate(frames, start=1):
        images.append({"id": image_id, **image})
        for ann in anns:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, **ann})
    return {"info": info, "categories": categories(), "images": images, "annotations": annotations}
