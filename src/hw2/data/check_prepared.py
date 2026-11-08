"""Draw ground truth on seeded random prepared frames per split and cross-check YOLO labels with COCO."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hw2.constants import CLASS_NAMES, SEED
from hw2.data import boxes as bx
from hw2.data import coco_utils as cu
from hw2.data.prepare import SPLITS
from hw2.paths import out_root, prep_root
from hw2.utils import get_logger

LOG = get_logger(__name__)
PALETTE = (
    (60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241),
    (182, 89, 155), (34, 126, 230), (160, 160, 22), (255, 0, 255),
)
IGNORE_COLOR = (255, 255, 255)
BOX_TOLERANCE_PX = 0.05


def sample_images(coco: dict[str, Any], n: int, seed: int) -> list[dict[str, Any]]:
    """Return n seeded random image entries of a COCO dict, ordered by id."""
    images = coco["images"]
    return sorted(random.Random(seed).sample(images, min(n, len(images))), key=lambda im: im["id"])


def read_yolo(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file and return classes and corner boxes in pixels."""
    rows = np.array([line.split() for line in path.read_text().splitlines() if line.strip()], dtype=np.float64)
    rows = rows.reshape(-1, 5)
    cx, cy, w, h = (rows[:, 1:] * [width, height, width, height]).T
    return rows[:, 0].astype(np.int64), np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def label_problems(prep: Path, image: dict[str, Any], frame: cu.FrameBoxes) -> list[str]:
    """Return mismatches between a frame's YOLO label file and its COCO object boxes."""
    width, height = image["width"], image["height"]
    label = prep / image["file_name"].replace("/images/", "/labels/", 1)
    label = label.with_suffix(".txt")
    if not label.is_file():
        return [f"missing label file {label}"]
    classes, boxes = read_yolo(label, width, height)
    expected = bx.clean_boxes(bx.xywh_to_xyxy(frame.boxes), (width, height), (width, height))
    if len(classes) != len(expected.keep):
        return [f"{label.name}: {len(classes)} YOLO boxes vs {len(expected.keep)} COCO boxes"]
    if not np.array_equal(classes, frame.classes[expected.keep]):
        return [f"{label.name}: class ids differ from COCO"]
    error = float(np.abs(boxes - expected.boxes).max()) if len(boxes) else 0.0
    return [f"{label.name}: boxes differ by {error:.3f} px"] if error > BOX_TOLERANCE_PX else []


def draw_frame(image: np.ndarray, frame: cu.FrameBoxes) -> np.ndarray:
    """Draw object boxes coloured by class, outline ignore regions and add a class legend."""
    canvas = image.copy()
    thick = max(1, round(max(canvas.shape[:2]) / 960))
    for x1, y1, x2, y2 in bx.round_half_up(bx.xywh_to_xyxy(frame.ignore)).tolist():
        cv2.rectangle(canvas, (x1, y1), (x2, y2), IGNORE_COLOR, thick)
    corners = bx.round_half_up(bx.xywh_to_xyxy(frame.boxes)).tolist()
    for cls, (x1, y1, x2, y2) in zip(frame.classes.tolist(), corners):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PALETTE[cls], thick)
    legend = [*zip(CLASS_NAMES, PALETTE), *([("ignore", IGNORE_COLOR)] if len(frame.ignore) else [])]
    scale, step = 0.5 * thick, 18 * thick
    cv2.rectangle(canvas, (0, 0), (110 * thick, step * len(legend) + 6 * thick), (0, 0, 0), -1)
    for row, (name, color) in enumerate(legend, start=1):
        cv2.putText(canvas, name, (6 * thick, step * row), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
    return canvas


def check_split(prep: Path, split: str, out_dir: Path, n: int, seed: int) -> list[str]:
    """Check and draw n seeded random frames of one split, returning any problems found."""
    coco_path = prep / split / "annotations_coco.json"
    if not coco_path.is_file():
        return [f"{split}: missing {coco_path}"]
    coco = cu.load_coco(coco_path)
    by_image = cu.annotations_by_image(coco["annotations"])
    problems: list[str] = []
    for image in sample_images(coco, n, seed):
        frame = cu.split_annotations(by_image.get(image["id"], []))
        pixels = cv2.imread(str(prep / image["file_name"]), cv2.IMREAD_COLOR)
        if pixels is None or (pixels.shape[1], pixels.shape[0]) != (image["width"], image["height"]):
            problems.append(f"{image['file_name']}: unreadable or size differs from COCO")
            continue
        problems += [f"{split}: {p}" for p in label_problems(prep, image, frame)]
        out = out_dir / f"{split}_{image['clip']}_{Path(image['file_name']).stem}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), draw_frame(pixels, frame), [cv2.IMWRITE_JPEG_QUALITY, 90])
        LOG.info("%s: %d boxes, %d ignore -> %s", split, len(frame.boxes), len(frame.ignore), out)
    return problems


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--n", type=int, default=3, help="frames per split")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prep", type=Path, default=None, help="prepared data, default prep_root()")
    parser.add_argument("--out", type=Path, default=None, help="default <out_root>/checks")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Check every requested split and return 1 when any problem is found."""
    args = parse_args(argv)
    prep, out_dir = args.prep or prep_root(), args.out or out_root() / "checks"
    problems = [p for split in args.splits for p in check_split(prep, split, out_dir, args.n, args.seed)]
    for problem in problems:
        LOG.error(problem)
    LOG.info("check_prepared: %s", f"{len(problems)} problem(s)" if problems else "ok")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
