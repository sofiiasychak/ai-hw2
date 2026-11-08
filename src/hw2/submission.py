"""Write and validate Kaggle submission CSVs built from test prediction tables."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hw2.constants import MAX_DETS, NUM_CLASSES
from hw2.eval.preds import load_preds, match_frames, top_k_per_image, validate_preds
from hw2.paths import data_root
from hw2.utils import get_logger

SUB_COLUMNS: tuple[str, ...] = ("image_id", "PredictionString")
TEST_COLUMNS: tuple[str, ...] = ("image_id", "file_name", "clip", "frame", "width", "height")
CLASS_TOKENS: frozenset[str] = frozenset(str(c) for c in range(NUM_CLASSES))
GROUP_SIZE = 6
BOX_TOLERANCE = 1.0
GZIP = {"method": "gzip", "mtime": 0}

log = get_logger(__name__)


class SubmissionError(ValueError):
    """Raised when a submission file breaks the competition format."""


def default_test_csv() -> Path:
    """Return the test_images.csv path inside the raw data folder."""
    return data_root() / "test_images.csv"


def read_test_csv(test_csv: str | Path) -> pd.DataFrame:
    """Read test_images.csv and check its columns and unique ids and frames."""
    test = pd.read_csv(test_csv, dtype={"image_id": str, "clip": str})
    missing = [col for col in TEST_COLUMNS if col not in test.columns]
    if missing:
        raise ValueError(f"{test_csv}: missing columns {missing}")
    if test["image_id"].duplicated().any() or test.duplicated(["clip", "frame"]).any():
        raise ValueError(f"{test_csv}: duplicate image ids or (clip, frame) pairs")
    return test


def clip_boxes(boxes: np.ndarray, width: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Clip xywh boxes to their image and round corners to 0.1 px, returning xywh."""
    x1 = np.round(np.clip(boxes[:, 0], 0, width), 1)
    y1 = np.round(np.clip(boxes[:, 1], 0, height), 1)
    x2 = np.round(np.clip(boxes[:, 0] + boxes[:, 2], 0, width), 1)
    y2 = np.round(np.clip(boxes[:, 1] + boxes[:, 3], 0, height), 1)
    return np.stack([x1, y1, np.round(x2 - x1, 1), np.round(y2 - y1, 1)], axis=1)


def submission_boxes(preds: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Match predictions to test rows, clip them, drop empty boxes and keep the top 500 per image."""
    table = {(clip, int(frame)): i for i, (clip, frame) in enumerate(zip(test["clip"], test["frame"]))}
    rows = match_frames(preds["file_name"], table, "test")
    sizes = test[["width", "height"]].to_numpy(np.float64)[rows]
    xywh = clip_boxes(preds[["x", "y", "w", "h"]].to_numpy(np.float64), sizes[:, 0], sizes[:, 1])
    boxes = pd.DataFrame(xywh, columns=["x", "y", "w", "h"])
    boxes.insert(0, "score", preds["score"].to_numpy())
    boxes.insert(0, "class_id", preds["class_id"].to_numpy())
    boxes.insert(0, "row", rows)
    boxes = boxes[(boxes["w"] > 0) & (boxes["h"] > 0)]
    return top_k_per_image(boxes, MAX_DETS, by="row")


def prediction_strings(boxes: pd.DataFrame, n_rows: int) -> list[str]:
    """Join each test row's boxes into 'cls score x y w h' groups, empty when it has none."""
    columns = (boxes[col].tolist() for col in ("class_id", "score", "x", "y", "w", "h"))
    groups = [f"{c} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}" for c, s, x, y, w, h in zip(*columns)]
    joined = pd.Series(groups, index=boxes["row"].to_numpy(), dtype=object).groupby(level=0, sort=False).agg(" ".join)
    return joined.reindex(range(n_rows), fill_value="").tolist()


def write_submission(preds: pd.DataFrame | str | Path, test_csv: str | Path, out_path: str | Path) -> Path:
    """Write one submission row per test image in test_images.csv order, gzipped when out_path ends in .gz."""
    test = read_test_csv(test_csv)
    preds = validate_preds(preds) if isinstance(preds, pd.DataFrame) else load_preds(preds)
    boxes = submission_boxes(preds, test)
    sub = pd.DataFrame({"image_id": test["image_id"], "PredictionString": prediction_strings(boxes, len(test))})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, compression=GZIP if out_path.suffix == ".gz" else None)
    log.info("wrote %s: %d rows, %d boxes", out_path, len(sub), len(boxes))
    return out_path


def check_ids(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise when the header, row count or image ids differ from the test index."""
    if tuple(sub.columns) != SUB_COLUMNS:
        raise SubmissionError(f"header must be {','.join(SUB_COLUMNS)}, got {','.join(map(str, sub.columns))}")
    duplicated = sub.loc[sub["image_id"].duplicated(), "image_id"].tolist()
    if duplicated:
        raise SubmissionError(f"{len(duplicated)} duplicate image_id(s), e.g. {duplicated[:3]}")
    missing = sorted(set(test["image_id"]) - set(sub["image_id"]))
    if missing:
        raise SubmissionError(f"{len(missing)} missing image_id(s), e.g. {missing[:3]}")
    unknown = sorted(set(sub["image_id"]) - set(test["image_id"]))
    if unknown:
        raise SubmissionError(f"{len(unknown)} unknown image_id(s), e.g. {unknown[:3]}")
    if len(sub) != len(test):
        raise SubmissionError(f"expected {len(test)} rows, got {len(sub)}")


def parse_groups(image_id: str, text: str) -> np.ndarray:
    """Parse one PredictionString into an (n, 6) float array, raising on malformed groups."""
    tokens = text.split()
    if len(tokens) % GROUP_SIZE:
        raise SubmissionError(f"{image_id}: {len(tokens)} tokens is not a multiple of {GROUP_SIZE}")
    n_groups = len(tokens) // GROUP_SIZE
    if n_groups > MAX_DETS:
        raise SubmissionError(f"{image_id}: {n_groups} boxes exceeds the limit of {MAX_DETS}")
    bad_class = [tok for tok in tokens[::GROUP_SIZE] if tok not in CLASS_TOKENS]
    if bad_class:
        raise SubmissionError(f"{image_id}: class {bad_class[0]!r} is not an integer in 0-{NUM_CLASSES - 1}")
    try:
        values = np.asarray(tokens, dtype=np.float64).reshape(n_groups, GROUP_SIZE)
    except ValueError as err:
        raise SubmissionError(f"{image_id}: non-numeric token ({err})") from None
    if not np.isfinite(values).all():
        raise SubmissionError(f"{image_id}: non-finite score or coordinate")
    return values


def check_boxes(image_id: str, values: np.ndarray, width: float, height: float) -> None:
    """Raise when scores leave [0, 1], sizes are not positive or boxes leave the image by over 1 px."""
    score, x, y, w, h = values[:, 1:].T
    if ((score < 0) | (score > 1)).any():
        raise SubmissionError(f"{image_id}: score outside [0, 1]")
    if ((w <= 0) | (h <= 0)).any():
        raise SubmissionError(f"{image_id}: box with w or h <= 0")
    tol = BOX_TOLERANCE
    outside = (x < -tol) | (y < -tol) | (x + w > width + tol) | (y + h > height + tol)
    if outside.any():
        box = values[np.argmax(outside), 2:].tolist()
        raise SubmissionError(f"{image_id}: box {box} leaves the {width:g}x{height:g} image")


def validate_submission(path: str | Path, test_csv: str | Path) -> dict[str, Any]:
    """Check a submission file against the competition format, raise SubmissionError on the first problem, else print one line."""
    test = read_test_csv(test_csv)
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    check_ids(sub, test)
    sizes = test.set_index("image_id").loc[sub["image_id"], ["width", "height"]].to_numpy(np.float64)
    n_boxes = 0
    for image_id, text, (width, height) in zip(sub["image_id"], sub["PredictionString"], sizes):
        values = parse_groups(image_id, text)
        check_boxes(image_id, values, width, height)
        n_boxes += len(values)
    n_empty = int((sub["PredictionString"].str.strip() == "").sum())
    print(f"submission ok: {path} | {len(sub)} rows | {n_boxes} boxes | {n_empty} empty")
    return {"rows": len(sub), "boxes": n_boxes, "empty": n_empty}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse options to write a submission from predictions or validate an existing one."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preds", type=Path, help="test predictions parquet")
    mode.add_argument("--validate", type=Path, help="submission file to validate")
    parser.add_argument("--out", type=Path, help="submission path, .gz for gzip")
    parser.add_argument("--test-csv", type=Path, default=None, help="defaults to data_root/test_images.csv")
    args = parser.parse_args(argv)
    if args.preds and not args.out:
        parser.error("--preds needs --out")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Write then validate a submission, or validate an existing one."""
    args = parse_args(argv)
    test_csv = args.test_csv or default_test_csv()
    path = write_submission(args.preds, test_csv, args.out) if args.preds else args.validate
    validate_submission(path, test_csv)


if __name__ == "__main__":
    main()
