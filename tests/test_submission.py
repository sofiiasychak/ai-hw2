"""Tests for writing and validating Kaggle submission files."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hw2.submission import SubmissionError, default_test_csv, validate_submission, write_submission

TEST_ROWS = [
    ("clip_001_0000001", "test/images/clip_001/0000001.jpg", "clip_001", 1, 100, 50),
    ("clip_001_0000002", "test/images/clip_001/0000002.jpg", "clip_001", 2, 100, 50),
    ("clip_002_0000001", "test/images/clip_002/0000001.jpg", "clip_002", 1, 200, 100),
]
COLUMNS = ["file_name", "class_id", "score", "x", "y", "w", "h"]


@pytest.fixture
def test_csv(tmp_path: Path) -> Path:
    """Write a three-image test index with two image sizes."""
    path = tmp_path / "test_images.csv"
    pd.DataFrame(TEST_ROWS, columns=["image_id", "file_name", "clip", "frame", "width", "height"]).to_csv(path, index=False)
    return path


@pytest.fixture
def preds() -> pd.DataFrame:
    """Return predictions with 600 boxes on one image, edge cases and a differently prefixed path."""
    rng = np.random.default_rng(0)
    many = pd.DataFrame({
        "file_name": "test/images/clip_001/0000001.jpg",
        "class_id": rng.integers(0, 8, 600),
        "score": rng.uniform(0.01, 0.9, 600),
        "x": rng.uniform(0, 50, 600),
        "y": rng.uniform(0, 25, 600),
        "w": rng.uniform(1, 40, 600),
        "h": rng.uniform(1, 20, 600),
    })
    other = "/kaggle/input/hw2/test/images/clip_002/0000001.jpg"
    edge = pd.DataFrame(
        [
            (other, 2, 0.95, 10.0, 20.0, 30.54, 40.0),
            (other, 7, 0.123456, 190.0, -5.0, 30.0, 20.0),
            (other, 1, 0.99, 250.0, 10.0, 5.0, 5.0),
            (other, 3, 0.98, 10.0, 10.0, 0.04, 5.0),
        ],
        columns=COLUMNS,
    )
    return pd.concat([many, edge], ignore_index=True)


def _groups(text: str) -> list[list[str]]:
    """Split a PredictionString into its six-token groups."""
    tokens = text.split()
    return [tokens[i : i + 6] for i in range(0, len(tokens), 6)]


def test_round_trip(tmp_path: Path, test_csv: Path, preds: pd.DataFrame) -> None:
    """A written gzip submission validates and keeps order, the top 500, clipping and formatting."""
    path = write_submission(preds, test_csv, tmp_path / "sub.csv.gz")
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert validate_submission(path, test_csv) == {"rows": 3, "boxes": 502, "empty": 1}
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert sub["image_id"].tolist() == [row[0] for row in TEST_ROWS]
    top = _groups(sub.loc[0, "PredictionString"])
    scores = [float(group[1]) for group in top]
    assert len(top) == 500 and scores == sorted(scores, reverse=True)
    assert min(scores) >= round(preds.loc[:599, "score"].nlargest(500).min(), 4)
    assert sub.loc[1, "PredictionString"] == ""
    assert sub.loc[2, "PredictionString"] == "2 0.9500 10.0 20.0 30.5 40.0 7 0.1235 190.0 0.0 10.0 15.0"


def test_plain_csv_when_not_gz(tmp_path: Path, test_csv: Path, preds: pd.DataFrame) -> None:
    """A path without .gz is written as plain text."""
    path = write_submission(preds, test_csv, tmp_path / "sub.csv")
    assert path.read_text().startswith("image_id,PredictionString\n")


def test_unknown_frame_raises(tmp_path: Path, test_csv: Path, preds: pd.DataFrame) -> None:
    """Predictions for a frame missing from the test index are rejected."""
    preds.loc[0, "file_name"] = "test/images/clip_009/0000001.jpg"
    with pytest.raises(ValueError, match="match no test frame"):
        write_submission(preds, test_csv, tmp_path / "sub.csv")


def _corrupt(sub: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Return a copy of a valid submission broken in one named way."""
    sub = sub.copy()
    text = sub.loc[2, "PredictionString"]
    if kind == "bad_class":
        sub.loc[2, "PredictionString"] = "8" + text[1:]
    elif kind == "501_boxes":
        sub.loc[1, "PredictionString"] = " ".join(["1 0.5000 1.0 1.0 2.0 2.0"] * 501)
    elif kind == "missing_id":
        sub = sub.drop(index=1)
    elif kind == "duplicate_id":
        sub.loc[1, "image_id"] = sub.loc[0, "image_id"]
    elif kind == "token_count":
        sub.loc[2, "PredictionString"] = text + " 3"
    elif kind == "score_range":
        sub.loc[2, "PredictionString"] = text.replace("0.9500", "1.5000")
    elif kind == "outside":
        sub.loc[2, "PredictionString"] = text.replace("190.0 0.0 10.0", "190.0 0.0 12.0")
    elif kind == "zero_width":
        sub.loc[2, "PredictionString"] = text.replace("30.5 40.0", "0.0 40.0")
    elif kind == "header":
        sub = sub.rename(columns={"PredictionString": "prediction"})
    return sub


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("bad_class", "class '8'"),
        ("501_boxes", "501 boxes exceeds"),
        ("missing_id", "missing image_id"),
        ("duplicate_id", "duplicate image_id"),
        ("token_count", "multiple of 6"),
        ("score_range", "score outside"),
        ("outside", "leaves the 200x100 image"),
        ("zero_width", "w or h <= 0"),
        ("header", "header must be"),
    ],
)
def test_validator_catches_errors(tmp_path: Path, test_csv: Path, preds: pd.DataFrame, kind: str, message: str) -> None:
    """Each kind of broken submission raises a SubmissionError naming the problem."""
    good = write_submission(preds, test_csv, tmp_path / "good.csv")
    bad = tmp_path / f"{kind}.csv"
    _corrupt(pd.read_csv(good, dtype=str, keep_default_na=False), kind).to_csv(bad, index=False)
    with pytest.raises(SubmissionError, match=message):
        validate_submission(bad, test_csv)


def test_one_pixel_tolerance(tmp_path: Path, test_csv: Path) -> None:
    """A box ending half a pixel past the border passes validation."""
    path = tmp_path / "sub.csv"
    rows = [(row[0], "") for row in TEST_ROWS]
    rows[2] = (rows[2][0], "0 0.5000 150.0 50.0 50.5 50.0")
    pd.DataFrame(rows, columns=["image_id", "PredictionString"]).to_csv(path, index=False)
    assert validate_submission(path, test_csv)["boxes"] == 1


@pytest.mark.skipif(not default_test_csv().exists(), reason="test_images.csv not available")
def test_empty_predictions_cover_real_test_index(tmp_path: Path) -> None:
    """Empty predictions still give one valid row per real test image."""
    empty = pd.DataFrame(columns=COLUMNS).astype({"file_name": str, "class_id": np.int64}).astype(dict.fromkeys(COLUMNS[2:], float))
    path = write_submission(empty, default_test_csv(), tmp_path / "s.csv.gz")
    assert validate_submission(path, default_test_csv()) == {"rows": 3811, "boxes": 0, "empty": 3811}
