"""Prediction table schema, validation, (clip, frame) matching and per-image top-k selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from hw2.constants import MAX_DETS, NUM_CLASSES

PRED_COLUMNS: tuple[str, ...] = ("file_name", "class_id", "score", "x", "y", "w", "h")
FLOAT_COLUMNS: tuple[str, ...] = ("score", "x", "y", "w", "h")

FrameKey = tuple[str, int]


def frame_key(file_name: str) -> FrameKey:
    """Return the (clip, frame number) parsed from a path ending in <clip>/<frame>.jpg."""
    path = PurePosixPath(str(file_name).replace("\\", "/"))
    if not path.stem.isdigit() or not path.parent.name:
        raise ValueError(f"cannot parse clip and frame from file name {file_name!r}")
    return path.parent.name, int(path.stem)


def match_frames(file_names: pd.Series, table: Mapping[FrameKey, int], target: str) -> np.ndarray:
    """Return the table id of each file name's (clip, frame) key, raising when any frame is unknown."""
    ids = {name: table.get(frame_key(name), -1) for name in pd.unique(file_names)}
    missing = [name for name, value in ids.items() if value < 0]
    if missing:
        raise ValueError(f"{len(missing)} file name(s) match no {target} frame, e.g. {missing[:3]}")
    return file_names.map(ids).to_numpy(dtype=np.int64)


def _check_dtypes(df: pd.DataFrame) -> None:
    """Raise when a prediction column has the wrong dtype or holds nulls."""
    missing = [col for col in PRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"predictions miss columns {missing}; expected {list(PRED_COLUMNS)}")
    nulls = [col for col in PRED_COLUMNS if df[col].isna().any()]
    if nulls:
        raise ValueError(f"predictions hold nulls in {nulls}")
    if not ptypes.is_string_dtype(df["file_name"]):
        raise TypeError(f"file_name must hold strings, got {df['file_name'].dtype}")
    if not ptypes.is_integer_dtype(df["class_id"]) or ptypes.is_bool_dtype(df["class_id"]):
        raise TypeError(f"class_id must be an integer dtype, got {df['class_id'].dtype}")
    for col in FLOAT_COLUMNS:
        if not ptypes.is_numeric_dtype(df[col]) or ptypes.is_bool_dtype(df[col]):
            raise TypeError(f"{col} must be numeric, got {df[col].dtype}")


def _check_values(df: pd.DataFrame) -> None:
    """Raise when class ids, scores or coordinates fall outside their valid ranges."""
    bad_class = ~df["class_id"].between(0, NUM_CLASSES - 1)
    if bad_class.any():
        raise ValueError(f"class_id outside 0-{NUM_CLASSES - 1}: {sorted(df.loc[bad_class, 'class_id'].unique())[:5]}")
    if not np.isfinite(df[list(FLOAT_COLUMNS)].to_numpy()).all():
        raise ValueError("score, x, y, w and h must be finite")
    if not df["score"].between(0.0, 1.0).all():
        raise ValueError(f"score outside [0, 1]: min {df['score'].min()}, max {df['score'].max()}")


def validate_preds(df: pd.DataFrame) -> pd.DataFrame:
    """Check prediction columns, dtypes and value ranges and return a copy with canonical dtypes."""
    _check_dtypes(df)
    out = df.reset_index(drop=True).astype({"file_name": str, "class_id": np.int64})
    out = out.astype({col: np.float64 for col in FLOAT_COLUMNS})
    _check_values(out)
    extras = [col for col in out.columns if col not in PRED_COLUMNS]
    return out[[*PRED_COLUMNS, *extras]]


def load_preds(path: str | Path) -> pd.DataFrame:
    """Read a predictions parquet and return it validated."""
    return validate_preds(pd.read_parquet(path))


def top_k_per_image(df: pd.DataFrame, k: int = MAX_DETS, by: str = "file_name") -> pd.DataFrame:
    """Keep the k highest-scoring rows of each image group, ties kept in original order."""
    ranked = df.sort_values("score", ascending=False, kind="stable")
    return ranked.groupby(by, sort=False).head(k)
