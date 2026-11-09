"""Tests for temporal track linking, rescoring and gap filling on synthetic tracks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hw2.config import load_config
from hw2.eval.preds import PRED_COLUMNS, frame_key, validate_preds
from hw2.infer import temporal as tp

Row = tuple[str, int, int, float, float, float, float, float]


def _name(clip: str, frame: int) -> str:
    """Return a val-style file name for a clip frame."""
    return f"val/images/{clip}/{frame:07d}.jpg"


def _preds(rows: list[Row]) -> pd.DataFrame:
    """Build a prediction table from (clip, frame, class, score, x, y, w, h) rows."""
    data = [(_name(clip, frame), cls, score, x, y, w, h) for clip, frame, cls, score, x, y, w, h in rows]
    df = pd.DataFrame(data, columns=list(PRED_COLUMNS))
    return df.astype({"file_name": str, "class_id": np.int64, **{col: float for col in PRED_COLUMNS[2:]}})


def _moving(clip: str, frames: list[int], cls: int, scores: list[float], x0: float, dx: float = 2.0) -> list[Row]:
    """Return rows of one object moving right by dx pixels per frame."""
    return [(clip, f, cls, s, x0 + dx * f, 50.0, 20.0, 20.0) for f, s in zip(frames, scores)]


def _labels(ts: tp.TrackSet) -> pd.Series:
    """Return track labels indexed by (frame, class, rounded x)."""
    frames = ts.preds["file_name"].map(lambda n: frame_key(n)[1])
    index = pd.MultiIndex.from_arrays([frames, ts.preds["class_id"], ts.preds["x"].round()])
    return pd.Series(ts.track, index=index)


def test_iou_matrix_values() -> None:
    """Identical, disjoint and half-shifted boxes give IoU 1, 0 and 1/3."""
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [20, 20, 5, 5], [5, 0, 10, 10]], float)
    np.testing.assert_allclose(tp.iou_matrix(a, b), [[1.0, 0.0, 1 / 3]])


def test_hungarian_global_assignment_and_threshold() -> None:
    """Matching maximises total IoU across pairs and drops pairs below the threshold."""
    a = np.array([[0, 0, 10, 10], [8, 0, 10, 10], [100, 0, 10, 10]], float)
    b = np.array([[9, 0, 10, 10], [1, 0, 10, 10], [107, 0, 10, 10]], float)
    rows, cols = tp.hungarian(a, b, iou_thr=0.3)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0)]


def test_links_objects_per_class_and_clip() -> None:
    """Two moving objects form two tracks, and other classes or clips never join them."""
    frames = [1, 2, 3, 4, 5]
    rows = _moving("c1", frames, 2, [0.9] * 5, 0) + _moving("c1", frames, 2, [0.5] * 5, 200)
    rows += [("c1", 3, 0, 0.8, 6.0, 50.0, 20.0, 20.0), ("c2", 2, 2, 0.7, 5.0, 50.0, 20.0, 20.0)]
    ts = tp.build_tracks(_preds(rows), iou_thr=0.3)
    labels = _labels(ts)
    assert labels.loc[[(f, 2, 2.0 * f) for f in frames]].nunique() == 1
    assert labels.loc[[(f, 2, 200 + 2.0 * f) for f in frames]].nunique() == 1
    assert ts.track.max() + 1 == 4


def test_low_iou_breaks_track() -> None:
    """A jump below the IoU threshold starts a new track."""
    rows = _moving("c1", [1, 2], 2, [0.9, 0.9], 0) + [("c1", 3, 2, 0.9, 16.0, 50.0, 20.0, 20.0)]
    ts = tp.build_tracks(_preds(rows), iou_thr=0.5)
    assert ts.track.max() + 1 == 2


def test_track_stats_match_bruteforce() -> None:
    """Vectorised track lengths and top-k means equal a per-track loop."""
    rng = np.random.default_rng(42)
    track, score = rng.integers(0, 30, 400), rng.random(400)
    track[track == 7] = 8
    length, top = tp.track_stats(track, score, top_k=4)
    for t in range(30):
        values = np.sort(score[track == t])[::-1]
        assert length[t] == len(values)
        if len(values):
            assert top[t] == pytest.approx(values[:4].mean())


def test_rescore_long_tracks_only() -> None:
    """Long tracks blend score with their top-k mean while short tracks keep their scores."""
    long_scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    rows = _moving("c1", [1, 2, 3, 4, 5], 2, long_scores, 0) + _moving("c1", [1, 2], 2, [0.3, 0.4], 300)
    ts = tp.build_tracks(_preds(rows), iou_thr=0.3)
    out = tp.apply_params(ts, tp.Params(alpha=0.5, top_k=2, min_len=3, decay=0.0))
    out = out.sort_values(["x", "file_name"])
    expected = [0.5 * s + 0.5 * 0.85 for s in long_scores] + [0.3, 0.4]
    np.testing.assert_allclose(out["score"].to_numpy(), expected)


def test_gap_filled_with_interpolated_box() -> None:
    """A one-frame gap is stitched and filled at the midpoint with the decayed lower neighbour score."""
    frames = [1, 2, 4, 5]
    rows = _moving("c1", frames, 3, [0.6, 0.8, 0.4, 0.6], 0, dx=4.0)
    all_frames = [_name("c1", f) for f in range(1, 6)]
    params = tp.Params(alpha=1.0, min_len=3, decay=0.5)
    ts = tp.build_tracks(_preds(rows), params.iou_thr, all_frames)
    assert ts.track.max() + 1 == 1
    out = validate_preds(tp.apply_params(ts, params))
    filled = out[out["file_name"] == _name("c1", 3)]
    assert len(out) == 5 and len(filled) == 1
    assert filled.iloc[0][["class_id", "score", "x", "y", "w", "h"]].tolist() == pytest.approx([3, 0.2, 12.0, 50.0, 20.0, 20.0])


def test_no_fill_for_short_tracks_or_two_frame_gaps() -> None:
    """Gaps of short tracks stay empty and two missing frames are never stitched."""
    all_frames = [_name("c1", f) for f in range(1, 7)]
    short = _moving("c1", [1, 2, 4, 5], 3, [0.6] * 4, 0, dx=4.0)
    out = tp.temporal_refine(_preds(short), tp.Params(min_len=5, decay=0.5), all_frames)
    assert len(out) == 4
    wide = _moving("c1", [1, 2, 5, 6], 3, [0.6] * 4, 0, dx=1.0)
    ts = tp.build_tracks(_preds(wide), 0.3, all_frames)
    assert ts.track.max() + 1 == 2 and len(ts.gaps) == 0


def test_missing_frames_shift_positions() -> None:
    """Without a frame list, frames absent from the predictions count as consecutive."""
    rows = _moving("c1", [1, 2, 4, 5], 3, [0.6] * 4, 0, dx=4.0)
    ts = tp.build_tracks(_preds(rows), 0.3)
    assert ts.track.max() + 1 == 1 and len(ts.gaps) == 0


def test_empty_predictions() -> None:
    """An empty table stays empty."""
    out = tp.temporal_refine(_preds([]), tp.Params(), [_name("c1", 1)])
    assert out.empty


def test_tune_scans_grid_then_decay() -> None:
    """Tuning scores every rescoring combination, then each positive decay on the best one."""
    rows = _moving("c1", [1, 2, 4, 5, 6], 3, [0.6] * 5, 0, dx=4.0)
    grid = {"iou_thr": [0.3, 0.5], "alpha": [0.5], "top_k": [3], "min_len": [3, 4], "decay": [0.0, 0.5]}
    frames = [_name("c1", f) for f in range(1, 7)]
    best, scores = tp.tune(_preds(rows), len, grid, frames)
    assert len(scores) == 5
    assert best.decay == 0.5 and scores[best] == 6


def test_t1_config_found() -> None:
    """The T1 experiment id resolves to one temporal config with a grid."""
    cfg = load_config(tp.find_config("T1"))
    assert cfg["exp_id"] == "T1" and set(cfg["grid"]) == {*tp.GRID_KEYS, "decay"}
