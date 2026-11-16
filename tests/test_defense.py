"""Tests for the defense overview helpers on a temporary results folder."""

import json
from pathlib import Path

import pandas as pd
import pytest

from hw2.analysis import defense


def _result(exp_id: str, val_map: float) -> dict:
    """Return a minimal results json body with val and val_testlike metrics."""
    return {
        "exp_id": exp_id,
        "exp_name": f"{exp_id}_test",
        "metrics": {"val": {"map": val_map, "map50": 2 * val_map, "map_small": val_map / 2}, "val_testlike": {"map": val_map - 0.01}},
        "train_hours": 2.5,
        "epochs_done": 10,
        "ms_per_image": 15.0,
        "lb_public": None,
    }


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point results_dir at a folder with two runs, a smoke run and leaderboard scores."""
    for exp_id, val_map in (("A0", 0.16), ("A2", 0.17), ("A0_smoke", 0.5)):
        (tmp_path / f"{exp_id}.json").write_text(json.dumps(_result(exp_id, val_map)))
    (tmp_path / "lb_scores.csv").write_text("exp_id,lb_public\nbaseline,0.22\nA0,0.18\nQ0,0.30\n")
    monkeypatch.setenv("HW2_RESULTS_DIR", str(tmp_path))
    return tmp_path


def test_summary_sorts_by_val_map_and_marks_finals(results: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Runs come first by val mAP, score-only rows follow by score, smoke runs are dropped and finals are marked."""
    table = defense.summary_table(["Q0", "QF"])
    assert list(table.index) == ["A2", "A0", "Q0", "baseline"]
    assert table.loc["Q0", "final"] == defense.FINAL_MARK and table.loc["A0", "final"] == ""
    assert table.loc["A0", "LB"] == 0.18 and pd.isna(table.loc["Q0", "val mAP"])
    assert table.loc["Q0", "name"] == "Q0_rtdetr_l_nq300" and table.loc["baseline", "name"] == "baseline"
    assert capsys.readouterr().out.strip() == "final submissions without results yet: QF"


def test_family_table_keeps_score_only_rows(results: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A family without results jsons shows its scored runs and one line naming the missing ones."""
    table = defense.experiment_table("rtdetr")
    assert list(table.index) == ["Q0"] and table.loc["Q0", "LB"] == 0.30
    assert capsys.readouterr().out.strip() == "rtdetr: no results json for Q0, Q1, QF, QL"


def test_validation_table_gap(results: Path) -> None:
    """The val_testlike gap is testlike minus val mAP for every run with a json."""
    table = defense.validation_table()
    assert list(table.index) == ["A2", "A0"]
    assert table["gap"].tolist() == pytest.approx([-0.01, -0.01])


def test_load_curve_reads_both_formats(results: Path) -> None:
    """Ultralytics seconds and torchvision hours both become hours with val_monitor mAP."""
    (results / "curves").mkdir()
    pd.DataFrame({"time": [1800, 3600], defense.ULTRA_MAP: [0.1, 0.2]}).to_csv(results / "curves" / "A0.csv", index=False)
    pd.DataFrame({"hours": [0.5, 1.0], "map": [0.1, 0.2], "loss": [1.0, 0.9]}).to_csv(results / "curves" / "F0.csv", index=False)
    for exp_id in ("A0", "F0"):
        curve = defense.load_curve(exp_id)
        assert curve["hours"].tolist() == [0.5, 1.0] and curve["map"].tolist() == [0.1, 0.2]
    assert defense.load_curve("Q0") is None


def test_missing_figures_skip_with_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    """A grid of absent images returns None after one status line."""
    assert defense.figure_grid(["no_such_figure"]) is None
    assert capsys.readouterr().out.strip() == "figures not found: no_such_figure"


def test_demo_skips_quietly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Without local val predictions the demo returns None and prints nothing."""
    monkeypatch.setenv("HW2_OUT_ROOT", str(tmp_path))
    assert defense.demo_figure(["A0"]) is None
    assert capsys.readouterr().out == ""


def test_reproduce_commands_cover_bare_config() -> None:
    """Commands find FINAL.yaml, pick the runner and weights per family and launch both lanes."""
    text = defense.reproduce_commands(["QF", "FINAL"])
    assert 'bash scripts/run_lanes.sh "QF" "FINAL"' in text
    assert "--config configs/yolo/FINAL.yaml --device 0 --skip-train" in text
    assert '"$OUT_ROOT/QF/weights/last.pt"' in text
    assert 'python -m hw2.submission --validate "$OUT_ROOT/FINAL/submission.csv.gz"' in text


def _tiny_coco() -> dict:
    """Return a two-frame COCO dict with three objects and one ignore region."""
    images = [{"id": i, "file_name": f"images/clipA/{i:07d}.jpg", "width": 100, "height": 80, "frame_id": i} for i in (1, 2)]
    anns = [
        {"id": 1, "image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 40, 40], "iscrowd": 0},
        {"id": 3, "image_id": 2, "category_id": 3, "bbox": [5, 5, 100, 100], "iscrowd": 0},
        {"id": 4, "image_id": 2, "category_id": 3, "bbox": [50, 50, 20, 20], "iscrowd": 1},
    ]
    return {"images": images, "annotations": anns, "categories": [{"id": i + 1, "name": str(i)} for i in range(8)]}


def test_object_and_prediction_tables() -> None:
    """Objects skip ignore regions and ground truth becomes score-one predictions of one clip."""
    coco = _tiny_coco()
    objects = defense.object_table(coco)
    assert objects["class"].tolist() == ["car", "person", "car"]
    assert objects["size"].tolist() == ["small", "medium", "large"]
    assert defense.frame_with_ignore(coco) == 1
    preds, gt = defense.gt_as_predictions(coco, "clipA")
    assert len(preds) == 3 and len(gt["annotations"]) == 4
    assert preds["file_name"].iloc[0] == "val/images/clipA/0000001.jpg" and (preds["score"] == 1.0).all()


def test_toy_clip_has_a_gap() -> None:
    """The synthetic clip misses the car in frame 4 and holds one false person."""
    toy = defense.toy_predictions()
    assert len(defense.toy_frames()) == defense.TOY_FRAMES
    assert "val/images/toy_clip/0000004.jpg" not in set(toy["file_name"])
    assert toy["class_id"].tolist().count(0) == 1


def test_config_text_and_missing_ensembles(capsys: pytest.CaptureFixture[str]) -> None:
    """Config text names each file and missing ensembling code prints one line."""
    assert defense.config_text(["FINAL"]).startswith("== configs/yolo/FINAL.yaml\n")
    assert defense.ensemble_functions() == []
    assert capsys.readouterr().out.strip() == "WBF / soft-NMS: no implementation in hw2.infer, skipped"
