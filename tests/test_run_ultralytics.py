"""Tests for the pure helpers of the Ultralytics experiment runner."""

from pathlib import Path

import pytest
import yaml

from hw2.train import run_ultralytics as ru


def test_train_args_precedence_and_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config train values beat defaults, run settings are fixed and smoke shrinks the run."""
    monkeypatch.setenv("HW2_OUT_ROOT", str(tmp_path))
    cfg = {"train": {"time": 2.5, "epochs": 300, "imgsz": 1280, "workers": 4}, "pretrained": None}
    args = ru.train_args(cfg, "A0", tmp_path / "data.yaml", "0", smoke=False)
    assert args["workers"] == 4 and args["seed"] == 42 and args["deterministic"] is True
    assert (args["project"], args["name"], args["exist_ok"], args["device"]) == (str(tmp_path), "A0", True, "0")
    assert "pretrained" not in args
    smoke = ru.train_args(cfg, "A0_smoke", tmp_path / "data.yaml", "cpu", smoke=True)
    assert (smoke["fraction"], smoke["epochs"], smoke["time"]) == (0.05, 1, None)


def test_predict_args_maps_half_to_quantize() -> None:
    """Predict defaults hold the score threshold and 500 boxes, and half becomes quantize."""
    args = ru.predict_args({"predict": {"imgsz": 1280, "half": True, "iou": 0.7}})
    assert args == {"conf": 0.001, "max_det": 500, "imgsz": 1280, "iou": 0.7, "quantize": 16}
    assert ru.predict_args({"predict": {"half": False}})["quantize"] is None


@pytest.mark.parametrize(("device", "count"), [("cpu", 0), ("0", 1), ("0,1", 2), ("mps", 0)])
def test_gpu_count(device: str, count: int) -> None:
    """Device strings map to the number of GPUs they use."""
    assert ru.gpu_count(device) == count


def test_pick_spreads_evenly() -> None:
    """A limit keeps evenly spaced items and no limit keeps everything."""
    items = list(range(100))
    assert ru.pick(items, 4) == [0, 25, 50, 75]
    assert ru.pick(items, None) == items
    assert ru.pick(items[:3], 20) == [0, 1, 2]


def test_data_yaml_uses_absolute_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime data yaml points at absolute train and val_monitor image folders."""
    for split in ("train_s5", "val_monitor"):
        (tmp_path / split / "images").mkdir(parents=True)
    monkeypatch.setenv("HW2_PREP_ROOT", str(tmp_path))
    data = yaml.safe_load(ru.write_data_yaml(tmp_path / "run" / "data.yaml", "train_s5").read_text())
    assert data["train"] == str(tmp_path.resolve() / "train_s5" / "images")
    assert data["val"] == str(tmp_path.resolve() / "val_monitor" / "images")
    assert data["names"][7] == "motor" and len(data["names"]) == 8
    with pytest.raises(FileNotFoundError):
        ru.write_data_yaml(tmp_path / "run" / "data.yaml", "missing_variant")
