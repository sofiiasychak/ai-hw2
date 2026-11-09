"""Tests for finding the raw and prepared data inside the nested Kaggle input layout."""

from pathlib import Path

import pytest

from hw2 import paths

ENV_VARS = ("HW2_DATA_ROOT", "HW2_PREP_ROOT")


def _touch(path: Path) -> Path:
    """Create an empty file with its parent folders and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_nested_kaggle_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Competition and dataset mounts are found deep down while image, label and ignore folders are skipped."""
    comp = tmp_path / "competitions" / "ucu-ai-2026-hw2"
    prep = tmp_path / "datasets" / "someone" / "hw2-prepared-s5" / "versions" / "3" / "prepared"
    _touch(comp / "test_images.csv")
    _touch(comp / "train" / "images" / "clip_001" / "0000001.jpg")
    _touch(prep / "train_s5" / "labels" / "clip_001" / "0000001.txt")
    _touch(tmp_path / "images" / "test_images.csv")  # shallower decoy, pruned
    (tmp_path / "labels" / "train_s5").mkdir(parents=True)
    (tmp_path / "ignore" / "train_s5").mkdir(parents=True)
    monkeypatch.setattr(paths, "KAGGLE_INPUT", tmp_path)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert paths.data_root() == comp
    assert paths.prep_root() == prep
    monkeypatch.setenv("HW2_DATA_ROOT", str(tmp_path / "mine"))
    monkeypatch.setenv("HW2_PREP_ROOT", str(tmp_path / "mine_prep"))
    assert paths.data_root() == tmp_path / "mine"
    assert paths.prep_root() == tmp_path / "mine_prep"


def test_search_stops_at_seven_folders(tmp_path: Path) -> None:
    """A marker eight folders below the root is out of reach and seven folders below is found."""
    deep = tmp_path.joinpath(*(f"d{i}" for i in range(8)))
    _touch(deep / "test_images.csv")
    assert paths.find_dir_with(tmp_path, "test_images.csv") is None
    assert paths.find_dir_with(tmp_path / "d0", "test_images.csv") == deep


def test_shallowest_match_wins(tmp_path: Path) -> None:
    """When several folders hold the marker the shallowest one is returned."""
    (tmp_path / "a" / "b" / "c" / "train_s5").mkdir(parents=True)
    (tmp_path / "z" / "train_s5").mkdir(parents=True)
    assert paths.find_dir_with(tmp_path, "train_s5") == tmp_path / "z"
