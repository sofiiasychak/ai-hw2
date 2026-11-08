"""Tests for YAML config loading with base inheritance, overrides and experiment ids."""

from pathlib import Path

import pytest

from hw2.config import load_config


def _write(path: Path, text: str) -> Path:
    """Write a YAML file, creating parent folders, and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_base_deep_merge_child_wins(tmp_path: Path) -> None:
    """Child keys override base keys while untouched nested base keys survive."""
    _write(tmp_path / "A0_base.yaml", "model: {name: yolo11s, imgsz: 1280}\ntrain: {epochs: 10, lr: 0.01}\n")
    child = _write(tmp_path / "A1_child.yaml", "base: A0_base.yaml\ntrain: {epochs: 20}\nextra: [1, 2]\n")
    cfg = load_config(child)
    assert cfg["model"] == {"name": "yolo11s", "imgsz": 1280}
    assert cfg["train"] == {"epochs": 20, "lr": 0.01}
    assert cfg["extra"] == [1, 2]
    assert "base" not in cfg


def test_recursive_base_resolves_relative_to_each_file(tmp_path: Path) -> None:
    """Bases chain recursively and each base path is relative to the file naming it."""
    _write(tmp_path / "common" / "root.yaml", "a: 1\nb: {x: 1, y: 1, z: 1}\n")
    _write(tmp_path / "family.yaml", "base: common/root.yaml\nb: {y: 2}\nc: 2\n")
    child = _write(tmp_path / "yolo" / "A2_deep.yaml", "base: ../family.yaml\nb: {x: 3}\n")
    cfg = load_config(child)
    assert cfg["a"] == 1
    assert cfg["b"] == {"x": 3, "y": 2, "z": 1}
    assert cfg["c"] == 2


def test_list_of_bases_merges_in_order(tmp_path: Path) -> None:
    """A list of bases merges left to right before the child is applied."""
    _write(tmp_path / "one.yaml", "a: 1\nb: 1\n")
    _write(tmp_path / "two.yaml", "b: 2\nc: 2\n")
    child = _write(tmp_path / "D1_multi.yaml", "base: [one.yaml, two.yaml]\nc: 3\n")
    assert load_config(child) == {"a": 1, "b": 2, "c": 3, "exp_id": "D1", "exp_name": "D1_multi"}


def test_base_cycle_raises(tmp_path: Path) -> None:
    """Configs that inherit from each other raise a clear error."""
    _write(tmp_path / "x.yaml", "base: y.yaml\n")
    _write(tmp_path / "y.yaml", "base: x.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "x.yaml")


def test_overrides_are_parsed_as_yaml(tmp_path: Path) -> None:
    """Dotted overrides set nested values with YAML types and create missing keys."""
    child = _write(tmp_path / "A3_ovr.yaml", "train: {epochs: 10, lr: 0.01}\n")
    overrides = ["train.epochs=5", "train.lr=1e-3", "model.name=yolo11m.pt", "aug.flip=false", "classes=[0, 2]"]
    cfg = load_config(child, overrides)
    assert cfg["train"] == {"epochs": 5, "lr": 0.001}
    assert cfg["model"] == {"name": "yolo11m.pt"}
    assert cfg["aug"] == {"flip": False}
    assert cfg["classes"] == [0, 2]


def test_override_wins_over_base_and_child(tmp_path: Path) -> None:
    """Overrides apply after inheritance so they beat both base and child values."""
    _write(tmp_path / "base.yaml", "train: {epochs: 10}\n")
    child = _write(tmp_path / "A4_x.yaml", "base: base.yaml\ntrain: {epochs: 20}\n")
    assert load_config(child, ["train.epochs=30"])["train"]["epochs"] == 30


@pytest.mark.parametrize("bad", ["noequals", "=5", "a..b=1", "train.epochs.x=1"])
def test_bad_override_raises(tmp_path: Path, bad: str) -> None:
    """Malformed overrides and overrides through non-mappings raise ValueError."""
    child = _write(tmp_path / "A5_bad.yaml", "train: {epochs: 10}\n")
    with pytest.raises(ValueError):
        load_config(child, [bad])


def test_exp_id_and_name_come_from_child_filename(tmp_path: Path) -> None:
    """The exp_id is the stem before the first underscore and exp_name is the full stem."""
    _write(tmp_path / "Q0_base.yaml", "exp_id: wrong\nmodel: rtdetr-l.pt\n")
    child = _write(tmp_path / "Q12_rtdetr_l_long.yaml", "base: Q0_base.yaml\n")
    cfg = load_config(child)
    assert cfg["exp_id"] == "Q12"
    assert cfg["exp_name"] == "Q12_rtdetr_l_long"


def test_exp_id_without_underscore(tmp_path: Path) -> None:
    """A stem with no underscore is used whole as the exp_id."""
    cfg = load_config(_write(tmp_path / "M7.yaml", ""))
    assert cfg == {"exp_id": "M7", "exp_name": "M7"}
