"""Resolve data, prepared-data, output and results folders for local and Kaggle runs."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
SEARCH_DEPTH = 3


def repo_root() -> Path:
    """Return the repository root, the nearest parent of this file holding pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def is_kaggle() -> bool:
    """Return True when running inside a Kaggle kernel."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or KAGGLE_INPUT.is_dir()


def _env_path(name: str) -> Path | None:
    """Return the path stored in an environment variable, or None when it is unset or empty."""
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


@lru_cache(maxsize=None)
def find_kaggle_input(marker: str) -> Path | None:
    """Return the shallowest folder under /kaggle/input that contains an entry named marker."""
    if not KAGGLE_INPUT.is_dir():
        return None
    for depth in range(1, SEARCH_DEPTH + 1):
        hits = sorted(KAGGLE_INPUT.glob("/".join(["*"] * depth + [marker])))
        if hits:
            return hits[0].parent
    return None


def data_root() -> Path:
    """Return the raw data folder from HW2_DATA_ROOT, the Kaggle input with test_images.csv, or repo data/."""
    found = _env_path("HW2_DATA_ROOT") or find_kaggle_input("test_images.csv")
    return found or repo_root() / "data"


def prep_root() -> Path:
    """Return the prepared data folder from HW2_PREP_ROOT, a Kaggle input with train_s5, or a writable default."""
    found = _env_path("HW2_PREP_ROOT") or find_kaggle_input("train_s5")
    if found:
        return found
    return KAGGLE_WORKING / "prepared" if is_kaggle() else repo_root() / "data" / "prepared"


def out_root() -> Path:
    """Return the run output folder from HW2_OUT_ROOT, Kaggle working, or repo outputs/."""
    found = _env_path("HW2_OUT_ROOT")
    if found:
        return found
    return KAGGLE_WORKING / "outputs" if is_kaggle() else repo_root() / "outputs"


def results_dir() -> Path:
    """Return the committed results folder inside the repository."""
    return repo_root() / "results"


ROOTS: dict[str, Callable[[], Path]] = {
    "repo_root": repo_root,
    "data_root": data_root,
    "prep_root": prep_root,
    "out_root": out_root,
    "results_dir": results_dir,
}


def main() -> None:
    """Print every resolved path and whether it exists."""
    for name, resolve in ROOTS.items():
        path = resolve()
        status = "exists" if path.exists() else "missing"
        print(f"{name:<12} {status:<8} {path}")


if __name__ == "__main__":
    main()
