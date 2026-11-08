"""Shared helpers for seeding, logging, git metadata, JSON files, timing and environment info."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from hw2.constants import SEED
from hw2.paths import repo_root

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def seed_everything(seed: int = SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy and, when installed, PyTorch random generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "hw2", level: int | None = None) -> logging.Logger:
    """Return a logger in the hw2 namespace that shares one stream handler."""
    root = logging.getLogger("hw2")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    logger = logging.getLogger(name if name.split(".")[0] == "hw2" else f"hw2.{name}")
    if level is not None:
        logger.setLevel(level)
    return logger


def git_commit() -> str | None:
    """Return the HEAD commit hash of the repository, or None when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _json_default(obj: Any) -> Any:
    """Convert NumPy values and paths into JSON-serialisable Python objects."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write an object as indented JSON, creating parent folders, and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=_json_default) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    """Read and return the JSON content of a file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Timer:
    """Context manager that measures wall time in seconds and logs it when named."""

    def __init__(self, name: str | None = None, logger: logging.Logger | None = None) -> None:
        """Store the optional label and logger used on exit."""
        self.name = name
        self.logger = logger
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        """Start the clock and return the timer."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the clock and log the elapsed time when named."""
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            (self.logger or get_logger()).info("%s took %.1f s", self.name, self.elapsed)


def _package_version(name: str) -> str | None:
    """Return the installed version of a package, or None when it is missing."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _device_name() -> str:
    """Return the CUDA device name, mps or cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def env_info() -> dict[str, str | None]:
    """Return Python, torch, torchvision and ultralytics versions plus the compute device name."""
    info: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "ultralytics"):
        info[package] = _package_version(package)
    info["device"] = _device_name()
    return info
