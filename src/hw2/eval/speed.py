"""Measure detector latency in milliseconds per image on images preloaded into memory."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ImageLike = np.ndarray | str | Path


def preload(images: Sequence[ImageLike]) -> list[np.ndarray]:
    """Read image paths into BGR arrays and pass arrays through unchanged."""
    import cv2

    frames = []
    for image in images:
        frame = image if isinstance(image, np.ndarray) else cv2.imread(str(image), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"cannot read image {image}")
        frames.append(frame)
    return frames


def cuda_sync() -> Callable[[], None]:
    """Return a function that waits for queued CUDA work, or a no-op without a GPU."""
    try:
        import torch
    except ImportError:
        return lambda: None
    return torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)


def benchmark(predict_fn: Callable[[np.ndarray], Any], images: Sequence[ImageLike], warmup: int = 20, n: int = 200) -> float:
    """Return mean milliseconds per image of predict_fn over n timed calls after warmup untimed calls."""
    frames = preload(images)
    if not frames or n < 1:
        raise ValueError("benchmark needs at least one image and n >= 1")
    sync = cuda_sync()
    for i in range(warmup):
        predict_fn(frames[i % len(frames)])
    sync()
    total = 0.0
    for i in range(n):
        start = time.perf_counter()
        predict_fn(frames[i % len(frames)])
        sync()
        total += time.perf_counter() - start
    return 1000.0 * total / n
