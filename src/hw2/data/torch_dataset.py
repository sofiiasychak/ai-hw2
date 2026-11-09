"""Torch datasets of prepared COCO splits for torchvision detectors and of plain image files for prediction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from hw2.data import boxes as bx
from hw2.data import coco_utils as cu

LABEL_OFFSET = 1  # label 0 is background
JITTER = 0.1


def read_rgb(path: str | Path) -> torch.Tensor:
    """Read an image file as a uint8 RGB tensor of shape (3, H, W)."""
    pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if pixels is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return torch.from_numpy(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous()


def build_transforms(augment: bool, color_jitter: float = 0.0) -> v2.Compose:
    """Return flip and optional light colour jitter when augmenting, then float conversion and box sanitising."""
    steps: list[v2.Transform] = []
    if augment:
        steps.append(v2.RandomHorizontalFlip(0.5))
        if color_jitter:
            steps.append(v2.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter))
    steps += [v2.ToDtype(torch.float32, scale=True), v2.SanitizeBoundingBoxes(), v2.ToPureTensor()]
    return v2.Compose(steps)


def collate(batch: Sequence[tuple[Any, ...]]) -> tuple[tuple[Any, ...], ...]:
    """Group a batch of samples into one tuple per field, keeping variable-size images apart."""
    return tuple(zip(*batch))


class CocoDetectionDataset(Dataset):
    """Prepared split images as float tensors in [0, 1] with xyxy boxes and labels shifted by LABEL_OFFSET."""

    def __init__(self, prep_root: str | Path, split: str = "train_s5", augment: bool = True, color_jitter: float = 0.0):
        """Read the split's annotations_coco.json and set up the transforms."""
        self.root = Path(prep_root)
        self.split = split
        self.coco = cu.load_coco(self.root / split / "annotations_coco.json")
        self.images: list[dict[str, Any]] = self.coco["images"]
        self.by_image = cu.annotations_by_image(self.coco["annotations"])
        self.transforms = build_transforms(augment, color_jitter)

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return one transformed image and its target with boxes, labels and image_id."""
        image = self.images[index]
        frame = cu.split_annotations(self.by_image.get(image["id"], []))
        pixels = tv_tensors.Image(read_rgb(self.root / image["file_name"]))
        boxes = tv_tensors.BoundingBoxes(
            torch.as_tensor(bx.xywh_to_xyxy(frame.boxes), dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=tuple(pixels.shape[-2:]),
        )
        labels = torch.as_tensor(frame.classes + LABEL_OFFSET, dtype=torch.int64)
        return self.transforms(pixels, {"boxes": boxes, "labels": labels, "image_id": int(image["id"])})

    def items(self, indices: Sequence[int]) -> list[tuple[str, Path]]:
        """Return (file_name, absolute path) pairs of the given images."""
        return [(self.images[i]["file_name"], (self.root / self.images[i]["file_name"]).resolve()) for i in indices]

    def coco_subset(self, indices: Sequence[int]) -> dict[str, Any]:
        """Return a COCO dict holding only the given images and their annotations."""
        images = [self.images[i] for i in indices]
        anns = [ann for image in images for ann in self.by_image.get(image["id"], [])]
        return {"images": images, "annotations": anns, "categories": self.coco["categories"]}


class ImageFileDataset(Dataset):
    """Image files read one at a time as uint8 RGB tensors with their file names."""

    def __init__(self, items: Sequence[tuple[str, Path]]):
        """Store (file_name, path) pairs."""
        self.items = list(items)

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        """Return a file name and its uint8 RGB tensor."""
        name, path = self.items[index]
        return name, read_rgb(path)
