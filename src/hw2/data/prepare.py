"""Build train_s5, val_monitor and val_testlike with images, YOLO labels, COCO files and metadata."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from hw2.config import load_config
from hw2.constants import CLASS_NAMES, NUM_CLASSES
from hw2.data import boxes as bx
from hw2.data import coco_utils as cu
from hw2.paths import data_root, prep_root, repo_root
from hw2.utils import Timer, get_logger, load_json, save_json

LOG = get_logger(__name__)
DEFAULT_CONFIG = repo_root() / "configs" / "data.yaml"
JPEG_QUALITY = 95
MAX_IGNORE_FRAC = 0.5
MIN_BOX_PX = 1.0
CSV_COLUMNS = (
    "split", "clip", "frame", "src_file", "out_file", "src_w", "src_h",
    "out_w", "out_h", "n_boxes", "n_dropped", "ignore_frac",
)


@dataclass(frozen=True)
class SplitSpec:
    """Source split, frame step config key and painting mode of one prepared split."""

    source: str
    step_key: str
    painted: bool


SPLITS: dict[str, SplitSpec] = {
    "train_s5": SplitSpec("train", "step", painted=True),
    "val_monitor": SplitSpec("val", "val_monitor_step", painted=True),
    "val_testlike": SplitSpec("val", "val_testlike_step", painted=False),
}


@dataclass(frozen=True)
class Settings:
    """Frame steps and image and box processing settings read from configs/data.yaml."""

    steps: dict[str, int]
    ignore_fill: int
    long_side_cap: int
    testlike_scales: dict[Any, float]
    jpeg_quality: int = JPEG_QUALITY
    max_ignore_frac: float = MAX_IGNORE_FRAC
    min_box_px: float = MIN_BOX_PX

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Settings:
        """Build settings from a loaded data config, using module defaults for optional keys."""
        return cls(
            steps={name: int(cfg[spec.step_key]) for name, spec in SPLITS.items()},
            ignore_fill=int(cfg["ignore_fill"]),
            long_side_cap=int(cfg["long_side_cap"]),
            testlike_scales=dict(cfg["val_testlike_scales"]),
            jpeg_quality=int(cfg.get("jpeg_quality", JPEG_QUALITY)),
            max_ignore_frac=float(cfg.get("max_ignore_frac", MAX_IGNORE_FRAC)),
            min_box_px=float(cfg.get("min_box_px", MIN_BOX_PX)),
        )


@dataclass(frozen=True)
class ClipTask:
    """The kept frames of one clip with their annotations and where to read and write them."""

    split: str
    clip: str
    src_dir: Path
    out_dir: Path
    frames: list[tuple[dict[str, Any], list[dict[str, Any]]]]
    settings: Settings


@dataclass
class FrameResult:
    """One prepared frame as a metadata row plus its COCO image entry and annotations."""

    row: dict[str, Any]
    image: dict[str, Any]
    annotations: list[dict[str, Any]] = field(default_factory=list)


def scale_for_width(width: int, scales: Mapping[Any, float]) -> float:
    """Return the val_testlike scale for an original frame width, falling back to the default entry."""
    return float(scales.get(width, scales.get(str(width), scales.get("default", 1.0))))


def output_size(split: str, settings: Settings, width: int, height: int) -> tuple[int, int]:
    """Return the written frame size for a split given the original frame size."""
    if SPLITS[split].painted:
        return bx.fit_long_side(width, height, settings.long_side_cap)
    return bx.scaled_size(width, height, scale_for_width(width, settings.testlike_scales))


def write_image(
    src: Path, dst: Path, src_size: tuple[int, int], out_size: tuple[int, int], rects: np.ndarray, settings: Settings
) -> None:
    """Paint rectangles grey, downscale with INTER_AREA and save as JPEG, or copy an unchanged frame."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not len(rects) and out_size == src_size:
        shutil.copyfile(src, dst)
        return
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None or (image.shape[1], image.shape[0]) != src_size:
        raise ValueError(f"{src}: unreadable or not {src_size[0]}x{src_size[1]}")
    bx.paint_rects(image, rects, settings.ignore_fill)
    if out_size != src_size:
        image = cv2.resize(image, out_size, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]):
        raise OSError(f"failed to write {dst}")


def write_labels(path: Path, lines: Sequence[str]) -> None:
    """Write YOLO label lines to a file, leaving it empty when there are none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))


def process_frame(task: ClipTask, image: dict[str, Any], anns: list[dict[str, Any]]) -> FrameResult:
    """Prepare one frame of a clip task and return its metadata, COCO image and annotations."""
    settings, painted = task.settings, SPLITS[task.split].painted
    src_size = (int(image["width"]), int(image["height"]))
    out_w, out_h = out_size = output_size(task.split, settings, *src_size)
    frame = cu.split_annotations(anns)
    rects = bx.nonempty_rects(bx.pixel_rects(bx.xywh_to_xyxy(frame.ignore), *src_size))
    mask = bx.region_mask(rects, *src_size) if len(rects) else None
    cleaned = bx.clean_boxes(
        bx.xywh_to_xyxy(frame.boxes), src_size, out_size,
        mask if painted else None, settings.max_ignore_frac, settings.min_box_px,
    )
    classes = frame.classes[cleaned.keep]
    name = Path(image["file_name"])
    out_file = f"{task.split}/images/{task.clip}/{name.name}"
    paint = rects if painted else rects[:0]
    write_image(task.src_dir / name, task.out_dir / out_file, src_size, out_size, paint, settings)
    label_file = task.out_dir / task.split / "labels" / task.clip / f"{name.stem}.txt"
    write_labels(label_file, bx.yolo_lines(classes, cleaned.boxes, out_w, out_h))
    if painted:
        kept = zip((frame.objects[i] for i in cleaned.keep), classes, bx.xyxy_to_xywh(cleaned.boxes))
        annotations = [cu.box_annotation(obj, cls, box) for obj, cls, box in kept]
    else:
        sx, sy = out_w / src_size[0], out_h / src_size[1]
        annotations = [cu.scale_annotation(a, sx, sy) for a in anns]
    row = {
        "split": task.split, "clip": task.clip, "frame": int(image["frame_id"]),
        "src_file": f"{task.src_dir.name}/{name.as_posix()}", "out_file": out_file,
        "src_w": src_size[0], "src_h": src_size[1], "out_w": out_w, "out_h": out_h,
        "n_boxes": len(cleaned.keep), "n_dropped": cleaned.n_dropped,
        "ignore_frac": round(float(mask.mean()), 6) if mask is not None else 0.0,
        "n_dropped_ignore": cleaned.n_dropped_ignore, "n_dropped_small": cleaned.n_dropped_small,
        "n_clipped": cleaned.n_clipped, "n_ignore": len(frame.ignore),
        "class_counts": np.bincount(classes, minlength=NUM_CLASSES).tolist(),
    }
    coco_image = {"file_name": out_file, "width": out_w, "height": out_h, "clip": task.clip, "frame": row["frame"]}
    return FrameResult(row, coco_image, annotations)


def init_worker() -> None:
    """Keep OpenCV single-threaded inside pool workers."""
    cv2.setNumThreads(1)


def process_clip(task: ClipTask) -> list[FrameResult]:
    """Prepare every kept frame of one clip in frame order."""
    return [process_frame(task, image, anns) for image, anns in task.frames]


def build_tasks(
    splits: Sequence[str], settings: Settings, data_dir: Path, out_dir: Path, max_clips: int | None = None
) -> list[ClipTask]:
    """Load each needed source COCO file once and return clip tasks ordered by split and clip."""
    tasks: list[ClipTask] = []
    for source in sorted({SPLITS[s].source for s in splits}):
        coco = cu.load_coco(data_dir / source / "annotations_coco.json")
        clips = cu.frames_by_clip(coco["images"])
        names = sorted(clips)[:max_clips]
        chosen = {
            split: {clip: clips[clip][:: settings.steps[split]] for clip in names}
            for split in splits
            if SPLITS[split].source == source
        }
        wanted = {im["id"] for per_clip in chosen.values() for frames in per_clip.values() for im in frames}
        by_image = cu.annotations_by_image(coco["annotations"], wanted)
        del coco
        for split, per_clip in chosen.items():
            for clip, frames in per_clip.items():
                pairs = [(im, by_image.get(im["id"], [])) for im in frames]
                tasks.append(ClipTask(split, clip, data_dir / source, out_dir, pairs, settings))
    order = list(SPLITS)
    return sorted(tasks, key=lambda t: (order.index(t.split), t.clip))


def run_tasks(tasks: Sequence[ClipTask], workers: int) -> list[list[FrameResult]]:
    """Run clip tasks serially or in a process pool, returning results in task order."""
    progress = {"total": len(tasks), "desc": "clips", "mininterval": 5}
    if workers <= 1:
        return [process_clip(task) for task in tqdm(tasks, **progress)]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        return list(tqdm(pool.map(process_clip, tasks), **progress))


def split_stats(frames: Sequence[FrameResult]) -> dict[str, Any]:
    """Return image, box, drop, clip and per-class box counts for one split."""
    rows = [f.row for f in frames]
    counts = np.sum([r["class_counts"] for r in rows], axis=0) if rows else np.zeros(NUM_CLASSES, int)
    totals = {
        "boxes": "n_boxes", "dropped": "n_dropped", "dropped_ignore": "n_dropped_ignore",
        "dropped_small": "n_dropped_small", "clipped": "n_clipped", "ignore_regions": "n_ignore",
    }
    stats: dict[str, Any] = {"clips": len({r["clip"] for r in rows}), "images": len(rows)}
    stats.update({key: int(sum(r[col] for r in rows)) for key, col in totals.items()})
    stats["boxes_per_class"] = {name: int(n) for name, n in zip(CLASS_NAMES, counts)}
    return stats


def split_info(split: str, settings: Settings) -> dict[str, Any]:
    """Return the COCO info block describing how a split was prepared."""
    spec = SPLITS[split]
    info = {"description": f"VisDrone2019-VID {split}", "source": spec.source, "step": settings.steps[split]}
    if spec.painted:
        info.update(ignore_painted=settings.ignore_fill, long_side_cap=settings.long_side_cap)
    else:
        info["scales"] = settings.testlike_scales
    return info


def write_data_yaml(out_dir: Path) -> Path:
    """Write the Ultralytics dataset yaml for train_s5 with val_monitor validation."""
    names = "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    text = (
        "# path omitted: relative to yaml\n"
        "train: train_s5/images\n"
        "val: val_monitor/images\n"
        f"nc: {NUM_CLASSES}\n"
        f"names:\n{names}"
    )
    path = out_dir / "data_s5.yaml"
    path.write_text(text)
    return path


def write_frames_csv(path: Path, rows: Iterable[Mapping[str, Any]], splits: Sequence[str]) -> Path:
    """Write meta/frames.csv, keeping existing rows of splits that were not rebuilt."""
    kept: list[Mapping[str, Any]] = []
    if path.exists():
        with path.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["split"] not in splits]
    order = list(SPLITS)
    merged = sorted([*kept, *rows], key=lambda r: (order.index(r["split"]), r["clip"], int(r["frame"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    return path


def write_summary(path: Path, stats: Mapping[str, dict[str, Any]], settings: Settings) -> dict[str, Any]:
    """Write meta/summary.json, keeping existing entries of splits that were not rebuilt."""
    summary = load_json(path) if path.exists() else {}
    summary["settings"] = asdict(settings)
    summary["splits"] = {**summary.get("splits", {}), **stats}
    summary["splits"] = {s: summary["splits"][s] for s in SPLITS if s in summary["splits"]}
    save_json(summary, path)
    return summary


def prepare(
    splits: Sequence[str],
    out_dir: Path,
    cfg: Mapping[str, Any],
    workers: int = 4,
    max_clips: int | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Prepare the requested splits under out_dir and return the summary dict."""
    settings = Settings.from_config(cfg)
    data_dir = data_dir or data_root()
    splits = [s for s in SPLITS if s in splits]
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        shutil.rmtree(out_dir / split, ignore_errors=True)
    tasks = build_tasks(splits, settings, data_dir, out_dir, max_clips)
    per_split: dict[str, list[FrameResult]] = {split: [] for split in splits}
    for task, frames in zip(tasks, run_tasks(tasks, workers)):
        per_split[task.split].extend(frames)
    for split, frames in per_split.items():
        coco = cu.build_coco(((f.image, f.annotations) for f in frames), split_info(split, settings))
        cu.save_coco(coco, out_dir / split / "annotations_coco.json")
    write_data_yaml(out_dir)
    write_frames_csv(out_dir / "meta" / "frames.csv", [f.row for fs in per_split.values() for f in fs], splits)
    stats = {split: split_stats(frames) for split, frames in per_split.items()}
    for split, s in stats.items():
        LOG.info("%s: %d images, %d boxes, %d dropped, %d clipped",
                 split, s["images"], s["boxes"], s["dropped"], s["clipped"])
    return write_summary(out_dir / "meta" / "summary.json", stats, settings)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--clips", type=int, default=None, help="only the first N clips per split")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="output folder, default prep_root()")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run data preparation from the command line."""
    args = parse_args(argv)
    out_dir = (args.out or prep_root()).resolve()
    LOG.info("data %s -> %s, splits %s, workers %d", data_root(), out_dir, " ".join(args.splits), args.workers)
    with Timer("prepare", LOG):
        prepare(args.splits, out_dir, load_config(args.config), args.workers, args.clips)


if __name__ == "__main__":
    main()
