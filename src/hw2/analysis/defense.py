"""Tables, figure grids and commands for the defense overview notebook, built only from saved results."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Code
from matplotlib.figure import Figure

import hw2.infer
from hw2.analysis import failures, runs
from hw2.config import find_config, load_config
from hw2.constants import CLASS_NAMES, MAX_DETS, SCORE_THRESH, SEED
from hw2.data import boxes as bx
from hw2.data import coco_utils as cu
from hw2.data.prepare import Settings
from hw2.eval.preds import load_preds
from hw2.paths import data_root, out_root, repo_root, results_dir
from hw2.train.common import gt_json, preds_path

FAMILY_DIRS: dict[str, tuple[str, ...]] = {
    "frcnn": ("frcnn",),
    "retinanet": ("retinanet",),
    "yolo": ("yolo",),
    "rtdetr": ("rtdetr",),
    "pipeline": ("inference", "ensemble", "temporal"),
}
RUNNERS: dict[str, tuple[str, str]] = {
    "yolo": ("hw2.train.run_ultralytics", "weights/last.pt"),
    "rtdetr": ("hw2.train.run_ultralytics", "weights/last.pt"),
    "frcnn": ("hw2.train.run_torchvision", "last.pt"),
    "retinanet": ("hw2.train.run_torchvision", "last.pt"),
}
SPLITS: tuple[tuple[str, int, int, str, str], ...] = (
    ("train", 56, 24201, "up to 3840x2160", "train_s5 (training)"),
    ("val", 7, 2846, "up to 3840x2160", "val_monitor, official val, val_testlike"),
    ("test", 48, 3811, "680x382 to 1920x1080", "Kaggle submission"),
)
EDA_FIGURES: tuple[str, ...] = ("eda_classes", "eda_resolutions", "eda_sizes_per_class", "eda_objects_per_frame")
PROTOCOL_FIGURES: tuple[str, ...] = ("eda_track_iou", "eda_sizes_1280", "eda_ignore", "eda_overlays")
FRCNN_FIGURES: tuple[str, ...] = ("anchor_coverage", "frcnn_rpn_recall", "F0_tide", "F0_fail_1", "F0_fail_2", "F0_fail_3")
RETINA_FIGURES: tuple[str, ...] = ("R0_tide", "R0_fail_1", "R0_fail_2", "R0_fail_3")
TABLE_COLS: list[str] = ["name", "mAP", "AP50", "APs", "APm", "APl", "AR500", "testlike mAP", "train h", "epochs", "ms/img", "LB"]
SUMMARY_COLS: dict[str, str] = {"name": "name", "mAP": "val mAP", "AP50": "val AP50", "APs": "val APs", "ms/img": "ms/img", "LB": "LB"}
DECIMALS: dict[str, int] = {"train h": 2, "epochs": 1, "ms/img": 1}
TEXT_COLS: tuple[str, ...] = ("name", "final")
ULTRA_MAP = "metrics/mAP50-95(B)"
SMOKE_SUFFIX = "_smoke"
PROTOCOL_SIDE = 1280
BUDGET_HOURS = 2.5
GRID_DPI = 72
GRID_WIDTH = 12.0
DEMO_FRAMES: tuple[int, ...] = (150, 900)
FINAL_MARK = "✓"
SIZE_EDGES: tuple[float, ...] = (0.0, 32.0, 96.0, np.inf)
SIZE_LABELS: tuple[str, ...] = ("small", "medium", "large")
ENSEMBLE_NAMES = re.compile(r"wbf|soft_?nms|weighted_box", re.IGNORECASE)
TOY_CLIP = "toy_clip"
TOY_FRAMES = 6


def note(message: str) -> None:
    """Print one short status line in the notebook."""
    print(message)


def all_results() -> dict[str, dict[str, Any]]:
    """Return every committed results json by experiment id, leaving out smoke runs."""
    return {r["exp_id"]: r for r in runs.load_results([""]) if not r["exp_id"].endswith(SMOKE_SUFFIX)}


def config_path(exp_id: str) -> Path:
    """Return the config of an experiment id, also accepting a bare <exp_id>.yaml such as FINAL.yaml."""
    exact = sorted((repo_root() / "configs").glob(f"*/{exp_id}.yaml"))
    return exact[0] if len(exact) == 1 else find_config(exp_id)


def exp_name(exp_id: str) -> str:
    """Return the config name of an experiment id, or the id itself when it has no config."""
    try:
        return load_config(config_path(exp_id))["exp_name"]
    except FileNotFoundError:
        return exp_id


def family_ids(family: str) -> list[str]:
    """Return the experiment ids of every config in the folders of one family."""
    folders = [repo_root() / "configs" / name for name in FAMILY_DIRS[family]]
    return [load_config(path)["exp_id"] for folder in folders for path in sorted(folder.glob("*.yaml"))]


def lb_row(exp_id: str, lb: dict[str, float]) -> dict[str, Any]:
    """Return a table row holding only the name and public score of a run without a results json."""
    return {"exp": exp_id, "name": exp_name(exp_id), "LB": lb[exp_id]}


def tidy(table: pd.DataFrame) -> pd.DataFrame:
    """Make every non-text column numeric, rounding metrics to four decimals and cost columns to fewer."""
    table = table.astype({col: float for col in table.columns if col not in TEXT_COLS})
    return table.round({**{col: 4 for col in table.columns}, **DECIMALS})


def experiment_table(family: str) -> pd.DataFrame | None:
    """Return one row per run of a family, public-score-only rows for runs without a json, or None when nothing exists."""
    results, lb = all_results(), runs.lb_scores()
    ids = family_ids(family)
    missing = [exp_id for exp_id in ids if exp_id not in results]
    if missing:
        note(f"{family}: no results json for {', '.join(missing)}")
    rows = [runs.result_row(results[i], lb) if i in results else lb_row(i, lb) for i in ids if i in results or i in lb]
    if not rows:
        note(f"{family}: nothing to show, section skipped")
        return None
    return tidy(pd.DataFrame(rows).set_index("exp").reindex(columns=TABLE_COLS))


def summary_table(finals: Sequence[str]) -> pd.DataFrame:
    """Return every run and public-score-only entry sorted by val mAP, with the final submissions marked."""
    results, lb = all_results(), runs.lb_scores()
    rows = [runs.result_row(r, lb) for r in results.values()]
    rows += [lb_row(exp_id, lb) for exp_id in lb if exp_id not in results]
    table = pd.DataFrame(rows).set_index("exp").reindex(columns=list(SUMMARY_COLS)).rename(columns=SUMMARY_COLS)
    table["final"] = [FINAL_MARK if exp_id in finals else "" for exp_id in table.index]
    absent = [exp_id for exp_id in finals if exp_id not in table.index]
    if absent:
        note(f"final submissions without results yet: {', '.join(absent)}")
    return tidy(table.sort_values(["val mAP", "LB"], ascending=False, na_position="last"))


def validation_table() -> pd.DataFrame | None:
    """Return val mAP, val_testlike mAP, their gap and the public score of every run with a results json, sorted by val mAP."""
    results, lb = all_results(), runs.lb_scores()
    if not results:
        note("no results json found, section skipped")
        return None
    full = pd.DataFrame([runs.result_row(r, lb) for r in results.values()]).set_index("exp")
    table = pd.DataFrame({"val mAP": full["mAP"], "testlike mAP": full["testlike mAP"], "LB": full["LB"]})
    table.insert(2, "gap", table["testlike mAP"] - table["val mAP"])
    return tidy(table.sort_values("val mAP", ascending=False))


def class_table() -> pd.DataFrame:
    """Return the eight class names by class id as one row."""
    return pd.DataFrame({"class": CLASS_NAMES}).rename_axis("id").T


def split_table() -> pd.DataFrame:
    """Return clips, frames, image sizes and use of the train, val and test splits."""
    return pd.DataFrame(SPLITS, columns=["split", "clips", "frames", "image size", "used for"]).set_index("split")


def protocol_table() -> pd.DataFrame:
    """Return the preprocessing and validation protocol as rows built from configs/data.yaml and project constants."""
    data = load_config(repo_root() / "configs" / "data.yaml")
    scales = ", ".join(f"{width}: x{scale}" for width, scale in data["val_testlike_scales"].items())
    splits = {name: (clips, frames) for name, clips, frames, *_ in SPLITS}
    rows = {
        "split": f"by sequence: train {splits['train'][0]} clips, val {splits['val'][0]} other clips",
        "train_s5": f"1 in {data['step']} train frames, ignore regions grey {data['ignore_fill']}, boxes clipped, long side <= {data['long_side_cap']}",
        "val_monitor": f"1 in {data['val_monitor_step']} val frames, prepared like train_s5, checked during training",
        "val (official)": f"all {splits['val'][1]:,} original val frames, ignore regions as iscrowd=1",
        "val_testlike": f"1 in {data['val_testlike_step']} val frames downscaled to test sizes (by width {scales})",
        "metric": f"COCO mAP@[.5:.95], at most {MAX_DETS} boxes per image",
        "runs": f"long side {PROTOCOL_SIDE}, {BUDGET_HOURS} h on one T4, last.pt, score >= {SCORE_THRESH}, seed {SEED}",
    }
    return pd.DataFrame({"protocol": rows})


def coverage(family: str) -> pd.DataFrame | None:
    """Return the saved anchor coverage table of one family, or None when the analysis is missing."""
    if not (results_dir() / runs.ANALYSIS_DIR / "anchors.json").is_file():
        note("anchor coverage: results/analysis/anchors.json missing, skipped")
        return None
    return runs.coverage_table(family).round(1)


def figure_grid(names: Sequence[str], cols: int = 2) -> Figure | None:
    """Show the existing images/<name>.png files in one grid, noting missing ones, or return None when none exist."""
    paths = {name: repo_root() / "images" / f"{name}.png" for name in names}
    found = [name for name, path in paths.items() if path.is_file()]
    missing = [name for name in names if name not in found]
    if missing:
        note(f"figures not found: {', '.join(missing)}")
    if not found:
        return None
    images = [mpimg.imread(paths[name]) for name in found]
    cols = min(cols, len(found))
    rows = -(-len(found) // cols)
    aspect = max(image.shape[0] / image.shape[1] for image in images)
    size = (GRID_WIDTH, GRID_WIDTH / cols * aspect * rows + 0.3 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=size, dpi=GRID_DPI, squeeze=False)
    for ax, name, image in zip(axes.flat, found, images):
        ax.imshow(image)
        ax.set_title(name, fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    return fig


def load_curve(exp_id: str) -> pd.DataFrame | None:
    """Return training hours and val_monitor mAP of a saved torchvision or Ultralytics curve, or None without one."""
    path = results_dir() / "curves" / f"{exp_id}.csv"
    if not path.is_file():
        return None
    curve = pd.read_csv(path)
    if {"hours", "map"} <= set(curve.columns):
        return curve[["hours", "map"]]
    return pd.DataFrame({"hours": curve["time"] / 3600, "map": curve[ULTRA_MAP]})


def map_vs_hours(family: str, reference: str) -> Figure | None:
    """Plot val_monitor mAP against training hours for one family against a reference family, or return None without curves."""
    curves = {fam: {i: c for i in family_ids(fam) if (c := load_curve(i)) is not None} for fam in (family, reference)}
    if not curves[family]:
        note(f"{family}: no curves in results/curves, comparison with {reference} skipped")
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for fam, style in ((family, "-"), (reference, "--")):
        for exp_id, curve in curves[fam].items():
            ax.plot(curve["hours"], curve["map"], style, marker="o", ms=2, label=exp_id)
    ax.set(xlabel="training hours", ylabel="val_monitor mAP (logged)", title=f"{family} vs {reference}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def demo_figure(exp_ids: Sequence[str]) -> Figure | None:
    """Draw ground truth and confident predictions on two val frames of the first run with a local preds_val.parquet, else return None quietly."""
    gt = gt_json("val")
    parquet = next((p for p in (preds_path(out_root() / e, "val") for e in exp_ids) if p.is_file()), None)
    if parquet is None or not gt.is_file():
        return None
    frames = islice(failures.iter_frames(load_preds(parquet), gt), max(DEMO_FRAMES) + 1)
    picked = [frame for index, frame in enumerate(frames) if index in DEMO_FRAMES]
    fig, axes = plt.subplots(1, len(picked), figsize=(GRID_WIDTH, 3.8), dpi=GRID_DPI, squeeze=False)
    for ax, frame in zip(axes.flat, picked):
        title = f"{parquet.parent.name} | {frame.key[0]}/{frame.key[1]:07d}"
        ax.imshow(failures.draw_failure(frame, title)[..., ::-1])
        ax.axis("off")
    fig.tight_layout()
    return fig


def reproduce_commands(exp_ids: Sequence[str]) -> str:
    """Return the shell commands that prepare the data, then train, predict and submit each experiment."""
    lanes = " ".join(f'"{exp_id}"' for exp_id in exp_ids)
    lines = [
        "# setup and data, once",
        "pip install -e .",
        "OUT_ROOT=\"$(python -c 'from hw2.paths import out_root; print(out_root())')\"",
        "python -m hw2.data.prepare",
        "python -m hw2.data.check_prepared",
        "# train, predict and submit: one T4 lane per run",
        f"bash scripts/run_lanes.sh {lanes}",
    ]
    for exp_id in exp_ids:
        cfg = config_path(exp_id)
        module, weights = RUNNERS[load_config(cfg)["family"]]
        base = f"python -m {module} --config {cfg.relative_to(repo_root()).as_posix()} --device 0"
        run = f"$OUT_ROOT/{exp_id}"
        lines += [
            f"# {exp_id} step by step: train, predict from last.pt, submit",
            base,
            f'{base} --skip-train --weights "{run}/{weights}"',
            f'python -m hw2.submission --preds "{run}/preds_test.parquet" --out "{run}/submission.csv.gz"',
            f'python -m hw2.submission --validate "{run}/submission.csv.gz"',
        ]
    return "\n".join(lines)


def show_source(obj: Any) -> Code:
    """Return the real source of a function or class as a highlighted code block."""
    return Code(inspect.getsource(obj), language="python")


def load_val_coco() -> dict[str, Any] | None:
    """Return the original val COCO annotations, or None with one line when data/val is missing."""
    path = gt_json("val")
    if not path.is_file():
        note(f"{path} missing, data demos skipped")
        return None
    return cu.load_coco(path)


def clip_of(image: dict[str, Any]) -> str:
    """Return the clip folder name of a COCO image."""
    return Path(image["file_name"]).parent.name


def object_table(coco: dict[str, Any]) -> pd.DataFrame:
    """Return one row per labelled object with image id, clip, class name, width, height and size bucket."""
    clips = {image["id"]: clip_of(image) for image in coco["images"]}
    anns = [ann for ann in coco["annotations"] if not ann.get("iscrowd")]
    table = pd.DataFrame({
        "image_id": [ann["image_id"] for ann in anns],
        "clip": [clips[ann["image_id"]] for ann in anns],
        "class": [CLASS_NAMES[ann["category_id"] - 1] for ann in anns],
        "w": [ann["bbox"][2] for ann in anns],
        "h": [ann["bbox"][3] for ann in anns],
    })
    table["side"] = np.sqrt(table["w"] * table["h"])
    table["size"] = pd.cut(table["side"], SIZE_EDGES, labels=SIZE_LABELS, right=False)
    return table


def size_table(objects: pd.DataFrame) -> pd.DataFrame:
    """Return object count, median side and COCO size shares (%) per class."""
    shares = pd.crosstab(objects["class"], objects["size"], normalize="index") * 100
    table = objects.groupby("class").agg(objects=("side", "size"), median_side=("side", "median")).join(shares)
    return table.reindex(CLASS_NAMES).round(1)


def per_frame_table(coco: dict[str, Any], objects: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics of labelled objects per frame, counting empty frames as zero."""
    counts = objects.groupby("image_id").size().reindex([image["id"] for image in coco["images"]], fill_value=0)
    return counts.describe().round(1).to_frame("objects per frame").T


def plot_sizes(objects: pd.DataFrame) -> Figure:
    """Plot the histogram of object sides sqrt(w*h) with the COCO small and medium edges."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(objects["side"], bins=np.logspace(0, 3, 60), color="tab:blue")
    for edge in SIZE_EDGES[1:3]:
        ax.axvline(edge, color="k", ls="--", lw=1)
    ax.set(xscale="log", xlabel="object side sqrt(w*h), original px", ylabel="objects")
    fig.tight_layout()
    return fig


def sizes_of_test_set() -> pd.DataFrame | None:
    """Return the frame-size counts of the test images, or None when test_images.csv is missing."""
    path = data_root() / "test_images.csv"
    if not path.is_file():
        note(f"{path} missing, skipped")
        return None
    test = pd.read_csv(path)
    return test.groupby(["width", "height"]).agg(clips=("clip", "nunique"), frames=("image_id", "size")).sort_values("frames", ascending=False)


def frame_with_ignore(coco: dict[str, Any]) -> int:
    """Return the index of the first val frame that has ignore regions."""
    crowd = {ann["image_id"] for ann in coco["annotations"] if ann.get("iscrowd")}
    return next(index for index, image in enumerate(coco["images"]) if image["id"] in crowd)


def frame_boxes(coco: dict[str, Any], index: int) -> tuple[dict[str, Any], cu.FrameBoxes]:
    """Return one COCO image and its object and ignore boxes."""
    image = coco["images"][index]
    return image, cu.split_annotations([ann for ann in coco["annotations"] if ann["image_id"] == image["id"]])


def draw_boxes(ax: plt.Axes, image: np.ndarray, boxes: np.ndarray, ignore: np.ndarray, title: str) -> None:
    """Show an image with corner boxes in green and ignore rectangles in red."""
    ax.imshow(image[..., ::-1])
    for color, rects in (("lime", boxes), ("red", ignore)):
        for x1, y1, x2, y2 in rects:
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=0.8))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_gt_frame(coco: dict[str, Any], index: int) -> Figure:
    """Draw ground-truth boxes and ignore regions on one original val frame."""
    image, frame = frame_boxes(coco, index)
    pixels = cv2.imread(str(gt_json("val").parent / image["file_name"]))
    fig, ax = plt.subplots(figsize=(9, 5), dpi=GRID_DPI)
    title = f"{image['file_name']}: {len(frame.boxes)} objects (green), {len(frame.ignore)} ignore regions (red)"
    draw_boxes(ax, pixels, bx.xywh_to_xyxy(frame.boxes), bx.xywh_to_xyxy(frame.ignore), title)
    fig.tight_layout()
    return fig


def subsample_table(coco: dict[str, Any], step: int) -> pd.DataFrame:
    """Return frames per val clip before and after keeping every step-th frame, as build_tasks does."""
    clips = cu.frames_by_clip(coco["images"])
    rows = {clip: {"frames": len(frames), "kept": len(frames[::step]), "first kept": [im["frame_id"] for im in frames[::step][:4]]} for clip, frames in clips.items()}
    return pd.DataFrame(rows).T


def preprocess_frame(coco: dict[str, Any], index: int) -> tuple[Figure, pd.DataFrame]:
    """Run the train_s5 ignore painting, box clipping and resizing on one val frame and plot it before and after."""
    settings = Settings.from_config(load_config(repo_root() / "configs" / "data.yaml"))
    image, frame = frame_boxes(coco, index)
    src_size = (int(image["width"]), int(image["height"]))
    out_size = bx.fit_long_side(*src_size, settings.long_side_cap)
    rects = bx.nonempty_rects(bx.pixel_rects(bx.xywh_to_xyxy(frame.ignore), *src_size))
    mask = bx.region_mask(rects, *src_size) if len(rects) else None
    cleaned = bx.clean_boxes(bx.xywh_to_xyxy(frame.boxes), src_size, out_size, mask, settings.max_ignore_frac, settings.min_box_px)
    before = cv2.imread(str(gt_json("val").parent / image["file_name"]))
    after = before.copy()
    bx.paint_rects(after, rects, settings.ignore_fill)
    after = cv2.resize(after, out_size, interpolation=cv2.INTER_AREA)
    fig, axes = plt.subplots(1, 2, figsize=(GRID_WIDTH, 3.6), dpi=GRID_DPI)
    draw_boxes(axes[0], before, bx.xywh_to_xyxy(frame.boxes), rects, f"before {src_size[0]}x{src_size[1]}")
    draw_boxes(axes[1], after, cleaned.boxes, np.zeros((0, 4)), f"after {out_size[0]}x{out_size[1]}, ignore painted {settings.ignore_fill}")
    fig.tight_layout()
    counts = {"boxes": len(frame.boxes), "kept": len(cleaned.keep), "clipped": cleaned.n_clipped,
              "dropped in ignore": cleaned.n_dropped_ignore, "dropped tiny": cleaned.n_dropped_small}
    return fig, pd.DataFrame([counts], index=[clip_of(image)])


def gt_as_predictions(coco: dict[str, Any], clip: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one clip's ground truth as a COCO dict and its objects as score-one predictions."""
    images = [image for image in coco["images"] if clip_of(image) == clip]
    ids = {image["id"]: f"val/{image['file_name']}" for image in images}
    anns = [ann for ann in coco["annotations"] if ann["image_id"] in ids]
    objects = [ann for ann in anns if not ann.get("iscrowd")]
    preds = pd.DataFrame({
        "file_name": [ids[ann["image_id"]] for ann in objects],
        "class_id": [ann["category_id"] - 1 for ann in objects],
        "score": 1.0,
        **{key: [ann["bbox"][i] for ann in objects] for i, key in enumerate("xywh")},
    })
    return preds, {"images": images, "annotations": anns, "categories": coco["categories"]}


def param_millions(model: torch.nn.Module) -> float:
    """Return the parameter count of a model in millions."""
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 2)


def detector_summary(model: torch.nn.Module, cfg: dict[str, Any]) -> pd.Series:
    """Return anchors, RPN and RoI or matcher and loss settings and the size of a built torchvision detector."""
    settings = cfg.get("model") or {}
    generator = model.rpn.anchor_generator if hasattr(model, "rpn") else model.anchor_generator
    info: dict[str, Any] = {"anchor sizes": generator.sizes, "aspect ratios": generator.aspect_ratios[0]}
    if hasattr(model, "rpn"):
        info |= {
            "rpn pre-nms train/test": (model.rpn._pre_nms_top_n["training"], model.rpn._pre_nms_top_n["testing"]),
            "rpn post-nms train/test": (model.rpn._post_nms_top_n["training"], model.rpn._post_nms_top_n["testing"]),
            "roi samples/img": model.roi_heads.fg_bg_sampler.batch_size_per_image,
            "roi positive fraction": model.roi_heads.fg_bg_sampler.positive_fraction,
            "box loss": settings.get("box_loss"),
        }
    else:
        info |= {
            "fg/bg IoU": (model.proposal_matcher.high_threshold, model.proposal_matcher.low_threshold),
            "box loss": model.head.regression_head._loss_type,
            "focal gamma/alpha": (settings.get("focal_gamma"), settings.get("focal_alpha")),
        }
    info["params (M)"] = param_millions(model)
    return pd.Series(info, name=cfg["exp_id"])


def ultralytics_summary(model: Any, name: str) -> pd.Series:
    """Return strides, detection head, query count and size of a built Ultralytics model."""
    head = model.model.model[-1]
    info = {"strides": model.model.stride.int().tolist(), "head": type(head).__name__,
            "queries": getattr(head, "num_queries", None), "params (M)": param_millions(model.model)}
    return pd.Series(info, name=name)


def config_text(exp_ids: Sequence[str]) -> str:
    """Return the raw yaml of each experiment config with its path, showing inheritance."""
    paths = [config_path(exp_id) for exp_id in exp_ids]
    return "\n".join(f"== {p.relative_to(repo_root()).as_posix()}\n{p.read_text().strip()}" for p in paths)


def ensemble_functions() -> list[Any]:
    """Return WBF or soft-NMS functions found in hw2.infer, with one line when there are none."""
    modules = [importlib.import_module(f"hw2.infer.{info.name}") for info in pkgutil.iter_modules(hw2.infer.__path__)]
    found = [obj for module in modules for name, obj in vars(module).items() if ENSEMBLE_NAMES.search(name) and callable(obj)]
    if not found:
        note("WBF / soft-NMS: no implementation in hw2.infer, skipped")
    return found


def toy_frames() -> list[str]:
    """Return the file names of a six-frame synthetic clip."""
    return [f"val/images/{TOY_CLIP}/{frame:07d}.jpg" for frame in range(1, TOY_FRAMES + 1)]


def toy_predictions() -> pd.DataFrame:
    """Return a synthetic clip: one moving car missed in frame 4 and a one-frame false person."""
    names = toy_frames()
    car = [(names[i], 2, 0.5 + 0.08 * i, 100 + 4 * i, 50.0, 40.0, 20.0) for i in range(TOY_FRAMES) if i != 3]
    noise = [(names[1], 0, 0.6, 300.0, 200.0, 10.0, 25.0)]
    return pd.DataFrame(car + noise, columns=["file_name", "class_id", "score", "x", "y", "w", "h"])
