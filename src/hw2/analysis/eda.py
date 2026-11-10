"""Exploratory data analysis tables and figures of the VisDrone splits for notebooks/01_eda.ipynb."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from hw2.constants import CLASS_NAMES, NUM_CLASSES, SEED
from hw2.data import boxes as bx
from hw2.data import coco_utils as cu
from hw2.data.check_prepared import draw_frame, read_yolo
from hw2.data.prepare import SPLITS as PREPARED_SPLITS
from hw2.paths import data_root, prep_root, repo_root
from hw2.utils import get_logger

LOG = get_logger(__name__)
LABELLED = ("train", "val")
OVERLAY_SPLITS = ("train_s5", "val_testlike")
FIG_DPI = 150
NET_SIDE = 1280
TINY_SIDES = (8, 16)
SIZE_BUCKETS = ("small", "medium", "large")
SIZE_EDGES = (0.0, 32.0, 96.0, np.inf)  # COCO edges on sqrt(area)
SIDE_BINS = np.logspace(0.5, 3.3, 57)
TRACK_GAPS = (1, 2, 5, 10, 20)
BOX_COLS = ["x", "y", "w", "h"]
FRAME_COLS = ["image_id", "clip", "frame", "width", "height"]
CLASS_BY_ID = dict(enumerate(CLASS_NAMES))
SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
CLASS_COLORS = dict(zip(CLASS_NAMES, plt.get_cmap("tab10").colors))


@dataclass(frozen=True)
class SplitData:
    """Frames, object boxes and deduplicated ignore boxes of one labelled split in original pixels."""

    images: pd.DataFrame
    objects: pd.DataFrame
    ignore: pd.DataFrame


def load_split(split: str) -> SplitData:
    """Load a raw labelled split's COCO file into frame, object and ignore tables with box sides."""
    coco = cu.load_coco(data_root() / split / "annotations_coco.json")
    images = pd.DataFrame(coco["images"]).rename(columns={"id": "image_id", "video": "clip", "frame_id": "frame"})
    images = images[FRAME_COLS]
    anns = pd.DataFrame(coco["annotations"])
    xywh = pd.DataFrame(anns.pop("bbox").tolist(), columns=BOX_COLS, index=anns.index, dtype=np.float64)
    anns = pd.concat([anns, xywh], axis=1).merge(images, on="image_id")
    crowd = anns["iscrowd"].astype(bool)
    objects = anns.loc[~crowd, [*FRAME_COLS, "category_id", "track_id", *BOX_COLS]].reset_index(drop=True)
    objects["class_id"] = objects.pop("category_id").map(cu.category_to_class)
    objects["side"] = np.sqrt(objects["w"] * objects["h"])
    objects["side_1280"] = objects["side"] * NET_SIDE / np.maximum(objects["width"], objects["height"])
    ignore = anns.loc[crowd, [*FRAME_COLS, *BOX_COLS]].drop_duplicates(["image_id", *BOX_COLS])
    LOG.info("%s: %d frames, %d boxes, %d ignore regions", split, len(images), len(objects), len(ignore))
    return SplitData(images, objects, ignore.reset_index(drop=True))


def load_labelled(splits: Sequence[str] = LABELLED) -> dict[str, SplitData]:
    """Load every labelled raw split."""
    return {split: load_split(split) for split in splits}


def load_test_images() -> pd.DataFrame:
    """Return the test frame table from test_images.csv."""
    return pd.read_csv(data_root() / "test_images.csv")


def save_figure(fig: Figure, name: str, prefix: str = "eda_") -> Path:
    """Save a figure as images/<prefix><name>.png at 150 dpi and return the path."""
    path = repo_root() / "images" / f"{prefix}{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    return path


def resolution(frames: pd.DataFrame) -> pd.Series:
    """Return each row's frame size as a WxH label."""
    return frames["width"].astype(str) + "x" + frames["height"].astype(str)


def split_table(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return clips, frames, frames-per-clip statistics and the number of resolutions of every split."""
    rows = {}
    for split, table in frames.items():
        per_clip = table.groupby("clip").size()
        rows[split] = {
            "clips": per_clip.size,
            "frames": len(table),
            "per_clip_min": per_clip.min(),
            "per_clip_median": per_clip.median(),
            "per_clip_max": per_clip.max(),
            "resolutions": resolution(table).nunique(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def resolution_table(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return clips and frames per resolution for every split, largest resolution first."""
    parts = {}
    for split, table in frames.items():
        grouped = table.assign(resolution=resolution(table)).groupby("resolution")
        parts[split] = pd.DataFrame({"clips": grouped["clip"].nunique(), "frames": grouped.size()})
    out = pd.concat(parts, axis=1).fillna(0).astype(int)
    order = sorted(out.index, key=lambda label: -np.prod([int(v) for v in label.split("x")]))
    return out.loc[order].rename_axis("resolution")


def plot_resolution_mix(table: pd.DataFrame) -> Figure:
    """Plot each split's share of frames per resolution as stacked horizontal bars."""
    frames = table.xs("frames", axis=1, level=1)
    shares = frames / frames.sum() * 100
    colors = plt.get_cmap("viridis")(np.linspace(0.95, 0.05, len(shares)))
    fig, ax = plt.subplots(figsize=(10, 2.8), layout="constrained")
    left = np.zeros(shares.shape[1])
    for color, (label, row) in zip(colors, shares.iterrows()):
        ax.barh(shares.columns, row.to_numpy(), left=left, color=color, label=label, edgecolor="white", lw=0.5)
        left += row.to_numpy()
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="share of frames (%)", title="Resolution mix per split")
    ax.legend(title="resolution", fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.05))
    return fig


def class_table(data: Mapping[str, SplitData]) -> pd.DataFrame:
    """Return box counts and percentage shares per class for every labelled split."""
    counts = pd.DataFrame({
        split: d.objects["class_id"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        for split, d in data.items()
    })
    counts.index = pd.Index(CLASS_NAMES, name="class")
    return pd.concat({"boxes": counts, "share_%": counts / counts.sum() * 100}, axis=1)


def plot_class_shares(table: pd.DataFrame) -> Figure:
    """Plot the share of boxes per class as grouped bars, one bar per split."""
    shares, counts = table["share_%"], table["boxes"]
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    x, width = np.arange(len(shares)), 0.8 / shares.shape[1]
    for i, split in enumerate(shares.columns):
        offset = (i - (shares.shape[1] - 1) / 2) * width
        label = f"{split} ({counts[split].sum():,} boxes)"
        ax.bar(x + offset, shares[split], width, label=label, color=SPLIT_COLORS[split])
    ax.set_xticks(x, shares.index)
    ax.set(ylabel="share of boxes (%)", title="Class distribution")
    ax.legend()
    return fig


def size_bucket(sides: pd.Series) -> pd.Series:
    """Return the COCO size bucket of each box side sqrt(area) as a string."""
    return pd.cut(sides, SIZE_EDGES, right=False, labels=SIZE_BUCKETS).astype(str)


def class_names(class_ids: pd.Series) -> pd.Series:
    """Return the class name of each class id."""
    return class_ids.map(CLASS_BY_ID)


def mark_coco_edges(ax: plt.Axes) -> None:
    """Draw the COCO small and large size edges as dashed vertical lines."""
    for edge in SIZE_EDGES[1:-1]:
        ax.axvline(edge, color="grey", ls="--", lw=0.8)


def share_hist(ax: plt.Axes, values: pd.Series, bins: np.ndarray, **style: object) -> None:
    """Draw a step histogram whose bar heights are percentages of the values."""
    weights = np.full(len(values), 100 / max(len(values), 1))
    ax.hist(values, bins=bins, weights=weights, histtype="step", lw=1.3, **style)


def size_bucket_table(data: Mapping[str, SplitData]) -> pd.DataFrame:
    """Return the percentage of COCO small, medium and large boxes per class and overall for every split."""
    parts = {}
    for split, d in data.items():
        counts = pd.crosstab(class_names(d.objects["class_id"]), size_bucket(d.objects["side"]))
        counts = counts.reindex(index=list(CLASS_NAMES), columns=list(SIZE_BUCKETS), fill_value=0)
        counts.loc["all"] = counts.sum()
        parts[split] = counts.div(counts.sum(axis=1), axis=0) * 100
    return pd.concat(parts, axis=1).rename_axis(index="class", columns=["split", "size_%"])


def size_summary(data: Mapping[str, SplitData]) -> pd.DataFrame:
    """Return box side percentiles, tiny-box shares and COCO size shares at original size and at long side 1280."""
    rows = {}
    for split, d in data.items():
        for scale, column in (("original", "side"), (f"long side {NET_SIDE}", "side_1280")):
            sides = d.objects[column]
            shares = size_bucket(sides).value_counts(normalize=True).reindex(SIZE_BUCKETS, fill_value=0) * 100
            rows[(split, scale)] = {
                "p10_px": sides.quantile(0.1),
                "median_px": sides.median(),
                "p90_px": sides.quantile(0.9),
                **{f"under_{t}px_%": (sides < t).mean() * 100 for t in TINY_SIDES},
                **{f"{bucket}_%": shares[bucket] for bucket in SIZE_BUCKETS},
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["split", "scale"])


def plot_size_hist(data: Mapping[str, SplitData]) -> Figure:
    """Plot per-class histograms of the box side sqrt(w*h) in original pixels on a log axis, one line per split."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharex=True, layout="constrained")
    for class_id, ax in enumerate(axes.flat):
        for split, d in data.items():
            sides = d.objects.loc[d.objects["class_id"] == class_id, "side"]
            share_hist(ax, sides, SIDE_BINS, color=SPLIT_COLORS[split], label=f"{split} (n={len(sides):,})")
        mark_coco_edges(ax)
        ax.set(xscale="log", title=CLASS_NAMES[class_id])
        ax.legend(fontsize=7)
    fig.supxlabel("box side sqrt(w*h), original px (dashed: COCO 32 and 96)")
    fig.supylabel("share of class boxes (%)")
    return fig


def plot_size_1280(data: Mapping[str, SplitData]) -> Figure:
    """Plot box side histograms at original size and after resizing the long side to 1280, one panel per split."""
    fig, axes = plt.subplots(1, len(data), figsize=(12, 3.6), sharey=True, layout="constrained", squeeze=False)
    for ax, (split, d) in zip(axes.flat, data.items()):
        share_hist(ax, d.objects["side"], SIDE_BINS, color="grey", label="original")
        share_hist(ax, d.objects["side_1280"], SIDE_BINS, color=SPLIT_COLORS[split], label=f"long side {NET_SIDE}")
        mark_coco_edges(ax)
        ax.set(xscale="log", title=split, xlabel="box side sqrt(w*h) (px)")
        ax.legend(fontsize=8)
    axes.flat[0].set_ylabel("share of boxes (%)")
    return fig


def objects_per_frame(data: Mapping[str, SplitData]) -> pd.DataFrame:
    """Return the box count of every frame of every labelled split, zero for frames without boxes."""
    parts = []
    for split, d in data.items():
        counts = d.objects.groupby("image_id").size()
        objects = d.images["image_id"].map(counts).fillna(0).astype(int)
        parts.append(d.images.assign(split=split, objects=objects))
    return pd.concat(parts, ignore_index=True)


def plot_objects_hist(per_frame: pd.DataFrame) -> Figure:
    """Plot the histogram of boxes per frame, one line per split."""
    bins = np.arange(0, per_frame["objects"].max() + 10, 5)
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    for split, table in per_frame.groupby("split", sort=False):
        label = f"{split} (median {table['objects'].median():.0f})"
        share_hist(ax, table["objects"], bins, color=SPLIT_COLORS[split], label=label)
    ax.set(xlabel="boxes per frame", ylabel="share of frames (%)", title="Objects per frame")
    ax.legend()
    return fig


def plot_objects_per_clip(per_frame: pd.DataFrame) -> Figure:
    """Plot per-clip boxplots of boxes per frame sorted by median, coloured by split, outliers hidden."""
    grouped = per_frame.groupby("clip")
    order = grouped["objects"].median().sort_values(kind="stable").index
    splits = grouped["split"].first()
    samples = dict(list(grouped["objects"]))
    fig, ax = plt.subplots(figsize=(8, 0.18 * len(order) + 1.2), layout="constrained")
    box = ax.boxplot(
        [samples[clip] for clip in order], orientation="horizontal", tick_labels=list(order),
        patch_artist=True, showfliers=False, widths=0.6, medianprops={"color": "black"},
    )
    for patch, clip in zip(box["boxes"], order):
        patch.set_facecolor(SPLIT_COLORS[splits[clip]])
    ax.tick_params(axis="y", labelsize=6)
    ax.set(xlabel="boxes per frame", title="Objects per frame per clip, sorted by median")
    ax.legend(handles=[Patch(color=SPLIT_COLORS[s], label=s) for s in splits.unique()], loc="lower right")
    return fig


def top_classes(class_ids: pd.Series, n: int = 3) -> str:
    """Return the n most frequent classes with their percentage shares as one string."""
    shares = class_names(class_ids).value_counts(normalize=True).head(n) * 100
    return ", ".join(f"{name} {share:.0f}%" for name, share in shares.items())


def clip_table(data: Mapping[str, SplitData], per_frame: pd.DataFrame) -> pd.DataFrame:
    """Return resolution, frames, mean boxes per frame, median box side and top classes for every clip."""
    objects = pd.concat([d.objects for d in data.values()]).groupby("clip")
    frames = per_frame.groupby("clip")
    first = frames[["split", "width", "height"]].first()
    table = pd.DataFrame({
        "split": first["split"],
        "resolution": resolution(first),
        "frames": frames.size(),
        "mean_objects": frames["objects"].mean(),
        "median_side_px": objects["side"].median(),
        "top_classes": objects["class_id"].apply(top_classes),
    })
    return table.sort_values(["split", "mean_objects"], ascending=[True, False])


def union_area(rects: np.ndarray) -> float:
    """Return the exact area covered by a union of corner boxes using coordinate compression."""
    xs, ys = np.unique(rects[:, [0, 2]]), np.unique(rects[:, [1, 3]])
    cx, cy = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    in_x = (rects[:, None, 0] < cx) & (cx < rects[:, None, 2])
    in_y = (rects[:, None, 1] < cy) & (cy < rects[:, None, 3])
    covered = (in_x[:, :, None] & in_y[:, None, :]).any(axis=0)
    return float((np.diff(xs)[:, None] * np.diff(ys)[None, :])[covered].sum())


def ignore_shares(data: Mapping[str, SplitData]) -> pd.DataFrame:
    """Return the percentage of every frame's area covered by ignore regions, zero for frames without any."""
    parts = []
    for split, d in data.items():
        ignore = d.ignore.sort_values("image_id", kind="stable")
        ids, starts = np.unique(ignore["image_id"].to_numpy(), return_index=True)
        sizes = ignore[["width", "height"]].to_numpy()[starts]
        chunks = np.split(bx.xywh_to_xyxy(ignore[BOX_COLS].to_numpy()), starts[1:])
        covered = {i: union_area(bx.clip_xyxy(c, w, h)) for i, c, (w, h) in zip(ids, chunks, sizes)}
        area = d.images["width"] * d.images["height"]
        pct = d.images["image_id"].map(covered).fillna(0.0) / area * 100
        parts.append(d.images.assign(split=split, ignore_pct=pct))
    return pd.concat(parts, ignore_index=True)


def ignore_summary(shares: pd.DataFrame) -> pd.DataFrame:
    """Return per split the share of frames and clips with ignore regions and statistics of the covered area."""
    frames = shares.groupby("split", sort=False)["ignore_pct"]
    clips = shares.groupby(["split", "clip"], sort=False)["ignore_pct"].mean().groupby("split", sort=False)
    return pd.DataFrame({
        "frames_with_ignore_%": frames.apply(lambda pct: (pct > 0).mean() * 100),
        "mean_area_%": frames.mean(),
        "p90_area_%": frames.quantile(0.9),
        "max_area_%": frames.max(),
        "clips_with_ignore": clips.apply(lambda pct: int((pct > 0).sum())),
        "clips": clips.size(),
        "max_clip_mean_area_%": clips.max(),
    })


def plot_ignore(shares: pd.DataFrame) -> Figure:
    """Plot the per-frame histogram of ignore area share and the per-clip mean share sorted descending."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 3.8), width_ratios=(1, 2), layout="constrained")
    bins = np.arange(0, shares["ignore_pct"].max() + 2.5, 2.5)
    for split, table in shares.groupby("split", sort=False):
        share_hist(left, table["ignore_pct"], bins, color=SPLIT_COLORS[split], label=split)
    left.set(yscale="log", xlabel="ignore area per frame (% of frame)", ylabel="share of frames (%)")
    left.legend()
    per_clip = shares.groupby(["clip", "split"])["ignore_pct"].mean().sort_values(ascending=False).reset_index()
    right.bar(np.arange(len(per_clip)), per_clip["ignore_pct"], color=per_clip["split"].map(SPLIT_COLORS))
    right.set_xticks(np.arange(len(per_clip)), per_clip["clip"], rotation=90, fontsize=5)
    right.set(ylabel="mean ignore area (% of frame)", xlim=(-0.6, len(per_clip) - 0.4))
    right.legend(handles=[Patch(color=SPLIT_COLORS[s], label=s) for s in per_clip["split"].unique()])
    fig.suptitle("Ignore regions: per frame (left) and per clip (right)")
    return fig


def paired_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the IoU of each row pair of two equally long x, y, w, h box arrays."""
    a, b = bx.xywh_to_xyxy(a), bx.xywh_to_xyxy(b)
    inter = np.clip(np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2]), 0, None).prod(axis=1)
    union = (a[:, 2:] - a[:, :2]).prod(axis=1) + (b[:, 2:] - b[:, :2]).prod(axis=1) - inter
    return np.divide(inter, union, out=np.zeros(len(a)), where=union > 0)


def track_iou(objects: pd.DataFrame, gaps: Sequence[int] = TRACK_GAPS) -> pd.DataFrame:
    """Return the number of box pairs and the mean IoU of one track between frames t and t+k, per class and overall."""
    keys = ["clip", "track_id", "frame"]
    base = objects[[*keys, "class_id", *BOX_COLS]]
    later_cols = [f"{col}_later" for col in BOX_COLS]
    rows = {}
    for gap in gaps:
        later = base[[*keys, *BOX_COLS]].assign(frame=base["frame"] - gap)
        pairs = base.merge(later, on=keys, suffixes=("", "_later"))
        pairs["iou"] = paired_iou(pairs[BOX_COLS].to_numpy(), pairs[later_cols].to_numpy())
        by_class = pairs.groupby(class_names(pairs["class_id"]))["iou"].mean().reindex(CLASS_NAMES)
        rows[gap] = {"pairs": len(pairs), "all": pairs["iou"].mean(), **by_class.to_dict()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("k")


def plot_track_iou(table: pd.DataFrame) -> Figure:
    """Plot mean same-track IoU against the frame gap k, one thin line per class and a thick overall line."""
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for name in CLASS_NAMES:
        ax.plot(table.index, table[name], marker="o", ms=3, lw=1, color=CLASS_COLORS[name], label=name)
    ax.plot(table.index, table["all"], marker="o", lw=2.5, color="black", label="all")
    ax.set_xscale("log")
    ax.set_xticks(table.index, [str(k) for k in table.index])
    ax.minorticks_off()
    ax.set(ylim=(0, 1), xlabel="frame gap k", ylabel="mean IoU of same track_id", title="Temporal redundancy (train)")
    ax.legend(fontsize=8, ncol=3)
    return fig


def pick_overlay_frames(
    splits: Sequence[str] = OVERLAY_SPLITS, per_split: int = 2, seed: int = SEED
) -> pd.DataFrame:
    """Return seeded random prepared frames that have ignore regions, from distinct clips, for each split."""
    meta = pd.read_csv(prep_root() / "meta" / "frames.csv")
    rng = np.random.default_rng(seed)
    picked = []
    for split in splits:
        pool = meta[(meta["split"] == split) & (meta["ignore_frac"] > 0)]
        for clip in rng.choice(sorted(pool["clip"].unique()), per_split, replace=False):
            frames = pool[pool["clip"] == clip]
            picked.append(frames.iloc[rng.integers(len(frames))])
    return pd.DataFrame(picked).reset_index(drop=True)


def overlay_image(row: pd.Series, ignore: pd.DataFrame) -> np.ndarray:
    """Return a prepared frame in RGB with its YOLO boxes and the raw ignore regions scaled onto it."""
    root = prep_root()
    width, height = int(row["out_w"]), int(row["out_h"])
    pixels = cv2.imread(str(root / row["out_file"]), cv2.IMREAD_COLOR)
    label = root / row["split"] / "labels" / row["clip"] / f"{Path(row['out_file']).stem}.txt"
    classes, corners = read_yolo(label, width, height)
    regions = ignore.loc[(ignore["clip"] == row["clip"]) & (ignore["frame"] == row["frame"]), BOX_COLS]
    scaled = bx.scale_xyxy(bx.xywh_to_xyxy(regions.to_numpy()), width / row["src_w"], height / row["src_h"])
    frame = cu.FrameBoxes(classes, bx.xyxy_to_xywh(corners), bx.xyxy_to_xywh(scaled), [])
    return cv2.cvtColor(draw_frame(pixels, frame), cv2.COLOR_BGR2RGB)


def plot_overlays(frames: pd.DataFrame, data: Mapping[str, SplitData]) -> Figure:
    """Plot prepared frames with YOLO boxes coloured by class and the source ignore regions outlined in white."""
    rows = -(-len(frames) // 2)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.9 * rows), squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, frames.iterrows()):
        ax.imshow(overlay_image(row, data[PREPARED_SPLITS[row["split"]].source].ignore))
        ax.set_title(f"{row['split']}  {row['clip']}  frame {row['frame']}  ({row['out_w']}x{row['out_h']})", fontsize=9)
    return fig
