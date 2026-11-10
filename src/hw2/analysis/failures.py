"""TIDE error breakdown and a gallery of failure frames for one predictions parquet against COCO ground truth."""

from __future__ import annotations

import argparse
import contextlib
import io
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from hw2.analysis import eda
from hw2.constants import MAX_DETS
from hw2.data import boxes as bx
from hw2.data import coco_utils as cu
from hw2.eval.coco_eval import gt_frame_index, gt_frame_key
from hw2.eval.preds import load_preds, match_frames, top_k_per_image
from hw2.infer.temporal import iou_matrix
from hw2.paths import repo_root, results_dir
from hw2.train.common import gt_json
from hw2.utils import get_logger, save_json

LOG = get_logger(__name__)
IOU_THR = 0.5
IGNORE_IOA = 0.5
SCORE_THR = 0.3
FRAME_STRIDE = 10
SMALL_SIDE = eda.SIZE_EDGES[1]
GALLERY: tuple[str, ...] = ("missed_small", "confusions", "duplicates")
MAIN_ERRORS: tuple[str, ...] = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")
SPECIAL_ERRORS: tuple[str, ...] = ("FalsePos", "FalseNeg")
GT_COLOR = (0, 200, 0)
PRED_COLOR = (0, 0, 255)
GALLERY_SIDE = 1920


@dataclass(frozen=True)
class Frame:
    """One ground-truth image with its path, size, boxes and the detections kept for analysis."""

    image_id: int
    key: tuple[str, int]
    path: Path
    width: int
    height: int
    gt: cu.FrameBoxes
    dets: pd.DataFrame


def drop_ignored(dets: pd.DataFrame, ignore: np.ndarray, width: int, height: int) -> pd.DataFrame:
    """Drop detections lying at least half inside ignore regions, as COCO crowd matching does."""
    if not len(ignore) or dets.empty:
        return dets
    mask = bx.region_mask(bx.pixel_rects(bx.xywh_to_xyxy(ignore), width, height), width, height)
    inside = bx.covered_fraction(bx.xywh_to_xyxy(dets[["x", "y", "w", "h"]].to_numpy()), mask)
    return dets[inside < IGNORE_IOA]


def iter_frames(preds: pd.DataFrame, gt_path: Path) -> Iterator[Frame]:
    """Yield every ground-truth frame with its boxes and top-500 detections outside ignore regions."""
    coco = cu.load_coco(gt_path)
    image_ids = match_frames(preds["file_name"], gt_frame_index(coco["images"]), "ground-truth")
    preds = top_k_per_image(preds.assign(image_id=image_ids), MAX_DETS, by="image_id")
    groups = preds.groupby("image_id").indices
    by_image = cu.annotations_by_image(coco["annotations"])
    for image in coco["images"]:
        gt = cu.split_annotations(by_image.get(image["id"], []))
        dets = preds.iloc[groups.get(image["id"], [])].sort_values("score", ascending=False)
        dets = drop_ignored(dets, gt.ignore, image["width"], image["height"])
        yield Frame(image["id"], gt_frame_key(image), gt_path.parent / image["file_name"], image["width"], image["height"], gt, dets)


def tide_data(frames: Sequence[Frame], name: str) -> tuple[Any, Any]:
    """Return TIDE ground-truth and detection data built from frames, allowing 500 detections per image."""
    from tidecv import Data

    gt, dets = Data("gt", max_dets=MAX_DETS), Data(name, max_dets=MAX_DETS)
    for frame in frames:
        for cls, box in zip(frame.gt.classes.tolist(), frame.gt.boxes.tolist()):
            gt.add_ground_truth(frame.image_id, cls, box)
        rows = zip(frame.dets["class_id"].tolist(), frame.dets["score"].tolist(), frame.dets[["x", "y", "w", "h"]].to_numpy().tolist())
        for cls, score, box in rows:
            dets.add_detection(frame.image_id, cls, score, box)
    return gt, dets


def tide_errors(frames: Sequence[Frame], name: str) -> dict[str, Any]:
    """Run TIDE box evaluation and return AP50 with the main and special error dAP values."""
    from tidecv import TIDE

    gt, dets = tide_data(frames, name)
    tide = TIDE()
    with contextlib.redirect_stdout(io.StringIO()):
        run = tide.evaluate(gt, dets, mode=TIDE.BOX, name=name)
        errors = tide.get_all_errors()
    return {"ap50": run.ap, "main": errors["main"][name], "special": errors["special"][name], "n_images": len(frames)}


def plot_tide(errors: dict[str, Any], title: str) -> Figure:
    """Draw horizontal bars of TIDE main and special error dAP."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [3, 1.3]})
    for ax, key, names in ((axes[0], "main", MAIN_ERRORS), (axes[1], "special", SPECIAL_ERRORS)):
        values = [errors[key][name] for name in names]
        bars = ax.barh(names, values, color="tab:red" if key == "main" else "tab:grey")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("dAP50 if fixed")
        ax.set_xlim(0, max(values) * 1.25 + 1e-6)
    fig.suptitle(f"{title} | AP50 {errors['ap50']:.1f} on {errors['n_images']} frames")
    fig.tight_layout()
    return fig


def frame_errors(frame: Frame, score_thr: float = SCORE_THR, iou_thr: float = IOU_THR) -> dict[str, int]:
    """Count missed small objects, class confusions and duplicates among confident detections by greedy matching."""
    dets = frame.dets[frame.dets["score"] >= score_thr]
    iou = iou_matrix(dets[["x", "y", "w", "h"]].to_numpy(), frame.gt.boxes)
    same = dets["class_id"].to_numpy()[:, None] == frame.gt.classes[None, :]
    matched = np.zeros(len(frame.gt.classes), bool)
    confusions = duplicates = 0
    for row_iou, row_same in zip(iou, same):
        hit = row_iou >= iou_thr
        if not (hit & row_same).any():
            confusions += int((hit & ~row_same).any())
        elif (free := hit & row_same & ~matched).any():
            matched[np.argmax(np.where(free, row_iou, -1.0))] = True
        else:
            duplicates += 1
    small = np.sqrt(frame.gt.boxes[:, 2] * frame.gt.boxes[:, 3]) < SMALL_SIDE
    return {"missed_small": int((small & ~matched).sum()), "confusions": confusions, "duplicates": duplicates}


def pick_gallery(counts: pd.DataFrame) -> list[tuple[str, int]]:
    """Return the frame with the most errors of each gallery kind, never repeating a frame."""
    picks: list[tuple[str, int]] = []
    for kind in GALLERY:
        ranked = counts.drop(index=[i for _, i in picks]).sort_values(kind, ascending=False, kind="stable")
        picks.append((kind, int(ranked.index[0])))
    return picks


def draw_failure(frame: Frame, title: str, score_thr: float = SCORE_THR) -> np.ndarray:
    """Draw ground truth in green and confident detections in red with a title banner, downscaled for the gallery."""
    image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
    thick = max(1, round(max(image.shape[:2]) / 960))
    dets = frame.dets.loc[frame.dets["score"] >= score_thr, ["x", "y", "w", "h"]].to_numpy()
    for boxes, color in ((frame.gt.boxes, GT_COLOR), (dets, PRED_COLOR)):
        for x1, y1, x2, y2 in bx.round_half_up(bx.xywh_to_xyxy(boxes)).tolist():
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
    scale = min(1.0, GALLERY_SIDE / max(image.shape[:2]))
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def save_gallery(frames: Sequence[Frame], counts: pd.DataFrame, exp: str) -> list[dict[str, Any]]:
    """Write images/<exp>_fail_<n>.png for each gallery pick and return what each shows."""
    by_id = {frame.image_id: frame for frame in frames}
    picks = []
    for n, (kind, image_id) in enumerate(pick_gallery(counts), start=1):
        frame, row = by_id[image_id], counts.loc[image_id]
        stats = ", ".join(f"{k} {int(row[k])}" for k in GALLERY)
        label = f"{frame.key[0]}/{frame.key[1]:07d}"
        title = f"{exp} | {label} | most {kind} | {stats} | green GT, red score>={SCORE_THR}"
        path = repo_root() / "images" / f"{exp}_fail_{n}.png"
        cv2.imwrite(str(path), draw_failure(frame, title))
        picks.append({"n": n, "kind": kind, "frame": label, **{k: int(v) for k, v in row.items()}})
    return picks


def analyse(preds_path: Path, gt_path: Path, exp: str, frame_stride: int = FRAME_STRIDE) -> dict[str, Any]:
    """Run TIDE on every frame_stride-th frame and the failure gallery on all frames, saving figures and a json."""
    preds = load_preds(preds_path)
    frames = list(iter_frames(preds, gt_path))
    counts = pd.DataFrame([frame_errors(frame) for frame in frames], index=[frame.image_id for frame in frames])
    gallery = save_gallery(frames, counts, exp)
    sampled = [frame for frame in frames if (frame.key[1] - 1) % frame_stride == 0]
    errors = tide_errors(sampled, exp)
    eda.save_figure(plot_tide(errors, f"{exp} TIDE (val, every {frame_stride}th frame)"), "tide", prefix=f"{exp}_")
    result = {"exp_id": exp, "frame_stride": frame_stride, "tide": errors, "totals": counts.sum().to_dict(), "gallery": gallery}
    save_json(result, results_dir() / "analysis" / f"{exp}_failures.json")
    LOG.info("%s TIDE main %s", exp, {k: round(v, 2) for k, v in errors["main"].items()})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the predictions, ground truth, experiment id and TIDE frame stride."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True, type=Path, help="val predictions parquet")
    parser.add_argument("--gt", type=Path, default=None, help="COCO json, default raw val")
    parser.add_argument("--exp", default=None, help="figure prefix, default the preds folder name")
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE, help="TIDE uses every n-th frame")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Analyse one predictions parquet from the command line."""
    args = parse_args(argv)
    return analyse(args.preds, args.gt or gt_json("val"), args.exp or args.preds.parent.name, args.frame_stride)


if __name__ == "__main__":
    main()
