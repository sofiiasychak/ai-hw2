"""Train, predict, benchmark and finalise an Ultralytics YOLO or RT-DETR experiment from a config file."""

from __future__ import annotations

import argparse
import importlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hw2.config import load_config
from hw2.constants import CLASS_NAMES, MAX_DETS, SCORE_THRESH, SEED
from hw2.eval.speed import benchmark
from hw2.paths import out_root, prep_root, repo_root, results_dir
from hw2.train.common import finalize_run, preds_path, save_preds, split_images
from hw2.utils import Timer, get_logger, seed_everything

LOG = get_logger(__name__)
WANDB_PROJECT = "hw2-visdrone"
DEFAULT_SPLITS: tuple[str, ...] = ("val", "val_testlike", "test")
DEFAULT_VARIANT = "train_s5"
MONITOR_SPLIT = "val_monitor"
TRAIN_DEFAULTS: dict[str, Any] = {"seed": SEED, "deterministic": True, "workers": 2}
SMOKE_TRAIN: dict[str, Any] = {"fraction": 0.05, "epochs": 1, "time": None, "plots": False}
PREDICT_DEFAULTS: dict[str, Any] = {"conf": SCORE_THRESH, "max_det": MAX_DETS}
SMOKE_IMAGES = 20
BENCH_IMAGES = 200


def model_class(family: str) -> type:
    """Return the Ultralytics model class for a config family."""
    from ultralytics import RTDETR, YOLO

    classes = {"yolo": YOLO, "rtdetr": RTDETR}
    if family not in classes:
        raise ValueError(f"family must be one of {sorted(classes)}, got {family!r}")
    return classes[family]


def resolve_weights(name: str | Path | None) -> str | None:
    """Return a repo-relative model path made absolute when it exists, else the name for Ultralytics to fetch."""
    if not name:
        return None
    path = repo_root() / name
    return str(path) if path.exists() else str(name)


def gpu_count(device: str) -> int:
    """Return how many GPUs a device string such as 0, 0,1 or cpu uses."""
    return 0 if device in ("cpu", "mps") else len([d for d in device.split(",") if d.strip()])


def default_device() -> str:
    """Return the first CUDA device when available, else cpu."""
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def pick(items: Sequence[Any], limit: int | None) -> list[Any]:
    """Return at most limit items spread evenly across the sequence, or all items without a limit."""
    if not limit or limit >= len(items):
        return list(items)
    return list(items[:: len(items) // limit][:limit])


def write_data_yaml(path: Path, variant: str, extra: Sequence[str] = ()) -> Path:
    """Write an Ultralytics data yaml with absolute image folders, training on variant plus extra splits and validating on val_monitor."""
    root = prep_root().resolve()
    trains, val = [root / split / "images" for split in (variant, *extra)], root / MONITOR_SPLIT / "images"
    for folder in (*trains, val):
        if not folder.is_dir():
            raise FileNotFoundError(f"missing prepared images {folder}; run hw2.data.prepare first")
    if MONITOR_SPLIT in extra:
        LOG.warning("%s is also trained on, so val metrics are only a sanity check", MONITOR_SPLIT)
    train = str(trains[0]) if len(trains) == 1 else [str(folder) for folder in trains]
    data = {"path": str(root), "train": train, "val": str(val), "names": dict(enumerate(CLASS_NAMES))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def setup_wandb(run: str, cfg: Mapping[str, Any], out_dir: Path, ddp: bool) -> bool:
    """Log Ultralytics training to W&B project hw2-visdrone when WANDB_API_KEY is set, else disable W&B quietly."""
    from ultralytics import settings

    enabled = bool(os.environ.get("WANDB_API_KEY"))
    if enabled and not ddp:
        try:
            import wandb

            wandb.init(project=WANDB_PROJECT, name=run, config=dict(cfg), dir=str(out_dir))
        except Exception as err:  # noqa: BLE001
            LOG.warning("W&B disabled, init failed: %s", err)
            enabled = False
    if not enabled:
        os.environ["WANDB_MODE"] = "disabled"
    if settings["wandb"] != enabled:
        settings.update({"wandb": enabled})
    return enabled


def apply_loss_patch(spec: str | None) -> None:
    """Import and call a module:function hook that patches Ultralytics before the model is built."""
    if not spec:
        return
    module, _, func = spec.partition(":")
    LOG.info("loss patch %s", spec)
    getattr(importlib.import_module(module), func)()


def build_model(cfg: Mapping[str, Any]) -> Any:
    """Build the configured YOLO or RT-DETR model, loading pretrained weights into a yaml model when given."""
    model = model_class(cfg["family"])(resolve_weights(cfg["model"]))
    pretrained = resolve_weights(cfg.get("pretrained"))
    if pretrained:
        model.load(pretrained)
    return model


def train_args(cfg: Mapping[str, Any], run: str, data_yaml: Path, device: str, smoke: bool) -> dict[str, Any]:
    """Return Ultralytics train arguments from fixed defaults, the config's train block and run settings."""
    args = {**TRAIN_DEFAULTS, **cfg.get("train", {})}
    args.update(data=str(data_yaml), project=str(out_root()), name=run, exist_ok=True, device=device)
    pretrained = resolve_weights(cfg.get("pretrained"))
    if pretrained:
        args["pretrained"] = pretrained
    if smoke:
        args.update(SMOKE_TRAIN)
    return args


def train(cfg: Mapping[str, Any], run: str, out_dir: Path, device: str, smoke: bool) -> tuple[Path, float]:
    """Train the configured model into out_dir and return its last.pt and the training wall hours."""
    variant = cfg.get("data_variant") or DEFAULT_VARIANT
    data_yaml = write_data_yaml(out_dir / "data.yaml", variant, cfg.get("extra_train") or ())
    (out_dir / "results.csv").unlink(missing_ok=True)
    setup_wandb(run, cfg, out_dir, ddp=gpu_count(device) > 1)
    apply_loss_patch(cfg.get("loss_patch"))
    model = build_model(cfg)
    with Timer(f"train {run}", LOG) as timer:
        model.train(**train_args(cfg, run, data_yaml, device, smoke))
    last = out_dir / "weights" / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(f"training finished without {last}")
    return last, timer.elapsed / 3600


def epochs_done(out_dir: Path) -> int | None:
    """Return the number of epochs logged in a run's results.csv, or None without one."""
    curves = out_dir / "results.csv"
    return len(pd.read_csv(curves)) if curves.is_file() else None


def copy_curves(out_dir: Path, run: str) -> Path | None:
    """Copy a run's results.csv to results/curves/<run>.csv and return the copy."""
    curves = out_dir / "results.csv"
    if not curves.is_file():
        return None
    target = results_dir() / "curves" / f"{run}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(curves.read_bytes())
    return target


def predict_split(model: Any, split: str, pcfg: Mapping[str, Any], device: str, limit: int | None) -> pd.DataFrame:
    """Predict one split and return boxes in original pixels with top-left x, y, w and h."""
    items = pick(split_images(split), limit)
    names = {str(path.resolve()): name for name, path in items}
    files, classes, scores, boxes = [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / f"{split}.txt"  # list sources preload everything
        source.write_text("".join(f"{path}\n" for _, path in items))
        for result in model.predict(source=str(source), stream=True, verbose=False, device=device, **pcfg):
            found = result.boxes
            files.append(np.full(len(found), names[str(Path(result.path).resolve())], dtype=object))
            classes.append(found.cls.cpu().numpy().astype(np.int64))
            scores.append(found.conf.cpu().numpy().astype(np.float64))
            boxes.append(found.xyxy.cpu().numpy().astype(np.float64).reshape(-1, 4))
    xyxy = np.concatenate(boxes) if boxes else np.zeros((0, 4))
    LOG.info("%s: %d images, %d boxes", split, len(items), len(xyxy))
    return pd.DataFrame({
        "file_name": np.concatenate(files).astype(str) if files else np.array([], dtype=str),
        "class_id": np.concatenate(classes) if classes else np.array([], dtype=np.int64),
        "score": np.concatenate(scores) if scores else np.array([], dtype=np.float64),
        "x": xyxy[:, 0],
        "y": xyxy[:, 1],
        "w": xyxy[:, 2] - xyxy[:, 0],
        "h": xyxy[:, 3] - xyxy[:, 1],
    })


def predict_args(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return Ultralytics predict arguments, mapping the config's half flag onto quantize."""
    args = {**PREDICT_DEFAULTS, **cfg.get("predict", {})}
    if "half" in args:
        args["quantize"] = 16 if args.pop("half") else None
    return args


def measure_speed(model: Any, pcfg: Mapping[str, Any], device: str) -> float:
    """Return mean end-to-end milliseconds per image over 200 val images at batch size one."""
    images = [path for _, path in pick(split_images("val"), BENCH_IMAGES)]
    args = {key: value for key, value in pcfg.items() if key != "batch"}
    return benchmark(lambda frame: model.predict(frame, verbose=False, device=device, **args), images, n=BENCH_IMAGES)


def run(cfg: Mapping[str, Any], device: str, smoke: bool = False, weights: Path | None = None) -> dict[str, Any]:
    """Train unless weights are given, predict every split with last.pt, benchmark and finalise the run."""
    run_id = f"{cfg['exp_id']}_smoke" if smoke else cfg["exp_id"]
    out_dir = out_root() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.get("train", {}).get("seed", SEED)))
    info: dict[str, Any] = {"train_hours": None, "gpu_hours": None, "ms_per_image": None}
    if weights is None:
        weights, hours = train(cfg, run_id, out_dir, device, smoke)
        info.update(train_hours=hours, gpu_hours=hours * gpu_count(device))
    info["epochs_done"] = epochs_done(out_dir)
    copy_curves(out_dir, run_id)
    model = model_class(cfg["family"])(str(weights))
    predict_device = device.split(",")[0]
    pcfg = predict_args(cfg)
    for split in cfg.get("splits") or DEFAULT_SPLITS:
        preds = predict_split(model, split, pcfg, predict_device, SMOKE_IMAGES if smoke else None)
        save_preds(preds, preds_path(out_dir, split))
    if not smoke:
        try:
            info["ms_per_image"] = measure_speed(model, pcfg, predict_device)
        except Exception:  # noqa: BLE001
            LOG.exception("speed benchmark failed, ms_per_image left empty")
    return finalize_run(run_id, cfg, out_dir, info)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None, help="0, 0,1 or cpu; default first GPU else cpu")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU-friendly end-to-end run")
    parser.add_argument("--set", nargs="*", default=[], metavar="a.b=value", help="config overrides")
    parser.add_argument("--skip-train", action="store_true", help="predict and finalise with --weights")
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.skip_train and not args.weights:
        parser.error("--skip-train needs --weights")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run one Ultralytics experiment from the command line."""
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    run(cfg, args.device or default_device(), args.smoke, args.weights if args.skip_train else None)


if __name__ == "__main__":
    main()
