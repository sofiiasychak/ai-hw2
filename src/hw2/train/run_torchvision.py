"""Train, predict, benchmark and finalise a torchvision Faster R-CNN or RetinaNet experiment from a config file."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from hw2.config import load_config
from hw2.constants import SEED
from hw2.data import boxes as bx
from hw2.data.torch_dataset import LABEL_OFFSET, CocoDetectionDataset, ImageFileDataset, collate
from hw2.eval.coco_eval import evaluate
from hw2.eval.speed import benchmark
from hw2.models.frcnn import build_frcnn
from hw2.models.patches_tv import configure_patches
from hw2.models.retinanet import build_retinanet
from hw2.paths import out_root, prep_root, results_dir
from hw2.train.common import finalize_run, preds_path, save_preds, split_images
from hw2.train.run_ultralytics import (
    BENCH_IMAGES,
    DEFAULT_SPLITS,
    DEFAULT_VARIANT,
    MONITOR_SPLIT,
    SMOKE_IMAGES,
    WANDB_PROJECT,
    pick,
)
from hw2.utils import get_logger, seed_everything

LOG = get_logger(__name__)
BUILDERS: dict[str, Callable[..., torch.nn.Module]] = {"frcnn": build_frcnn, "retinanet": build_retinanet}
TRAIN_DEFAULTS: dict[str, Any] = {
    "lr": 0.005,
    "batch": 4,
    "workers": 2,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "warmup_iters": 500,
    "grad_clip": 10.0,
    "color_jitter": 0.0,
    "epochs": None,
}
PREDICT_DEFAULTS: dict[str, Any] = {"batch": 4, "workers": 2, "amp": False}  # fp16 decoding loses pixels
DEFAULT_BUDGET_H = 2.5
PLAN_AFTER = 200
UNPLANNED_EPOCHS = 1000
WARMUP_START = 0.001
LOG_EVERY = 50
SMOKE_TRAIN_IMAGES = 8
SMOKE_ITERS = 4
SMOKE_MONITOR_IMAGES = 8
OVERFIT_ITERS = 150
OVERFIT_MIN_MAP = 0.5
OVERFIT_TRAIN: dict[str, Any] = {"warmup_iters": 30, "weight_decay": 0.0}
CHECKPOINT = "last.pt"


class WarmupCosine:
    """Learning-rate factor with a linear warm-up then cosine decay to zero at a total that can be re-planned."""

    def __init__(self, warmup: int, total: int, decay: bool = True) -> None:
        """Store the warm-up length, the planned total iterations and whether to decay after warm-up."""
        self.warmup = warmup
        self.total = total
        self.decay = decay

    def __call__(self, step: int) -> float:
        """Return the factor at an iteration, with warm-up capped at a fifth of the total."""
        warm = min(self.warmup, self.total // 5)
        if step < warm:
            return WARMUP_START + (1 - WARMUP_START) * step / warm
        if not self.decay:
            return 1.0
        done = min(1.0, (step - warm) / max(1, self.total - warm))
        return 0.5 * (1 + math.cos(math.pi * done))


@dataclass
class Progress:
    """Training progress kept in every checkpoint."""

    epoch: int = 0
    iteration: int = 0
    total_iters: int = 0
    iters_per_epoch: int = 1
    train_seconds: float = 0.0
    overhead_seconds: float = 0.0
    eval_seconds: float | None = None
    planned: bool = False

    @property
    def epochs_done(self) -> float:
        """Return completed epochs, fractional when training stopped mid-epoch."""
        return round(self.iteration / self.iters_per_epoch, 2)


@dataclass(frozen=True)
class Monitor:
    """Images and COCO ground truth evaluated after every epoch."""

    items: list[tuple[str, Path]]
    gt: dict[str, Any]


def plan_iterations(budget_s: float, progress: Progress, sec_per_iter: float, eval_s: float) -> int:
    """Return the total iteration count whose remaining steps and per-epoch evaluations fit the time budget."""
    left = budget_s - progress.train_seconds
    total = progress.iteration + int(left / sec_per_iter)
    for _ in range(3):
        evals = max(1, math.ceil(total / progress.iters_per_epoch) - progress.epoch)
        total = progress.iteration + max(0, int((left - evals * eval_s) / sec_per_iter))
    return total


def torch_device(spec: str | None) -> torch.device:
    """Return the device for cpu, mps, cuda:N or a bare GPU index list such as 0 or 0,1, defaulting to cuda:0."""
    if not spec:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    first = spec.split(",")[0].strip()
    if "," in spec:
        LOG.warning("multi-GPU device %s not supported, using %s only", spec, first)
    return torch.device(f"cuda:{first}" if first.isdigit() else first)


def autocast(device: torch.device, enabled: bool = True) -> contextlib.AbstractContextManager:
    """Return a float16 autocast context on CUDA when enabled and a no-op otherwise."""
    if device.type != "cuda" or not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def to_float(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a uint8 image tensor to the device and scale it to float in [0, 1]."""
    return image.to(device, non_blocking=True).float().div_(255)


def settings(cfg: Mapping[str, Any], block: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Return a config block merged over its defaults."""
    return {**defaults, **(cfg.get(block) or {})}


def build_model(cfg: Mapping[str, Any], pretrained: bool = True) -> torch.nn.Module:
    """Build the configured detector family."""
    family = cfg.get("family")
    if family not in BUILDERS:
        raise ValueError(f"family must be one of {sorted(BUILDERS)}, got {family!r}")
    return BUILDERS[family](cfg, pretrained=pretrained)


def preds_frame(names: Sequence[str], outputs: Sequence[Mapping[str, np.ndarray]]) -> pd.DataFrame:
    """Return predictions in original pixels with class_id = label - 1, dropping the background label."""
    counts = [len(out["scores"]) for out in outputs]
    labels = np.concatenate([out["labels"] for out in outputs]) if outputs else np.zeros(0, np.int64)
    scores = np.concatenate([out["scores"] for out in outputs]) if outputs else np.zeros(0)
    xyxy = np.concatenate([out["boxes"] for out in outputs]) if outputs else np.zeros((0, 4))
    xywh = bx.xyxy_to_xywh(xyxy)
    frame = pd.DataFrame({
        "file_name": np.repeat(np.asarray(names, dtype=object), counts).astype(str),
        "class_id": labels.astype(np.int64) - LABEL_OFFSET,
        "score": scores.astype(np.float64),
        "x": xywh[:, 0],
        "y": xywh[:, 1],
        "w": xywh[:, 2],
        "h": xywh[:, 3],
    })
    return frame[frame["class_id"] >= 0].reset_index(drop=True)


@torch.inference_mode()
def predict(model: torch.nn.Module, items: Sequence[tuple[str, Path]], device: torch.device, pcfg: Mapping[str, Any]) -> pd.DataFrame:
    """Predict images streamed in batches by a loader and return their boxes in original pixels."""
    model.eval()
    loader = DataLoader(
        ImageFileDataset(items), batch_size=pcfg["batch"], num_workers=pcfg["workers"],
        collate_fn=collate, pin_memory=device.type == "cuda",
    )
    names: list[str] = []
    outputs: list[dict[str, np.ndarray]] = []
    for batch_names, images in loader:
        with autocast(device, pcfg["amp"]):
            results = model([to_float(image, device) for image in images])
        names += batch_names
        outputs += [
            {
                "boxes": out["boxes"].float().cpu().numpy(),
                "scores": out["scores"].float().cpu().numpy(),
                "labels": out["labels"].cpu().numpy(),
            }
            for out in results
        ]
    return preds_frame(names, outputs)


def evaluate_monitor(model: torch.nn.Module, monitor: Monitor, device: torch.device, pcfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return COCO metrics of the model on the monitor images."""
    preds = predict(model, monitor.items, device, pcfg)
    if preds.empty:
        return {"map": 0.0, "map50": 0.0}
    return evaluate(preds, monitor.gt, breakdown=False)


def init_wandb(run_id: str, cfg: Mapping[str, Any], out_dir: Path) -> Any:
    """Start a W&B run in project hw2-visdrone when WANDB_API_KEY is set, else return None."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb

        return wandb.init(project=WANDB_PROJECT, name=run_id, config=dict(cfg), dir=str(out_dir))
    except Exception as err:  # noqa: BLE001
        LOG.warning("W&B disabled, init failed: %s", err)
        return None


def append_curve(path: Path, row: Mapping[str, Any]) -> None:
    """Append one row to a curves CSV, writing the header for a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


class Trainer:
    """SGD loop with warm-up cosine schedule, AMP, gradient clipping, time budget, checkpoints and monitoring."""

    def __init__(
        self,
        model: torch.nn.Module,
        tcfg: Mapping[str, Any],
        dataset: Dataset,
        device: torch.device,
        total_iters: int,
        budget_s: float | None = None,
        decay: bool = True,
    ) -> None:
        """Set up the optimiser, schedule, scaler, seeded loader and progress."""
        self.model, self.tcfg, self.device = model.to(device), dict(tcfg), device
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            self.params, lr=tcfg["lr"], momentum=tcfg["momentum"], weight_decay=tcfg["weight_decay"]
        )
        self.schedule = WarmupCosine(int(tcfg["warmup_iters"]), total_iters, decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.schedule)
        self.amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.generator = torch.Generator().manual_seed(SEED)
        self.loader = DataLoader(
            dataset, batch_size=tcfg["batch"], sampler=RandomSampler(dataset, generator=self.generator),
            num_workers=tcfg["workers"], collate_fn=collate, generator=self.generator, pin_memory=self.amp,
            persistent_workers=tcfg["workers"] > 0,
        )
        self.progress = Progress(total_iters=total_iters, iters_per_epoch=len(self.loader))
        self.budget_s = budget_s
        self.epoch_cap = tcfg["epochs"] and int(tcfg["epochs"]) * len(self.loader)
        self.monitor: Monitor | None = None
        self.pcfg: Mapping[str, Any] = PREDICT_DEFAULTS
        self.checkpoint: Path | None = None
        self.curves: Path | None = None
        self.wandb: Any = None
        self.replanned = False
        self.clock = 0.0

    def state(self) -> dict[str, Any]:
        """Return everything --resume needs."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "progress": asdict(self.progress),
        }

    def save(self) -> None:
        """Write last.pt atomically when a checkpoint path is set."""
        if self.checkpoint is None:
            return
        tmp = self.checkpoint.with_suffix(".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, self.checkpoint)

    def load(self, path: Path) -> None:
        """Restore model, optimiser, scaler, schedule and progress from a checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.progress = Progress(**state["progress"])
        LOG.info("resumed %s at epoch %d, iteration %d/%d", path, self.progress.epoch, self.progress.iteration, self.progress.total_iters)

    def elapsed(self) -> float:
        """Return wall seconds of training so far, including earlier sessions."""
        return time.perf_counter() - self.clock

    def set_total(self, total: int, reason: str) -> None:
        """Set the planned iterations, capped by the epoch limit, and stretch the cosine schedule to them."""
        total = max(self.progress.iteration, min(total, self.epoch_cap or total))
        self.progress.total_iters = self.schedule.total = total
        LOG.info("%s: %d iterations (%.2f epochs)", reason, total, total / self.progress.iters_per_epoch)

    def plan(self, eval_s: float) -> None:
        """Fit the remaining iterations to the time budget from the measured step rate."""
        steps = self.progress.train_seconds - self.progress.overhead_seconds
        rate = steps / max(1, self.progress.iteration)
        self.set_total(plan_iterations(self.budget_s, self.progress, rate, eval_s), f"budget {rate:.3f} s/it")
        self.progress.planned = True

    def step(self, images: Sequence[torch.Tensor], targets: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, float]:
        """Run one optimisation step and return the loss terms and their total."""
        images = [image.to(self.device, non_blocking=True) for image in images]
        targets = [{key: t[key].to(self.device, non_blocking=True) for key in ("boxes", "labels")} for t in targets]
        with autocast(self.device):
            losses = self.model(images, targets)
            loss = sum(losses.values())
        self.optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.params, self.tcfg["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            LOG.warning("non-finite loss at iteration %d, step skipped", self.progress.iteration)
        self.scheduler.step()
        return {**{key: float(value.detach()) for key, value in losses.items()}, "loss": float(loss.detach())}

    def after_step(self, losses: Mapping[str, float]) -> None:
        """Plan the budget after 200 iterations, log progress and stop when the budget runs out."""
        progress = self.progress
        progress.train_seconds = self.elapsed()
        if self.budget_s and not progress.planned and progress.iteration >= PLAN_AFTER:
            per_image = progress.train_seconds / progress.iteration / self.tcfg["batch"]
            self.plan(progress.eval_seconds or per_image * len(self.monitor.items if self.monitor else []))
        if progress.iteration % LOG_EVERY == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            LOG.info("it %d/%d | loss %.4f | lr %.5f | %.1f min", progress.iteration, progress.total_iters, losses["loss"], lr, progress.train_seconds / 60)
            if self.wandb:
                self.wandb.log({**losses, "lr": lr}, step=progress.iteration)
        if self.budget_s and progress.train_seconds >= self.budget_s and progress.iteration < progress.total_iters:
            self.set_total(progress.iteration, "time budget reached, stopping")

    def run_epoch(self) -> tuple[float, bool]:
        """Train until the epoch ends or the planned iterations are done, returning the mean loss and completeness."""
        self.model.train()
        self.generator.manual_seed(SEED + self.progress.epoch)
        totals: list[float] = []
        batch = 0
        for batch, (images, targets) in enumerate(self.loader, start=1):
            losses = self.step(images, targets)
            if math.isfinite(losses["loss"]):
                totals.append(losses["loss"])
            self.progress.iteration += 1
            self.after_step(losses)
            if self.progress.iteration >= self.progress.total_iters:
                break
        complete = batch == len(self.loader)
        return (float(np.mean(totals)) if totals else float("nan")), complete

    def end_epoch(self, loss: float, complete: bool) -> None:
        """Evaluate the monitor split, append a curve row, save last.pt and re-plan once the evaluation cost is known."""
        start = time.perf_counter()
        self.progress.epoch += int(complete)
        metrics = evaluate_monitor(self.model, self.monitor, self.device, self.pcfg) if self.monitor else {}
        row = {
            "epoch": self.progress.epochs_done,
            "iteration": self.progress.iteration,
            "hours": round(self.elapsed() / 3600, 4),
            "loss": loss,
            "map": metrics.get("map"),
            "map50": metrics.get("map50"),
        }
        if self.curves:
            append_curve(self.curves, row)
        if self.wandb:
            self.wandb.log({f"epoch/{key}": value for key, value in row.items()}, step=self.progress.iteration)
        if self.monitor:
            LOG.info("epoch %.2f | loss %.4f | val_monitor mAP %.4f | mAP50 %.4f", row["epoch"], loss, row["map"] or 0, row["map50"] or 0)
        overhead = time.perf_counter() - start
        self.progress.eval_seconds = overhead
        self.progress.overhead_seconds += overhead
        self.progress.train_seconds = self.elapsed()
        if self.budget_s and self.progress.planned and not self.replanned:
            self.plan(overhead)
            self.replanned = True
        self.save()

    def fit(self) -> Progress:
        """Train until the planned iterations are done and return the final progress."""
        self.clock = time.perf_counter() - self.progress.train_seconds
        while self.progress.iteration < self.progress.total_iters:
            self.end_epoch(*self.run_epoch())
        return self.progress


def train(model: torch.nn.Module, cfg: Mapping[str, Any], run_id: str, out_dir: Path, device: torch.device, smoke: bool, resume: bool) -> Progress:
    """Train the model on the prepared split with val_monitor checks and return the progress."""
    tcfg = settings(cfg, "train", TRAIN_DEFAULTS)
    root = prep_root()
    dataset: Dataset = CocoDetectionDataset(root, cfg.get("data_variant") or DEFAULT_VARIANT, color_jitter=tcfg["color_jitter"])
    monitor_set = CocoDetectionDataset(root, MONITOR_SPLIT, augment=False)
    monitor_idx = pick(range(len(monitor_set)), SMOKE_MONITOR_IMAGES if smoke else None)
    if smoke:
        dataset = Subset(dataset, pick(range(len(dataset)), SMOKE_TRAIN_IMAGES))
    budget_h = None if smoke else float(cfg.get("time_budget_h") or DEFAULT_BUDGET_H)
    iters_per_epoch = math.ceil(len(dataset) / tcfg["batch"])
    total = SMOKE_ITERS if smoke else iters_per_epoch * int(tcfg["epochs"] or UNPLANNED_EPOCHS)
    trainer = Trainer(model, tcfg, dataset, device, total, budget_h and budget_h * 3600)
    trainer.monitor = Monitor(monitor_set.items(monitor_idx), monitor_set.coco_subset(monitor_idx))
    trainer.pcfg = settings(cfg, "predict", PREDICT_DEFAULTS)
    trainer.checkpoint = out_dir / CHECKPOINT
    trainer.curves = results_dir() / "curves" / f"{run_id}.csv"
    if resume:
        trainer.load(trainer.checkpoint)
    else:
        trainer.curves.unlink(missing_ok=True)
    trainer.wandb = init_wandb(run_id, cfg, out_dir)
    try:
        return trainer.fit()
    finally:
        if trainer.wandb:
            trainer.wandb.finish()


def load_weights(model: torch.nn.Module, path: Path) -> float | None:
    """Load model weights from a checkpoint or plain state dict and return its epochs done when recorded."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    progress = state.get("progress")
    return Progress(**progress).epochs_done if progress else None


def measure_speed(model: torch.nn.Module, device: torch.device, pcfg: Mapping[str, Any]) -> float:
    """Return mean end-to-end milliseconds per image over 200 val images at batch size one."""
    model.eval()
    images = [path for _, path in pick(split_images("val"), BENCH_IMAGES)]

    def predict_one(frame: np.ndarray) -> Any:
        """Predict one BGR frame."""
        image = torch.from_numpy(np.ascontiguousarray(frame[:, :, ::-1])).permute(2, 0, 1)
        with torch.inference_mode(), autocast(device, pcfg["amp"]):
            return model([to_float(image, device)])

    return benchmark(predict_one, images, n=BENCH_IMAGES)


def overfit(cfg: Mapping[str, Any], device: torch.device, n_images: int) -> float:
    """Memorise n train images in 150 constant-lr iterations without augmentation and require mAP above 0.5."""
    model = build_model(cfg)
    configure_patches(cfg, model)
    dataset = CocoDetectionDataset(prep_root(), cfg.get("data_variant") or DEFAULT_VARIANT, augment=False)
    idx = pick(range(len(dataset)), n_images)
    tcfg = {**settings(cfg, "train", TRAIN_DEFAULTS), **OVERFIT_TRAIN}
    Trainer(model, tcfg, Subset(dataset, idx), device, OVERFIT_ITERS, decay=False).fit()
    preds = predict(model, dataset.items(idx), device, settings(cfg, "predict", PREDICT_DEFAULTS))
    value = evaluate(preds, dataset.coco_subset(idx), breakdown=False)["map"] if len(preds) else 0.0
    LOG.info("overfit %s on %d images: mAP %.4f", cfg["exp_id"], n_images, value or 0.0)
    if not value or value <= OVERFIT_MIN_MAP:
        raise AssertionError(f"overfit mAP {value} is not above {OVERFIT_MIN_MAP}")
    return value


def run(cfg: Mapping[str, Any], device: torch.device, smoke: bool = False, resume: bool = False, weights: Path | None = None) -> dict[str, Any]:
    """Train unless weights are given, predict every split with the last weights, benchmark and finalise the run."""
    run_id = f"{cfg['exp_id']}_smoke" if smoke else cfg["exp_id"]
    out_dir = out_root() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    model = build_model(cfg, pretrained=weights is None).to(device)
    configure_patches(cfg, model)
    info: dict[str, Any] = {"train_hours": None, "gpu_hours": None, "epochs_done": None, "ms_per_image": None}
    if weights is None:
        progress = train(model, cfg, run_id, out_dir, device, smoke, resume)
        hours = progress.train_seconds / 3600
        info.update(train_hours=hours, gpu_hours=hours if device.type == "cuda" else 0.0, epochs_done=progress.epochs_done)
        LOG.info("trained %d iterations, %.2f epochs, %.2f h", progress.iteration, progress.epochs_done, hours)
    else:
        info["epochs_done"] = load_weights(model, weights)
    pcfg = settings(cfg, "predict", PREDICT_DEFAULTS)
    for split in cfg.get("splits") or DEFAULT_SPLITS:
        items = pick(split_images(split), SMOKE_IMAGES if smoke else None)
        preds = predict(model, items, device, pcfg)
        LOG.info("%s: %d images, %d boxes", split, len(items), len(preds))
        save_preds(preds, preds_path(out_dir, split))
    if not smoke:
        try:
            info["ms_per_image"] = measure_speed(model, device, pcfg)
        except Exception:  # noqa: BLE001
            LOG.exception("speed benchmark failed, ms_per_image left empty")
    return finalize_run(run_id, cfg, out_dir, info)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cuda:0, cpu or a GPU index; default cuda:0 else cpu")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU-friendly end-to-end run")
    parser.add_argument("--set", nargs="*", default=[], metavar="a.b=value", help="config overrides")
    parser.add_argument("--resume", action="store_true", help="continue from <OUT_ROOT>/<exp>/last.pt")
    parser.add_argument("--skip-train", action="store_true", help="predict and finalise with --weights")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--overfit", type=int, default=None, metavar="N", help="overfit N train images and check mAP")
    args = parser.parse_args(argv)
    if args.skip_train and not args.weights:
        parser.error("--skip-train needs --weights")
    if args.resume and args.skip_train:
        parser.error("--resume and --skip-train exclude each other")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run one torchvision experiment, or the overfit check, from the command line."""
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    device = torch_device(args.device)
    if args.overfit:
        seed_everything(SEED)
        overfit(cfg, device, args.overfit)
        return
    run(cfg, device, args.smoke, args.resume, args.weights if args.skip_train else None)


if __name__ == "__main__":
    main()
