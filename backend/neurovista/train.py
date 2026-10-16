"""Training loop. Run as a module so the API can launch/cancel it as a subprocess:

    python -m neurovista.train --run-dir var/runs/<id>      (reads <run-dir>/config.json)
"""
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import REGIONS
from .data import SliceDataset
from .losses import build_loss
from .model import UNet, count_parameters

DEFAULTS = {
    "name": "run", "preset": None, "processed_dir": "data/processed", "seed": 42, "device": "auto",
    "model": "unet", "base": 32, "dropout": 0.2, "in_ch": 4, "modalities": ["t1", "t1ce", "t2", "flair"],
    "loss": "dice_ce", "optimizer": "adamw", "lr": 3e-4, "weight_decay": 1e-4, "scheduler": "cosine",
    "batch_size": 16, "epochs": 60, "patience": 15, "grad_clip": 1.0, "amp": True, "augment": True,
    "num_workers": 0, "max_train_batches": None, "resume": None,
}


def seed_all(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def validate(model, loader, dev, loss_fn, channels=None) -> dict:
    """Dice accumulated over the whole validation set (sum 2|P∩G| / sum(|P|+|G|)), per region."""
    model.eval()
    tot = {k: [0.0, 0.0] for k in REGIONS}
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = _select(x, channels).to(dev), y.to(dev)
        logits = model(x)
        loss_sum += float(loss_fn(logits, y)) * x.shape[0]
        n += x.shape[0]
        pred = logits.argmax(1)
        for k, ids in REGIONS.items():
            t = torch.tensor(ids, device=dev)
            p, g = torch.isin(pred, t), torch.isin(y, t)
            tot[k][0] += 2.0 * float((p & g).sum()); tot[k][1] += float(p.sum() + g.sum())
    dice = {k: (a / b if b > 0 else 1.0) for k, (a, b) in tot.items()}
    return {"val_loss": loss_sum / max(n, 1), "dice": dice, "mean_dice": float(np.mean(list(dice.values())))}


def _select(x: torch.Tensor, channels: list[int] | None) -> torch.Tensor:
    return x if channels is None else x[:, channels]


def _write_status(run_dir: Path, **kw) -> None:
    p = run_dir / "status.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw)
    p.write_text(json.dumps(cur, indent=1))


def run_training(run_dir: Path) -> dict:
    cfg = {**DEFAULTS, **json.loads((run_dir / "config.json").read_text())}
    mods = cfg["modalities"]
    all_mods = ["t1", "t1ce", "t2", "flair"]
    channels = None if mods == all_mods else [all_mods.index(m) for m in mods]
    in_ch = len(mods)
    dev = torch.device("cuda" if (cfg["device"] == "auto" and torch.cuda.is_available()) else
                       ("cpu" if cfg["device"] == "auto" else cfg["device"]))
    seed_all(int(cfg["seed"]))
    _write_status(run_dir, state="running", started=time.time(), device=str(dev))
    processed = Path(cfg["processed_dir"])
    aug = {"gamma": True, "noise": True} if cfg["augment"] else None
    tr = SliceDataset(processed, "train", train=bool(cfg["augment"]), aug_cfg=aug, seed=cfg["seed"])
    va = SliceDataset(processed, "val", train=False)
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError(f"Empty split (train={len(tr)}, val={len(va)}). Check the dataset and splits.json")
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    tl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                    generator=g, drop_last=len(tr) > cfg["batch_size"])
    vl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    model = UNet(in_ch=in_ch, n_classes=4, base=cfg["base"], p=cfg["dropout"],
                 attention=cfg["model"] == "attention_unet").to(dev)
    loss_fn = build_loss(cfg["loss"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"]) if cfg["scheduler"] == "cosine"
             else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5))
    use_amp = bool(cfg["amp"]) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, bad, start_epoch = -1.0, 0, 0
    if cfg.get("resume"):
        ck = torch.load(cfg["resume"], map_location=dev, weights_only=True)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
    hist_path = run_dir / "history.jsonl"
    ckpt_dir = run_dir / "checkpoints"
    meta_base = {"model": cfg["model"], "params": count_parameters(model), "seed": cfg["seed"],
                 "loss": cfg["loss"], "modalities": mods, "run": run_dir.name,
                 "trained_on": cfg.get("dataset_label", str(processed)), "synthetic": bool(cfg.get("synthetic", False))}
    _write_status(run_dir, state="running", params=count_parameters(model), train_slices=len(tr), val_slices=len(va),
                  total_epochs=cfg["epochs"])
    for epoch in range(start_epoch, cfg["epochs"]):
        if (run_dir / "CANCEL").exists():
            _write_status(run_dir, state="cancelled"); return {"state": "cancelled"}
        model.train(); t0, run_loss, nb = time.time(), 0.0, 0
        for x, y in tl:
            x, y = _select(x, channels).to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast(dev.type, enabled=use_amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError("GPU out of memory: reduce batch_size or base channels")
            run_loss += float(loss.detach()); nb += 1
            if cfg["max_train_batches"] and nb >= cfg["max_train_batches"]:
                break
            if (run_dir / "CANCEL").exists():
                break
        val = validate(model, vl, dev, loss_fn, channels)
        lr = opt.param_groups[0]["lr"]
        sched.step() if cfg["scheduler"] == "cosine" else sched.step(val["mean_dice"])
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(nb, 1), "val_loss": val["val_loss"],
               "dice": val["dice"], "mean_dice": val["mean_dice"], "lr": lr, "seconds": round(time.time() - t0, 2)}
        with hist_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        improved = val["mean_dice"] > best
        if improved:
            best, bad = val["mean_dice"], 0
            save_checkpoint(ckpt_dir / "best.pt", model, {**meta_base, "epoch": epoch + 1, "val_mean_dice": best,
                                                          "val_dice": val["dice"]})
        else:
            bad += 1
        save_checkpoint(ckpt_dir / "last.pt", model, {**meta_base, "epoch": epoch + 1},
                        {"optimizer": opt.state_dict(), "epoch": epoch, "best": best})
        extra = {"best_epoch": epoch + 1} if improved else {}
        _write_status(run_dir, epoch=epoch + 1, best_mean_dice=best, updated=time.time(), **extra)
        if bad >= cfg["patience"]:
            _write_status(run_dir, state="completed", stopped_early=True); return {"state": "completed"}
    _write_status(run_dir, state="completed", stopped_early=False, finished=time.time())
    return {"state": "completed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    a = ap.parse_args()
    try:
        run_training(a.run_dir)
    except Exception as e:  # surfaced to the UI through status.json + train.log
        (a.run_dir / "train.log").open("a").write(traceback.format_exc())
        _write_status(a.run_dir, state="failed", error=f"{type(e).__name__}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
