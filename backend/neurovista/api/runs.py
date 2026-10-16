"""Training runs and experiment registry (file-based; no database needed)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from ..checkpoint import read_json
from ..config import Settings
from ..train import DEFAULTS

PRESETS: list[dict[str, Any]] = [
    {"id": "baseline_ce", "title": "U-Net + cross-entropy (baseline)", "kind": "training",
     "overrides": {"model": "unet", "loss": "ce", "augment": False}},
    {"id": "dice_ce", "title": "U-Net + Dice + cross-entropy", "kind": "training",
     "overrides": {"model": "unet", "loss": "dice_ce", "augment": False}},
    {"id": "augmentation", "title": "Dice+CE with augmentation", "kind": "training",
     "overrides": {"model": "unet", "loss": "dice_ce", "augment": True}},
    {"id": "attention_unet", "title": "Attention U-Net", "kind": "training",
     "overrides": {"model": "attention_unet", "loss": "dice_ce", "augment": True}},
    {"id": "flair_only", "title": "FLAIR-only input vs four channels", "kind": "training",
     "overrides": {"model": "unet", "loss": "dice_ce", "augment": True, "modalities": ["flair"], "in_ch": 1}},
    {"id": "tta_postproc", "title": "Test-time augmentation + post-processing", "kind": "evaluation",
     "note": "Evaluation-time options applied to an existing run (evaluate with TTA / min component size)."},
    {"id": "2d_vs_3d", "title": "2D vs 3D baseline", "kind": "unsupported",
     "note": "A 3D model is not implemented in this release; only the 2D slice-wise U-Net is available."},
    {"id": "mc_dropout_vs_error", "title": "MC-dropout uncertainty vs segmentation error", "kind": "evaluation",
     "note": "Evaluate an existing run with MC-dropout; the report includes AUROC of uncertainty vs voxel error."},
]
_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class RunManager:
    def __init__(self, settings: Settings):
        self.s = settings
        settings.runs_dir.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        if not _ID.match(run_id):
            raise KeyError(run_id)
        p = self.s.runs_dir / run_id
        if not p.is_dir():
            raise KeyError(run_id)
        return p

    def dataset_dir(self, name: str) -> Path:
        if not _ID.match(name):
            raise ValueError("Invalid dataset name")
        return self.s.datasets_dir / name

    def datasets(self) -> list[dict]:
        out = []
        if self.s.datasets_dir.is_dir():
            for d in sorted(p for p in self.s.datasets_dir.iterdir() if p.is_dir()):
                man = read_json(d / "splits.json")
                out.append({"name": d.name, "has_raw": (d / "raw").is_dir(),
                            "has_processed": (d / "processed").is_dir() and any((d / "processed").glob("*.npz")),
                            "splits": {k: len(v) for k, v in man["splits"].items()} if man else None,
                            "seed": man.get("seed") if man else None})
        return out

    def create(self, cfg: dict) -> str:
        if not self.s.training_enabled:
            raise PermissionError("Training is disabled on this deployment (NV_TRAINING_ENABLED=false)")
        preset = next((p for p in PRESETS if p["id"] == cfg.get("preset")), None)
        if cfg.get("preset") and not preset:
            raise ValueError("Unknown preset")
        if preset and preset["kind"] != "training":
            raise ValueError(f"Preset '{preset['id']}' is not a training preset")
        merged = {**DEFAULTS, **(preset["overrides"] if preset else {}), **{k: v for k, v in cfg.items() if v is not None}}
        for k in ("resume",):
            merged.pop(k, None)
        ds = self.dataset_dir(str(merged.get("dataset", "")))
        if not (ds / "processed").is_dir() or not any((ds / "processed").glob("*.npz")):
            raise FileNotFoundError("Dataset has no processed slices. See DATASET_GUIDE.md (neurovista prepare).")
        merged["processed_dir"] = str(ds / "processed")
        merged["dataset_label"] = merged["dataset"]
        self._validate(merged)
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = self.s.runs_dir / run_id
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(merged, indent=1))
        (d / "status.json").write_text(json.dumps({"state": "queued", "created": time.time()}))
        self._spawn(d)
        return run_id

    @staticmethod
    def _validate(c: dict) -> None:
        if c["model"] not in ("unet", "attention_unet"):
            raise ValueError("model must be unet or attention_unet")
        if c["loss"] not in ("ce", "dice", "dice_ce", "focal_tversky"):
            raise ValueError("unsupported loss")
        if not (1 <= int(c["epochs"]) <= 1000 and 1 <= int(c["batch_size"]) <= 256 and 0 < float(c["lr"]) < 1):
            raise ValueError("epochs/batch_size/lr out of range")
        if c["device"] not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if c["optimizer"] != "adamw" or c["scheduler"] not in ("cosine", "plateau"):
            raise ValueError("optimizer must be adamw; scheduler cosine|plateau")
        if not set(c["modalities"]) <= {"t1", "t1ce", "t2", "flair"} or not c["modalities"]:
            raise ValueError("invalid modalities")

    def _spawn(self, d: Path) -> None:
        log = (d / "train.log").open("a")
        p = subprocess.Popen([sys.executable, "-m", "neurovista.train", "--run-dir", str(d)],
                             stdout=log, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[2]))
        st = read_json(d / "status.json", {})
        st.update(pid=p.pid, state="running")
        (d / "status.json").write_text(json.dumps(st))

    def cancel(self, run_id: str) -> None:
        d = self.path(run_id)
        (d / "CANCEL").write_text("1")

    def resume(self, run_id: str) -> None:
        d = self.path(run_id)
        last = d / "checkpoints" / "last.pt"
        if not last.exists():
            raise FileNotFoundError("No last.pt checkpoint to resume from")
        cfg = read_json(d / "config.json", {})
        cfg["resume"] = str(last)
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        (d / "CANCEL").unlink(missing_ok=True)
        self._spawn(d)

    def get(self, run_id: str) -> dict:
        d = self.path(run_id)
        st = read_json(d / "status.json", {})
        hist = []
        hp = d / "history.jsonl"
        if hp.exists():
            hist = [json.loads(line) for line in hp.read_text().splitlines() if line.strip()]
        log = ""
        lp = d / "train.log"
        if lp.exists():
            log = "\n".join(lp.read_text(errors="replace").splitlines()[-40:])
        cfg = read_json(d / "config.json", {})
        cfg.pop("processed_dir", None)
        evals = {p.stem.replace("eval_", ""): read_json(p) for p in d.glob("eval_*.json")}
        st = self._reconcile(d, st)
        return {"id": run_id, "config": cfg, "status": st, "history": hist, "log": log, "evals": evals,
                "has_best": (d / "checkpoints" / "best.pt").exists()}

    @staticmethod
    def _reconcile(d: Path, st: dict) -> dict:
        """If the process died without writing a final state, mark it failed."""
        if st.get("state") == "running" and st.get("pid") and not _alive(int(st["pid"])):
            st = {**st, "state": "failed", "error": st.get("error") or "training process exited unexpectedly"}
            (d / "status.json").write_text(json.dumps(st))
        return st

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.s.runs_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            try:
                r = self.get(d.name)
            except Exception:
                continue
            h = r["history"]
            out.append({"id": r["id"], "name": r["config"].get("name"), "preset": r["config"].get("preset"),
                        "model": r["config"].get("model"), "loss": r["config"].get("loss"),
                        "state": r["status"].get("state"), "epochs_done": len(h),
                        "best_mean_dice": r["status"].get("best_mean_dice"),
                        "evals": sorted(r["evals"].keys()), "synthetic": bool(r["config"].get("synthetic")),
                        "created": r["status"].get("created") or r["status"].get("started")})
        return out

    def experiments(self) -> list[dict]:
        runs = self.list()
        res = []
        for p in PRESETS:
            mine = [r for r in runs if r["preset"] == p["id"]]
            if p["kind"] == "unsupported":
                status = "not_implemented"
            elif p["kind"] == "evaluation":
                ev = [r for r in runs if any(k.startswith(p["id"]) for k in r["evals"])]
                status, mine = ("completed" if ev else "planned"), ev
            else:
                done = [r for r in mine if r["state"] == "completed" and r["evals"]]
                status = ("completed" if done else "running" if any(r["state"] == "running" for r in mine)
                          else "trained_not_evaluated" if any(r["state"] == "completed" for r in mine) else "planned")
            res.append({**{k: v for k, v in p.items() if k != "overrides"}, "status": status,
                        "runs": [r["id"] for r in mine]})
        return res

    def activate(self, run_id: str) -> Path:
        src = self.path(run_id) / "checkpoints" / "best.pt"
        if not src.exists():
            raise FileNotFoundError("Run has no best.pt checkpoint")
        dst = self.s.checkpoint_file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        cal = self.path(run_id) / "calibration.json"
        cal_dst = dst.parent / "calibration.json"
        if cal.exists():
            shutil.copy2(cal, cal_dst)
        else:
            cal_dst.unlink(missing_ok=True)
        return dst

    def evaluate(self, run_id: str, split: str, tta: bool, mc: int, min_component: int, label: str) -> None:
        d = self.path(run_id)
        cfg = read_json(d / "config.json", {})
        ds = self.dataset_dir(cfg.get("dataset", ""))
        if split not in ("val", "test") or not (ds / "raw").is_dir():
            raise ValueError("split must be val|test and the dataset needs raw cases")
        if not re.match(r"^[A-Za-z0-9_\-]{1,40}$", label):
            raise ValueError("invalid label")
        out = d / f"eval_{label}.json"
        cmd = [sys.executable, "-m", "neurovista.evaluate", "--checkpoint", str(d / "checkpoints" / "best.pt"),
               "--raw-dir", str(ds / "raw"), "--processed-dir", str(ds / "processed"), "--split", split,
               "--out", str(out), "--mc", str(mc), "--min-component", str(min_component)] + (["--tta"] if tta else [])
        log = (d / "eval.log").open("a")
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[2]))


def _alive(pid: int) -> bool:
    try:
        import os
        if sys.platform == "win32":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False
