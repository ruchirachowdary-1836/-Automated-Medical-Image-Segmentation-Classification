"""Checkpoint save/load with architecture + metadata validation (weights_only loading)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .model import UNet, build_model

REQUIRED_KEYS = ("state_dict", "arch", "meta")


class CheckpointError(RuntimeError):
    pass


def save_checkpoint(path: Path, model: UNet, meta: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "arch": model.arch_config(),
               "meta": {**meta, "code_version": __version__}}
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[UNet, dict[str, Any]]:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path.name}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        raise CheckpointError(f"Checkpoint could not be loaded safely ({type(e).__name__})") from e
    if not isinstance(payload, dict) or any(k not in payload for k in REQUIRED_KEYS):
        raise CheckpointError("Checkpoint is missing required keys (state_dict, arch, meta)")
    arch = payload["arch"]
    mods = payload["meta"].get("modalities", ["t1", "t1ce", "t2", "flair"])
    if arch.get("n_classes") != 4 or arch.get("in_ch") != len(mods):
        raise CheckpointError(f"Unsupported architecture (need 4 classes and in_ch == len(modalities)): {arch}")
    model = build_model(arch)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as e:
        raise CheckpointError("Checkpoint weights do not match the declared architecture") from e
    model.to(device).eval()
    return model, payload["meta"]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
