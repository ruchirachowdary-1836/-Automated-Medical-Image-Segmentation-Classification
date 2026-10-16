"""Reusable inference engine used by the API and the CLI.

Confidence definitions (documented in MODEL_CARD.md):
  * voxel confidence      = max softmax probability (a model score, NOT a guarantee of correctness)
  * region confidence     = mean, over voxels predicted inside the region, of the summed probability of the
                            region's classes (WT = classes 1+2+3, TC = 1+3, ET = 3)
  * scan confidence       = mean voxel confidence over predicted whole-tumor voxels (None if no tumor predicted)
  * uncertainty           = predictive entropy of the (TTA / MC averaged) probabilities, normalised to [0, 1]
  * mutual information    = epistemic part, only when MC-dropout is enabled
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from .checkpoint import load_checkpoint
from .config import CLASS_NAMES, MODALITIES, NUM_CLASSES, REGIONS, Settings
from .metrics import volume_ml
from .nifti_io import Scan
from .preprocess import GridBox, brain_mask, prepare_volume

LOG_C = math.log(NUM_CLASSES)


@dataclass
class InferenceOptions:
    tta: bool = False
    mc_samples: int = 0  # 0 disables MC-dropout
    min_component_voxels: int = 0
    batch_size: int = 8


@dataclass
class InferenceResult:
    labels: np.ndarray  # (H, W, D) uint8, contiguous classes 0..3, original grid
    confidence: np.ndarray  # float32 max-softmax, 0 outside brain
    uncertainty: np.ndarray  # float32 normalised entropy in [0,1]
    mutual_info: np.ndarray | None
    summary: dict[str, Any] = field(default_factory=dict)


def resolve_device(pref: str) -> torch.device:
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def labels_to_brats(labels: np.ndarray) -> np.ndarray:
    out = labels.copy()
    out[labels == 3] = 4
    return out


def filter_small_components(labels: np.ndarray, min_voxels: int) -> tuple[np.ndarray, int]:
    """Remove 3D connected whole-tumor components smaller than `min_voxels`. Returns (labels, n_removed)."""
    if min_voxels <= 0:
        return labels, 0
    cc, n = ndimage.label(labels > 0, structure=np.ones((3, 3, 3)))
    if n == 0:
        return labels, 0
    sizes = ndimage.sum(np.ones_like(cc), cc, index=np.arange(1, n + 1))
    remove = np.flatnonzero(sizes < min_voxels) + 1
    if remove.size == 0:
        return labels, 0
    out = labels.copy()
    out[np.isin(cc, remove)] = 0
    return out, int(remove.size)


class Segmenter:
    def __init__(self, model: torch.nn.Module, meta: dict[str, Any], device: torch.device, settings: Settings,
                 checkpoint_name: str = "checkpoint"):
        self.model, self.meta, self.device, self.settings = model, meta, device, settings
        self.checkpoint_name = checkpoint_name
        mods = list(meta.get("modalities", MODALITIES))
        self.channels = None if mods == list(MODALITIES) else [MODALITIES.index(m) for m in mods]
        self.has_dropout = any(isinstance(m, torch.nn.Dropout2d) and m.p > 0 for m in model.modules())

    @classmethod
    def from_checkpoint(cls, path: Path, settings: Settings) -> "Segmenter":
        device = resolve_device(settings.device)
        model, meta = load_checkpoint(path, str(device))
        return cls(model, meta, device, settings, path.name)

    # ---- probability engines ---------------------------------------------------------------
    def _enable_dropout(self) -> None:
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout2d):
                m.train()

    @torch.no_grad()
    def _probs(self, x: torch.Tensor, tta: bool) -> torch.Tensor:
        p = self.model(x).softmax(1)
        if tta:  # horizontal-flip TTA (flip along W)
            pf = self.model(torch.flip(x, dims=[-1])).softmax(1)
            p = (p + torch.flip(pf, dims=[-1])) / 2
        return p

    @torch.no_grad()
    def _batch_probs(self, x: torch.Tensor, opts: InferenceOptions) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.model.eval()
        if opts.mc_samples > 1 and self.has_dropout:
            self._enable_dropout()
            try:
                samples = torch.stack([self._probs(x, opts.tta) for _ in range(opts.mc_samples)])
            finally:
                self.model.eval()
            mean = samples.mean(0)
            h_mean = -(mean * mean.clamp_min(1e-8).log()).sum(1)
            h_each = -(samples * samples.clamp_min(1e-8).log()).sum(2).mean(0)
            return mean, (h_mean - h_each).clamp_min(0) / LOG_C
        return self._probs(x, opts.tta), None

    # ---- main entry ---------------------------------------------------------------------------
    def predict(self, scan: Scan, opts: InferenceOptions | None = None) -> InferenceResult:
        opts = opts or InferenceOptions()
        if opts.mc_samples > 1 and not self.has_dropout:
            raise ValueError("MC-dropout requested but the model has no active dropout layers")
        t0 = time.perf_counter()
        stack, box = prepare_volume(scan.volumes)
        inside = np.any(stack != 0, axis=0)
        if self.channels is not None:
            stack = stack[self.channels]
        h, w, d = stack.shape[1:]
        labels = np.zeros((h, w, d), np.uint8)
        conf = np.zeros((h, w, d), np.float32)
        unc = np.zeros((h, w, d), np.float32)
        mi = np.zeros((h, w, d), np.float32) if opts.mc_samples > 1 else None
        region_prob = {r: np.zeros((h, w, d), np.float16) for r in REGIONS}
        todo = [z for z in range(d) if inside[..., z].any()]
        for i in range(0, len(todo), opts.batch_size):
            zs = todo[i:i + opts.batch_size]
            x = torch.from_numpy(np.stack([stack[..., z] for z in zs])).to(self.device)
            probs, mut = self._batch_probs(x, opts)
            cf, lab = probs.max(1)
            ent = -(probs * probs.clamp_min(1e-8).log()).sum(1) / LOG_C
            for j, z in enumerate(zs):
                labels[..., z] = lab[j].cpu().numpy()
                conf[..., z] = cf[j].cpu().numpy()
                unc[..., z] = ent[j].clamp(0, 1).cpu().numpy()
                if mi is not None and mut is not None:
                    mi[..., z] = mut[j].cpu().numpy()
                for r, ids in REGIONS.items():
                    region_prob[r][..., z] = probs[j, list(ids)].sum(0).cpu().numpy()
        labels[~inside] = 0
        labels, removed = filter_small_components(labels, opts.min_component_voxels)
        conf[~inside] = 0
        unc[~inside] = 0
        duration = time.perf_counter() - t0
        labels_o = box.restore(labels)
        conf_o = box.restore(conf)
        unc_o = box.restore(unc)
        mi_o = box.restore(mi) if mi is not None else None
        rp_o = {r: box.restore(v) for r, v in region_prob.items()}
        summary = self._summarize(labels_o, conf_o, unc_o, mi_o, rp_o, scan, opts, duration, removed, box)
        return InferenceResult(labels_o, conf_o, unc_o, mi_o, summary)

    def confidence_level(self, value: float | None) -> str | None:
        if value is None:
            return None
        hi, med = self._thresholds()["high"], self._thresholds()["medium"]
        return "high" if value >= hi else "medium" if value >= med else "low"

    def _thresholds(self) -> dict[str, Any]:
        cal = self.meta.get("calibration_thresholds")
        if cal:
            return {"high": float(cal["high"]), "medium": float(cal["medium"]), "source": "validation-calibrated"}
        return {"high": self.settings.confidence_high, "medium": self.settings.confidence_medium,
                "source": "configured-default (not calibrated)"}

    def _summarize(self, labels, conf, unc, mi, rp, scan: Scan, opts, duration, removed, box: GridBox) -> dict:
        spacing = scan.spacing
        regions: dict[str, Any] = {}
        for r, ids in REGIONS.items():
            m = np.isin(labels, ids)
            n = int(m.sum())
            rc = float(rp[r][m].astype(np.float32).mean()) if n else None
            regions[r] = {
                "voxels": n, "volume_ml": volume_ml(m, spacing), "confidence": rc,
                "confidence_level": self.confidence_level(rc),
                "mean_uncertainty": float(unc[m].mean()) if n else None,
            }
        wt = labels > 0
        scan_conf = float(conf[wt].mean()) if wt.any() else None
        per_class = {CLASS_NAMES[c]: {"voxels": int((labels == c).sum()),
                                      "volume_ml": volume_ml(labels == c, spacing)} for c in range(1, NUM_CLASSES)}
        bm = brain_mask(scan.volumes)
        warnings: list[str] = []
        if bm.mean() > 0.85:
            warnings.append("More than 85% of voxels are non-zero: the scan may not be skull-stripped "
                            "(heuristic check). The model was designed for skull-stripped BraTS-style input.")
        if not wt.any():
            warnings.append("No tumor voxels were predicted. This does not exclude disease; outputs require review.")
        return {
            "regions": regions, "classes": per_class,
            "scan_confidence": scan_conf, "scan_confidence_level": self.confidence_level(scan_conf),
            "scan_confidence_definition": "mean max-softmax probability over predicted whole-tumor voxels",
            "confidence_thresholds": self._thresholds(),
            "mean_uncertainty_wt": float(unc[wt].mean()) if wt.any() else None,
            "mean_mutual_information_wt": float(mi[wt].mean()) if (mi is not None and wt.any()) else None,
            "uncertainty_method": "predictive entropy (normalised)" + (
                f" + MC-dropout x{opts.mc_samples} mutual information" if mi is not None else ""),
            "inference_seconds": round(duration, 3),
            "device": str(self.device),
            "options": {"tta": opts.tta, "mc_samples": opts.mc_samples,
                        "min_component_voxels": opts.min_component_voxels,
                        "components_removed": removed},
            "preprocessing": {"normalisation": "z-score in brain mask after 0.5-99.5 percentile clipping",
                              "modality_order": list(MODALITIES), "orig_inplane": list(box.orig),
                              "model_inplane": list(box.target)},
            "spacing_mm": list(spacing), "shape": list(labels.shape), "orientation": "".join(scan.axcodes),
            "model": {"checkpoint": self.checkpoint_name, **{k: v for k, v in self.meta.items()
                                                              if k not in ("calibration_thresholds",)}},
            "warnings": warnings,
        }
