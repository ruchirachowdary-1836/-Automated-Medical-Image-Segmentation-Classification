"""Evaluation metrics (NumPy; operate on whole volumes).

Empty-region convention for Dice / IoU (BraTS): if a region is absent in BOTH prediction and ground
truth the score is 1.0; if only one is empty the score is 0.0. Sensitivity is NaN when the ground truth
region is empty (no positives exist); specificity is NaN when the whole volume is positive.
HD95 is 0.0 when both masks are empty and NaN when exactly one is empty (distance undefined).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .config import REGIONS


def region_mask(labels: np.ndarray, ids: tuple[int, ...] | list[int]) -> np.ndarray:
    return np.isin(labels, list(ids))


def dice(p: np.ndarray, g: np.ndarray) -> float:
    p, g = p.astype(bool), g.astype(bool)
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0
    return float(2.0 * (p & g).sum() / s)


def iou(p: np.ndarray, g: np.ndarray) -> float:
    p, g = p.astype(bool), g.astype(bool)
    u = (p | g).sum()
    if u == 0:
        return 1.0
    return float((p & g).sum() / u)


def sensitivity(p: np.ndarray, g: np.ndarray) -> float:
    p, g = p.astype(bool), g.astype(bool)
    pos = g.sum()
    return float("nan") if pos == 0 else float((p & g).sum() / pos)


def specificity(p: np.ndarray, g: np.ndarray) -> float:
    p, g = p.astype(bool), g.astype(bool)
    neg = (~g).sum()
    return float("nan") if neg == 0 else float((~p & ~g).sum() / neg)


def hd95(p: np.ndarray, g: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """95th percentile symmetric surface distance in the units of `spacing` (mm)."""
    p, g = p.astype(bool), g.astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("nan")
    ps = p & ~ndimage.binary_erosion(p)
    gs = g & ~ndimage.binary_erosion(g)
    d_to_g = ndimage.distance_transform_edt(~gs, sampling=spacing)
    d_to_p = ndimage.distance_transform_edt(~ps, sampling=spacing)
    d = np.concatenate([d_to_g[ps], d_to_p[gs]])
    return float(np.percentile(d, 95))


def volume_ml(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Physical volume: voxel count x voxel volume (mm^3) / 1000 -> mL."""
    return float(mask.astype(bool).sum() * float(np.prod(spacing)) / 1000.0)


def volume_error(p: np.ndarray, g: np.ndarray) -> float:
    """Relative absolute volume error |Vp - Vg| / Vg (NaN if ground truth is empty)."""
    vg = int(g.astype(bool).sum())
    vp = int(p.astype(bool).sum())
    return float("nan") if vg == 0 else abs(vp - vg) / vg


def evaluate_case(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    """Per-region metrics for one patient volume (labels already remapped to 0..3)."""
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs ground truth {gt.shape}")
    out: dict[str, dict[str, float]] = {}
    for name, ids in REGIONS.items():
        p, g = region_mask(pred, ids), region_mask(gt, ids)
        out[name] = {
            "dice": dice(p, g), "iou": iou(p, g), "sensitivity": sensitivity(p, g),
            "specificity": specificity(p, g), "hd95_mm": hd95(p, g, spacing),
            "volume_pred_ml": volume_ml(p, spacing), "volume_gt_ml": volume_ml(g, spacing),
            "volume_error": volume_error(p, g),
        }
    return out


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """Rows = ground truth class, cols = predicted class (voxel counts)."""
    idx = gt.astype(np.int64).ravel() * n_classes + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


# ---- calibration ----------------------------------------------------------------------------------
def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf = np.asarray(conf, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = ids == b
        n = int(m.sum())
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": n,
                     "confidence": float(conf[m].mean()) if n else None,
                     "accuracy": float(correct[m].mean()) if n else None})
    return bins


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    total = np.asarray(conf).size
    if total == 0:
        return float("nan")
    return float(sum(b["count"] / total * abs(b["accuracy"] - b["confidence"])
                     for b in reliability_bins(conf, correct, n_bins) if b["count"]))


def brier_score(probs: np.ndarray, target: np.ndarray) -> float:
    """Multi-class Brier score. probs (N, C), target (N,) integer labels."""
    probs = np.asarray(probs, dtype=np.float64)
    oh = np.eye(probs.shape[1])[np.asarray(target).astype(int)]
    return float(((probs - oh) ** 2).sum(axis=1).mean())


def is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def nan_to_none(obj):
    """Recursively convert NaN/inf floats to None so results are valid JSON."""
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
