"""Shared preprocessing. Training and inference MUST both go through these functions.

Conventions (from the build guide):
  * per-modality z-score computed inside the brain mask (voxels > 0), after 0.5-99.5 percentile clipping
  * background voxels are set to 0 after normalisation
  * BraTS labels 0,1,2,4 are remapped to contiguous 0,1,2,3
  * axial slices run along the LAST array axis; in-plane grid is cropped 240 -> 192 when lossless
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MODALITIES

CLIP_PERCENTILES = (0.5, 99.5)
TARGET_INPLANE = 192
MULTIPLE = 16  # four pooling levels


class PreprocessError(ValueError):
    """Raised when data cannot be preprocessed safely."""


def zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    m = v > 0
    if not m.any():
        return np.zeros_like(v, dtype=np.float32)
    lo, hi = np.percentile(v[m], CLIP_PERCENTILES)
    v = np.clip(v, lo, hi)
    mu, sd = float(v[m].mean()), float(v[m].std()) + 1e-8
    out = (v - mu) / sd
    out[~m] = 0
    return out.astype(np.float32)


def normalize_volumes(volumes: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the four modalities in the canonical order -> (4, H, W, D) float32."""
    missing = [m for m in MODALITIES if m not in volumes]
    if missing:
        raise PreprocessError(f"Missing modalities: {', '.join(missing)}")
    shapes = {volumes[m].shape for m in MODALITIES}
    if len(shapes) != 1:
        raise PreprocessError(f"Modalities have different shapes: {sorted(shapes)}")
    return np.stack([zscore(volumes[m]) for m in MODALITIES])


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """BraTS labels {0,1,2,4} -> {0,1,2,3}. Any other value is an error."""
    seg = np.rint(np.asarray(seg)).astype(np.int16)
    bad = np.setdiff1d(np.unique(seg), [0, 1, 2, 3, 4])
    if bad.size:
        raise PreprocessError(f"Unexpected label values: {bad.tolist()}")
    out = seg.copy()
    out[seg == 4] = 3
    if (seg == 3).any():
        raise PreprocessError("Label 3 present: input already remapped or not BraTS-style (expected 0,1,2,4)")
    return out.astype(np.uint8)


@dataclass(frozen=True)
class GridBox:
    """Centered crop/pad box mapping the original in-plane grid to the model grid."""

    orig: tuple[int, int]
    target: tuple[int, int]

    def _axis(self, o: int, t: int) -> tuple[int, int, int, int]:
        # returns (src_start, src_end, dst_start, dst_end)
        if t <= o:
            s = (o - t) // 2
            return s, s + t, 0, t
        d = (t - o) // 2
        return 0, o, d, d + o

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Crop/pad axes (-3, -2) of an array shaped (..., H, W, D)."""
        h0, h1, th0, th1 = self._axis(self.orig[0], self.target[0])
        w0, w1, tw0, tw1 = self._axis(self.orig[1], self.target[1])
        shape = list(arr.shape)
        shape[-3], shape[-2] = self.target
        out = np.zeros(shape, dtype=arr.dtype)
        out[..., th0:th1, tw0:tw1, :] = arr[..., h0:h1, w0:w1, :]
        return out

    def restore(self, arr: np.ndarray, fill: float = 0) -> np.ndarray:
        """Inverse of apply: place model-grid array back into the original grid."""
        h0, h1, th0, th1 = self._axis(self.orig[0], self.target[0])
        w0, w1, tw0, tw1 = self._axis(self.orig[1], self.target[1])
        shape = list(arr.shape)
        shape[-3], shape[-2] = self.orig
        out = np.full(shape, fill, dtype=arr.dtype)
        out[..., h0:h1, w0:w1, :] = arr[..., th0:th1, tw0:tw1, :]
        return out


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def plan_grid(stack: np.ndarray) -> GridBox:
    """Choose the model grid for a (C,H,W,D) stack.

    Crop to 192x192 only when that discards no non-zero voxel (BraTS 240x240 has empty borders);
    otherwise pad up to a multiple of 16 so nothing is lost.
    """
    h, w = stack.shape[-3], stack.shape[-2]
    th, tw = _round_up(h, MULTIPLE), _round_up(w, MULTIPLE)
    if h > TARGET_INPLANE or w > TARGET_INPLANE:
        cand = GridBox((h, w), (min(h, TARGET_INPLANE), min(w, TARGET_INPLANE)))
        if np.array_equal(cand.restore(cand.apply(stack != 0)), stack != 0):
            return GridBox((h, w), (_round_up(cand.target[0], MULTIPLE), _round_up(cand.target[1], MULTIPLE)))
    return GridBox((h, w), (th, tw))


def prepare_volume(volumes: dict[str, np.ndarray]) -> tuple[np.ndarray, GridBox]:
    """Raw modality volumes -> model-ready (4, H', W', D) tensor + grid box."""
    stack = normalize_volumes(volumes)
    box = plan_grid(stack)
    return box.apply(stack), box


def brain_mask(volumes: dict[str, np.ndarray]) -> np.ndarray:
    return np.any(np.stack([volumes[m] > 0 for m in MODALITIES]), axis=0)


def extract_slices(stack: np.ndarray, seg: np.ndarray | None, rng: np.random.Generator, keep_empty: float = 0.1):
    """Yield (z, x[4,H,W], y[H,W]) for the 2D baseline: all tumor slices + ~keep_empty of empty ones."""
    for z in range(stack.shape[-1]):
        x = stack[..., z]
        if np.abs(x).max() == 0:
            continue
        y = seg[..., z] if seg is not None else np.zeros(x.shape[1:], dtype=np.uint8)
        if y.max() == 0 and rng.random() > keep_empty:
            continue
        yield z, x, y
