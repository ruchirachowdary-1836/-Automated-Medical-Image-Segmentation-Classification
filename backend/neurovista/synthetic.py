"""Synthetic BraTS-like phantoms for tests, smoke training and demo uploads.

These are NOT real MRI. They exist only to exercise the pipeline; never use them to claim accuracy.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .config import MODALITIES


def make_case(shape=(64, 64, 24), seed: int = 0, spacing=(1.0, 1.0, 2.0), tumor: bool = True):
    """Returns (volumes: dict[modality -> float32], seg_brats: uint8 with labels 0,1,2,4, affine)."""
    rng = np.random.default_rng(seed)
    h, w, d = shape
    xx, yy, zz = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing="ij")
    c = np.array([h / 2, w / 2, d / 2])
    r = np.array([h * 0.40, w * 0.40, d * 0.45])
    brain = (((xx - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((zz - c[2]) / r[2]) ** 2) < 1
    seg = np.zeros(shape, np.uint8)
    if tumor:
        t = c + rng.uniform(-0.12, 0.12, 3) * np.array([h, w, d])
        rad = rng.uniform(0.10, 0.16) * min(h, w)
        dist = np.sqrt((xx - t[0]) ** 2 + (yy - t[1]) ** 2 + ((zz - t[2]) * (spacing[2] / spacing[0])) ** 2)
        seg[(dist < rad) & brain] = 2  # edema
        seg[(dist < rad * 0.65) & brain] = 4  # enhancing
        seg[(dist < rad * 0.35) & brain] = 1  # necrotic core
    base = {"t1": 0.6, "t1ce": 0.6, "t2": 0.5, "flair": 0.4}
    vols = {}
    for m in MODALITIES:
        v = (base[m] + 0.05 * rng.standard_normal(shape)) * brain
        if m == "flair":
            v = v + 0.5 * ((seg == 2) | (seg == 1) | (seg == 4))
        if m == "t2":
            v = v + 0.4 * ((seg == 2) | (seg == 1))
        if m == "t1ce":
            v = v + 0.6 * (seg == 4)
        if m == "t1":
            v = v - 0.2 * (seg == 1)
        vols[m] = (np.clip(v, 0, None) * 1000 * brain).astype(np.float32)
    affine = np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])  # LPS-like, like BraTS
    return vols, seg, affine


def write_case(root: Path, name: str, **kw) -> Path:
    vols, seg, affine = make_case(**kw)
    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    for m, v in vols.items():
        nib.save(nib.Nifti1Image(v, affine), str(case / f"{name}_{m}.nii.gz"))
    nib.save(nib.Nifti1Image(seg, affine), str(case / f"{name}_seg.nii.gz"))
    return case


def write_dataset(root: Path, n: int = 10, shape=(64, 64, 24), seed: int = 0) -> list[Path]:
    return [write_case(root, f"SYN_{i:03d}", shape=shape, seed=seed + i) for i in range(n)]
