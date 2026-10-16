"""Safe NIfTI loading and validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel import orientations as ornt_mod

from .config import MODALITIES, Settings


class InputValidationError(ValueError):
    """User-facing validation failure. `code` is a stable machine-readable identifier."""

    def __init__(self, message: str, code: str = "invalid_input", modality: str | None = None):
        super().__init__(message)
        self.code, self.modality = code, modality


@dataclass
class Scan:
    volumes: dict[str, np.ndarray]  # float32, shape (H, W, D), native orientation
    affine: np.ndarray
    spacing: tuple[float, float, float]
    axcodes: tuple[str, str, str]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.volumes[MODALITIES[0]].shape)  # type: ignore[return-value]


def has_nifti_extension(name: str) -> bool:
    n = name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def load_nifti(path: Path, modality: str, settings: Settings) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    if not has_nifti_extension(path.name) and not has_nifti_extension(str(path)):
        raise InputValidationError(f"{modality}: expected a .nii or .nii.gz file", "unsupported_file_type", modality)
    try:
        img = nib.load(str(path))
    except Exception as e:  # nibabel raises many types for corrupt files
        raise InputValidationError(f"{modality}: file is not a readable NIfTI volume ({type(e).__name__})",
                                   "unreadable_file", modality) from e
    shape = tuple(int(s) for s in img.shape)
    if len(shape) == 4 and shape[3] == 1:
        shape = shape[:3]
    if len(shape) != 3:
        raise InputValidationError(f"{modality}: expected a 3D volume, got shape {shape}", "bad_dimensions", modality)
    if (max(shape[0], shape[1]) > settings.max_inplane_dim or shape[2] > settings.max_depth
            or min(shape[0], shape[1]) < settings.min_inplane_dim or shape[2] < 1):
        raise InputValidationError(
            f"{modality}: volume shape {shape} is outside the supported range "
            f"(in-plane {settings.min_inplane_dim}-{settings.max_inplane_dim}, depth <= {settings.max_depth})",
            "dimensions_out_of_range", modality)
    try:
        data = np.asarray(img.dataobj, dtype=np.float32).reshape(shape)
    except Exception as e:
        raise InputValidationError(f"{modality}: could not read voxel data ({type(e).__name__})",
                                   "unreadable_file", modality) from e
    if not np.isfinite(data).all():
        raise InputValidationError(f"{modality}: volume contains NaN or infinite values", "non_finite", modality)
    if not (data != 0).any():
        raise InputValidationError(f"{modality}: volume is empty (all zeros)", "empty_volume", modality)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if len(zooms) != 3 or not all(np.isfinite(z) and z > 0 for z in zooms):
        raise InputValidationError(f"{modality}: invalid voxel spacing {zooms}", "bad_spacing", modality)
    affine = np.asarray(img.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
        raise InputValidationError(f"{modality}: invalid spatial affine", "bad_affine", modality)
    return data, affine, zooms  # type: ignore[return-value]


def load_scan(paths: dict[str, Path], settings: Settings) -> Scan:
    """Load all four modalities and check that they share one spatial grid."""
    missing = [m for m in MODALITIES if m not in paths]
    if missing:
        raise InputValidationError(f"Missing required modalities: {', '.join(missing)}", "missing_modality")
    vols: dict[str, np.ndarray] = {}
    ref_aff: np.ndarray | None = None
    ref_zoom: tuple[float, float, float] | None = None
    for m in MODALITIES:
        data, aff, zoom = load_nifti(paths[m], m, settings)
        if ref_aff is None:
            ref_aff, ref_zoom = aff, zoom
            first = m
        else:
            if data.shape != vols[first].shape:
                raise InputValidationError(
                    f"{m} has shape {data.shape} but {first} has shape {vols[first].shape}; "
                    "all modalities must be co-registered on the same grid", "shape_mismatch", m)
            if not np.allclose(aff, ref_aff, atol=1e-2) or not np.allclose(zoom, ref_zoom, atol=1e-3):
                raise InputValidationError(
                    f"{m} has a different orientation/spacing than {first}; "
                    "modalities must share one affine", "affine_mismatch", m)
        vols[m] = data
    assert ref_aff is not None and ref_zoom is not None
    axcodes = tuple(nib.aff2axcodes(ref_aff))
    if axcodes[2] not in ("S", "I"):
        raise InputValidationError(
            f"Third array axis must be the superior-inferior (axial slice) axis; this scan is {''.join(axcodes)}. "
            "Reorient to an axial BraTS-style layout before upload.", "unsupported_orientation")
    return Scan(vols, ref_aff, ref_zoom, axcodes)  # type: ignore[arg-type]


def reorient_to_ras(arr: np.ndarray, affine: np.ndarray, spacing: tuple[float, float, float]):
    """Return (array_in_RAS, spacing_in_RAS). Works on (H, W, D) or (C, H, W, D) when channel_first is False."""
    ornt = ornt_mod.io_orientation(affine)
    target = ornt_mod.axcodes2ornt(("R", "A", "S"))
    transform = ornt_mod.ornt_transform(ornt, target)
    out = ornt_mod.apply_orientation(arr, transform)
    perm = [int(i) for i in transform[:, 0]]  # new axis i comes from old axis perm[i]
    new_spacing = tuple(spacing[p] for p in perm)
    return np.ascontiguousarray(out), new_spacing


def save_mask_nifti(labels_brats: np.ndarray, affine: np.ndarray, path: Path) -> None:
    img = nib.Nifti1Image(labels_brats.astype(np.uint8), affine)
    img.header.set_data_dtype(np.uint8)
    nib.save(img, str(path))
