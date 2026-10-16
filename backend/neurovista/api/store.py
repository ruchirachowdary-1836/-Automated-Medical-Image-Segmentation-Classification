"""Result artifact store with a retention window. Uploaded scans are never persisted."""
from __future__ import annotations

import io
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import MODALITIES, Settings
from ..inference import InferenceResult, labels_to_brats
from ..metrics import nan_to_none
from ..nifti_io import Scan, reorient_to_ras, save_mask_nifti

REGION_COLORS = {"necrotic_non_enhancing_core": "#3b82f6", "edema": "#f5c542", "enhancing_tumor": "#ef4444"}
PALETTE = {1: (59, 130, 246), 2: (245, 197, 66), 3: (239, 68, 68)}
LAYERS = (*MODALITIES, "labels", "confidence", "uncertainty")


def _to_u8_window(v: np.ndarray) -> np.ndarray:
    m = v > 0
    if not m.any():
        return np.zeros(v.shape, np.uint8)
    lo, hi = np.percentile(v[m], [1, 99])
    out = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    out[~m] = 0
    return out.astype(np.uint8)


def display_slice(arr2d: np.ndarray) -> np.ndarray:
    """RAS slice (axis0=R, axis1=A) -> image with anterior up and patient-right on screen-left."""
    return arr2d[::-1, ::-1].T


def render_preview(flair: np.ndarray, labels: np.ndarray, z: int) -> bytes:
    g = display_slice(flair[:, :, z])
    rgb = np.stack([g] * 3, -1).astype(np.float32)
    lab = display_slice(labels[:, :, z])
    ov = rgb.copy()
    for c, col in PALETTE.items():
        ov[lab == c] = 0.45 * rgb[lab == c] + 0.55 * np.array(col)
    panel = np.concatenate([rgb, ov], axis=1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(panel).resize((panel.shape[1] * 2, panel.shape[0] * 2), Image.NEAREST).save(buf, "PNG")
    return buf.getvalue()


class JobStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self.root = settings.artifacts_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, job_id: str) -> Path:
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise KeyError(job_id)
        return self.root / job_id

    def save(self, scan: Scan, res: InferenceResult, options: dict) -> dict:
        job_id = str(uuid.uuid4())
        d = self.dir(job_id)
        (d / "layers").mkdir(parents=True)
        spacing = scan.spacing
        ras_spacing = spacing
        ras_labels = None
        for name in LAYERS:
            if name in MODALITIES:
                src = _to_u8_window(scan.volumes[name])
            elif name == "labels":
                src = res.labels
            elif name == "confidence":
                src = np.rint(res.confidence * 255).astype(np.uint8)
            else:
                src = np.rint(res.uncertainty * 255).astype(np.uint8)
            arr, ras_spacing = reorient_to_ras(src, scan.affine, spacing)
            (d / "layers" / f"{name}.bin").write_bytes(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
            if name == "labels":
                ras_labels, ras_shape = arr, arr.shape
            if name == "flair":
                ras_flair = arr
        save_mask_nifti(labels_to_brats(res.labels), scan.affine, d / "mask.nii.gz")
        areas = (ras_labels > 0).sum(axis=(0, 1))
        z = int(areas.argmax()) if areas.max() > 0 else ras_labels.shape[2] // 2
        (d / "preview.png").write_bytes(render_preview(ras_flair, ras_labels, z))
        meta = {
            "id": job_id, "status": "completed", "created": time.time(),
            "expires": time.time() + self.s.artifact_ttl_minutes * 60 if self.s.artifact_ttl_minutes > 0 else None,
            "viewer": {"shape": list(ras_shape), "spacing_mm": list(ras_spacing), "layers": list(LAYERS),
                       "orientation": "RAS", "default_slice": z,
                       "region_colors": REGION_COLORS},
            "request": options, "results": res.summary,
            "downloads": {"mask": f"/jobs/{job_id}/download/mask", "preview": f"/jobs/{job_id}/download/preview",
                          "metrics": f"/jobs/{job_id}/download/metrics"},
        }
        meta = nan_to_none(meta)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        return meta

    def get(self, job_id: str) -> dict:
        d = self.dir(job_id)
        p = d / "meta.json"
        if not p.exists():
            raise KeyError(job_id)
        meta = json.loads(p.read_text())
        if meta.get("expires") and meta["expires"] < time.time():
            self.delete(job_id)
            raise KeyError(job_id)
        return meta

    def list(self) -> list[dict]:
        self.cleanup()
        out = []
        for d in self.root.iterdir():
            try:
                m = self.get(d.name)
            except (KeyError, OSError, ValueError):
                continue
            r = m["results"]
            out.append({"id": m["id"], "created": m["created"], "expires": m["expires"], "status": m["status"],
                        "model": r["model"].get("run") or r["model"]["checkpoint"],
                        "wt_volume_ml": r["regions"]["WT"]["volume_ml"], "scan_confidence": r["scan_confidence"],
                        "shape": r["shape"], "options": r["options"], "synthetic_model": r["model"].get("synthetic", False)})
        return sorted(out, key=lambda x: -x["created"])

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self.dir(job_id), ignore_errors=True)

    def cleanup(self) -> int:
        n, now = 0, time.time()
        for d in self.root.iterdir():
            p = d / "meta.json"
            try:
                exp = json.loads(p.read_text()).get("expires") if p.exists() else now - 1
            except (OSError, ValueError):
                exp = now - 1
            if exp is not None and exp < now:
                shutil.rmtree(d, ignore_errors=True); n += 1
        return n
