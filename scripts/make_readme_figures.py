"""Generate README figures from REAL pipeline outputs.

Runs the actual preprocessing + checkpoint on a synthetic phantom from samples/ and renders:
  docs/figures/pipeline_panels.png  original | normalised | ground truth | prediction | overlay | confidence | uncertainty
  docs/figures/slice_sweep.gif      predicted overlay swept through the axial slices

The phantom is NOT patient data and the checkpoint is a synthetic-data smoke-test model, so these figures
demonstrate the software pipeline only. Run from backend/ with the venv active:

    python ../scripts/make_readme_figures.py --case ../samples/SYNTHETIC_PHANTOM_2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurovista.config import MODALITIES, Settings
from neurovista.inference import InferenceOptions, Segmenter
from neurovista.nifti_io import load_scan, reorient_to_ras
from neurovista.preprocess import prepare_volume, remap_labels
from neurovista.api.store import PALETTE, display_slice

NAVY = (11, 22, 40)
SCALE = 4


def gray(a: np.ndarray) -> np.ndarray:
    m = a > 0
    lo, hi = (np.percentile(a[m], [1, 99]) if m.any() else (0, 1))
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255 * m).astype(np.uint8)


def overlay(g: np.ndarray, lab: np.ndarray, alpha=0.55) -> np.ndarray:
    rgb = np.stack([g] * 3, -1).astype(np.float32)
    for c, col in PALETTE.items():
        rgb[lab == c] = (1 - alpha) * rgb[lab == c] + alpha * np.array(col)
    return rgb.astype(np.uint8)


def heat(v: np.ndarray, stops) -> np.ndarray:
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    s = np.array(stops, dtype=np.float32)
    return (s[i] * (1 - f) + s[i + 1] * f).astype(np.uint8)


VIRIDIS = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
INFERNO = [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("../docs/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st = Settings(device="cpu", _env_file=None)
    name = a.case.name
    paths = {m: a.case / f"{name}_{m}.nii.gz" for m in MODALITIES}
    scan = load_scan(paths, st)
    gt = remap_labels(np.asarray(nib.load(str(a.case / f"{name}_seg.nii.gz")).dataobj))
    seg = Segmenter.from_checkpoint(st.checkpoint_file, st)
    res = seg.predict(scan, InferenceOptions(tta=True, mc_samples=4))
    stack, box = prepare_volume(scan.volumes)
    norm = box.restore(stack[3])  # normalised FLAIR on the original grid

    def ras(x):
        return reorient_to_ras(x, scan.affine, scan.spacing)[0]

    flair, norm, gt, pred = ras(scan.volumes["flair"]), ras(norm), ras(gt), ras(res.labels)
    conf, unc = ras(res.confidence), ras(res.uncertainty)
    z = int((pred > 0).sum(axis=(0, 1)).argmax())
    g = display_slice(gray(flair)[:, :, z])
    n = display_slice(gray(norm - norm.min() * (norm > 0))[:, :, z])
    panels = [
        ("1  FLAIR (input)", np.stack([g] * 3, -1)),
        ("2  Normalised", np.stack([n] * 3, -1)),
        ("3  Ground truth", overlay(g, display_slice(gt[:, :, z]), 0.8)),
        ("4  Predicted mask", overlay(np.zeros_like(g), display_slice(pred[:, :, z]), 1.0)),
        ("5  Overlay", overlay(g, display_slice(pred[:, :, z]))),
        ("6  Confidence", heat(display_slice(conf[:, :, z]), VIRIDIS)),
        ("7  Uncertainty", heat(display_slice(unc[:, :, z]), INFERNO)),
    ]
    h, w = g.shape
    cap_h, pad = 34, 8
    W = len(panels) * (w * SCALE + pad) + pad
    canvas = Image.new("RGB", (W, h * SCALE + cap_h + 2 * pad + 22), NAVY)
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    for i, (title, img) in enumerate(panels):
        x = pad + i * (w * SCALE + pad)
        canvas.paste(Image.fromarray(img).resize((w * SCALE, h * SCALE), Image.NEAREST), (x, pad + cap_h))
        d.text((x, pad + 8), title, fill=(210, 225, 245), font=font)
    d.text((pad, h * SCALE + cap_h + pad + 6),
           "Synthetic phantom + synthetic-data smoke-test checkpoint: demonstrates the pipeline, not clinical accuracy.",
           fill=(140, 160, 190), font=small)
    canvas.save(a.out / "pipeline_panels.png", optimize=True)

    frames = []
    for k in range(pred.shape[2]):
        gk = display_slice(gray(flair)[:, :, k])
        fr = Image.fromarray(overlay(gk, display_slice(pred[:, :, k]))).resize((w * 3, h * 3), Image.NEAREST)
        ImageDraw.Draw(fr).text((6, 4), f"axial {k + 1}/{pred.shape[2]}", fill=(220, 230, 245), font=small)
        frames.append(fr.convert("P", palette=Image.ADAPTIVE, colors=96))
    frames[0].save(a.out / "slice_sweep.gif", save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)
    print("slice", z, "WT mL", res.summary["regions"]["WT"]["volume_ml"], "wrote", list(a.out.iterdir()))


if __name__ == "__main__":
    main()
