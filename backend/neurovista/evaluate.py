"""Volume-level evaluation of a checkpoint on a patient split, including calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import rankdata

from .config import MODALITIES, REGIONS, get_settings
from .data import _case_file, load_splits
from .inference import InferenceOptions, Segmenter
from .metrics import confusion_matrix, ece, evaluate_case, nan_to_none, reliability_bins
from .nifti_io import Scan
from .preprocess import remap_labels

METRICS = ("dice", "iou", "sensitivity", "specificity", "hd95_mm", "volume_error")


def _aggregate(cases: list[dict]) -> dict:
    agg: dict = {}
    for r in REGIONS:
        agg[r] = {}
        for k in METRICS:
            vals = np.array([c["regions"][r][k] for c in cases], dtype=float)
            vals = vals[~np.isnan(vals)]
            agg[r][k] = {"mean": float(vals.mean()) if vals.size else None,
                         "std": float(vals.std()) if vals.size else None, "n": int(vals.size)}
    return agg


def calibration_thresholds(conf: np.ndarray, correct: np.ndarray) -> dict | None:
    """Smallest confidence cut whose accuracy (on the VALIDATION set) reaches 0.95 ('high') / 0.85 ('medium')."""
    if conf.size < 100:
        return None
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order]
    cum_acc = np.cumsum(ok) / np.arange(1, ok.size + 1)

    def cut(target: float):
        idx = np.flatnonzero(cum_acc >= target)
        return float(c[idx.max()]) if idx.size else None

    hi, med = cut(0.95), cut(0.85)
    if hi is None or med is None or med >= hi:
        return None
    return {"high": hi, "medium": med, "targets": {"high": 0.95, "medium": 0.85}}


def uncertainty_vs_error(unc: np.ndarray, correct: np.ndarray) -> dict | None:
    """AUROC of normalised entropy as a detector of mis-classified voxels (needs ground truth)."""
    err = correct < 0.5
    n_err, n_ok = int(err.sum()), int((~err).sum())
    if n_err == 0 or n_ok == 0:
        return None
    ranks = rankdata(unc)
    auroc = (ranks[err].sum() - n_err * (n_err + 1) / 2) / (n_err * n_ok)
    return {"auroc": float(auroc), "mean_uncertainty_correct": float(unc[~err].mean()),
            "mean_uncertainty_error": float(unc[err].mean()), "n_error_voxels": n_err}


def evaluate_split(seg: Segmenter, raw_dir: Path, patients: list[str], opts: InferenceOptions,
                   max_calib_voxels: int = 2_000_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cases, confs, oks, uncs, fg_c, fg_ok = [], [], [], [], [], []
    cm = np.zeros((4, 4), np.int64)
    for pid in patients:
        case = Path(raw_dir) / pid
        vols = {m: np.asarray(nib.load(str(_case_file(case, m))).dataobj, dtype=np.float32) for m in MODALITIES}
        gt_img = nib.load(str(_case_file(case, "seg")))
        gt = remap_labels(np.asarray(gt_img.dataobj))
        zooms = tuple(float(z) for z in gt_img.header.get_zooms()[:3])
        scan = Scan(vols, gt_img.affine, zooms, tuple(nib.aff2axcodes(gt_img.affine)))  # type: ignore[arg-type]
        res = seg.predict(scan, opts)
        m = evaluate_case(res.labels, gt, zooms)
        cases.append({"patient": pid, "regions": m, "inference_seconds": res.summary["inference_seconds"]})
        cm += confusion_matrix(res.labels, gt)
        inside = np.any(np.stack([vols[k] > 0 for k in MODALITIES]), 0)
        c, ok = res.confidence[inside], (res.labels == gt)[inside]
        take = rng.choice(c.size, size=min(c.size, max_calib_voxels // max(len(patients), 1)), replace=False)
        confs.append(c[take]); oks.append(ok[take]); uncs.append(res.uncertainty[inside][take])
        fg = (res.labels > 0)[inside]
        fg_c.append(c[fg][:500_000]); fg_ok.append(ok[fg][:500_000])
    conf, ok, unc = np.concatenate(confs), np.concatenate(oks).astype(float), np.concatenate(uncs)
    out = {
        "n_patients": len(cases), "cases": cases, "aggregate": _aggregate(cases),
        "confusion_matrix": cm.tolist(),
        "calibration": {"ece": ece(conf, ok), "bins": reliability_bins(conf, ok),
                        "brier_top1": float(((conf - ok) ** 2).mean()),
                        "voxels_sampled": int(conf.size), "scope": "brain voxels, max-softmax vs correctness",
                        "thresholds": calibration_thresholds(np.concatenate(fg_c), np.concatenate(fg_ok).astype(float)),
                        "thresholds_scope": "voxels predicted as tumor (validation split only)"},
        "uncertainty_vs_error": uncertainty_vs_error(unc, ok),
        "options": {"tta": opts.tta, "mc_samples": opts.mc_samples, "min_component_voxels": opts.min_component_voxels},
    }
    return nan_to_none(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on a patient split")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mc", type=int, default=0)
    ap.add_argument("--min-component", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seg = Segmenter.from_checkpoint(a.checkpoint, get_settings())
    patients = load_splits(a.processed_dir)[a.split]
    res = evaluate_split(seg, a.raw_dir, patients, InferenceOptions(a.tta, a.mc, a.min_component))
    res["split"] = a.split
    a.out.write_text(json.dumps(res, indent=1))
    th = res["calibration"]["thresholds"]
    if a.split == "val" and th and not (a.tta or a.mc or a.min_component):
        # thresholds are chosen on the validation split only, never on test
        (a.out.parent / "calibration.json").write_text(json.dumps({**th, "split": "val"}, indent=1))
    print(json.dumps(res["aggregate"], indent=1))


if __name__ == "__main__":
    main()
