"""Command-line helpers:  python -m neurovista.cli {prepare|synth|predict|serve}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MODALITIES, get_settings


def main() -> None:
    ap = argparse.ArgumentParser(prog="neurovista")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="BraTS-style raw cases -> normalised slices + patient-wise splits")
    p.add_argument("--name", required=True, help="dataset name; raw cases must be in <data_dir>/datasets/<name>/raw")
    p.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("synth", help="generate a SYNTHETIC phantom dataset for smoke tests (not real MRI)")
    s.add_argument("--name", default="synthetic")
    s.add_argument("--n", type=int, default=12)
    pr = sub.add_parser("predict", help="run inference on four NIfTI files")
    for m in MODALITIES:
        pr.add_argument(f"--{m}", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--tta", action="store_true")
    pr.add_argument("--mc", type=int, default=0)
    a = ap.parse_args()
    st = get_settings()

    if a.cmd == "prepare":
        from .data import preprocess_dataset
        root = st.datasets_dir / a.name
        man = preprocess_dataset(root / "raw", root / "processed", a.seed)
        print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in man["splits"].items()}))
    elif a.cmd == "synth":
        from .data import preprocess_dataset
        from .synthetic import write_dataset
        root = st.datasets_dir / a.name
        write_dataset(root / "raw", a.n)
        man = preprocess_dataset(root / "raw", root / "processed", 42, keep_empty=0.5)
        print("synthetic dataset ready:", {k: len(v) for k, v in man["splits"].items()})
    elif a.cmd == "predict":
        from .checkpoint import CheckpointError
        from .inference import InferenceOptions, Segmenter, labels_to_brats
        from .nifti_io import load_scan, save_mask_nifti
        try:
            seg = Segmenter.from_checkpoint(st.checkpoint_file, st)
        except CheckpointError as e:
            raise SystemExit(f"Model unavailable: {e}")
        scan = load_scan({m: getattr(a, m) for m in MODALITIES}, st)
        res = seg.predict(scan, InferenceOptions(a.tta, a.mc))
        a.out.mkdir(parents=True, exist_ok=True)
        save_mask_nifti(labels_to_brats(res.labels), scan.affine, a.out / "segmentation.nii.gz")
        (a.out / "metrics.json").write_text(json.dumps(res.summary, indent=1, default=str))
        print(json.dumps(res.summary["regions"], indent=1))


if __name__ == "__main__":
    main()
