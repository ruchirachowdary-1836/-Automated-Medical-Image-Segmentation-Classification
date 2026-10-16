"""BraTS case discovery, patient-wise splitting, slice export and the PyTorch Dataset."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import MODALITIES
from .preprocess import extract_slices, normalize_volumes, plan_grid, remap_labels


def find_cases(raw_dir: Path) -> list[Path]:
    """A case is a folder containing <id>_{t1,t1ce,t2,flair,seg}.nii[.gz]."""
    return sorted(p for p in Path(raw_dir).iterdir() if p.is_dir() and _case_file(p, "flair") is not None)


def _case_file(case: Path, key: str) -> Path | None:
    for ext in (".nii.gz", ".nii"):
        f = case / f"{case.name}_{key}{ext}"
        if f.exists():
            return f
    return None


def make_splits(patient_ids: list[str], seed: int = 42, fractions=(0.70, 0.15, 0.15)) -> dict[str, list[str]]:
    ids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    # guarantee non-empty val/test when possible
    if n >= 3:
        a = max(1, min(a, n - 2))
        b = max(a + 1, min(b, n - 1))
    s = {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}
    assert_no_leakage(s)
    return s


def assert_no_leakage(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        for pid in ids:
            if pid in seen:
                raise ValueError(f"Patient {pid} appears in both '{seen[pid]}' and '{name}'")
            seen[pid] = name


def preprocess_dataset(raw_dir: Path, out_dir: Path, seed: int = 42, keep_empty: float = 0.1,
                       fractions=(0.70, 0.15, 0.15)) -> dict:
    """NIfTI cases -> normalised 2D slices (.npz) + splits.json (patient-wise, seeded, persisted)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = find_cases(raw_dir)
    if not cases:
        raise FileNotFoundError(f"No cases found in {raw_dir} (expected <id>/<id>_flair.nii.gz etc.)")
    rng = np.random.default_rng(seed)
    n_slices: dict[str, int] = {}
    for case in cases:
        files = {m: _case_file(case, m) for m in (*MODALITIES, "seg")}
        absent = [k for k, v in files.items() if v is None]
        if absent:
            raise FileNotFoundError(f"Case {case.name} is missing: {', '.join(absent)}")
        vols = {m: np.asarray(nib.load(str(files[m])).dataobj, dtype=np.float32) for m in MODALITIES}
        seg = remap_labels(np.asarray(nib.load(str(files["seg"])).dataobj))
        stack = normalize_volumes(vols)
        box = plan_grid(stack)
        stack, seg = box.apply(stack), box.apply(seg)
        count = 0
        for z, x, y in extract_slices(stack, seg, rng, keep_empty):
            np.savez_compressed(out_dir / f"{case.name}_{z:03d}.npz", x=x.astype(np.float32), y=y.astype(np.uint8))
            count += 1
        n_slices[case.name] = count
    splits = make_splits(list(n_slices), seed, fractions)
    manifest = {"seed": seed, "fractions": list(fractions), "splits": splits, "slices_per_patient": n_slices}
    (out_dir.parent / "splits.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_splits(processed_dir: Path) -> dict[str, list[str]]:
    manifest = json.loads((Path(processed_dir).parent / "splits.json").read_text(encoding="utf-8"))
    assert_no_leakage(manifest["splits"])
    return manifest["splits"]


def patient_of(file: Path) -> str:
    return file.stem.rsplit("_", 1)[0]


def augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, cfg: dict | None = None):
    """Spatial transforms are applied identically to image and label; intensity ones to the image only."""
    cfg = cfg or {}
    if rng.random() < cfg.get("flip_p", 0.5):
        x, y = x[:, :, ::-1], y[:, ::-1]
    if rng.random() < cfg.get("rot_p", 0.3):
        k = int(rng.integers(0, 4))
        if k:
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(0, 1))
    x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
    if cfg.get("gamma", False) and rng.random() < 0.3:
        lo, hi = float(x.min()), float(x.max())
        if hi > lo:
            g = rng.uniform(0.8, 1.25)
            x = ((x - lo) / (hi - lo)) ** g * (hi - lo) + lo
    if cfg.get("noise", False) and rng.random() < 0.3:
        x = x + rng.normal(0, 0.05, x.shape)
    return x, y


class SliceDataset(Dataset):
    def __init__(self, processed_dir: Path, split: str, train: bool = False, aug_cfg: dict | None = None,
                 seed: int = 42, patients: list[str] | None = None):
        pids = set(patients if patients is not None else load_splits(processed_dir)[split])
        self.files = [f for f in sorted(Path(processed_dir).glob("*.npz")) if patient_of(f) in pids]
        self.train, self.aug_cfg, self.seed = train, aug_cfg, seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.files)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:  # created lazily so each DataLoader worker gets its own stream
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            self._rng = np.random.default_rng([self.seed, wid, int(torch.initial_seed() % (2**31))])
        return self._rng

    def __getitem__(self, i: int):
        d = np.load(self.files[i])
        x, y = d["x"], d["y"]
        if self.train:
            x, y = augment(x, y, self._get_rng(), self.aug_cfg)
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y).astype(np.int64)))
