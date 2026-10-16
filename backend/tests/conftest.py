from __future__ import annotations

from pathlib import Path

import nibabel as nib
import pytest
import torch

from neurovista.checkpoint import save_checkpoint
from neurovista.config import MODALITIES, Settings
from neurovista.model import UNet
from neurovista.synthetic import make_case


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "var", device="cpu", rate_limit_per_minute=10_000, max_upload_mb=5,
                    artifact_ttl_minutes=5, _env_file=None)


@pytest.fixture()
def checkpoint(settings: Settings) -> Path:
    torch.manual_seed(0)
    model = UNet(base=8, p=0.2)
    save_checkpoint(settings.checkpoint_file, model, {"modalities": list(MODALITIES), "synthetic": True,
                                                      "trained_on": "untrained test fixture", "run": "test"})
    return settings.checkpoint_file


@pytest.fixture()
def scan_files(tmp_path: Path):
    vols, seg, affine = make_case((48, 48, 12), seed=1)
    d = tmp_path / "scan"
    d.mkdir()
    paths = {}
    for m, v in vols.items():
        paths[m] = d / f"{m}.nii.gz"
        nib.save(nib.Nifti1Image(v, affine), str(paths[m]))
    seg_path = d / "seg.nii.gz"
    nib.save(nib.Nifti1Image(seg, affine), str(seg_path))
    return paths, seg_path, (vols, seg, affine)
