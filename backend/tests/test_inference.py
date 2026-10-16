import nibabel as nib
import numpy as np
import pytest
import torch

from neurovista.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from neurovista.inference import InferenceOptions, Segmenter, filter_small_components, labels_to_brats
from neurovista.model import UNet
from neurovista.nifti_io import InputValidationError, load_scan, save_mask_nifti

MODS = ["t1", "t1ce", "t2", "flair"]


def test_checkpoint_roundtrip_and_rejections(checkpoint, tmp_path):
    m, meta = load_checkpoint(checkpoint)
    assert meta["synthetic"] is True and isinstance(m, UNet)
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "missing.pt")
    torch.save({"state_dict": {}}, tmp_path / "bad.pt")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "bad.pt")
    save_checkpoint(tmp_path / "mismatch.pt", UNet(base=8), {"modalities": MODS})
    payload = torch.load(tmp_path / "mismatch.pt", weights_only=True)
    payload["arch"]["base"] = 16
    torch.save(payload, tmp_path / "mismatch.pt")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "mismatch.pt")


def test_predict_shapes_affine_and_volumes(settings, checkpoint, scan_files, tmp_path):
    paths, _, (_, _, affine) = scan_files
    scan = load_scan(paths, settings)
    res = Segmenter.from_checkpoint(checkpoint, settings).predict(scan, InferenceOptions(batch_size=4))
    assert res.labels.shape == scan.shape == res.confidence.shape == res.uncertainty.shape
    assert set(np.unique(res.labels)) <= {0, 1, 2, 3}
    assert 0 <= res.confidence.min() and res.confidence.max() <= 1
    assert 0 <= res.uncertainty.min() and res.uncertainty.max() <= 1
    wt = res.summary["regions"]["WT"]  # voxel spacing is 1 x 1 x 2 mm
    assert wt["volume_ml"] == pytest.approx(wt["voxels"] * 2.0 / 1000)
    out = tmp_path / "m.nii.gz"
    save_mask_nifti(labels_to_brats(res.labels), scan.affine, out)
    img = nib.load(str(out))
    assert np.allclose(img.affine, affine) and img.shape == scan.shape
    assert set(np.unique(np.asarray(img.dataobj))) <= {0, 1, 2, 4}
    assert res.summary["preprocessing"]["modality_order"] == MODS


def test_no_tumor_prediction_is_safe(settings, scan_files):
    paths, _, _ = scan_files
    scan = load_scan(paths, settings)
    model = UNet(base=8)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([10.0, 0, 0, 0]))
    seg = Segmenter(model.eval(), {"modalities": MODS}, torch.device("cpu"), settings)
    res = seg.predict(scan)
    s = res.summary
    assert res.labels.max() == 0
    assert s["scan_confidence"] is None and s["regions"]["WT"]["confidence"] is None
    assert s["regions"]["WT"]["volume_ml"] == 0 and s["warnings"]


def test_tta_and_mc_dropout(settings, checkpoint, scan_files):
    paths, _, _ = scan_files
    scan = load_scan(paths, settings)
    s = Segmenter.from_checkpoint(checkpoint, settings)
    r = s.predict(scan, InferenceOptions(tta=True, mc_samples=3))
    assert r.mutual_info is not None and r.mutual_info.shape == r.labels.shape
    assert np.array_equal(s.predict(scan).labels, s.predict(scan).labels)  # deterministic again afterwards
    no_drop = Segmenter(UNet(base=8, p=0.0).eval(), {}, torch.device("cpu"), settings)
    with pytest.raises(ValueError):
        no_drop.predict(scan, InferenceOptions(mc_samples=4))


def test_small_component_filter():
    lab = np.zeros((20, 20, 20), np.uint8)
    lab[2:12, 2:12, 2:12] = 2
    lab[18, 18, 18] = 1
    out, n = filter_small_components(lab, 5)
    assert n == 1 and out[18, 18, 18] == 0 and (out[2:12, 2:12, 2:12] == 2).all()
    same, n0 = filter_small_components(lab, 0)
    assert n0 == 0 and np.array_equal(same, lab)


def _write(tmp_path, name, arr, affine):
    p = tmp_path / name
    nib.save(nib.Nifti1Image(arr, affine), str(p))
    return p


def _code(settings, paths):
    with pytest.raises(InputValidationError) as e:
        load_scan(paths, settings)
    return e.value.code


def test_load_scan_rejections(settings, scan_files, tmp_path):
    paths, _, (vols, _, affine) = scan_files
    assert _code(settings, {k: v for k, v in paths.items() if k != "t2"}) == "missing_modality"
    assert _code(settings, {**paths, "t1": _write(tmp_path, "a.nii.gz", vols["t1"][:32, :32], affine)}) == "shape_mismatch"
    assert _code(settings, {**paths, "t1": _write(tmp_path, "b.nii.gz", vols["t1"], np.diag([-1, -1, 3, 1.0]))}) == "affine_mismatch"
    junk = tmp_path / "junk.nii.gz"
    junk.write_bytes(b"not a nifti")
    assert _code(settings, {**paths, "flair": junk}) == "unreadable_file"
    zero = _write(tmp_path, "z.nii.gz", np.zeros_like(vols["t1"]), affine)
    assert _code(settings, {**paths, "t2": zero}) == "empty_volume"
    sag = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    allsag = {k: _write(tmp_path, f"s_{k}.nii.gz", vols[k], sag) for k in paths}
    assert _code(settings, allsag) == "unsupported_orientation"
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG")
    assert _code(settings, {**paths, "t1": png}) == "unsupported_file_type"
