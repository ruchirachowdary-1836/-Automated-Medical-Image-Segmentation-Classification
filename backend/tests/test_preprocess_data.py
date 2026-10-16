import numpy as np
import pytest
import torch

from neurovista.config import MODALITIES
from neurovista.data import SliceDataset, assert_no_leakage, augment, load_splits, make_splits, preprocess_dataset
from neurovista.preprocess import (
    GridBox,
    PreprocessError,
    normalize_volumes,
    plan_grid,
    prepare_volume,
    remap_labels,
    zscore,
)
from neurovista.synthetic import make_case, write_dataset


def test_zscore_brain_mask_stats():
    v = np.zeros((20, 20, 20), np.float32)
    v[5:15, 5:15, 5:15] = np.random.default_rng(0).uniform(100, 500, (10, 10, 10))
    z = zscore(v)
    m = v > 0
    assert abs(z[m].mean()) < 0.2 and 0.7 < z[m].std() < 1.2
    assert (z[~m] == 0).all() and z.dtype == np.float32
    assert (zscore(np.zeros((4, 4, 4))) == 0).all()


def test_remap_labels():
    assert remap_labels(np.array([0, 1, 2, 4, 4])).tolist() == [0, 1, 2, 3, 3]
    with pytest.raises(PreprocessError):
        remap_labels(np.array([0, 3]))
    with pytest.raises(PreprocessError):
        remap_labels(np.array([0, 7]))


def test_modality_order_and_missing():
    vols, _, _ = make_case((32, 32, 8))
    stack = normalize_volumes(vols)
    assert stack.shape == (4, 32, 32, 8)
    for i, m in enumerate(MODALITIES):
        assert np.allclose(stack[i], zscore(vols[m]))
    vols.pop("t2")
    with pytest.raises(PreprocessError):
        normalize_volumes(vols)


def test_grid_roundtrip_and_lossless_crop():
    a = np.zeros((4, 240, 240, 3), np.float32)
    a[:, 60:180, 60:180] = 1
    box = plan_grid(a)
    assert box.target == (192, 192)
    assert np.array_equal(box.restore(box.apply(a)), a)
    b = np.zeros((1, 240, 240, 2), np.float32)
    b[:, 0:5] = 1  # non-zero border: cropping would lose data
    assert plan_grid(b).target == (240, 240)
    c = np.ones((1, 50, 70, 2), np.float32)
    box = plan_grid(c)
    assert box.target == (64, 80) and np.array_equal(box.restore(box.apply(c)), c)


def test_image_mask_alignment_in_grid():
    box = GridBox((50, 50), (64, 64))
    img = np.random.default_rng(0).random((1, 50, 50, 2))
    seg = (img[0] > 0.5).astype(np.uint8)
    a, b = box.apply(img), box.apply(seg)
    assert np.array_equal(a[0] > 0.5, b > 0)


def test_splits_no_leakage_and_reproducible():
    ids = [f"P{i:03d}" for i in range(40)]
    s1, s2 = make_splits(ids, 7), make_splits(ids, 7)
    assert s1 == s2 and make_splits(ids, 8) != s1
    assert sorted(sum(s1.values(), [])) == ids
    assert_no_leakage(s1)
    with pytest.raises(ValueError):
        assert_no_leakage({"train": ["a"], "val": ["a"]})


def test_prepare_dataset_slices_respect_split(tmp_path):
    raw = tmp_path / "ds" / "raw"
    write_dataset(raw, 8, shape=(32, 32, 8))
    man = preprocess_dataset(raw, tmp_path / "ds" / "processed", seed=3, keep_empty=1.0)
    splits = load_splits(tmp_path / "ds" / "processed")
    assert splits == man["splits"] and (tmp_path / "ds" / "splits.json").exists()
    seen = {}
    for name in splits:
        ds = SliceDataset(tmp_path / "ds" / "processed", name)
        for f in ds.files:
            pid = f.stem.rsplit("_", 1)[0]
            assert pid in splits[name]
            assert seen.setdefault(pid, name) == name
    x, y = SliceDataset(tmp_path / "ds" / "processed", "train")[0]
    assert x.shape == (4, 32, 32) and y.shape == (32, 32) and y.dtype == torch.int64


def test_augment_keeps_image_mask_alignment():
    rng = np.random.default_rng(0)
    y = np.zeros((16, 16), np.uint8)
    y[3:8, 2:6] = 1
    x = np.stack([y.astype(np.float32)] * 4)
    for _ in range(30):
        xa, ya = augment(x, y, rng, {})
        thr = xa[0].min() + 0.5 * (xa[0].max() - xa[0].min())
        assert np.array_equal(xa[0] > thr, ya > 0)


def test_prepare_volume_matches_training_path():
    vols, _, _ = make_case((48, 48, 8))
    stack, box = prepare_volume(vols)
    assert stack.shape == (4, 48, 48, 8) and box.target == (48, 48)
