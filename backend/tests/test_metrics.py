import numpy as np
import pytest

from neurovista.metrics import (
    brier_score,
    dice,
    ece,
    evaluate_case,
    hd95,
    iou,
    nan_to_none,
    sensitivity,
    specificity,
    volume_error,
    volume_ml,
)


def test_dice_iou_known_values():
    p = np.zeros((4, 4), bool)
    g = np.zeros((4, 4), bool)
    p[:2] = True
    g[:, :2] = True
    assert dice(p, g) == pytest.approx(0.5)
    assert iou(p, g) == pytest.approx(1 / 3)


def test_empty_region_convention():
    z = np.zeros((4, 4), bool)
    o = np.ones((4, 4), bool)
    assert dice(z, z) == 1.0 and iou(z, z) == 1.0
    assert dice(z, o) == 0.0 and dice(o, z) == 0.0
    assert np.isnan(sensitivity(o, z))
    assert hd95(z, z) == 0.0 and np.isnan(hd95(z, o))
    assert np.isnan(volume_error(o, z))


def test_sensitivity_specificity():
    g = np.array([1, 1, 0, 0], bool)
    p = np.array([1, 0, 0, 1], bool)
    assert sensitivity(p, g) == 0.5 and specificity(p, g) == 0.5


def test_hd95_respects_spacing():
    a = np.zeros((20, 20, 20), bool)
    b = a.copy()
    a[5:10, 5:10, 5:10] = True
    b[5:10, 5:10, 8:13] = True
    d1 = hd95(a, b, (1, 1, 1))
    d2 = hd95(a, b, (1, 1, 2))
    assert d1 > 0 and d2 == pytest.approx(2 * d1, rel=0.3)


def test_volume_ml_uses_spacing():
    m = np.ones((10, 10, 10), bool)
    assert volume_ml(m, (1, 1, 1)) == pytest.approx(1.0)  # 1000 mm3 = 1 mL
    assert volume_ml(m, (2, 1, 1)) == pytest.approx(2.0)


def test_evaluate_case_perfect_and_shape_check():
    gt = np.zeros((8, 8, 8), np.uint8)
    gt[2:6, 2:6, 2:6] = 2
    gt[3:5, 3:5, 3:5] = 3
    r = evaluate_case(gt, gt, (1, 1, 1))
    assert all(r[k]["dice"] == 1.0 for k in ("WT", "TC", "ET"))
    with pytest.raises(ValueError):
        evaluate_case(gt, gt[:4], (1, 1, 1))


def test_ece_perfectly_calibrated_and_overconfident():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 20000)
    ok = (rng.uniform(size=conf.size) < conf).astype(float)
    assert ece(conf, ok) < 0.02
    assert ece(np.full(1000, 0.99), np.zeros(1000)) == pytest.approx(0.99)
    assert np.isnan(ece(np.array([]), np.array([])))


def test_brier():
    p = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    assert brier_score(p, np.array([0, 1])) == 0.0
    assert brier_score(p, np.array([1, 0])) == pytest.approx(2.0)


def test_nan_to_none():
    assert nan_to_none({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}
