import pytest
import torch

from neurovista.losses import DiceCELoss, DiceLoss, FocalTverskyLoss, build_loss
from neurovista.model import UNet, count_parameters


@pytest.mark.parametrize("attention", [False, True])
def test_unet_forward_shape(attention):
    m = UNet(base=8, attention=attention).eval()
    assert m(torch.randn(2, 4, 32, 48)).shape == (2, 4, 32, 48)
    assert count_parameters(m) > 0


def test_unet_rejects_bad_size():
    with pytest.raises(ValueError):
        UNet(base=8)(torch.randn(1, 4, 30, 32))


def test_default_unet_matches_guide_param_scale():
    m = UNet()  # base=32
    assert 7e6 < count_parameters(m) < 9e6


def test_losses_finite_and_perfect_prediction_low():
    y = torch.randint(0, 4, (2, 16, 16))
    perfect = torch.nn.functional.one_hot(y, 4).permute(0, 3, 1, 2).float() * 20
    for loss in (DiceLoss(), DiceCELoss(), FocalTverskyLoss()):
        v = loss(perfect, y)
        assert torch.isfinite(v) and v < 0.1
        assert loss(torch.randn(2, 4, 16, 16), y) > v


def test_dice_loss_all_background_is_finite():
    y = torch.zeros(1, 16, 16, dtype=torch.long)
    assert torch.isfinite(DiceCELoss()(torch.randn(1, 4, 16, 16), y))


def test_build_loss_unknown():
    with pytest.raises(ValueError):
        build_loss("nope")
