"""Segmentation losses. Dice terms ignore the background class."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(target, n).permute(0, 3, 1, 2).float()


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.n, self.s = n_classes, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = logits.softmax(1)
        oh = _one_hot(target, self.n)
        inter = (p * oh).sum((0, 2, 3))
        denom = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        dice = (2 * inter + self.s) / (denom + self.s)
        return 1 - dice[1:].mean()


class DiceCELoss(nn.Module):
    def __init__(self, n_classes: int = 4, w_dice: float = 0.5, smooth: float = 1e-5,
                 class_weights: list[float] | None = None):
        super().__init__()
        self.dice = DiceLoss(n_classes, smooth)
        self.w = w_dice
        self.register_buffer("cw", torch.tensor(class_weights, dtype=torch.float32) if class_weights else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.cw)
        return (1 - self.w) * ce + self.w * self.dice(logits, target)


class FocalTverskyLoss(nn.Module):
    """Weights false negatives more than false positives (alpha > beta)."""

    def __init__(self, n_classes: int = 4, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75,
                 smooth: float = 1e-5):
        super().__init__()
        self.n, self.a, self.b, self.g, self.s = n_classes, alpha, beta, gamma, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = logits.softmax(1)
        oh = _one_hot(target, self.n)
        tp = (p * oh).sum((0, 2, 3))
        fn = ((1 - p) * oh).sum((0, 2, 3))
        fp = (p * (1 - oh)).sum((0, 2, 3))
        t = (tp + self.s) / (tp + self.a * fn + self.b * fp + self.s)
        return ((1 - t[1:]) ** self.g).mean()


def build_loss(name: str, n_classes: int = 4, class_weights: list[float] | None = None) -> nn.Module:
    name = name.lower()
    if name in ("ce", "cross_entropy"):
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        return _CE(w)
    if name == "dice":
        return DiceLoss(n_classes)
    if name in ("dice_ce", "dicece"):
        return DiceCELoss(n_classes, class_weights=class_weights)
    if name in ("focal_tversky", "ft"):
        return FocalTverskyLoss(n_classes)
    raise ValueError(f"Unknown loss '{name}'")


class _CE(nn.Module):
    def __init__(self, weight: torch.Tensor | None):
        super().__init__()
        self.register_buffer("w", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.w)
