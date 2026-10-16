"""2D U-Net and Attention U-Net (4 input channels -> 4 classes)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, i: int, o: int, p: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        ]
        if p > 0:
            layers.append(nn.Dropout2d(p))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttnGate(nn.Module):
    """Additive attention gate on a skip connection (Oktay et al., 2018)."""

    def __init__(self, cg: int, cx: int, ci: int):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(cg, ci, 1), nn.BatchNorm2d(ci))
        self.wx = nn.Sequential(nn.Conv2d(cx, ci, 1), nn.BatchNorm2d(ci))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(ci, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.wg(g) + self.wx(x))


class UNet(nn.Module):
    def __init__(self, in_ch: int = 4, n_classes: int = 4, base: int = 32, p: float = 0.2, attention: bool = False):
        super().__init__()
        self.in_ch, self.n_classes, self.base, self.p, self.attention = in_ch, n_classes, base, p, attention
        b = base
        self.e1, self.e2 = DoubleConv(in_ch, b), DoubleConv(b, 2 * b)
        self.e3, self.e4 = DoubleConv(2 * b, 4 * b), DoubleConv(4 * b, 8 * b, p / 2)
        self.bott = DoubleConv(8 * b, 16 * b, p)
        self.pool = nn.MaxPool2d(2)
        self.u4, self.d4 = nn.ConvTranspose2d(16 * b, 8 * b, 2, 2), DoubleConv(16 * b, 8 * b, p / 2)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), DoubleConv(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), DoubleConv(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), DoubleConv(2 * b, b)
        self.head = nn.Conv2d(b, n_classes, 1)
        if attention:
            self.g4 = AttnGate(8 * b, 8 * b, 4 * b)
            self.g3 = AttnGate(4 * b, 4 * b, 2 * b)
            self.g2 = AttnGate(2 * b, 2 * b, b)
            self.g1 = AttnGate(b, b, max(b // 2, 1))

    def _skip(self, name: str, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return getattr(self, name)(g, x) if self.attention else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 16 or x.shape[-2] % 16:
            raise ValueError(f"Spatial size must be divisible by 16, got {tuple(x.shape[-2:])}")
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        z = self.bott(self.pool(e4))
        u = self.u4(z); d4 = self.d4(torch.cat([u, self._skip("g4", u, e4)], 1))
        u = self.u3(d4); d3 = self.d3(torch.cat([u, self._skip("g3", u, e3)], 1))
        u = self.u2(d3); d2 = self.d2(torch.cat([u, self._skip("g2", u, e2)], 1))
        u = self.u1(d2); d1 = self.d1(torch.cat([u, self._skip("g1", u, e1)], 1))
        return self.head(d1)

    def arch_config(self) -> dict:
        return {"in_ch": self.in_ch, "n_classes": self.n_classes, "base": self.base, "p": self.p,
                "attention": self.attention}


def build_model(cfg: dict) -> UNet:
    return UNet(**{k: cfg[k] for k in ("in_ch", "n_classes", "base", "p", "attention") if k in cfg})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
