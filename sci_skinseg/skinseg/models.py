from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        groups: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        if p is None:
            p = dilation * (k // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, dilation=dilation, groups=groups, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3, stride, dilation, groups=in_ch, dilation=dilation),
            ConvBNAct(in_ch, out_ch, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResDSBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = DSConv(in_ch, out_ch, stride=stride)
        self.conv2 = DSConv(out_ch, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else ConvBNAct(in_ch, out_ch, 1, stride, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.skip(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(in_ch, out_ch), ConvBNAct(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicDoubleConv(nn.Module):
    """Conv-BN-ReLU block used for the EGE/MALUNet-style U-Net baseline."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.e1 = DoubleConv(3, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)
        self.b = DoubleConv(base * 8, base * 16)
        self.d4 = DoubleConv(base * 16 + base * 8, base * 8)
        self.d3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.d2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.d1 = DoubleConv(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e4], 1))
        d3 = self.d3(torch.cat([F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


class ClassicUNet(nn.Module):
    """Classic U-Net baseline with Conv-BN-ReLU blocks and about 7.77M params."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.e1 = ClassicDoubleConv(3, base)
        self.e2 = ClassicDoubleConv(base, base * 2)
        self.e3 = ClassicDoubleConv(base * 2, base * 4)
        self.e4 = ClassicDoubleConv(base * 4, base * 8)
        self.b = ClassicDoubleConv(base * 8, base * 16)
        self.d4 = ClassicDoubleConv(base * 16 + base * 8, base * 8)
        self.d3 = ClassicDoubleConv(base * 8 + base * 4, base * 4)
        self.d2 = ClassicDoubleConv(base * 4 + base * 2, base * 2)
        self.d1 = ClassicDoubleConv(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e4], 1))
        d3 = self.d3(torch.cat([F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


class TinyUNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.e1 = ResDSBlock(3, base)
        self.e2 = ResDSBlock(base, base * 2, 2)
        self.e3 = ResDSBlock(base * 2, base * 4, 2)
        self.b = ResDSBlock(base * 4, base * 8, 2)
        self.d3 = ResDSBlock(base * 8 + base * 4, base * 4)
        self.d2 = ResDSBlock(base * 4 + base * 2, base * 2)
        self.d1 = ResDSBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b = self.b(e3)
        d3 = self.d3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


class GroupEnhanceBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        parts = max(1, min(4, channels // 8))
        self.parts = parts
        per = channels // parts
        self.branches = nn.ModuleList([DSConv(per, per, dilation=d) for d in (1, 2, 3, 5)[:parts]])
        self.proj = ConvBNAct(channels, channels, 1, 1, 0)
        self.attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.parts, dim=1)
        y = torch.cat([branch(ch) for branch, ch in zip(self.branches, chunks)], dim=1)
        y = self.proj(y)
        return x + y * self.attn(y)


class MultiScaleContextBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        branch_ch = channels // 4
        self.reduce = ConvBNAct(channels, branch_ch * 4, 1, 1, 0)
        self.b1 = DSConv(branch_ch, branch_ch, dilation=1)
        self.b2 = DSConv(branch_ch, branch_ch, dilation=2)
        self.b3 = DSConv(branch_ch, branch_ch, dilation=4)
        self.b4 = SelectiveScan2D(branch_ch)
        self.fuse = ConvBNAct(branch_ch * 4, channels, 1, 1, 0)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, channels // 4), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1, c2, c3, c4 = torch.chunk(self.reduce(x), 4, dim=1)
        y = torch.cat([self.b1(c1), self.b2(c2), self.b3(c3), self.b4(c4)], dim=1)
        y = self.fuse(y)
        return x + y * self.attn(y)


class EGELite(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.stem = ConvBNAct(3, base)
        self.e1 = nn.Sequential(ResDSBlock(base, base), GroupEnhanceBlock(base))
        self.e2 = nn.Sequential(ResDSBlock(base, base * 2, 2), GroupEnhanceBlock(base * 2))
        self.e3 = nn.Sequential(ResDSBlock(base * 2, base * 4, 2), GroupEnhanceBlock(base * 4))
        self.e4 = nn.Sequential(ResDSBlock(base * 4, base * 8, 2), GroupEnhanceBlock(base * 8))
        self.d3 = ResDSBlock(base * 12, base * 4)
        self.d2 = ResDSBlock(base * 6, base * 2)
        self.d1 = ResDSBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


class SelectiveScan2D(nn.Module):
    """Mamba-style lightweight selective scan without custom CUDA dependencies."""

    def __init__(self, channels: int):
        super().__init__()
        self.in_proj = ConvBNAct(channels, channels * 2, 1, 1, 0)
        self.local = DSConv(channels, channels)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.out_proj = ConvBNAct(channels, channels, 1, 1, 0)

    @staticmethod
    def _scan(x: torch.Tensor) -> torch.Tensor:
        h_f = torch.cumsum(x, dim=2)
        h_b = torch.flip(torch.cumsum(torch.flip(x, dims=[2]), dim=2), dims=[2])
        w_f = torch.cumsum(x, dim=3)
        w_b = torch.flip(torch.cumsum(torch.flip(x, dims=[3]), dim=3), dims=[3])
        h = x.shape[2]
        w = x.shape[3]
        return (h_f + h_b) / max(h, 1) + (w_f + w_b) / max(w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = torch.chunk(self.in_proj(x), 2, dim=1)
        local = self.local(u)
        scanned = self._scan(local)
        y = scanned * self.gate(v)
        return x + self.out_proj(y)


class BoundaryGuidedFusion(nn.Module):
    def __init__(self, high_ch: int, low_ch: int, out_ch: int):
        super().__init__()
        self.high = ConvBNAct(high_ch, out_ch, 1, 1, 0)
        self.low = ConvBNAct(low_ch, out_ch, 1, 1, 0)
        self.boundary = nn.Conv2d(out_ch, 1, 1)
        self.mix = ResDSBlock(out_ch * 2 + 1, out_ch)

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        high = F.interpolate(self.high(high), size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low(low)
        b = self.boundary(high + low)
        fused = self.mix(torch.cat([high, low, torch.sigmoid(b)], dim=1))
        return fused, b


class BLMNet(nn.Module):
    """Boundary-aware Lightweight Mamba-CNN network."""

    def __init__(self, base: int = 16, use_scan: bool = True, use_boundary: bool = True):
        super().__init__()
        self.use_boundary = use_boundary
        self.stem = ConvBNAct(3, base)
        self.e1 = ResDSBlock(base, base)
        self.e2 = nn.Sequential(ResDSBlock(base, base * 2, 2), GroupEnhanceBlock(base * 2))
        self.e3 = nn.Sequential(ResDSBlock(base * 2, base * 4, 2), GroupEnhanceBlock(base * 4))
        self.e4 = nn.Sequential(ResDSBlock(base * 4, base * 8, 2), GroupEnhanceBlock(base * 8))
        self.context = MultiScaleContextBlock(base * 8) if use_scan else GroupEnhanceBlock(base * 8)
        if use_boundary:
            self.f3 = BoundaryGuidedFusion(base * 8, base * 4, base * 4)
            self.f2 = BoundaryGuidedFusion(base * 4, base * 2, base * 2)
            self.f1 = BoundaryGuidedFusion(base * 2, base, base)
        else:
            self.f3 = ResDSBlock(base * 12, base * 4)
            self.f2 = ResDSBlock(base * 6, base * 2)
            self.f1 = ResDSBlock(base * 3, base)
        self.aux3 = nn.Conv2d(base * 4, 1, 1)
        self.aux2 = nn.Conv2d(base * 2, 1, 1)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        x = self.stem(x)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.context(self.e4(e3))
        if self.use_boundary:
            d3, b3 = self.f3(e4, e3)
            d2, b2 = self.f2(d3, e2)
            d1, b1 = self.f1(d2, e1)
            logits = self.out(d1)
            boundary = b1 + F.interpolate(b2, size=b1.shape[-2:], mode="bilinear", align_corners=False)
            boundary = boundary + F.interpolate(b3, size=b1.shape[-2:], mode="bilinear", align_corners=False)
            return {
                "logits": logits,
                "boundary": boundary / 3.0,
                "aux": [
                    F.interpolate(self.aux3(d3), size=logits.shape[-2:], mode="bilinear", align_corners=False),
                    F.interpolate(self.aux2(d2), size=logits.shape[-2:], mode="bilinear", align_corners=False),
                ],
            }
        d3 = self.f3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.f2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.f1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        logits = self.out(d1)
        return {
            "logits": logits,
            "aux": [
                F.interpolate(self.aux3(d3), size=logits.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(self.aux2(d2), size=logits.shape[-2:], mode="bilinear", align_corners=False),
            ],
        }


class ShiftMLPBlock(nn.Module):
    def __init__(self, channels: int, shift_size: int = 5):
        super().__init__()
        self.pad = shift_size // 2
        self.shift_size = shift_size
        self.fc1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.fc2 = nn.Conv2d(channels * 2, channels, 1)
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def _shift(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        chunks = torch.chunk(x, self.shift_size, dim=1)
        shifts = range(-self.pad, self.pad + 1)
        shifted = [torch.roll(ch, shifts=s, dims=dim) for ch, s in zip(chunks, shifts)]
        return torch.cat(shifted, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._shift(x, 2)
        y = F.gelu(self.dw(self.fc1(y)))
        y = self._shift(y, 3)
        y = self.fc2(y)
        return x + self.norm(y)


class UNeXtLite(nn.Module):
    """Dependency-light UNeXt-style baseline with convolutional encoder and shifted MLP bottleneck."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.e1 = ConvBNAct(3, base)
        self.e2 = ConvBNAct(base, base * 2)
        self.e3 = ConvBNAct(base * 2, base * 4)
        self.down = nn.MaxPool2d(2)
        self.b1 = nn.Sequential(ConvBNAct(base * 4, base * 8), ShiftMLPBlock(base * 8), ShiftMLPBlock(base * 8))
        self.d3 = ConvBNAct(base * 8 + base * 4, base * 4)
        self.d2 = ConvBNAct(base * 4 + base * 2, base * 2)
        self.d1 = ConvBNAct(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.down(e1))
        e3 = self.e3(self.down(e2))
        b = self.b1(self.down(e3))
        d3 = self.d3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


class LinearAttention2D(nn.Module):
    """MobileViTv2-style separable self-attention for 2D feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = ConvBNAct(channels, channels, 1, 1, 0)
        self.scale = channels**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = torch.chunk(self.qkv(x), 3, dim=1)
        b, c, h, w = q.shape
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)
        weights = torch.softmax(k * self.scale, dim=-1)
        context = torch.bmm(weights, v)
        y = torch.bmm(q, context).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(y)


class MobileViTv2Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Sequential(DSConv(channels, channels), ConvBNAct(channels, channels, 1, 1, 0))
        self.attn = LinearAttention2D(channels)
        self.fuse = ResDSBlock(channels * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        global_ctx = self.attn(local)
        return self.fuse(torch.cat([local, global_ctx], dim=1))


class MobileViTv2Seg(nn.Module):
    """Dependency-light MobileViTv2-style segmentation baseline."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.e1 = nn.Sequential(ConvBNAct(3, base), ResDSBlock(base, base))
        self.e2 = nn.Sequential(ResDSBlock(base, base * 2, 2), MobileViTv2Block(base * 2))
        self.e3 = nn.Sequential(ResDSBlock(base * 2, base * 4, 2), MobileViTv2Block(base * 4))
        self.e4 = nn.Sequential(ResDSBlock(base * 4, base * 8, 2), MobileViTv2Block(base * 8))
        self.d3 = ResDSBlock(base * 12, base * 4)
        self.d2 = ResDSBlock(base * 6, base * 2)
        self.d1 = ResDSBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


class OfficialMobileViTv2Seg(nn.Module):
    """Apple ml-cvnets MobileViTv2 backbone with a lightweight binary decoder."""

    def __init__(self, width_multiplier: float = 0.5, decoder_ch: int = 96):
        super().__init__()
        root = _repo_root() / "MobileViTv2" / "ml-cvnets-main"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from cvnets.models.classification.mobilevit_v2 import MobileViTv2
        except Exception as exc:  # pragma: no cover - depends on optional external repo.
            raise ImportError(
                "Cannot import official MobileViTv2 from MobileViTv2/ml-cvnets-main. "
                "Keep that folder at the repository root on Kaggle."
            ) from exc

        opts = SimpleNamespace()
        option_values = {
            "common.enable_coreml_compatible_module": False,
            "dev.device": "cuda",
            "model.activation.name": "swish",
            "model.activation.inplace": False,
            "model.activation.neg_slope": 0.1,
            "model.classification.activation.inplace": False,
            "model.classification.activation.name": None,
            "model.classification.activation.neg_slope": 0.1,
            "model.classification.enable_layer_wise_lr_decay": False,
            "model.classification.gradient_checkpointing": False,
            "model.classification.layer_wise_lr_decay_rate": 1.0,
            "model.classification.mitv2.attn_dropout": 0.0,
            "model.classification.mitv2.attn_norm_layer": "layer_norm_2d",
            "model.classification.mitv2.dropout": 0.0,
            "model.classification.mitv2.ffn_dropout": 0.0,
            "model.classification.mitv2.width_multiplier": width_multiplier,
            "model.classification.n_classes": 1000,
            "model.layer.global_pool": "mean",
            "model.layer.linear_init": "normal",
            "model.normalization.groups": 1,
            "model.normalization.momentum": 0.1,
            "model.normalization.name": "batch_norm",
            "scheduler.is_iteration_based": True,
            "scheduler.max_iterations": 100000,
            "scheduler.warmup_iterations": 10000,
        }
        for key, value in option_values.items():
            setattr(opts, key, value)

        self.backbone = MobileViTv2(opts=opts)
        conf = self.backbone.model_conf_dict
        c1 = conf["layer1"]["out"]
        c2 = conf["layer2"]["out"]
        c3 = conf["layer3"]["out"]
        c4 = conf["layer4"]["out"]
        c5 = conf["layer5"]["out"]
        self.p5 = ConvBNAct(c5, decoder_ch, 1, 1, 0)
        self.p4 = ConvBNAct(c4, decoder_ch, 1, 1, 0)
        self.p3 = ConvBNAct(c3, decoder_ch, 1, 1, 0)
        self.p2 = ConvBNAct(c2, decoder_ch // 2, 1, 1, 0)
        self.p1 = ConvBNAct(c1, decoder_ch // 2, 1, 1, 0)
        self.f4 = ResDSBlock(decoder_ch * 2, decoder_ch)
        self.f3 = ResDSBlock(decoder_ch * 2, decoder_ch)
        self.f2 = ResDSBlock(decoder_ch + decoder_ch // 2, decoder_ch // 2)
        self.f1 = ResDSBlock(decoder_ch, decoder_ch // 2)
        self.out = nn.Conv2d(decoder_ch // 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = self.backbone.extract_end_points_all(x, use_l5=True, use_l5_exp=False)
        l1, l2, l3, l4, l5 = feats["out_l1"], feats["out_l2"], feats["out_l3"], feats["out_l4"], feats["out_l5"]
        d4 = self.f4(torch.cat([F.interpolate(self.p5(l5), size=l4.shape[-2:], mode="bilinear", align_corners=False), self.p4(l4)], 1))
        d3 = self.f3(torch.cat([F.interpolate(d4, size=l3.shape[-2:], mode="bilinear", align_corners=False), self.p3(l3)], 1))
        d2 = self.f2(torch.cat([F.interpolate(d3, size=l2.shape[-2:], mode="bilinear", align_corners=False), self.p2(l2)], 1))
        d1 = self.f1(torch.cat([F.interpolate(d2, size=l1.shape[-2:], mode="bilinear", align_corners=False), self.p1(l1)], 1))
        return F.interpolate(self.out(d1), size=size, mode="bilinear", align_corners=False)


class SoftExpertContext(nn.Module):
    """Mamba Goes HoME-inspired lightweight mixture-of-experts context block."""

    def __init__(self, channels: int, experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                DSConv(channels, channels, dilation=1),
                DSConv(channels, channels, dilation=2),
                DSConv(channels, channels, dilation=3),
                SelectiveScan2D(channels),
            ][:experts]
        )
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, len(self.experts), 1))
        self.fuse = ConvBNAct(channels, channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.router(x).flatten(1), dim=1)
        ys = torch.stack([expert(x) for expert in self.experts], dim=1)
        y = (ys * weights[:, :, None, None, None]).sum(dim=1)
        return x + self.fuse(y)


class MambaHoMESeg(nn.Module):
    """2D skin-lesion baseline adapted from the HoME mixture-of-experts idea."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.stem = ConvBNAct(3, base)
        self.e1 = ResDSBlock(base, base)
        self.e2 = ResDSBlock(base, base * 2, 2)
        self.e3 = ResDSBlock(base * 2, base * 4, 2)
        self.e4 = ResDSBlock(base * 4, base * 8, 2)
        self.context = SoftExpertContext(base * 8)
        self.d3 = ResDSBlock(base * 12, base * 4)
        self.d2 = ResDSBlock(base * 6, base * 2)
        self.d1 = ResDSBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.context(self.e4(e3))
        d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


class LiteMambaBoundSeg(nn.Module):
    """LiteMamba-Bound-style baseline without mamba_ssm or custom CUDA kernels."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.stem = ConvBNAct(3, base)
        self.e1 = ResDSBlock(base, base)
        self.e2 = nn.Sequential(ResDSBlock(base, base * 2, 2), SelectiveScan2D(base * 2))
        self.e3 = nn.Sequential(ResDSBlock(base * 2, base * 4, 2), SelectiveScan2D(base * 4))
        self.e4 = nn.Sequential(ResDSBlock(base * 4, base * 8, 2), MultiScaleContextBlock(base * 8))
        self.f3 = BoundaryGuidedFusion(base * 8, base * 4, base * 4)
        self.f2 = BoundaryGuidedFusion(base * 4, base * 2, base * 2)
        self.f1 = BoundaryGuidedFusion(base * 2, base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3, b3 = self.f3(e4, e3)
        d2, b2 = self.f2(d3, e2)
        d1, b1 = self.f1(d2, e1)
        logits = self.out(d1)
        boundary = b1 + F.interpolate(b2, size=b1.shape[-2:], mode="bilinear", align_corners=False)
        boundary = boundary + F.interpolate(b3, size=b1.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits, "boundary": boundary / 3.0}


class ProbabilityOutputWrapper(nn.Module):
    """Adapts external models that return probabilities or nested outputs to logits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @staticmethod
    def _last_tensor(outputs):
        if torch.is_tensor(outputs):
            return outputs
        if isinstance(outputs, dict):
            return outputs["logits"] if "logits" in outputs else next(v for v in outputs.values() if torch.is_tensor(v))
        if isinstance(outputs, (tuple, list)):
            for item in reversed(outputs):
                try:
                    return ProbabilityOutputWrapper._last_tensor(item)
                except StopIteration:
                    continue
        raise TypeError(f"Unsupported output type: {type(outputs)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._last_tensor(self.model(x))
        if y.min().detach() >= 0 and y.max().detach() <= 1:
            y = torch.logit(y.clamp(1e-4, 1 - 1e-4))
        return y


def _load_class(module_path: Path, class_name: str):
    module_name = f"_skinseg_external_{module_path.stem}_{class_name}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _external_model(name: str) -> nn.Module:
    root = _repo_root()
    if name == "malunet":
        cls = _load_class(root / "MALUNet" / "MALUNet-main" / "models" / "malunet.py", "MALUNet")
        return ProbabilityOutputWrapper(cls(num_classes=1, input_channels=3))
    if name in {"lbunet", "lb-unet"}:
        cls = _load_class(root / "LB-UNet" / "LB-UNet-main" / "lbunet.py", "LBUNet")
        return ProbabilityOutputWrapper(cls(num_classes=1, input_channels=3))
    if name in {"egeunet", "ege-unet"}:
        cls = _load_class(root / "EGE-UNet" / "EGE-UNet" / "models" / "egeunet.py", "EGEUNet")
        return ProbabilityOutputWrapper(cls(num_classes=1, input_channels=3, bridge=True, gt_ds=True))
    raise ValueError(name)

def get_model(name: str) -> nn.Module:
    key = name.lower()
    if key == "unet":
        return UNet(base=32)
    if key in {"classic_unet", "classic-unet", "unet_classic"}:
        return ClassicUNet(base=32)
    if key == "tinyunet":
        return TinyUNet(base=16)
    if key in {"egelite", "ege"}:
        return EGELite(base=16)
    if key in {"malunet", "lbunet", "lb-unet", "egeunet", "ege-unet"}:
        return _external_model(key)
    if key in {"unext", "unextlite", "unext-lite"}:
        return UNeXtLite(base=16)
    if key in {"mobilevitv2", "mobilevit-v2", "mobilevit"}:
        return MobileViTv2Seg(base=16)
    if key in {"mobilevitv2_paper", "mobilevitv2-paper", "mobilevitv2_187m"}:
        return OfficialMobileViTv2Seg(width_multiplier=0.5, decoder_ch=96)
    if key in {"mambahome", "mamba_home", "home"}:
        return MambaHoMESeg(base=16)
    if key in {"litemamba_bound", "litemambabound", "litemamba-bound"}:
        return LiteMambaBoundSeg(base=16)
    if key == "blmnet":
        return BLMNet(base=16, use_scan=True, use_boundary=True)
    if key == "blmnet_no_boundary":
        return BLMNet(base=16, use_scan=True, use_boundary=False)
    if key == "blmnet_no_scan":
        return BLMNet(base=16, use_scan=False, use_boundary=True)
    raise ValueError(f"Unknown model: {name}")


def unwrap_logits(outputs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    return outputs["logits"] if isinstance(outputs, dict) else outputs
