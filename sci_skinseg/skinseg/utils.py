from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    params = count_params(model)
    buffers = sum(b.numel() for b in model.buffers())
    return (params + buffers) * 4 / (1024**2)


def estimate_flops(model: nn.Module, img_size: int, device: torch.device) -> int:
    flops = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        out = output
        batch, out_ch, out_h, out_w = out.shape
        k_h, k_w = module.kernel_size
        in_ch = module.in_channels
        groups = module.groups
        flops += int(batch * out_ch * out_h * out_w * (in_ch // groups) * k_h * k_w)

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        batch = output.shape[0] if output.ndim > 1 else 1
        flops += int(batch * module.in_features * module.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        model(dummy)
    for h in hooks:
        h.remove()
    model.train(was_training)
    return flops


@torch.no_grad()
def measure_fps(model: nn.Module, img_size: int, device: torch.device, steps: int = 50, warmup: int = 10) -> float:
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return steps / max(elapsed, 1e-9)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return torch.autocast(device_type="cpu", enabled=False)
