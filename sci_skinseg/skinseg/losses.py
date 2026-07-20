from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    union = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def boundary_weighted_loss(logits: torch.Tensor, target: torch.Tensor, boundary: torch.Tensor, weight: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = 1.0 + weight * boundary
    return (bce * weights).mean() + dice_loss(logits, target)


def multitask_loss(outputs: dict[str, torch.Tensor] | torch.Tensor, target: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return bce_dice_loss(outputs, target)
    seg = boundary_weighted_loss(outputs["logits"], target, boundary)
    if "boundary" not in outputs:
        return seg
    bnd = F.binary_cross_entropy_with_logits(outputs["boundary"], boundary)
    return seg + 0.4 * bnd
