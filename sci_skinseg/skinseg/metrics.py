from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


EPS = 1e-7


def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def confusion(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    return float(tp), float(tn), float(fp), float(fn)


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float(max(pred.shape))
    pred_border = np.logical_xor(pred, binary_erosion(pred))
    target_border = np.logical_xor(target, binary_erosion(target))
    dt_pred = distance_transform_edt(~pred_border)
    dt_target = distance_transform_edt(~target_border)
    dists = np.concatenate([dt_target[pred_border], dt_pred[target_border]])
    if dists.size == 0:
        return 0.0
    return float(np.percentile(dists, 95))


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    tp, tn, fp, fn = confusion(pred, target)
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    acc = (tp + tn + EPS) / (tp + tn + fp + fn + EPS)
    sen = (tp + EPS) / (tp + fn + EPS)
    spe = (tn + EPS) / (tn + fp + EPS)
    pre = (tp + EPS) / (tp + fp + EPS)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "accuracy": float(acc),
        "sensitivity": float(sen),
        "specificity": float(spe),
        "precision": float(pre),
        "f1": float(dice),
        "hd95": hd95(pred, target),
    }


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
