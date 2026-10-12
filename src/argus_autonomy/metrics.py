"""Evaluation metrics. All compare the estimator's belief with the hidden truth."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .perception import binary_entropy


def mean_entropy(belief: np.ndarray) -> float:
    """Average per-cell uncertainty, bits/cell."""
    return float(binary_entropy(belief).mean())


def f1_iou(belief: np.ndarray, truth: np.ndarray, thr: float = 0.5) -> tuple[float, float]:
    est = belief >= thr
    tp = np.sum(est & truth)
    fp = np.sum(est & ~truth)
    fn = np.sum(~est & truth)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return float(f1), float(iou)


def perimeter(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)


def boundary_error(belief: np.ndarray, truth: np.ndarray, cell_m: float, thr: float = 0.5) -> float:
    """Mean symmetric distance between estimated and true fire perimeters, metres."""
    pe, pt = perimeter(belief >= thr), perimeter(truth)
    if not pe.any() or not pt.any():
        return float("nan")
    d_to_e = distance_transform_edt(~pe)
    d_to_t = distance_transform_edt(~pt)
    return float(0.5 * (d_to_e[pt].mean() + d_to_t[pe].mean()) * cell_m)


def confirmed(belief: np.ndarray, truth: np.ndarray, thr: float) -> tuple[int, int]:
    """(true confirmations, false confirmations) at threshold `thr`."""
    c = belief >= thr
    return int(np.sum(c & truth)), int(np.sum(c & ~truth))
