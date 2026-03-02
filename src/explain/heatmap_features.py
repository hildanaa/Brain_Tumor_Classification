# src/explain/heatmap_features.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class HeatmapFeatures:
    focus_ratio: float                 # fraction of pixels above threshold
    peak_intensity: float              # max value in heatmap
    mean_intensity: float              # average heat
    center_of_mass_xy: Tuple[float, float]  # (x,y) normalized [0,1]
    spread: float                      # rough dispersion (0=concentrated, 1=spread)
    num_regions: Optional[int]         # connected components above threshold (optional)
    threshold: float


def _safe_normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Ensure heatmap is float32 in [0,1], tolerate weird inputs."""
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be 2D [H,W], got shape={hm.shape}")
    hm = np.nan_to_num(hm, nan=0.0, posinf=1.0, neginf=0.0)
    hm_min, hm_max = float(hm.min()), float(hm.max())
    if hm_max - hm_min < 1e-8:
        return np.zeros_like(hm, dtype=np.float32)
    hm = (hm - hm_min) / (hm_max - hm_min)
    hm = np.clip(hm, 0.0, 1.0).astype(np.float32)
    return hm


def _center_of_mass_xy(hm: np.ndarray) -> Tuple[float, float]:
    """Weighted center of mass normalized to [0,1] in x,y."""
    H, W = hm.shape
    total = float(hm.sum())
    if total <= 1e-8:
        return (0.5, 0.5)  # neutral center when nothing is active

    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(ys, xs, indexing="ij")
    cy = float((hm * y_grid).sum() / total)
    cx = float((hm * x_grid).sum() / total)

    return (cx / max(W - 1, 1), cy / max(H - 1, 1))


def _spread(hm: np.ndarray, com_xy: Tuple[float, float]) -> float:
    """Rough dispersion: weighted average distance to COM, normalized ~[0,1]."""
    H, W = hm.shape
    total = float(hm.sum())
    if total <= 1e-8:
        return 1.0

    cx = com_xy[0] * max(W - 1, 1)
    cy = com_xy[1] * max(H - 1, 1)

    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    max_dist = float(np.sqrt((W - 1) ** 2 + (H - 1) ** 2)) + 1e-8
    return float((hm * dist).sum() / total / max_dist)


def _count_regions(binary_mask: np.ndarray) -> Optional[int]:
    """
    Count connected components in a binary mask.
    Returns None if no library is available.
    """
    try:
        # Option A: scipy
        from scipy.ndimage import label  # type: ignore
        _, n = label(binary_mask.astype(np.uint8))
        return int(n)
    except Exception:
        pass

    try:
        # Option B: skimage
        from skimage.measure import label as sklabel  # type: ignore
        n = int(sklabel(binary_mask.astype(np.uint8), connectivity=1).max())
        return n
    except Exception:
        return None


def extract_features(
    heatmap: np.ndarray,
    threshold: float = 0.60,
    compute_regions: bool = True,
) -> Dict:
    """
    Input:
      heatmap: float array [H,W] (expected [0,1], but we normalize safely)
    Output:
      dict with robust features (JSON-serializable)
    """
    hm = _safe_normalize_heatmap(heatmap)

    focus_ratio = float((hm > threshold).mean())
    peak = float(hm.max())
    mean = float(hm.mean())
    com_xy = _center_of_mass_xy(hm)
    spread = _spread(hm, com_xy)

    num_regions = None
    if compute_regions:
        mask = hm > threshold
        num_regions = _count_regions(mask)

    feats = HeatmapFeatures(
        focus_ratio=focus_ratio,
        peak_intensity=peak,
        mean_intensity=mean,
        center_of_mass_xy=com_xy,
        spread=spread,
        num_regions=num_regions,
        threshold=threshold,
    )

    return {
        "focus_ratio": feats.focus_ratio,
        "peak_intensity": feats.peak_intensity,
        "mean_intensity": feats.mean_intensity,
        "center_of_mass_xy": [feats.center_of_mass_xy[0], feats.center_of_mass_xy[1]],
        "spread": feats.spread,
        "num_regions": feats.num_regions,  # can be None
        "threshold": feats.threshold,
    }