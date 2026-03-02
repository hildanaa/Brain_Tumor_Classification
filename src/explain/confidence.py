# src/explain/confidence.py

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def topk(probs: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
    p = np.asarray(probs, dtype=float).ravel()
    if p.size == 0:
        return []
    idxs = np.argsort(-p)[:k]
    return [(int(i), float(p[i])) for i in idxs]


def confidence_from_probs(probs: np.ndarray) -> Dict:
    """
    Input:
      probs: array-like of class probabilities (sum ~ 1)
    Output:
      dict:
        level: "High" | "Medium" | "Low"
        p_top1, p_top2, margin
    """
    p = np.asarray(probs, dtype=float).ravel()
    if p.size < 2:
        p_top1 = float(p[0]) if p.size == 1 else 0.0
        return {"level": "Low", "p_top1": p_top1, "p_top2": 0.0, "margin": p_top1}

    order = np.argsort(-p)
    p_top1 = float(p[order[0]])
    p_top2 = float(p[order[1]])
    margin = float(p_top1 - p_top2)

    # Rules (tuneable)
    if p_top1 >= 0.80 and margin >= 0.20:
        level = "High"
    elif p_top1 >= 0.60:
        level = "Medium"
    else:
        level = "Low"

    return {"level": level, "p_top1": p_top1, "p_top2": p_top2, "margin": margin}