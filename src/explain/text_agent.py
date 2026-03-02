# src/explain/text_agent.py

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from src.explain.confidence import confidence_from_probs, topk


def _describe_location(center_xy: List[float]) -> str:
    """
    center_xy: [x,y] normalized in [0,1]
    Returns a human-friendly region description.
    """
    x, y = float(center_xy[0]), float(center_xy[1])

    def horiz(xv: float) -> str:
        if xv < 0.33:
            return "left"
        if xv > 0.66:
            return "right"
        return "center"

    def vert(yv: float) -> str:
        if yv < 0.33:
            return "upper"
        if yv > 0.66:
            return "lower"
        return "central"

    h = horiz(x)
    v = vert(y)

    if v == "central" and h == "center":
        return "the central area of the image"
    if v == "central":
        return f"the {h} side of the image"
    if h == "center":
        return f"the {v} area of the image"
    return f"the {v}-{h} area of the image"


def _describe_attention_pattern(features: Dict) -> str:
    """
    Decide whether attention is localized or diffuse using focus_ratio, spread, num_regions.
    """
    focus_ratio = float(features.get("focus_ratio", 0.0))
    spread = float(features.get("spread", 1.0))
    num_regions = features.get("num_regions", None)

    localized_votes = 0
    diffuse_votes = 0

    if focus_ratio <= 0.12:
        localized_votes += 1
    elif focus_ratio >= 0.25:
        diffuse_votes += 1

    if spread <= 0.22:
        localized_votes += 1
    elif spread >= 0.35:
        diffuse_votes += 1

    if isinstance(num_regions, int):
        if num_regions <= 2:
            localized_votes += 1
        elif num_regions >= 4:
            diffuse_votes += 1

    if diffuse_votes > localized_votes:
        return "diffuse (spread across multiple areas)"
    if localized_votes > diffuse_votes:
        return "localized (concentrated in a small area)"
    return "mixed (some concentration, but not fully localized)"


def _format_topk(probs: np.ndarray, class_names: Optional[List[str]] = None, k: int = 3) -> str:
    pairs = topk(probs, k=k)
    if not pairs:
        return ""

    parts = []
    for idx, p in pairs:
        label = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
        parts.append(f"{label}: {p:.2f}")
    return ", ".join(parts)


def explain(
    pred_label: str,
    probs: np.ndarray,
    heatmap_features: Dict,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Inputs:
      pred_label: e.g., "glioma" / "no_tumor"
      probs: np array of probabilities
      heatmap_features: dict from extract_features()
      class_names: optional list aligned with probs indices

    Output:
      short, safe, UX-friendly explanation string
    """
    conf = confidence_from_probs(probs)
    level = conf["level"]
    p_top1 = conf["p_top1"]
    margin = conf["margin"]

    loc = _describe_location(heatmap_features.get("center_of_mass_xy", [0.5, 0.5]))
    pattern = _describe_attention_pattern(heatmap_features)

    topk_str = _format_topk(probs, class_names=class_names, k=3)

    # Avoid "diagnosis" language. This is model behavior + uncertainty.
    if level == "High":
        headline = f"Prediction: **{pred_label}** (high confidence)"
        caution = ""
    elif level == "Medium":
        headline = f"Prediction: **{pred_label}** (medium confidence)"
        caution = "This result is moderately certain—small changes in the image could affect the prediction."
    else:
        headline = f"Prediction: **{pred_label}** (low confidence)"
        caution = "This result is uncertain. Treat it as a weak signal rather than a firm conclusion."

    rationale = (
        f"The model’s attention was mainly in {loc}, and the pattern looked {pattern}. "
        f"(Top probability: {p_top1:.2f}, margin vs. second choice: {margin:.2f}.)"
    )

    probs_line = f"Top probabilities: {topk_str}." if topk_str else ""

    safety = (
        "Note: This explanation describes what the model focused on in the image. "
        "It does not confirm the presence or absence of a medical condition."
    )

    parts = [headline, rationale]
    if probs_line:
        parts.append(probs_line)
    if caution:
        parts.append(caution)
    parts.append(safety)

    return "\n\n".join(parts)