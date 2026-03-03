# src/explain/gradcam.py
"""
Wrapper for Grad-CAM / overlay. Tries the real implementations you have:
- src/explain/gradcam_06.py
- src/explain/overlay_07.py
If none found, returns a demo heatmap + overlay for visuals.
Exposes: gradcam(pil_img, model=None, target_class=None) -> (heatmap, overlay_pil)
"""

from typing import Tuple, Any
import numpy as np
from PIL import Image

_real_gradcam = None

# try gradcam_06 first
try:
    from src.explain.gradcam_06 import gradcam as _real_gradcam  # type: ignore
except Exception:
    _real_gradcam = None

# try overlay_07 (might have make_overlay or gradcam_overlay)
if _real_gradcam is None:
    try:
        from src.explain.overlay_07 import gradcam_overlay as _real_gradcam  # type: ignore
    except Exception:
        try:
            from src.explain.overlay_07 import make_overlay as _real_gradcam  # type: ignore
        except Exception:
            _real_gradcam = None

def _make_demo_heatmap(H: int, W: int) -> np.ndarray:
    ys = np.linspace(-1, 1, H)[:, None]
    xs = np.linspace(-1, 1, W)[None, :]
    dist = np.sqrt(xs**2 + ys**2)
    hm = np.clip(1.0 - dist, 0.0, 1.0)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    return hm.astype(np.float32)

def _demo_overlay(pil_img: Image.Image, hm: np.ndarray) -> Image.Image:
    W, H = pil_img.size
    heat_uint8 = (hm * 255).astype("uint8")
    heat_rgb = np.stack([heat_uint8, np.zeros_like(heat_uint8), np.zeros_like(heat_uint8)], axis=2)
    heat_pil = Image.fromarray(heat_rgb).resize(pil_img.size).convert("RGBA")
    base = pil_img.convert("RGBA")
    overlay = Image.blend(base, heat_pil, alpha=0.4)
    return overlay

def gradcam(pil_img: Image.Image, model=None, target_class=None) -> Tuple[np.ndarray, Image.Image]:
    # try real implementation
    if callable(_real_gradcam):
        try:
            out = _real_gradcam(pil_img, model=model, target_class=target_class)
            if isinstance(out, tuple) and len(out) >= 2:
                a, b = out[0], out[1]
                # prefer (heatmap, overlay)
                if hasattr(a, "shape") and (isinstance(b, Image.Image) or hasattr(b, "convert")):
                    return a, b
                if hasattr(b, "shape") and (isinstance(a, Image.Image) or hasattr(a, "convert")):
                    return b, a
            if hasattr(out, "shape"):
                hm = out
                return hm, _demo_overlay(pil_img, hm)
        except Exception:
            pass

    # fallback demo
    W, H = pil_img.size
    hm = _make_demo_heatmap(H=H, W=W)
    ov = _demo_overlay(pil_img, hm)
    return hm, ov
