import numpy as np
import cv2
from PIL import Image

def overlay_heatmap_on_pil(
    pil_img: Image.Image,
    heatmap_t,
    alpha: float = 0.30,
    top_quantile: float = 0.80,
    colormap: int = cv2.COLORMAP_TURBO
):
    """
    pil_img: PIL RGB
    heatmap_t: torch tensor [H,W] in [0,1]
    alpha: strength of overlay
    top_quantile: keep only top activations (remove weak noise)
    """
    img = np.array(pil_img.convert("RGB"))
    heatmap = heatmap_t.detach().cpu().numpy().astype(np.float32)

    # Keep only top activations (cleaner visualization)
    thr = np.quantile(heatmap, top_quantile)
    heatmap = np.where(heatmap >= thr, heatmap, 0.0)

    # Re-normalize after thresholding
    maxv = heatmap.max()
    if maxv > 1e-8:
        heatmap = heatmap / (maxv + 1e-8)

    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # ✅ NEW: resize heatmap to match image size if needed
    if heatmap_color.shape[:2] != img.shape[:2]:
        H, W = img.shape[:2]
        heatmap_color = cv2.resize(heatmap_color, (W, H), interpolation=cv2.INTER_LINEAR)

    overlay = (img * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)
    return Image.fromarray(overlay)
