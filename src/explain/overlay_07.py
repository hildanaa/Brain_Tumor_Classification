import numpy as np
import cv2
from PIL import Image

def overlay_heatmap_on_pil(pil_img: Image.Image, heatmap_t, alpha=0.45):
    """
    heatmap_t: torch tensor [H,W] in [0,1]
    returns: PIL Image (overlay)
    """
    img = np.array(pil_img.convert("RGB"))
    heatmap = (heatmap_t.cpu().numpy() * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # ✅ NEW: resize heatmap to match image size if needed
    if heatmap_color.shape[:2] != img.shape[:2]:
        H, W = img.shape[:2]
        heatmap_color = cv2.resize(heatmap_color, (W, H), interpolation=cv2.INTER_LINEAR)

    overlay = (img * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)
    return Image.fromarray(overlay)
