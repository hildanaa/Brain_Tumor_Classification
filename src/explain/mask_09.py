import numpy as np
import cv2
import torch
from PIL import Image

def brain_mask_from_pil(pil_img: Image.Image, size=(224,224)) -> torch.Tensor:
    img = np.array(pil_img.convert("L").resize(size)).astype(np.uint8)

    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    mask = (mask > 0).astype(np.float32)
    return torch.tensor(mask)

def apply_mask(heatmap_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    hm = heatmap_t * mask_t.to(heatmap_t.device)
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    return hm