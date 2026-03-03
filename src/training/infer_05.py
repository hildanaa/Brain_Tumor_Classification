# src/training/infer_05.py

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.model_factory_03 import build_resnet18
from src.data.dataset_01 import ID2LABEL

CHECKPOINT_PATH = os.path.join("outputs", "checkpoints", "best_model.pt")


def build_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path: str = CHECKPOINT_PATH, device: str | None = None):
    """
    Returns a torch model ready for inference (eval mode).
    This is what Grad-CAM needs.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_resnet18(num_classes=4, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Your checkpoint key is "model_state"
    model.load_state_dict(ckpt["model_state"])

    model.to(device).eval()
    return model, device


@torch.no_grad()
def predict(pil_img: Image.Image, checkpoint_path: str = CHECKPOINT_PATH, device: str | None = None):
    """
    App expects: (probs, pred_label, pred_id)
    """
    model, device = load_model(checkpoint_path=checkpoint_path, device=device)

    tfm = build_eval_transform(224)
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = ID2LABEL[pred_id]

    return probs, pred_label, pred_id


# Keep your old API too (optional, for compatibility)
@torch.no_grad()
def predict_pil(pil_img: Image.Image, checkpoint_path: str = CHECKPOINT_PATH, device: str | None = None):
    probs, pred_label, pred_id = predict(pil_img, checkpoint_path=checkpoint_path, device=device)
    return probs, pred_id, pred_label
