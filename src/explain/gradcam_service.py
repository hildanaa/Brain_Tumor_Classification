import torch
from PIL import Image
from torchvision import transforms

from src.explain.gradcam_06 import GradCAM
from src.explain.mask_09 import brain_mask_from_pil, apply_mask
from src.explain.overlay_07 import overlay_heatmap_on_pil
from src.models.model_factory_03 import build_resnet18

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]

def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def gradcam_overlay(pil_img: Image.Image,
                    checkpoint_path="outputs/checkpoints/best_model.pt",
                    device=None,
                    skip_no_tumor=True):
    """
    Returns:
        overlay_pil (or None),
        heatmap_np (or None),
        pred_label
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, _ = build_resnet18(num_classes=4, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    pil_224 = pil_img.convert("RGB").resize((224, 224))
    tfm = build_eval_transform(224)
    x = tfm(pil_224).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_label = CLASS_NAMES[pred_idx]

    # If predicted no tumor → skip CAM
    if skip_no_tumor and pred_label == "notumor":
        return None, None, pred_label

    # Use layer4 + Grad-CAM++
    target_layer = model.layer4
    cam = GradCAM(model, target_layer)

    heatmap = cam.generate_pp(x, class_idx=pred_idx)

    # Mask outside brain
    mask = brain_mask_from_pil(pil_224, size=(224,224))
    heatmap = apply_mask(heatmap, mask)

    overlay = overlay_heatmap_on_pil(
        pil_224,
        heatmap,
        alpha=0.30,
        top_quantile=0.85
    )

    cam.remove_hooks()

    return overlay, heatmap.detach().cpu().numpy(), pred_label