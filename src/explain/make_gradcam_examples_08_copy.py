import os
import random
import torch
from PIL import Image
from torchvision import transforms

from src.explain.mask_09 import brain_mask_from_pil, apply_mask
from src.models.model_factory_03 import build_resnet18
from src.explain.gradcam_06 import GradCAM
from src.explain.overlay_07 import overlay_heatmap_on_pil

def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "outputs/checkpoints/best_model.pt"

    model, target_layer = build_resnet18(num_classes=4, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    cam = GradCAM(model, target_layer)
    tfm = build_eval_transform(224)

    # Toma imágenes desde los splits (elige rutas existentes en tu proyecto)
    # Aquí asumo que tienes data/raw/Training y data/raw/Testing.
    roots = [r"Brain Tumor data\Training", r"Brain Tumor data\Testing"]
    all_imgs = []
    for r in roots:
        if not os.path.exists(r):
            continue
        for root, _, files in os.walk(r):
            for f in files:
                if f.lower().endswith((".png",".jpg",".jpeg")):
                    all_imgs.append(os.path.join(root, f))

    if not all_imgs:
        raise RuntimeError("No images found under data/raw/Training or data/raw/Testing. Adjust paths in script.")

    os.makedirs("outputs/figures/gradcam_examples", exist_ok=True)

    sample = random.sample(all_imgs, k=min(12, len(all_imgs)))
    for i, path in enumerate(sample, start=1):
        pil = Image.open(path).convert("RGB")
        pil_224 = pil.resize((224,224))
        x = tfm(pil_224).unsqueeze(0).to(device)

        heatmap = cam.generate(x)
        mask = brain_mask_from_pil(pil_224, size=(224,224))
        heatmap = apply_mask(heatmap, mask)

        overlay = overlay_heatmap_on_pil(pil_224, heatmap, alpha=0.30, top_quantile=0.80)

        out = f"outputs/figures/gradcam_examples/example_{i:02d}.png"
        overlay.save(out)
        print("saved", out)

    cam.remove_hooks()

if __name__ == "__main__":
    main()