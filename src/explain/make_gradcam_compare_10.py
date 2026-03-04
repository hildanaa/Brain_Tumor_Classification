import os
import random
import torch
from PIL import Image
from torchvision import transforms

from src.explain.mask_09 import brain_mask_from_pil, apply_mask
from src.explain.gradcam_06 import GradCAM
from src.explain.overlay_07 import overlay_heatmap_on_pil
from src.models.model_factory_03 import build_resnet18

def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def list_images():
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
        raise RuntimeError("No images found. Check your dataset path roots.")
    return all_imgs

def load_model(ckpt_path, device):
    model, _ = build_resnet18(num_classes=4, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model

def get_target_layer(model, layer_name: str):
    if layer_name == "layer3":
        return model.layer3
    if layer_name == "layer4":
        return model.layer4
    raise ValueError("layer_name must be 'layer3' or 'layer4'")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "outputs/checkpoints/best_model.pt"
    out_dir = "outputs/figures/gradcam_compare"
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(ckpt_path, device)
    tfm = build_eval_transform(224)

    all_imgs = list_images()
    sample = random.sample(all_imgs, k=min(6, len(all_imgs)))  # 6 imágenes para comparar rápido

    for i, path in enumerate(sample, start=1):
        pil = Image.open(path).convert("RGB")
        pil_224 = pil.resize((224, 224))
        x = tfm(pil_224).unsqueeze(0).to(device)

        # 1) fija class_idx explícito (importantísimo)
        with torch.no_grad():
            logits = model(x)
            pred_idx = int(torch.argmax(logits, dim=1).item())

        # mask para que no se pinte fuera del cráneo
        mask = brain_mask_from_pil(pil_224, size=(224,224))

        for layer_name in ["layer3", "layer4"]:
            target_layer = get_target_layer(model, layer_name)
            cam = GradCAM(model, target_layer)

            # --- Grad-CAM normal ---
            heatmap = cam.generate(x, class_idx=pred_idx)
            heatmap = apply_mask(heatmap, mask)
            overlay = overlay_heatmap_on_pil(pil_224, heatmap, alpha=0.30, top_quantile=0.85)
            overlay.save(os.path.join(out_dir, f"img{i:02d}_{layer_name}_gradcam.png"))

            # --- Grad-CAM++ ---
            heatmap_pp = cam.generate_pp(x, class_idx=pred_idx)
            heatmap_pp = apply_mask(heatmap_pp, mask)
            overlay_pp = overlay_heatmap_on_pil(pil_224, heatmap_pp, alpha=0.30, top_quantile=0.85)
            overlay_pp.save(os.path.join(out_dir, f"img{i:02d}_{layer_name}_gradcampp.png"))

            cam.remove_hooks()

        print(f"Saved comparisons for image {i:02d}: {path}")

    print(f"\nDone. Check: {out_dir}")

if __name__ == "__main__":
    main()