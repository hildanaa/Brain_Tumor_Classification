import sys
import os
sys.path.append(os.path.abspath("."))

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.models.model_factory_03 import build_resnet18
from src.explain.gradcam_06 import GradCAM
from src.explain.overlay_07 import overlay_heatmap_on_pil
from src.explain.heatmap_features import extract_features
from src.explain.text_agent import explain

# ✅ Fallback directo (sin imports de dataset)
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]


def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@st.cache_resource
def load_model_and_cam(checkpoint_path: str, device: str):
    model, target_layer = build_resnet18(num_classes=len(CLASS_NAMES), pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    cam = GradCAM(model, target_layer)
    return model, cam


def predict_probs(model, pil_img: Image.Image, device: str):
    tfm = build_eval_transform(224)
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_id]
    return x, probs, pred_id, pred_label


def probs_table(probs: np.ndarray):
    pairs = [(CLASS_NAMES[i], float(probs[i])) for i in range(len(CLASS_NAMES))]
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pairs


def main():
    st.set_page_config(page_title="Brain Tumor MRI - Demo", layout="wide")

    st.title("Brain Tumor MRI Classifier (Demo)")
    st.caption("Upload → Prediction → Grad-CAM → Explanation")

    with st.sidebar:
        st.header("Settings")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"**Device:** {device}")

        default_ckpt = "outputs/checkpoints/best_model.pt"
        ckpt_path = st.text_input("Checkpoint path", value=default_ckpt)

        alpha = st.slider("Overlay alpha", min_value=0.0, max_value=0.9, value=0.45, step=0.05)
        threshold = st.slider("Heatmap threshold (features)", 0.1, 0.9, 0.60, 0.05)

        st.markdown("---")
        st.subheader("Disclaimer")
        st.info(
            "This app is a **technical demo**. It does **not** provide medical diagnosis. "
            "Predictions may be wrong or biased. Always consult qualified clinicians."
        )

    uploaded = st.file_uploader("Upload an MRI image (jpg/png)", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.stop()

    pil_img = Image.open(uploaded).convert("RGB")

    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()

    model, cam = load_model_and_cam(ckpt_path, device)

    # Predict
    x, probs, pred_id, pred_label = predict_probs(model, pil_img, device)

    # Grad-CAM sobre la clase predicha
    heatmap_t = cam.generate(x, class_idx=pred_id)  # torch [H,W] en [0,1]
    overlay_img = overlay_heatmap_on_pil(pil_img, heatmap_t, alpha=alpha)

    # Features + explanation
    heatmap_np = heatmap_t.detach().cpu().numpy()
    feats = extract_features(heatmap_np, threshold=threshold, compute_regions=True)
    explanation = explain(
        pred_label=pred_label,
        probs=probs,
        heatmap_features=feats,
        class_names=CLASS_NAMES
    )

    # UI
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Input")
        st.image(pil_img, use_container_width=True)

    with c2:
        st.subheader("Grad-CAM Overlay")
        st.image(overlay_img, use_container_width=True)

    st.markdown("---")

    c3, c4 = st.columns([1, 1])
    with c3:
        st.subheader("Probabilities")
        pairs = probs_table(probs)
        st.table({"class": [p[0] for p in pairs], "prob": [round(p[1], 4) for p in pairs]})
        st.bar_chart({p[0]: p[1] for p in pairs})

    with c4:
        st.subheader("Explanation")
        st.markdown(explanation)
        with st.expander("Heatmap features (debug)"):
            st.json(feats)

    st.markdown("---")
    st.warning(
        "**Important:** This output is an automated model prediction + visualization of attention. "
        "It should not be used for diagnosis or treatment decisions."
    )


if __name__ == "__main__":
    main()
