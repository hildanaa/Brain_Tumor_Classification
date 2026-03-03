import os
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
from src.explain.confidence import confidence_from_probs  # usa tu confidence.py :contentReference[oaicite:0]{index=0}

# ✅ Fallback directo para labels
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
PRETTY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary tumor",
    "notumor": "No tumor",
}


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


def badge_confidence(level: str) -> str:
    # Solo UI
    if level == "High":
        return "🟢 High"
    if level == "Medium":
        return "🟠 Medium"
    return "🔴 Low"


def main():
    st.set_page_config(page_title="Brain Tumor MRI - Demo", layout="wide")

    st.markdown(
        """
        <div style="padding: 0.5rem 0 0.2rem 0;">
            <h1 style="margin-bottom: 0.2rem;">🧠 Brain Tumor MRI Classifier</h1>
            <p style="margin-top: 0; opacity: 0.8;">
                Upload an MRI image → model prediction → Grad-CAM overlay → textual explanation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar
    with st.sidebar:
        st.header("Settings")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.caption(f"Running on: **{device}**")

        ckpt_path = st.text_input("Checkpoint path", value="outputs/checkpoints/best_model.pt")

        st.markdown("---")
        st.subheader("Explainability")
        alpha = st.slider("Overlay alpha", 0.0, 0.9, 0.45, 0.05)
        threshold = st.slider("Heatmap threshold (features)", 0.1, 0.9, 0.60, 0.05)

        st.markdown("---")
        st.subheader("Disclaimer")
        st.info(
            "This is a **research/educational demo** and is **not** a medical device. "
            "Do not use for diagnosis or treatment decisions. Consult qualified clinicians."
        )

    # --- Upload
    st.markdown("### 1) Upload")
    uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.stop()

    pil_img = Image.open(uploaded).convert("RGB")

    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found: {ckpt_path}. Make sure it exists in the deployed repo.")
        st.stop()

    # --- Load model
    with st.spinner("Loading model..."):
        model, cam = load_model_and_cam(ckpt_path, device)

    # --- Inference + CAM
    with st.spinner("Running prediction + Grad-CAM..."):
        x, probs, pred_id, pred_label = predict_probs(model, pil_img, device)
        heatmap_t = cam.generate(x, class_idx=pred_id)
        overlay_img = overlay_heatmap_on_pil(pil_img, heatmap_t, alpha=alpha)

        heatmap_np = heatmap_t.detach().cpu().numpy()
        feats = extract_features(heatmap_np, threshold=threshold, compute_regions=True)
        explanation = explain(
            pred_label=pred_label,
            probs=probs,
            heatmap_features=feats,
            class_names=CLASS_NAMES,
        )

        conf = confidence_from_probs(probs)  # :contentReference[oaicite:1]{index=1}

    pretty_pred = PRETTY_NAMES.get(pred_label, pred_label)

    # --- Tabs
    tab1, tab2, tab3 = st.tabs(["📌 Results", "🧠 Explanation", "ℹ️ About"])

    with tab1:
        st.markdown("### 2) Results")

        # Key metrics row
        m1, m2, m3 = st.columns([1, 1, 1])
        with m1:
            st.metric("Prediction", pretty_pred)
        with m2:
            st.metric("Top probability", f"{conf['p_top1']:.3f}")
        with m3:
            st.metric("Confidence", badge_confidence(conf["level"]))

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Input")
            st.image(pil_img, use_container_width=True)

        with c2:
            st.subheader("Grad-CAM overlay")
            st.image(overlay_img, use_container_width=True)

        st.markdown("---")
        st.subheader("Class probabilities")

        pairs = probs_table(probs)
        table_data = {
            "Class": [PRETTY_NAMES.get(p[0], p[0]) for p in pairs],
            "Probability": [round(p[1], 4) for p in pairs],
        }
        st.table(table_data)
        st.bar_chart({PRETTY_NAMES.get(p[0], p[0]): p[1] for p in pairs})

    with tab2:
        st.markdown("### 3) Explanation")
        st.write(explanation)

        with st.expander("Heatmap features (debug)"):
            st.json(feats)

        with st.expander("Confidence details (debug)"):
            st.json(conf)

    with tab3:
        st.markdown(
            """
            ### About this demo
            - **Model:** CNN classifier (ResNet18) trained on MRI images for 4 classes.
            - **Explainability:** Grad-CAM highlights regions that influenced the predicted class.
            - **Confidence:** Rule-based confidence derived from the top-1 probability and margin.

            ### Limitations
            - Performance depends on training data and preprocessing.
            - Grad-CAM is not a medical explanation; it is a visualization of model attention.
            - Outputs may be incorrect; do not use clinically.
            """
        )

    st.markdown("---")
    st.warning(
        "**Important:** This is an automated prediction and attention visualization. "
        "Not for clinical use."
    )


if __name__ == "__main__":
    main()
