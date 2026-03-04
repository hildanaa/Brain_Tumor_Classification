import sys
import os
sys.path.append(os.path.abspath("."))

import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from src.explain.mask_09 import brain_mask_from_pil, apply_mask
from src.models.model_factory_03 import build_resnet18
from src.explain.gradcam_06 import GradCAM
from src.explain.overlay_07 import overlay_heatmap_on_pil
from src.explain.heatmap_features import extract_features
from src.explain.text_agent import explain
from src.explain.confidence import confidence_from_probs


# -------- Labels --------
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
PRETTY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary tumor",
    "notumor": "No tumor",
}


# -------- Streamlit compatibility helpers --------
def st_image(img, **kwargs):
    """
    Streamlit compatibility:
    - Newer versions: use_container_width
    - Older versions: use_column_width
    """
    if "use_container_width" in kwargs:
        try:
            return st.image(img, **kwargs)
        except TypeError:
            v = kwargs.pop("use_container_width")
            kwargs["use_column_width"] = v
            return st.image(img, **kwargs)
    return st.image(img, **kwargs)


def st_dataframe(obj, **kwargs):
    """
    Some Streamlit versions don't support hide_index, so we fallback.
    """
    try:
        return st.dataframe(obj, **kwargs)
    except TypeError:
        kwargs.pop("hide_index", None)
        return st.dataframe(obj, **kwargs)


# -------- Transforms --------
def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# -------- Load model --------
@st.cache_resource
def load_model_and_cam(checkpoint_path: str, device: str):
    model, _ = build_resnet18(num_classes=len(CLASS_NAMES), pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # ✅ Force layer4 for best Grad-CAM++
    target_layer = model.layer4
    cam = GradCAM(model, target_layer)

    return model, cam


# -------- Prediction --------
def predict_probs(model, pil_img: Image.Image, device: str):
    tfm = build_eval_transform(224)
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_id]
    return x, probs, pred_id, pred_label


# -------- Confidence badge --------
def badge_confidence(level: str) -> str:
    if level == "High":
        return "🟢 High"
    if level == "Medium":
        return "🟠 Medium"
    return "🔴 Low"


# -------- PDF report --------
def _pil_to_png_bytes(pil_img: Image.Image, max_w: int = 900) -> bytes:
    img = pil_img.convert("RGB")
    if img.width > max_w:
        new_h = int(max_w * img.height / img.width)
        img = img.resize((max_w, new_h))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_pdf_report(
    banner_path: Optional[str],
    input_img: Image.Image,
    overlay_img: Image.Image,
    pretty_pred: str,
    conf: dict,
    df_probs: pd.DataFrame,
    explanation_text: str,
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    y = H - 50

    # Banner (optional)
    if banner_path and os.path.exists(banner_path):
        try:
            banner = Image.open(banner_path).convert("RGB")
            banner_bytes = _pil_to_png_bytes(banner, max_w=1100)
            banner_reader = ImageReader(io.BytesIO(banner_bytes))
            c.drawImage(
                banner_reader, 40, y - 120,
                width=W - 80, height=110,
                preserveAspectRatio=True, mask='auto'
            )
            y -= 140
        except Exception:
            pass

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Brain Tumor MRI Classifier - Report")
    y -= 22

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Prediction: {pretty_pred}")
    y -= 16
    c.drawString(40, y, f"Confidence: {conf.get('level','')} (p_top1={conf.get('p_top1',0):.3f}, margin={conf.get('margin',0):.3f})")
    y -= 22

    # Images (Input + Overlay)
    input_bytes = _pil_to_png_bytes(input_img, max_w=700)
    overlay_bytes = _pil_to_png_bytes(overlay_img, max_w=700)
    input_reader = ImageReader(io.BytesIO(input_bytes))
    overlay_reader = ImageReader(io.BytesIO(overlay_bytes))

    img_w = (W - 100) / 2
    img_h = 220

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Input image")
    c.drawString(40 + img_w + 20, y, "Grad-CAM overlay")
    y -= 10

    c.drawImage(input_reader, 40, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
    c.drawImage(overlay_reader, 40 + img_w + 20, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
    y -= img_h + 20

    # Probabilities table (text)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Class probabilities")
    y -= 16

    c.setFont("Helvetica", 10)
    for _, row in df_probs.iterrows():
        line = f"- {row['Class']}: {row['Probability']:.4f}"
        c.drawString(50, y, line)
        y -= 14
        if y < 120:
            c.showPage()
            y = H - 60
            c.setFont("Helvetica", 10)

    y -= 6

    # Explanation
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Explanation")
    y -= 16

    c.setFont("Helvetica", 10)
    # Basic wrapping
    max_chars = 110
    text = explanation_text.replace("\n", " ").strip()
    lines = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        cut = cut if cut != -1 else max_chars
        lines.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        lines.append(text)

    for line in lines:
        c.drawString(40, y, line)
        y -= 13
        if y < 120:
            c.showPage()
            y = H - 60
            c.setFont("Helvetica", 10)

    # Disclaimer
    if y < 120:
        c.showPage()
        y = H - 60

    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Disclaimer")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(
        40,
        y,
        "This is a research/educational demo and is NOT a medical device. Do not use for clinical decisions.",
    )

    c.showPage()
    c.save()
    return buf.getvalue()


# ==========================
#            MAIN
# ==========================
def main():
    st.set_page_config(page_title="Brain Tumor MRI - Demo", layout="wide")

    banner_path = "assests/banner_MRI.jpg"
    if os.path.exists(banner_path):
        st_image(banner_path, use_container_width=True)

    st.markdown(
        """
        <div style="padding: 0.5rem 0 0.2rem 0;">
            <h1 style="margin-bottom: 0.2rem;">🧠 Brain Tumor MRI Classifier</h1>
            <p style="margin-top: 0; opacity: 0.8;">
                Upload an MRI image and we’ll show the predicted class,
                highlight the regions that influenced it (Grad-CAM),
                and provide a short explanation.
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
            "This is a research/educational demo and is NOT a medical device. "
            "Do not use for diagnosis or treatment decisions."
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

        # ✅ Skip overlay for No Tumor
        if pred_label == "notumor":
            overlay_img = pil_img  # keep valid image for UI + PDF
            heatmap_np = None
            feats = {"note": "No heatmap generated because prediction is 'No tumor'."}
        else:
            # ✅ Grad-CAM++ (layer4 already fixed in load_model_and_cam)
            heatmap_t = cam.generate_pp(x, class_idx=pred_id)

            # ✅ Mask outside brain (use 224x224 to match heatmap)
            pil_224 = pil_img.resize((224, 224))
            mask = brain_mask_from_pil(pil_224, size=(224, 224))
            heatmap_t = apply_mask(heatmap_t, mask)

            overlay_img = overlay_heatmap_on_pil(pil_224, heatmap_t, alpha=alpha)
            heatmap_np = heatmap_t.detach().cpu().numpy()

            feats = extract_features(heatmap_np, threshold=threshold, compute_regions=True)

        explanation = explain(
            pred_label=pred_label,
            probs=probs,
            heatmap_features=feats,
            class_names=CLASS_NAMES,
        )

        conf = confidence_from_probs(probs)

    pretty_pred = PRETTY_NAMES.get(pred_label, pred_label)

    # Dataframe probs sorted
    pairs = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)
    df_probs = pd.DataFrame({
        "Class": [PRETTY_NAMES.get(p[0], p[0]) for p in pairs],
        "Probability": [float(p[1]) for p in pairs],
    })

    # --- PDF bytes (build once per run)
    pdf_bytes = build_pdf_report(
        banner_path=banner_path if os.path.exists(banner_path) else None,
        input_img=pil_img,
        overlay_img=overlay_img,
        pretty_pred=pretty_pred,
        conf=conf,
        df_probs=df_probs,
        explanation_text=str(explanation),
    )

    # --- Tabs
    tab1, tab2, tab3 = st.tabs(["📌 Results", "🧠 Explanation", "ℹ️ About"])

    with tab1:
        st.markdown("### 2) Results")

        st.download_button(
            label="⬇️ Download report (PDF)",
            data=pdf_bytes,
            file_name="brain_tumor_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", pretty_pred)
        m2.metric("Top probability", f"{conf['p_top1']:.3f}")
        m3.metric("Confidence", badge_confidence(conf["level"]))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Input")
            st_image(pil_img, use_container_width=True)

        with c2:
            st.subheader("Grad-CAM overlay")
            if pred_label == "notumor":
                st.info("No tumor detected. Grad-CAM overlay is not generated for 'No tumor' predictions.")
                st_image(pil_img, use_container_width=True)
            else:
                st_image(overlay_img, use_container_width=True)

        st.markdown("---")
        st.subheader("Class probabilities")

        top_class = df_probs.iloc[0]["Class"]
        styled = (
            df_probs.style
              .format({"Probability": "{:.2%}"})
              .bar(subset=["Probability"], align="left")
              .apply(lambda r: ["font-weight: 700;" if r["Class"] == top_class else "" for _ in r], axis=1)
        )
        st_dataframe(styled, use_container_width=True, hide_index=True)

        labels = df_probs["Class"].tolist()
        values = df_probs["Probability"].tolist()

        fig, ax = plt.subplots()
        y = np.arange(len(labels))

        ax.barh(y, values, height=0.95, color="#1f6f8b")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.grid(axis="x", alpha=0.25)

        st.pyplot(fig, clear_figure=True)

    with tab2:
        st.markdown("### 3) Explanation")
        st.write(explanation)

        with st.expander("Heatmap features"):
            st.json(feats)

        with st.expander("Confidence details"):
            st.json(conf)

    with tab3:
        st.markdown(
            """
            ### About this demo
            - **Model:** ResNet18 CNN trained on MRI images (4 classes).
            - **Explainability:** Grad-CAM highlights regions that influenced the prediction.
            - **Confidence:** Rule-based confidence from top-1 probability and margin.

            ### Limitations
            - Performance depends on training data and preprocessing.
            - Grad-CAM is not a medical explanation; it is model attention visualization.
            - Not intended for clinical use.
            """
        )

    st.markdown("---")
    st.warning("Important: This is an automated prediction and attention visualization. Not for clinical use.")


if __name__ == "__main__":
    main()