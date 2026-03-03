# src/app/app.py

import os
import sys
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# ✅ Streamlit Cloud fix: make repo root importable (so "src" works)
# ---------------------------------------------------------
sys.path.insert(0, os.path.abspath("."))

# ---------------------------------------------------------
# ✅ Real pipeline imports (YOUR real file names)
# ---------------------------------------------------------
REAL_PIPELINE = True
IMPORT_ERRORS: List[str] = []

try:
    from src.training.infer_05 import load_model, predict
except Exception as e:
    REAL_PIPELINE = False
    IMPORT_ERRORS.append(f"Import failed: src.training.infer_05 → {e}")

try:
    from src.explain.gradcam_06 import gradcam
except Exception as e:
    REAL_PIPELINE = False
    IMPORT_ERRORS.append(f"Import failed: src.explain.gradcam_06 → {e}")

try:
    from src.explain.heatmap_features import extract_features
except Exception as e:
    REAL_PIPELINE = False
    IMPORT_ERRORS.append(f"Import failed: src.explain.heatmap_features → {e}")

try:
    from src.explain.text_agent import explain
except Exception as e:
    REAL_PIPELINE = False
    IMPORT_ERRORS.append(f"Import failed: src.explain.text_agent → {e}")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def probs_to_dict(probs: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    p = np.asarray(probs, dtype=np.float32).ravel()
    if len(class_names) != len(p):
        class_names = [f"class_{i}" for i in range(len(p))]
    return {class_names[i]: float(p[i]) for i in range(len(p))}

def topk_items(d: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

def confidence_from_probs(probs: np.ndarray) -> Tuple[str, float, float]:
    p = np.asarray(probs, dtype=np.float32).ravel()
    if p.size == 0:
        return ("Low", 0.0, 0.0)
    if p.size == 1:
        return ("Low", float(p[0]), 0.0)
    s = np.sort(p)
    p_top1 = float(s[-1])
    p_top2 = float(s[-2])
    margin = float(p_top1 - p_top2)
    if p_top1 >= 0.80 and margin >= 0.20:
        return ("High", p_top1, margin)
    if p_top1 >= 0.60:
        return ("Medium", p_top1, margin)
    return ("Low", p_top1, margin)

@st.cache_resource(show_spinner=False)
def get_cached_model():
    """
    Loads model ONCE per app session.
    Your load_model() should return (model, device) according to the updated infer_05.py I gave you.
    """
    return load_model()

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Brain Tumor XAI Assistant", layout="wide")

st.title("🧠 Explainable Brain Tumor Detection Assistant")
st.caption("Upload MRI → prediction + Grad-CAM overlay + explanation text.")

st.markdown(
    "**Medical Disclaimer:** This tool is for educational/research purposes only and is not a medical diagnostic device. "
    "Always consult a qualified healthcare professional for medical decisions."
)

with st.expander("⚙️ Settings", expanded=False):
    force_mock = st.checkbox("Force MOCK mode", value=False)
    show_heatmap_raw = st.checkbox("Show raw heatmap (debug)", value=False)
    top_k = st.slider("Top-k probabilities to display", 2, 6, 4)

uploaded = st.file_uploader("Upload MRI image (png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

try:
    pil_img = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("Could not read the uploaded file as an image.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Input")
    st.image(pil_img, use_container_width=True)

analyze = st.button("🔍 Analyze image", type="primary")
if not analyze:
    st.stop()

use_real = REAL_PIPELINE and (not force_mock)

st.write(f"Pipeline status → REAL_PIPELINE={REAL_PIPELINE} | use_real={use_real}")

with st.expander("Import diagnostics", expanded=not REAL_PIPELINE):
    if REAL_PIPELINE:
        st.success("All imports OK ✅")
    else:
        for msg in IMPORT_ERRORS:
            st.error(msg)

# ---------------------------------------------------------
# Defaults (MOCK)
# ---------------------------------------------------------
class_names_default = ["glioma", "meningioma", "pituitary", "no_tumor"]
probs = np.array([0.10, 0.08, 0.05, 0.77], dtype=np.float32)
pred_id = int(np.argmax(probs))
pred_label = class_names_default[pred_id]

heatmap = np.zeros((224, 224), dtype=np.float32)
overlay_pil = pil_img.copy()
features: Dict[str, Any] = {
    "focus_ratio": 0.0,
    "peak_intensity": 0.0,
    "mean_intensity": 0.0,
    "center_of_mass_xy": [0.5, 0.5],
    "spread": 1.0,
    "num_regions": None,
    "threshold": 0.6,
}
explanation_text = "MOCK mode active."

# ---------------------------------------------------------
# RUN REAL PIPELINE
# ---------------------------------------------------------
with st.spinner("Running analysis..."):
    if use_real:
        try:
            # 1) Load model (once)
            model, device = get_cached_model()

            # 2) Predict
            # Expect: (probs, pred_label, pred_id) from updated infer_05.py
            probs, pred_label, pred_id = predict(pil_img)

            probs = np.asarray(probs, dtype=np.float32).ravel()

            # Try to infer class_names based on probabilities length
            if probs.size == 4:
                class_names = class_names_default
            else:
                class_names = [f"class_{i}" for i in range(probs.size)]

            # ✅ Marker to confirm REAL model ran
            st.success(f"Using REAL model ✅  pred={pred_label}  id={pred_id}")

            # 3) Grad-CAM
            # Most common signature:
            # gradcam(pil_img, model=model, target_class=pred_id) -> (heatmap, overlay)
            try:
                heatmap, overlay = gradcam(pil_img, model=model, target_class=pred_id)
            except TypeError:
                # fallback signature: gradcam(pil_img, model, pred_id)
                heatmap, overlay = gradcam(pil_img, model, pred_id)

            heatmap = np.asarray(heatmap, dtype=np.float32)

            if isinstance(overlay, Image.Image):
                overlay_pil = overlay
            else:
                overlay_pil = Image.fromarray(overlay)

            # 4) Heatmap features
            features = extract_features(heatmap)

            # 5) Explain
            try:
                explanation_text = explain(pred_label, probs, features, class_names=class_names)
            except TypeError:
                probs_dict = probs_to_dict(probs, class_names)
                explanation_text = explain(pred_label, probs_dict, features)

        except Exception as e:
            st.warning("Real pipeline failed — falling back to MOCK outputs.")
            st.exception(e)

# ---------------------------------------------------------
# RESULTS UI
# ---------------------------------------------------------
probs_dict = probs_to_dict(probs, class_names_default if len(probs) == 4 else [f"class_{i}" for i in range(len(probs))])
conf_label, top1_prob, margin = confidence_from_probs(probs)

with col2:
    st.subheader("2) Results")
    st.markdown(f"**Predicted class:** `{pred_label}`")
    st.markdown(f"**Top-1 probability:** `{top1_prob:.3f}`")
    st.markdown(f"**Confidence:** `{conf_label}`")

    ordered = topk_items(probs_dict, top_k)

    st.write("**Prediction probabilities (table)**")
    st.dataframe(
        [{"class": k, "prob": float(v)} for k, v in ordered],
        use_container_width=True
    )

    st.write("**Prediction probabilities (bar chart)**")
    st.bar_chart({k: float(v) for k, v in ordered})

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("3) Model attention (Grad-CAM)")
    st.image(overlay_pil, use_container_width=True)

    if show_heatmap_raw:
        st.caption("Heatmap (raw)")
        st.image(heatmap, clamp=True, use_container_width=True)

with col4:
    st.subheader("4) Explanation")
    st.markdown(explanation_text)

st.divider()

# ---------------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------------
st.subheader("📄 Download report")

report = f"""Brain Tumor XAI Assistant Report

Predicted class: {pred_label}
Top-1 probability: {top1_prob:.4f}
Confidence: {conf_label}
Margin (top1 - top2): {margin:.4f}

Probabilities:
{json.dumps(probs_dict, indent=2)}

Heatmap features:
{json.dumps(features, indent=2)}

Explanation:
{explanation_text}

Disclaimer:
Educational/research only. Not medical advice or diagnosis.
"""

st.download_button(
    "Download report (TXT)",
    data=report,
    file_name="report.txt",
    mime="text/plain",
)
