# src/app/app.py

import os
import sys
import json
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# ✅ Streamlit Cloud fix: make repo root importable (so "src" works)
# ---------------------------------------------------------
sys.path.insert(0, os.path.abspath("."))

# ---------------------------------------------------------
# ✅ Try real pipeline imports (your real file names)
# ---------------------------------------------------------
REAL_PIPELINE = False
IMPORT_ERRORS: List[str] = []

try:
    from src.training.infer_05 import load_model, predict
except Exception as e:
    IMPORT_ERRORS.append(f"Inference import failed: {e}")

try:
    from src.explain.gradcam_06 import gradcam
except Exception as e:
    IMPORT_ERRORS.append(f"Grad-CAM import failed: {e}")

try:
    from src.explain.heatmap_features import extract_features
except Exception as e:
    IMPORT_ERRORS.append(f"Heatmap features import failed: {e}")

try:
    from src.explain.text_agent import explain
except Exception as e:
    IMPORT_ERRORS.append(f"Text agent import failed: {e}")

REAL_PIPELINE = (len(IMPORT_ERRORS) == 0)


# ---------------------------------------------------------
# Helper: consistent probs dict + label mapping
# ---------------------------------------------------------
def _to_probs_dict(probs: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=np.float32).ravel()
    if len(probs) != len(class_names):
        # fallback labels if mismatch
        class_names = [f"class_{i}" for i in range(len(probs))]
    return {class_names[i]: float(probs[i]) for i in range(len(probs))}


def _topk(probs_dict: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:k]


def _confidence_from_probs(probs: np.ndarray) -> Tuple[str, float, float]:
    p = np.asarray(probs, dtype=np.float32).ravel()
    if p.size < 2:
        return ("Low", float(p.max()) if p.size else 0.0, 0.0)
    top2 = np.sort(p)[-2:]
    p_top1 = float(top2[-1])
    p_top2 = float(top2[-2])
    margin = float(p_top1 - p_top2)
    if p_top1 >= 0.80 and margin >= 0.20:
        return ("High", p_top1, margin)
    if p_top1 >= 0.60:
        return ("Medium", p_top1, margin)
    return ("Low", p_top1, margin)


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Brain Tumor XAI Assistant", layout="wide")

st.title("🧠 Explainable Brain Tumor Detection Assistant")
st.caption("Upload MRI → predict + Grad-CAM overlay + explanation text.")

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

pil_img = Image.open(uploaded).convert("RGB")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Input")
    st.image(pil_img, use_container_width=True)

analyze = st.button("🔍 Analyze image", type="primary")
if not analyze:
    st.stop()

use_real = REAL_PIPELINE and (not force_mock)

st.write(
    f"Pipeline status → REAL_PIPELINE={REAL_PIPELINE} | use_real={use_real}"
)

with st.expander("Import diagnostics", expanded=not REAL_PIPELINE):
    if REAL_PIPELINE:
        st.success("All imports OK ✅")
    else:
        for msg in IMPORT_ERRORS:
            st.error(msg)

# ---------------------------------------------------------
# Run pipeline (REAL or MOCK)
# ---------------------------------------------------------
# Default mock outputs (so UI never crashes)
mock_class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
mock_probs = np.array([0.10, 0.08, 0.05, 0.77], dtype=np.float32)
mock_pred_id = int(np.argmax(mock_probs))
mock_pred_label = mock_class_names[mock_pred_id]

probs = mock_probs
pred_label = mock_pred_label
pred_id = mock_pred_id
class_names = mock_class_names
heatmap = np.zeros((224, 224), dtype=np.float32)
overlay_pil = pil_img.copy()
explanation_text = "Using MOCK mode."

with st.spinner("Running analysis..."):
    if use_real:
        try:
            # 1) load model (cached so it doesn't reload every time)
            @st.cache_resource
            def _cached_model():
                return load_model()

            model = _cached_model()

            # 2) predict
            # Expect: probs, pred_label, pred_id OR probs, pred_label (handle both)
            pred_out = predict(pil_img)

            if isinstance(pred_out, tuple) and len(pred_out) == 3:
                probs, pred_label, pred_id = pred_out
            elif isinstance(pred_out, tuple) and len(pred_out) == 2:
                probs, pred_label = pred_out
                probs = np.asarray(probs, dtype=np.float32)
                pred_id = int(np.argmax(probs))
            else:
                raise ValueError("predict() must return (probs, pred_label) or (probs, pred_label, pred_id)")

            probs = np.asarray(probs, dtype=np.float32).ravel()

            # 3) infer class_names if your predict doesn't provide them:
            # If your repo has a saved mapping file, you can load it here.
            # For now, use defaults if lengths match.
            if probs.size == len(mock_class_names):
                class_names = mock_class_names
            else:
                class_names = [f"class_{i}" for i in range(probs.size)]

            # 4) Grad-CAM: try common signatures
            try:
                heatmap, overlay_pil = gradcam(pil_img, model=model, target_class=pred_id)
            except TypeError:
                # maybe gradcam(pil_img, model, pred_id)
                heatmap, overlay_pil = gradcam(pil_img, model, pred_id)

            # 5) features + explanation
            feats = extract_features(heatmap)

            # Some text_agent versions accept class_names; try both
            try:
                explanation_text = explain(pred_label, probs, feats, class_names=class_names)
            except TypeError:
                probs_dict = _to_probs_dict(probs, class_names)
                explanation_text = explain(pred_label, probs_dict, feats)

        except Exception as e:
            st.warning("Real pipeline failed — falling back to MOCK outputs.")
            st.exception(e)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
probs_dict = _to_probs_dict(probs, class_names)
conf_label, top1_prob, margin = _confidence_from_probs(probs)

with col2:
    st.subheader("2) Results")
    st.markdown(f"**Predicted class:** `{pred_label}`")
    st.markdown(f"**Top-1 probability:** `{top1_prob:.3f}`")
    st.markdown(f"**Confidence:** `{conf_label}`")

    ordered = _topk(probs_dict, top_k)

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

with col4:
    st.subheader("4) Explanation")
    st.markdown(explanation_text)

if show_heatmap_raw:
    st.subheader("Heatmap (raw)")
    st.image(np.asarray(heatmap, dtype=np.float32), clamp=True)

st.divider()

# ---------------------------------------------------------
# Download report
# ---------------------------------------------------------
st.subheader("📄 Download report")

report = f"""Brain Tumor XAI Assistant Report

Predicted class: {pred_label}
Top-1 probability: {top1_prob:.4f}
Confidence: {conf_label}
Margin (top1 - top2): {margin:.4f}

Probabilities:
{json.dumps(probs_dict, indent=2)}

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
