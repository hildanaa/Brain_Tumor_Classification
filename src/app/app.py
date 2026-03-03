import json
import os
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import streamlit as st
from PIL import Image

# ----------------------------
# UI COPY (Role 4)
# ----------------------------
try:
    from src.app.copy import (
        APP_TITLE,
        TAGLINE,
        UPLOAD_LABEL,
        PRIMARY_CTA,
        RESULTS_HEADER,
        EXPLANATION_HEADER,
        HEATMAP_HEADER,
        CONFIDENCE_LABEL,
        PROBABILITIES_LABEL,
        DOWNLOAD_REPORT_LABEL,
        DISCLAIMER_SHORT,
        DISCLAIMER_LONG,
        PRIVACY_NOTE,
        ERROR_IMAGE,
        ERROR_MODEL,
    )
except Exception:
    # Fallback strings if copy.py isn't present yet
    APP_TITLE = "Explainable Brain Tumor Detection Assistant"
    TAGLINE = "Upload an MRI image → prediction + Grad-CAM overlay + explanation."
    UPLOAD_LABEL = "Upload MRI image (PNG/JPG)"
    PRIMARY_CTA = "Analyze"
    RESULTS_HEADER = "Results"
    EXPLANATION_HEADER = "Explanation"
    HEATMAP_HEADER = "Model attention (Grad-CAM)"
    CONFIDENCE_LABEL = "Confidence"
    PROBABILITIES_LABEL = "Prediction probabilities"
    DOWNLOAD_REPORT_LABEL = "Download report"
    DISCLAIMER_SHORT = "⚠️ Educational/demo tool only — not medical advice and not a diagnosis."
    DISCLAIMER_LONG = DISCLAIMER_SHORT
    PRIVACY_NOTE = ""
    ERROR_IMAGE = "Could not read the uploaded file as an image."
    ERROR_MODEL = "Model is not available."


# ----------------------------
# PIPELINE IMPORTS (Roles 2–4)
# ----------------------------
REAL_PIPELINE = True
IMPORT_ERRORS: List[str] = []

# Inference (Role 2)
try:
    # Expected contract: predict(image) -> probs (np array), pred_label (str), pred_id (int)
    # Some teams may name it predict_pil; we support both.
    from src.models.infer import predict as predict_image  # type: ignore
except Exception as e:
    REAL_PIPELINE = False
    IMPORT_ERRORS.append(f"Inference import failed: {e}")
    predict_image = None  # type: ignore

# Grad-CAM (Role 3)
try:
    # Expected contract: gradcam(image, model, target_class=None) -> heatmap, overlay
    from src.explain.gradcam import gradcam  # type: ignore
except Exception as e:
    REAL_PIPELINE = False
    IMPORT_ERRORS.append(f"Grad-CAM import failed: {e}")
    gradcam = None  # type: ignore

# Heatmap features (Role 4)
try:
    from src.explain.heatmap_features import extract_features  # type: ignore
except Exception as e:
    REAL_PIPELINE = False
    IMPORT_ERRORS.append(f"Heatmap features import failed: {e}")
    extract_features = None  # type: ignore

# Text agent (Role 4)
try:
    from src.explain.text_agent import explain  # type: ignore
except Exception as e:
    REAL_PIPELINE = False
    IMPORT_ERRORS.append(f"Text agent import failed: {e}")
    explain = None  # type: ignore

# Confidence helper (Role 4)
try:
    from src.explain.confidence import confidence_from_probs  # type: ignore
except Exception:
    confidence_from_probs = None  # type: ignore


# ----------------------------
# OPTIONAL: Try to load class names if you store them
# (Role 2 common practice: outputs/checkpoints/class_to_idx.json)
# ----------------------------
def load_class_names() -> Optional[List[str]]:
    candidates = [
        os.path.join("outputs", "checkpoints", "class_to_idx.json"),
        os.path.join("outputs", "checkpoints", "classes.json"),
        os.path.join("outputs", "metrics", "class_to_idx.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                # class_to_idx: {"glioma":0,...} -> invert
                if isinstance(obj, dict) and all(isinstance(v, int) for v in obj.values()):
                    inv = {v: k for k, v in obj.items()}
                    return [inv[i] for i in range(len(inv))]
                # classes: ["glioma", ...]
                if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                    return obj
            except Exception:
                pass
    return None


CLASS_NAMES = load_class_names()


# ----------------------------
# MODEL ACCESS FOR GRADCAM
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model_for_gradcam():
    """
    Grad-CAM needs the model object. Different teams implement this differently.
    We try common patterns:
      - src.models.infer has load_model()
      - src.models.infer exposes a cached model (MODEL or model)
    If none exist, we return None and run in MOCK mode.
    """
    try:
        import src.models.infer as infer_mod  # type: ignore

        if hasattr(infer_mod, "load_model") and callable(infer_mod.load_model):
            return infer_mod.load_model()

        if hasattr(infer_mod, "MODEL"):
            return infer_mod.MODEL

        if hasattr(infer_mod, "model"):
            return infer_mod.model

    except Exception:
        return None

    return None


def probs_to_dict(probs: np.ndarray, class_names: Optional[List[str]]) -> Dict[str, float]:
    p = np.asarray(probs, dtype=float).ravel()
    if class_names and len(class_names) == len(p):
        return {class_names[i]: float(p[i]) for i in range(len(p))}
    # fallback labels
    return {f"class_{i}": float(p[i]) for i in range(len(p))}


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🧠 {APP_TITLE}")
st.caption(TAGLINE)

st.markdown(DISCLAIMER_SHORT)
if DISCLAIMER_LONG:
    with st.expander("Read full disclaimer", expanded=False):
        st.write(DISCLAIMER_LONG)
        if PRIVACY_NOTE:
            st.write(PRIVACY_NOTE)

with st.expander("⚙️ Settings", expanded=False):
    force_mock = st.checkbox("Force MOCK mode", value=False)
    show_heatmap_raw = st.checkbox("Show raw heatmap (debug)", value=False)
    top_k = st.slider("Top-k probabilities to display", 2, 6, 4)

uploaded = st.file_uploader(UPLOAD_LABEL, type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

try:
    pil_img = Image.open(uploaded).convert("RGB")
except Exception:
    st.error(ERROR_IMAGE)
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Input")
    st.image(pil_img, use_container_width=True)

analyze = st.button(f"🔍 {PRIMARY_CTA}", type="primary")
if not analyze:
    st.stop()

use_real = (REAL_PIPELINE and (not force_mock))

st.write(
    "Pipeline status →",
    f"REAL_PIPELINE={REAL_PIPELINE}",
    f"| use_real={use_real}",
    f"| class_names={'yes' if CLASS_NAMES else 'no'}"
)
if IMPORT_ERRORS:
    with st.expander("Import diagnostics", expanded=False):
        for msg in IMPORT_ERRORS:
            st.code(msg)


# ----------------------------
# RUN PIPELINE
# ----------------------------
with st.spinner("Running analysis..."):

    if use_real and predict_image is not None:
        # 1) Inference
        # Expected: probs (np array), pred_label (str), pred_id (int)
        out = predict_image(pil_img)

        # Support both (probs, pred_label, pred_id) and (probs_dict, pred_label)
        pred_id = None
        if isinstance(out, tuple) and len(out) == 3:
            probs, pred_label, pred_id = out
            probs = np.asarray(probs, dtype=float)
            probs_dict = probs_to_dict(probs, CLASS_NAMES)
        elif isinstance(out, tuple) and len(out) == 2:
            probs_or_dict, pred_label = out
            if isinstance(probs_or_dict, dict):
                probs_dict = {k: float(v) for k, v in probs_or_dict.items()}
                probs = np.array(list(probs_dict.values()), dtype=float)
                if CLASS_NAMES is None:
                    CLASS_NAMES = list(probs_dict.keys())  # type: ignore
            else:
                probs = np.asarray(probs_or_dict, dtype=float)
                probs_dict = probs_to_dict(probs, CLASS_NAMES)
        else:
            # Unknown output format -> fallback mock
            use_real = False

    if not use_real:
        # MOCK fallback
        probs_dict = {
            "glioma": 0.10,
            "meningioma": 0.08,
            "pituitary": 0.05,
            "no_tumor": 0.77,
        }
        pred_label = max(probs_dict, key=probs_dict.get)
        probs = np.array([probs_dict[k] for k in probs_dict.keys()], dtype=float)
        pred_id = None
        overlay_pil = pil_img
        heatmap = np.zeros((224, 224), dtype=np.float32)

    # 2) Grad-CAM (needs model)
    if use_real and gradcam is not None:
        model = load_model_for_gradcam()
        if model is None:
            st.warning("Could not load model object for Grad-CAM. Showing MOCK overlay.")
            overlay_pil = pil_img
            heatmap = np.zeros((224, 224), dtype=np.float32)
        else:
            # target_class: prefer pred_id if available, else None
            hm, overlay = gradcam(pil_img, model, target_class=pred_id)
            heatmap = np.asarray(hm, dtype=np.float32)
            overlay_pil = overlay if isinstance(overlay, Image.Image) else Image.fromarray(overlay)

    # 3) Heatmap features (Role 4)
    if extract_features is not None:
        try:
            feats = extract_features(heatmap)
        except Exception:
            feats = {
                "focus_ratio": float((heatmap > 0.6).mean()) if isinstance(heatmap, np.ndarray) else 0.0,
                "spread": 1.0,
                "num_regions": None,
                "center_of_mass_xy": [0.5, 0.5],
            }
    else:
        feats = {
            "focus_ratio": float((heatmap > 0.6).mean()) if isinstance(heatmap, np.ndarray) else 0.0,
            "spread": 1.0,
            "num_regions": None,
            "center_of_mass_xy": [0.5, 0.5],
        }

    # 4) Text explanation (Role 4)
    if explain is not None:
        explanation_text = explain(pred_label, probs, feats, class_names=CLASS_NAMES)
    else:
        explanation_text = "Text explainer not available."

    # 5) Confidence label (optional)
    conf_level = None
    if confidence_from_probs is not None:
        try:
            conf_level = confidence_from_probs(probs).get("level")
        except Exception:
            conf_level = None


# ----------------------------
# RESULTS UI
# ----------------------------
with col2:
    st.subheader(f"2) {RESULTS_HEADER}")

    top1_prob = float(probs_dict.get(pred_label, np.max(probs) if probs.size else 0.0))
    st.markdown(f"**Predicted class:** `{pred_label}`")
    st.markdown(f"**Top-1 probability:** `{top1_prob:.3f}`")
    if conf_level:
        st.markdown(f"**{CONFIDENCE_LABEL}:** `{conf_level}`")

    ordered = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

    st.write(f"**{PROBABILITIES_LABEL} (table)**")
    st.dataframe(
        [{"class": k, "prob": float(v)} for k, v in ordered],
        use_container_width=True
    )

    st.write(f"**{PROBABILITIES_LABEL} (bar chart)**")
    chart_dict = {k: float(v) for k, v in ordered}
    st.bar_chart(chart_dict)

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader(f"3) {HEATMAP_HEADER}")
    st.image(overlay_pil, use_container_width=True)

    if show_heatmap_raw:
        st.caption("Heatmap (raw)")
        # Streamlit can display 2D arrays as images
        st.image(heatmap, clamp=True, use_container_width=True)

with col4:
    st.subheader(f"4) {EXPLANATION_HEADER}")
    st.markdown(explanation_text)

st.divider()

# ----------------------------
# DOWNLOAD REPORT
# ----------------------------
st.subheader("📄 Download report")

report = f"""\
{APP_TITLE}

Predicted class: {pred_label}
Top-1 probability: {top1_prob:.4f}
Confidence: {conf_level if conf_level else "N/A"}

Probabilities:
{json.dumps(probs_dict, indent=2)}

Heatmap features:
{json.dumps(feats, indent=2)}

Explanation:
{explanation_text}

Disclaimer:
{DISCLAIMER_SHORT}
"""

st.download_button(
    DOWNLOAD_REPORT_LABEL,
    data=report,
    file_name="report.txt",
    mime="text/plain",
)