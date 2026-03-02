# src/app/copy.py

APP_TITLE = "Explainable Brain Tumor Detection Assistant"
TAGLINE = "Upload an MRI image to get a model prediction plus a visual + text explanation."

UPLOAD_LABEL = "Upload a brain MRI image (PNG/JPG)"
PRIMARY_CTA = "Analyze image"

RESULTS_HEADER = "Results"
EXPLANATION_HEADER = "Explanation"
HEATMAP_HEADER = "Model attention (Grad-CAM)"

CONFIDENCE_LABEL = "Confidence"
PROBABILITIES_LABEL = "Prediction probabilities"

DOWNLOAD_REPORT_LABEL = "Download report"

DISCLAIMER_SHORT = (
    "⚠️ Educational/demo tool only — not medical advice and not a diagnosis."
)

DISCLAIMER_LONG = (
    "Important: This system does not provide a medical diagnosis. "
    "It generates a prediction from an image using a machine learning model and highlights regions that influenced the prediction. "
    "If you have health concerns, consult a qualified medical professional."
)

PRIVACY_NOTE = (
    "Privacy note: If you deploy this publicly, avoid storing uploaded images unless users explicitly consent."
)

ERROR_IMAGE = "Could not read the uploaded file as an image. Please upload a valid PNG/JPG."
ERROR_MODEL = "Model is not available. Please check that the checkpoint is loaded correctly."