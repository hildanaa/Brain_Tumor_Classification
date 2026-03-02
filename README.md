# Brain Tumor MRI Classifier with Explainable AI

## Overview

This project implements a deep learning pipeline for brain tumor classification from MRI images using transfer learning and Grad-CAM for visual explainability.

The system includes:

- A multi-class MRI classifier (glioma, meningioma, pituitary tumor, no tumor)
- Grad-CAM visual explanations to highlight model attention
- A rule-based explanation agent that generates text grounded in model evidence
- An interactive Streamlit web interface for image upload and analysis
## How to Run (Development Setup)

1. Clone the repository:


git clone https://github.com/hildanaa/Brain_Tumor_Classification.git

cd brain-tumor-assistant


2. Create a virtual environment:


python -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows


3. Install dependencies:


pip install -r requirements.txt


4. Run the Streamlit app:


streamlit run src/app/app.py


---

## Dataset

This project uses a publicly available Brain Tumor MRI dataset for educational purposes.

The dataset is not included in the repository and must be downloaded separately.

---

## Medical Disclaimer

This project is intended strictly for educational and research purposes.

It is NOT a medical diagnostic tool and must not be used for clinical decision-making.

Model predictions and visual explanations are experimental and should not replace professional medical evaluation.

---

## Model Performance Summary

The current baseline model is a ResNet18 trained with transfer learning.

**Final Test Performance:**

- Test Accuracy: ~97.5%
- Test Macro-F1: ~97.4%
- Macro AUC (One-vs-Rest): ~0.99

Training curves show stable convergence with minimal overfitting.

---

## Limitations

While the model achieves strong performance on the current dataset, several important limitations must be acknowledged:

1. **Public Dataset**  
   The MRI dataset used in this project is publicly available and curated for academic purposes. It may not reflect the variability and complexity of real-world clinical data.

2. **No Patient-Level Split**  
   The dataset is split at the image level rather than at the patient level. If multiple slices from the same patient are present across splits, this may inflate performance metrics.

3. **No External Validation**  
   The model has not been evaluated on an independent external dataset from a different source or hospital.

4. **Not a Clinical Diagnostic Tool**  
   This project is strictly for educational and research demonstration purposes. It must not be used for medical diagnosis or clinical decision-making.

---

## Next Steps

- Improve Grad-CAM localization quality
- Explore Grad-CAM++ or smoothing techniques
- Investigate patient-level splitting strategies
- Integrate the model into a Streamlit-based explainable interface

---

## Authors

- Fernanda Hernandez
- Paula Cáceres
- Ivette Benavides
- Regina Villaseñor
- Hildana Aklilu
