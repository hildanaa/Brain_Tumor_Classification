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

## Authors

- Fernanda Hernandez
- Paula Cáceres
- Ivette Benavides
- Regina Villaseñor
- Hildana Aklilu
