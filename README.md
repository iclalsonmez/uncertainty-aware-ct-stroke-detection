# Uncertainty-Aware CT Stroke Detection

An explainable and uncertainty-aware AI framework for stroke detection in brain CT using a TrustNet-inspired CNN, Monte Carlo Dropout, Human-in-the-Loop referral, and Grad-CAM.

## Overview

This project presents a safety-oriented medical AI pipeline for binary stroke detection from brain CT slices.  
Instead of producing only a class label, the system is designed to:

- classify CT slices as **normal** or **stroke**
- estimate predictive uncertainty
- refer uncertain cases to a clinician
- provide visual explanations with Grad-CAM

The framework was developed as a proof-of-concept for **trustworthy AI in medical decision support**.

## Key Components

- **TrustNet-inspired CNN** for lightweight feature extraction
- **Monte Carlo Dropout (MCD)** for epistemic uncertainty estimation
- Predictive entropy for uncertainty-aware decision making
- **Human-in-the-Loop (HITL)** referral mechanism
- **Grad-CAM** for explainability and model interpretation

## Dataset

The study uses the **Teknofest 2021 Stroke Dataset**, organized into:

- Inme Yok → normal
- Iskemi → stroke
- Kanama → stroke

### Final binary setting
- Normal
- Stroke = ischemic + hemorrhagic

### Loaded data
- Total slices: 6,774
- Normal: 4,551
- Stroke: 2,223

## Preprocessing

- grayscale input
- resized to 224 × 224
- normalized to [0, 1]
- random 80/20 train-test split

## Methodology

### Model
- TrustNet-inspired CNN
- 5 residual convolutional blocks
- AdamW optimizer
- weighted cross-entropy loss
- 8 training epochs

### Uncertainty-aware inference
- Monte Carlo Dropout
- 20 stochastic forward passes
- predictive entropy
- referral threshold: **H > 0.45**

### Explainability
Grad-CAM was applied to multiple candidate layers:

- `features[-3].conv`
- `features[-2].conv`
- `features[-1].conv`

The intermediate layer `features[-2].conv` provided the most balanced and interpretable visualization.

## Results

### Quantitative performance
- Accuracy: 0.6362
- Precision: 0.4614
- Recall: 0.7821
- F1-score: 0.5804

### Behavioral result
- Accepted predictions: 39
- Referred to clinician: 1,316

These results show that the framework behaved in a **highly conservative and safety-oriented** manner.

## Main Takeaway

This project is not intended as a state-of-the-art clinical diagnostic model.  
Its main value lies in combining:

- prediction
- uncertainty estimation
- referral logic
- explainability

within a single **trustworthy AI pipeline** for stroke imaging support.

## Limitations

- slice-level split instead of patient-level split
- no separate validation set
- high referral rate
- moderate overall predictive performance
- no external validation

## Repository Contents

- `agentic_xai_stroke_detection.ipynb` — main notebook
- `README.md` — project description
- presentation/report files — optional supporting materials

## How to Run

### Option 1 — Google Colab
The notebook is designed to run in **Google Colab**.

Requirements:
- GPU runtime recommended
- Kaggle API access for dataset download
- Colab secret for Kaggle key if used

### Option 2 — Local environment
Install the required Python packages and run the notebook step by step.

Suggested dependencies:
- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `matplotlib`
- `scikit-learn`

## Future Work

- patient-level evaluation
- validation-based threshold calibration
- improved specificity
- external multi-center validation
- multi-class subtype analysis
- more robust uncertainty calibration

## License

This project is licensed under the **MIT License**.
