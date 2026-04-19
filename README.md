# CT-Based COVID-19 Diagnosis Using Machine Learning 🦠🩺

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-success.svg)

## 📌 Project Overview
This repository contains a robust, supervised machine learning pipeline designed to classify COVID-19 anomalies from high-dimensional CT scan radiomics data. 

To address the common "black-box" limitations in medical AI, this project implements a **Hybrid Pipeline Architecture**, combining the high predictive accuracy of a Support Vector Machine (SVM) with the human-readable interpretability of a Decision Tree. The final output is an automated, text-based **AI Diagnostic Report** for clinical triage.

### Problem Statement 🎯
Chest CT scans are a critical tool in detecting COVID-19 anomalies, but accurate diagnosis relies heavily on the limited time and subjective interpretation of expert radiologists. While quantitative "radiomics" feature extraction can pull thousands of hidden data points from a single scan, this creates high-dimensional data that is impossible for humans to evaluate manually.

**The Objective:** This project aims to efficiently process this high-dimensional radiomics data to automatically and accurately classify COVID-19 positive vs. negative scans. Specifically, it utilizes **Linear Discriminant Analysis (LDA)** for supervised dimensionality reduction, trains a **Support Vector Machine (SVM)** to find the optimal decision boundary, and deploys an auxiliary **Decision Tree** to extract interpretable diagnostic rules.

---

## 🛠️ Methodology & Pipeline Architecture

The pipeline uses the **UCSD COVID-CT-Dataset**, extracting quantitative features using PyRadiomics, and processes a total of **746 images** (596 Train / 150 Test). 

1. **Feature Extraction:** Over 1,000 continuous features (GLCM, GLRLM, First Order, etc.) are extracted from 2D CT scans using `pyradiomics`.
2. **Dimensionality Reduction (LDA):** To prevent the curse of dimensionality, supervised Linear Discriminant Analysis compresses the feature space down to 1 core component. *Note: Transformers are fit strictly on training folds to prevent data leakage.*
3. **Primary Predictive Model (SVM):** An SVM with a Radial Basis Function (RBF) kernel acts as the primary classifier, heavily regularized (`C=0.1`, `gamma=0.01`) to prevent overfitting.
4. **Explainer Model (Decision Tree):** A heavily pruned Classification Tree (`max_depth=3`) is trained in parallel to extract clinical rules.
5. **Validation:** Evaluated using 5-Fold Stratified Cross-Validation to ensure consistent generalization.

---

## 📊 Results & Performance

The model demonstrates excellent generalization from training to unseen holdout data, with minimal drop in ROC AUC, indicating a highly stable decision boundary.

| Metric | Score |
| :--- | :--- |
| **Cross-Validation Mean AUC** | 0.8807 (± 0.0337) |
| **Holdout ROC AUC** | 0.8482 |
| **Holdout Accuracy** | 78.67% |
| **Recall (COVID / Class 1)** | 0.80 |
| **Recall (Non-COVID / Class 0)** | 0.78 |

### 💡 Key Insight: The Interpretability Trade-off (Sample #0)
This project highlights the danger of relying purely on hard classification thresholds (Accuracy) in medical data. 

During testing, **Sample #0** triggered a model disagreement:
* **SVM Probability:** `45.47%` (Classified as *Negative / Borderline*)
* **Decision Tree:** (Classified as *Positive*)

Because the SVM predicted a probability just under the 50% cutoff, it labeled the scan as Negative. However, the Decision Tree, which relies on rigid, hard-coded spatial boundaries, forced the scan into a Positive classification. This edge case perfectly illustrates why the generated **AI Diagnostic Report** outputs the raw SVM Probability alongside a "Risk Band" (e.g., *Borderline — monitor and re-evaluate*), signaling to clinicians that human judgment is required rather than blind trust in a binary label.

---

## 🚀 How to Run in Google Colab

1. Clone this repository or download the `Final_ML_Project.ipynb` notebook.
2. Upload the notebook to Google Colab.
3. Ensure the raw image zip files (`CT_COVID.zip` and `CT_NonCOVID.zip`) are uploaded to your Colab session.
4. Run the notebook from top to bottom. The script will automatically:
   * Install necessary C-extensions for `pyradiomics`.
   * Unzip the image folders.
   * Extract features, train the models, and print the final **AI Diagnostic Report**.

---
## 🎓 Acknowledgments
* **Dataset:** The dataset used for feature extraction is the [UCSD COVID-CT-Dataset](https://github.com/UCSD-AI4H/COVID-CT).
* **Citation:** Zhao, Jinyu, et al. "COVID-CT-Dataset: a CT scan dataset about COVID-19." *arXiv preprint arXiv:2003.13865* (2020).
* **Feature Extraction:** [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/) (Harvard-MGH AIM).
## 🎓 Acknowledgments
* **Dataset:** The dataset used for feature extraction is the [UCSD COVID-CT-Dataset](https://github.com/UCSD-AI4H/COVID-CT).
* **Feature Extraction:** [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/) (Harvard-MGH AIM).
