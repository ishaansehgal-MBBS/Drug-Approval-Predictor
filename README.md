# Drug-Approval-Predictor
# 🧬 Clinical Trial Success Predictor & Quantitative Equity Model

A machine learning pipeline built in Python to predict Phase II/III clinical trial outcomes and generate risk-adjusted probability scores for event-driven long/short healthcare equity strategies.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Biopharmaceutical equities experience extreme volatility around clinical trial readouts (Phase II/III catalysts). This project builds an end-to-end predictive machine learning framework using historical trial data (~1,300 observations across 40 features) to quantify the likelihood of trial success. 

By mapping model probability outputs directly to binary event risk, the model serves as an algorithmic decision-support tool for long/short healthcare portfolio management.

---

## Key Features

* **Data Preprocessing & Scaling:** Cleans categorical and continuous variables, utilizing `MinMaxScaler` on target patient accrual size (`taracc`).
* **Stratified Validation:** Employs a 70/30 train/test split and 5-fold `StratifiedKFold` cross-validation to maintain consistent target class distribution across splits [1].
* **Model Benchmarking & Tuning:** Trains and compares Logistic Regression and Random Forest Classifiers using `GridSearchCV` hyperparameter optimization [1].
* **Feature Explainability:** Extracts coefficient weights and tree importances to identify key structural drivers of clinical trial success (e.g., administration route, study design, sponsor tier) [1].

---

## Performance & Results

Given the class imbalance in historical trial outcomes (~18.7% baseline success rate) [1], performance is evaluated using Area Under the Receiver Operating Characteristic Curve (**ROC-AUC**):

| Model | Setup | Out-of-Sample AUC |
| :--- | :--- | :--- |
| **Logistic Regression** | Baseline | `0.7561` |
| **Logistic Regression** | Tuned (`GridSearchCV`) | `0.7586` |
| **Random Forest** | Baseline | `0.7905` |
| **Random Forest** | **Tuned (`GridSearchCV`)** | **`0.8254`** |

### Top Clinical Success Drivers (Logistic Regression Weights)
1. **Oral Administration (`route.1`):** Positively impacts success probability over complex routes [1].
2. **Double-Blind Design (`dkw.13`):** High study design rigor strongly correlates with trial success [1].
3. **Non-Top 20 Pharma Sponsor (`stid.1`):** Specialized mid-tier pharma backing provides positive probability lift [1].
4. **Top 20 Pharma Sponsor (`stid.31`):** Institutional resource backing improves success odds [1].
5. **Biologics / Antibodies (`origin.25`):** Protein/antibody drug origins show structural advantages over traditional small molecules [1].

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/clinical-trial-predictor.git](https://github.com/yourusername/clinical-trial-predictor.git)
   cd clinical-trial-predictor
