# Credit Default Risk Intelligence

An end-to-end, explainable machine-learning portfolio project using the **Give Me Some Credit** dataset. The project estimates the probability of serious delinquency within two years, compares several classification approaches, combines tuned XGBoost and Random Forest models, selects a decision threshold using out-of-fold F2-score, and explains predictions with SHAP.

## What is included

- A Streamlit application with sidebar navigation — **all application code is in one `app.py` file**.
- The original modeling notebook.
- A companion notebook for exporting the model and SHAP artifacts required by the app.
- All eight supplied result/SHAP chart images.
- Project documentation, data dictionary, and model card.

## Application navigation

1. **Project Overview** — problem statement, dataset scale, target definition, predictors, and capabilities.
2. **Project Workflow** — EDA, data preparation, model comparison, tuning, ensemble weighting, threshold selection, SHAP, and deployment.
3. **Model Findings** — class imbalance, ranking performance, threshold trade-off, and interpretation.
4. **Model Results** — metric table, model comparison charts, threshold metrics, and held-out confusion matrix.
5. **SHAP Explainability** — global beeswarm, mean absolute SHAP importance, and an individual waterfall example.
6. **Try the Model** — enter borrower characteristics, estimate default probability, apply the saved threshold, and calculate an individual SHAP explanation when the required artifacts are present.

## Dataset and target

The notebook reads the Give Me Some Credit training data (`cs-training.csv`). The original dataset contains **150,000 observations**, 10 predictors, and 10,026 positive target observations (approximately **6.68%**).

The target, `SeriousDlqin2yrs`, is renamed to `Target`:

- `Target = 1`: serious delinquency within two years.
- `Target = 0`: no recorded serious delinquency within two years.

The notebook removes one row with `Age = 0`, leaving 149,999 observations, and then performs a stratified 70/30 train/test split using `random_state=42`:

- Training set: 104,999 observations.
- Held-out test set: 45,000 observations.

The raw dataset is **not included**. Obtain it from the original competition/data provider and comply with its terms.

## Modeling workflow

- Exploratory analysis of target imbalance, missingness, distributions, outliers, correlations, and delinquency patterns.
- Comparison of three Logistic Regression preprocessing pipelines.
- Random Forest and XGBoost model development.
- Randomized hyperparameter search using Average Precision.
- Soft-voting probability ensemble of tuned XGBoost and tuned Random Forest.
- Selection of ensemble weights using out-of-fold predictions.
- Classification threshold selection by maximizing out-of-fold F2-score.
- SHAP permutation explanations for global and individual predictions.
- Streamlit deployment interface.

### Final ensemble

The selected ensemble uses equal probability weights:

- XGBoost weight: 1
- Random Forest weight: 1
- Best tested OOF Average Precision for weight selection: **0.387872**

### Threshold selection

The selected threshold is **0.1834**, chosen by maximizing F2-score on out-of-fold training predictions. F2 gives recall more weight than precision, matching the project's stated priority of identifying more positive/default cases.

| OOF threshold metric | Result |
|---|---:|
| Selected threshold | 0.1834 |
| Precision | 0.2451 |
| Recall | 0.7164 |
| F2-score | 0.5175 |

### Held-out test results

At the selected threshold of 0.1834:

| Metric | Held-out test result |
|---|---:|
| ROC-AUC | **0.8644** |
| Average Precision | **0.3910** |
| Precision | 0.2473 |
| Recall | **0.7211** |
| F1-score | 0.3683 |

Confusion matrix (rows = actual, columns = predicted):

| | Predicted non-default | Predicted default |
|---|---:|---:|
| Actual non-default | 35,390 | 6,602 |
| Actual default | 839 | 2,169 |

The lower threshold increases recall compared with the default 0.50 cutoff, while also increasing false positives and lowering precision. This is an explicit operating-point trade-off, not a universal optimum for every lender or business.

## Repository structure

```text
credit-default-risk-intelligence/
├── app.py
├── requirements.txt
├── README.md
├── MODEL_CARD.md
├── DATA_DICTIONARY.md
├── GITHUB_UPLOAD_GUIDE.md
├── .gitignore
├── assets/
│   ├── model_performance_comparison.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── multi_metric_radar.png
│   ├── performance_heatmap.png
│   ├── shap_beeswarm.png
│   ├── shap_global_importance.png
│   └── shap_waterfall.png
└── notebooks/
    ├── Fraud Detection.ipynb
    └── Streamlit_Artifact_Export.ipynb
```

There are no separate Streamlit page modules and no additional `.py` files to create.

## Run locally

Recommended: Python 3.10 or 3.11.

```bash
# 1. Unzip the repository and open a terminal in this folder
cd credit-default-risk-intelligence

# 2. Create a virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The Overview, Workflow, Findings, Results, and static SHAP pages can display using the included chart assets. **Live predictions require the trained model artifact** described below. Live individual SHAP explanations additionally require the SHAP background artifact.

## Required artifacts for live prediction

The model pickle files were not present in the files supplied for packaging, so they are intentionally not fabricated or included.

### 1. `credit_risk_model.pkl` — required for live predictions

The original modeling notebook already saves this artifact in its final model-export cell. Make sure the saved dictionary contains:

```python
{
    "model": best_ensemble,
    "threshold": best_threshold,
    "features": X_train.columns.tolist()
}
```

Run the final model-export cell in `Fraud Detection.ipynb`, then place `credit_risk_model.pkl` in the repository root beside `app.py`.

### 2. `shap_background.pkl` — required for live individual SHAP

Run `notebooks/Streamlit_Artifact_Export.ipynb` after loading the trained model and the saved train/test data. It creates a representative training background sample and saves `shap_background.pkl`.

### 3. `shap_global.pkl`

The companion notebook can also save precomputed global SHAP values. The app currently uses the supplied static global SHAP images for the Explainability page; the global pickle is not required for the normal app navigation.


## Methodological limitations and next steps

This repository reproduces the supplied notebook and its reported results. It should not be presented as a production-validated credit scoring system.

The notebook calculates clipping percentiles before the train/test split. This means the held-out feature distribution contributes to the clipping boundaries. Also, the hyperparameters are selected on the training set before the same training set is used to generate out-of-fold predictions for ensemble-weight and threshold selection. These choices can make reported validation estimates somewhat optimistic.

For a stronger next iteration:

1. Split the data first and estimate all preprocessing parameters using training folds only.
2. Put clipping/winsorization, imputation, and resampling inside a cross-validated pipeline.
3. Use nested cross-validation or a separate validation set for hyperparameter, ensemble-weight, and threshold selection.
4. Keep the final test set untouched until all decisions are fixed.
5. Evaluate probability calibration, subgroup performance, and stability over time.
6. Assess the operational cost of false positives and false negatives with a clearly defined decision-cost framework.

## Interpretation notes

- ROC-AUC measures ranking discrimination across thresholds.
- Average Precision is especially useful when the positive class is rare.
- F2-score weights recall more heavily than precision.
- Positive SHAP values increase the model output relative to its expected output; negative SHAP values decrease it.
- Mean absolute SHAP values measure average contribution magnitude, not direction.
- SHAP explanations describe the fitted model's behavior; they are not causal effects.
- The classification threshold changes the predicted class, not the probability explained by SHAP.


