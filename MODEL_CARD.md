# Model Card — Credit Default Risk Prediction

## Model purpose

Estimate the probability that a record will have `Target = 1`, corresponding to serious delinquency within two years in the Give Me Some Credit dataset.

## Intended use

- Educational demonstration of imbalanced binary classification.
- Portfolio demonstration of model comparison, ensemble learning, threshold selection, and SHAP explainability.
- Exploration of the precision/recall trade-off.

## Out-of-scope use

- Automated credit approvals, denials, pricing, or credit-limit decisions.
- Real-world underwriting without independent validation, legal review, governance, and monitoring.
- Treating SHAP contributions as causal explanations.

## Data

- Original observations: 150,000.
- Predictors: 10.
- Positive cases: 10,026 (about 6.68%).
- One record with `Age = 0` removed.
- Stratified split: 104,999 train / 45,000 test.

See `DATA_DICTIONARY.md` for feature definitions.

## Model

Soft-voting ensemble of tuned XGBoost and tuned Random Forest with equal weights (1:1). The decision threshold is 0.1834, selected by maximizing out-of-fold F2-score.

## Performance

Held-out test results at threshold 0.1834:

- ROC-AUC: 0.8644
- Average Precision: 0.3910
- Precision: 0.2473
- Recall: 0.7211
- F1: 0.3683

OOF threshold-selection results:

- Precision: 0.2451
- Recall: 0.7164
- F2: 0.5175

## Trade-offs

The selected threshold favors recall. On the held-out test set, it identifies 2,169 of 3,008 positive cases but also flags 6,602 negative cases. The relative costs of these errors depend on the real-world decision context and are not specified by this portfolio exercise.

## Limitations

- The notebook computes clipping bounds before splitting, allowing test-set feature information to influence preprocessing.
- Hyperparameter tuning precedes OOF ensemble/threshold selection; nested validation would provide a more rigorous estimate.
- No calibration, temporal validation, subgroup fairness assessment, or external validation is reported here.
- Dataset labels and historical patterns may not represent current populations or lending environments.
- SHAP values describe model associations, not causal effects.

## Ethical considerations

Credit-related predictions can affect access to financial services. This project is for education only and is not suitable for actual lending decisions. Any real deployment would require legal/compliance review, fairness analysis, explainability governance, data quality controls, model monitoring, and meaningful human oversight.
