# Model Card: Customer Churn Classifier

## Model
- **Version:** 1.0-20261031163548 (trained 2026-10-31T16:35:48+00:00)
- **Algorithm:** logreg (selected by CV ROC-AUC, tuned with RandomizedSearchCV)
- **Best params:** `{"model__C": 10}`
- **Decision threshold:** 0.5
- **Pipeline:** impute + scale numeric, impute + one-hot categorical, then classifier (one joblib).

## Data
- **Source:** real Telco Customer Churn dataset
- **Rows:** 5634 train / 1409 test (stratified split, seed fixed)
- **Churn rate (train):** 26.5%
- **Target:** Churn (1 = customer left)

## Metrics (hold-out test set)
| Metric | Value |
|---|---|
| Accuracy | 0.740 |
| Precision | 0.506 |
| Recall | 0.783 |
| F1 | 0.615 |
| ROC-AUC | 0.841 |
| PR-AUC | 0.627 |

## Cross-validation (stratified 5-fold, train set)
| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| logreg | 0.846 ± 0.012 | 0.660 | 0.628 |
| rf | 0.821 ± 0.012 | 0.610 | 0.543 |
| gb | 0.843 ± 0.012 | 0.657 | 0.626 |

## Limitations
- Single telecom dataset (US-style plans): may not generalise to other markets/periods.
- Class imbalance handled with class weights, so probabilities are not calibrated.
- Uses `gender` as a feature; fairness across groups has not been audited.
- Static snapshot: no time-based split, so performance under drift is unknown (see `src/drift.py`).
- Intended for ranking/prioritising retention outreach, not for automated decisions about customers.
