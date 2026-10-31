"""Metrics, plots and model card."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CARD_PATH, PROC_DIR, REPORT_DIR, TARGET, THRESH  # noqa: E402
from src.features import feat_names  # noqa: E402


def get_metrics(y, proba, thr: float = THRESH) -> dict:
    """All headline metrics for true labels y and churn probabilities proba."""
    pred = (np.asarray(proba) >= thr).astype(int)  # hard labels
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def make_plots(pipe, y, proba, rep_dir: Path = REPORT_DIR, thr: float = THRESH) -> None:
    """Save ROC, PR, confusion matrix and feature importance plots."""
    rep_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, proba)  # roc points
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rep_dir / "roc_curve.png", dpi=120)
    plt.close(fig)

    pre, rec, _ = precision_recall_curve(y, proba)  # pr points
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, pre, label=f"AP={average_precision_score(y, proba):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rep_dir / "pr_curve.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, (np.asarray(proba) >= thr).astype(int), ax=ax)
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(rep_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    mdl = pipe.named_steps["model"]  # fitted estimator
    imp = getattr(mdl, "feature_importances_", None)  # tree importances
    if imp is None:
        imp = np.abs(mdl.coef_[0])  # linear coefs
    ser = pd.Series(imp, index=feat_names(pipe)).nlargest(15).sort_values()  # top 15
    fig, ax = plt.subplots(figsize=(6, 5))
    ser.plot.barh(ax=ax)
    ax.set_title("Top 15 feature importances")
    fig.tight_layout()
    fig.savefig(rep_dir / "feature_importance.png", dpi=120)
    plt.close(fig)


def write_card(m: dict, path: Path = CARD_PATH) -> None:
    """Write model_card.md from the metrics dict."""
    t = m["test_metrics"]  # test metrics
    cv = "\n".join(
        f"| {k} | {v['roc_auc']:.3f} ± {v['roc_auc_std']:.3f} | {v['pr_auc']:.3f} | {v['f1']:.3f} |"
        for k, v in m["cv_results"].items()
    )  # cv table rows
    synth = m["data_source"] == "synthetic"  # synthetic flag
    lim = (  # first limitation line
        "Trained on SYNTHETIC data: metrics say nothing about real-world performance."
        if synth
        else "Single telecom dataset (US-style plans): may not generalise to other markets/periods."
    )
    txt = f"""# Model Card: Customer Churn Classifier

## Model
- **Version:** {m['version']} (trained {m['trained_at']})
- **Algorithm:** {m['best_model']} (selected by CV ROC-AUC, tuned with RandomizedSearchCV)
- **Best params:** `{json.dumps(m['best_params'], default=str)}`
- **Decision threshold:** {m['threshold']}
- **Pipeline:** impute + scale numeric, impute + one-hot categorical, then classifier (one joblib).

## Data
- **Source:** {m['data_source']} Telco Customer Churn dataset
- **Rows:** {m['n_train']} train / {m['n_test']} test (stratified split, seed fixed)
- **Churn rate (train):** {m['churn_rate_train']:.1%}
- **Target:** Churn (1 = customer left)

## Metrics (hold-out test set)
| Metric | Value |
|---|---|
| Accuracy | {t['accuracy']:.3f} |
| Precision | {t['precision']:.3f} |
| Recall | {t['recall']:.3f} |
| F1 | {t['f1']:.3f} |
| ROC-AUC | {t['roc_auc']:.3f} |
| PR-AUC | {t['pr_auc']:.3f} |

## Cross-validation (stratified {m['n_folds']}-fold, train set)
| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
{cv}

## Limitations
- {lim}
- Class imbalance handled with class weights, so probabilities are not calibrated.
- Uses `gender` as a feature; fairness across groups has not been audited.
- Static snapshot: no time-based split, so performance under drift is unknown (see `src/drift.py`).
- Intended for ranking/prioritising retention outreach, not for automated decisions about customers.
"""
    path.write_text(txt)


def main() -> None:
    """Re-evaluate the saved model on data/processed/test.csv."""
    from src.predict import load_bundle  # local import avoids a cycle

    b = load_bundle()  # saved bundle
    df = pd.read_csv(PROC_DIR / "test.csv")  # processed test set
    proba = b["pipeline"].predict_proba(df.drop(columns=[TARGET]))[:, 1]  # churn proba
    print(json.dumps(get_metrics(df[TARGET], proba), indent=2))
    make_plots(b["pipeline"], df[TARGET], proba)


if __name__ == "__main__":
    main()
