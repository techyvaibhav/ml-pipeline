"""Train: compare 3 models with stratified CV, tune the best, evaluate, save."""
import json
from datetime import UTC, datetime
from math import prod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    CARD_PATH,
    FEATURES,
    MAJOR,
    MODEL_DIR,
    N_FOLDS,
    N_ITER,
    PROC_DIR,
    REPORT_DIR,
    SEED,
    TARGET,
    TEST_SIZE,
    THRESH,
)
from src.evaluate import get_metrics, make_plots, write_card
from src.features import build_pipeline
from src.ingest import clean, data_source, load_raw
from src.validate import validate


def get_models(quick: bool = False) -> dict:
    """The 3 candidate models. Imbalance: class_weight (LR, RF) / sample_weight (GB)."""
    n = 30 if quick else 200  # trees
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=n, class_weight="balanced_subsample",
                                     random_state=SEED, n_jobs=1),
        "gb": GradientBoostingClassifier(n_estimators=n, random_state=SEED),
    }


GRIDS = {  # search spaces (keys are pipeline params)
    "logreg": {"model__C": [0.01, 0.1, 1, 10]},
    "rf": {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 6, 10, 16],
           "model__min_samples_leaf": [1, 3, 5, 10], "model__max_features": ["sqrt", 0.3]},
    "gb": {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.05, 0.1],
           "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 0.85, 1.0],
           "model__min_samples_leaf": [1, 5, 10]},
}


def fit_kw(name: str, y: pd.Series) -> dict:
    """Extra fit args: balanced sample weights for GB (it has no class_weight)."""
    if name == "gb":
        return {"model__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def cv_eval(name: str, pipe, X: pd.DataFrame, y: pd.Series, k: int) -> dict:
    """Stratified k-fold CV; returns mean/std ROC-AUC and mean PR-AUC, F1."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)  # splitter
    rows = []  # per-fold scores
    for tr, va in skf.split(X, y):
        p = clone(pipe)  # fresh pipeline per fold (no leakage)
        p.fit(X.iloc[tr], y.iloc[tr], **fit_kw(name, y.iloc[tr]))
        pr = p.predict_proba(X.iloc[va])[:, 1]  # val proba
        rows.append((roc_auc_score(y.iloc[va], pr), average_precision_score(y.iloc[va], pr),
                     f1_score(y.iloc[va], pr >= THRESH)))
    arr = np.array(rows)  # folds x 3
    return {"roc_auc": arr[:, 0].mean(), "roc_auc_std": arr[:, 0].std(),
            "pr_auc": arr[:, 1].mean(), "f1": arr[:, 2].mean()}


def run(df: pd.DataFrame | None = None, model_dir: Path = MODEL_DIR, rep_dir: Path = REPORT_DIR,
        proc_dir: Path | None = PROC_DIR, quick: bool = False) -> dict:
    """Full training run. Returns the metrics dict (also saved to disk)."""
    raw = load_raw() if df is None else df  # raw df
    data = validate(clean(raw))  # cleaned + validated
    X = data[FEATURES]  # features
    y = data[TARGET].astype(int)  # target
    Xtr, Xte, ytr, yte = train_test_split(  # split FIRST, then fit on train only
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    if proc_dir is not None:
        proc_dir.mkdir(parents=True, exist_ok=True)
        Xtr.assign(**{TARGET: ytr}).to_csv(proc_dir / "train.csv", index=False)
        Xte.assign(**{TARGET: yte}).to_csv(proc_dir / "test.csv", index=False)

    k = 3 if quick else N_FOLDS  # CV folds
    cv = {}  # model name -> cv scores
    for name, mdl in get_models(quick).items():
        cv[name] = cv_eval(name, build_pipeline(mdl), Xtr, ytr, k)
        print(f"[cv] {name}: ROC-AUC {cv[name]['roc_auc']:.4f} PR-AUC {cv[name]['pr_auc']:.4f}")
    best = max(cv, key=lambda n: cv[n]["roc_auc"])  # best by CV ROC-AUC
    print(f"[train] best = {best}; tuning...")

    grid = GRIDS[best]  # search space
    n_it = min(2 if quick else N_ITER, prod(len(v) for v in grid.values()))  # iterations
    search = RandomizedSearchCV(
        build_pipeline(get_models(quick)[best]), grid, n_iter=n_it, scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=-1)
    search.fit(Xtr, ytr, **fit_kw(best, ytr))
    pipe = search.best_estimator_  # tuned full pipeline

    proba = pipe.predict_proba(Xte)[:, 1]  # test proba
    test_m = get_metrics(yte, proba)  # hold-out metrics
    ts = datetime.now(UTC)  # train time
    ver = f"{MAJOR}-{ts:%Y%m%d%H%M%S}"  # model version
    m = {
        "version": ver, "trained_at": ts.isoformat(timespec="seconds"),
        "data_source": data_source() if df is None else "synthetic",
        "best_model": best, "best_params": search.best_params_, "threshold": THRESH,
        "n_folds": k, "cv_results": cv, "test_metrics": test_m,
        "n_train": len(Xtr), "n_test": len(Xte), "churn_rate_train": float(ytr.mean()),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "version": ver, "trained_at": m["trained_at"],
                 "algorithm": best, "features": FEATURES, "threshold": THRESH,
                 "metrics": test_m}, model_dir / "model.joblib", compress=3)
    (model_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    write_card(m, model_dir / CARD_PATH.name)
    make_plots(pipe, yte, proba, rep_dir)
    print(f"[train] saved {ver}; test: " + ", ".join(f"{a}={b:.3f}" for a, b in test_m.items()))
    return m


if __name__ == "__main__":
    run()
