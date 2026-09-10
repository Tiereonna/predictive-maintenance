"""
Train a simple baseline and XGBoost, then print the metrics a resume mentions.

Why two models: logistic regression is the honest baseline (linear, readable
coefficients). XGBoost is the stronger tree model. If XGBoost is not clearly
better on recall / F1, you should say that in an interview — "I compared,
I did not assume boosting wins."

Class imbalance: failures are rarer than healthy rows. Accuracy would look
great if the model always predicted 0. Precision / recall / F1 / ROC-AUC
are what you quote instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generate_data import generate_sensor_frame

try:
    from xgboost import XGBClassifier

    _XGB_IMPORT_ERROR = None
except Exception as exc:  # libomp missing on some Macs
    XGBClassifier = None
    _XGB_IMPORT_ERROR = exc

FEATURE_COLS = [
    "hours_run",
    "vibration_mm_s",
    "temperature_c",
    "pressure_psi",
    "current_amps",
]
LABEL = "failed_within_7_days"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def print_metrics(name: str, scores: dict[str, float]) -> None:
    print(f"\n{name}")
    for key, value in scores.items():
        print(f"  {key:9s} {value:.3f}")


def main() -> None:
    frame = generate_sensor_frame()
    fail_rate = frame[LABEL].mean()
    print(f"rows={len(frame)}  failure_rate={fail_rate:.1%}")
    print("A dummy 'always healthy' model gets accuracy ≈ 1 − failure_rate.")
    print("That is why we quote precision / recall / F1 / ROC-AUC instead.")

    X = frame[FEATURE_COLS]
    y = frame[LABEL]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Scale for logistic regression only. Trees do not need it.
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    logistic.fit(X_train, y_train)
    log_prob = logistic.predict_proba(X_test)[:, 1]
    log_hat = (log_prob >= 0.5).astype(int)
    log_scores = metrics_dict(y_test.to_numpy(), log_prob, log_hat)

    # Boosted trees. Prefer XGBoost; fall back if the native lib is missing
    # (common on Mac without `brew install libomp`). Same idea either way:
    # many small trees, failures weighted more than healthy rows.
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if XGBClassifier is not None:
        booster_name = "XGBoost"
        booster = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="logloss",
            random_state=42,
        )
    else:
        booster_name = "HistGradientBoosting (XGBoost unavailable)"
        print(f"XGBoost not loaded ({_XGB_IMPORT_ERROR}). Using sklearn HGB.")
        booster = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.08,
            max_iter=120,
            class_weight="balanced",
            random_state=42,
        )

    booster.fit(X_train, y_train)
    boost_prob = booster.predict_proba(X_test)[:, 1]
    boost_hat = (boost_prob >= 0.5).astype(int)
    boost_scores = metrics_dict(y_test.to_numpy(), boost_prob, boost_hat)

    print_metrics("Logistic regression (baseline)", log_scores)
    print_metrics(booster_name, boost_scores)

    print(f"\n{booster_name} classification report")
    print(classification_report(y_test, boost_hat, digits=3, zero_division=0))

    coefs = logistic.named_steps["clf"].coef_[0]
    log_importance = pd.Series(coefs, index=FEATURE_COLS).sort_values(key=abs, ascending=False)
    print("\nLogistic coefficients (sign = direction after scaling)")
    print(log_importance.to_string())

    if hasattr(booster, "feature_importances_"):
        boost_importance = pd.Series(
            booster.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        print(f"\n{booster_name} feature importance (not causation)")
        print(boost_importance.to_string())
    else:
        # HistGradientBoosting has no .feature_importances_ on some versions.
        # Permutation importance: shuffle one column, see how ROC-AUC drops.
        perm = permutation_importance(
            booster,
            X_test,
            y_test,
            n_repeats=8,
            random_state=42,
            scoring="roc_auc",
        )
        boost_importance = pd.Series(perm.importances_mean, index=FEATURE_COLS).sort_values(
            ascending=False
        )
        print(f"\n{booster_name} permutation importance (ROC-AUC drop, not causation)")
        print(boost_importance.to_string())

    ARTIFACTS.mkdir(exist_ok=True)
    summary = pd.DataFrame(
        [{"model": "logistic", **log_scores}, {"model": booster_name, **boost_scores}]
    )
    out = ARTIFACTS / "metrics.csv"
    summary.to_csv(out, index=False)
    joblib.dump(
        {"logistic": logistic, "booster": booster, "booster_name": booster_name},
        ARTIFACTS / "models.joblib",
    )
    (ARTIFACTS / "importance.json").write_text(
        json.dumps(
            {
                "logistic": log_importance.to_dict(),
                "booster": boost_importance.to_dict(),
                "booster_name": booster_name,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")
    print(f"Wrote {ARTIFACTS / 'models.joblib'}")


if __name__ == "__main__":
    main()
