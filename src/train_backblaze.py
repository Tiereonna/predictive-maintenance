# train on cleaned backblaze rows
# split by date so the test set is later than the train set (no peeking at the future)

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train import ARTIFACTS, print_metrics

try:
    from xgboost import XGBClassifier

    _XGB_IMPORT_ERROR = None
except Exception as exc:
    XGBClassifier = None
    _XGB_IMPORT_ERROR = exc

FEATURE_CANDIDATES = [
    "smart_5_raw",
    "smart_9_raw",
    "smart_12_raw",
    "smart_187_raw",
    "smart_197_raw",
    "smart_198_raw",
]
LABEL = "failure"


def metrics_dict(y_true, y_prob, y_hat) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
    }


def time_split(frame: pd.DataFrame, frac: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("date")
    cut = ordered["date"].quantile(frac)
    train = ordered[ordered["date"] <= cut]
    test = ordered[ordered["date"] > cut]
    return train, test


def main() -> None:
    path = ARTIFACTS / "backblaze_clean.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    features = [c for c in FEATURE_CANDIDATES if c in frame.columns and frame[c].notna().any()]
    if not features:
        raise SystemExit("no smart columns after etl")

    train, test = time_split(frame)
    print(f"rows={len(frame)} train={len(train)} test={len(test)}")
    print(f"failure_rate train={train[LABEL].mean():.4%} test={test[LABEL].mean():.4%}")
    print(f"features={features}")

    X_train = train[features]
    y_train = train[LABEL]
    X_test = test[features]
    y_test = test[LABEL]

    # SMART columns are often blank for some drive models. fill with the median
    # so logistic regression can run; trees can handle nan on their own.
    logistic = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    logistic.fit(X_train, y_train)
    log_prob = logistic.predict_proba(X_test)[:, 1]
    log_hat = (log_prob >= 0.5).astype(int)
    log_scores = metrics_dict(y_test.to_numpy(), log_prob, log_hat)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if XGBClassifier is not None:
        booster_name = "XGBoost"
        booster = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="logloss",
            random_state=42,
        )
    else:
        booster_name = "HistGradientBoosting"
        print(f"xgboost missing ({_XGB_IMPORT_ERROR})")
        booster = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.08,
            max_iter=80,
            class_weight="balanced",
            random_state=42,
        )

    booster.fit(X_train, y_train)
    boost_prob = booster.predict_proba(X_test)[:, 1]
    boost_hat = (boost_prob >= 0.5).astype(int)
    boost_scores = metrics_dict(y_test.to_numpy(), boost_prob, boost_hat)

    print_metrics("logistic", log_scores)
    print_metrics(booster_name, boost_scores)

    ARTIFACTS.mkdir(exist_ok=True)
    summary = {
        "rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": features,
        "logistic": log_scores,
        "tree": {"name": booster_name, **boost_scores},
    }
    out = ARTIFACTS / "backblaze_metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
