"""
Task: Model Training
=====================
Trains XGBoost on engineered features.
Saves model + metrics for validation task.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
from datetime import datetime

logger = logging.getLogger(__name__)


def train_model(input_path: str, output_dir: str = "artifacts") -> dict:
    """Train XGBoost model on engineered features."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(input_path)
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=8, eval_metric="logloss",
        verbosity=0, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred, zero_division=0)

    # CV score
    cv_scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")

    # Save
    model_path = os.path.join(output_dir, "model.json")
    model.save_model(model_path)

    features_path = os.path.join(output_dir, "feature_order.json")
    with open(features_path, "w") as f:
        json.dump(list(X.columns), f)

    meta = {
        "auc":           round(float(auc), 4),
        "f1":            round(float(f1), 4),
        "cv_auc_mean":   round(float(cv_scores.mean()), 4),
        "cv_auc_std":    round(float(cv_scores.std()), 4),
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "model_path":    model_path,
        "features_path": features_path,
        "trained_at":    datetime.now().isoformat(),
    }

    print(f"Train AUC: {auc:.4f} | F1: {f1:.4f} | CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return meta
