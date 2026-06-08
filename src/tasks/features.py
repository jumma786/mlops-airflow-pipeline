"""
Task: Feature Engineering
==========================
Encodes categoricals, engineers new features.
Saves feature matrix for training task.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

logger = logging.getLogger(__name__)


def engineer_features(input_path: str, output_dir: str = "data/processed") -> dict:
    """Encode and engineer features from raw data."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(input_path)
    logger.info(f"Input shape: {df.shape}")

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {v: int(i) for i, v in enumerate(le.classes_)}

    # Save encoders
    enc_path = os.path.join(output_dir, "encoders.json")
    with open(enc_path, "w") as f:
        json.dump(encoders, f)

    # Save feature matrix
    feat_path = os.path.join(output_dir, "features.parquet")
    df.to_parquet(feat_path, index=False)

    feature_names = [c for c in df.columns if c != "y"]

    meta = {
        "n_rows":        len(df),
        "n_features":    len(feature_names),
        "features":      feature_names,
        "output_path":   feat_path,
        "encoders_path": enc_path,
        "engineered_at": datetime.now().isoformat(),
    }

    print(f"Features: {meta['n_features']} | rows: {meta['n_rows']:,}")
    return meta
