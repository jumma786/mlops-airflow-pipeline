"""
Task: Data Ingestion
=====================
Loads and validates raw Bank Marketing data.
Returns summary statistics for downstream tasks.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def ingest_data(data_path: str = None, output_dir: str = "data/processed") -> dict:
    """
    Ingest and validate raw data.
    Returns metadata dict passed to downstream tasks via XCom.
    """
    os.makedirs(output_dir, exist_ok=True)

    if data_path and os.path.exists(data_path):
        logger.info(f"Loading real data: {data_path}")
        df = pd.read_csv(data_path, sep=";")
        df["y"] = (df["y"] == "yes").astype(int)
        source = "real"
    else:
        logger.info("Generating synthetic data")
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "age":           np.random.randint(18, 95, n),
            "job":           np.random.choice(["admin.","blue-collar","management","retired","technician"], n),
            "marital":       np.random.choice(["divorced","married","single"], n),
            "education":     np.random.choice(["basic.4y","high.school","university.degree"], n),
            "default":       np.random.choice(["no","unknown"], n),
            "housing":       np.random.choice(["no","yes"], n),
            "loan":          np.random.choice(["no","yes"], n),
            "contact":       np.random.choice(["cellular","telephone"], n),
            "month":         np.random.choice(["mar","apr","may","jun","jul","aug"], n),
            "day_of_week":   np.random.choice(["mon","tue","wed","thu","fri"], n),
            "campaign":      np.random.randint(1, 10, n),
            "pdays":         np.where(np.random.rand(n) < 0.13, np.random.randint(1,30,n), 999),
            "previous":      np.random.randint(0, 5, n),
            "poutcome":      np.random.choice(["failure","nonexistent","success"], n),
            "emp.var.rate":  np.random.choice([-1.8, 1.1, 1.4], n),
            "cons.price.idx": np.random.uniform(92.2, 94.8, n).round(3),
            "cons.conf.idx": np.random.uniform(-50.8, -26.9, n).round(1),
            "euribor3m":     np.random.uniform(0.6, 5.1, n).round(3),
            "nr.employed":   np.random.choice([4963.6, 5099.1, 5228.1], n),
            "y":             (np.random.rand(n) < 0.11).astype(int),
        })
        source = "synthetic"

    # Drop leakage
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Validate
    assert len(df) > 0, "Empty dataset"
    assert "y" in df.columns, "Missing target column"

    # Save
    out_path = os.path.join(output_dir, "raw_data.parquet")
    df.to_parquet(out_path, index=False)

    meta = {
        "n_rows":         len(df),
        "n_cols":         len(df.columns),
        "positive_rate":  round(float(df["y"].mean()), 4),
        "source":         source,
        "output_path":    out_path,
        "ingested_at":    datetime.now().isoformat(),
    }

    print(f"Ingested: {meta['n_rows']:,} rows | positive rate: {meta['positive_rate']:.1%}")
    return meta
