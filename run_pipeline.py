"""
Standalone Pipeline Runner
===========================
Runs the full ML pipeline without requiring Airflow.
Useful for local testing and CI/CD.

Usage:
    python run_pipeline.py
    python run_pipeline.py --data-path data/bank-additional-full.csv
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tasks.ingest   import ingest_data
from src.tasks.features import engineer_features
from src.tasks.train    import train_model
from src.tasks.validate import validate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(data_path: str = None):
    """Run full pipeline sequentially."""
    print("\n" + "="*55)
    print("MLOps Training Pipeline — Starting")
    print("="*55)

    # Step 1 — Ingest
    print("\n[1/4] Ingesting data...")
    ingest_meta = ingest_data(data_path=data_path)

    # Step 2 — Feature engineering
    print("\n[2/4] Engineering features...")
    feature_meta = engineer_features(input_path=ingest_meta["output_path"])

    # Step 3 — Train
    print("\n[3/4] Training model...")
    train_meta = train_model(input_path=feature_meta["output_path"])

    # Step 4 — Validate
    print("\n[4/4] Validating model...")
    validate_meta = validate_model(train_meta=train_meta)

    print("\n" + "="*55)
    print("PIPELINE COMPLETE")
    print("="*55)
    print(f"  Source:   {ingest_meta['source']}")
    print(f"  Rows:     {ingest_meta['n_rows']:,}")
    print(f"  Features: {feature_meta['n_features']}")
    print(f"  AUC:      {train_meta['auc']}")
    print(f"  CV AUC:   {train_meta['cv_auc_mean']} ± {train_meta['cv_auc_std']}")
    print(f"  Passed:   {validate_meta['passed']}")
    print(f"  Model:    {train_meta['model_path']}")
    print("="*55)

    return {
        "ingest":   ingest_meta,
        "features": feature_meta,
        "train":    train_meta,
        "validate": validate_meta,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    args = p.parse_args()
    run_pipeline(data_path=args.data_path)
