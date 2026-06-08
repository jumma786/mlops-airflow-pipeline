"""
Task: Model Validation
=======================
Gates model promotion based on AUC threshold.
Passes or fails the pipeline.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

AUC_THRESHOLD = 0.75


def validate_model(train_meta: dict, auc_threshold: float = AUC_THRESHOLD) -> dict:
    """
    Validate model metrics against thresholds.
    Raises if model doesn't meet the bar.
    """
    auc    = train_meta["auc"]
    cv_auc = train_meta["cv_auc_mean"]
    passed = auc >= auc_threshold

    meta = {
        "auc":           auc,
        "cv_auc":        cv_auc,
        "threshold":     auc_threshold,
        "passed":        passed,
        "validated_at":  datetime.now().isoformat(),
    }

    if passed:
        print(f"✅ PASSED — AUC={auc:.4f} >= threshold={auc_threshold}")
    else:
        raise ValueError(f"❌ FAILED — AUC={auc:.4f} < threshold={auc_threshold}")

    return meta
