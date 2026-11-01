"""
Model Evaluation & Diagnostic Reporting Module.
HCL Data Science Project
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_evaluation_summary(base_dir: str):
    """Loads metadata and prints comprehensive diagnostic report."""
    metadata_path = os.path.join(base_dir, "models", "model_metadata.json")
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found at {metadata_path}")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("=" * 70)
    print("HCL DATA SCIENCE PROJECT: DIABETIC PATIENT HEALTH RECORD DASHBOARD")
    print("MODEL BENCHMARKING & VALIDATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Target Variable: {meta['target']}")
    print(f"Training Encounters: {meta['train_samples']} | Testing Encounters: {meta['test_samples']}")
    print(f"Positive Readmission Rate (<30 days): {meta['positive_rate_pct']}%\n")

    print(f"{'Model Architecture':<22} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Tuned Thresh':<12} | {'Recall':<8} | {'Precision':<10} | {'F1':<6}")
    print("-" * 88)

    for m_name, metrics in meta["models_comparison"].items():
        tuned = metrics["tuned_threshold_recall_60"]
        print(
            f"{m_name:<22} | "
            f"{metrics['roc_auc']:<8.4f} | "
            f"{metrics['pr_auc']:<8.4f} | "
            f"{tuned['threshold']:<12.4f} | "
            f"{tuned['recall']:<8.4f} | "
            f"{tuned['precision']:<10.4f} | "
            f"{tuned['f1']:<6.4f}"
        )

    print("-" * 88)
    print(f"Champion Model Selected: {meta['best_model']} (ROC-AUC: {meta['best_roc_auc']:.4f})")
    print("\nTop Clinical Predictors (Feature Importance):")
    for idx, item in enumerate(meta["top_features"][:10], 1):
        print(f"  {idx:2d}. {item['feature']:<40} -> {item['importance'] * 100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    project_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    generate_evaluation_summary(project_base)
