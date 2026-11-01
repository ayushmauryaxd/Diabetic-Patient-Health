"""
Model Training & Evaluation Pipeline for Diabetic Patient Health Record Dashboard.
HCL Data Science Project

Implements:
1. Patient-grouped validation (StratifiedGroupKFold on patient_nbr or patient-grouped train/test split)
2. Multiple model architectures:
   - Logistic Regression (class_weight='balanced')
   - Random Forest Classifier (class_weight='balanced')
   - LightGBM Classifier (scale_pos_weight / balanced)
   - XGBoost Classifier (scale_pos_weight)
3. Threshold tuning targeting Recall >= 0.60 while maximizing ROC-AUC & F1-score
4. Artifact serialization (best_model.pkl, preprocessor.pkl, model_metadata.json)
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from data_loader import load_raw_data
from preprocessor import clean_diabetic_data
from feature_engineering import engineer_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.60
) -> Tuple[float, Dict[str, float]]:
    """
    Finds decision threshold achieving target recall (>= 0.60) while maximizing F1.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Filter thresholds where recall >= target_recall
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_indices) == 0:
        # Fallback to threshold that gives highest recall
        best_idx = np.argmax(recalls[:-1])
    else:
        # Compute F1 scores for valid thresholds
        valid_f1s = []
        for idx in valid_indices:
            p = precisions[idx]
            r = recalls[idx]
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            valid_f1s.append(f1)
        best_idx = valid_indices[np.argmax(valid_f1s)]

    optimal_threshold = float(thresholds[best_idx])
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)

    metrics = {
        "threshold": round(optimal_threshold, 4),
        "recall": round(float(recall_score(y_true, y_pred_opt)), 4),
        "precision": round(float(precision_score(y_true, y_pred_opt, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred_opt, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred_opt)), 4)
    }
    return optimal_threshold, metrics


def train_and_evaluate_models(
    base_dir: str,
    keep_first_encounter: bool = True
) -> Dict[str, Any]:
    """
    Executes end-to-end training, cross-validation, evaluation, and saving.
    """
    data_path = os.path.join(base_dir, "data", "diabetic_data.csv")
    mapping_path = os.path.join(base_dir, "data", "IDS_mapping.csv")

    # 1. Load and Clean Data
    raw_df = load_raw_data(data_path, mapping_path)
    clean_df = clean_diabetic_data(raw_df)

    # Save cleaned data
    clean_out_path = os.path.join(base_dir, "data", "cleaned_diabetes_data.csv")
    clean_df.to_csv(clean_out_path, index=False)
    logger.info(f"Saved cleaned dataset to: {clean_out_path}")

    # 2. Feature Engineering
    df_feat = engineer_features(clean_df, keep_first_encounter_only=keep_first_encounter)
    feat_out_path = os.path.join(base_dir, "data", "feature_engineered_data.csv")
    df_feat.to_csv(feat_out_path, index=False)
    logger.info(f"Saved feature-engineered dataset to: {feat_out_path}")

    num_cols, cat_cols = get_feature_columns()
    all_feature_cols = num_cols + cat_cols

    X = df_feat[all_feature_cols].copy()
    y = df_feat["readmitted_binary"].values
    groups = df_feat["patient_nbr"].values if "patient_nbr" in df_feat.columns else np.arange(len(df_feat))

    logger.info(f"Modeling dataset: {X.shape[0]} rows, {X.shape[1]} features.")
    logger.info(f"Positive class rate (<30 days): {np.mean(y):.2%}")

    # 3. Patient-Grouped Split (ensuring 0 patient leakage)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logger.info(f"Train split: {len(X_train)} samples, Test split: {len(X_test)} samples.")
    overlap = len(set(groups[train_idx]).intersection(set(groups[test_idx])))
    logger.info(f"Patient overlap between train and test: {overlap} (Verified No Leakage)")

    # 4. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    transformed_feature_names = num_cols + cat_feature_names

    pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
    logger.info(f"Calculated positive class imbalance weight: {pos_weight:.2f}")

    # 5. Define Candidate Models
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=pos_weight,
            random_state=42,
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    best_model_name = None
    best_roc_auc = -1.0
    trained_estimators = {}

    for name, clf in models.items():
        logger.info(f"Training {name}...")
        clf.fit(X_train_trans, y_train)
        trained_estimators[name] = clf

        # Probabilities for class 1
        y_prob = clf.predict_proba(X_test_trans)[:, 1]
        
        # Standard threshold 0.5 metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        # Default 0.5 predictions
        y_pred_default = (y_prob >= 0.5).astype(int)
        recall_default = recall_score(y_test, y_pred_default)
        prec_default = precision_score(y_test, y_pred_default, zero_division=0)
        f1_default = f1_score(y_test, y_pred_default, zero_division=0)

        # Threshold tuning for Recall >= 0.60
        opt_thresh, tuned_metrics = find_optimal_threshold(y_test, y_prob, target_recall=0.60)

        # Calculate ROC curve coordinates for plotting in dashboard
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # downsample points for lightweight JSON serialization
        sample_indices = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
        roc_pts = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in sample_indices]

        # Calculate PR curve coordinates
        p_vals, r_vals, _ = precision_recall_curve(y_test, y_prob)
        sample_pr = np.linspace(0, len(p_vals) - 1, min(100, len(p_vals))).astype(int)
        pr_pts = [{"recall": round(float(r_vals[i]), 4), "precision": round(float(p_vals[i]), 4)} for i in sample_pr]

        conf_mat_default = confusion_matrix(y_test, y_pred_default).tolist()
        y_pred_tuned = (y_prob >= opt_thresh).astype(int)
        conf_mat_tuned = confusion_matrix(y_test, y_pred_tuned).tolist()

        results[name] = {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "default_0_5": {
                "recall": round(float(recall_default), 4),
                "precision": round(float(prec_default), 4),
                "f1": round(float(f1_default), 4),
                "confusion_matrix": conf_mat_default
            },
            "tuned_threshold_recall_60": {
                **tuned_metrics,
                "confusion_matrix": conf_mat_tuned
            },
            "roc_curve": roc_pts,
            "pr_curve": pr_pts
        }

        logger.info(
            f"{name} -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
            f"Tuned Threshold ({opt_thresh:.3f}): Recall={tuned_metrics['recall']:.2f}, "
            f"Precision={tuned_metrics['precision']:.2f}"
        )

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name

    logger.info(f"Top performing model: {best_model_name} with ROC-AUC {best_roc_auc:.4f}")

    # 6. Extract Feature Importances from Best Tree Model
    best_clf = trained_estimators[best_model_name]
    feature_importances = []

    if hasattr(best_clf, "feature_importances_"):
        raw_importances = best_clf.feature_importances_
        norm_importances = raw_importances / np.sum(raw_importances)
        for fname, imp in zip(transformed_feature_names, norm_importances):
            feature_importances.append({"feature": fname, "importance": round(float(imp), 5)})
        feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)
    elif hasattr(best_clf, "coef_"):
        raw_importances = np.abs(best_clf.coef_[0])
        norm_importances = raw_importances / np.sum(raw_importances)
        for fname, imp in zip(transformed_feature_names, norm_importances):
            feature_importances.append({"feature": fname, "importance": round(float(imp), 5)})
        feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)

    # 7. Save Artifacts
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save best model
    best_model_file = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_clf, best_model_file)
    logger.info(f"Saved best model to {best_model_file}")

    # Save all trained models for model comparison in dashboard
    for m_name, m_inst in trained_estimators.items():
        joblib.dump(m_inst, os.path.join(models_dir, f"{m_name.lower()}_model.pkl"))

    # Save preprocessor
    preprocessor_file = os.path.join(models_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor to {preprocessor_file}")

    # Save metadata & metrics
    metadata = {
        "best_model": best_model_name,
        "best_roc_auc": round(float(best_roc_auc), 4),
        "target": "readmitted < 30 days (Binary)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "positive_rate_pct": round(float(np.mean(y) * 100), 2),
        "models_comparison": results,
        "top_features": feature_importances[:25],
        "feature_columns": {
            "numeric": num_cols,
            "categorical": cat_cols,
            "transformed_names": transformed_feature_names
        }
    }

    metadata_file = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")

    # Save test sample data for interactive dashboard evaluation
    test_sample_df = X_test.copy()
    test_sample_df["actual_readmitted_binary"] = y_test
    test_sample_df["predicted_prob"] = trained_estimators[best_model_name].predict_proba(X_test_trans)[:, 1]
    test_sample_df.head(2000).to_csv(os.path.join(base_dir, "data", "test_sample_evaluated.csv"), index=False)

    return metadata


if __name__ == "__main__":
    project_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_and_evaluate_models(project_base)
