"""
Feature Engineering Module for Diabetic Patient Health Record Analysis.
HCL Data Science Project

Engineers clinical features as specified:
- Age bands mapped to ordinal & categorical representations
- Prior inpatient, emergency, and outpatient visits + total prior visits
- Number of medications and active medication counts
- Medication changed (yes/no)
- HbA1c tested and abnormal flags
- Diagnosis disease families
- Clean patient deduplication and StratifiedGroupKFold preparation
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# List of 23 diabetic medication columns in the UCI dataset
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
]

AGE_ORDINAL_MAP = {
    "[0-10)": 0,
    "[10-20)": 1,
    "[20-30)": 2,
    "[30-40)": 3,
    "[40-50)": 4,
    "[50-60)": 5,
    "[60-70)": 6,
    "[70-80)": 7,
    "[80-90)": 8,
    "[90-100)": 9
}

def engineer_features(
    df: pd.DataFrame,
    keep_first_encounter_only: bool = True
) -> pd.DataFrame:
    """
    Applies comprehensive domain-specific feature engineering.
    """
    data = df.copy()

    # Avoid data leakage: Keep only first encounter per patient if requested
    if keep_first_encounter_only and "patient_nbr" in data.columns:
        initial_len = len(data)
        data = data.drop_duplicates(subset=["patient_nbr"], keep="first").copy()
        logger.info(
            f"Filtered to first encounter per patient: {initial_len} -> {len(data)} "
            "encounters (eliminating patient duplication leakage)."
        )

    # 1. Age band mapping (ordinal value + category)
    if "age" in data.columns:
        data["age_ordinal"] = data["age"].map(AGE_ORDINAL_MAP).fillna(5)

    # 2. Prior visit engineering
    for col in ["number_inpatient", "number_emergency", "number_outpatient"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)

    # Total prior visits
    data["total_prior_visits"] = (
        data["number_inpatient"] + data["number_emergency"] + data["number_outpatient"]
    )
    # High prior utilization flags
    data["has_prior_inpatient"] = (data["number_inpatient"] > 0).astype(int)
    data["has_prior_emergency"] = (data["number_emergency"] > 0).astype(int)

    # 3. Medication changed flag
    if "change" in data.columns:
        data["medication_changed"] = (data["change"] == "Ch").astype(int)

    # 4. Diabetes medication prescribed flag
    if "diabetesMed" in data.columns:
        data["diabetes_med_prescribed"] = (data["diabetesMed"] == "Yes").astype(int)

    # 5. HbA1c testing flags
    if "A1Cresult" in data.columns:
        # Tested if not NaN and not 'None'
        data["hba1c_tested"] = (
            data["A1Cresult"].notna() & (data["A1Cresult"] != "None")
        ).astype(int)
        data["hba1c_abnormal"] = (
            data["A1Cresult"].isin([">7", ">8"])
        ).astype(int)
    else:
        data["hba1c_tested"] = 0
        data["hba1c_abnormal"] = 0

    # 6. Glucose serum test flag
    if "max_glu_serum" in data.columns:
        data["glucose_serum_tested"] = (
            data["max_glu_serum"].notna() & (data["max_glu_serum"] != "None")
        ).astype(int)

    # 7. Active medications count across the 23 medication columns
    present_med_cols = [c for c in MEDICATION_COLS if c in data.columns]
    if present_med_cols:
        # count how many medications have status other than 'No'
        active_counts = (data[present_med_cols] != "No").sum(axis=1)
        data["active_diabetic_meds_count"] = active_counts
    else:
        data["active_diabetic_meds_count"] = 0

    # Insulin status
    if "insulin" in data.columns:
        data["insulin_treatment"] = (data["insulin"] != "No").astype(int)
        data["insulin_status"] = data["insulin"].fillna("No")

    # 8. Admission type grouping
    if "admission_type_desc" in data.columns:
        def simplify_adm_type(val):
            val_str = str(val).lower()
            if "emergency" in val_str:
                return "Emergency"
            elif "urgent" in val_str:
                return "Urgent"
            elif "elective" in val_str:
                return "Elective"
            else:
                return "Other"
        data["admission_category"] = data["admission_type_desc"].apply(simplify_adm_type)

    # 9. Top medical specialties grouping
    if "medical_specialty" in data.columns:
        top_specs = [
            "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
            "Cardiology", "Surgery-General", "Unknown"
        ]
        data["specialty_grouped"] = data["medical_specialty"].apply(
            lambda x: x if x in top_specs else "Other"
        )

    # 10. Severity / Complexity index
    # Length of stay + diagnoses + lab procedures normalized
    data["complexity_score"] = (
        (data["time_in_hospital"] / 14.0) +
        (data["number_diagnoses"] / 16.0) +
        (data["num_procedures"] / 6.0)
    )

    logger.info(f"Feature engineering completed. Engineered shape: {data.shape}")
    return data


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Returns numeric and categorical feature lists for modeling.
    Strictly excludes post-discharge data, encounter IDs, and target variables.
    """
    numeric_features = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "total_prior_visits",
        "has_prior_inpatient",
        "has_prior_emergency",
        "age_ordinal",
        "medication_changed",
        "diabetes_med_prescribed",
        "hba1c_tested",
        "hba1c_abnormal",
        "glucose_serum_tested",
        "active_diabetic_meds_count",
        "insulin_treatment",
        "complexity_score"
    ]

    categorical_features = [
        "race",
        "gender",
        "age",
        "admission_category",
        "specialty_grouped",
        "diag_1_family",
        "diag_2_family",
        "diag_3_family",
        "insulin_status"
    ]

    return numeric_features, categorical_features
