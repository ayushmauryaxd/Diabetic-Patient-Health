"""
Data Preprocessing Module for Diabetic Patient Health Record Analysis.
HCL Data Science Project

Implements data cleaning and transformation procedures per specifications:
1. '?' -> NaN
2. weight dropped (almost entirely missing)
3. medical_specialty, payer_code heavily missing -> 'Unknown' category, keep
4. race -> 'Unknown' for missing values
5. Remove encounters where patient died or went to hospice (discharge_disposition_id in [11, 13, 14, 19, 20, 21])
6. Group ICD-9 diagnosis codes into standard 9 disease families
7. Binary classification target: readmitted < 30 days (1: Yes, 0: No)
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Discharge dispositions corresponding to Expired or Hospice care
EXPIRED_HOSPICE_IDS = [11, 13, 14, 19, 20, 21]

def categorize_icd9(val: Optional[str]) -> str:
    """
    Groups ICD-9 diagnostic codes into ~9 clinical disease families:
    1. Circulatory: 390-459, 785
    2. Respiratory: 460-519, 786
    3. Digestive: 520-579, 787
    4. Diabetes: 250.xx
    5. Injury: 800-999
    6. Musculoskeletal: 710-739
    7. Genitourinary: 580-629, 788
    8. Neoplasms: 140-239
    9. Other: 780-782, 784, 790-799, 240-279 (except 250), 680-709, 001-139, V/E codes
    """
    if pd.isna(val) or val is None or str(val).strip() in ["?", "", "nan"]:
        return "Missing/Unknown"
    
    val_str = str(val).strip()
    
    # Diabetes specific
    if val_str.startswith("250"):
        return "Diabetes"
    
    # Supplementary classifications (V and E codes)
    if val_str.startswith(("V", "E")):
        return "Other"
    
    try:
        code = float(val_str)
    except ValueError:
        return "Other"
        
    if (390 <= code <= 459) or code == 785:
        return "Circulatory"
    elif (460 <= code <= 519) or code == 786:
        return "Respiratory"
    elif (520 <= code <= 579) or code == 787:
        return "Digestive"
    elif (580 <= code <= 629) or code == 788:
        return "Genitourinary"
    elif 710 <= code <= 739:
        return "Musculoskeletal"
    elif 140 <= code <= 239:
        return "Neoplasms"
    elif 800 <= code <= 999:
        return "Injury"
    else:
        return "Other"


def clean_diabetic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the formal cleaning and imputation pipeline.
    """
    logger.info(f"Starting data cleaning. Initial records: {len(df)}")
    data = df.copy()

    # Rule 1: Replace '?' with NaN across all object columns
    data = data.replace("?", np.nan)

    # Rule 2: weight is almost entirely empty (>96% missing) -> drop
    if "weight" in data.columns:
        logger.info("Dropping 'weight' column due to severe missingness.")
        data = data.drop(columns=["weight"])

    # Rule 3: medical_specialty, payer_code heavily missing -> 'Unknown' category, keep
    if "medical_specialty" in data.columns:
        data["medical_specialty"] = data["medical_specialty"].fillna("Unknown")
    if "payer_code" in data.columns:
        data["payer_code"] = data["payer_code"].fillna("Unknown")

    # Rule 4: race -> 'Unknown'
    if "race" in data.columns:
        data["race"] = data["race"].fillna("Unknown")

    # Filter out invalid genders (only 3 encounters with 'Unknown/Invalid')
    if "gender" in data.columns:
        valid_genders = data["gender"].isin(["Male", "Female"])
        data = data[valid_genders].copy()

    # Rule 5: Remove encounters where the patient died or went to hospice
    # (these patients cannot be readmitted within 30 days)
    if "discharge_disposition_id" in data.columns:
        before_count = len(data)
        data = data[~data["discharge_disposition_id"].isin(EXPIRED_HOSPICE_IDS)].copy()
        logger.info(
            f"Removed {before_count - len(data)} encounters where patient died or entered hospice."
        )

    # Rule 6: Group ICD-9 diagnosis codes into disease families
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in data.columns:
            group_col_name = f"{diag_col}_family"
            data[group_col_name] = data[diag_col].apply(categorize_icd9)
            logger.info(f"Grouped {diag_col} into {group_col_name}")

    # Rule 7: Target encoding: Binary classification (readmitted < 30 days: yes/no)
    if "readmitted" in data.columns:
        data["readmitted_binary"] = (data["readmitted"] == "<30").astype(int)
        logger.info(
            f"Binary target created: {data['readmitted_binary'].sum()} positive cases "
            f"({data['readmitted_binary'].mean():.2%})"
        )

    logger.info(f"Cleaning complete. Remaining records: {len(data)}")
    return data


if __name__ == "__main__":
    import os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    from data_loader import load_raw_data

    raw = load_raw_data(
        os.path.join(base_dir, "data", "diabetic_data.csv"),
        os.path.join(base_dir, "data", "IDS_mapping.csv"),
    )
    cleaned = clean_diabetic_data(raw)
    out_path = os.path.join(base_dir, "data", "cleaned_diabetes_data.csv")
    cleaned.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path} with shape: {cleaned.shape}")
