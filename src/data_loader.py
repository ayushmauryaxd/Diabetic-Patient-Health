"""
Data Loader Module for Diabetic Patient Health Record Analysis.
HCL Data Science Project
"""

import os
import csv
import logging
import pandas as pd
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_ids_mapping(filepath: str) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    """
    Parses IDS_mapping.csv which contains 3 sections:
    1. admission_type_id,description
    2. discharge_disposition_id,description
    3. admission_source_id,description
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"IDS mapping file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    admission_type_map = {}
    discharge_disposition_map = {}
    admission_source_map = {}

    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line == ",":
            continue

        if "admission_type_id" in line:
            current_section = "admission_type"
            continue
        elif "discharge_disposition_id" in line:
            current_section = "discharge_disposition"
            continue
        elif "admission_source_id" in line:
            current_section = "admission_source"
            continue

        parts = list(csv.reader([line]))[0]
        if len(parts) >= 2:
            id_str = parts[0].strip()
            desc = parts[1].strip()
            if id_str.isdigit():
                id_val = int(id_str)
                if current_section == "admission_type":
                    admission_type_map[id_val] = desc
                elif current_section == "discharge_disposition":
                    discharge_disposition_map[id_val] = desc
                elif current_section == "admission_source":
                    admission_source_map[id_val] = desc

    logger.info(
        f"Parsed IDS_mapping: {len(admission_type_map)} admission types, "
        f"{len(discharge_disposition_map)} discharge dispositions, "
        f"{len(admission_source_map)} admission sources"
    )
    return admission_type_map, discharge_disposition_map, admission_source_map


def load_raw_data(
    data_path: str,
    ids_mapping_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads diabetic_data.csv and optionally merges descriptions from IDS_mapping.csv.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    logger.info(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Raw dataset loaded with shape: {df.shape}")

    if ids_mapping_path and os.path.exists(ids_mapping_path):
        adm_type_map, disch_disp_map, adm_src_map = parse_ids_mapping(ids_mapping_path)
        
        df["admission_type_desc"] = df["admission_type_id"].map(adm_type_map).fillna("Unknown/Other")
        df["discharge_disposition_desc"] = df["discharge_disposition_id"].map(disch_disp_map).fillna("Unknown/Other")
        df["admission_source_desc"] = df["admission_source_id"].map(adm_src_map).fillna("Unknown/Other")
        logger.info("Mapped code columns (admission_type_id, discharge_disposition_id, admission_source_id) to descriptive labels.")

    return df


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "data", "diabetic_data.csv")
    mapping_file = os.path.join(base_dir, "data", "IDS_mapping.csv")
    
    data = load_raw_data(data_file, mapping_file)
    print("Sample mapped rows:")
    print(data[["admission_type_desc", "discharge_disposition_desc", "admission_source_desc"]].head())
