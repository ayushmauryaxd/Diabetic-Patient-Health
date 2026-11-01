"""
Pipeline Validation & Unit Tests
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)

from src.preprocessor import categorize_icd9, clean_diabetic_data
from src.feature_engineering import engineer_features, get_feature_columns
from src.explainability import PatientRiskExplainer

class TestDiabeticPipeline(unittest.TestCase):

    def test_icd9_categorization(self):
        self.assertEqual(categorize_icd9("250.02"), "Diabetes")
        self.assertEqual(categorize_icd9("414"), "Circulatory")
        self.assertEqual(categorize_icd9("486"), "Respiratory")
        self.assertEqual(categorize_icd9("530"), "Digestive")
        self.assertEqual(categorize_icd9("585"), "Genitourinary")
        self.assertEqual(categorize_icd9("715"), "Musculoskeletal")
        self.assertEqual(categorize_icd9("174"), "Neoplasms")
        self.assertEqual(categorize_icd9("820"), "Injury")
        self.assertEqual(categorize_icd9("V45"), "Other")
        self.assertEqual(categorize_icd9("?"), "Missing/Unknown")

    def test_model_and_explainer(self):
        model_path = os.path.join(base_dir, "models", "best_model.pkl")
        prep_path = os.path.join(base_dir, "models", "preprocessor.pkl")
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(prep_path))

        explainer = PatientRiskExplainer(model_path, prep_path)
        test_df = pd.read_csv(os.path.join(base_dir, "data", "test_sample_evaluated.csv"), nrows=2)
        res = explainer.predict_and_explain(test_df.iloc[[0]])
        self.assertIn("readmission_risk_pct", res)
        self.assertIn("top_drivers", res)
        self.assertEqual(len(res["top_drivers"]), 3)

if __name__ == "__main__":
    unittest.main()
