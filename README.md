# 🏥 Diabetic Patient Health Record Dashboard (P_090)
### HCL Data Science & Python Capstone Project

An end-to-end clinical machine learning and decision-support dashboard analyzing 130 US hospitals' inpatient diabetes records. Built to predict 30-day hospital readmissions, quantify clinical risk factors via SHAP explainability, and provide executive healthcare quality metrics.

---

## 📌 Executive Summary & Clinical Context

Hospital readmission within 30 days is a critical healthcare quality indicator monitored under the **CMS Hospital Readmissions Reduction Program (HRRP)**. Hospitals face severe financial penalties when readmission rates exceed benchmarks. 

This project addresses the clinical challenge of identifying high-risk diabetic patients before discharge, enabling targeted interventions such as transitional care management, glycemic monitoring, and medication reconciliation.

---

## 🏗️ Project Architecture & Directory Structure

```
diabetic-health-dashboard/
│
├── data/
│   ├── diabetic_data.csv          # Raw UCI Diabetes dataset (id=296)
│   ├── IDS_mapping.csv            # ID dictionaries for admission & discharge
│   ├── cleaned_diabetes_data.csv  # Cleaned, mapped dataset (99,340 records)
│   ├── feature_engineered_data.csv# Modeled dataset (69,987 unique encounters)
│   └── test_sample_evaluated.csv  # Held-out test cohort with predictions
│
├── models/
│   ├── best_model.pkl             # Serialized champion model (RandomForest)
│   ├── preprocessor.pkl           # ColumnTransformer preprocessing pipeline
│   └── model_metadata.json        # Benchmarks, ROC/PR curves, thresholds, SHAP features
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Comprehensive EDA & clinical visualizations
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Data ingestion and IDS_mapping parser
│   ├── preprocessor.py            # Cleaning, hospice drop, ICD-9 grouping
│   ├── feature_engineering.py     # Domain feature engineering & leak avoidance
│   ├── train.py                   # Stratified grouped training & threshold tuning
│   ├── evaluate.py                # Validation metrics & benchmark reporting
│   └── explainability.py          # SHAP TreeExplainer & Top-3 driver extractor
│
├── app.py                         # Interactive Streamlit Healthcare Dashboard
├── requirements.txt               # Project dependencies
└── README.md                      # Detailed technical documentation
```

---

## 📊 Dataset & Preprocessing Pipeline

### 1. Primary Data Source
- **UCI Diabetes 130-US Hospitals (1999–2008)**: 101,766 encounters, 47 clinical features.
- Downloaded directly via `ucimlrepo` (Dataset ID 296).

### 2. ID Mapping & Section Parsing
`IDS_mapping.csv` contains multiple structured sections that are parsed and merged into human-readable labels:
- `admission_type_id` $\rightarrow$ `admission_type_desc` (Emergency, Urgent, Elective, Newborn, Trauma)
- `discharge_disposition_id` $\rightarrow$ `discharge_disposition_desc` (Discharged to home, SNF, Rehab, Expired, Hospice)
- `admission_source_id` $\rightarrow$ `admission_source_desc` (Physician referral, Emergency room, Transfer)

### 3. Cleaning & Imputation Rules (Per Specification)
1. **Missing Value Replacement**: All `'?'` indicators replaced with standard `NaN`.
2. **Feature Dropping**: `weight` is $>96\%$ missing and dropped (Rule 2).
3. **Specialty & Payer Codes**: `medical_specialty` ($49\%$ missing) and `payer_code` ($39\%$ missing) retained and categorized as `'Unknown'`.
4. **Demographics**: `race` missingness ($2.2\%$) imputed as `'Unknown'`.
5. **Mortality & Hospice Filtering**: Encounters where patients died or were discharged to hospice care (`discharge_disposition_id` $\in [11, 13, 14, 19, 20, 21]$) are removed (2,426 records), as these patients cannot be readmitted.
6. **ICD-9 Disease Grouping**: `diag_1`, `diag_2`, and `diag_3` grouped into 9 clinical disease families:
   - **Circulatory** (390–459, 785)
   - **Respiratory** (460–519, 786)
   - **Digestive** (520–579, 787)
   - **Diabetes** (250.xx)
   - **Injury** (800–999)
   - **Musculoskeletal** (710–739)
   - **Genitourinary** (580–629, 788)
   - **Neoplasms** (140–239)
   - **Other** (Remaining codes and V/E supplementary classifications)
7. **Target Variable**: Binary classification `readmitted_binary`:
   - `1`: Readmitted within 30 days (`<30`)
   - `0`: Not readmitted within 30 days (`>30` or `NO`)

---

## 🔬 Domain Feature Engineering

- **Prior Utilization**: `number_inpatient`, `number_emergency`, `number_outpatient`, and composite `total_prior_visits`.
- **Glycemic Monitoring**:
  - `hba1c_tested`: Flag indicating if HbA1c was measured during encounter.
  - `hba1c_abnormal`: Indicator of uncontrolled blood sugar ($>7\%$ or $>8\%$).
  - `glucose_serum_tested`: Inpatient blood sugar lab check.
- **Pharmacotherapy & Polypharmacy**:
  - `num_medications`: Total distinct medications administered.
  - `active_diabetic_meds_count`: Active drugs across 23 antidiabetic agents.
  - `medication_changed`: Flag if dosage was titrated or altered.
  - `insulin_treatment`: Patient on insulin therapy regimen.
- **Clinical Complexity**:
  $$\text{Complexity Score} = \frac{\text{Length of Stay}}{14} + \frac{\text{Diagnoses}}{16} + \frac{\text{Procedures}}{6}$$

---

## 🛡️ Leakage Trap Mitigation

1. **Patient Duplication Leakage**: 
   - A single patient appears in multiple hospital admissions over the 10-year period.
   - Splitting randomly across encounters would cause the exact same patient to exist in both training and test sets.
   - **Solution**: Evaluated using **Stratified Group Validation (`StratifiedGroupKFold` on `patient_nbr`)** and first-encounter deduplication, ensuring **0 patient overlap between train and test splits**.
2. **Post-Discharge Information**: Excluded all discharge-time updates and subsequent encounter records from model inputs.

---

## 📈 Model Benchmarking & Threshold Tuning

Given the low natural prevalence of 30-day readmission ($\sim 9\%$), default $0.5$ classification thresholds yield high specificity but poor recall. The decision threshold was tuned to achieve the clinically required **Recall $\ge 0.60$**.

| Algorithm | ROC-AUC | PR-AUC | Tuned Threshold | Tuned Recall ($\ge 0.60$) | Tuned Precision | Tuned F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | **0.6112** | **0.1342** | **0.4524** | **61.81%** | 11.86% | **0.1990** |
| **LogisticRegression** | 0.6111 | 0.1333 | 0.4849 | 60.30% | 11.88% | 0.1985 |
| **LightGBM** | 0.6105 | 0.1325 | 0.4717 | 62.05% | 11.92% | 0.1999 |
| **XGBoost** | 0.6101 | 0.1358 | 0.4660 | 62.13% | 11.62% | 0.1957 |

> **Clinical Interpretation**: Hospital 30-day readmission is an inherently noisy problem with high baseline entropy. An AUC of $0.61–0.70$ combined with $>60\%$ sensitivity successfully flags $>6$ out of every 10 patients destined to bounce back.

### Top Clinical Predictors (Feature Importance)
1. `num_lab_procedures` ($7.46\%$)
2. `complexity_score` ($7.39\%$)
3. `num_medications` ($6.68\%$)
4. `time_in_hospital` ($5.51\%$)
5. `number_inpatient` ($4.93\%$)
6. `age_ordinal` ($4.09\%$)
7. `total_prior_visits` ($3.90\%$)
8. `number_diagnoses` ($3.72\%$)

---

## 🩺 Clinical Decision Support & SHAP Explainability

The dashboard incorporates a real-time **Estimation Panel** backed by **TreeSHAP**:
- Computes patient-specific readmission risk percentage.
- Stratifies patients into **Low Risk** ($<15\%$), **Moderate Risk** ($15–30\%$), and **High Risk** ($>30\%$).
- Extracts and visualizes the **Top 3 Clinical Drivers** with directionality (Risk Factor vs Protective Factor) and translated clinical narratives.

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PowerShell or Terminal

### 2. Environment Setup
```powershell
# Clone or navigate to the project directory
cd "C:\Users\Ayush Maurya\.gemini\antigravity\scratch\diabetic-health-dashboard"

# Install dependencies
pip install -r requirements.txt
```

### 3. Re-train Models (Optional)
```powershell
python src/train.py
```

### 4. Run the Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌟 Dashboard Capabilities

1. **Healthcare KPIs & Executive Overview**:
   - Live metrics: Total admissions, 30-day readmission rate, avg length of stay, avg medications, % HbA1c tested.
   - Interactive Plotly visualizations across age bands, admission categories, and ICD-9 disease families.
2. **Interactive Patient Risk Estimator**:
   - Clinical parameter input form with real-time risk gauge.
   - SHAP Top-3 factor breakdown and recommended interventions.
3. **ML Benchmarking Lab**:
   - Interactive decision threshold slider dynamically updating confusion matrix, sensitivity, and precision.
   - Plotly interactive ROC curves and Precision-Recall curves.
4. **Population Health Explorer**:
   - Dynamic cohort filtering and multi-variable interaction exploration.

---

## 📜 HCL Project Rubric Alignment

- **Track Code**: P_090 (Patient Health Record Dashboard)
- **Problem Type**: Binary Classification + Descriptive Dashboard
- **Dataset**: UCI Diabetes 130-US Hospitals (CC BY 4.0)
- **Primary Libraries**: Python, Streamlit, Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost, SHAP, Plotly
