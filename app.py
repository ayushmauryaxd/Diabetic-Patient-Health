"""
Diabetic Patient Health Record Dashboard - Streamlit Application
HCL Data Science Project (P_090: Patient Health Record Dashboard)

A clinical decision support & executive analytics platform:
1. Executive Healthcare KPIs
2. Population Risk & Clinical Visualizations
3. Real-Time Patient Risk Estimator with SHAP Top-3 Drivers
4. ML Model Benchmarking Lab & Dynamic Threshold Tuning
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explainability import PatientRiskExplainer
from src.feature_engineering import AGE_ORDINAL_MAP, get_feature_columns

# Page configuration
st.set_page_config(
    page_title="Diabetic Patient Health Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-top: 4px solid #2563eb;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        margin-top: 0.3rem;
    }
    .kpi-delta {
        font-size: 0.85rem;
        color: #059669;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .driver-card-risk {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .driver-card-protect {
        background: #f0fdf4;
        border-left: 5px solid #10b981;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_project_data():
    """Loads cleaned data and sample evaluation data."""
    cleaned_file = os.path.join(PROJECT_ROOT, "data", "cleaned_diabetes_data.csv")
    meta_file = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    
    if os.path.exists(cleaned_file):
        df_clean = pd.read_csv(cleaned_file, low_memory=False)
    else:
        df_clean = pd.DataFrame()

    metadata = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return df_clean, metadata


@st.cache_resource
def load_ml_pipeline():
    """Loads model and explainer."""
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
    prep_path = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")
    if os.path.exists(model_path) and os.path.exists(prep_path):
        explainer = PatientRiskExplainer(model_path, prep_path)
        return explainer
    return None


df_clean, metadata = load_project_data()
explainer = load_ml_pipeline()

# Title banner
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.markdown('<div class="main-title">🏥 Diabetic Patient Health Record Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">HCL Data Science Analytics • 30-Day Hospital Readmission Risk Prediction & Decision Support</div>', unsafe_allow_html=True)
with col_head2:
    st.markdown("""
        <div style="text-align: right; padding-top: 10px;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                HCL Healthcare AI
            </span>
        </div>
    """, unsafe_allow_html=True)

# Navigation tabs
tab_kpis, tab_estimator, tab_models, tab_eda = st.tabs([
    "📊 Healthcare KPIs & Executive Overview",
    "🩺 Patient Risk Estimator (SHAP Explainer)",
    "🧪 Model Benchmarking & Lab",
    "🔍 Clinical Population Insights"
])


# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW & HEALTHCARE KPIS
# ==============================================================================
with tab_kpis:
    st.markdown("### Executive Performance Metrics")
    st.caption("Standard hospital quality indicators derived from 99,340 acute diabetic inpatient encounters (post-hospice filtering).")

    if not df_clean.empty:
        total_admissions = len(df_clean)
        readmit_30d_rate = (df_clean["readmitted_binary"].mean()) * 100
        avg_los = df_clean["time_in_hospital"].mean()
        avg_meds = df_clean["num_medications"].mean()
        hba1c_tested_pct = ((df_clean["A1Cresult"].notna()) & (df_clean["A1Cresult"] != "None")).mean() * 100

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Encounters</div>
                <div class="kpi-value">{total_admissions:,}</div>
                <div class="kpi-delta">130 US Hospitals</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #dc2626;">
                <div class="kpi-title">30-Day Readmission Rate</div>
                <div class="kpi-value">{readmit_30d_rate:.1f}%</div>
                <div class="kpi-delta" style="color: #dc2626;">Target: &lt; 9.5%</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #059669;">
                <div class="kpi-title">Avg Length of Stay</div>
                <div class="kpi-value">{avg_los:.1f} <span style="font-size: 1rem; color: #6b7280;">days</span></div>
                <div class="kpi-delta">Inpatient duration</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #d97706;">
                <div class="kpi-title">Avg Medications</div>
                <div class="kpi-value">{avg_meds:.1f}</div>
                <div class="kpi-delta">Active drugs / stay</div>
            </div>
            """, unsafe_allow_html=True)

        with k5:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #7c3aed;">
                <div class="kpi-title">HbA1c Tested %</div>
                <div class="kpi-value">{hba1c_tested_pct:.1f}%</div>
                <div class="kpi-delta">Standard of Care</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Primary KPI visual breakdowns
        c1, c2 = st.columns(2)
        
        with c1:
            # Readmission rate by age band
            age_rates = df_clean.groupby("age")["readmitted_binary"].mean().reset_index()
            age_rates["readmission_pct"] = age_rates["readmitted_binary"] * 100
            
            fig_age = px.bar(
                age_rates,
                x="age",
                y="readmission_pct",
                title="<b>30-Day Readmission Rate by Age Band</b>",
                labels={"age": "Age Cohort", "readmission_pct": "Readmission Rate (%)"},
                color="readmission_pct",
                color_continuous_scale="Blues"
            )
            fig_age.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)

        with c2:
            # Readmission rate by admission type
            adm_rates = df_clean.groupby("admission_type_desc")["readmitted_binary"].agg(["count", "mean"]).reset_index()
            adm_rates = adm_rates[adm_rates["count"] > 100]
            adm_rates["readmission_pct"] = adm_rates["mean"] * 100
            adm_rates = adm_rates.sort_values("readmission_pct", ascending=True)

            fig_adm = px.bar(
                adm_rates,
                y="admission_type_desc",
                x="readmission_pct",
                orientation="h",
                title="<b>30-Day Readmission Rate by Admission Type</b>",
                labels={"admission_type_desc": "Admission Category", "readmission_pct": "Readmission Rate (%)"},
                color="readmission_pct",
                color_continuous_scale="Reds"
            )
            fig_adm.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_adm, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # Readmission rate by diagnosis disease family
            diag_rates = df_clean.groupby("diag_1_family")["readmitted_binary"].mean().reset_index()
            diag_rates["readmission_pct"] = diag_rates["readmitted_binary"] * 100
            diag_rates = diag_rates.sort_values("readmission_pct", ascending=False)

            fig_diag = px.bar(
                diag_rates,
                x="diag_1_family",
                y="readmission_pct",
                title="<b>Readmission Rate Across ICD-9 Primary Disease Families</b>",
                labels={"diag_1_family": "ICD-9 Disease Group", "readmission_pct": "Readmission Rate (%)"},
                color="readmission_pct",
                color_continuous_scale="Purples"
            )
            fig_diag.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_diag, use_container_width=True)

        with c4:
            # HbA1c testing impact
            hba1c_rates = df_clean.copy()
            hba1c_rates["A1C_Status"] = hba1c_rates["A1Cresult"].fillna("Not Tested")
            hba1c_summary = hba1c_rates.groupby("A1C_Status")["readmitted_binary"].agg(["count", "mean"]).reset_index()
            hba1c_summary["readmission_pct"] = hba1c_summary["mean"] * 100

            fig_hba1c = px.pie(
                hba1c_summary,
                names="A1C_Status",
                values="count",
                title="<b>HbA1c Blood Testing Coverage in Hospital</b>",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_hba1c.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hba1c, use_container_width=True)


# ==============================================================================
# TAB 2: ESTIMATION PANEL (SHAP EXPLAINABILITY)
# ==============================================================================
with tab_estimator:
    st.markdown("### 🩺 Patient 30-Day Readmission Risk Estimator")
    st.caption("Enter patient clinical parameters to calculate individualized risk probability and extract the **Top-3 Clinical Drivers (SHAP Explanations)**.")

    if explainer is None:
        st.warning("Model and Preprocessor pipeline artifacts not found. Please train models first using `python src/train.py`.")
    else:
        with st.form("patient_risk_form"):
            col_p1, col_p2, col_p3 = st.columns(3)
            
            with col_p1:
                st.markdown("##### 👤 Demographics & Stays")
                age_val = st.selectbox("Age Band", list(AGE_ORDINAL_MAP.keys()), index=6) # [60-70)
                gender_val = st.selectbox("Gender", ["Female", "Male"], index=0)
                race_val = st.selectbox("Race / Ethnicity", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"], index=0)
                time_in_hosp = st.slider("Length of Hospital Stay (Days)", 1, 14, 4)
                admission_cat = st.selectbox("Admission Category", ["Emergency", "Urgent", "Elective", "Other"], index=0)

            with col_p2:
                st.markdown("##### 🧪 Clinical Labs & Procedures")
                num_lab = st.number_input("Number of Lab Procedures", min_value=1, max_value=132, value=45)
                num_proc = st.number_input("Number of Non-Lab Procedures", min_value=0, max_value=6, value=1)
                num_diag = st.number_input("Total Diagnoses Recorded", min_value=1, max_value=16, value=8)
                diag1_fam = st.selectbox("Primary Diagnosis Family", [
                    "Circulatory", "Diabetes", "Respiratory", "Digestive",
                    "Genitourinary", "Musculoskeletal", "Injury", "Neoplasms", "Other"
                ], index=0)
                diag2_fam = st.selectbox("Secondary Diagnosis Family", [
                    "Diabetes", "Circulatory", "Respiratory", "Digestive",
                    "Genitourinary", "Musculoskeletal", "Injury", "Neoplasms", "Other"
                ], index=0)
                diag3_fam = st.selectbox("Tertiary Diagnosis Family", [
                    "Diabetes", "Circulatory", "Respiratory", "Digestive",
                    "Genitourinary", "Musculoskeletal", "Injury", "Neoplasms", "Other"
                ], index=0)

            with col_p3:
                st.markdown("##### 💊 Medications & Prior Utilization")
                num_meds = st.slider("Total Prescribed Medications", 1, 60, 16)
                med_changed = st.checkbox("Medication Dosage Changed During Stay", value=True)
                diabetes_prescribed = st.checkbox("Diabetes Medication Prescribed", value=True)
                insulin_opt = st.selectbox("Insulin Status", ["No", "Steady", "Up", "Down"], index=1)
                hba1c_res = st.selectbox("HbA1c Blood Test Result", ["None", "Norm", ">7", ">8"], index=0)
                
                prior_inpatient = st.number_input("Prior Inpatient Admissions (Past Year)", min_value=0, max_value=20, value=0)
                prior_emergency = st.number_input("Prior Emergency Room Visits (Past Year)", min_value=0, max_value=25, value=0)
                prior_outpatient = st.number_input("Prior Outpatient Visits (Past Year)", min_value=0, max_value=30, value=0)

            submit_btn = st.form_submit_button("⚡ Predict Readmission Risk & Calculate SHAP Drivers", use_container_width=True)

        if submit_btn:
            # Build patient record DataFrame matching feature engineering inputs
            total_prior = prior_inpatient + prior_emergency + prior_outpatient
            hba1c_test_flag = 1 if hba1c_res != "None" else 0
            hba1c_abn_flag = 1 if hba1c_res in [">7", ">8"] else 0
            ins_treat = 1 if insulin_opt != "No" else 0
            comp_score = (time_in_hosp / 14.0) + (num_diag / 16.0) + (num_proc / 6.0)

            patient_dict = {
                "time_in_hospital": [time_in_hosp],
                "num_lab_procedures": [num_lab],
                "num_procedures": [num_proc],
                "num_medications": [num_meds],
                "number_outpatient": [prior_outpatient],
                "number_emergency": [prior_emergency],
                "number_inpatient": [prior_inpatient],
                "number_diagnoses": [num_diag],
                "total_prior_visits": [total_prior],
                "has_prior_inpatient": [1 if prior_inpatient > 0 else 0],
                "has_prior_emergency": [1 if prior_emergency > 0 else 0],
                "age_ordinal": [AGE_ORDINAL_MAP[age_val]],
                "medication_changed": [1 if med_changed else 0],
                "diabetes_med_prescribed": [1 if diabetes_prescribed else 0],
                "hba1c_tested": [hba1c_test_flag],
                "hba1c_abnormal": [hba1c_abn_flag],
                "glucose_serum_tested": [0],
                "active_diabetic_meds_count": [2 if diabetes_prescribed else 0],
                "insulin_treatment": [ins_treat],
                "complexity_score": [comp_score],
                # Categoricals
                "race": [race_val],
                "gender": [gender_val],
                "age": [age_val],
                "admission_category": [admission_cat],
                "specialty_grouped": ["Unknown"],
                "diag_1_family": [diag1_fam],
                "diag_2_family": [diag2_fam],
                "diag_3_family": [diag3_fam],
                "insulin_status": [insulin_opt]
            }

            p_df = pd.DataFrame(patient_dict)
            explanation = explainer.predict_and_explain(p_df)

            st.markdown("---")
            st.markdown("### 📋 Risk Assessment Results")

            res_c1, res_c2 = st.columns([1.2, 2])
            
            with res_c1:
                risk_pct = explanation["readmission_risk_pct"]
                tier = explanation["risk_tier"]
                
                gauge_color = "#10b981" if risk_pct < 15 else ("#f59e0b" if risk_pct < 30 else "#ef4444")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    number={"suffix": "%"},
                    title={"text": f"<b>{tier}</b><br><span style='font-size:0.8em;color:gray'>30-Day Readmission Risk</span>"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 15], "color": "#f0fdf4"},
                            {"range": [15, 30], "color": "#fefce8"},
                            {"range": [30, 100], "color": "#fef2f2"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 45.2 # optimal tuned threshold
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

                if risk_pct >= 30:
                    st.error("🚨 **High Readmission Alert**: Candidate for aggressive post-discharge transitional care program.")
                elif risk_pct >= 15:
                    st.warning("⚠️ **Moderate Readmission Risk**: Recommend follow-up appointment within 7 days.")
                else:
                    st.success("✅ **Low Readmission Risk**: Standard outpatient transition protocol applies.")

            with res_c2:
                st.markdown("#### 🔍 Top 3 Clinical Risk Drivers (SHAP Value Contribution)")
                st.caption("SHAP (SHapley Additive exPlanations) quantifies the exact mathematical contribution of each patient variable:")

                for idx, driver in enumerate(explanation["top_drivers"], 1):
                    card_class = "driver-card-risk" if driver["direction"] == "Increases Risk" else "driver-card-protect"
                    icon = "🔺" if driver["direction"] == "Increases Risk" else "🛡️"
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div style="font-weight: 700; font-size: 1rem; color: #1f2937;">
                            {icon} #{idx}: {driver['clinical_label']}
                        </div>
                        <div style="font-size: 0.9rem; color: #4b5563; margin-top: 4px;">
                            Impact: <b>{driver['direction']}</b> (SHAP: {driver['impact']:+.4f})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Recommended clinical interventions
                st.markdown("##### 💡 Recommended Clinical Interventions:")
                if prior_inpatient > 0 or total_prior > 1:
                    st.write("• **Frequent Attender Protocol**: Assign dedicated care coordinator to address social determinants & medication access.")
                if hba1c_test_flag == 0:
                    st.write("• **Missing Glycemic Baseline**: Order HbA1c prior to discharge per ADA inpatient guidelines.")
                if num_meds >= 16:
                    st.write("• **Polypharmacy Review**: Pharmacist-led discharge reconciliation recommended.")


# ==============================================================================
# TAB 3: MODEL BENCHMARKING & EVALUATION LAB
# ==============================================================================
with tab_models:
    st.markdown("### 🧪 Machine Learning Model Benchmark & Diagnostic Lab")
    st.caption("Comparison of 4 algorithms evaluated with Stratified Patient-Grouped Validation (0 patient overlap between train/test).")

    if metadata and "models_comparison" in metadata:
        comp_data = []
        for m_name, vals in metadata["models_comparison"].items():
            comp_data.append({
                "Algorithm": m_name,
                "ROC-AUC": vals["roc_auc"],
                "PR-AUC": vals["pr_auc"],
                "Default Thresh Recall": vals["default_0_5"]["recall"],
                "Default Thresh Precision": vals["default_0_5"]["precision"],
                "Tuned Threshold": vals["tuned_threshold_recall_60"]["threshold"],
                "Tuned Recall (Target >= 0.60)": vals["tuned_threshold_recall_60"]["recall"],
                "Tuned Precision": vals["tuned_threshold_recall_60"]["precision"],
                "Tuned F1": vals["tuned_threshold_recall_60"]["f1"]
            })
        
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df.style.highlight_max(subset=["ROC-AUC", "PR-AUC", "Tuned Recall (Target >= 0.60)"], color="#dbeafe"), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎚️ Dynamic Decision Threshold Simulator")
        st.caption("Adjust the decision threshold to explore the clinical trade-off between sensitivity (Recall) and false alarm burden (Precision):")

        # Threshold slider
        thresh_col, metric_col = st.columns([1.5, 2.5])
        with thresh_col:
            selected_thresh = st.slider("Classification Probability Threshold", 0.10, 0.90, 0.45, step=0.01)
            st.info(f"Targeting **Recall &ge; 0.60** requires threshold &le; {comp_df.loc[comp_df['Algorithm']=='RandomForest', 'Tuned Threshold'].values[0]:.2f}")

        # If evaluated test samples exist, show dynamic confusion matrix
        test_eval_path = os.path.join(PROJECT_ROOT, "data", "test_sample_evaluated.csv")
        if os.path.exists(test_eval_path):
            test_eval_df = pd.read_csv(test_eval_path)
            y_true_s = test_eval_df["actual_readmitted_binary"].values
            y_prob_s = test_eval_df["predicted_prob"].values
            y_pred_s = (y_prob_s >= selected_thresh).astype(int)

            from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
            rec_sim = recall_score(y_true_s, y_pred_s, zero_division=0)
            prec_sim = precision_score(y_true_s, y_pred_s, zero_division=0)
            f1_sim = f1_score(y_true_s, y_pred_s, zero_division=0)
            cm_sim = confusion_matrix(y_true_s, y_pred_s)

            with metric_col:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Simulated Recall", f"{rec_sim:.1%}")
                mc2.metric("Simulated Precision", f"{prec_sim:.1%}")
                mc3.metric("Simulated F1", f"{f1_sim:.3f}")

                fig_cm = px.imshow(
                    cm_sim,
                    text_auto=True,
                    labels=dict(x="Predicted Class", y="Actual Class"),
                    x=["No Readmit (0)", "Readmitted &lt;30d (1)"],
                    y=["No Readmit (0)", "Readmitted &lt;30d (1)"],
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")
        # ROC and PR Curves
        c_curve1, c_curve2 = st.columns(2)
        with c_curve1:
            fig_roc = go.Figure()
            for m_name, vals in metadata["models_comparison"].items():
                roc_pts = vals["roc_curve"]
                fpr_list = [p["fpr"] for p in roc_pts]
                tpr_list = [p["tpr"] for p in roc_pts]
                fig_roc.add_trace(go.Scatter(x=fpr_list, y=tpr_list, mode="lines", name=f"{m_name} (AUC={vals['roc_auc']:.2f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random Guess"))
            fig_roc.update_layout(title="<b>ROC Curves Comparison</b>", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)", height=380)
            st.plotly_chart(fig_roc, use_container_width=True)

        with c_curve2:
            fig_pr = go.Figure()
            for m_name, vals in metadata["models_comparison"].items():
                pr_pts = vals["pr_curve"]
                r_list = [p["recall"] for p in pr_pts]
                p_list = [p["precision"] for p in pr_pts]
                fig_pr.add_trace(go.Scatter(x=r_list, y=p_list, mode="lines", name=f"{m_name} (PR-AUC={vals['pr_auc']:.2f})"))
            fig_pr.update_layout(title="<b>Precision-Recall Curves</b>", xaxis_title="Recall", yaxis_title="Precision", height=380)
            st.plotly_chart(fig_pr, use_container_width=True)

        # Top Features Chart
        if "top_features" in metadata:
            feat_df = pd.DataFrame(metadata["top_features"][:12])
            fig_feat = px.bar(
                feat_df,
                x="importance",
                y="feature",
                orientation="h",
                title="<b>Top 12 Overall Model Clinical Predictors</b>",
                labels={"importance": "Relative Importance", "feature": "Predictor"},
                color="importance",
                color_continuous_scale="Viridis"
            )
            fig_feat.update_layout(yaxis=dict(autorange="reversed"), height=380)
            st.plotly_chart(fig_feat, use_container_width=True)


# ==============================================================================
# TAB 4: CLINICAL POPULATION INSIGHTS & EXPLORATION
# ==============================================================================
with tab_eda:
    st.markdown("### 🔍 Population Health & Exploratory Data Analysis")
    st.caption("Deep-dive cohort segmentation and multi-variable interaction exploration.")

    if not df_clean.empty:
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            sel_race = st.multiselect("Filter by Race/Ethnicity", df_clean["race"].unique().tolist(), default=df_clean["race"].unique().tolist()[:3])
        with filter_col2:
            sel_adm = st.multiselect("Filter by Admission Category", df_clean["admission_type_desc"].unique().tolist(), default=df_clean["admission_type_desc"].unique().tolist()[:3])
        with filter_col3:
            sel_diag = st.multiselect("Filter by Primary Disease Family", df_clean["diag_1_family"].unique().tolist(), default=df_clean["diag_1_family"].unique().tolist()[:4])

        filtered_df = df_clean[
            (df_clean["race"].isin(sel_race)) &
            (df_clean["admission_type_desc"].isin(sel_adm)) &
            (df_clean["diag_1_family"].isin(sel_diag))
        ]

        st.write(f"Filtered Cohort Size: **{len(filtered_df):,} encounters**")

        eda_c1, eda_c2 = st.columns(2)
        with eda_c1:
            # Length of Stay vs Number of Medications scatter / density
            fig_scatter = px.box(
                filtered_df.sample(min(2000, len(filtered_df))),
                x="readmitted",
                y="time_in_hospital",
                color="readmitted",
                title="<b>Length of Hospital Stay by Readmission Status</b>",
                labels={"readmitted": "Readmission Status", "time_in_hospital": "Days in Hospital"}
            )
            fig_scatter.update_layout(height=360)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with eda_c2:
            fig_meds = px.histogram(
                filtered_df.sample(min(2000, len(filtered_df))),
                x="num_medications",
                color="readmitted_binary",
                barmode="overlay",
                title="<b>Distribution of Prescribed Medications (0: No, 1: &lt;30d Readmit)</b>",
                labels={"num_medications": "Number of Medications", "readmitted_binary": "Readmitted &lt;30d"}
            )
            fig_meds.update_layout(height=360)
            st.plotly_chart(fig_meds, use_container_width=True)

        st.markdown("#### Sample Patient Population Table")
        display_cols = [
            "encounter_id", "patient_nbr", "race", "gender", "age",
            "admission_type_desc", "time_in_hospital", "num_medications",
            "diag_1_family", "A1Cresult", "readmitted"
        ]
        available_display_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[available_display_cols].head(100), use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.85rem;">
    HCL Data Science Capstone Project • Diabetic Patient Health Record Dashboard • Track P_090<br>
    Validated with 5-Fold Stratified Group Split (Zero Patient Duplication Leakage) • Powered by Python & Streamlit
</div>
""", unsafe_allow_html=True)
