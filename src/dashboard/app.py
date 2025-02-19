import streamlit as st
import requests
import time

# 🌟 Set Page Title & Icon
st.set_page_config(page_title="Healthcare Readmission Predictor", page_icon="🏥", layout="wide")

# ---- Title & Description ----
st.title("🏥 Healthcare Readmission Risk Predictor")
st.markdown("### 🔍 Predict the likelihood of a patient being readmitted to the hospital.")

# API Endpoint
API_URL = "http://127.0.0.1:8000/predict"

# ---- Mapping Dictionaries ----
race_mapping = {
    "American Indian/Alaska Native": 1,
    "Asian": 2,
    "Black/African American": 3,
    "Hispanic": 4,
    "White": 5
}

gender_mapping = {
    "Male": 1,
    "Female": 0
}

admission_type_mapping = {
    "Emergency": 1,
    "Urgent": 2,
    "Elective": 3,
    "Newborn": 4,
    "Not Available": 5,
    "Trauma Center": 6,
    "Not Mapped": 7,
    "Unknown": 8
}

discharge_mapping = {
    "Discharged to home": 1,
    "Discharged to another facility": 2,
    "Discharged to hospice": 3,
    "Left against medical advice": 4,
    "Expired": 5
}

admission_source_mapping = {
    "Physician Referral": 1,
    "Clinic Referral": 2,
    "HMO Referral": 3,
    "Transfer from Hospital": 4,
    "Transfer from Skilled Nursing Facility": 5
}

max_glu_mapping = {
    "None": "None",
    "Normal": "Norm",
    "Above 200": ">200",
    "Above 300": ">300"
}

A1C_mapping = {
    "None": "None",
    "Normal": "Norm",
    "Above 7": ">7",
    "Above 8": ">8"
}

yes_no_mapping = {
    "Yes": 1,
    "No": 0
}

# ---- Layout with Columns ----
col1, col2 = st.columns(2)

# ---- Patient Information ----
with col1:
    st.header("📌 Patient Information")
    race_display = st.selectbox("🌍 Race", list(race_mapping.keys()))
    race = race_mapping[race_display]

    gender_display = st.radio("⚤ Gender", list(gender_mapping.keys()))
    gender = gender_mapping[gender_display]

    age = st.slider("🎂 Age", 0, 100, 50)

    admission_type_display = st.selectbox("🏥 Admission Type", list(admission_type_mapping.keys()))
    admission_type_id = admission_type_mapping[admission_type_display]

    discharge_display = st.selectbox("🚪 Discharge Disposition", list(discharge_mapping.keys()))
    discharge_disposition_id = discharge_mapping[discharge_display]

    admission_source_display = st.selectbox("📋 Admission Source", list(admission_source_mapping.keys()))
    admission_source_id = admission_source_mapping[admission_source_display]

    time_in_hospital = st.slider("🕒 Time in Hospital (Days)", 1, 20, 5)

with col2:
    st.header("🔬 Medical Information")
    num_lab_procedures = st.slider("🧪 Lab Procedures", 1, 100, 30)
    num_procedures = st.slider("🏥 Number of Procedures", 0, 10, 1)
    num_medications = st.slider("💊 Medications", 1, 50, 10)
    number_diagnoses = st.slider("🦠 Number of Diagnoses", 1, 20, 7)
    total_visits = st.slider("🔁 Total Visits", 1, 50, 5)
    comorbidity_score = st.slider("⚕️ Comorbidity Score", 1, 10, 5)

    # ✅ Add missing visit count fields
    number_outpatient = st.number_input("🏠 Outpatient Visits", min_value=0, max_value=50, value=0)
    number_emergency = st.number_input("🚨 Emergency Visits", min_value=0, max_value=50, value=0)
    number_inpatient = st.number_input("🏩 Inpatient Visits", min_value=0, max_value=50, value=0)

# ---- Diagnosis & Lab Results ----
st.header("🩺 Diagnosis & Lab Results")
diag_1 = st.text_input("📄 Primary Diagnosis Code (ICD-9)", value="250.00", help="E.g., '250.00' for Diabetes")
diag_2 = st.text_input("📄 Secondary Diagnosis Code (ICD-9)", value="276.8")
diag_3 = st.text_input("📄 Tertiary Diagnosis Code (ICD-9)", value="414.01")

max_glu_display = st.radio("🩸 Max Glucose Serum Level", list(max_glu_mapping.keys()))
max_glu_serum = max_glu_mapping[max_glu_display]

A1C_display = st.radio("🔬 A1C Test Result", list(A1C_mapping.keys()))
A1Cresult = A1C_mapping[A1C_display]

# ✅ Include "Change in Condition" and "Diabetes Medication"
st.header("📌 Additional Information")
change_display = st.selectbox("⚠️ Change in Condition", list(yes_no_mapping.keys()))
change = yes_no_mapping[change_display]

diabetesMed_display = st.selectbox("💉 Diabetes Medication", list(yes_no_mapping.keys()))
diabetesMed = yes_no_mapping[diabetesMed_display]

# ---- Medications ----
st.header("💊 Medications")
medications_list = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide_metformin", "glipizide_metformin",
    "glimepiride_pioglitazone", "metformin_rosiglitazone", "metformin_pioglitazone"
]

col_med1, col_med2 = st.columns(2)
medication_values = {}

for i, med in enumerate(medications_list):
    if i % 2 == 0:
        with col_med1:
            medication_values[med] = st.selectbox(med.capitalize(), list(yes_no_mapping.keys()))
    else:
        with col_med2:
            medication_values[med] = st.selectbox(med.capitalize(), list(yes_no_mapping.keys()))

# Convert "Yes"/"No" to 1/0
medication_values = {k: yes_no_mapping[v] for k, v in medication_values.items()}

# ---- Predict Button ----
if st.button("🔍 Predict Readmission Risk"):
    with st.spinner("Processing... 🔄"):
        time.sleep(1)

        # Prepare API Payload
        payload = {
            "race": race, "gender": gender, "age": age, "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id, "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital, "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures, "num_medications": num_medications,
            "number_outpatient": number_outpatient, "number_emergency": number_emergency, 
            "number_inpatient": number_inpatient, "number_diagnoses": number_diagnoses, 
            "total_visits": total_visits, "comorbidity_score": comorbidity_score,
            "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
            "max_glu_serum": max_glu_serum, "A1Cresult": A1Cresult, "change": change,
            "diabetesMed": diabetesMed, **medication_values
        }
        
        response = requests.post(API_URL, json=payload)
        result = response.json()
        st.success(f"✅ Readmission Probability: {result['probability']}%")
