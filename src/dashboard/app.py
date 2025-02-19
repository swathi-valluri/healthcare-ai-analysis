import streamlit as st
import requests

st.title("🏥 Healthcare Readmission Risk Predictor")

API_URL = "http://127.0.0.1:8000/predict"

# Define categorical dropdown options
medications = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide_metformin", "glipizide_metformin",
    "glimepiride_pioglitazone", "metformin_rosiglitazone", "metformin_pioglitazone"
]

# User Inputs
race = st.selectbox("Race", [1, 2, 3, 4, 5])
gender = st.selectbox("Gender", [0, 1])
age = st.number_input("Age", min_value=0, max_value=120, value=50)
admission_type_id = st.number_input("Admission Type ID", min_value=1, max_value=8, value=1)
discharge_disposition_id = st.number_input("Discharge Disposition ID", min_value=1, max_value=30, value=1)
admission_source_id = st.number_input("Admission Source ID", min_value=1, max_value=30, value=1)
time_in_hospital = st.number_input("Time in Hospital", min_value=1, max_value=20, value=5)

number_outpatient = st.number_input("Number of Outpatient Visits", min_value=0, max_value=50, value=0)
number_emergency = st.number_input("Number of Emergency Visits", min_value=0, max_value=50, value=0)
number_inpatient = st.number_input("Number of Inpatient Visits", min_value=0, max_value=50, value=0)

num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=100, value=30)
num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=10, value=1)
num_medications = st.number_input("Number of Medications", min_value=0, max_value=50, value=10)
number_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=20, value=7)
total_visits = st.number_input("Total Visits", min_value=0, max_value=50, value=5)
comorbidity_score = st.number_input("Comorbidity Score", min_value=0, max_value=10, value=5)

change = st.selectbox("Change", [0, 1])
diabetesMed = st.selectbox("Diabetes Medication", [0, 1])

# ✅ New Inputs (Fixing 422 error)
diag_1 = st.text_input("Primary Diagnosis (diag_1)", value="250.00")
diag_2 = st.text_input("Secondary Diagnosis (diag_2)", value="276.8")
diag_3 = st.text_input("Tertiary Diagnosis (diag_3)", value="414.01")
max_glu_serum = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"], index=0)
A1Cresult = st.selectbox("A1C Test Result", ["None", "Norm", ">7", ">8"], index=0)

# Medication Inputs
medication_values = {med: st.selectbox(med.capitalize(), ["No", "Yes"], key=med) for med in medications}
medication_values = {k: (1 if v == "Yes" else 0) for k, v in medication_values.items()}

if st.button("🔍 Predict Readmission Risk"):
    payload = {
        "race": race, "gender": gender, "age": age, "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id, "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital, "num_lab_procedures": num_lab_procedures, "num_procedures": num_procedures,
        "num_medications": num_medications, "number_outpatient": number_outpatient,
        "number_emergency": number_emergency, "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses, "total_visits": total_visits, "comorbidity_score": comorbidity_score,
        "change": change, "diabetesMed": diabetesMed,
        "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
        "max_glu_serum": max_glu_serum, "A1Cresult": A1Cresult,
        **medication_values
    }

    response = requests.post(API_URL, json=payload)
    st.write(response.json())
