import streamlit as st
import requests

# Define API URL
API_URL = "http://127.0.0.1:8000/predict"

# Streamlit UI Title
st.set_page_config(page_title="Healthcare Readmission Prediction", layout="wide")

# 🏥 Header
st.markdown("# 🏥 **Healthcare Readmission Prediction**")

# Collect Patient Information
st.sidebar.header("Enter Patient Details")

race = st.sidebar.selectbox("Race", options=[1, 2, 3, 4, 5], format_func=lambda x: {1: "Caucasian", 2: "African American", 3: "Asian", 4: "Hispanic", 5: "Other"}[x])
gender = st.sidebar.radio("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
age = st.sidebar.slider("Age", 0, 100, 50)
admission_type_id = st.sidebar.selectbox("Admission Type ID", options=[1, 2, 3, 4, 5])
discharge_disposition_id = st.sidebar.selectbox("Discharge Disposition ID", options=[1, 2, 3, 4, 5])
admission_source_id = st.sidebar.selectbox("Admission Source ID", options=[1, 2, 3, 4, 5])
time_in_hospital = st.sidebar.slider("Time in Hospital (days)", 1, 14, 5)
num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 1, 100, 30)
num_procedures = st.sidebar.slider("Number of Procedures", 0, 10, 1)
num_medications = st.sidebar.slider("Number of Medications", 1, 50, 10)
number_outpatient = st.sidebar.slider("Number of Outpatient Visits", 0, 20, 0)
number_emergency = st.sidebar.slider("Number of Emergency Visits", 0, 20, 0)
number_inpatient = st.sidebar.slider("Number of Inpatient Visits", 0, 20, 0)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 7)
total_visits = st.sidebar.slider("Total Visits", 1, 50, 5)
comorbidity_score = st.sidebar.slider("Comorbidity Score", 1, 10, 5)

# Binary Categorical Inputs (Checkboxes)
change = st.sidebar.checkbox("Change in Medication", value=False)
diabetesMed = st.sidebar.checkbox("On Diabetes Medication", value=False)

# Medications (Checkbox)
medications = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide_metformin",
    "glipizide_metformin", "glimepiride_pioglitazone", "metformin_rosiglitazone",
    "metformin_pioglitazone"
]

meds_data = {med: int(st.sidebar.checkbox(med.capitalize(), value=False)) for med in medications}

# Prepare JSON request
input_data = {
    "race": race,
    "gender": gender,
    "age": age,
    "admission_type_id": admission_type_id,
    "discharge_disposition_id": discharge_disposition_id,
    "admission_source_id": admission_source_id,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses,
    "total_visits": total_visits,
    "comorbidity_score": comorbidity_score,
    "change": int(change),
    "diabetesMed": int(diabetesMed),
    **meds_data
}

# Debugging: Show API Request Data
# st.write("🔍 **Debug - API Request Data:**")
# st.json(input_data)

# Predict Button
if st.button("Predict Readmission Risk"):
    try:
        # Send request to FastAPI
        response = requests.post(API_URL, json=input_data)

        # Check if API responded successfully
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract Predictions
            prediction = response_data.get("readmission_prediction", "Error")
            probability = response_data.get("readmission_probability", "Error")

            # Display Prediction Results
            st.markdown("### **📊 Prediction Results:**")

            if prediction == "Yes":
                st.markdown("### ⚠ **:red[Yes Readmission Risk]**")
            else:
                st.markdown("### ✅ **:green[No Readmission Risk]**")

            # Display Probability
            st.markdown(f"### **📊 Readmission Probability:** `{probability}`")

            # Show Probability as Progress Bar
            st.progress(float(probability))

            # Debugging: Show API Response
            st.write("🔍 **API Response Debug:**")
            st.json(response_data)

        else:
            st.error("⚠ API Error: Unable to get prediction.")

    except Exception as e:
        st.error(f"⚠ API Error: {e}")
