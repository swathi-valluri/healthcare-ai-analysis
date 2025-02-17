import pickle
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np
import lightgbm as lgb

## Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Load trained LightGBM model with error handling
model_path = "models/lightgbm_model.pkl"
scaler_path = "models/scaler.pkl"

model, scaler = None, None  # Initialize to None

try:
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        raise FileNotFoundError("❌ LightGBM model not found! Please train and save it first.")

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        print("⚠️ Warning: Scaler file not found! Continuing without scaling.")
except Exception as e:
    print(f"🚨 Model loading error: {e}")

# Initialize FastAPI app
app = FastAPI(title="Healthcare Readmission Prediction API",
              description="API for predicting patient readmission risk using LightGBM",
              version="1.0")

# Define request format using Pydantic
class PatientData(BaseModel):
    race: int
    gender: int
    age: int
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    metformin: int
    repaglinide: int
    nateglinide: int
    chlorpropamide: int
    glimepiride: int
    acetohexamide: int
    glipizide: int
    glyburide: int
    tolbutamide: int
    pioglitazone: int
    rosiglitazone: int
    acarbose: int
    miglitol: int
    troglitazone: int
    tolazamide: int
    examide: int
    citoglipton: int
    insulin: int
    glyburide_metformin: int
    glipizide_metformin: int
    glimepiride_pioglitazone: int
    metformin_rosiglitazone: int
    metformin_pioglitazone: int
    change: int
    diabetesMed: int
    total_visits: int
    comorbidity_score: int

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict_readmission(data: PatientData):
    print("🚀 Incoming API Request Data:", data.dict())  # Debug print
    # Ensure the model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert data to model input
    try:
        input_data = np.array([list(data.dict().values())])  # Convert to NumPy array
        print("📊 Processed Input Data for Model:", input_data)  # Debug print

        # Ensure the model supports `predict_proba`
        if hasattr(model, "predict_proba"):
            prediction_prob = model.predict_proba(input_data)[:, 1]  # Probability of readmission
        else:
            prediction_prob = model.predict(input_data)  # Use `predict()` if `predict_proba` is unavailable

        print("🔍 Model Prediction Probability:", prediction_prob)  # Debug print

        # Convert probability to label
        prediction_label = "Yes" if prediction_prob[0] > 0.5 else "No"

        # Return response
        return {
            "readmission_probability": round(float(prediction_prob[0]), 4),
            "readmission_prediction": prediction_label
        }

    except Exception as e:
        print(f"🚨 Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check model compatibility.")

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
