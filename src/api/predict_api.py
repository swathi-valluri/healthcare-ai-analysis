import pickle
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import os
import numpy as np
import lightgbm as lgb

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Load trained LightGBM model
model_path = "models/lightgbm_model.pkl"
scaler_path = "models/scaler.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
else:
    raise FileNotFoundError("LightGBM model not found! Train and save it first.")

# Initialize FastAPI app
app = FastAPI(title="Healthcare Readmission Prediction API",
              description="API for predicting patient readmission risk using LightGBM",
              version="1.0")

# Define request format using Pydantic
class PatientData(BaseModel):
    age: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    total_visits: int
    comorbidity_score: int
    change: int
    diabetesMed: int
    metformin: int
    insulin: int
    glipizide: int
    glyburide: int
    rosiglitazone: int

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict_readmission(data: PatientData):
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction_prob = model.predict_proba(input_scaled)[:, 1]
        prediction = int(prediction_prob[0] > 0.5)

        return {
            "readmission_probability": round(float(prediction_prob[0]), 4),
            "readmission_prediction": "Yes" if prediction == 1 else "No"
        }
    except Exception as e:
        return {"error": str(e)}

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
