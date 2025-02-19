import joblib
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model_path = "models/lightgbm_model.pkl"
try:
    model = joblib.load(model_path)
    logging.info(f"✅ Model loaded from {model_path}")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    model = None

# Define Input Schema
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
    total_visits: int
    comorbidity_score: int
    change: int
    diabetesMed: int
    diag_1: str
    diag_2: str
    diag_3: str
    max_glu_serum: str
    A1Cresult: str
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

@app.post("/predict")
async def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Log input before transformation
        logging.info(f"📊 Raw Input Features: {input_df.dtypes}")

        # ✅ Step 1: Encode `diag_1`, `diag_2`, `diag_3` (Convert to category codes)
        for col in ["diag_1", "diag_2", "diag_3"]:
            input_df[col] = pd.factorize(input_df[col])[0]  # Assign unique integer codes
        
        # ✅ Step 2: Encode `max_glu_serum` and `A1Cresult` (Map to numeric)
        glu_mapping = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
        a1c_mapping = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}

        input_df["max_glu_serum"] = input_df["max_glu_serum"].map(glu_mapping).fillna(0).astype(int)
        input_df["A1Cresult"] = input_df["A1Cresult"].map(a1c_mapping).fillna(0).astype(int)

        # Log transformed input data
        logging.info(f"📊 Processed Input Features: {input_df.dtypes}")

        # ✅ Step 3: Ensure all numeric fields are cast properly
        input_df = input_df.apply(pd.to_numeric, errors="coerce")

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {"readmission": int(prediction), "probability": round(probability * 100, 2)}

    except Exception as e:
        logging.error(f"🚨 Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
