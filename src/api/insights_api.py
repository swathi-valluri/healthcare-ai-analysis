from fastapi import APIRouter, HTTPException
import pandas as pd
import os

# ✅ Use APIRouter to prevent circular import issues
router = APIRouter()

# 📂 Define Data Path (Ensure file exists)
DATA_PATH = "data/patient_readmissions.csv"

# ✅ Load Data (Handle Missing File)
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"❌ Error: File '{DATA_PATH}' not found!")

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Error loading data: {e}")

# 📊 API: Readmission Rates by Age & Race
@router.get("/readmission_rates")
async def get_readmission_rates():
    try:
        insights = df.groupby(["age", "race"])["readmission"].mean().reset_index()
        return insights.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {e}")

# 📊 API: Readmission Trends by Admission Type
@router.get("/trends")
async def get_readmission_trends():
    try:
        # ✅ Debug: Print available columns inside FastAPI
        print("\n📊 Available Columns in CSV:", df.columns.tolist())

        required_columns = {"admission_type", "time_in_hospital", "readmission"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"400: Missing required columns: {missing_columns}"
            )

        trends = df.groupby(["admission_type", "time_in_hospital"])["readmission"].mean().reset_index()
        return {"data": trends.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating trends: {e}")

# ✅ Force FastAPI to reload the latest CSV on startup
@router.on_event("startup")
async def startup_event():
    global df  # ✅ Ensures FastAPI loads the latest CSV on restart
    df = pd.read_csv("data/patient_readmissions.csv")
    print("\n✅ Data Reloaded on Startup!\n")
    print("\n📊 Available Columns at Startup:", df.columns.tolist())

from fastapi.responses import FileResponse
import csv

@router.get("/export")
async def export_data():
    try:
        export_path = "data/exported_insights.csv"

        # ✅ Save DataFrame to CSV
        df.to_csv(export_path, index=False)

        # ✅ Return CSV file as a downloadable response
        return FileResponse(export_path, media_type="text/csv", filename="insights_data.csv")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {e}")
