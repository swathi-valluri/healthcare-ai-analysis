from fastapi import FastAPI
from src.api.predict_api import app as predict_app  # ✅ Import Predict API
from src.api.insights_api import router as insights_router  # ✅ Import Insights API

# ✅ Create FastAPI application
app = FastAPI(title="Healthcare AI Analysis", version="1.0")

# ✅ Register APIs
app.include_router(insights_router, prefix="/insights", tags=["Insights API"])
app.mount("/predict", predict_app)  # Mount the Predict API

# ✅ Root Route
@app.get("/")
def home():
    return {"message": "Welcome to Healthcare AI API!"}
