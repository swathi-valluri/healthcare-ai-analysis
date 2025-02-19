import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

# ✅ Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# ✅ Define the dataset path
DATA_FILE = "data/processed/diabetic_data_final.csv"

# ✅ Define categorical columns to encode
CATEGORICAL_FEATURES = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id", "max_glu_serum", "A1Cresult"]

# 📌 Step 1: Load the dataset
try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Successfully loaded dataset with shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ ERROR: The dataset file '{DATA_FILE}' was not found. Check the path and try again.")
    exit(1)

# 📌 Step 2: Ensure required columns exist
missing_cols = [col for col in CATEGORICAL_FEATURES if col not in df.columns]
if missing_cols:
    print(f"❌ ERROR: Missing columns in dataset: {missing_cols}")
    exit(1)

# 📌 Step 3: Train the OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(df[CATEGORICAL_FEATURES])

# 📌 Step 4: Save the trained encoder inside the models directory
encoder_path = "models/encoder.pkl"
joblib.dump(encoder, encoder_path)
print(f"✅ Encoder successfully saved at '{encoder_path}'")
