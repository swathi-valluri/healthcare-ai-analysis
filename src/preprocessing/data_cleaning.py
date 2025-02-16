import pandas as pd

# Load dataset
file_path = "data/raw/diabetic_data.csv"
df = pd.read_csv(file_path)

# Step 1: Drop highly missing and irrelevant columns
df.drop(columns=["weight", "medical_specialty", "payer_code", "encounter_id", "patient_nbr"], inplace=True)

# Step 2: Fill missing values in categorical columns
df["race"] = df["race"].replace("?", "Unknown")
df["diag_1"] = df["diag_1"].replace("?", "Unknown")
df["diag_2"] = df["diag_2"].replace("?", "Unknown")
df["diag_3"] = df["diag_3"].replace("?", "Unknown")

# Step 3: Save cleaned dataset
df.to_csv("data/processed/diabetic_data_cleaned.csv", index=False)

print("✅ Data Cleaning Completed. Cleaned dataset saved.")
