import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load transformed dataset
file_path = "data/processed/diabetic_data_transformed.csv"
df = pd.read_csv(file_path)

# Step 1: Scale numerical features using MinMaxScaler
scaler = MinMaxScaler()
numerical_columns = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                     "number_outpatient", "number_emergency", "number_inpatient"]

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 2: Feature Engineering

# 2.1 Create "Total Visits" Feature (Sum of previous visits)
df["total_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]

# 2.2 Create a Comorbidity Score from Diagnosis Codes
# Define function to assign comorbidity scores
def assign_comorbidity(diag):
    if diag.startswith("250"):  # Diabetes-related
        return 2
    elif diag.startswith("401") or diag.startswith("402"):  # Hypertension
        return 1
    elif diag.startswith("410") or diag.startswith("411"):  # Cardiovascular disease
        return 3
    elif diag.startswith("585"):  # Kidney disease
        return 3
    else:
        return 1  # Default risk score

df["comorbidity_score"] = df["diag_1"].apply(assign_comorbidity) + \
                          df["diag_2"].apply(assign_comorbidity) + \
                          df["diag_3"].apply(assign_comorbidity)

# Step 3: Save processed dataset
df.to_csv("data/processed/diabetic_data_final.csv", index=False)

print("✅ Feature Scaling & Engineering Completed. Final dataset saved.")
