import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
file_path = "data/processed/diabetic_data_cleaned.csv"
df = pd.read_csv(file_path)

# Step 1: Convert 'age' column from range to numeric midpoint
def convert_age(age_range):
    age_dict = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
                "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
                "[80-90)": 85, "[90-100)": 95}
    return age_dict.get(age_range, 55)  # Default to 55 if missing

df["age"] = df["age"].apply(convert_age)

# Step 2: Convert 'readmitted' column to binary classification (1 = Readmitted, 0 = Not Readmitted)
df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

# Step 3: Encode categorical variables
categorical_columns = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id"]

label_encoders = {}  # Store encoders for future reference
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder if needed later

# Step 4: Save transformed dataset
df.to_csv("data/processed/diabetic_data_transformed.csv", index=False)

print("✅ Data Encoding & Transformation Completed. Transformed dataset saved.")
