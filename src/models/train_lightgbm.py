import os
import pickle
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = "data/processed/diabetic_data_final.csv"
df = pd.read_csv(file_path)

# Drop non-numeric columns
df.drop(columns=["diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult"], errors='ignore', inplace=True)

# Convert categorical variables to numeric
binary_columns = ["change", "diabetesMed"]
for col in binary_columns:
    df[col] = df[col].map({"No": 0, "Yes": 1})

medications = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone"
]
for med in medications:
    df[med] = df[med].map({"No": 0, "Steady": 1, "Up": 2, "Down": -1})

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Split into features & target
X = df.drop(columns=["readmitted"])
y = df["readmitted"]

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train LightGBM model with correct feature names
lgbm_model = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train_scaled, y_train, feature_name=list(X.columns))  # Save correct feature names

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model
with open("models/lightgbm_model.pkl", "wb") as f:
    pickle.dump(lgbm_model, f)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ LightGBM model and scaler saved successfully in 'models/' directory with correct feature names.")
