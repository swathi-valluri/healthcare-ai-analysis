import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import os

# Ensure reports folder exists
os.makedirs("reports", exist_ok=True)

# Step 1: Load Processed Dataset
file_path = "data/processed/diabetic_data_final.csv"
df = pd.read_csv(file_path)

# Step 2: Drop non-numeric columns
df.drop(columns=["diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult"], errors='ignore', inplace=True)

# Step 3: Convert categorical medication columns to numerical values
medications = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone"
]
for med in medications:
    df[med] = df[med].map({"No": 0, "Steady": 1, "Up": 2, "Down": -1})

# Step 4: Convert categorical variables to numeric values
binary_columns = ["change", "diabetesMed"]
for col in binary_columns:
    df[col] = df[col].map({"No": 0, "Yes": 1})

# Step 5: Handle Missing Values
df.fillna(df.mean(), inplace=True)

# Step 6: Split Data into Features & Target
X = df.drop(columns=["readmitted"])  
y = df["readmitted"]

# Step 7: Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Step 9: Standardize Numerical Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Hyperparameter Tuning & Model Training

# Dictionary to store model results
model_results = {}

def evaluate_model(model, model_name):
    """Train and evaluate a model, then store results."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    model_results[model_name] = metrics
    print(f"\n📊 {model_name} Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# 1️⃣ **Optimized Logistic Regression**
log_reg = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l2', C=1.0, random_state=42)
evaluate_model(log_reg, "Optimized Logistic Regression")

# 2️⃣ **Optimized Decision Tree**
dt_params = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring="roc_auc")
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
evaluate_model(best_dt, "Optimized Decision Tree")

# 3️⃣ **Random Forest**
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring="roc_auc")
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
evaluate_model(best_rf, "Random Forest")

# 4️⃣ **XGBoost**
xgb_params = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'learning_rate': [0.01, 0.1]}
grid_xgb = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"), xgb_params, cv=5, scoring="roc_auc")
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
evaluate_model(best_xgb, "XGBoost")

# 5️⃣ **LightGBM**
lgbm_params = {'n_estimators': [50, 100], 'max_depth': [-1, 5, 10], 'learning_rate': [0.01, 0.1]}
grid_lgbm = GridSearchCV(LGBMClassifier(random_state=42), lgbm_params, cv=5, scoring="roc_auc")
grid_lgbm.fit(X_train, y_train)
best_lgbm = grid_lgbm.best_estimator_
evaluate_model(best_lgbm, "LightGBM")

# Step 11: Save Model Comparison Report
comparison_df = pd.DataFrame(model_results).T
comparison_df.to_csv("reports/optimized_model_comparison.csv")

print("\n✅ Model Optimization Completed. Reports saved in 'reports/' folder.")
