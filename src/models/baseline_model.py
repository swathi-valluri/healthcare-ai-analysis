import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

# Step 5: Convert any remaining categorical columns using Label Encoding
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for future use

# Step 6: Handle Missing Values
# Fill missing values with column mean for numerical features
df.fillna(df.mean(), inplace=True)  

# Step 7: Split Data into Features & Target
X = df.drop(columns=["readmitted"])  
y = df["readmitted"]

# Step 8: Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 9: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Step 10: Standardize Numerical Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 11: Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Step 12: Evaluate Logistic Regression Model
y_pred_log = log_reg.predict(X_test)
y_prob_log = log_reg.predict_proba(X_test)[:, 1]

log_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_log),
    "Precision": precision_score(y_test, y_pred_log),
    "Recall": recall_score(y_test, y_pred_log),
    "F1-Score": f1_score(y_test, y_pred_log),
    "ROC-AUC": roc_auc_score(y_test, y_prob_log)
}

print("\n🔍 Logistic Regression Model Performance:")
for metric, value in log_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save classification report
report_log = classification_report(y_test, y_pred_log, output_dict=True)
pd.DataFrame(report_log).transpose().to_csv("reports/logistic_regression_report.csv")

# Step 13: Train Decision Tree Model for Comparison
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)

# Step 14: Evaluate Decision Tree Model
y_pred_tree = tree_clf.predict(X_test)
y_prob_tree = tree_clf.predict_proba(X_test)[:, 1]

tree_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_tree),
    "Precision": precision_score(y_test, y_pred_tree),
    "Recall": recall_score(y_test, y_pred_tree),
    "F1-Score": f1_score(y_test, y_pred_tree),
    "ROC-AUC": roc_auc_score(y_test, y_prob_tree)
}

print("\n🌳 Decision Tree Model Performance:")
for metric, value in tree_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save comparison report
comparison_df = pd.DataFrame([log_metrics, tree_metrics], index=["Logistic Regression", "Decision Tree"])
comparison_df.to_csv("reports/model_comparison.csv")

print("\n✅ Model Training Completed. Reports saved in 'reports/' folder.")
