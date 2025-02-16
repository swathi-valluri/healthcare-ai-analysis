import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# Ensure reports folder exists
os.makedirs("reports", exist_ok=True)

# Load final processed dataset
file_path = "data/processed/diabetic_data_final.csv"
df = pd.read_csv(file_path)

# Step 1: Basic Statistics & Distributions
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum().sum())
print("\nClass Distribution (Readmitted Cases):\n", df["readmitted"].value_counts(normalize=True))

# Step 2: Visualize Readmission Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="readmitted", hue="readmitted", palette="viridis", legend=False)
plt.title("Distribution of Readmission Cases")
plt.xlabel("Readmitted (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.savefig("reports/readmission_distribution.png")  
plt.close()

# Step 3: Age Distribution Analysis
plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=10, kde=True, color="blue")
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("reports/age_distribution.png")
plt.close()

# Step 4: Heatmap of Correlations (Drop non-numeric columns)
df_numeric = df.select_dtypes(include=['number'])  # Keep only numeric columns
plt.figure(figsize=(10,6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.savefig("reports/correlation_matrix.png")
plt.close()

# Step 5: Readmission Rate by Age Group (Fixed barplot warning)
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="age", y="readmitted", errorbar=None, palette="magma", hue=None)
plt.xticks(rotation=45)
plt.title("Readmission Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Readmission Rate")
plt.savefig("reports/readmission_by_age.png")
plt.close()

# Step 6: Convert Categorical Variables to Numeric (Fixing 'No' Error)
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

# Step 7: Feature Importance Analysis using SHAP
print("\nRunning SHAP Analysis (Feature Importance)...")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Drop problematic non-numeric columns
df.drop(columns=["max_glu_serum", "A1Cresult", "diag_1", "diag_2", "diag_3"], inplace=True)

# Select features & target
X = df.drop(columns=["readmitted"])  
y = df["readmitted"]

# Train simple Decision Tree Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# SHAP Analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("reports/shap_feature_importance.png")
plt.close()

print("✅ EDA Completed. Check the generated visualizations in the 'reports/' folder!")
