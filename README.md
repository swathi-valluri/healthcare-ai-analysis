
# 🏥 Healthcare AI Analysis

## 📌 Project Overview
Healthcare AI Analysis is a **machine learning-based predictive model** designed to analyze patient readmission risks. This project leverages **FastAPI**, **Streamlit**, and **machine learning models** to provide insights into patient readmissions and visualize trends using an interactive dashboard.

---

## 📂 **Project Structure**
```
swathi-valluri-healthcare-ai-analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── exported_insights.csv
│   ├── patient_readmissions.csv
│   ├── predictions.csv
│   ├── processed/
│   │   ├── diabetic_data_balanced.csv
│   │   └── diabetic_data_final.csv
│   └── raw/
│       └── diabetic_data.csv
├── models/
│   ├── encoder.pkl
│   ├── lightgbm_model.pkl
│   ├── recreate_encoder.py
│   └── scaler.pkl
├── reports/
│   ├── logistic_regression_report.csv
│   ├── model_comparison.csv
│   └── optimized_model_comparison.csv
├── src/
│   ├── api/
│   │   ├── insights_api.py
│   │   ├── main.py
│   │   ├── predict_api.py
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── insights_dashboard.py
│   ├── eda/
│   │   ├── exploratory_data_analysis.py
│   ├── models/
│   │   ├── baseline_model.py
│   │   ├── model_evaluation.py
│   │   ├── optimized_model.py
│   │   ├── train_lightgbm.py
│   ├── preprocessing/
│   │   ├── data_cleaning.py
│   │   ├── data_transformation.py
│   │   ├── feature_engineering.py
```

---

## 🏗 **Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/swathi-valluri/healthcare-ai-analysis.git
cd healthcare-ai-analysis
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 **How to Use**
### **1️⃣ Start FastAPI Backend**
Run the following command to start the API:
```bash
uvicorn src.api.main:app --reload
```
**API Endpoints:**
- `GET /insights/readmission_rates` → Returns readmission rates by age & race.
- `GET /insights/trends` → Returns readmission trends based on admission type.
- `POST /predict` → Predicts patient readmission probability.

### **2️⃣ Start the Streamlit Dashboard**
To launch the interactive dashboard:
```bash
streamlit run src/dashboard/app.py
```
- **Predict Readmission Risk**: Enter patient details to get readmission probability.
- **Visualize Trends**: Readmission rates based on age, race, and admission type.

---

## 🏥 **Data & Outputs Explained**
### **📊 Data Files**
- **`diabetic_data.csv`** → Raw dataset of hospital admissions.
- **`diabetic_data_final.csv`** → Processed dataset with transformed features.
- **`predictions.csv`** → Model predictions for patient readmission.

### **📈 Model Reports**
- **`logistic_regression_report.csv`** → Performance metrics for logistic regression.
- **`model_comparison.csv`** → Accuracy, precision, recall of different models.
- **`optimized_model_comparison.csv`** → Fine-tuned model performance (LightGBM, XGBoost, etc.).

### **🖼️ Visualizations**
- **`readmission_distribution.png`** → Readmission rate distribution.
- **`age_distribution.png`** → Age-wise patient distribution.
- **`correlation_matrix.png`** → Feature correlation heatmap.
- **`roc_auc_curve.png`** → Model ROC curve comparison.
- **`shap_feature_importance.png`** → Feature importance based on SHAP.

---

## 📜 **APIs & Scripts**
### **📌 API Services (`src/api/`)**
- `predict_api.py` → Predicts readmission probability.
- `insights_api.py` → Fetches insights on readmissions.

### **📌 Dashboard (`src/dashboard/`)**
- `app.py` → Streamlit-based predictor UI.
- `insights_dashboard.py` → Visualizes readmission trends.

### **📌 Model Training (`src/models/`)**
- `baseline_model.py` → Logistic regression and decision tree baseline models.
- `optimized_model.py` → Hyperparameter-tuned models.
- `train_lightgbm.py` → LightGBM model training.

### **📌 Data Processing (`src/preprocessing/`)**
- `data_cleaning.py` → Cleans missing and redundant data.
- `data_transformation.py` → Encodes categorical features.
- `feature_engineering.py` → Generates new predictive features.

---

## 🔍 **Example API Usage**
### **Predict Readmission using cURL**
```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
-H 'Content-Type: application/json' \
-d '{
    "race": 1, "gender": 0, "age": 50,
    "admission_type_id": 1, "discharge_disposition_id": 1,
    "admission_source_id": 1, "time_in_hospital": 5,
    "num_lab_procedures": 30, "num_procedures": 1,
    "num_medications": 10, "number_diagnoses": 7,
    "total_visits": 5, "comorbidity_score": 3,
    "change": 1, "diabetesMed": 1, "insulin": 0
}'
```
**Response:**
```json
{
    "readmission": 1,
    "probability": 99.75
}
```

---

## 📝 **License**
This project is licensed under the MIT License.

---

## 📌 **Contributing**
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-name`
5. Open a Pull Request.

---

## 📧 **Contact**
For any questions or feedback, reach out via [GitHub Issues](https://github.com/swathi-valluri/healthcare-ai-analysis/issues).

---
