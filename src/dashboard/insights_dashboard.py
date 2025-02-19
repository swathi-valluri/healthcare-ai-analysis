import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# 🌟 Set Page Title
st.set_page_config(page_title="Healthcare Insights Dashboard", page_icon="📊", layout="wide")

st.title("📊 Healthcare Insights Dashboard")
st.markdown("### 🔍 Analyze patient readmission trends and generate reports.")

API_URL = "http://127.0.0.1:8000/insights/"

# ---- Fetch Data ----
@st.cache_data
def fetch_data(endpoint):
    response = requests.get(API_URL + endpoint)
    return pd.DataFrame(response.json()) if response.status_code == 200 else None

# 📊 Readmission Rates by Age & Race
st.header("📌 Readmission Rates by Age & Race")
df_rates = fetch_data("readmission_rates")
if df_rates is not None:
    fig = px.bar(df_rates, x="age", y="readmission", color="race", barmode="group", title="Readmission Rates by Age & Race")
    st.plotly_chart(fig)
else:
    st.error("Failed to load readmission rates.")

# 📈 Readmission Trends by Admission Type
st.header("📌 Readmission Trends by Admission Type")
df_trends = fetch_data("trends")
if df_trends is not None:
    fig = px.line(df_trends, x="time_in_hospital", y="readmission", color="admission_type",
                  markers=True, title="Readmission Trends by Hospital Stay Duration")
    st.plotly_chart(fig)
else:
    st.error("Failed to load readmission trends.")

# 📂 Export Data for BI Tools
st.header("📂 Export Data for Tableau/Power BI")
st.write("Download healthcare insights data for further analysis.")

if st.button("📥 Download CSV"):
    response = requests.get(API_URL + "export")
    if response.status_code == 200:
        st.download_button(label="Download Data", data=response.text, file_name="readmission_data.csv", mime="text/csv")
    else:
        st.error("Failed to export data.")
