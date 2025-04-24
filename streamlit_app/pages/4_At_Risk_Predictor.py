import streamlit as st
import pandas as pd
import joblib
import os
from utils.loader import load_data
from utils.shap_helper import load_shap_explainer, get_shap_values
import shap

# Page layout
st.set_page_config(page_title="At-Risk Analyzer", page_icon="⚠️", layout="wide")
st.markdown("<h1 style='text-align: center;'>⚠️ At-Risk Student Predictor</h1><hr>", unsafe_allow_html=True)

# Dynamically resolve absolute model path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_path = os.path.join(BASE_DIR, "models", "at_risk_model.pkl")
model = joblib.load(model_path)

# Load dataset
df = load_data()

# Sidebar Inputs
st.sidebar.header("🔍 Enter Student Information")

gender = st.sidebar.selectbox("Gender", df["gender"].unique())
race = st.sidebar.selectbox("Race/Ethnicity", df["race/ethnicity"].unique())
parent = st.sidebar.selectbox("Parental Education", df["parent_edu"].unique())
lunch = st.sidebar.selectbox("Lunch Type", df["lunch"].unique())
prep = st.sidebar.selectbox("Test Prep", df["prep_course"].unique())

# Score Sliders
st.markdown("### 🎛️ What-If Simulator: Adjust Student Scores")
col1, col2, col3 = st.columns(3)
math = col1.slider("Math Score", 0, 100, 50)
reading = col2.slider("Reading Score", 0, 100, 50)
writing = col3.slider("Writing Score", 0, 100, 50)

# Create input DataFrame
input_df = pd.DataFrame([{
    "gender": gender,
    "race/ethnicity": race,
    "parent_edu": parent,
    "lunch": lunch,
    "prep_course": prep,
    "math_score": math,
    "reading_score": reading,
    "writing_score": writing
}])

# Encode input
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction
pred = model.predict(input_encoded)[0]
risk_label = "🚨 At Risk" if pred == 1 else "✅ Not At Risk"
st.success(f"**Predicted Status:** {risk_label}**")

# SHAP Explanation
# SHAP Explanation
import matplotlib.pyplot as plt

# SHAP Explanation
st.markdown("### 📌 Feature Impact (SHAP Explanation)")

explainer = load_shap_explainer(model_path, input_encoded)
shap_values = get_shap_values(explainer, input_encoded)

# Plot SHAP bar chart for top features
fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.bar(shap_values[0], max_display=10, show=False)
st.pyplot(fig)


