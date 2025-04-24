import streamlit as st
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
from utils.loader import load_data
from utils.shap_helper import load_shap_explainer, get_shap_values

# --- Page Setup ---
st.set_page_config(page_title="Model Insights", page_icon="📊", layout="wide")
st.markdown("""
    <h1 style='text-align: center;'>📊 Global Model Insights</h1><hr>
""", unsafe_allow_html=True)

# --- Load Model and Data ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_path = os.path.join(BASE_DIR, "models", "at_risk_model.pkl")
model = joblib.load(model_path)
df = load_data()

# --- Prepare Data for SHAP ---
if 'at_risk' not in df.columns:
    df['at_risk'] = df['average_score'].apply(lambda x: 1 if x < 60 else 0)

drop_cols = [col for col in ["performance", "average_score", "at_risk"] if col in df.columns]
X = pd.get_dummies(df.drop(columns=drop_cols), drop_first=True)
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

# --- SHAP Explainer ---
st.markdown("### 🔍 SHAP Feature Importance Explorer")
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# --- SHAP DataFrame ---
shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False).reset_index()
mean_abs_shap.columns = ["Feature", "Mean SHAP Value"]

# --- Interactive Plotly Bar ---
st.markdown("### 📈 Interactive Global SHAP Summary")
fig = px.bar(mean_abs_shap.head(15), x="Mean SHAP Value", y="Feature", orientation='h',
             color="Mean SHAP Value", color_continuous_scale="Blues", title="Top 15 Important Features")
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

# --- Raw Table Option ---
with st.expander("🔎 Show SHAP Value Table"):
    st.dataframe(mean_abs_shap)

st.markdown("---")
st.caption("Built for UTA Master's Data Science Project · Global SHAP Insights · 2025")