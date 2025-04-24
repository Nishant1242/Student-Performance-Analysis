import streamlit as st
import pandas as pd
import joblib
import os
from utils.loader import load_data
from utils.shap_helper import load_shap_explainer, get_shap_values
from utils.pdf_generator import generate_risk_report
import shap
import plotly.express as px

# Page layout
st.set_page_config(page_title="At-Risk Analyzer", page_icon="⚠️", layout="wide")
st.markdown("""
    <h1 style='text-align: center;'>⚠️ At-Risk Student Predictor</h1><hr>
""", unsafe_allow_html=True)

# Load model and dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_path = os.path.join(BASE_DIR, "models", "at_risk_model.pkl")
model = joblib.load(model_path)
df = load_data()

# Sidebar Inputs
st.sidebar.header("🔍 Enter Student Information")
gender = st.sidebar.selectbox("Gender", df["gender"].unique())
race = st.sidebar.selectbox("Race/Ethnicity", df["race/ethnicity"].unique())
parent = st.sidebar.selectbox("Parental Education", df["parent_edu"].unique())
lunch = st.sidebar.selectbox("Lunch Type", df["lunch"].unique())
prep = st.sidebar.selectbox("Test Prep", df["prep_course"].unique())

# Scores
st.markdown("### 🎛️ What-If Simulator: Adjust Student Scores")
col1, col2, col3 = st.columns(3)
math = col1.slider("Math Score", 0, 100, 50)
reading = col2.slider("Reading Score", 0, 100, 50)
writing = col3.slider("Writing Score", 0, 100, 50)

# Prepare input
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
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict + Confidence
proba = model.predict_proba(input_encoded)[0]
pred = proba.argmax()
confidence = proba[pred] * 100
risk_label = "🚨 At Risk" if pred == 1 else "✅ Not At Risk"
st.success(f"**Predicted Status:** {risk_label} | 🧠 Confidence: {confidence:.2f}%")



# SHAP
st.markdown("### 📌 Feature Impact (SHAP Explanation)")
explainer = load_shap_explainer(model_path, input_encoded)
shap_values = get_shap_values(explainer, input_encoded)
shap_df = pd.DataFrame({
    "Feature": input_encoded.columns,
    "SHAP Impact": shap_values.values[0]
}).sort_values(by="SHAP Impact", key=abs, ascending=False)

shap_df_clipped = shap_df.copy()
shap_df_clipped["SHAP Impact"] = shap_df_clipped["SHAP Impact"].apply(lambda x: 0 if abs(x) < 0.01 else x)

# Use shap_df_clipped for plot
# Use shap_df (original) for report


# SHAP Chart
fig = px.bar(shap_df.head(10), x="SHAP Impact", y="Feature", orientation='h',
             color="SHAP Impact", color_continuous_scale="RdBu",
             title="Top 10 Feature Contributions")
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

with st.expander("🔍 Show SHAP Table"):
    st.dataframe(shap_df)

# 📚 Study Recommendations
# Study Recommendations
st.markdown("### 📚 Smart Study Recommendations")
subject_keywords = ["math", "reading", "writing"]
rec_lines = []

top_feature = shap_df.iloc[0]["Feature"]
if any(sub in top_feature for sub in subject_keywords):
    top_subj = next(sub for sub in subject_keywords if sub in top_feature)
    rec_lines.append(f"- Focus more on **{top_subj.title()}**. Allocate at least 5 extra study hours/week.")

if prep == "none":
    rec_lines.append("- 📝 **Enroll in a test prep course** for improved performance.")

if rec_lines:
    st.info("\\n".join(rec_lines))
else:
    st.success("🎉 Your study pattern looks well balanced!")

# PDF Download
if st.button("📄 Download Risk Report as PDF"):
    report_path = generate_risk_report(input_df, risk_label, shap_df.head(10))
    with open(report_path, "rb") as f:
        st.download_button(label="⬇️ Click to Download PDF", data=f,
                           file_name="student_risk_report.pdf", mime="application/pdf")
