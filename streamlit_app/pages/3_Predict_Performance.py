import streamlit as st
import pandas as pd
from utils.loader import load_model, load_data

# Page config
st.set_page_config(page_title="Performance Predictor", page_icon="🧠", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #4A4A4A;'>🧠 Predict Student Performance</h1>
    <hr style='margin-top: 0px;'>
""", unsafe_allow_html=True)

# Load model and data
model, le = load_model()
df = load_data()

# Sidebar styling
st.sidebar.header("📥 Input Student Details")

# Form
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", df["gender"].unique())
        race = st.selectbox("Race/Ethnicity", df["race/ethnicity"].unique())
        parent = st.selectbox("Parental Education", df["parent_edu"].unique())

    with col2:
        lunch = st.selectbox("Lunch Type", df["lunch"].unique())
        prep = st.selectbox("Test Preparation", df["prep_course"].unique())
        math = st.slider("Math Score", 0, 100, 60)
        reading = st.slider("Reading Score", 0, 100, 60)
        writing = st.slider("Writing Score", 0, 100, 60)

    submit = st.form_submit_button("🔮 Predict")

# Handle prediction
if submit:
    input_df = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parent_edu": [parent],
        "lunch": [lunch],
        "prep_course": [prep],
        "math_score": [math],
        "reading_score": [reading],
        "writing_score": [writing],
    })

    # Preprocessing
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    pred = model.predict(input_encoded)[0]
    label = le.inverse_transform([pred])[0]

    st.success(f"🎯 Predicted Performance Category: **{label}**")
    st.markdown("---")
    st.caption("🔍 Prediction powered by Decision Tree Classifier")
