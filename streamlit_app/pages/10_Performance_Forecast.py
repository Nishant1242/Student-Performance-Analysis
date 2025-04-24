import streamlit as st
import pandas as pd
import plotly.express as px
from utils.loader import load_data

st.set_page_config(page_title="Performance Forecast", page_icon="📈", layout="wide")
st.title("📈 Student Score Forecasting")

# Load synthetic or historical data
df = load_data()

# --- Patch: simulate semester/year score trend ---
if 'term' not in df.columns:
    import numpy as np
    st.info("ℹ️ Historical data not found. Generating sample scores across 4 terms.")
    terms = ['Term 1', 'Term 2', 'Term 3', 'Term 4']
    df_long = pd.concat([
        df.assign(term=t,
                  math_score=lambda x: x['math_score'] + np.random.randint(-5, 5, size=len(x)),
                  reading_score=lambda x: x['reading_score'] + np.random.randint(-5, 5, size=len(x)),
                  writing_score=lambda x: x['writing_score'] + np.random.randint(-5, 5, size=len(x)))
        for t in terms
    ])
else:
    df_long = df.copy()

# Filter by gender or group
st.sidebar.header("🔍 Filter")
selected_gender = st.sidebar.selectbox("Gender", ["All"] + sorted(df_long["gender"].unique()))
if selected_gender != "All":
    df_long = df_long[df_long["gender"] == selected_gender]

# Melt scores
melted = pd.melt(df_long, id_vars=["term"], value_vars=["math_score", "reading_score", "writing_score"],
                 var_name="Subject", value_name="Score")

# Plot
st.subheader("📊 Average Score Trends Over Time")
fig = px.line(melted.groupby(["term", "Subject"]).Score.mean().reset_index(),
              x="term", y="Score", color="Subject", markers=True,
              title="Forecasted Subject Trends Across Terms")
st.plotly_chart(fig, use_container_width=True)

# Download option
st.subheader("📥 Download Synthetic Time-Series Data")
st.download_button("Download CSV", melted.to_csv(index=False), "term_score_trends.csv", "text/csv")

st.markdown("---")
st.caption("⏳ Simulated Trend Forecast · Group 7 · UTA MSDS")
