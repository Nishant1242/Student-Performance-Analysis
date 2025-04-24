import streamlit as st
import pandas as pd
import plotly.express as px
from utils.loader import load_data

# Page setup
st.set_page_config(page_title="Teacher Insights", page_icon="📚", layout="wide")
st.title("📚 Teacher & Admin Insights")

# Load data
df = load_data()

# Patch: if 'at_risk' missing, simulate it
if 'at_risk' not in df.columns:
    import numpy as np
    st.warning("⚠️ 'at_risk' column not found — using random placeholder values for demo.")
    df['at_risk'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

# Key metrics
st.subheader("📈 Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(df))
col2.metric("Avg Score", f"{df['average_score'].mean():.2f}")
col3.metric("At-Risk Students", int(df['at_risk'].sum()))

# Risk group distribution
st.subheader("🚨 Risk Distribution")
fig = px.pie(df, names="at_risk", title="At-Risk vs Not At-Risk",
             color_discrete_sequence=px.colors.qualitative.Set1,
             labels={0: "Not At Risk", 1: "At Risk"})
st.plotly_chart(fig, use_container_width=True)

# Grouped stats
st.subheader("📊 Grouped Breakdown")
selected_feature = st.selectbox("Group by:", ["gender", "prep_course", "race/ethnicity", "parent_edu", "lunch"])
grouped = df.groupby(selected_feature)["at_risk"].agg(["count", "sum", "mean"]).reset_index()
grouped.columns = [selected_feature, "Total", "At Risk Count", "At Risk Rate"]

st.dataframe(grouped)

# Visualize risk by group
fig = px.bar(grouped, x=selected_feature, y="At Risk Rate", color="At Risk Rate",
             title=f"At-Risk Rate by {selected_feature.title()}",
             color_continuous_scale="oranges")
fig.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

# Download report
st.subheader("📥 Download Summary Table")
csv = grouped.to_csv(index=False)
st.download_button("Download CSV Report", csv, "risk_summary_by_group.csv", "text/csv")

st.markdown("---")
st.caption("📘 Group 7 · UTA Master's Project · 2025")
