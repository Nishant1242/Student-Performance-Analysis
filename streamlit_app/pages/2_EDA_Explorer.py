import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from utils.loader import load_data

# --- Page Configuration ---
st.set_page_config(page_title="Student Performance Dashboard", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem !important;
            }
            h1 {
                font-size: 1.5rem !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #4A4A4A;'>🎓 Student Performance Analysis</h1>
    <hr style='margin-top: 0px;'>
""", unsafe_allow_html=True)

# --- Load Data ---
df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Dataset")
st.sidebar.markdown("### 🎯 Select Filters")
gender = st.sidebar.selectbox("Gender", ["All"] + sorted(df["gender"].unique()))
prep = st.sidebar.selectbox("Test Preparation", ["All"] + sorted(df["prep_course"].unique()))
race = st.sidebar.selectbox("Race/Ethnicity", ["All"] + sorted(df["race/ethnicity"].unique()))
lunch = st.sidebar.selectbox("Lunch Type", ["All"] + sorted(df["lunch"].unique()))

# --- Filter Data ---
filtered_df = df.copy()
if gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender]
if prep != "All":
    filtered_df = filtered_df[filtered_df["prep_course"] == prep]
if race != "All":
    filtered_df = filtered_df[filtered_df["race/ethnicity"] == race]
if lunch != "All":
    filtered_df = filtered_df[filtered_df["lunch"] == lunch]

# --- Key Metrics ---
st.subheader("📊 Key Statistics")
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Students", len(filtered_df))
col2.metric("📈 Avg. Score", f"{filtered_df['average_score'].mean():.2f}")
col3.metric("🏅 Most Common Category", filtered_df['performance'].value_counts().idxmax())

# --- Summary Table ---
st.subheader("📋 Dataset Overview")
st.dataframe(filtered_df.describe(include='all').astype(str), use_container_width=True)

# --- Performance Distribution ---
st.subheader("🎯 Performance Category Distribution")
fig = px.histogram(filtered_df, x="performance", color="performance", template="plotly_white",
                   title="Distribution of Performance Categories")
fig.update_layout(title_x=0.3, margin=dict(l=40, r=40, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# --- Subject Score Distributions ---
st.subheader("📚 Subject-wise Score Distribution")
score_option = st.selectbox("Select Subject", ["math_score", "reading_score", "writing_score"])
fig = px.histogram(filtered_df, x=score_option, color="gender", marginal="violin", nbins=20,
                   title=f"{score_option.replace('_', ' ').title()} Distribution by Gender",
                   template="plotly_white")
fig.update_layout(title_x=0.3)
st.plotly_chart(fig, use_container_width=True)

# --- Parental Education Boxplot ---
st.subheader("🎓 Score by Parental Education")
fig = px.box(filtered_df, x="parent_edu", y="average_score", color="gender",
             title="Average Score by Parental Education",
             template="plotly_white")
fig.update_layout(title_x=0.3)
st.plotly_chart(fig, use_container_width=True)

# --- Correlation Heatmap ---
st.subheader("🧠 Correlation Heatmap")
corr_matrix = filtered_df.select_dtypes(include=np.number).corr()
z = np.round(corr_matrix.values, 2)
x = list(corr_matrix.columns)
y = list(corr_matrix.index)
heatmap = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z,
                                      colorscale='RdBu', showscale=True, zmin=-1, zmax=1)
heatmap.update_layout(title="Correlation Between Score Features", template="plotly_white")
st.plotly_chart(heatmap, use_container_width=True)

# --- Donut Chart ---
st.subheader("🍩 Performance Donut Chart")
perf_counts = filtered_df['performance'].value_counts().sort_index()
fig = px.pie(values=perf_counts.values, names=perf_counts.index, hole=0.5,
             title="Performance Category Share", template="plotly_white",
             color_discrete_sequence=px.colors.qualitative.Set3)
fig.update_layout(title_x=0.3)
st.plotly_chart(fig, use_container_width=True)

# --- Sunburst Chart ---
st.subheader("🌞 Sunburst: Gender > Parent Edu > Performance")
fig = px.sunburst(filtered_df, path=["gender", "parent_edu", "performance"], color="performance",
                  template="plotly_white", title="Performance by Gender and Parental Education")
fig.update_layout(title_x=0.3)
st.plotly_chart(fig, use_container_width=True)

# --- Parallel Coordinates ---
st.subheader("🔗 Parallel Coordinates")
parallel_df = filtered_df.copy()
performance_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
parallel_df['performance_code'] = parallel_df['performance'].map(performance_map)
fig = px.parallel_coordinates(parallel_df, color="performance_code",
                              dimensions=["math_score", "reading_score", "writing_score", "average_score"],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              labels={"performance_code": "Performance Level"},
                              title="Score Profiles by Performance", template="plotly_white")
fig.update_layout(title_x=0.3)
st.plotly_chart(fig, use_container_width=True)

# --- Scatter Matrix ---
st.subheader("📈 Score Interactions (Scatter Matrix)")
fig = px.scatter_matrix(filtered_df,
                        dimensions=["math_score", "reading_score", "writing_score", "average_score"],
                        color="performance",
                        title="Pairwise Score Relationships",
                        template="plotly_white")
fig.update_layout(title_x=0.3)
st.plotly_chart(fig, use_container_width=True)

# --- Download Filtered Dataset ---
st.subheader("📥 Download Filtered Data")
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_student_performance.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("🚀 Built for UTA Master's Data Science Project · © 2025")
