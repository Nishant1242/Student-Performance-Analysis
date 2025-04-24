import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.loader import load_data

st.set_page_config(page_title="Student Clustering", page_icon="🧩", layout="wide")
st.markdown("""
<h1 style='text-align: center;'>🧩 Student Persona Clustering</h1><hr>
""", unsafe_allow_html=True)

# Load data
df = load_data()

# Prepare numeric features for clustering
X = df[["math_score", "reading_score", "writing_score"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Label personas manually (optional refinement)
persona_map = {
    0: "📘 Struggling",
    1: "📗 Consistent Performer",
    2: "📕 High Achiever",
    3: "📙 Improving"
}
df['persona'] = df['cluster'].map(persona_map)

# Visualization
st.markdown("### 🎯 Student Clusters by Subject Scores")
fig = px.scatter_3d(
    df, x="math_score", y="reading_score", z="writing_score",
    color="persona", symbol="gender",
    title="3D Cluster of Student Personas",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Show cluster distribution
st.markdown("### 📊 Persona Distribution")
st.dataframe(df['persona'].value_counts().reset_index().rename(columns={"index": "Persona", "persona": "Count"}))

st.markdown("---")
st.caption("UTA MSDS 2025 · KMeans Student Clustering Insight")
