import streamlit as st

# Page config
st.set_page_config(page_title="About This Project", page_icon="📘", layout="wide")

# Responsive style block
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

# Title
st.markdown("""
    <h1 style='text-align: center; color: #4A4A4A;'>📘 About This Project</h1>
    <hr style='margin-top: 0px;'>
""", unsafe_allow_html=True)

# Project content
st.markdown("""
### 🎯 Objective
Analyze and predict student performance using demographic and academic features through interactive data visualization and machine learning.

### 🧠 Features Used
- Gender
- Race/Ethnicity
- Parental Education
- Lunch Type
- Test Preparation Course
- Math, Reading, and Writing Scores

### 🔍 EDA Highlights
- Interactive filters for deep-dive analysis
- Distribution plots, boxplots, and violin plots by category
- Correlation heatmaps, parallel coordinates, scatter matrix
- Sunburst and donut visualizations for hierarchical and proportional analysis

### 🧪 Machine Learning Models Evaluated
- Logistic Regression
- Decision Tree ✅ *(Best Accuracy)*
- Random Forest

### 🧰 Tools & Libraries
- **Python**, **Pandas**, **Scikit-learn**, **Joblib**
- **Plotly**, **Streamlit**, **SHAP**, **pdfkit**

### 🏆 Results & Deliverables
- Achieved **93.5% accuracy** using a Decision Tree Classifier
- Built a live prediction engine within an interactive Streamlit dashboard
- Integrated SHAP explanations for individual and global insights
- Clustering via KMeans for student personas
- PDF risk reports and What-if simulator for interventions

---
📅 **Developed for:** Final Project – Master's in Data Science  
🏫 **University:** University of Texas at Arlington  
👥 **Group 7 Members:** Nishant · Ayush · Shilp · Harsh  
📍 **Year:** 2025
---
""")