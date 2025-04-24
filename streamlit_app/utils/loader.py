import pandas as pd
import joblib
import streamlit as st
import os

@st.cache_resource
def load_model():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(project_root, "models", "decision_tree_model.pkl")
    encoder_path = os.path.join(project_root, "models", "performance_label_encoder.pkl")
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

@st.cache_data
def load_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(project_root, "data", "cleaned", "students_cleaned.csv")
    return pd.read_csv(data_path)
