# utils/shap_helper.py

import shap
import pandas as pd
import joblib

def load_shap_explainer(model_path, X_sample):
    model = joblib.load(model_path)
    explainer = shap.Explainer(model, X_sample)
    return explainer

def get_shap_values(explainer, X_input):
    shap_values = explainer(X_input)
    return shap_values
