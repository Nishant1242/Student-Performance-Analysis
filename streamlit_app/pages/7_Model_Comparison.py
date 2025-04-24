import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from utils.loader import load_data

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
st.markdown("""
<h1 style='text-align: center;'>⚖️ Model Comparison Dashboard</h1><hr>
""", unsafe_allow_html=True)

# Load Data
df = load_data()
df['at_risk'] = df['average_score'].apply(lambda x: 1 if x < 60 else 0)

# Prepare Data
X = pd.get_dummies(df.drop(columns=["performance", "average_score", "at_risk"]), drop_first=True)
y = df['at_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": round(acc, 3)})

# Show Results
st.markdown("### 🧪 Accuracy Comparison")
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
st.dataframe(results_df)

# Confusion Matrix
st.markdown("### 📉 Confusion Matrix (Top Model)")
top_model_name = results_df.iloc[0]['Model']
top_model = models[top_model_name]
y_pred_top = top_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_top)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not At Risk", "At Risk"], yticklabels=["Not At Risk", "At Risk"])
plt.title(f"Confusion Matrix - {top_model_name}")
st.pyplot(fig)

# Report
st.markdown("### 📋 Classification Report")
st.text(classification_report(y_test, y_pred_top, target_names=["Not At Risk", "At Risk"]))

st.markdown("---")
st.caption("UTA MSDS 2025 · Binary Model Evaluation")