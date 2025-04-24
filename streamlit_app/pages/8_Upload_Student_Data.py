import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Upload Student Data", page_icon="📥", layout="wide")
st.markdown("""
<h1 style='text-align: center;'>📥 Upload Student Data (Excel / Google Forms)</h1><hr>
""", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("Upload .xlsx or .csv file exported from Excel/Google Form", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.markdown("### 🧾 Preview of Uploaded Data")
        st.dataframe(df.head())

        # Optional save
        os.makedirs("data/uploads", exist_ok=True)
        save_path = os.path.join("data/uploads", uploaded_file.name)
        df.to_csv(save_path, index=False)
        st.info(f"✅ Saved to: {save_path}")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")

else:
    st.info("Upload a spreadsheet to get started.")

st.markdown("---")
st.caption("UTA MSDS 2025 · Google Form/Excel Data Integration")
