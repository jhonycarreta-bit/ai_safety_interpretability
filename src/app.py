import streamlit as st
import pandas as pd
from train_model import train

st.title("Model Training & Interpretability Demo")

uploaded = st.file_uploader("Upload CSV data", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Data preview:", df.head())
    df_clean = df.dropna()
    st.write("Processed data:", df_clean.describe())

    if st.button("Train Model"):
        train()
        st.success("Model trained and SHAP interpretability report generated.")
        with open("shap_summary.html", "r") as f:
            html = f.read()
        st.components.v1.html(html, height=600)
