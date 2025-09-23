import os
import streamlit as st
import pandas as pd
from agent.data_ingest import load_any
from agent.utils import basic_clean

st.set_page_config(page_title="Agentic AI Business Analyst", page_icon="📊", layout="wide")

st.title("Agentic AI Business Analyst")
st.markdown("Upload a CSV/Excel file to get started. Use the sidebar to navigate to EDA, Modeling, the Agent run, or Q&A.")

if "df" not in st.session_state:
    st.session_state.df = None

file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
if file is not None:
    try:
        df = load_any(file)
        st.session_state.df = df
        st.success(f"Loaded dataset with shape {df.shape}.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to load file: {e}")

st.info("➡️ Next: Click **EDA** in the sidebar to explore, then **Modeling** or **Agent Run**.")
