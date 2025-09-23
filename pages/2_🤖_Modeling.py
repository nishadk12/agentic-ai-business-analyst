import streamlit as st
import pandas as pd
from agent.utils import detect_problem_type
from agent.modeling import fit_and_score

st.set_page_config(page_title="Modeling · Agentic AI BA", page_icon="🤖", layout="wide")
st.title("🤖 Modeling")

df = st.session_state.get("df")
if df is None:
    st.warning("Please upload a dataset on the Home page first.")
    st.stop()

cols = df.columns.tolist()
target = st.selectbox("Select target column", options=cols, index=len(cols)-1 if cols else 0)

if target:
    problem_guess = detect_problem_type(df, target)
    problem_type = st.radio("Problem Type", options=["regression","classification"], index=0 if problem_guess=="regression" else 1, help="Auto-guess based on target dtype/uniques.")

    if st.button("Train & Compare Models"):
        with st.spinner("Training models..."):
            best, leaderboard = fit_and_score(df, target, problem_type)
        st.success(f"Best model: {best['name']}")
        st.write("**Best metrics**", best["metrics"])
        st.write("**Leaderboard**")
        st.json(leaderboard)
else:
    st.info("Select a target to continue.")
