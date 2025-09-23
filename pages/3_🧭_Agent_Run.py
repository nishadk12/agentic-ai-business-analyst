import streamlit as st
import pandas as pd
from agent.agent import AnalystAgent
from agent.reporting import render_report
from agent.utils import detect_problem_type

st.set_page_config(page_title="Agent Run · Agentic AI BA", page_icon="🧭", layout="wide")
st.title("🧭 Agentic Run (Plan → Execute → Reflect)")

df = st.session_state.get("df")
if df is None:
    st.warning("Please upload a dataset on the Home page first.")
    st.stop()

target = st.selectbox("Target column", options=df.columns.tolist())

goal = st.text_input("Agent Goal", value="Find the best model and explain key drivers.")
if st.button("Run Agent"):
    with st.spinner("Agent is working..."):
        agent = AnalystAgent(goal=goal)
        result = agent.run(df, target)

    st.success("Agent finished!")
    st.write("**Plan**", result.plan.steps)
    st.write("**Problem Type**", result.problem_type)
    st.write("**Best Model**", result.best_model_name)
    st.write("**Best Metrics**", result.best_metrics)
    st.write("**Leaderboard**")
    st.json(result.leaderboard)

    # Render report
    path = "executive_report.html"
    render_report(path, result.overview, result.problem_type, result.best_model_name, result.best_metrics, result.leaderboard)
    with open(path, "rb") as f:
        st.download_button("⬇️ Download Executive Summary (HTML)", f, file_name="executive_report.html", mime="text/html")
