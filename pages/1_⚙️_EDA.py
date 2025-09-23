import streamlit as st
import pandas as pd
from agent.eda import profile_overview, histograms, correlation_heatmap, time_series_plot

st.set_page_config(page_title="EDA · Agentic AI BA", page_icon="⚙️", layout="wide")
st.title("⚙️ Exploratory Data Analysis (EDA)")

df = st.session_state.get("df")
if df is None:
    st.warning("Please upload a dataset on the Home page first.")
    st.stop()

overview = profile_overview(df)
st.subheader("Overview")
c1,c2,c3 = st.columns(3)
c1.metric("Rows", overview["rows"])
c2.metric("Columns", overview["cols"])
c3.metric("Missing Cols", len(overview["missing"]) if overview["missing"] else 0)

st.write("**Dtypes**")
st.json(overview["dtypes"])
if overview["missing"]:
    st.write("**Missing Values**")
    st.json(overview["missing"])

st.subheader("Distributions")
for fig in histograms(df, max_cols=9):
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Correlation Heatmap")
fig_corr = correlation_heatmap(df)
if fig_corr:
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Not enough numeric columns for a heatmap.")

st.subheader("Time Series (Auto-detected)")
fig_ts = time_series_plot(df)
if fig_ts:
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("No clear time column + numeric target detected.")
