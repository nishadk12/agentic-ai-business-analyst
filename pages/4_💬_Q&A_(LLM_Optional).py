import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Q&A · Agentic AI BA", page_icon="💬", layout="wide")
st.title("💬 Q&A (LLM Optional)")

df = st.session_state.get("df")
if df is None:
    st.warning("Please upload a dataset on the Home page first.")
    st.stop()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.info("Set OPENAI_API_KEY to enable LLM-based Q&A. (The rest of the app works without it.)")
    st.stop()

# Lightweight LLM usage without tight coupling; keep it simple
from openai import OpenAI
client = OpenAI(api_key=api_key)

st.write("Ask natural-language questions about your data. Example: 'Which category has the highest average sales?'")
q = st.text_area("Your question")
if st.button("Ask"):
    if not q.strip():
        st.warning("Type a question.")
    else:
        # We send a compact schema + head sample to the LLM for context.
        schema = {c: str(t) for c,t in df.dtypes.items()}
        head_json = df.head(20).to_dict(orient="list")
        prompt = f"""You are a data analyst. A user uploaded a dataset.
Schema: {schema}
Sample (first 20 rows): {head_json}

Task: Answer the user's question STRICTLY using Python/pandas logic in a short explanation.
If aggregation is needed, describe briefly how you'd compute it. Do not fabricate columns.
User question: {q}
"""
        with st.spinner("Thinking..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content": prompt}],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content
        st.markdown("**Answer:**")
        st.write(answer)
