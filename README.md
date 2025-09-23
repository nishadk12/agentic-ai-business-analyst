# Agentic AI Business Analyst
An interactive **Streamlit** app that behaves like a junior **agentic** data/business analyst. 
Users upload CSV/Excel files; the agent plans and executes a workflow: **ingestion → EDA → modeling → reflection → reporting**, 
then produces **visual insights** and a downloadable **executive summary report**. Optional **Q&A** is powered by an LLM if an API key is provided.

---

## Demo (Local)
```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # on macOS/Linux
# .\.venv\Scripts\activate      # on Windows PowerShell

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) set your OpenAI key for Q&A
export OPENAI_API_KEY="sk-..."     # macOS/Linux
# $env:OPENAI_API_KEY="sk-..."     # Windows PowerShell

# 4) Run Streamlit
streamlit run app.py
```

> If no `OPENAI_API_KEY` is set, the Q&A tab will be disabled. The rest of the app works fully offline.

---

## Features
- **Upload & Profile**: CSV/Excel ingestion, schema detection, missing/outlier overview.
- **EDA**: auto summaries, distributions, correlation heatmap, time-series detection.
- **Modeling** (auto or manual):
  - **Regression**: LinearRegression, Ridge, Lasso, RandomForestRegressor, XGBRegressor
  - **Classification**: LogisticRegression, RandomForestClassifier, XGBClassifier
  - Auto metric selection (R² / RMSE for regression; F1 / ROC-AUC for classification)
- **Agentic Workflow**: planner → executor → reflector chooses the best-performing model and hyperparams from a small search space.
- **Reporting**: generates an **HTML executive summary** with charts and key findings for download.
- **Q&A (Optional)**: natural-language queries against your dataset using an LLM to plan small analyses.

---

## Project Structure
```
agentic-ai-business-analyst/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── agent/
│   ├── data_ingest.py
│   ├── eda.py
│   ├── modeling.py
│   ├── agent.py
│   ├── reporting.py
│   └── utils.py
├── pages/
│   ├── 1_⚙️_EDA.py
│   ├── 2_🤖_Modeling.py
│   ├── 3_🧭_Agent_Run.py
│   └── 4_💬_Q&A_(LLM_Optional).py
└── sample_data/
    └── retail_sales_demo.csv
```

---

## Deployment (Streamlit Community Cloud / Hugging Face)
- Push this repo to GitHub.
- On **Streamlit Community Cloud**, set `OPENAI_API_KEY` secret if you want Q&A.
- On **Hugging Face Spaces**, choose **Streamlit** SDK and add the same env secret.

---

## Resume Bullet (example)
> Built an **agentic AI Business Analyst** (Streamlit + Python) that autonomously ingests user datasets, performs EDA and predictive modeling (regression/classification with AutoML-style selection), generates executive reports, and supports optional LLM-based Q&A. Demonstrated end-to-end data engineering, ML, orchestration, and explainability.
