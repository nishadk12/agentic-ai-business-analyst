from __future__ import annotations
import io
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from .utils import summarize_missing, infer_datetime_cols

def profile_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing": summarize_missing(df).to_dict("index") if df.isna().any().any() else {},
    }

def histograms(df: pd.DataFrame, max_cols: int = 12):
    figs = []
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns][:max_cols]
    for c in num_cols:
        fig = px.histogram(df, x=c, nbins=30, title=f"Distribution: {c}")
        figs.append(fig)
    return figs

def correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr(numeric_only=True)
    fig = px.imshow(corr, title="Correlation Heatmap")
    return fig

def time_series_plot(df: pd.DataFrame):
    dt_cols = infer_datetime_cols(df)
    if not dt_cols:
        return None
    # pick the first datetime column and a numeric column
    dt = dt_cols[0]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != dt]
    if not num_cols:
        return None
    y = num_cols[0]
    try:
        sorted_df = df.sort_values(dt)
        fig = px.line(sorted_df, x=dt, y=y, title=f"Time Series: {y} over {dt}")
        return fig
    except Exception:
        return None
