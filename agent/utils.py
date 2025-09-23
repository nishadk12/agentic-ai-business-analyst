from __future__ import annotations
import pandas as pd
import numpy as np

def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    y = df[target]
    # Heuristic classification vs regression
    if y.dtype.kind in "ifu":
        # numeric; if few unique values, maybe classification
        nunique = y.nunique(dropna=True)
        if nunique <= 10:
            return "classification"
        return "regression"
    else:
        return "classification"

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Simple cleaning: strip column names, drop duplicates
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()
    return df

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().sort_values(ascending=False)
    return miss[miss>0].to_frame(name="missing_count").assign(missing_pct=lambda x: x['missing_count']/len(df))

def infer_datetime_cols(df: pd.DataFrame) -> list[str]:
    dt_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_cols.append(c)
        else:
            try:
                parsed = pd.to_datetime(df[c], errors="raise", utc=False, infer_datetime_format=True)
                # if conversion works for >= 80% non-null rows, consider datetime
                non_null = df[c].notna().sum()
                if non_null>0 and (parsed.notna().sum()/non_null) > 0.8:
                    dt_cols.append(c)
            except Exception:
                pass
    return dt_cols
