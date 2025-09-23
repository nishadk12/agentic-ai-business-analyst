from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    return ColumnTransformer(transformers=transformers)

def split_data(df: pd.DataFrame, target: str, test_size: float=0.2, random_state: int=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_models(problem_type: str):
    if problem_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.001),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "XGBRegressor": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1),
        }
    else:
        return {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=42),
            "XGBClassifier": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, eval_metric="logloss"),
        }

def evaluate(problem_type: str, y_true, y_pred, y_proba=None) -> Dict[str, float]:
    if problem_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {"RMSE": rmse, "R2": r2}
    else:
        # for binary/multiclass we'll do weighted F1; ROC-AUC only for binary if probas available
        f1 = float(f1_score(y_true, y_pred, average="weighted"))
        metrics = {"F1_weighted": f1}
        if y_proba is not None and (np.array(y_true).nunique() if hasattr(np.array(y_true), "nunique") else pd.Series(y_true).nunique()) == 2:
            try:
                auc = float(roc_auc_score(y_true, y_proba[:,1]))
                metrics["ROC_AUC"] = auc
            except Exception:
                pass
        return metrics

def fit_and_score(df: pd.DataFrame, target: str, problem_type: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    X_train, X_test, y_train, y_test = split_data(df, target)
    pre = build_preprocessor(X_train)
    models = get_models(problem_type)

    results = {}
    best = {"name": None, "pipeline": None, "metrics": None}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        y_proba = None
        if problem_type != "regression":
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception:
                pass
        metrics = evaluate(problem_type, y_test, y_pred, y_proba)
        results[name] = metrics

        # select best by R2 (regression) or F1 (classification)
        if best["metrics"] is None:
            best = {"name": name, "pipeline": pipe, "metrics": metrics}
        else:
            if problem_type == "regression":
                if metrics.get("R2", -999) > best["metrics"].get("R2", -999):
                    best = {"name": name, "pipeline": pipe, "metrics": metrics}
            else:
                if metrics.get("F1_weighted", -999) > best["metrics"].get("F1_weighted", -999):
                    best = {"name": name, "pipeline": pipe, "metrics": metrics}

    return best, results
