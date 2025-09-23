from __future__ import annotations
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any
from .utils import detect_problem_type, basic_clean
from .modeling import fit_and_score
from .eda import profile_overview

@dataclass
class Plan:
    goal: str
    steps: list[str] = field(default_factory=lambda: [
        "clean_data", "profile", "determine_problem_type", "train_models", "select_best", "summarize"
    ])

@dataclass
class AgentResult:
    plan: Plan
    problem_type: str
    best_model_name: str
    best_metrics: Dict[str, float]
    leaderboard: Dict[str, Dict[str, float]]
    overview: Dict[str, Any]

class AnalystAgent:
    def __init__(self, goal: str = "Find the best model and explain key drivers."):
        self.plan = Plan(goal=goal)

    def run(self, df: pd.DataFrame, target: str) -> AgentResult:
        # Step 1: clean
        df = basic_clean(df)

        # Step 2: profile
        overview = profile_overview(df)

        # Step 3: determine problem type
        problem_type = detect_problem_type(df, target)

        # Step 4 & 5: train & select
        best, leaderboard = fit_and_score(df, target, problem_type)
        best_name = best["name"]
        best_metrics = best["metrics"]

        return AgentResult(
            plan=self.plan,
            problem_type=problem_type,
            best_model_name=best_name,
            best_metrics=best_metrics,
            leaderboard=leaderboard,
            overview=overview
        )
