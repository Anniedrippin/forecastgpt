"""
Runs every implemented model on a held-out validation window, scores them
with real metrics, and reports/selects the best one. This is the function
behind both `POST /forecast` (compare_all=True) and the agent's
"compare_models" tool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.baselines import moving_average_forecast, naive_forecast
from app.forecasting.evaluation import evaluate
from app.forecasting.preprocessing import time_based_split
from app.forecasting.statsmodels_forecast import sarima_forecast
from app.forecasting.xgboost_forecast import xgboost_forecast

MODEL_FUNCS = {
    "naive": lambda train, h, freq: naive_forecast(train, h),
    "moving_average": lambda train, h, freq: moving_average_forecast(train, h),
    "sarima": lambda train, h, freq: sarima_forecast(train, h, freq)[0],
    "xgboost": lambda train, h, freq: xgboost_forecast(train, h),
}


def compare_models(
    df: pd.DataFrame,
    horizon: int,
    inferred_freq: str | None = None,
    models: list[str] | None = None,
) -> list[dict]:
    """Fit every requested model on train data and evaluate against a
    validation window of length `horizon` held out from the end of `df`.
    Returns a list of {model, mae, rmse, mape} sorted best (lowest RMSE) first.
    """
    models = models or list(MODEL_FUNCS.keys())
    validation_size = min(horizon, max(1, len(df) // 5))
    train, val = time_based_split(df, validation_size)

    results = []
    y_true = val["value"].values.astype(float)

    for name in models:
        try:
            func = MODEL_FUNCS[name]
            y_pred = func(train, len(val), inferred_freq)
            y_pred = np.asarray(y_pred)[: len(y_true)]
            metrics = evaluate(y_true, y_pred)
            results.append({"model": name, **metrics})
        except Exception as e:  # keep comparing even if one model fails
            results.append({"model": name, "mae": float("inf"), "rmse": float("inf"), "mape": None, "error": str(e)})

    results.sort(key=lambda r: r["rmse"])
    return results


def best_model(comparison: list[dict]) -> str:
    return comparison[0]["model"]
