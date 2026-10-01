"""
Forecast service.

`model_selector.compare_models` answers "which model is best?" by testing
models on a held-out validation window. This service answers the follow-up
question: "ok, now actually forecast the future with that model", producing
the real forward-looking forecast (dates beyond the end of the dataset)
plus a confidence interval where the model supports one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.baselines import moving_average_forecast, naive_forecast
from app.forecasting.evaluation import evaluate
from app.forecasting.model_selector import compare_models
from app.forecasting.preprocessing import time_based_split
from app.forecasting.statsmodels_forecast import sarima_forecast
from app.forecasting.xgboost_forecast import xgboost_forecast


def _future_dates(last_date: pd.Timestamp, horizon: int, freq: str | None) -> list[pd.Timestamp]:
    freq = freq or "D"
    offset = pd.tseries.frequencies.to_offset(freq)
    dates = []
    current = last_date
    for _ in range(horizon):
        current = current + offset
        dates.append(current)
    return dates


def _residual_std_band(df: pd.DataFrame, model_name: str, inferred_freq: str | None) -> float:
    """For models without native confidence intervals (naive, moving_average,
    xgboost), approximate an uncertainty band from the model's own
    validation-set residual standard deviation. This is a simplification
    (a real system might use bootstrapping/quantile regression) but it is
    honestly labeled as such and still gives a useful visual band."""
    validation_size = min(14, max(1, len(df) // 5))
    train, val = time_based_split(df, validation_size)
    funcs = {
        "naive": lambda: naive_forecast(train, len(val)),
        "moving_average": lambda: moving_average_forecast(train, len(val)),
        "xgboost": lambda: xgboost_forecast(train, len(val)),
    }
    y_pred = funcs[model_name]()
    residuals = val["value"].values.astype(float) - np.asarray(y_pred)[: len(val)]
    return float(np.std(residuals)) if len(residuals) > 1 else 0.0


def run_forecast(
    df: pd.DataFrame,
    horizon: int,
    model: str,
    inferred_freq: str | None,
) -> dict:
    last_date = df["date"].iloc[-1]
    future_dates = _future_dates(last_date, horizon, inferred_freq)

    lower = upper = None

    if model == "naive":
        point = naive_forecast(df, horizon)
    elif model == "moving_average":
        point = moving_average_forecast(df, horizon)
    elif model == "sarima":
        point, lower, upper = sarima_forecast(df, horizon, inferred_freq)
    elif model == "xgboost":
        point = xgboost_forecast(df, horizon)
    else:
        raise ValueError(f"Unknown model '{model}'")

    if lower is None:
        band = _residual_std_band(df, model, inferred_freq)
        lower = point - 1.96 * band
        upper = point + 1.96 * band

    forecast_points = [
        {"date": str(d.date()), "value": float(v), "lower": float(lo), "upper": float(up)}
        for d, v, lo, up in zip(future_dates, point, lower, upper)
    ]

    # Validation metrics for the chosen model, so the UI/agent can always
    # show "how good is this forecast expected to be"
    validation_size = min(horizon, max(1, len(df) // 5))
    train, val = time_based_split(df, validation_size)
    val_comparison = compare_models(df, validation_size, inferred_freq, models=[model])
    validation_metrics = val_comparison[0] if val_comparison else None

    return {
        "model_used": model,
        "horizon": horizon,
        "history": [
            {"date": str(row.date.date()), "value": float(row.value)} for row in df.itertuples()
        ],
        "forecast": forecast_points,
        "validation_metrics": validation_metrics,
    }
