"""
Baseline forecasting models.

Baselines matter for a portfolio project because they answer the question
every stakeholder asks: "is your fancy model actually better than doing
almost nothing?" If XGBoost or SARIMA can't beat a naive baseline, that's an
important (and honest) finding, not a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every step of the horizon."""
    last_value = train["value"].iloc[-1]
    return np.full(horizon, last_value, dtype=float)


def moving_average_forecast(train: pd.DataFrame, horizon: int, window: int = 7) -> np.ndarray:
    """Repeat the mean of the last `window` observations for every step of
    the horizon. A simple but often surprisingly strong baseline for noisy,
    non-trending series."""
    window = min(window, len(train))
    avg = train["value"].iloc[-window:].mean()
    return np.full(horizon, avg, dtype=float)
