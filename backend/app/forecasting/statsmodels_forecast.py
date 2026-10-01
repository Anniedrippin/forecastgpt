"""
Classical statistical forecasting (SARIMA) via statsmodels.

SARIMA = Seasonal AutoRegressive Integrated Moving Average. Unlike the
baselines, it explicitly models trend (via differencing / the "I") and
seasonality (via the seasonal (P, D, Q, s) terms), and it produces proper
confidence intervals from the model's estimated forecast error distribution
-- which is why this is the model used whenever the UI needs to show an
uncertainty band.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.core.logging_config import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore")  # statsmodels is chatty about convergence


def _guess_seasonal_period(n_obs: int, inferred_freq: str | None) -> int:
    """Pick a reasonable seasonal period `s` for SARIMA based on the data's
    frequency. This keeps the model from trying (and failing) to fit a
    52-week seasonal term on 40 rows of data."""
    if inferred_freq is None:
        return 7 if n_obs >= 21 else 1
    freq = inferred_freq.upper()
    if freq.startswith("D"):
        candidate = 7
    elif freq.startswith("W"):
        candidate = 52
    elif freq.startswith("M"):
        candidate = 12
    elif freq.startswith("Q"):
        candidate = 4
    else:
        candidate = 7
    # need at least 2 full seasonal cycles to estimate seasonality at all
    return candidate if n_obs >= candidate * 2 else 1


def sarima_forecast(
    train: pd.DataFrame,
    horizon: int,
    inferred_freq: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit SARIMA on the training series and forecast `horizon` steps ahead.

    Returns:
        (point_forecast, lower_ci, upper_ci) each of length `horizon`
    """
    y = train["value"].astype(float).values
    s = _guess_seasonal_period(len(y), inferred_freq)

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, s) if s > 1 else (0, 0, 0, 0)

    try:
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        pred = fitted.get_forecast(steps=horizon)
        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)
        lower, upper = ci[:, 0], ci[:, 1]
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.warning(f"SARIMA fit failed ({e}); falling back to simple trend model.")
        order = (1, 1, 0)
        model = SARIMAX(y, order=order, seasonal_order=(0, 0, 0, 0))
        fitted = model.fit(disp=False)
        pred = fitted.get_forecast(steps=horizon)
        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)
        lower, upper = ci[:, 0], ci[:, 1]

    return np.asarray(mean), np.asarray(lower), np.asarray(upper)
