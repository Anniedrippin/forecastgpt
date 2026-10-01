"""
Descriptive time-series analysis: trend, seasonality, anomalies, and a
dataset profile. These are the functions the "dataset analysis" node and
the agent's tools call directly -- always real statistics, never an LLM
guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose

from app.forecasting.statsmodels_forecast import _guess_seasonal_period


def profile_dataset(df: pd.DataFrame, inferred_freq: str | None) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_missing": int(df["value"].isna().sum()),
        "mean": float(df["value"].mean()),
        "std": float(df["value"].std()),
        "min": float(df["value"].min()),
        "max": float(df["value"].max()),
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "inferred_frequency": inferred_freq,
    }


def analyze_trend(df: pd.DataFrame) -> dict:
    """Fit a simple linear regression of value ~ time index to characterize
    the overall trend direction, slope, and how well a straight line
    explains the data (R^2)."""
    x = np.arange(len(df))
    y = df["value"].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if abs(slope) < 1e-9 or r_squared < 0.05:
        direction = "flat / no clear linear trend"
    elif slope > 0:
        direction = "upward"
    else:
        direction = "downward"

    summary = (
        f"The series shows a {direction} trend "
        f"(slope \u2248 {slope:.4f} per period, R\u00b2 = {r_squared:.3f})."
    )
    return {
        "direction": direction,
        "slope_per_period": float(slope),
        "r_squared": float(r_squared),
        "summary": summary,
    }


def analyze_seasonality(df: pd.DataFrame, inferred_freq: str | None) -> dict:
    """Use statsmodels' classical seasonal_decompose to estimate how much of
    the series' variance is explained by a repeating seasonal pattern."""
    n = len(df)
    period = _guess_seasonal_period(n, inferred_freq)

    if period <= 1 or n < period * 2:
        return {
            "seasonal_period_detected": None,
            "seasonality_strength": 0.0,
            "is_seasonal": False,
            "summary": "Not enough data to reliably detect seasonality "
            f"(need at least {max(period * 2, 14)} rows).",
        }

    series = df.set_index("date")["value"]
    try:
        decomposition = seasonal_decompose(series, period=period, model="additive", extrapolate_trend="period")
        seasonal = decomposition.seasonal.values
        resid = decomposition.resid.dropna().values
        detrended = series.values - decomposition.trend.values
        detrended = detrended[~np.isnan(detrended)]
        var_seasonal = np.var(seasonal)
        var_total = np.var(detrended) if len(detrended) else np.var(series.values)
        strength = float(max(0.0, min(1.0, 1 - np.var(resid) / (np.var(series.values) + 1e-9))))
    except Exception:
        strength = 0.0

    is_seasonal = strength > 0.3
    summary = (
        f"{'Strong' if strength > 0.6 else 'Moderate' if is_seasonal else 'Weak/no'} "
        f"seasonality detected with an estimated period of {period} "
        f"(strength score {strength:.2f} on a 0-1 scale)."
    )
    return {
        "seasonal_period_detected": int(period),
        "seasonality_strength": strength,
        "is_seasonal": is_seasonal,
        "summary": summary,
    }


def detect_anomalies(df: pd.DataFrame, method: str = "zscore", contamination: float = 0.05) -> dict:
    """Detect anomalous points using one of two real statistical methods:

      - "zscore": flags points whose rolling z-score exceeds 3 standard
        deviations from a local rolling mean. Simple, explainable, fast.
      - "isolation_forest": uses scikit-learn's IsolationForest, an
        unsupervised ensemble method, on (value, rolling_diff) features.
        Better at catching contextual anomalies zscore misses.
    """
    values = df["value"].astype(float).values
    dates = df["date"]

    if method == "isolation_forest":
        window = min(7, max(2, len(df) // 10))
        rolling_mean = df["value"].rolling(window=window, min_periods=1, center=True).mean()
        diff = (df["value"] - rolling_mean).values.reshape(-1, 1)
        features = np.hstack([values.reshape(-1, 1), diff])
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(features)
        scores = -clf.score_samples(features)  # higher = more anomalous
        anomaly_idx = np.where(preds == -1)[0]
    else:
        method = "zscore"
        window = min(14, max(3, len(df) // 5))
        rolling_mean = df["value"].rolling(window=window, min_periods=1, center=True).mean()
        rolling_std = df["value"].rolling(window=window, min_periods=1, center=True).std().replace(0, np.nan)
        z = ((df["value"] - rolling_mean) / rolling_std).fillna(0).values
        scores = np.abs(z)
        anomaly_idx = np.where(scores > 3.0)[0]

    anomalies = []
    for idx in anomaly_idx:
        anomalies.append(
            {
                "index": int(idx),
                "date": str(dates.iloc[idx].date()),
                "value": float(values[idx]),
                "reason": "far from local rolling mean" if method == "zscore" else "isolated by IsolationForest",
                "score": float(scores[idx]),
            }
        )
    anomalies.sort(key=lambda a: -a["score"])

    summary = (
        f"Detected {len(anomalies)} anomalous point(s) out of {len(df)} using the '{method}' method."
        if anomalies
        else f"No significant anomalies detected using the '{method}' method."
    )
    return {"method": method, "n_anomalies": len(anomalies), "anomalies": anomalies, "summary": summary}
