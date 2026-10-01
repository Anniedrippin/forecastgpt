"""
Preprocessing utilities shared by every forecasting model.

Why time-based splitting matters
---------------------------------
A random train/test split leaks future information into the training set
(the model would "see" data points that occur after the ones it's being
validated on). For time series you must always split *chronologically*:
everything before a cutoff date is train, everything after is validation.
This file has exactly one function for that so every model uses the same
splitting logic and metrics are comparable across models.
"""
from __future__ import annotations

import pandas as pd


def time_based_split(df: pd.DataFrame, validation_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a (date, value) dataframe chronologically.

    Args:
        df: sorted dataframe with columns ["date", "value"]
        validation_size: number of most-recent rows to hold out for validation

    Returns:
        (train_df, val_df)
    """
    if validation_size >= len(df):
        raise ValueError("validation_size must be smaller than the dataset length.")
    train_df = df.iloc[: len(df) - validation_size].reset_index(drop=True)
    val_df = df.iloc[len(df) - validation_size :].reset_index(drop=True)
    return train_df, val_df


def make_supervised_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Turn a (date, value) series into a supervised-learning feature table
    for tree-based models (XGBoost).

    Features created:
      - calendar features: day of week, day of month, month, quarter, week of year
      - lag features: value at t-1, t-7, t-14, t-28 (configurable)
      - rolling features: rolling mean/std over 7 and 28 periods (shifted by 1
        to avoid leaking the current value)
    """
    lags = lags or [1, 2, 3, 7, 14, 28]
    out = df.copy()
    out["dayofweek"] = out["date"].dt.dayofweek
    out["day"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)

    for lag in lags:
        out[f"lag_{lag}"] = out["value"].shift(lag)

    out["rolling_mean_7"] = out["value"].shift(1).rolling(window=7, min_periods=1).mean()
    out["rolling_std_7"] = out["value"].shift(1).rolling(window=7, min_periods=1).std()
    out["rolling_mean_28"] = out["value"].shift(1).rolling(window=28, min_periods=1).mean()

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("date", "value")]
