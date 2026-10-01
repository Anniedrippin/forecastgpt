"""
XGBoost forecaster.

Gradient-boosted trees don't understand "time" natively -- they only see
rows of features. So we convert the series into a supervised-learning
problem (see preprocessing.make_supervised_features: lag values, rolling
statistics, calendar features) and then forecast recursively: predict step
t+1, append it to the history, recompute lag/rolling features, predict
t+2, and so on. This is the standard way to do multi-step forecasting with
tree-based models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.forecasting.preprocessing import feature_columns, make_supervised_features


def xgboost_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    feat_df = make_supervised_features(train)
    feat_df = feat_df.dropna().reset_index(drop=True)

    if len(feat_df) < 10:
        # Not enough history to build lag features reliably; fall back to
        # the mean of available data repeated across the horizon.
        return np.full(horizon, train["value"].mean(), dtype=float)

    cols = feature_columns(feat_df)
    X_train = feat_df[cols]
    y_train = feat_df["value"]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    # Recursive multi-step forecasting
    history = train.copy().reset_index(drop=True)
    freq = pd.infer_freq(history["date"]) or "D"
    preds = []
    for _ in range(horizon):
        next_date = history["date"].iloc[-1] + pd.tseries.frequencies.to_offset(freq)
        extended = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "value": [np.nan]})],
            ignore_index=True,
        )
        feat_row = make_supervised_features(extended).iloc[[-1]]
        x_next = feat_row[cols].fillna(0)
        pred = float(model.predict(x_next)[0])
        preds.append(pred)
        history = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "value": [pred]})],
            ignore_index=True,
        )

    return np.array(preds)
