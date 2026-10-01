"""
Unit tests for the pure forecasting/analysis functions. These do not need
FastAPI, Qdrant, or Groq running -- they test the deterministic math in
isolation, which is exactly the layer where correctness matters most
(the agent and API are just orchestration around this).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.forecasting import analysis, evaluation, model_selector
from app.forecasting.baselines import moving_average_forecast, naive_forecast
from app.forecasting.preprocessing import make_supervised_features, time_based_split
from app.forecasting.xgboost_forecast import xgboost_forecast


@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    t = np.arange(120)
    values = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 7) + np.random.RandomState(0).normal(0, 2, 120)
    return pd.DataFrame({"date": dates, "value": values})


def test_time_based_split(sample_df):
    train, val = time_based_split(sample_df, validation_size=14)
    assert len(val) == 14
    assert len(train) == len(sample_df) - 14
    assert train["date"].max() < val["date"].min()


def test_naive_forecast(sample_df):
    preds = naive_forecast(sample_df, horizon=5)
    assert len(preds) == 5
    assert np.all(preds == sample_df["value"].iloc[-1])


def test_moving_average_forecast(sample_df):
    preds = moving_average_forecast(sample_df, horizon=5, window=7)
    assert len(preds) == 5
    expected = sample_df["value"].iloc[-7:].mean()
    assert np.allclose(preds, expected)


def test_evaluation_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    result = evaluation.evaluate(y_true, y_pred)
    assert result["mae"] > 0
    assert result["rmse"] >= result["mae"]  # RMSE >= MAE always holds
    assert result["mape"] is not None


def test_evaluation_mape_undefined_with_zero():
    y_true = np.array([0.0, 20.0])
    y_pred = np.array([1.0, 18.0])
    assert evaluation.mape(y_true, y_pred) is None


def test_analyze_trend_detects_upward(sample_df):
    result = analysis.analyze_trend(sample_df)
    assert result["direction"] == "upward"
    assert result["slope_per_period"] > 0


def test_detect_anomalies_zscore_flags_injected_spike(sample_df):
    df = sample_df.copy()
    df.loc[50, "value"] = df["value"].max() * 5  # inject an obvious spike
    result = analysis.detect_anomalies(df, method="zscore")
    flagged_indices = {a["index"] for a in result["anomalies"]}
    assert 50 in flagged_indices


def test_make_supervised_features_has_no_lookahead(sample_df):
    feat = make_supervised_features(sample_df)
    # lag_1 at row i must equal value at row i-1
    valid = feat.dropna(subset=["lag_1"])
    assert np.allclose(valid["lag_1"].values, sample_df["value"].shift(1).dropna().values)


def test_xgboost_forecast_runs_and_returns_horizon(sample_df):
    preds = xgboost_forecast(sample_df, horizon=5)
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))


def test_compare_models_returns_sorted_by_rmse(sample_df):
    comparison = model_selector.compare_models(sample_df, horizon=7)
    rmses = [c["rmse"] for c in comparison]
    assert rmses == sorted(rmses)
    assert model_selector.best_model(comparison) == comparison[0]["model"]
