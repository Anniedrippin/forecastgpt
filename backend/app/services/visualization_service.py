"""
Visualization service.

Rather than serializing full Plotly Figure objects across the API boundary
(which is fragile and heavier than necessary), this service returns plain
JSON-able chart specs: a chart `type` plus `series` of (x, y) data. The
Streamlit frontend turns these into actual Plotly figures with
`plotly.graph_objects`. This keeps the backend framework-agnostic -- any
frontend (React, another Streamlit app, a notebook) could consume the same
JSON.
"""
from __future__ import annotations

import pandas as pd


def timeseries_chart(df: pd.DataFrame, title: str = "Time Series") -> dict:
    return {
        "type": "line",
        "title": title,
        "series": [
            {
                "name": "value",
                "x": df["date"].astype(str).tolist(),
                "y": df["value"].tolist(),
            }
        ],
    }


def forecast_chart(history: list[dict], forecast: list[dict], title: str = "Forecast") -> dict:
    return {
        "type": "forecast",
        "title": title,
        "history": {
            "x": [h["date"] for h in history],
            "y": [h["value"] for h in history],
        },
        "forecast": {
            "x": [f["date"] for f in forecast],
            "y": [f["value"] for f in forecast],
            "lower": [f.get("lower") for f in forecast],
            "upper": [f.get("upper") for f in forecast],
        },
    }


def anomaly_chart(df: pd.DataFrame, anomalies: list[dict], title: str = "Anomalies") -> dict:
    anomaly_dates = {a["date"] for a in anomalies}
    return {
        "type": "anomaly",
        "title": title,
        "series": {
            "x": df["date"].astype(str).tolist(),
            "y": df["value"].tolist(),
        },
        "anomalies": {
            "x": [a["date"] for a in anomalies],
            "y": [a["value"] for a in anomalies],
        },
    }


def model_comparison_chart(comparison: list[dict], title: str = "Model Comparison (RMSE)") -> dict:
    return {
        "type": "bar",
        "title": title,
        "x": [c["model"] for c in comparison],
        "y": [c["rmse"] for c in comparison],
    }
