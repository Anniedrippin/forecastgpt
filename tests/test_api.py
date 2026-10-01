"""
API-level tests using FastAPI's TestClient.

Note: /chat is intentionally NOT tested here because it requires a live
Groq API key and a running Qdrant instance -- it is exercised manually /
via the Streamlit UI instead. Everything else (upload, analyze, forecast,
dataset info, health) only depends on pandas/statsmodels/xgboost and runs
fully offline, so it's safe to test in CI without secrets.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture
def sample_csv_bytes():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    t = np.arange(100)
    values = 200 + 0.3 * t + 5 * np.sin(2 * np.pi * t / 7)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": values})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_and_analyze_and_forecast_flow(sample_csv_bytes):
    upload_resp = client.post(
        "/upload", files={"file": ("sample.csv", sample_csv_bytes, "text/csv")}
    )
    assert upload_resp.status_code == 200
    data = upload_resp.json()
    dataset_id = data["dataset_id"]
    assert data["n_rows"] == 100
    assert data["value_column"] == "sales"

    info_resp = client.get(f"/dataset/{dataset_id}")
    assert info_resp.status_code == 200
    assert info_resp.json()["dataset_id"] == dataset_id

    analyze_resp = client.post("/analyze", json={"dataset_id": dataset_id})
    assert analyze_resp.status_code == 200
    body = analyze_resp.json()
    assert body["trend"]["direction"] in {"upward", "downward", "flat / no clear linear trend"}
    assert body["profile"]["n_rows"] == 100

    forecast_resp = client.post(
        "/forecast", json={"dataset_id": dataset_id, "model": "sarima", "horizon": 7}
    )
    assert forecast_resp.status_code == 200
    fc_body = forecast_resp.json()
    assert len(fc_body["forecast"]) == 7
    assert fc_body["model_used"] == "sarima"


def test_forecast_unknown_dataset_returns_404():
    resp = client.post("/forecast", json={"dataset_id": "does-not-exist", "model": "naive", "horizon": 3})
    assert resp.status_code == 404


def test_upload_rejects_non_csv():
    resp = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
