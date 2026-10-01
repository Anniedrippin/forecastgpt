"""
Pydantic models (schemas) for API request/response validation.

Keeping these separate from business logic (services/) and from the agent
(agents/) means the API's public "contract" is defined in exactly one file.
FastAPI uses these to auto-generate OpenAPI docs at /docs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    n_rows: int
    date_column: str
    value_column: str
    start_date: str
    end_date: str
    inferred_frequency: Optional[str] = None
    preview: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Dataset info
# ---------------------------------------------------------------------------
class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    n_rows: int
    date_column: str
    value_column: str
    start_date: str
    end_date: str
    uploaded_at: str


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    dataset_id: str
    checks: list[Literal["profile", "trend", "seasonality", "anomalies"]] = Field(
        default_factory=lambda: ["profile", "trend", "seasonality", "anomalies"]
    )


class TrendResult(BaseModel):
    direction: str
    slope_per_period: float
    r_squared: float
    summary: str


class SeasonalityResult(BaseModel):
    seasonal_period_detected: Optional[int]
    seasonality_strength: float
    is_seasonal: bool
    summary: str


class AnomalyPoint(BaseModel):
    index: int
    date: str
    value: float
    reason: str
    score: float


class AnomalyResult(BaseModel):
    method: str
    n_anomalies: int
    anomalies: list[AnomalyPoint]
    summary: str


class ProfileResult(BaseModel):
    n_rows: int
    n_missing: int
    mean: float
    std: float
    min: float
    max: float
    start_date: str
    end_date: str
    inferred_frequency: Optional[str]


class AnalyzeResponse(BaseModel):
    dataset_id: str
    profile: Optional[ProfileResult] = None
    trend: Optional[TrendResult] = None
    seasonality: Optional[SeasonalityResult] = None
    anomalies: Optional[AnomalyResult] = None


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
ModelName = Literal["naive", "moving_average", "sarima", "xgboost"]


class ForecastRequest(BaseModel):
    dataset_id: str
    model: ModelName = "sarima"
    horizon: int = Field(14, gt=0, le=365)
    compare_all: bool = False


class ForecastPoint(BaseModel):
    date: str
    value: float
    lower: Optional[float] = None
    upper: Optional[float] = None


class ModelMetrics(BaseModel):
    model: str
    mae: float
    rmse: float
    mape: Optional[float] = None


class ForecastResponse(BaseModel):
    dataset_id: str
    model_used: str
    horizon: int
    history: list[ForecastPoint]
    forecast: list[ForecastPoint]
    validation_metrics: Optional[ModelMetrics] = None
    all_model_comparison: Optional[list[ModelMetrics]] = None


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    dataset_id: Optional[str] = None
    message: str
    conversation_id: Optional[str] = None


class ChatToolCall(BaseModel):
    tool: str
    args: dict[str, Any]


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    tool_calls_made: list[ChatToolCall] = Field(default_factory=list)
    chart_data: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    llm_provider: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
