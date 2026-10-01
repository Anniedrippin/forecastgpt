"""
Agent tools.

Every tool here is a thin wrapper around a *deterministic* Python function
(pandas/numpy/statsmodels/scikit-learn/xgboost) -- never an LLM call. This is
the mechanism that satisfies "the LLM should call tools rather than
hallucinate numerical results": the LLM's only job is to decide *which*
tool to call and *how to phrase its explanation* of the tool's output; it
never invents a number itself.

Design note: tools are built by `build_tools(dataset_id)` instead of being
plain module-level functions. Binding the active dataset_id into a closure
means the LLM's tool-calling schema never has to include (and therefore can
never hallucinate) a dataset_id -- it only ever operates on "the currently
uploaded dataset", which the API layer sets for the conversation.
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.logging_config import get_logger
from app.forecasting import analysis, model_selector
from app.services.dataset_service import get_dataset_service
from app.services.forecast_service import run_forecast
from app.services.qdrant_service import get_qdrant_service
from app.services.visualization_service import (
    anomaly_chart,
    forecast_chart,
    model_comparison_chart,
    timeseries_chart,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------
class NoArgs(BaseModel):
    pass


class AnomalyArgs(BaseModel):
    method: str = Field("zscore", description="Either 'zscore' or 'isolation_forest'")


class ForecastArgs(BaseModel):
    model: str = Field(
        "sarima", description="One of: naive, moving_average, sarima, xgboost"
    )
    horizon: int = Field(14, description="Number of future periods to forecast")


class CompareArgs(BaseModel):
    horizon: int = Field(14, description="Validation horizon used to compare models")


class VisualizationArgs(BaseModel):
    chart_type: str = Field(
        "timeseries", description="One of: timeseries, anomaly, model_comparison"
    )


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(..., description="Natural-language question about forecasting concepts/methods")


class WhatIfArgs(BaseModel):
    percent_change: float = Field(
        ..., description="Hypothetical percentage change to the recent trend, e.g. 10 for +10%, -5 for -5%"
    )
    horizon: int = Field(14, description="Number of future periods to project under this scenario")


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------
def build_tools(dataset_id: Optional[str]) -> list[StructuredTool]:
    dataset_service = get_dataset_service()
    qdrant = get_qdrant_service()

    def _require_dataset():
        if not dataset_id:
            raise ValueError("No dataset is currently loaded. Ask the user to upload a CSV first.")
        return dataset_service.load(dataset_id)

    # -- profile_dataset ---------------------------------------------------
    def _profile_dataset() -> str:
        loaded = _require_dataset()
        meta = dataset_service.get_metadata(dataset_id)
        result = analysis.profile_dataset(loaded.df, meta.get("inferred_frequency"))
        return json.dumps(result)

    # -- analyze_trend -------------------------------------------------------
    def _analyze_trend() -> str:
        loaded = _require_dataset()
        return json.dumps(analysis.analyze_trend(loaded.df))

    # -- analyze_seasonality --------------------------------------------------
    def _analyze_seasonality() -> str:
        loaded = _require_dataset()
        meta = dataset_service.get_metadata(dataset_id)
        return json.dumps(analysis.analyze_seasonality(loaded.df, meta.get("inferred_frequency")))

    # -- detect_anomalies ------------------------------------------------------
    def _detect_anomalies(method: str = "zscore") -> str:
        loaded = _require_dataset()
        result = analysis.detect_anomalies(loaded.df, method=method)
        return json.dumps(result)

    # -- generate_forecast -------------------------------------------------------
    def _generate_forecast(model: str = "sarima", horizon: int = 14) -> str:
        loaded = _require_dataset()
        meta = dataset_service.get_metadata(dataset_id)
        result = run_forecast(loaded.df, horizon, model, meta.get("inferred_frequency"))
        # Trim history in the tool's return value (the LLM doesn't need every
        # historical row to write an explanation -- just the forecast + metrics)
        trimmed = {**result, "history": result["history"][-10:]}
        return json.dumps(trimmed)

    # -- compare_models ---------------------------------------------------------
    def _compare_models(horizon: int = 14) -> str:
        loaded = _require_dataset()
        meta = dataset_service.get_metadata(dataset_id)
        comparison = model_selector.compare_models(loaded.df, horizon, meta.get("inferred_frequency"))
        return json.dumps({"comparison": comparison, "best_model": model_selector.best_model(comparison)})

    # -- create_visualization -------------------------------------------------
    def _create_visualization(chart_type: str = "timeseries") -> str:
        loaded = _require_dataset()
        if chart_type == "anomaly":
            anomalies = analysis.detect_anomalies(loaded.df)["anomalies"]
            chart = anomaly_chart(loaded.df, anomalies)
        elif chart_type == "model_comparison":
            meta = dataset_service.get_metadata(dataset_id)
            comparison = model_selector.compare_models(loaded.df, 14, meta.get("inferred_frequency"))
            chart = model_comparison_chart(comparison)
        else:
            chart = timeseries_chart(loaded.df)
        # Tools return text to the LLM; the actual chart JSON is re-derived
        # by the API layer for the frontend so the LLM only has to describe
        # it, not carry the payload through its context window.
        return json.dumps({"chart_type": chart["type"], "title": chart["title"], "note": "chart rendered in UI"})

    # -- search_knowledge_base (RAG) --------------------------------------------
    def _search_knowledge_base(query: str) -> str:
        hits = qdrant.search(query, top_k=3, doc_type="method_doc")
        if not hits:
            return json.dumps({"results": [], "note": "No relevant knowledge base entries found."})
        return json.dumps({"results": [{"text": h["text"], "score": round(h["score"], 3)} for h in hits]})

    # -- what_if_analysis -----------------------------------------------------------
    def _what_if_analysis(percent_change: float, horizon: int = 14) -> str:
        """Simplified scenario analysis: shifts the moving-average baseline
        forecast by the given percentage to illustrate 'what if growth were
        X% different'. Explicitly not a causal model."""
        loaded = _require_dataset()
        meta = dataset_service.get_metadata(dataset_id)
        base = run_forecast(loaded.df, horizon, "moving_average", meta.get("inferred_frequency"))
        factor = 1 + (percent_change / 100.0)
        scenario_forecast = [
            {**pt, "value": pt["value"] * factor, "lower": pt["lower"] * factor, "upper": pt["upper"] * factor}
            for pt in base["forecast"]
        ]
        return json.dumps(
            {
                "scenario": f"{percent_change:+.1f}% adjustment to baseline trend",
                "baseline_forecast": base["forecast"],
                "scenario_forecast": scenario_forecast,
                "note": (
                    "This is a simplified sensitivity analysis (baseline forecast scaled by a "
                    "percentage), not a causal model."
                ),
            }
        )

    return [
        StructuredTool.from_function(
            func=_profile_dataset,
            name="profile_dataset",
            description="Get summary statistics of the currently loaded dataset: row count, missing values, mean/std/min/max, date range, inferred frequency.",
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            func=_analyze_trend,
            name="analyze_trend",
            description="Analyze the overall trend (direction, slope, R-squared) of the currently loaded time series.",
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            func=_analyze_seasonality,
            name="analyze_seasonality",
            description="Detect and quantify seasonality in the currently loaded time series using classical decomposition.",
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            func=_detect_anomalies,
            name="detect_anomalies",
            description="Detect anomalous points in the currently loaded time series using 'zscore' or 'isolation_forest'.",
            args_schema=AnomalyArgs,
        ),
        StructuredTool.from_function(
            func=_generate_forecast,
            name="generate_forecast",
            description="Generate a real numeric forecast for the currently loaded dataset using one of: naive, moving_average, sarima, xgboost. Returns forecast values, confidence bounds, and validation metrics.",
            args_schema=ForecastArgs,
        ),
        StructuredTool.from_function(
            func=_compare_models,
            name="compare_models",
            description="Fit all four forecasting models on a held-out validation window and compare them by MAE/RMSE/MAPE, returning the best model.",
            args_schema=CompareArgs,
        ),
        StructuredTool.from_function(
            func=_create_visualization,
            name="create_visualization",
            description="Prepare chart data for the currently loaded dataset. chart_type one of: timeseries, anomaly, model_comparison.",
            args_schema=VisualizationArgs,
        ),
        StructuredTool.from_function(
            func=_search_knowledge_base,
            name="search_knowledge_base",
            description="Semantic search over documentation about forecasting methods and evaluation metrics (RAG). Use this for conceptual 'what is/why/when should I use X' questions, not for numeric results.",
            args_schema=KnowledgeSearchArgs,
        ),
        StructuredTool.from_function(
            func=_what_if_analysis,
            name="what_if_analysis",
            description="Simplified what-if / scenario analysis: projects a forecast under a hypothetical percentage change to the trend.",
            args_schema=WhatIfArgs,
        ),
    ]
