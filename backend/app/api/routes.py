"""
FastAPI route definitions.

Routes are intentionally thin: they validate input (via Pydantic models in
models/schemas.py), call a service or forecasting function, and shape the
response. All real logic lives in services/ and forecasting/ so it can be
unit-tested without spinning up FastAPI at all (see tests/).
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.logging_config import get_logger
from app.forecasting import analysis, model_selector
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnomalyResult,
    ChatRequest,
    ChatResponse,
    ChatToolCall,
    DatasetInfo,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ProfileResult,
    SeasonalityResult,
    TrendResult,
    UploadResponse,
)
from app.services import chat_service
from app.services.dataset_service import get_dataset_service
from app.services.forecast_service import run_forecast
from app.services.qdrant_service import get_qdrant_service

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    from app.core.config import get_settings

    settings = get_settings()
    qdrant_ok = False
    try:
        qdrant_ok = get_qdrant_service().is_connected()
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
    return HealthResponse(status="ok", qdrant_connected=qdrant_ok, llm_provider=settings.llm_provider)


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported.")
    content = await file.read()
    service = get_dataset_service()
    try:
        loaded = service.ingest_csv(content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    meta = service.get_metadata(loaded.dataset_id)
    preview = loaded.df.head(10).copy()
    preview["date"] = preview["date"].astype(str)

    # Store a short natural-language summary of the upload in Qdrant so a
    # later semantic question like "what datasets have I uploaded about
    # sales?" can retrieve it.
    try:
        qdrant = get_qdrant_service()
        qdrant.add_document(
            f"Dataset '{file.filename}' uploaded with {meta['n_rows']} rows, "
            f"from {meta['start_date']} to {meta['end_date']}, "
            f"value column '{meta['value_column']}'.",
            {"doc_type": "dataset_metadata", "dataset_id": loaded.dataset_id},
        )
    except Exception as e:
        logger.warning(f"Could not index dataset metadata in Qdrant: {e}")

    return UploadResponse(
        dataset_id=loaded.dataset_id,
        filename=file.filename,
        n_rows=meta["n_rows"],
        date_column=meta["date_column"],
        value_column=meta["value_column"],
        start_date=meta["start_date"],
        end_date=meta["end_date"],
        inferred_frequency=meta["inferred_frequency"],
        preview=preview.to_dict(orient="records"),
    )


@router.get("/dataset/{dataset_id}", response_model=DatasetInfo)
def get_dataset(dataset_id: str):
    service = get_dataset_service()
    try:
        meta = service.get_metadata(dataset_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return DatasetInfo(**meta)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    service = get_dataset_service()
    try:
        loaded = service.load(req.dataset_id)
        meta = service.get_metadata(req.dataset_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e

    response = AnalyzeResponse(dataset_id=req.dataset_id)
    if "profile" in req.checks:
        response.profile = ProfileResult(**analysis.profile_dataset(loaded.df, meta.get("inferred_frequency")))
    if "trend" in req.checks:
        response.trend = TrendResult(**analysis.analyze_trend(loaded.df))
    if "seasonality" in req.checks:
        response.seasonality = SeasonalityResult(
            **analysis.analyze_seasonality(loaded.df, meta.get("inferred_frequency"))
        )
    if "anomalies" in req.checks:
        response.anomalies = AnomalyResult(**analysis.detect_anomalies(loaded.df))
    return response


@router.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    service = get_dataset_service()
    try:
        loaded = service.load(req.dataset_id)
        meta = service.get_metadata(req.dataset_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e

    try:
        result = run_forecast(loaded.df, req.horizon, req.model, meta.get("inferred_frequency"))
    except Exception as e:
        raise HTTPException(400, f"Forecast failed: {e}") from e

    comparison = None
    if req.compare_all:
        comparison = model_selector.compare_models(loaded.df, req.horizon, meta.get("inferred_frequency"))

    return ForecastResponse(
        dataset_id=req.dataset_id,
        model_used=result["model_used"],
        horizon=result["horizon"],
        history=result["history"],
        forecast=result["forecast"],
        validation_metrics=result["validation_metrics"],
        all_model_comparison=comparison,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = chat_service.chat(req.message, req.dataset_id, req.conversation_id)
    return ChatResponse(
        conversation_id=result["conversation_id"],
        reply=result["reply"],
        tool_calls_made=[ChatToolCall(**c) for c in result["tool_calls_made"]],
    )
