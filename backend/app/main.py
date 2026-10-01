"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
(from the backend/ directory), or via Docker Compose (see docker-compose.yml).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting ForecastGPT backend...")
    try:
        from app.services.knowledge_base import seed_knowledge_base
        from app.services.qdrant_service import get_qdrant_service

        seed_knowledge_base(get_qdrant_service())
    except Exception as e:
        logger.warning(
            f"Could not seed Qdrant knowledge base on startup ({e}). "
            "Is Qdrant running? (docker compose up qdrant)"
        )
    yield
    logger.info("Shutting down ForecastGPT backend.")


app = FastAPI(
    title="ForecastGPT API",
    description="Conversational AI assistant for time-series analysis and forecasting.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")


@app.get("/")
def root():
    return {"message": "ForecastGPT API is running. See /docs for the API reference."}
