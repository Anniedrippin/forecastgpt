"""
Qdrant service.

Why Qdrant / why RAG here at all
---------------------------------
Most of what this app does (forecasts, trend slopes, anomaly points) is
*computed*, not *retrieved* -- so it must never go through a vector search.
Qdrant is used only for the minority of things that are genuinely served
better by semantic retrieval:

  1. **Forecasting-method documentation** -- e.g. "when should I use SARIMA
     vs XGBoost?" is a conceptual question best answered by retrieving a
     relevant knowledge snippet and having the LLM explain it, not by
     running a forecast.
  2. **Past analysis summaries** -- every time the agent analyzes a dataset,
     a short natural-language summary of the findings is embedded and
     stored, so a later question like "what did we find out about dataset
     X last time?" can retrieve it instead of recomputing everything.
  3. **Dataset metadata** -- lets a "which of my datasets looks like it has
     strong seasonality?" style question retrieve semantically instead of
     requiring an exact filter.

Numeric questions ("what's the RMSE of the SARIMA model") are always
answered by calling the forecasting tools directly -- RAG is not used to
avoid recomputation of numbers, only to avoid recomputation/re-derivation
of *explanations*.
"""
from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.embedding_service import get_embedding_service

logger = get_logger(__name__)


class QdrantService:
    def __init__(self, host: str, port: int, collection: str):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        embedder = get_embedding_service()
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            logger.info(f"Creating Qdrant collection '{self.collection}' (dim={embedder.dimension})")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=embedder.dimension, distance=qmodels.Distance.COSINE
                ),
            )

    def is_connected(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def add_document(self, text: str, metadata: dict[str, Any]) -> str:
        embedder = get_embedding_service()
        vector = embedder.embed(text)
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": text, **metadata},
                )
            ],
        )
        return point_id

    def search(self, query: str, top_k: int = 3, doc_type: str | None = None) -> list[dict]:
        embedder = get_embedding_service()
        vector = embedder.embed(query)
        query_filter = None
        if doc_type:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value=doc_type))]
            )
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            {"text": h.payload.get("text", ""), "score": h.score, "metadata": h.payload}
            for h in hits
        ]


@lru_cache
def get_qdrant_service() -> QdrantService:
    settings = get_settings()
    return QdrantService(settings.qdrant_host, settings.qdrant_port, settings.qdrant_collection)
