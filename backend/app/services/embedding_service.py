"""
Embedding service.

Uses a local Sentence-Transformers model (all-MiniLM-L6-v2 by default) to
turn text into vectors for Qdrant. This runs entirely on your CPU, downloads
once from HuggingFace (free, no API key), and produces 384-dimensional
vectors -- there is no OpenAI embeddings call anywhere in this project.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str):
        logger.info(f"Loading local embedding model '{model_name}' (first run downloads weights)...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(settings.embedding_model)
