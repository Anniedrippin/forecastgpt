"""
Centralized application configuration.

Why this exists
----------------
Every other module (LLM client, Qdrant client, embedding model, API routes)
reads its configuration from a single `Settings` object instead of calling
`os.getenv(...)` scattered everywhere. This means:

  1. There is exactly one place to look when something is misconfigured.
  2. Swapping the LLM provider, embedding model, or Qdrant host is a one-line
     change in `.env` -- no code changes required.
  3. Tests can override settings cleanly by instantiating `Settings(...)`
     with different kwargs instead of monkeypatching environment variables
     all over the place.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider -- kept generic on purpose. Today it's Groq (free tier).
    # If you later want to swap to a local Ollama model or another provider,
    # you only need to add a branch in agents/llm.py; nothing else changes.
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Local, free embedding model (Sentence-Transformers). Never OpenAI.
    embedding_model: str = "all-MiniLM-L6-v2"

    # Qdrant (self-hosted, free, runs in Docker)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "forecastgpt_knowledge"

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "INFO"
    data_dir: str = "./data"

    # Frontend
    backend_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so we parse `.env` once per process."""
    return Settings()
