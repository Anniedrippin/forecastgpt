"""
LLM provider abstraction.

The rest of the agent code (graph.py) only ever imports `get_chat_model()`
from this file -- it never imports `ChatGroq` directly. That means switching
providers later (a local Ollama model, another hosted API) is a change
contained entirely to this one function, driven by the `LLM_PROVIDER`
environment variable.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def get_chat_model() -> BaseChatModel:
    settings = get_settings()

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and put it in your .env file."
            )
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.2,
        )

    # Extension point: add more providers here, e.g.
    # if settings.llm_provider == "ollama":
    #     from langchain_ollama import ChatOllama
    #     return ChatOllama(model=settings.groq_model)

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
        "Currently supported: 'groq'. Add a new branch in agents/llm.py to extend."
    )
