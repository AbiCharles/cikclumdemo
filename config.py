# config.py
"""
Configuration
-------------
Purpose
~~~~~~~
Centralized, typed configuration using pydantic-settings. This reads from:
- Environment variables
- A local `.env` file (if present)

Why it matters
~~~~~~~~~~~~~~
- Makes the app configurable without code changes
- Keeps environment (models, keys, ports, thresholds) in one consistent place
- Plays well with Docker Compose and local dev

Usage
~~~~~
from config import get_settings
settings = get_settings()
print(settings.together_chat_model)
"""

from __future__ import annotations

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    All tunables for the demo app. Most fields map directly to environment
    variables (see `alias=...`) so they can be overridden via `.env` or CI.
    """

    # --- App ---
    app_port: int = Field(
        default=8080,
        description="FastAPI/Gradio port the app listens on (host-mapped in docker-compose).",
    )
    trace_keep_raw_messages: bool = Field(
        default=True,
        description="Include human-friendly trace messages in responses for the demo UI.",
    )

    # --- Qdrant (vector DB) ---
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        description="Internal URL for Qdrant (Docker network).",
    )
    collection_name: str = Field(
        default="prior_auth_demo",
        description="Name of the Qdrant collection to use for the demo.",
    )
    ingest_on_startup: bool = Field(
        default=True,
        description="Ingest synthetic corpus at startup if the collection is empty.",
    )
    top_k: int = Field(
        default=3,
        description="Top K hits to return from vector search.",
        alias="TOP_K",
    )

    # Optional: score threshold for retrieval (cosine similarity returned by Qdrant).
    # When set (e.g., MIN_SCORE=0.65), retrieval will drop hits with score < min_score.
    # Leave as None (unset) to disable thresholding.
    min_score: Optional[float] = Field(
        default=None,
        description="Optional cosine-score threshold for Qdrant results. Set via MIN_SCORE env; None disables.",
        alias="MIN_SCORE",
    )

    # --- Together.ai (LLM and embeddings) ---
    together_api_key: str = Field(
        default="",
        description="Together.ai API key. Set via TOGETHER_API_KEY env or .env.",
        alias="TOGETHER_API_KEY",
    )
    together_chat_model: str = Field(
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        description="Chat/completions model name.",
        alias="TOGETHER_CHAT_MODEL",
    )
    together_embed_model: str = Field(
        default="togethercomputer/m2-bert-80M-32k-retrieval",
        description="Embedding model name.",
        alias="TOGETHER_EMBED_MODEL",
    )
    temperature: float = Field(
        default=0.2,
        description="Default generation temperature (passed to Together chat).",
        alias="TEMPERATURE",
    )
    max_new_tokens: int = Field(
        default=800,
        description="Max tokens for Together chat responses.",
        alias="MAX_NEW_TOKENS",
    )

    class Config:
        # Allow overrides from a local .env file (dev convenience)
        env_file = ".env"
        # Ignore unknown envs to avoid crashes when extra vars are present
        extra = "ignore"


# Cached singleton instance to avoid re-parsing env on every import
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    We construct it once on first call; subsequent callers get the same object.
    """
    global _settings
    if _settings is None:
        _settings = Settings()  # pydantic-settings picks up .env automatically
    return _settings
