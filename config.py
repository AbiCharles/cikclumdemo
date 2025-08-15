from __future__ import annotations
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # --- App / server ---
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8080)

    # --- Together ---
    together_api_key: str = Field(default="", description="Set via TOGETHER_API_KEY env or .env")
    together_chat_model: str = Field(default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    together_embed_model: str = Field(default="togethercomputer/m2-bert-80M-32k-retrieval")
    temperature: float = Field(default=0.2)
    max_new_tokens: int = Field(default=800)

    # --- Vector DB (Qdrant) ---
    # Default to container hostname when running under docker-compose; fallback to localhost for CLI runs
    qdrant_url: str = Field(default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection_name: str = Field(default="prior_auth_demo")
    top_k: int = Field(default=5)

    # --- Demo behavior ---
    ingest_on_startup: bool = Field(default=True)
    trace_keep_raw_messages: bool = Field(default=True)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
