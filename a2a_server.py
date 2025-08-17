"""
a2a_server.py
-------------
FastAPI application factory for the Agent-to-Agent (A2A) Prior Authorization demo.

This module mounts both agents under a single FastAPI app:
- Retrieval Agent    -> /agents/retrieval
- Summarization Agent -> /agents/summarization

Also provides simple operational endpoints:
- GET /health                 -> lightweight liveness check
- GET /                       -> human-friendly index
- GET /.well-known/agents     -> basic agent directory (helpful when exploring the API)
"""


from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from agent.retrieval_agent import get_router as retrieval_router
from agent.summarization_agent import get_router as summarization_router

settings = get_settings()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="A2A Prior Auth Demo",
        version="0.1.0",
        description=(
            "Demo of Agent-to-Agent (A2A) communication: a Retrieval Agent and a "
            "Summarization Agent collaborate to produce prior authorization guidance."
        ),
    )

    # CORS (wide-open for demo; restrict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount agents (single-process topology)
    app.include_router(retrieval_router("/agents/retrieval"))
    app.include_router(summarization_router("/agents/summarization"))

    @app.get("/", tags=["meta"])
    def index() -> dict:
        """Human-friendly index for quick navigation."""
        return {
            "name": "A2A Prior Auth Demo",
            "version": "0.1.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "agent_directory": "/.well-known/agents",
            "agents": {
                "retrieval": {
                    "card": "/agents/retrieval/.well-known/agent-card.json",
                    "task": "/agents/retrieval/task",
                },
                "summarization": {
                    "card": "/agents/summarization/.well-known/agent-card.json",
                    "task": "/agents/summarization/task",
                },
            },
        }

    @app.get("/.well-known/agents", tags=["meta"])
    def agent_directory() -> dict:
        """Minimal agent directory to help A2A discovery and manual testing."""
        return {
            "agents": [
                {
                    "id": "retrieval-agent",
                    "card_url": "/agents/retrieval/.well-known/agent-card.json",
                    "rpc_url": "/agents/retrieval/task",
                    "a2a_schema_version": 1,
                },
                {
                    "id": "summarization-agent",
                    "card_url": "/agents/summarization/.well-known/agent-card.json",
                    "rpc_url": "/agents/summarization/task",
                    "a2a_schema_version": 1,
                },
            ]
        }

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        """Liveness/readiness probe."""
        return {"ok": True}

    return app


# For `uvicorn a2a_server:app`
app = create_app()
