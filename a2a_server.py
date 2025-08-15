# a2a_server.py

from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import get_settings
from agent.retrieval_agent import get_router as retrieval_router
from agent.summarization_agent import get_router as summarization_router

settings = get_settings()

def create_app() -> FastAPI:
    app = FastAPI(title="A2A Prior Auth Demo", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount agents under one app
    app.include_router(retrieval_router("/agents/retrieval"))
    app.include_router(summarization_router("/agents/summarization"))

    @app.get("/health")
    def health():
        return {"ok": True}

    return app

# For uvicorn a2a_server:app if you want
app = create_app()
