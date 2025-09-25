

from __future__ import annotations

"""
Retrieval Agent
---------------
Purpose:
    Provides vector-based retrieval of prior authorization policy snippets
    for a given patient and drug. Uses Together.ai embeddings with Qdrant
    as the vector database.

Responsibilities:
    - Uses Together embeddings (m2-bert-80M-32k-retrieval by default)
    - Ingest synthetic prior auth corpus into Qdrant (if empty).
    - Encode queries into embeddings and perform similarity search.
    - Filter results by section if requested (eligibility, step_therapy, etc.).
    - Expose an A2A-compatible FastAPI router with /task and agent-card endpoints
    - Optional min-score post-filter (respects `settings.min_score` if defined).

Why it matters:
    Decouples evidence retrieval from summarization logic, enabling clean
    Agent-to-Agent orchestration where this agent specializes in retrieval only.
"""

"""

Endpoints:
  GET  /.well-known/agent-card.json
  POST /task
"""



import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)
from together import Together

from config import get_settings
from data.prior_auth_docs import build_synthetic_corpus, patient_plan_map

settings = get_settings()


def _utc_ts() -> float:
    """Return current UNIX timestamp."""
    return time.time()


class RetrievalAgent:
    """
    Vector search over Qdrant using Together embeddings.

    Lifecycle:
      - Ensure collection exists, sized to embedding dimension
      - Ingest synthetic corpus if empty (configurable)
      - On /task, embed a query tailored to (patient_id, drug_name, focus_sections)
        and return top-K hits with payloads.
    """

    def __init__(self) -> None:
        if not settings.together_api_key:
            raise RuntimeError("TOGETHER_API_KEY not set.")

        # LLM embeddings client
        self.client = Together(api_key=settings.together_api_key)

        # Vector DB client (qdrant)
        self.qdrant = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.collection_name

        # Create collection (and ingest corpus if empty)
        self._ensure_collection_and_ingest()

    # ----------------------------
    # Embeddings
    # ----------------------------
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embedding vectors via Together embeddings API with basic retries.
        """
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self.client.embeddings.create(
                    model=settings.together_embed_model,
                    input=texts,
                )
                return [d.embedding for d in resp.data]
            except Exception as e:
                last_err = e
                time.sleep(2**attempt)
        raise last_err or RuntimeError("Embedding failed")

    def _auto_dimension(self) -> int:
        """
        Probe the embedding dimensionality once (e.g., 768) by embedding a short token.
        """
        return len(self.embed(["dimension_probe"])[0])

    # ----------------------------
    # Qdrant collection & ingestion
    # ----------------------------
    def _ensure_collection_and_ingest(self) -> None:
        """
        Ensure the target collection exists with correct dimensionality & distance.
        If collection is empty and ingest_on_startup is True, ingest the synthetic corpus.
        """
        try:
            _ = self.qdrant.get_collection(self.collection)
        except Exception:
            # If collection missing, recreate with detected dimension
            dim = self._auto_dimension()
            self.qdrant.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        # If collection empty, ingest synthetic corpus once (optional)
        _, count = self.qdrant.scroll(self.collection, limit=1)
        if count == 0 and settings.ingest_on_startup:
            self._ingest_corpus()

    def _ingest_corpus(self) -> None:
        """
        Embed all synthetic snippets and upsert them to Qdrant with payload fields.
        Payload fields include: doc_id, drug, plan, section, content.
        """
        corpus = build_synthetic_corpus()
        vectors = self.embed([d["content"] for d in corpus])
        points = [
            PointStruct(id=i + 1, vector=vectors[i], payload=corpus[i])
            for i in range(len(corpus))
        ]
        self.qdrant.upsert(collection_name=self.collection, points=points)

    # ----------------------------
    # Retrieval (the core task)
    # ----------------------------
    def retrieve(
        self,
        patient_id: str,
        drug_name: str,
        focus_sections: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform a similarity search.

        - Filters by plan and drug to reduce off-topic hits.
        - Optionally filters by section (eligibility, step_therapy, documentation, forms)
          to support focused re-query after reflection.

        Returns:
            Dict with plan, results (list of payloads + score)
        """
        limit = limit or settings.top_k
        plan = patient_plan_map().get(patient_id, "Unknown")

        # Simple query “prompt” text for embedding
        query = (
            f"prior authorization policy for {drug_name} under plan {plan}; "
            f"eligibility, step therapy, documentation, forms"
        )
        qvec = self.embed([query])[0]

        # Build filter: always constrain by plan + drug; optionally by section
        must = [
            FieldCondition(key="plan", match=MatchValue(value=plan)),
            FieldCondition(key="drug", match=MatchValue(value=drug_name)),
        ]
        if focus_sections:
            must.append(FieldCondition(key="section", match=MatchAny(any=focus_sections)))

        q_filter = Filter(must=must)

        # Use the new API (query_points) instead of deprecated search()
        hits = self.qdrant.query_points(
            collection_name=self.collection,
            query=qvec,                # vector
            limit=limit,
            query_filter=q_filter,
            with_payload=True,
            with_vectors=False,
        ).points

        # Optional score threshold (if present in settings)
        min_score = getattr(settings, "min_score", None)
        if isinstance(min_score, (int, float)):
            hits = [h for h in hits if h.score is None or h.score >= float(min_score)]

        items = [
            {
                "doc_id": h.payload["doc_id"],
                "drug": h.payload["drug"],
                "plan": h.payload["plan"],
                "section": h.payload["section"],
                "content": h.payload["content"],
                "score": h.score,
            }
            for h in hits
        ]

        return {"plan": plan, "results": items}


# =============================================================================
# FastAPI Router (A2A-style)
# =============================================================================
# Expose A2A discovery + RPC endpoints so other agents can discover
# and invoke retrieval over HTTP (e.g., the summarization agent).
def get_router(base_path: str = "/agents/retrieval") -> APIRouter:
    """
    Build the FastAPI router for the RetrievalAgent.
    Exposes:
      - GET /.well-known/agent-card.json
      - POST /task
    """
    router = APIRouter(prefix=base_path, tags=["retrieval-agent"])
    agent = RetrievalAgent()

    @router.get("/.well-known/agent-card.json")
    def agent_card() -> Dict[str, Any]:
        """Advertise the retrieval agent for A2A discovery."""
        return {
            "id": "retrieval-agent",
            "name": "PriorAuth Retrieval Agent",
            "description": "Retrieves plan-specific prior authorization snippets using vector search.",
            "a2a_schema_version": 1,  # <— version your A2A contract
            "rpc_url": f"{base_path}/task",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "drug_name": {"type": "string"},
                    "focus_sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: eligibility | step_therapy | documentation | forms",
                    },
                },
                "required": ["patient_id", "drug_name"],
            },
            "output_schema": {"type": "object", "properties": {"results": {"type": "array"}}},
        }

    # A2A task endpoint consumed by peer agents via HTTP POST.
    @router.post("/task")
    def handle_task(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        A2A Task entrypoint for retrieval.
        Validates goal args, runs search, and returns hits with a small trace.
        """
        ts0 = _utc_ts()
        goal = payload.get("goal") or {}
        patient_id = goal.get("patient_id")
        drug_name = goal.get("drug_name")
        focus_sections = goal.get("focus_sections")

        if not patient_id or not drug_name:
            raise HTTPException(400, "Missing patient_id or drug_name")

        retrieval = agent.retrieve(
            patient_id, drug_name, focus_sections=focus_sections
        )
        ts1 = _utc_ts()

        # Minimal human-readable messages for your Trace tab
        raw_msgs = [
            {"role": "system", "content": "RetrievalAgent received task.", "ts": ts0},
            {
                "role": "assistant",
                "content": f"Searching for {drug_name} under patient {patient_id}.",
                "ts": ts0,
            },
            {
                "role": "assistant",
                "content": f"Found {len(retrieval['results'])} results.",
                "ts": ts1,
            },
        ]

        return {
            "task_id": payload.get("task_id"),
            "agent": "retrieval-agent",
            "state": "completed",
            "duration_s": round(ts1 - ts0, 3),
            "output": {
                "plan": retrieval["plan"],
                "top_k": len(retrieval["results"]),
                "results": retrieval["results"],
            },
            "trace": {
                "agent": "retrieval-agent",
                "states": [{"state": "submitted", "ts": ts0}, {"state": "completed", "ts": ts1}],
                "messages": raw_msgs,
            },
        }

    return router
