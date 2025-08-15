# agent/retrieval_agent.py
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny
from together import Together
from config import get_settings
from data.prior_auth_docs import build_synthetic_corpus, patient_plan_map

settings = get_settings()

def _utc_ts() -> float:
    return time.time()

class RetrievalAgent:
    """
    Retrieval over Qdrant using Together embeddings.
    """
    def __init__(self):
        if not settings.together_api_key:
            raise RuntimeError("TOGETHER_API_KEY not set.")
        self.client = Together(api_key=settings.together_api_key)
        self.qdrant = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.collection_name
        self._ensure_collection_and_ingest()

    # ---------- Embeddings via Together ----------
    def embed(self, texts: List[str]) -> List[List[float]]:
        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.embeddings.create(
                    model=settings.together_embed_model,
                    input=texts
                )
                return [d.embedding for d in resp.data]
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise last_err or RuntimeError("Embedding failed")

    def _auto_dimension(self) -> int:
        return len(self.embed(["dimension_probe"])[0])

    # ---------- Qdrant collection / ingestion ----------
    def _ensure_collection_and_ingest(self) -> None:
        try:
            _ = self.qdrant.get_collection(self.collection)
        except Exception:
            dim = self._auto_dimension()
            self.qdrant.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        # ingest if empty
        _, count = self.qdrant.scroll(self.collection, limit=1)
        if count == 0 and settings.ingest_on_startup:
            self._ingest_corpus()

    def _ingest_corpus(self) -> None:
        corpus = build_synthetic_corpus()
        vectors = self.embed([d["content"] for d in corpus])
        points = [PointStruct(id=i + 1, vector=vectors[i], payload=corpus[i]) for i in range(len(corpus))]
        self.qdrant.upsert(collection_name=self.collection, points=points)

    # ---------- Retrieval ----------
    def retrieve(self, patient_id: str, drug_name: str, focus_sections: Optional[List[str]] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        limit = limit or settings.top_k
        plan = patient_plan_map().get(patient_id, "Unknown")

        query = (
            f"prior authorization policy for {drug_name} under plan {plan}; "
            f"eligibility, step therapy, documentation, forms"
        )
        qvec = self.embed([query])[0]

        q_filter = None
        if focus_sections:
            q_filter = Filter(
                must=[FieldCondition(key="section", match=MatchAny(any=focus_sections))]
            )

        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=qvec,
            limit=limit,
            query_filter=q_filter,
        )
        items = [{
            "doc_id": h.payload["doc_id"],
            "drug": h.payload["drug"],
            "plan": h.payload["plan"],
            "section": h.payload["section"],
            "content": h.payload["content"],
            "score": h.score,
        } for h in hits]

        return {"plan": plan, "results": items}

# --------- FastAPI / A2A-style endpoints ----------
def get_router(base_path: str = "/agents/retrieval") -> APIRouter:
    router = APIRouter(prefix=base_path, tags=["retrieval-agent"])
    agent = RetrievalAgent()

    @router.get("/.well-known/agent-card.json")
    def agent_card() -> Dict[str, Any]:
        return {
            "id": "retrieval-agent",
            "name": "PriorAuth Retrieval Agent",
            "description": "Retrieves plan-specific prior authorization snippets using vector search.",
            "rpc_url": f"{base_path}/task",
            "input_schema": {"type": "object", "properties": {"patient_id": {"type": "string"}, "drug_name": {"type": "string"}}},
            "output_schema": {"type": "object", "properties": {"results": {"type": "array"}}},
        }

    @router.post("/task")
    def handle_task(payload: Dict[str, Any]) -> Dict[str, Any]:
        ts0 = _utc_ts()
        goal = payload.get("goal") or {}
        patient_id = goal.get("patient_id")
        drug_name = goal.get("drug_name")
        focus_sections = goal.get("focus_sections")

        if not patient_id or not drug_name:
            raise HTTPException(400, "Missing patient_id or drug_name")

        retrieval = agent.retrieve(patient_id, drug_name, focus_sections=focus_sections)
        ts1 = _utc_ts()

        raw_msgs = [
            {"role": "system", "content": "RetrievalAgent received task.", "ts": ts0},
            {"role": "assistant", "content": f"Searching for {drug_name} under patient {patient_id}.", "ts": ts0},
            {"role": "assistant", "content": f"Found {len(retrieval['results'])} results.", "ts": ts1},
        ]

        return {
            "task_id": payload.get("task_id"),
            "agent": "retrieval-agent",
            "state": "completed",
            "duration_s": round(ts1 - ts0, 3),
            "output": {"plan": retrieval["plan"], "top_k": len(retrieval["results"]), "results": retrieval["results"]},
            "trace": {
                "agent": "retrieval-agent",
                "states": [{"state": "submitted", "ts": ts0}, {"state": "completed", "ts": ts1}],
                "messages": raw_msgs,
            },
        }

    return router
