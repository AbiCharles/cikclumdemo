# data/debug_qdrant.py
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from together import Together

from config import get_settings
from data.prior_auth_docs import build_synthetic_corpus, patient_plan_map


def echo(msg: str):
    print(msg, flush=True)


def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)


def err(msg: str):
    print(f"[ERROR] {msg}", flush=True)


def embed_probe(together: Together, text: str, model: str) -> int:
    resp = together.embeddings.create(model=model, input=[text])
    vec = resp.data[0].embedding
    return len(vec)


def ensure_collection(qc: QdrantClient, collection: str, dim: int):
    try:
        qc.get_collection(collection)
        return
    except Exception:
        pass
    from qdrant_client.models import Distance, VectorParams
    qc.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    echo(f"Recreated collection: {collection} (dim={dim})")


def ingest_if_empty(qc: QdrantClient, collection: str, together: Together, embed_model: str) -> int:
    # OPTION A: len-based emptiness check
    pts, _ = qc.scroll(collection, limit=1, with_payload=False, with_vectors=False)
    if pts:
        return 0  # nothing ingested

    corpus = build_synthetic_corpus()
    texts = [d["content"] for d in corpus]
    vecs = together.embeddings.create(model=embed_model, input=texts).data

    from qdrant_client.models import PointStruct
    points = [
        PointStruct(id=i + 1, vector=vecs[i].embedding, payload=corpus[i])
        for i in range(len(corpus))
    ]
    qc.upsert(collection_name=collection, points=points)

    # Report how many we *attempted* to ingest
    return len(points)


def list_points(qc: QdrantClient, collection: str, limit: int):
    points, _ = qc.scroll(collection, limit=limit, with_payload=True, with_vectors=False)
    if not points:
        echo("(no points to display)")
        return
    for p in points:
        pl = p.payload or {}
        echo(f"- id={p.id} doc_id={pl.get('doc_id')} | drug={pl.get('drug')} | plan={pl.get('plan')} | section={pl.get('section')}")


def dry_run_search(
    qc: QdrantClient,
    together: Together,
    collection: str,
    patient_id: str,
    drug_name: str,
    embed_model: str,
    top_k: int = 5,
    focus_sections: Optional[List[str]] = None,
):
    plan = patient_plan_map().get(patient_id, "Unknown")
    whitelist = {"eligibility", "step_therapy", "documentation", "forms", "notes"}

    focus = None
    if focus_sections:
        focus = sorted({s.strip().lower() for s in focus_sections if isinstance(s, str)})
        focus = [s for s in focus if s in whitelist]
        if not focus:
            focus = None

    hint = ""
    if focus:
        hint = " Focus on: " + ", ".join(focus) + "."

    query_text = (
        f"prior authorization policy for {drug_name} under plan {plan}; "
        f"eligibility, step therapy, documentation, forms.{hint}"
    ).strip()

    vec = together.embeddings.create(model=embed_model, input=[query_text]).data[0].embedding

    q_filter = None
    if focus:
        q_filter = Filter(must=[FieldCondition(key="section", match=MatchAny(any=focus))])

    # Using deprecated search is fine for now; this is only a debug tool.
    hits = qc.search(collection_name=collection, query_vector=vec, limit=top_k, query_filter=q_filter)

    echo(f"\nSearch: drug={drug_name} plan={plan} focus={focus or '-'} top_k={top_k}")
    if not hits:
        echo("=> (no hits)")
        return

    for h in hits:
        pl = h.payload or {}
        echo(
            f"  score={h.score:.4f}  doc_id={pl.get('doc_id')}  "
            f"section={pl.get('section')}  plan={pl.get('plan')}  drug={pl.get('drug')}"
        )
        echo(f"    content: {pl.get('content')}")


def main():
    parser = argparse.ArgumentParser(description="Qdrant/Together sanity check")
    parser.add_argument("--limit", type=int, default=10, help="How many points to list from the collection")
    parser.add_argument("--patient", type=str, default="P001", help="Patient ID (for dry-run search)")
    parser.add_argument("--drug", type=str, default="Humira", help="Drug name (for dry-run search)")
    parser.add_argument("--focus", nargs="*", help="Optional focus sections (e.g., eligibility documentation)")
    parser.add_argument("--reingest", action="store_true", help="Force ingest if collection is empty")
    args = parser.parse_args()

    settings = get_settings()
    echo(f"Qdrant URL: {settings.qdrant_url}")
    echo(f"Collection: {settings.collection_name}")
    echo(f"Together chat model: {settings.together_chat_model}")
    echo(f"Together embed model: {settings.together_embed_model}")

    # Clients
    try:
        qc = QdrantClient(url=settings.qdrant_url)
        qc.get_collections()
    except Exception as e:
        err(f"Failed to reach Qdrant at {settings.qdrant_url}: {e}")
        sys.exit(1)

    try:
        tg = Together(api_key=settings.together_api_key)
    except Exception as e:
        err(f"Failed to init Together client: {e}")
        sys.exit(1)

    # Probe embedding dimension
    try:
        dim = embed_probe(tg, "dimension_probe", settings.together_embed_model)
        echo(f"Embedding dimension (probe): {dim}")
    except Exception as e:
        err(f"Embedding probe failed (Together): {e}")
        sys.exit(1)

    # Ensure collection exists
    try:
        ensure_collection(qc, settings.collection_name, dim)
    except Exception as e:
        err(f"Failed to ensure collection: {e}")
        sys.exit(1)

    # Ingest if empty (Option A len-check)
    try:
        pts, _ = qc.scroll(settings.collection_name, limit=1, with_payload=False, with_vectors=False)
        is_empty = len(pts) == 0
        if is_empty and args.reingest:
            n = ingest_if_empty(qc, settings.collection_name, tg, settings.together_embed_model)
            echo(f"Ingested {n} points into {settings.collection_name}.")
        else:
            echo(f"Collection empty? {is_empty}  (use --reingest to ingest if empty)")
    except Exception as e:
        err(f"Failed during ingest/scroll: {e}")
        sys.exit(1)

    # List a few points
    echo("\n--- Sample points ---")
    try:
        list_points(qc, settings.collection_name, limit=max(1, args.limit))
    except Exception as e:
        err(f"Failed to list points: {e}")

    # Dry-run search
    try:
        top_k = max(1, int(getattr(settings, "top_k", 5)))
        dry_run_search(
            qc,
            tg,
            settings.collection_name,
            args.patient,
            args.drug,
            settings.together_embed_model,
            top_k=top_k,
            focus_sections=args.focus,
        )
    except Exception as e:
        err(f"Dry-run search failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
