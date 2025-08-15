"""
summarization_agent.py
----------------------
SummarizationAgent orchestrates:
1) First-pass retrieval from the RetrievalAgent
2) (Optional) Reflection LLM step to detect missing sections
3) Focused re-query for missing sections
4) Final policy summary

It also computes section coverage, missing-for-approval, and stage latencies
exposed in the response for UI tabs: Summary and Reflection Details.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from together import Together

from config import get_settings
from data.prior_auth_docs import patient_plan_map

settings = get_settings()


def _utc_ts() -> float:
    """Return current UNIX timestamp (float seconds)."""
    return time.time()


# LLM prompts kept short & single-responsibility
JSON_FALLBACK_PROMPT = """
You are a careful assistant. Reply ONLY in JSON with keys: summary (string), missing_sections (array of strings from: eligibility, step_therapy, documentation, forms).
"""

FINAL_SUMMARY_PROMPT = """
You are a clinical policy summarizer. Produce a concise, actionable summary for prior authorization for the given drug and plan.
Use bullet points. Include a short "Checklist" of items to include in the PA request.
End with a "Citations" section listing doc_id and section for the sources you used.
"""


def extract_json_block(text: str) -> Dict[str, Any]:
    """
    Robustly parse the most JSON-looking portion of a model response.
    Falls back to a minimal structure if nothing parses.
    """
    try:
        return json.loads(text)
    except Exception:
        # Try extracting the largest {...} block
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {"summary": text.strip(), "missing_sections": []}


class SummarizationAgent:
    """
    Coordinates the summarization workflow with optional reflection and re-query.

    High-level:
      - call_retrieval() -> first pass results
      - (if not fast) ask LLM which sections are missing
      - (if missing) call_retrieval(focus_sections=...)
      - final LLM summarize() over merged context
      - compute section_coverage & missing_for_approval for UI
    """

    def __init__(self) -> None:
        if not settings.together_api_key:
            raise RuntimeError("TOGETHER_API_KEY not set.")
        # Together client for chat completions
        self.client = Together(api_key=settings.together_api_key)
        # URL of our own FastAPI app where the RetrievalAgent is mounted
        self.base_url = f"http://127.0.0.1:{settings.app_port}"

    # ----------------------------
    # LLM client wrappers
    # ----------------------------
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send chat messages to Together chat model with basic retries.

        Returns:
            content string of the first choice
        """
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=settings.together_chat_model,
                    messages=messages,
                    temperature=settings.temperature,
                    max_tokens=settings.max_new_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                time.sleep(2**attempt)  # backoff: 1s, 2s, 4s
        raise last_err or RuntimeError("Chat failed")

    # ----------------------------
    # A2A call to RetrievalAgent
    # ----------------------------
    def call_retrieval(
        self,
        patient_id: str,
        drug_name: str,
        focus_sections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        POST to /agents/retrieval/task (local FastAPI).
        Adds a measured wall clock duration for UI latencies.
        """
        body: Dict[str, Any] = {
            "task_id": f"ret-{int(time.time() * 1000)}",
            "goal": {"patient_id": patient_id, "drug_name": drug_name},
        }
        if focus_sections:
            body["goal"]["focus_sections"] = focus_sections

        t0 = time.time()
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{self.base_url}/agents/retrieval/task", json=body)
            r.raise_for_status()
            out = r.json()
        out["_wall_duration_s"] = round(time.time() - t0, 3)
        return out

    # ----------------------------
    # Main workflow
    # ----------------------------
    def summarize_with_reflection(
        self, patient_id: str, drug_name: str, fast: bool = False
    ) -> Dict[str, Any]:
        """
        Execute end-to-end summarization with optional reflection (fast=False).
        Returns a response compatible with your UI expectations: output, trace, raw_intermediate, meta.
        """
        ts0 = _utc_ts()
        plan = patient_plan_map().get(patient_id, "Unknown")
        expected_sections = ["eligibility", "step_therapy", "documentation", "forms"]

        # 1) First-pass retrieval
        ret1_t0 = time.time()
        ret1 = self.call_retrieval(patient_id, drug_name)
        ret1_wall = ret1.get("_wall_duration_s")  # measured by us
        first_results = list(ret1.get("output", {}).get("results", []))
        child_traces = [ret1.get("trace")]
        first_pass_count = len(first_results)
        first_sections = {
            str(r.get("section", "")).strip().lower() for r in first_results
        }
        ret1_ll = round(time.time() - ret1_t0, 3)

        # Defaults for reflection paths
        missing: List[str] = []
        parsed: Dict[str, Any] = {"summary": "", "missing_sections": []}
        ret2: Optional[Dict[str, Any]] = None
        requery_count: Optional[int] = None
        requery_new_items: List[Dict[str, Any]] = []
        reflect_llm_s: Optional[float] = None
        ret2_wall: Optional[float] = None

        if not fast:
            # 2) Reflection: ask LLM which sections are missing (force JSON)
            context_blob = "\n".join(
                f"- [{r.get('section')}] {r.get('content')} (doc:{r.get('doc_id')} score:{float(r.get('score', 0.0)):.3f})"
                for r in first_results
            )
            messages = [
                {"role": "system", "content": JSON_FALLBACK_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Drug: {drug_name}\nPlan: {plan}\nEvidence:\n{context_blob}\n\n"
                        "What is the PA policy summary? What sections are still missing?"
                    ),
                },
            ]
            ref_t0 = time.time()
            raw_json_text = self.chat(messages)
            reflect_llm_s = round(time.time() - ref_t0, 3)

            parsed = extract_json_block(raw_json_text)
            if isinstance(parsed, dict):
                missing = [
                    m.strip().lower()
                    for m in parsed.get("missing_sections", [])
                    if isinstance(m, str)
                ]

            # 3) Focused re-query only for whitelisted sections
            section_whitelist = {"eligibility", "step_therapy", "documentation", "forms"}
            focus = [m for m in missing if m in section_whitelist]
            if focus:
                ret2 = self.call_retrieval(patient_id, drug_name, focus_sections=focus)
                ret2_wall = ret2.get("_wall_duration_s")
                child_traces.append(ret2.get("trace"))
                focused_results = list(ret2.get("output", {}).get("results", []))
                requery_count = len(focused_results)

                # capture “new” sections/snippets coming specifically from re-query
                for item in focused_results:
                    sec = str(item.get("section", "")).strip().lower()
                    if sec not in first_sections:
                        snippet = item.get("content", "")
                        if len(snippet) > 120:
                            snippet = snippet[:120].rstrip() + "…"
                        requery_new_items.append(
                            {
                                "doc_id": item.get("doc_id"),
                                "section": sec,
                                "snippet": snippet,
                                "score": float(item.get("score", 0.0)),
                            }
                        )

                # Merge re-query results into our working set
                first_results += focused_results
        # Fast mode: skip reflection & re-query

        # 4) Compute coverage and approval requirements after merge
        all_sections = {
            str(r.get("section", "")).strip().lower() for r in first_results
        }
        requery_sections = (
            set(s.get("section") for s in requery_new_items) if requery_new_items else set()
        )

        section_coverage: Dict[str, str] = {}
        for s in expected_sections:
            if s in first_sections:
                section_coverage[s] = "first_pass"
            elif s in requery_sections or s in all_sections:
                section_coverage[s] = "requery"  # satisfied by re-query or post-merge
            else:
                section_coverage[s] = "missing"

        missing_for_approval = [s for s, where in section_coverage.items() if where == "missing"]

        # 5) Final LLM summarization over merged context
        final_context = "\n".join(
            f"- [{r.get('section')}] {r.get('content')} (doc:{r.get('doc_id')})"
            for r in first_results
        )
        final_messages = [
            {"role": "system", "content": FINAL_SUMMARY_PROMPT},
            {
                "role": "user",
                "content": f"Drug: {drug_name}\nPlan: {plan}\nEvidence:\n{final_context}\n\nWrite the final summary now.",
            },
        ]
        final_text = self.chat(final_messages)

        ts1 = _utc_ts()

        # Optional human-readable notes for the Trace tab
        raw_msgs: List[Dict[str, Any]] = []
        if getattr(settings, "trace_keep_raw_messages", False):
            note = (
                "Fast mode ON: skipped reflection."
                if fast
                else f"Reflection identified missing sections: {missing}"
            )
            raw_msgs = [
                {
                    "role": "system",
                    "content": "SummarizationAgent received task.",
                    "ts": ts0,
                },
                {
                    "role": "assistant",
                    "content": f"Delegated retrieval for {drug_name} on plan {plan}.",
                    "ts": ts0,
                },
                {"role": "assistant", "content": note, "ts": ts1},
            ]

        # Prefer server-measured retrieval duration if available
        first_retrieval_s = ret1_wall if isinstance(ret1_wall, (int, float)) else ret1_ll

        # Final response structure consumed by UI
        return {
            "agent": "summarization-agent",
            "state": "completed",
            "duration_s": round(ts1 - ts0, 3),
            "output": {
                "plan": plan,
                "drug_name": drug_name,
                "summary_markdown": final_text,
                "citations": [
                    {"doc_id": r.get("doc_id"), "section": r.get("section")}
                    for r in first_results
                ],
                # For Summary tab: show what’s missing & coverage
                "missing_for_approval": missing_for_approval,
                "section_coverage": section_coverage,
            },
            "trace": {
                "agent": "summarization-agent",
                "states": [{"state": "submitted", "ts": ts0}, {"state": "completed", "ts": ts1}],
                "messages": raw_msgs,
                "children": [t for t in child_traces if t],
            },
            # For Reflection Details tab and inline panel
            "raw_intermediate": {
                "expected_sections": expected_sections,
                "first_pass_llm_output": parsed,
                "focused_sections": [] if fast else missing,
                "fast_mode": fast,
                "first_pass_hit_count": first_pass_count,
                "requery_hit_count": requery_count,
                "requery_new_items": requery_new_items,  # {doc_id, section, snippet, score}
                # Stage latencies
                "duration_first_retrieval_s": first_retrieval_s,
                "duration_reflection_llm_s": reflect_llm_s,
                "duration_requery_retrieval_s": ret2_wall,
            },
            "meta": {"duration_s": round(ts1 - ts0, 3)},
        }


# =============================================================================
# FastAPI Router (A2A-style)
# =============================================================================
def get_router(base_path: str = "/agents/summarization") -> APIRouter:
    """
    Build the FastAPI router for the SummarizationAgent.
    Exposes:
      - GET /.well-known/agent-card.json
      - POST /task
    """
    router = APIRouter(prefix=base_path, tags=["summarization-agent"])
    agent = SummarizationAgent()

    @router.get("/.well-known/agent-card.json")
    def agent_card() -> Dict[str, Any]:
        """Advertise agent capabilities to peers via A2A card."""
        return {
            "id": "summarization-agent",
            "name": "PriorAuth Summarization Agent",
            "description": "Orchestrates retrieval and produces a policy summary with optional fast-mode (skips reflection).",
            "rpc_url": f"{base_path}/task",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "drug_name": {"type": "string"},
                    "fast": {"type": "boolean"},
                },
                "required": ["patient_id", "drug_name"],
            },
            "output_schema": {"type": "object", "properties": {"summary_markdown": {"type": "string"}}},
        }

    @router.post("/task")
    def handle_task(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        A2A Task entrypoint. Validates goal args, invokes agent, returns task result.
        Never raises to callers; failures are returned as {"state": "failed", "error": "..."}.
        """
        goal = payload.get("goal") or {}
        patient_id = goal.get("patient_id")
        drug_name = goal.get("drug_name")
        fast = bool(goal.get("fast", False))
        if not patient_id or not drug_name:
            raise HTTPException(400, "Missing patient_id or drug_name")
        try:
            result = agent.summarize_with_reflection(patient_id, drug_name, fast=fast)
            return {
                "task_id": payload.get("task_id"),
                "agent": "summarization-agent",
                "state": result["state"],
                "output": result["output"],
                "trace": result["trace"],
                "meta": {"duration_s": result["duration_s"]},
                "raw_intermediate": result.get("raw_intermediate"),
            }
        except Exception as e:
            return {
                "task_id": payload.get("task_id"),
                "agent": "summarization-agent",
                "state": "failed",
                "error": str(e),
            }

    return router
