"""
Summarization Agent
-------------------
Purpose
~~~~~~~
This agent orchestrates a small Agent-to-Agent (A2A) workflow to produce an
actionable prior authorization summary given a (patient_id, drug_name).

Architecture
~~~~~~~~~~~~
- Delegates policy evidence retrieval to the Retrieval Agent via an HTTP RPC
  (`/agents/retrieval/task`).
- Optionally performs a "reflection" step:
    1) Ask the LLM to identify missing policy sections (eligibility, step_therapy,
       documentation, forms) based on the first retrieval results.
    2) If any sections are missing, re-call the Retrieval Agent with a section
       filter to fill gaps.
- Produces a final markdown summary + citations, returns a trace tree (for UI).

Why it matters
~~~~~~~~~~~~~~
- Demonstrates A2A delegation and optional iterative refinement ("reflection")
- Separates concerns: retrieval vs summarization/orchestration
- Returns machine-friendly outputs (for automation) and human-friendly traces


It also computes section coverage, missing-for-approval, and stage latencies
exposed in the response for UI tabs: Summary and Reflection Details.
(for debugging and demos)
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

# Read all configuration from a single settings object (pydantic-settings)
settings = get_settings()


def _utc_ts() -> float:
    """Return current UNIX timestamp (float). Useful for trace timings."""
    return time.time()


# System prompts kept short and explicit to reduce latency & parsing issues.
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
    Try to parse the model output as JSON; if it contains extra wrapper text,
    attempt to slice out the largest {...} block. Fall back to a minimal dict.

    Args:
        text: Raw model output

    Returns:
        dict with at least "summary" and "missing_sections"
    """
    try:
        return json.loads(text)
    except Exception:
        # Try to salvage JSON if model wrapped content with prose
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
    The orchestrator agent that:
      - Calls Retrieval Agent for evidence
      - Optionally reflects to find missing sections, then re-queries
      - Produces the final markdown summary + citations
    """

    def __init__(self) -> None:
        if not settings.together_api_key:
            raise RuntimeError("TOGETHER_API_KEY not set.")
        # Together.ai client (used for both reflection and final summary)
        self.client = Together(api_key=settings.together_api_key)
        # Base URL for local FastAPI app; used to call Retrieval Agent
        self.base_url = f"http://127.0.0.1:{settings.app_port}"

    # --------------------------------------------------------------------- #
    # LLM Invocation
    # --------------------------------------------------------------------- #
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a chat.completions request to Together.ai with a small retry loop.

        Args:
            messages: role/content messages (system|user|assistant)

        Returns:
            Model text output (string)
        """
        last_err = None
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
        # If all retries fail, surface last error
        raise last_err or RuntimeError("Chat failed")

    # --------------------------------------------------------------------- #
    # A2A RPC: Retrieval Agent
    # --------------------------------------------------------------------- #
    def call_retrieval(
        self, patient_id: str, drug_name: str, focus_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Invoke the Retrieval Agent with an A2A task payload.

        Args:
            patient_id: Patient ID key for plan lookup
            drug_name: Target drug name
            focus_sections: Optional list of sections to narrow the re-query

        Returns:
            Retrieval agent response (JSON dict)
        """
        body = {
            "task_id": f"ret-{int(time.time() * 1000)}",
            "goal": {"patient_id": patient_id, "drug_name": drug_name},
        }
        if focus_sections:
            body["goal"]["focus_sections"] = focus_sections

        with httpx.Client(timeout=60) as client:
            r = client.post(f"{self.base_url}/agents/retrieval/task", json=body)
            r.raise_for_status()
            return r.json()

    # --------------------------------------------------------------------- #
    # Orchestrated Flow: optional reflection, final summary
    # --------------------------------------------------------------------- #
    def summarize_with_reflection(
        self, patient_id: str, drug_name: str, fast: bool = False
    ) -> Dict[str, Any]:
        """
        Core flow:
          1) Call retrieval agent for initial evidence
          2) If not "fast":
                - Ask the LLM to identify which policy sections are missing
                - Re-call retrieval with a section filter to fill gaps
          3) Generate final markdown summary + citations

        Returns:
            Full agent response dict including outputs, trace, and raw_intermediate
        """
        ts0 = _utc_ts()
        plan = patient_plan_map().get(patient_id, "Unknown")

        # -- Step 1: initial retrieval
        ret1 = self.call_retrieval(patient_id, drug_name)
        results = list(ret1["output"]["results"])  # copy to allow append
        child_traces = [ret1["trace"]]

        missing: List[str] = []
        ret2 = None
        parsed: Dict[str, Any] = {"summary": "", "missing_sections": []}

        if not fast:
            # -- Step 2a: ask LLM which sections are still missing
            # We show evidence with section labels and scores to improve grounding
            context_blob = "\n".join(
                [
                    f"- [{r['section']}] {r['content']} (doc:{r['doc_id']} score:{r['score']:.3f})"
                    for r in results
                ]
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
            raw_json_text = self.chat(messages)
            parsed = extract_json_block(raw_json_text)

            # Normalize & validate missing sections
            if isinstance(parsed, dict):
                missing = [
                    m.strip().lower()
                    for m in parsed.get("missing_sections", [])
                    if isinstance(m, str)
                ]

            # -- Step 2b: if anything is missing, do a focused re-query
            if missing:
                allowed = {"eligibility", "step_therapy", "documentation", "forms"}
                focus = [m for m in missing if m in allowed]
                if focus:
                    ret2 = self.call_retrieval(patient_id, drug_name, focus_sections=focus)
                    child_traces.append(ret2["trace"])
                    results += ret2["output"]["results"]
        else:
            # Fast mode: skip reflection to reduce latency.
            parsed = {"summary": "", "missing_sections": []}

        # -- Step 3: final summary over all gathered evidence
        final_context = "\n".join(
            [f"- [{r['section']}] {r['content']} (doc:{r['doc_id']})" for r in results]
        )
        final_messages = [
            {"role": "system", "content": FINAL_SUMMARY_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Drug: {drug_name}\nPlan: {plan}\nEvidence:\n{final_context}\n\n"
                    "Write the final summary now."
                ),
            },
        ]
        final_text = self.chat(final_messages)

        ts1 = _utc_ts()

        # Small human-friendly trace (optional; controlled by settings)
        raw_msgs: List[Dict[str, Any]] = []
        if settings.trace_keep_raw_messages:
            note = (
                "Fast mode ON: skipped reflection."
                if fast
                else f"Reflection identified missing sections: {missing}"
            )
            raw_msgs = [
                {"role": "system", "content": "SummarizationAgent received task.", "ts": ts0},
                {
                    "role": "assistant",
                    "content": f"Delegated retrieval for {drug_name} on plan {plan}.",
                    "ts": ts0,
                },
                {"role": "assistant", "content": note, "ts": ts1},
            ]

        # Compose final response payload
        return {
            "agent": "summarization-agent",
            "state": "completed",
            "duration_s": round(ts1 - ts0, 3),
            "output": {
                "plan": plan,
                "drug_name": drug_name,
                "summary_markdown": final_text,
                "citations": [
                    {"doc_id": r["doc_id"], "section": r["section"]} for r in results
                ],
            },
            "trace": {
                "agent": "summarization-agent",
                "states": [{"state": "submitted", "ts": ts0}, {"state": "completed", "ts": ts1}],
                "messages": raw_msgs,
                "children": child_traces,
            },
            "raw_intermediate": {
                "first_pass_llm_output": parsed,     # model JSON about missing sections
                "focused_sections": [] if fast else missing,
                "fast_mode": fast,
            },
        }


# ------------------------------------------------------------------------- #
# FastAPI Router (A2A-facing)
# ------------------------------------------------------------------------- #
def get_router(base_path: str = "/agents/summarization") -> APIRouter:
    """
    Build the FastAPI router that exposes the Summarization Agent as:
      - GET  /.well-known/agent-card.json (A2A discovery)
      - POST /task                          (A2A RPC)
    """
    router = APIRouter(prefix=base_path, tags=["summarization-agent"])
    agent = SummarizationAgent()

    @router.get("/.well-known/agent-card.json")
    def agent_card() -> Dict[str, Any]:
        """
        Lightweight "agent card" for discovery by other agents.
        Includes a version tag (a2a_schema_version) so callers can adapt
        to changes over time.
        """
        return {
            "id": "summarization-agent",
            "name": "PriorAuth Summarization Agent",
            "description": (
                "Orchestrates retrieval and produces a policy summary with optional fast-mode (skips reflection)."
            ),
            "a2a_schema_version": 1,  # <— mirrors retrieval agent
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
            "output_schema": {
                "type": "object",
                "properties": {"summary_markdown": {"type": "string"}},
            },
        }

    @router.post("/task")
    def handle_task(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for A2A tasks. Validates input, runs orchestration, and
        returns final outputs and trace.

        Request shape:
            {
              "task_id": "...",
              "goal": { "patient_id": "...", "drug_name": "...", "fast": bool }
            }
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
            # Keep failures machine-readable for orchestrators
            return {
                "task_id": payload.get("task_id"),
                "agent": "summarization-agent",
                "state": "failed",
                "error": str(e),
            }

    return router
