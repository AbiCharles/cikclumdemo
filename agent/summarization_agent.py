from __future__ import annotations

"""
Summarization Agent
-------------------
Coordinates retrieval + (optional) reflection and produces an actionable PA summary.

This version adds:
- first_pass_hit_count / requery_hit_count
- per-step durations
- section_coverage + missing_for_approval
So the UI’s Reflection Details and “Needed for Prior Auth Approval” render correctly.
"""

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
    return time.time()


JSON_FALLBACK_PROMPT = """
You are a careful assistant. Reply ONLY in JSON with keys: summary (string), missing_sections (array of strings from: eligibility, step_therapy, documentation, forms).
"""

FINAL_SUMMARY_PROMPT = """
You are a clinical policy summarizer. Produce a concise, actionable summary for prior authorization for the given drug and plan.
Use bullet points. Include a short "Checklist" of items to include in the PA request.
End with a "Citations" section listing doc_id and section for the sources you used.
"""


def extract_json_block(text: str) -> Dict[str, Any]:
    """Tolerant JSON extractor (handles accidental prose around a JSON object)."""
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {"summary": text.strip(), "missing_sections": []}


class SummarizationAgent:
    """Uses Together chat for reasoning and delegates retrieval to RetrievalAgent."""

    def __init__(self):
        if not settings.together_api_key:
            raise RuntimeError("TOGETHER_API_KEY not set.")
        self.client = Together(api_key=settings.together_api_key)
        self.base_url = f"http://127.0.0.1:{settings.app_port}"

    # ------ Together Chat with simple retry ------
    def chat(self, messages: List[Dict[str, str]]) -> str:
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
                time.sleep(2**attempt)
        raise last_err or RuntimeError("Chat failed")

    # ------ A2A call to RetrievalAgent ------
    def call_retrieval(
        self,
        patient_id: str,
        drug_name: str,
        focus_sections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """POST to retrieval agent; optionally constrain by sections."""
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

    def _compute_coverage(
        self,
        first_results: List[Dict[str, Any]],
        requery_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """Return section coverage map: {eligibility|step_therapy|documentation|forms -> first_pass|requery|missing}."""
        required = ["eligibility", "step_therapy", "documentation", "forms"]
        first_sections = {r["section"] for r in first_results}
        re_sections = {r["section"] for r in (requery_results or [])}

        coverage: Dict[str, str] = {}
        for sec in required:
            if sec in first_sections:
                coverage[sec] = "first_pass"
            elif sec in re_sections:
                coverage[sec] = "requery"
            else:
                coverage[sec] = "missing"
        return coverage

    def summarize_with_reflection(
        self, patient_id: str, drug_name: str, fast: bool = False
    ) -> Dict[str, Any]:
        """Main orchestration: retrieval → (optional) reflection → final summary."""
        ts0 = _utc_ts()
        plan = patient_plan_map().get(patient_id, "Unknown")

        # ---- 1) First-pass retrieval
        t_r1_a = _utc_ts()
        ret1 = self.call_retrieval(patient_id, drug_name)
        t_r1_b = _utc_ts()
        first_results = list(ret1["output"]["results"])
        first_pass_hit_count = len(first_results)
        child_traces = [ret1["trace"]]

        # Track reflection details
        missing: List[str] = []
        ret2 = None
        requery_results: List[Dict[str, Any]] = []
        duration_reflection_llm_s = None

        # ---- 2) Optional reflection
        if not fast:
            # Ask LLM what's missing (force JSON)
            context_blob = "\n".join(
                [
                    f"- [{r['section']}] {r['content']} (doc:{r['doc_id']} score:{r['score']:.3f})"
                    for r in first_results
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
            t_ref_a = _utc_ts()
            raw_json_text = self.chat(messages)
            t_ref_b = _utc_ts()
            duration_reflection_llm_s = round(t_ref_b - t_ref_a, 3)

            parsed = extract_json_block(raw_json_text)
            if isinstance(parsed, dict):
                missing = [
                    m.strip().lower()
                    for m in parsed.get("missing_sections", [])
                    if isinstance(m, str)
                ]

            # 3) Focused re-query if we have actionable missing sections
            if missing:
                allowed = {"eligibility", "step_therapy", "documentation", "forms"}
                focus = [m for m in missing if m in allowed]
                if focus:
                    t_r2_a = _utc_ts()
                    ret2 = self.call_retrieval(patient_id, drug_name, focus_sections=focus)
                    t_r2_b = _utc_ts()
                    requery_results = list(ret2["output"]["results"])
                    child_traces.append(ret2["trace"])
        else:
            # Fast mode: skip reflection entirely
            parsed = {"summary": "", "missing_sections": []}

        # ---- 4) Final summarization over combined evidence
        combined_results = first_results + requery_results
        final_context = "\n".join(
            [f"- [{r['section']}] {r['content']} (doc:{r['doc_id']})" for r in combined_results]
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

        # ---- 5) Coverage + approval gap
        coverage = self._compute_coverage(first_results, requery_results)
        missing_for_approval = [sec for sec, st in coverage.items() if st == "missing"]

        # ---- 6) Build raw metrics expected by the UI
        raw_msgs = []
        if settings.trace_keep_raw_messages:
            note = "Fast mode ON: skipped reflection." if fast else f"Reflection identified missing sections: {missing}"
            raw_msgs = [
                {"role": "system", "content": "SummarizationAgent received task.", "ts": ts0},
                {"role": "assistant", "content": f"Delegated retrieval for {drug_name} on plan {plan}.", "ts": ts0},
                {"role": "assistant", "content": note, "ts": ts1},
            ]

        raw_intermediate: Dict[str, Any] = {
            "first_pass_llm_output": parsed,
            "focused_sections": [] if fast else missing,
            "fast_mode": fast,
            # NEW: counts + durations for the Reflection Details panel
            "first_pass_hit_count": first_pass_hit_count,
            "requery_hit_count": len(requery_results) if requery_results else 0 if not fast else None,
            "duration_first_retrieval_s": round(t_r1_b - t_r1_a, 3),
            "duration_reflection_llm_s": duration_reflection_llm_s,
            "duration_requery_retrieval_s": round(t_r2_b - t_r2_a, 3) if not fast and ret2 else None,
        }

        # Show what the re-query added that wasn't already present (nice for the Reflection tab)
        first_keys = {(r["doc_id"], r["section"]) for r in first_results}
        re_new = []
        for r in requery_results:
            key = (r["doc_id"], r["section"])
            if key not in first_keys:
                re_new.append(
                    {
                        "doc_id": r["doc_id"],
                        "section": r["section"],
                        "snippet": r["content"][:180],
                        "score": r["score"],
                    }
                )
        raw_intermediate["requery_new_items"] = re_new

        return {
            "agent": "summarization-agent",
            "state": "completed",
            "duration_s": round(ts1 - ts0, 3),
            "output": {
                "plan": plan,
                "drug_name": drug_name,
                "summary_markdown": final_text,
                "citations": [{"doc_id": r["doc_id"], "section": r["section"]} for r in combined_results],
                # NEW: power the UI's Needed-for-Approval and Reflection panels
                "section_coverage": coverage,
                "missing_for_approval": missing_for_approval,
            },
            "trace": {
                "agent": "summarization-agent",
                "states": [{"state": "submitted", "ts": ts0}, {"state": "completed", "ts": ts1}],
                "messages": raw_msgs,
                "children": child_traces,
            },
            "raw_intermediate": raw_intermediate,
        }


# --------- FastAPI / A2A-style endpoints ----------
def get_router(base_path: str = "/agents/summarization") -> APIRouter:
    """Expose agent card + task endpoint under one FastAPI app."""
    router = APIRouter(prefix=base_path, tags=["summarization-agent"])
    agent = SummarizationAgent()

    @router.get("/.well-known/agent-card.json")
    def agent_card() -> Dict[str, Any]:
        return {
            "a2a_schema_version": "1.0",
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
