# agent/summarization_agent.py

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
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{"); end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except Exception:
            pass
    return {"summary": text.strip(), "missing_sections": []}

class SummarizationAgent:
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
                time.sleep(2 ** attempt)
        raise last_err or RuntimeError("Chat failed")

    # ------ A2A call to RetrievalAgent ------
    def call_retrieval(self, patient_id: str, drug_name: str, focus_sections: Optional[List[str]] = None) -> Dict[str, Any]:
        body = {
            "task_id": f"ret-{int(time.time()*1000)}",
            "goal": {"patient_id": patient_id, "drug_name": drug_name}
        }
        if focus_sections:
            body["goal"]["focus_sections"] = focus_sections

        with httpx.Client(timeout=60) as client:
            r = client.post(f"{self.base_url}/agents/retrieval/task", json=body)
            r.raise_for_status()
            return r.json()

    def summarize_with_reflection(self, patient_id: str, drug_name: str, fast: bool = False) -> Dict[str, Any]:
        ts0 = _utc_ts()
        plan = patient_plan_map().get(patient_id, "Unknown")

        # 1) delegate to retrieval agent
        ret1 = self.call_retrieval(patient_id, drug_name)
        results = list(ret1["output"]["results"])  # copy
        child_traces = [ret1["trace"]]

        missing: List[str] = []
        ret2 = None

        if not fast:
            # 2) ask the LLM what is missing (force JSON)
            context_blob = "\n".join([f"- [{r['section']}] {r['content']} (doc:{r['doc_id']} score:{r['score']:.3f})" for r in results])
            messages = [
                {"role": "system", "content": JSON_FALLBACK_PROMPT},
                {"role": "user", "content": f"Drug: {drug_name}\nPlan: {plan}\nEvidence:\n{context_blob}\n\nWhat is the PA policy summary? What sections are still missing?"},
            ]
            raw_json_text = self.chat(messages)
            parsed = extract_json_block(raw_json_text)
            if isinstance(parsed, dict):
                missing = [m.strip().lower() for m in parsed.get("missing_sections", []) if isinstance(m, str)]

            if missing:
                # 3) Focused re-query for specific sections (reflection)
                section_map = {"eligibility","step_therapy","documentation","forms"}
                focus = [m for m in missing if m in section_map]
                if focus:
                    ret2 = self.call_retrieval(patient_id, drug_name, focus_sections=focus)
                    child_traces.append(ret2["trace"])
                    results += ret2["output"]["results"]
        else:
            # Fast mode: skip reflection entirely
            parsed = {"summary": "", "missing_sections": []}

        # 4) Final summarization
        final_context = "\n".join([f"- [{r['section']}] {r['content']} (doc:{r['doc_id']})" for r in results])
        final_messages = [
            {"role": "system", "content": FINAL_SUMMARY_PROMPT},
            {"role": "user", "content": f"Drug: {drug_name}\nPlan: {plan}\nEvidence:\n{final_context}\n\nWrite the final summary now."},
        ]
        final_text = self.chat(final_messages)

        ts1 = _utc_ts()
        raw_msgs = []
        if settings.trace_keep_raw_messages:
            note = "Fast mode ON: skipped reflection." if fast else f"Reflection identified missing sections: {missing}"
            raw_msgs = [
                {"role": "system", "content": "SummarizationAgent received task.", "ts": ts0},
                {"role": "assistant", "content": f"Delegated retrieval for {drug_name} on plan {plan}.", "ts": ts0},
                {"role": "assistant", "content": note, "ts": ts1},
            ]

        return {
            "agent": "summarization-agent",
            "state": "completed",
            "duration_s": round(ts1 - ts0, 3),
            "output": {
                "plan": plan,
                "drug_name": drug_name,
                "summary_markdown": final_text,
                "citations": [{"doc_id": r["doc_id"], "section": r["section"]} for r in results],
            },
            "trace": {
                "agent": "summarization-agent",
                "states": [{"state": "submitted", "ts": ts0}, {"state": "completed", "ts": ts1}],
                "messages": raw_msgs,
                "children": child_traces,
            },
            "raw_intermediate": {
                "first_pass_llm_output": parsed,
                "focused_sections": [] if fast else missing,
                "fast_mode": fast,
            }
        }

# --------- FastAPI / A2A-style endpoints ----------
def get_router(base_path: str = "/agents/summarization") -> APIRouter:
    router = APIRouter(prefix=base_path, tags=["summarization-agent"])
    agent = SummarizationAgent()

    @router.get("/.well-known/agent-card.json")
    def agent_card() -> Dict[str, Any]:
        return {
            "id": "summarization-agent",
            "name": "PriorAuth Summarization Agent",
            "description": "Orchestrates retrieval and produces a policy summary with optional fast-mode (skips reflection).",
            "rpc_url": f"{base_path}/task",
            "input_schema": {"type": "object", "properties": {"patient_id": {"type": "string"}, "drug_name": {"type": "string"}, "fast": {"type": "boolean"}}},
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
