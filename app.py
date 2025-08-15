# app.py
from __future__ import annotations
import json
from typing import Any, Dict
import httpx
import gradio as gr
from a2a_server import create_app
from config import get_settings

# -----------------------------
# App / settings
# -----------------------------
settings = get_settings()
fastapi_app = create_app()

DEFAULT_PATIENT = "P001"
DEFAULT_DRUG = "Humira"

# -----------------------------
# Helpers
# -----------------------------
def _pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)

def run_task(patient_id: str, drug_name: str, fast_mode: bool):
    """
    Call the summarization agent A2A endpoint (non-streaming) and format outputs.
    When fast_mode=True, we send a hint to skip reflection in the agent.
    """
    body = {
        "task_id": "ui-call",
        "goal": {
            "patient_id": patient_id.strip(),
            "drug_name": drug_name.strip(),
            "fast": bool(fast_mode),
        }
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(
            f"http://127.0.0.1:{settings.app_port}/agents/summarization/task",
            json=body
        )
        r.raise_for_status()
        resp = r.json()

    # Human-friendly trace
    def timeline(trace: Dict[str, Any], depth=0) -> str:
        pad = "  " * depth
        out = f"{pad}- Agent: **{trace.get('agent')}**\n"
        for st in trace.get("states", []):
            out += f"{pad}  - {st['state']} at {st['ts']}\n"
        for msg in trace.get("messages", []):
            out += f"{pad}  - {msg['role']}: {msg['content']}\n"
        for ch in trace.get("children", []) or []:
            out += timeline(ch, depth + 1)
        return out

    summary_md = resp.get("output", {}).get("summary_markdown", "")
    raw_json = _pretty_json(resp)
    human_trace_md = timeline(resp.get("trace", {}))
    return summary_md, raw_json, human_trace_md

def reset_form():
    """Reset inputs to defaults and clear outputs and status; also re-enable Run button."""
    return (
        DEFAULT_PATIENT,  # patient
        DEFAULT_DRUG,     # drug
        True,             # fast_mode
        "",               # out_summary
        "",               # out_json
        "",               # out_trace
        "",               # status
        gr.update(interactive=True),  # run_btn re-enabled
    )

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="A2A Prior Auth Demo") as demo:
    gr.Markdown("# Agent-to-Agent Demo — Prior Auth Retrieval & Summarization")

    # Big status banner shown while the request is in-flight
    status = gr.HTML(value="")

    with gr.Row():
        patient = gr.Textbox(label="Patient ID", value=DEFAULT_PATIENT)
        drug = gr.Textbox(label="Drug Name", value=DEFAULT_DRUG)

    with gr.Row():
        fast_mode = gr.Checkbox(label="Fast mode (skip reflection)", value=True)

    with gr.Row():
        run_btn = gr.Button("Run", variant="primary")
        reset_btn = gr.Button("Reset Form")

    with gr.Tab("Summary"):
        out_summary = gr.Markdown()
    with gr.Tab("Raw A2A Messages"):
        out_json = gr.Code(language="json")
    with gr.Tab("Human Trace"):
        out_trace = gr.Markdown()

    # Show a large in-progress banner immediately when user clicks Run and disable the button
    def show_busy(fast: bool):
        label = " (fast mode)" if fast else ""
        html = (
            "<div style='padding:14px;"
            "background:#fff3cd;border:1px solid #ffeeba;border-radius:10px;"
            "font-size:20px;font-weight:600;'>"
            f"⏳ Retrieval in process{label}…</div>"
        )
        return html, gr.update(interactive=False)

    run_click = run_btn.click(
        fn=show_busy,
        inputs=[fast_mode],
        outputs=[status, run_btn],
        queue=False,
    )

    # Then run the task
    run_click.then(
        fn=run_task,
        inputs=[patient, drug, fast_mode],
        outputs=[out_summary, out_json, out_trace]
    ).then(
        # Clear the status and re-enable the button
        fn=lambda: ("", gr.update(interactive=True)),
        inputs=None,
        outputs=[status, run_btn],
        queue=False
    )

    # Reset inputs and outputs (also clears the status line and re-enables the button)
    reset_btn.click(
        fn=reset_form,
        inputs=None,
        outputs=[patient, drug, fast_mode, out_summary, out_json, out_trace, status, run_btn]
    )

# Mount Gradio under the FastAPI app
fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

# Export for uvicorn
app = fastapi_app
