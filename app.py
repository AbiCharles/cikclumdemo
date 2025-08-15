# app.py
from __future__ import annotations
import json
import time
import statistics
from typing import Any, Dict, List, Tuple
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
# Helpers (shared)
# -----------------------------
def _pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)

def _timeline(trace: Dict[str, Any], depth=0) -> str:
    pad = "  " * depth
    out = f"{pad}- Agent: **{trace.get('agent')}**\n"
    for st in trace.get("states", []):
        out += f"{pad}  - {st['state']} at {st['ts']}\n"
    for msg in trace.get("messages", []):
        out += f"{pad}  - {msg['role']}: {msg['content']}\n"
    for ch in trace.get("children", []) or []:
        out += _timeline(ch, depth + 1)
    return out

def _post_summarization(patient_id: str, drug_name: str, fast: bool, timeout: float = 60.0) -> Dict[str, Any]:
    body = {
        "task_id": "ui-call",
        "goal": {
            "patient_id": patient_id.strip(),
            "drug_name": drug_name.strip(),
            "fast": bool(fast),
        }
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"http://127.0.0.1:{settings.app_port}/agents/summarization/task", json=body)
        r.raise_for_status()
        return r.json()

# -----------------------------
# Main Run handlers (Summary tabs)
# -----------------------------
def run_task(patient_id: str, drug_name: str, fast_mode: bool):
    resp = _post_summarization(patient_id, drug_name, fast_mode)
    summary_md = resp.get("output", {}).get("summary_markdown", "")
    raw_json = _pretty_json(resp)
    human_trace_md = _timeline(resp.get("trace", {}))
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
# Benchmark tab helpers
# -----------------------------
def _call_once(patient: str, drug: str, fast: bool, timeout: float = 60.0) -> float:
    t0 = time.perf_counter()
    data = _post_summarization(patient, drug, fast, timeout=timeout)
    t1 = time.perf_counter()
    meta = data.get("meta", {})
    # Use server-reported duration if present; else wall time
    return float(meta.get("duration_s", t1 - t0))

def _run_trials(patient: str, drug: str, fast: bool, trials: int, warmup: int) -> Tuple[List[float], List[str]]:
    # Warmups (not logged)
    for _ in range(max(0, int(warmup))):
        try:
            _call_once(patient, drug, fast)
        except Exception:
            # ignore warmup errors
            pass
    # Timed trials
    durations: List[float] = []
    logs: List[str] = []
    label = "FAST" if fast else "REFL"
    for i in range(max(1, int(trials))):
        d = _call_once(patient, drug, fast)
        durations.append(d)
        logs.append(f"{label} trial {i+1}/{trials}: {d:.3f}s")
    return durations, logs

def _summarize(name: str, durations: List[float]) -> Dict[str, Any]:
    if not durations:
        return {"name": name, "count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "name": name,
        "count": len(durations),
        "mean": statistics.fmean(durations),
        "median": statistics.median(durations),
        "p95": sorted(durations)[max(0, int(len(durations) * 0.95) - 1)],
        "min": min(durations),
        "max": max(durations),
    }

def _mk_summary_markdown(rows: List[Dict[str, Any]]) -> str:
    def f(x): return "-" if x is None else f"{x:.3f}"
    md = []
    md.append("### Latency Summary (seconds)")
    md.append("")
    md.append("| Mode | N | Mean | Median | p95 | Min | Max |")
    md.append("|:-----|--:|-----:|------:|----:|----:|----:|")
    for r in rows:
        md.append(f"| {r['name']} | {r['count']} | {f(r['mean'])} | {f(r['median'])} | {f(r['p95'])} | {f(r['min'])} | {f(r['max'])} |")
    md.append("")
    return "\n".join(md)

def benchmark_ui(patient: str, drug: str, trials: int, warmup: int) -> Tuple[str, str]:
    # Run FAST first, then REFL
    fast_durs, fast_logs = _run_trials(patient, drug, fast=True, trials=trials, warmup=warmup)
    refl_durs, refl_logs = _run_trials(patient, drug, fast=False, trials=trials, warmup=warmup)

    rows = [
        _summarize("FAST", fast_durs),
        _summarize("REFL", refl_durs),
    ]
    summary_md = _mk_summary_markdown(rows)
    logs_md = "### Per-trial logs\n\n```\n" + "\n".join(fast_logs + refl_logs) + "\n```"
    return summary_md, logs_md

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="A2A Prior Auth Demo") as demo:
    gr.Markdown("# Agent-to-Agent Demo — Prior Auth Retrieval & Summarization")

    # Big status banner (main run)
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

    # Show large in-progress banner and disable Run
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

    # Then run the task and clear banner / re-enable button
    run_click.then(
        fn=run_task,
        inputs=[patient, drug, fast_mode],
        outputs=[out_summary, out_json, out_trace]
    ).then(
        fn=lambda: ("", gr.update(interactive=True)),
        inputs=None,
        outputs=[status, run_btn],
        queue=False
    )

    # Reset inputs/outputs/status and re-enable button
    reset_btn.click(
        fn=reset_form,
        inputs=None,
        outputs=[patient, drug, fast_mode, out_summary, out_json, out_trace, status, run_btn]
    )

    # -------------------------
    # Benchmark tab
    # -------------------------
    with gr.Tab("Benchmark"):
        gr.Markdown("Compare latency between **Fast Mode** and **Reflection Mode** for a fixed patient/drug.")
        with gr.Row():
            b_patient = gr.Textbox(label="Patient ID", value=DEFAULT_PATIENT)
            b_drug = gr.Textbox(label="Drug Name", value=DEFAULT_DRUG)
        with gr.Row():
            b_trials = gr.Number(label="Trials", value=5, precision=0)
            b_warmup = gr.Number(label="Warmup", value=1, precision=0)
        with gr.Row():
            bench_btn = gr.Button("Run Benchmark", variant="secondary")
        bench_status = gr.HTML("")
        bench_summary = gr.Markdown()
        bench_logs = gr.Markdown()

        # Show busy banner and disable button quickly
        def bench_busy():
            html = (
                "<div style='padding:14px;"
                "background:#e7f1ff;border:1px solid #b6daff;border-radius:10px;"
                "font-size:18px;font-weight:600;'>"
                "📊 Benchmark running…</div>"
            )
            return html, gr.update(interactive=False)

        bench_click = bench_btn.click(
            fn=bench_busy,
            inputs=None,
            outputs=[bench_status, bench_btn],
            queue=False
        )
        bench_click.then(
            fn=benchmark_ui,
            inputs=[b_patient, b_drug, b_trials, b_warmup],
            outputs=[bench_summary, bench_logs]
        ).then(
            fn=lambda: ("", gr.update(interactive=True)),
            inputs=None,
            outputs=[bench_status, bench_btn],
            queue=False
        )

# Mount Gradio under the FastAPI app
fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

# Export for uvicorn
app = fastapi_app
