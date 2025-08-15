"""
app.py
------
Gradio UI + FastAPI mount for the Agent-to-Agent (A2A) Prior Authorization demo.

This app allows users to:
- Select a Patient ID and Drug Name (from synthetic data).
- Run a summarization workflow that calls a Retrieval Agent and optionally
  performs a Reflection step to re-query for missing sections.
- View outputs in multiple tabs:
    - Summary (with "Needed for Prior Auth Approval" section)
    - Raw A2A messages
    - Human-readable execution trace
    - Detailed Reflection info
    - Benchmark latency for Fast vs Reflection modes
"""

from __future__ import annotations
import json
import time
import statistics
from typing import Any, Dict, List, Tuple

import httpx
import gradio as gr

from a2a_server import create_app
from config import get_settings
from data.prior_auth_docs import patient_plan_map, build_synthetic_corpus

# -----------------------------
# App & Settings Initialization
# -----------------------------
settings = get_settings()
fastapi_app = create_app()

# Build dropdown options from the synthetic corpus
PATIENT_OPTIONS = sorted(patient_plan_map().keys())
DRUG_OPTIONS = sorted({d["drug"] for d in build_synthetic_corpus()})

# Defaults for dropdowns
DEFAULT_PATIENT = PATIENT_OPTIONS[0] if PATIENT_OPTIONS else "P001"
DEFAULT_DRUG = "Humira" if "Humira" in DRUG_OPTIONS else (DRUG_OPTIONS[0] if DRUG_OPTIONS else "Humira")

# Timeout values (in seconds)
SUMMARIZER_TIMEOUT_S = 120.0
BENCHMARK_TIMEOUT_S = 120.0

# ============================================================
# Utility Functions
# ============================================================

def _pretty_json(data: Dict[str, Any]) -> str:
    """Return pretty-printed JSON string for display."""
    return json.dumps(data, indent=2, ensure_ascii=False)

def _timeline(trace: Dict[str, Any], depth: int = 0) -> str:
    """Recursively format a trace dictionary into human-readable markdown."""
    if not trace:
        return "_(no trace)_"
    pad = "  " * depth
    out = f"{pad}- Agent: **{trace.get('agent', 'unknown')}**\n"
    for st in trace.get("states", []) or []:
        ts = st.get("ts", "")
        out += f"{pad}  - {st.get('state', 'unknown')} at {ts}\n"
    for msg in trace.get("messages", []) or []:
        role = msg.get("role", "log")
        content = msg.get("content", "")
        out += f"{pad}  - {role}: {content}\n"
    for ch in trace.get("children", []) or []:
        out += _timeline(ch, depth + 1)
    return out

def _error_banner(text: str) -> str:
    """Generate an HTML error banner."""
    return (
        "<div style='padding:14px;background:#fdecea;border:1px solid #f5c6cb;"
        "border-radius:10px;font-size:16px;font-weight:600;color:#721c24;'>"
        f"❌ {text}</div>"
    )

def _info_banner(text: str) -> str:
    """Generate an HTML info banner."""
    return (
        "<div style='padding:14px;background:#fff3cd;border:1px solid #ffeeba;"
        "border-radius:10px;font-size:20px;font-weight:600;'>"
        f"⏳ {text}</div>"
    )

def _post_summarization(patient_id: str, drug_name: str, fast: bool, timeout: float) -> Dict[str, Any]:
    """
    Call the Summarization Agent's /task endpoint.
    
    Args:
        patient_id: Patient ID (string)
        drug_name: Drug name (string)
        fast: Whether to run in Fast mode (skip reflection)
        timeout: Timeout in seconds for HTTP request

    Returns:
        JSON response as a Python dictionary.
    """
    body = {
        "task_id": "ui-call",
        "goal": {"patient_id": patient_id.strip(), "drug_name": drug_name.strip(), "fast": bool(fast)},
    }
    url = f"http://127.0.0.1:{settings.app_port}/agents/summarization/task"
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=body)
        r.raise_for_status()
        return r.json()

# ============================================================
# Reflection Display Helpers
# ============================================================

def _reflection_md_inline(resp: Dict[str, Any]) -> str:
    """Minimal reflection summary for inline banner below Run button."""
    ri = resp.get("raw_intermediate") or {}
    fast = bool(ri.get("fast_mode"))
    first_hits = ri.get("first_pass_hit_count")
    re_hits = ri.get("requery_hit_count")
    if fast:
        return (
            "### Reflection Details\n"
            "- **Mode:** Fast (reflection skipped)\n"
            f"- **Missing sections:** _n/a_\n"
            f"- **First pass hits:** {first_hits if first_hits is not None else '-'}\n"
            f"- **Focused re-query hits:** _n/a_\n"
        )
    focused = ri.get("focused_sections") or []
    lines = ["### Reflection Details", "- **Mode:** Reflection (fast = false)"]
    lines.append(
        f"- **Missing sections detected:** {', '.join(focused)}" if focused else
        "- **Missing sections detected:** _none_"
    )
    if first_hits is not None:
        lines.append(f"- **First pass hits:** {first_hits}")
    if re_hits is not None:
        lines.append(f"- **Focused re-query hits:** {re_hits}")
    return "\n".join(lines) + "\n"

def _reflection_md_full(resp: Dict[str, Any]) -> str:
    """
    Detailed reflection info for the Reflection Details tab:
    - Mode, hit counts, durations
    - Missing sections
    - Section coverage table
    - Re-query snippets
    """
    ri = resp.get("raw_intermediate") or {}
    out = ["### Reflection Details"]
    fast = bool(ri.get("fast_mode"))
    first_hits = ri.get("first_pass_hit_count")
    re_hits = ri.get("requery_hit_count")
    out.append(f"- **Mode:** {'Fast (reflection skipped)' if fast else 'Reflection (fast = false)'}")
    out.append(f"- **First pass hits:** {first_hits if first_hits is not None else '-'}"
               + (f" ({ri.get('duration_first_retrieval_s')}s)" if ri.get('duration_first_retrieval_s') is not None else ""))
    if not fast:
        focused = ri.get("focused_sections") or []
        out.append("- **Missing sections detected:** " + (", ".join(focused) if focused else "_none_"))
        if re_hits is not None:
            suffix = f" ({ri.get('duration_requery_retrieval_s')}s)" if ri.get('duration_requery_retrieval_s') is not None else ""
            out.append(f"- **Focused re-query hits:** {re_hits}{suffix}")
        if ri.get("duration_reflection_llm_s") is not None:
            out.append(f"- **Reflection LLM duration:** {ri.get('duration_reflection_llm_s')}s")

    # Section coverage table
    coverage = resp.get("output", {}).get("section_coverage") or {}
    if coverage:
        out.append("\n#### Section coverage")
        out.append("| Section | Status |")
        out.append("|:--|:--|")
        status_emoji = {"first_pass": "✅ first pass", "requery": "⚠️ re-query", "missing": "❌ missing"}
        for sec in ["eligibility", "step_therapy", "documentation", "forms"]:
            st = coverage.get(sec, "missing")
            out.append(f"| {sec} | {status_emoji.get(st, st)} |")

    # Re-query snippets
    rq_items = ri.get("requery_new_items") or []
    if rq_items:
        out.append("\n#### Re-query additions")
        for it in rq_items:
            out.append(f"- **{it.get('section')}** — `{it.get('doc_id')}` — {it.get('snippet')} (score {it.get('score'):.3f})")

    out.append("")
    return "\n".join(out)

def _needed_for_pa_md(resp: Dict[str, Any]) -> str:
    """Render 'Needed for Prior Auth Approval' section based on missing_for_approval."""
    out = ["\n## Needed for Prior Auth Approval"]
    coverage = resp.get("output", {}).get("section_coverage") or {}
    missing = resp.get("output", {}).get("missing_for_approval") or []

    if not coverage:
        out.append("_(Coverage not available)_")
        return "\n".join(out) + "\n"

    if not missing:
        out.append("**All necessary sections found — PA can be approved.**")
        return "\n".join(out) + "\n"

    status_emoji = {"first_pass": "✅", "requery": "⚠️", "missing": "❌"}
    for sec in ["eligibility", "step_therapy", "documentation", "forms"]:
        st = coverage.get(sec, "missing")
        out.append(f"- {status_emoji.get(st, '❌')} **{sec}**")
    out.append("\n**Missing:** " + ", ".join(missing))
    return "\n".join(out) + "\n"

# ============================================================
# Main Run Logic
# ============================================================

def run_task(patient_id: str, drug_name: str, fast_mode: bool):
    """Handler for Run button click: calls summarizer and prepares outputs for all tabs."""
    pid = (patient_id or "").strip()
    drg = (drug_name or "").strip()
    if not pid or not drg:
        err = _error_banner("Please select both Patient ID and Drug Name.")
        return err, "{}", "_(no trace)_", "_(no reflection info)_", "_(no reflection info)_"

    try:
        resp = _post_summarization(pid, drg, bool(fast_mode), timeout=SUMMARIZER_TIMEOUT_S)
    except httpx.ReadTimeout:
        err = _error_banner("Request timed out. Try **Fast Mode**, reduce `MAX_NEW_TOKENS`, or retry later.")
        return err, "{}", "_(no trace)_", "_(no reflection info)_", "_(no reflection info)_"
    except Exception as e:
        err = _error_banner(f"Summarization failed: {e}")
        return err, "{}", "_(no trace)_", "_(no reflection info)_", "_(no reflection info)_"

    base_summary = resp.get("output", {}).get("summary_markdown", "") or "_(no summary)_"
    needed_md = _needed_for_pa_md(resp)
    summary_md = base_summary + "\n" + needed_md

    raw_json = _pretty_json(resp)
    human_trace_md = _timeline(resp.get("trace", {}))
    reflection_tab_md = _reflection_md_full(resp)
    reflection_inline_md = _reflection_md_inline(resp)

    return summary_md, raw_json, human_trace_md, reflection_tab_md, reflection_inline_md

def reset_form():
    """Reset inputs and outputs to defaults."""
    return (
        DEFAULT_PATIENT, DEFAULT_DRUG, True,
        "", "", "",  # summary/json/trace
        "_(Reflection details will appear here after a run.)_",  # Reflection tab placeholder
        "_(Reflection details will appear here after a run.)_",  # Inline placeholder
        "",  # status
        gr.update(interactive=True),
    )
# ============================================================
# Benchmark helpers (FAST vs REFLECTION)
# ============================================================

def _call_once(patient: str, drug: str, fast: bool, timeout: float) -> Tuple[float, str]:
    """
    Run a single summarization call and return (duration_s, error_msg_if_any).
    Uses server-reported duration when available; falls back to wall time.
    """
    t0 = time.perf_counter()
    try:
        data = _post_summarization(patient, drug, fast, timeout=timeout)
        t1 = time.perf_counter()
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        dur = float(meta.get("duration_s", t1 - t0))
        return dur, ""
    except Exception as e:
        t1 = time.perf_counter()
        return (t1 - t0), f"{type(e).__name__}: {e}"

def _run_trials(patient: str, drug: str, fast: bool, trials: int, warmup: int) -> Tuple[List[float], List[str]]:
    """
    Perform optional warmup calls, then measured trials.
    Returns (durations_list, per_trial_log_lines).
    """
    # Warmup (ignored errors)
    for _ in range(max(0, int(warmup))):
        try:
            _call_once(patient, drug, fast, timeout=BENCHMARK_TIMEOUT_S)
        except Exception:
            pass

    # Measured trials
    durations: List[float] = []
    logs: List[str] = []
    label = "FAST" if fast else "REFL"
    n = max(1, int(trials))
    for i in range(n):
        d, maybe_err = _call_once(patient, drug, fast, timeout=BENCHMARK_TIMEOUT_S)
        durations.append(d)
        line = f"{label} trial {i+1}/{n}: {d:.3f}s"
        if maybe_err:
            line += f"  ⚠️ {maybe_err}"
        logs.append(line)
    return durations, logs

def _summarize(name: str, durations: List[float]) -> Dict[str, Any]:
    """
    Compute simple aggregates for a set of durations.
    """
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
    """
    Produce a compact markdown table for latency aggregates.
    """
    def fmt(x): return "-" if x is None else f"{x:.3f}"
    md = [
        "### Latency Summary (seconds)",
        "",
        "| Mode | N | Mean | Median | p95 | Min | Max |",
        "|:-----|--:|-----:|------:|----:|----:|----:|",
    ]
    for r in rows:
        md.append(f"| {r['name']} | {r['count']} | {fmt(r['mean'])} | {fmt(r['median'])} | {fmt(r['p95'])} | {fmt(r['min'])} | {fmt(r['max'])} |")
    md.append("")
    return "\n".join(md)

def benchmark_ui(patient: str, drug: str, trials: int, warmup: int) -> Tuple[str, str]:
    """
    Entry used by the Benchmark tab.
    Runs FAST trials then REFLECTION trials and returns:
      (summary_markdown_table, per_trial_logs_markdown)
    """
    pid, drg = (patient or "").strip(), (drug or "").strip()
    if not pid or not drg:
        return _error_banner("Please set Patient ID and Drug before benchmarking."), ""

    fast_durs, fast_logs = _run_trials(pid, drg, fast=True, trials=trials, warmup=warmup)
    refl_durs, refl_logs = _run_trials(pid, drg, fast=False, trials=trials, warmup=warmup)

    rows = [_summarize("FAST", fast_durs), _summarize("REFL", refl_durs)]
    summary_md = _mk_summary_markdown(rows)
    logs_md = "### Per-trial logs\n\n```\n" + "\n".join(fast_logs + refl_logs) + "\n```"
    return summary_md, logs_md

# ============================================================
# UI Layout (Gradio Blocks)
# ============================================================

with gr.Blocks(title="A2A Prior Auth Demo") as demo:
    gr.Markdown("# Agent-to-Agent Demo — Prior Auth Retrieval & Summarization")
    gr.Markdown(
        f"_Model: **{settings.together_chat_model}**, Embeddings: **{settings.together_embed_model}**, Top-K: **{settings.top_k}**_",
        elem_id="model-info",
    )

    status = gr.HTML(value="")
    with gr.Row():
        # Dropdown for Patient ID
        patient = gr.Dropdown(
            label="Patient ID",
            choices=PATIENT_OPTIONS,
            value=DEFAULT_PATIENT,
            allow_custom_value=False,
            interactive=True,
        )
        # Dropdown for Drug Name
        drug = gr.Dropdown(
            label="Drug Name",
            choices=DRUG_OPTIONS,
            value=DEFAULT_DRUG,
            allow_custom_value=False,
            interactive=True,
        )

    with gr.Row():
        fast_mode = gr.Checkbox(label="Fast mode (skip reflection)", value=True)

    with gr.Row():
        run_btn = gr.Button("Run", variant="primary")
        reset_btn = gr.Button("Reset Form")

    reflection_inline = gr.Markdown(
        value="_(Reflection details will appear here after a run.)_",
        elem_id="reflection-inline",
        visible=True,
    )

    with gr.Tabs():
        with gr.Tab("Summary"):
            out_summary = gr.Markdown()
        with gr.Tab("Raw A2A Messages"):
            out_json = gr.Code(language="json")
        with gr.Tab("Human Trace"):
            out_trace = gr.Markdown()
        with gr.Tab("Reflection Details", elem_id="reflection-tab"):
            out_reflection = gr.Markdown(
                value="_(Reflection details will appear here after a run.)_", elem_id="reflection-md"
            )
        with gr.Tab("Benchmark"):
            # Benchmark tab UI
            gr.Markdown("Compare latency between **Fast Mode** and **Reflection Mode** for a fixed patient/drug.")
            with gr.Row():
                b_patient = gr.Dropdown(
                    label="Patient ID",
                    choices=PATIENT_OPTIONS,
                    value=DEFAULT_PATIENT,
                    allow_custom_value=False,
                    interactive=True,
                )
                b_drug = gr.Dropdown(
                    label="Drug Name",
                    choices=DRUG_OPTIONS,
                    value=DEFAULT_DRUG,
                    allow_custom_value=False,
                    interactive=True,
                )
            with gr.Row():
                b_trials = gr.Number(label="Trials", value=5, precision=0)
                b_warmup = gr.Number(label="Warmup", value=1, precision=0)
            with gr.Row():
                bench_btn = gr.Button("Run Benchmark", variant="secondary")
            bench_status = gr.HTML("")
            bench_summary = gr.Markdown()
            bench_logs = gr.Markdown()

            # Busy banner for benchmark
            def bench_busy():
                return (
                    "<div style='padding:14px;background:#e7f1ff;border:1px solid #b6daff;"
                    "border-radius:10px;font-size:18px;font-weight:600;'>📊 Benchmark running…</div>",
                    gr.update(interactive=False),
                )

            bench_click = bench_btn.click(
                fn=bench_busy, inputs=None, outputs=[bench_status, bench_btn], queue=False
            )
            bench_click.then(
                fn=lambda p, d, t, w: benchmark_ui(p, d, t, w),
                inputs=[b_patient, b_drug, b_trials, b_warmup],
                outputs=[bench_summary, bench_logs]
            ).then(
                fn=lambda: ("", gr.update(interactive=True)),
                inputs=None, outputs=[bench_status, bench_btn], queue=False
            )

    def show_busy(fast: bool):
        """Display 'in-progress' banner and disable Run button."""
        label = " (fast mode)" if fast else ""
        return _info_banner(f"Retrieval in process{label}…"), gr.update(interactive=False)

    run_click = run_btn.click(
        fn=show_busy, inputs=[fast_mode], outputs=[status, run_btn], queue=False
    )
    run_click.then(
        fn=run_task, inputs=[patient, drug, fast_mode],
        outputs=[out_summary, out_json, out_trace, out_reflection, reflection_inline]
    ).then(
        fn=lambda: ("", gr.update(interactive=True)),
        inputs=None, outputs=[status, run_btn], queue=False
    )

    reset_btn.click(
        fn=reset_form, inputs=None,
        outputs=[patient, drug, fast_mode, out_summary, out_json, out_trace, out_reflection, reflection_inline, status, run_btn]
    )

# Mount Gradio app under FastAPI
fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
app = fastapi_app
