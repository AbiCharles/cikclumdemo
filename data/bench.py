#!/usr/bin/env python3
from __future__ import annotations
import argparse, time, statistics, csv
import requests
from config import get_settings

def call_once(patient: str, drug: str, fast: bool, timeout: float = 120.0) -> float:
    s = get_settings()
    url = f"http://127.0.0.1:{s.app_port}/agents/summarization/task"
    payload = {"task_id": "bench", "goal": {"patient_id": patient, "drug_name": drug, "fast": fast}}
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    t1 = time.perf_counter()
    r.raise_for_status()
    data = r.json()
    # Prefer server-reported duration if present; otherwise fall back to wall time
    meta = data.get("meta", {})
    dur = float(meta.get("duration_s", t1 - t0))
    return dur

def run_trials(patient: str, drug: str, fast: bool, trials: int, warmup: int) -> list[float]:
    # Warmup calls (don’t record)
    for _ in range(warmup):
        try:
            call_once(patient, drug, fast)
        except Exception:
            pass
    # Timed trials
    durations = []
    for i in range(trials):
        dur = call_once(patient, drug, fast)
        durations.append(dur)
        print(f"{'FAST' if fast else 'REFL'} trial {i+1}/{trials}: {dur:.3f}s")
    return durations

def summarize(name: str, durations: list[float]) -> dict:
    if not durations:
        return {"name": name, "count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "name": name,
        "count": len(durations),
        "mean": statistics.fmean(durations),
        "median": statistics.median(durations),
        "p95": sorted(durations)[max(0, int(len(durations)*0.95)-1)],
        "min": min(durations),
        "max": max(durations),
    }

def print_table(rows: list[dict]):
    # Simple text table
    headers = ["name","count","mean","median","p95","min","max"]
    print("\nLATENCY SUMMARY (seconds)")
    print("-" * 72)
    print("{:<7} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}".format(*headers))
    for r in rows:
        vals = [
            r["name"],
            r["count"],
            f"{r['mean']:.3f}" if r["mean"] is not None else "-",
            f"{r['median']:.3f}" if r["median"] is not None else "-",
            f"{r['p95']:.3f}" if r["p95"] is not None else "-",
            f"{r['min']:.3f}" if r["min"] is not None else "-",
            f"{r['max']:.3f}" if r["max"] is not None else "-",
        ]
        print("{:<7} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}".format(*vals))
    print("-" * 72)

def maybe_write_csv(path: str | None, rows: list[dict]):
    if not path:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","count","mean","median","p95","min","max"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote summary CSV -> {path}")

def main():
    ap = argparse.ArgumentParser(description="Benchmark fast vs reflection modes")
    ap.add_argument("--patient", default="P001")
    ap.add_argument("--drug", default="Humira")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    print(f"Benchmarking patient={args.patient} drug={args.drug} trials={args.trials} (warmup={args.warmup})")

    fast_durations = run_trials(args.patient, args.drug, fast=True, trials=args.trials, warmup=args.warmup)
    refl_durations = run_trials(args.patient, args.drug, fast=False, trials=args.trials, warmup=args.warmup)

    rows = [summarize("FAST", fast_durations), summarize("REFL", refl_durations)]
    print_table(rows)
    maybe_write_csv(args.csv or None, rows)

if __name__ == "__main__":
    main()
