#!/usr/bin/env python3
from __future__ import annotations
import argparse
import requests
from config import get_settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="P001")
    parser.add_argument("--drug", default="Humira")
    args = parser.parse_args()

    s = get_settings()
    url = f"http://127.0.0.1:{s.app_port}/agents/summarization/task"
    payload = {"goal": {"patient_id": args.patient, "drug_name": args.drug}}
    print("POST", url, payload)
    r = requests.post(url, json=payload, timeout=120)
    print("Status:", r.status_code)
    print(r.json())

if __name__ == "__main__":
    main()
