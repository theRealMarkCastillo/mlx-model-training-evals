"""Fetch BFCL v3 simple and convert it into this repo's chat format (data/bfcl_simple.jsonl).

BFCL is Apache-2.0 and ships one JSON object per line (questions) with ground truth in
`possible_answer/`. This downloads those two files, converts them via `src/bfcl.py`, and
writes the committed subset used by the real-data pilot.

    uv run python scripts/fetch_bfcl.py

Run it from the repo root. The output is deterministic given the dataset revision, and the
skip report tells you which records were left out because their schema uses a feature the
grammar does not support (e.g. `tuple`, `any`, or an object with no `properties`).
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bfcl import load_bfcl, save_records  # noqa: E402

BASE = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
OUT = ROOT / "data" / "bfcl_simple.jsonl"
CACHE = Path("/tmp") / "bfcl-fetch"


def fetch(relative):
    local = CACHE / relative.replace("/", "__")
    CACHE.mkdir(parents=True, exist_ok=True)
    if not local.is_file():
        url = f"{BASE}/{relative}"
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, local)
    return local


def main():
    questions = fetch("BFCL_v3_simple.json")
    answers = fetch("possible_answer/BFCL_v3_simple.json")
    records, skipped = load_bfcl(questions, answers)
    save_records(records, OUT)
    print(f"Wrote {len(records)} records to {OUT.relative_to(ROOT)}")
    print(f"Skipped: {json.dumps(skipped)}")
    return records


if __name__ == "__main__":
    main()
