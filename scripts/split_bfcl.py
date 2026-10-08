"""Split the committed BFCL simple data into train/valid/test for the self-split LoRA experiment.

BFCL is eval-only, so this split is deliberately *in-distribution*: it answers "does a LoRA
trained on real function-calling schemas beat the base model's prompting on the same kind of
schema?", not "does it generalize to unseen tasks?" The README/docs label it accordingly.

    uv run python scripts/split_bfcl.py

Writes data/bfcl_split/{train,valid,test}.jsonl with a fixed seed (270/45/80 of 395).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "data" / "bfcl_simple.jsonl"
OUT = ROOT / "data" / "bfcl_split"
SPLITS = (("train", 270), ("valid", 45), ("test", 80))
SEED = 42


def main():
    records = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    random.Random(SEED).shuffle(records)
    OUT.mkdir(parents=True, exist_ok=True)
    start = 0
    for name, count in SPLITS:
        chunk = records[start:start + count]
        start += count
        (OUT / f"{name}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in chunk))
        print(f"{name}: {len(chunk)} records -> {OUT / f'{name}.jsonl'}")
    assert start == len(records), f"split sizes {sum(c for _, c in SPLITS)} != {len(records)}"


if __name__ == "__main__":
    main()
