"""Canonical chat records shared by training validation and both evaluation paths."""

import json
from pathlib import Path

from src.schema import parse_and_validate


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise ValueError("Expected a positive integer")
    return number


def load_samples(path, max_samples=None):
    if max_samples is not None:
        positive_int(max_samples)
    records = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record.get("messages", [])
        if ([m.get("role") for m in messages] != ["system", "user", "assistant"]
                or any(not isinstance(m.get("content"), str) for m in messages)):
            raise ValueError(f"{path}:{line_number}: expected system, user, assistant text messages")
        target = parse_and_validate(messages[-1]["content"])
        if not target["is_pure_json"] or not target["is_schema_valid"]:
            raise ValueError(f"{path}:{line_number}: invalid target: {target['error']}")
        records.append({
            "id": f"{Path(path).name}:{line_number}", "messages": messages,
            "prompt": messages[1]["content"], "expected": target["parsed_data"],
            "normalized_expected": target["normalized_data"],
        })
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records[:max_samples]


def validate_splits(data_dir):
    seen = set()
    for name in ("train", "valid", "test"):
        records = load_samples(Path(data_dir) / f"{name}.jsonl")
        prompts = [r["prompt"] for r in records]
        if len(set(prompts)) != len(prompts) or seen.intersection(prompts):
            raise ValueError(f"Duplicate prompts in or across splits: {name}")
        seen.update(prompts)
