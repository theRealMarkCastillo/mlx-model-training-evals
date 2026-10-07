"""Canonical chat records shared by training validation and both evaluation paths."""

from collections import defaultdict
from itertools import zip_longest
import json
from pathlib import Path

from src.runs import REPO_ROOT
from src.schema import parse_and_validate

DATA_DIR = REPO_ROOT / "data"
STANDARD_SPLITS = ("train", "valid", "test")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise ValueError("Expected a positive integer")
    return number


def challenge_files(data_dir=DATA_DIR):
    return {p.stem: p for p in sorted(Path(data_dir).glob("challenge_*.jsonl"))}


def _balanced_prefix(records, count):
    """Round-robin across expected tools so a subset keeps the tool mix."""
    by_tool = defaultdict(list)
    for record in records:
        by_tool[record["expected"]["tool"]].append(record)
    interleaved = [r for group in zip_longest(*by_tool.values()) for r in group if r is not None]
    return interleaved[:count]


def load_samples(path, max_samples=None):
    """Load and validate chat records. A max_samples subset is tool-balanced."""
    if max_samples is not None:
        positive_int(max_samples)
    path = Path(path)
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
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
            "id": f"{path.name}:{line_number}", "messages": messages,
            "prompt": messages[1]["content"], "expected": target["parsed_data"],
            "normalized_expected": target["normalized_data"], "meta": record.get("meta"),
        })
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    if max_samples is None or max_samples >= len(records):
        return records
    return _balanced_prefix(records, max_samples)


def validate_splits(data_dir=DATA_DIR):
    """Every split and challenge set loads, and no prompt appears twice anywhere."""
    seen = set()
    names = [Path(data_dir) / f"{name}.jsonl" for name in STANDARD_SPLITS] + list(challenge_files(data_dir).values())
    for path in names:
        prompts = [r["prompt"] for r in load_samples(path)]
        if len(set(prompts)) != len(prompts) or seen.intersection(prompts):
            raise ValueError(f"Duplicate prompts in or across splits: {path.name}")
        seen.update(prompts)


def select_shots(path=DATA_DIR / "train.jsonl", count=5):
    """Few-shot demonstrations from the training split: one per tool when count == tool count."""
    return _balanced_prefix(load_samples(path), count) if count else []


def fewshot_messages(messages, shots):
    """Insert demonstration turns between the system prompt and the request.

    The same system prompt is kept, so the only difference from zero-shot is
    the worked examples. Training-split shots never overlap evaluation prompts.
    """
    demos = []
    for shot in shots:
        demos += [{"role": "user", "content": shot["prompt"]},
                  {"role": "assistant", "content": shot["messages"][-1]["content"]}]
    return [messages[0], *demos, *messages[1:]]
