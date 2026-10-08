"""Load BFCL records into this repo's chat format, with per-record system prompts.

BFCL ([gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard),
Apache-2.0) is one JSON object per line, one file per category, with ground truth in
`possible_answer/`. This module maps a question record into the same `{system, user,
assistant}` shape the rest of the repo trains and evaluates on, so the existing eval
machinery can run against real schemas.

Two deviations are deliberate and must be reported wherever these numbers are quoted:

* **Envelope.** BFCL's canonical output is a list of `{name, arguments}` calls; this repo
  uses `{"tool": ..., "parameters": ...}`. The loader writes the latter so results stay
  comparable to the synthetic task.
* **Reference choice.** BFCL ground truth gives, per argument, a *set* of acceptable
  values (and `""` meaning "may be omitted"). The reference completion picks one canonical
  value, and the acceptable sets are kept in `meta` so a BFCL-style metric can be scored
  later. This repo's strict exact match is therefore *not* the right metric for BFCL.

Only records whose schemas fit the supported subset are converted; the rest are skipped
and counted, never guessed at.
"""

import json
from pathlib import Path

# BFCL uses Python type names; `number`/`object`/`list` are accepted synonyms.
SUPPORTED_TYPES = {"dict", "object", "array", "list", "string", "integer", "float", "number", "boolean"}
SCALAR_TYPES = {"string", "integer", "float", "number", "boolean"}
NO_ACTION = '{"tool":"no_action","parameters":{"reason":"unsupported_request"}}'


def is_supported(schema):
    """True if the grammar (and this loader) can handle this parameter schema subtree."""
    if not isinstance(schema, dict):
        return False
    kind = schema.get("type")
    if kind not in SUPPORTED_TYPES:
        return False
    if kind in ("array", "list"):
        return "items" in schema and is_supported(schema["items"])
    if kind in ("dict", "object"):
        return "properties" in schema and all(is_supported(prop) for prop in schema["properties"].values())
    return True


def _type_name(kind):
    return "float" if kind == "number" else "list" if kind == "array" else "object" if kind == "dict" else kind


def _describe_param(schema, required):
    kind = _type_name(schema.get("type"))
    bits = [kind]
    if kind == "list":
        bits.append(f"of {_type_name(schema['items'].get('type'))}" if "items" in schema else "of unknown")
    description = schema.get("description")
    if description:
        bits.append(f"({description})")
    if required:
        bits.append("(required)")
    else:
        bits.append("(optional)")
    return " ".join(parts for parts in bits if parts)


def build_system_prompt(functions):
    """The per-record system prompt: the function list plus the envelope contract."""
    lines = [
        "You are an automated tool-calling assistant.",
        "Respond to the user's request with exactly one JSON object in this envelope, using exact types:",
        "",
        '{"tool": "<function name>", "parameters": {"<argument>": <value>, ...}}',
        "",
        "Available functions:",
    ]
    for function in functions:
        params = function.get("parameters") or {}
        required = set(params.get("required") or [])
        lines.append(f"- {function['name']}: {function.get('description', '')}")
        for name, schema in (params.get("properties") or {}).items():
            lines.append(f"    {name}: {_describe_param(schema, name in required)}")
    lines += [
        "",
        "Use exact parameter types. Omit optional parameters unless the request specifies them.",
        "Respond with only the JSON object — no prose, no markdown fences.",
        "If no function satisfies the request, respond with: " + NO_ACTION,
    ]
    return "\n".join(lines)


def canonical_arguments(acceptable_args):
    """Pick one value per argument from its acceptable set; `""` means the argument is absent."""
    arguments = {}
    for name, acceptable in acceptable_args.items():
        values = [value for value in acceptable if value != ""]
        if values:
            arguments[name] = values[0]
    return arguments


def canonical_completion(function_name, acceptable_args):
    return json.dumps({"tool": function_name, "parameters": canonical_arguments(acceptable_args)},
                      separators=(",", ":"))


def load_bfcl(questions_path, answers_path=None, *, max_records=None):
    """Convert a BFCL category into this repo's chat records.

    Returns `(records, skipped, skip_reasons)` where each record carries `meta` with the
    BFCL id, the function schema, and the per-argument acceptable-value sets. `answers_path`
    may be omitted (for categories like irrelevance that ship no ground truth).
    """
    questions_path, max_records = Path(questions_path), None if max_records is None else int(max_records)
    answers = {}
    if answers_path:
        answers = {rec["id"]: rec for rec in _read_jsonl(Path(answers_path))}
    records, skipped = [], {"no_answer": 0, "unsupported_schema": 0, "multiple_calls": 0}
    for record in _read_jsonl(questions_path):
        if max_records is not None and len(records) >= max_records:
            break
        bfcl_id = record["id"]
        functions = record.get("function") or []
        if any(not is_supported(fn.get("parameters") or {}) for fn in functions):
            skipped["unsupported_schema"] += 1
            continue
        answer = answers.get(bfcl_id)
        if answer is None:
            skipped["no_answer"] += 1
            continue
        ground_truth = answer["ground_truth"]
        if len(ground_truth) != 1:
            skipped["multiple_calls"] += 1
            continue
        (function_name, acceptable_args), = ground_truth[0].items()
        question = " ".join(turn["content"] for turn in record["question"][0] if turn["role"] == "user")
        meta = {
            "tool": function_name, "bfcl_id": bfcl_id, "source": "bfcl",
            "function": functions[0],
            "acceptable": {name: acceptable for name, acceptable in acceptable_args.items()},
        }
        records.append({
            "id": f"{Path(questions_path).stem}:{bfcl_id}",
            "prompt": question,
            "expected": {"tool": function_name, "parameters": canonical_arguments(acceptable_args)},
            "meta": meta,
            "messages": [
                {"role": "system", "content": build_system_prompt(functions)},
                {"role": "user", "content": question},
                {"role": "assistant", "content": canonical_completion(function_name, acceptable_args)},
            ],
        })
    return records, skipped


def _read_jsonl(path):
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def save_records(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
