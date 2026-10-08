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
# The refusal schema for `irrelevance`, where the correct answer is "do not call anything".
NO_ACTION_SCHEMA = {"type": "dict", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}


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
    """Pick one value per argument from its acceptable set; `""` means the argument is absent.

    Flat (scalar) fields only; nested values go through `collapse_arguments`, which is
    schema-aware because BFCL wraps *every* value — including nested dicts and lists — in
    a list of acceptable alternatives.
    """
    arguments = {}
    for name, acceptable in acceptable_args.items():
        values = [value for value in acceptable if value != ""]
        if values:
            arguments[name] = values[0]
    return arguments


def _collapse_field(schema, acceptable):
    """Resolve one field: pick the first non-empty alternative, then collapse by schema."""
    alternatives = [value for value in acceptable if value != ""]
    if not alternatives:
        return None
    return _collapse_value(schema, alternatives[0])


def _collapse_value(schema, value):
    """Collapse a value that is *not* wrapped in an acceptable-set, by its schema type."""
    kind = schema.get("type")
    if kind in ("dict", "object"):
        result = {}
        for name, sub_schema in (schema.get("properties") or {}).items():
            if name in value:
                collapsed = _collapse_field(sub_schema, value[name])
                if collapsed is not None:
                    result[name] = collapsed
        return result
    if kind in ("array", "list"):
        item_schema = schema.get("items")
        return [_collapse_item(item_schema, element) if item_schema is not None else element
                for element in value]
    return value


def _collapse_item(item_schema, element):
    if item_schema.get("type") in ("dict", "object") and isinstance(element, dict):
        return _collapse_value(item_schema, element)
    if item_schema.get("type") in ("array", "list") and isinstance(element, list):
        return _collapse_value(item_schema, element)
    return element


def _matches_type(value, schema):
    kind = schema.get("type")
    if kind in ("dict", "object"):
        return isinstance(value, dict)
    if kind in ("array", "list"):
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in ("float", "number"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    return True


def matches_type(value, schema):
    """True if `value` is a legal instance of `schema` (the schema's `type` field)."""
    return _matches_type(value, schema)


def collapse_arguments(parameters, acceptable_args):
    """Schema-aware canonical arguments, or None if a value contradicts its schema.

    Returns None when BFCL's own ground truth disagrees with the function schema (this
    happens in the wild, e.g. a `string` field whose answer is a boolean), so the record
    is skipped rather than scored against an impossible reference.
    """
    properties = (parameters or {}).get("properties") or {}
    arguments = {}
    for name, acceptable in acceptable_args.items():
        schema = properties.get(name) or {}
        value = _collapse_field(schema, acceptable)
        if value is None:
            continue
        if not _matches_type(value, schema):
            return None
        arguments[name] = value
    return arguments


def canonical_completion(function_name, acceptable_args, parameters=None):
    if parameters is not None:
        arguments = collapse_arguments(parameters, acceptable_args)
        if arguments is None:
            raise ValueError("acceptable args contradict the function schema")
    else:
        arguments = canonical_arguments(acceptable_args)
    return json.dumps({"tool": function_name, "parameters": arguments}, separators=(",", ":"), ensure_ascii=False)


def parse_bfcl_call(raw):
    """Recover a `{tool, parameters}` JSON object from model output, or None.

    Mirrors `src.schema.parse_and_validate`'s strictness (duplicate keys and non-finite
    numbers rejected) but stops at the envelope: the function name and parameter schema are
    arbitrary here, so no Pydantic model is applied.
    """
    from src.schema import _finite_float, _first_embedded_object, _invalid_constant, _unique_object

    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_invalid_constant,
                               parse_float=_finite_float)
    text = raw.strip()
    try:
        try:
            parsed = decoder.decode(text)
        except json.JSONDecodeError:
            parsed = _first_embedded_object(decoder, text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, RecursionError):
        return None
    return None


def score_bfcl_call(parsed, meta):
    """Score one parsed call against BFCL ground truth (acceptable-value sets).

    A call is right when the function name matches and every provided argument's value is in
    the ground-truth acceptable set, with every required argument present — BFCL's answers
    are sets of alternatives, so exact match is the wrong standard.
    """
    expected_tool = meta["tool"]
    acceptable = meta.get("acceptable") or {}
    schema = (meta.get("function") or {}).get("parameters") or {}
    required = set(schema.get("required") or [])
    properties = schema.get("properties") or {}

    if not isinstance(parsed, dict) or not isinstance(parsed.get("tool"), str) or \
            not isinstance(parsed.get("parameters"), dict):
        return {"is_schema_valid": False, "tool_correct": False, "args_correct": False, "error_category": "json"}

    tool = parsed["tool"]
    params = parsed["parameters"]
    schema_valid = all(name in params for name in required) and all(
        matches_type(params[name], properties.get(name) or {}) for name in params)
    tool_correct = tool == expected_tool
    args_correct = tool_correct and schema_valid and all(
        params[name] in acceptable.get(name, []) for name in params)
    category = None if args_correct else ("schema" if not schema_valid else "tool" if not tool_correct else "parameters")
    return {"is_schema_valid": schema_valid, "tool_correct": tool_correct, "args_correct": args_correct,
            "error_category": category}


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
    records, skipped = [], {"no_answer": 0, "unsupported_schema": 0, "multiple_calls": 0,
                            "schema_mismatch": 0, "name_mismatch": 0}
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
        called = next((fn for fn in functions if fn["name"] == function_name), None)
        if called is None:
            skipped["name_mismatch"] += 1
            continue
        parameters = called.get("parameters") or {}
        arguments = collapse_arguments(parameters, acceptable_args)
        if arguments is None:
            skipped["schema_mismatch"] = skipped.get("schema_mismatch", 0) + 1
            continue
        question = " ".join(turn["content"] for turn in record["question"][0] if turn["role"] == "user")
        meta = {
            "tool": function_name, "bfcl_id": bfcl_id, "source": "bfcl",
            "function": called,                              # the function the ground truth calls
            "functions": functions,                          # every function the model had to choose from
            "acceptable": {name: acceptable for name, acceptable in acceptable_args.items()},
        }
        records.append({
            "id": f"{Path(questions_path).stem}:{bfcl_id}",
            "prompt": question,
            "expected": {"tool": function_name, "parameters": arguments},
            "meta": meta,
            "messages": [
                {"role": "system", "content": build_system_prompt(functions)},
                {"role": "user", "content": question},
                {"role": "assistant", "content": canonical_completion(function_name, acceptable_args, parameters)},
            ],
        })
    return records, skipped


def load_bfcl_irrelevance(questions_path, *, max_records=None):
    """Convert the BFCL irrelevance category (which ships no answer file) into records.

    The correct behaviour is to *not* call the provided function, so each record has no
    assistant turn and `meta["irrelevant"] = True`; the hallucination rate is scored
    separately in `src/bfcl_eval.run_bfcl_irrelevance`.
    """
    questions_path, max_records = Path(questions_path), None if max_records is None else int(max_records)
    records, skipped = [], {"unsupported_schema": 0}
    for record in _read_jsonl(questions_path):
        if max_records is not None and len(records) >= max_records:
            break
        functions = record.get("function") or []
        if any(not is_supported(fn.get("parameters") or {}) for fn in functions):
            skipped["unsupported_schema"] += 1
            continue
        question = " ".join(turn["content"] for turn in record["question"][0] if turn["role"] == "user")
        records.append({
            "id": f"{questions_path.stem}:{record['id']}",
            "prompt": question,
            "meta": {"source": "bfcl_irrelevance", "bfcl_id": record["id"], "function": functions[0],
                     "irrelevant": True},
            "messages": [
                {"role": "system", "content": build_system_prompt(functions)},
                {"role": "user", "content": question},
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
