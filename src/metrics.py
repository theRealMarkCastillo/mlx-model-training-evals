"""Per-sample scoring and aggregate statistics, kept separate from model code.

A rate measured on n samples is an estimate. Every rate here comes with a 95%
Wilson score interval. With 15 samples per tool, a measured 80% could plausibly
be anywhere from about 55% to 93%, so small differences between variants are
often noise.
"""

import json
import math
from collections import Counter, defaultdict

RATE_FLAGS = {
    "pure_json_rate": "is_pure_json", "valid_json_rate": "is_valid_json",
    "schema_valid_rate": "is_schema_valid", "tool_accuracy": "tool_match",
    "exact_match_rate": "param_exact", "normalized_match_rate": "normalized_match",
}
CATEGORY_ORDER = ("json", "schema", "format", "tool", "omitted_default", "parameters")
CATEGORY_HELP = {
    "json": "no JSON object could be recovered",
    "schema": "JSON does not satisfy the tool schema (wrong type, unknown field, missing required)",
    "format": "valid call wrapped in prose or markdown fences",
    "tool": "valid call to the wrong tool",
    "omitted_default": "right values, but an optional parameter was left out instead of written with its default",
    "parameters": "right tool, at least one wrong parameter value",
}


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion; well behaved at 0% and 100%."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _canonical(value):
    # json.dumps distinguishes 1 / 1.0 / true, which Python's == does not.
    return json.dumps(value, sort_keys=True)


def score_sample(expected, normalized_expected, parsed):
    """Compare one parsed model output against its reference call."""
    actual = parsed["parsed_data"]
    tool_match = isinstance(actual, dict) and actual.get("tool") == expected["tool"]
    exact = parsed["is_pure_json"] and parsed["is_schema_valid"] and _canonical(actual) == _canonical(expected)
    normalized_match = parsed["is_schema_valid"] and _canonical(parsed["normalized_data"]) == _canonical(normalized_expected)
    params = actual.get("parameters") if tool_match else None
    fields = {
        name: isinstance(params, dict) and name in params and _canonical(params[name]) == _canonical(value)
        for name, value in expected["parameters"].items()
    }
    if not parsed["is_valid_json"]:
        category = "json"
    elif not parsed["is_schema_valid"]:
        category = "schema"
    elif not parsed["is_pure_json"]:
        category = "format"
    elif not tool_match:
        category = "tool"
    elif exact:
        category = None
    elif normalized_match:
        category = "omitted_default"
    else:
        category = "parameters"
    return {
        "tool_match": tool_match, "param_exact": exact, "normalized_match": normalized_match,
        "field_correct": fields, "wrong_fields": [name for name, ok in fields.items() if not ok],
        "error_category": category,
    }


def _rate(results, flag):
    successes = sum(bool(r[flag]) for r in results)
    low, high = wilson_interval(successes, len(results))
    return successes / len(results), [low, high]


def summarize(results):
    """Aggregate rates with intervals, plus per-tool, per-field, and error breakdowns."""
    n = len(results)
    if not n:
        raise ValueError("Cannot summarize zero results")
    summary = {"num_samples": n, "intervals": {}}
    for metric, flag in RATE_FLAGS.items():
        summary[metric], summary["intervals"][metric] = _rate(results, flag)

    by_tool = defaultdict(list)
    for r in results:
        by_tool[r["expected"]["tool"]].append(r)
    summary["per_tool"] = {}
    summary["per_field"] = {}
    for tool, group in sorted(by_tool.items()):
        rate, interval = _rate(group, "param_exact")
        summary["per_tool"][tool] = {"n": len(group), "exact_match_rate": rate, "ci95": interval,
                                     "tool_accuracy": _rate(group, "tool_match")[0]}
        summary["per_field"][tool] = {
            name: sum(r["field_correct"][name] for r in group) / len(group)
            for name in group[0]["field_correct"]
        }

    counts = Counter(r["error_category"] for r in results if r["error_category"])
    summary["error_categories"] = {c: counts.get(c, 0) for c in CATEGORY_ORDER}

    # Slices that depend on how the request was written, when the data records it.
    slices = {
        "omitted_optional": [r for r in results if (r.get("meta") or {}).get("omitted")],
        "all_explicit": [r for r in results if r.get("meta") is not None and not r["meta"].get("omitted")],
    }
    summary["slices"] = {
        name: {"n": len(group), "exact_match_rate": _rate(group, "param_exact")[0], "ci95": _rate(group, "param_exact")[1]}
        for name, group in slices.items() if group
    }
    return summary


def failure_examples(results, limit=5):
    """Up to `limit` failures, cycling through error categories for variety."""
    by_category = defaultdict(list)
    for r in results:
        if r["error_category"]:
            by_category[r["error_category"]].append(r)
    picked = []
    while len(picked) < limit and any(by_category.values()):
        for category in CATEGORY_ORDER:
            if by_category.get(category) and len(picked) < limit:
                r = by_category[category].pop(0)
                picked.append({
                    "id": r["id"], "category": category, "prompt": r["prompt"],
                    "expected": r["expected"], "raw_output": r["raw_output"],
                    "wrong_fields": r["wrong_fields"], "error": r.get("error"),
                })
    return picked


def format_rate(rate, interval):
    return f"{100 * rate:5.1f}% [{100 * interval[0]:.0f}–{100 * interval[1]:.0f}]"


def paired_comparison(results_a, results_b, flag="param_exact"):
    """Compare two variants on the same samples (McNemar's exact test).

    Samples both variants get right (or both get wrong) say nothing about which
    is better; only the discordant pairs do. The p-value is the chance of a
    split at least this lopsided if the two variants were equally good.
    """
    pairs = list(zip(results_a, results_b, strict=False))
    if not pairs or any(a["id"] != b["id"] for a, b in pairs) or len(results_a) != len(results_b):
        raise ValueError("Paired comparison needs results for the same samples in the same order")
    only_a = sum(bool(a[flag]) and not b[flag] for a, b in pairs)
    only_b = sum(bool(b[flag]) and not a[flag] for a, b in pairs)
    both = sum(bool(a[flag]) and bool(b[flag]) for a, b in pairs)
    n = only_a + only_b
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n if n else 1.0
    return {"both_correct": both, "only_a": only_a, "only_b": only_b,
            "neither": len(pairs) - both - n, "p_value": min(1.0, 2 * tail)}


def sign_test(deltas):
    """Two-sided exact sign test for paired continuous measurements.

    Used by the forgetting check: for each general-capability record we compare the
    assistant loss before and after fine-tuning, and ask whether the losses went up
    more often than down by a margin that sampling noise cannot explain. Records with
    an unchanged loss are ignored, exactly like ties in a Wilcoxon test.
    """
    increased = sum(1 for delta in deltas if delta > 0)
    decreased = sum(1 for delta in deltas if delta < 0)
    n = increased + decreased
    if not n:
        return {"increased": 0, "decreased": 0, "unchanged": len(deltas), "p_value": 1.0}
    tail = sum(math.comb(n, k) for k in range(min(increased, decreased) + 1)) / 2 ** n
    return {"increased": increased, "decreased": decreased, "unchanged": len(deltas) - n,
            "p_value": min(1.0, 2 * tail)}


def sweep_summary(values):
    """Mean ± spread across the points of a sweep (for example, one adapter per seed).

    A single-seed number is one draw from an unknown distribution; when a sweep exists,
    this is the honest way to quote its headline: mean with the population standard
    deviation and the range, so the reader can see whether the points agree.
    """
    values = list(values)
    if not values:
        raise ValueError("Cannot summarize an empty sweep")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"n": len(values), "mean": mean, "stdev": math.sqrt(variance),
            "min": min(values), "max": max(values)}
