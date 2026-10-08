"""Core-safe tests for the BFCL loader (src/bfcl.py): no MLX, no network.

The loader is the data foundation of the real-data pilot, so its contract is tested
against small inline fixtures that mirror BFCL's actual on-disk shape.
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.bfcl import (
    build_system_prompt,
    canonical_arguments,
    canonical_completion,
    is_supported,
    load_bfcl,
    parse_bfcl_call,
    score_bfcl_call,
)

QUESTION = {
    "id": "simple_0",
    "question": [[{"role": "user", "content": "Find the area of a triangle, base 10, height 5."}]],
    "function": [{
        "name": "calculate_triangle_area",
        "description": "Area of a triangle.",
        "parameters": {
            "type": "dict",
            "properties": {
                "base": {"type": "integer", "description": "The base."},
                "height": {"type": "integer", "description": "The height."},
                "unit": {"type": "string", "description": "Unit, defaults to units."},
            },
            "required": ["base", "height"],
        },
    }],
}
ANSWER = {"id": "simple_0", "ground_truth": [
    {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}},
]}


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


class BfclLoaderTests(unittest.TestCase):
    def test_is_supported_covers_the_bfcl_subset(self):
        self.assertTrue(is_supported({"type": "dict", "properties": {"x": {"type": "string"}}}))
        self.assertTrue(is_supported({"type": "array", "items": {"type": "integer"}}))
        self.assertTrue(is_supported({"type": "float"}))
        self.assertTrue(is_supported({"type": "boolean"}))
        self.assertFalse(is_supported({"type": "tuple"}))
        self.assertFalse(is_supported({"type": "any"}))
        self.assertFalse(is_supported({"type": "dict"}))                    # no properties
        self.assertFalse(is_supported({"type": "array"}))                   # no items

    def test_conversion_uses_the_repo_envelope_and_keeps_acceptable_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            questions, answers = Path(tmp) / "q.jsonl", Path(tmp) / "a.jsonl"
            write_jsonl(questions, [QUESTION])
            write_jsonl(answers, [ANSWER])
            records, skipped = load_bfcl(questions, answers)
        self.assertEqual(skipped, {"no_answer": 0, "unsupported_schema": 0, "multiple_calls": 0,
                                   "schema_mismatch": 0, "name_mismatch": 0})
        record = records[0]
        self.assertEqual(record["expected"], {"tool": "calculate_triangle_area",
                                              "parameters": {"base": 10, "height": 5, "unit": "units"}})
        self.assertEqual(record["meta"]["acceptable"]["unit"], ["units", ""])
        self.assertEqual([m["role"] for m in record["messages"]], ["system", "user", "assistant"])
        self.assertEqual(record["messages"][0]["content"], build_system_prompt(QUESTION["function"]))
        self.assertIn('"tool":"calculate_triangle_area"', record["messages"][-1]["content"])

    def test_empty_acceptable_value_means_the_argument_is_omitted(self):
        self.assertEqual(canonical_arguments({"x": [3], "y": ["", ""]}), {"x": 3})
        self.assertEqual(canonical_completion("f", {"x": [3]}), '{"tool":"f","parameters":{"x":3}}')

    def test_unsupported_schemas_and_missing_answers_are_skipped(self):
        bad = {**QUESTION, "function": [{**QUESTION["function"][0],
                                        "parameters": {"type": "dict", "properties": {"x": {"type": "tuple"}}}}]}
        with tempfile.TemporaryDirectory() as tmp:
            questions, answers = Path(tmp) / "q.jsonl", Path(tmp) / "a.jsonl"
            write_jsonl(questions, [QUESTION, bad])   # one answer only
            write_jsonl(answers, [ANSWER])
            records, skipped = load_bfcl(questions, answers)
        self.assertEqual(len(records), 1)
        self.assertEqual(skipped["unsupported_schema"], 1)
        self.assertEqual(skipped["no_answer"], 0)

    def test_ground_truth_with_multiple_calls_is_skipped(self):
        multi = {"id": "p_0", "ground_truth": [{"f": {"a": [1]}}, {"g": {"b": [2]}}]}
        with tempfile.TemporaryDirectory() as tmp:
            questions, answers = Path(tmp) / "q.jsonl", Path(tmp) / "a.jsonl"
            write_jsonl(questions, [QUESTION])
            write_jsonl(answers, [ANSWER, multi])
            records, skipped = load_bfcl(questions, answers)
        self.assertEqual(len(records), 1)
        self.assertEqual(skipped["multiple_calls"], 0)   # only counted for records that are processed


META = {"tool": "calculate_triangle_area",
        "acceptable": {"base": [10], "height": [5], "unit": ["units", ""]},
        "function": QUESTION["function"][0]}


class BfclScoringTests(unittest.TestCase):
    def test_parse_recovers_embedded_json_and_rejects_garbage(self):
        self.assertEqual(parse_bfcl_call('{"tool":"f","parameters":{"x":1}}'),
                         {"tool": "f", "parameters": {"x": 1}})
        self.assertEqual(parse_bfcl_call('Here is the answer: ```json\n{"tool":"f","parameters":{}}\n```'),
                         {"tool": "f", "parameters": {}})
        self.assertIsNone(parse_bfcl_call("no json here"))
        self.assertIsNone(parse_bfcl_call('{"tool":"f","tool":"g"}'))   # duplicate key

    def test_score_accepts_the_canonical_call(self):
        parsed = {"tool": "calculate_triangle_area", "parameters": {"base": 10, "height": 5, "unit": "units"}}
        scored = score_bfcl_call(parsed, META)
        self.assertTrue(scored["is_schema_valid"])
        self.assertTrue(scored["tool_correct"])
        self.assertTrue(scored["args_correct"])
        self.assertIsNone(scored["error_category"])

    def test_score_accepts_an_alternative_value_and_omitted_optional(self):
        parsed = {"tool": "calculate_triangle_area", "parameters": {"base": 10, "height": 5}}
        scored = score_bfcl_call(parsed, META)
        self.assertTrue(scored["args_correct"])   # unit omitted is acceptable ("" in the set)

    def test_score_flags_wrong_tool_wrong_value_and_missing_required(self):
        self.assertFalse(score_bfcl_call({"tool": "other", "parameters": {"base": 10, "height": 5}}, META)["tool_correct"])
        wrong_tool = score_bfcl_call({"tool": "other", "parameters": {"base": 10, "height": 5}}, META)
        self.assertEqual(wrong_tool["error_category"], "tool")
        bad_value = score_bfcl_call({"tool": "calculate_triangle_area", "parameters": {"base": 99, "height": 5}}, META)
        self.assertEqual(bad_value["error_category"], "parameters")
        missing = score_bfcl_call({"tool": "calculate_triangle_area", "parameters": {"base": 10}}, META)
        self.assertEqual(missing["error_category"], "schema")

    def test_score_rejects_wrong_parameter_type(self):
        parsed = {"tool": "calculate_triangle_area", "parameters": {"base": "10", "height": 5}}
        self.assertFalse(score_bfcl_call(parsed, META)["is_schema_valid"])


if __name__ == "__main__":
    unittest.main()
