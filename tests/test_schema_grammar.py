"""Tests for the JSON-Schema-driven grammar and processor (src/schema_grammar.py).

The grammar is validated two ways: `feed` accepts every canonical BFCL-style completion
and rejects malformed prefixes, and the *processor* round-trip (mask -> argmax -> feed)
stays in sync with it — the same invariant the tool grammar's tests pin down.
"""
import json
import unittest

import mlx.core as mx
import numpy as np

from src.bfcl import load_bfcl
from src.schema_grammar import JsonSchemaGrammar, SchemaGrammarProcessor, bfcl_envelope

FLAT = {"type": "dict", "properties": {
    "base": {"type": "integer"}, "height": {"type": "integer"},
    "unit": {"type": "string"},
}, "required": ["base", "height"]}

NESTED = {"type": "dict", "properties": {
    "conditions": {"type": "array", "items": {"type": "dict", "properties": {
        "field": {"type": "string"}, "operation": {"type": "string"}, "value": {"type": "string"}},
        "required": ["field", "operation", "value"]}},
    "cash_flows": {"type": "array", "items": {"type": "integer"}},
    "discount_rate": {"type": "float"},
    "years": {"type": "array", "items": {"type": "integer"}},
}, "required": ["conditions", "cash_flows"]}


def envelope(functions):
    return JsonSchemaGrammar(bfcl_envelope(functions))


def ok(text, functions):
    grammar = envelope(functions)
    return grammar.feed(text) and grammar.complete


class SchemaGrammarTests(unittest.TestCase):
    def test_flat_schema_accepts_and_rejects(self):
        function = {"name": "calculate_triangle_area", "parameters": FLAT}
        good = '{"tool":"calculate_triangle_area","parameters":{"base":10,"height":5,"unit":"units"}}'
        self.assertTrue(ok(good, [function]))
        for bad in ('{"tool":"calculate_triangle_area","parameters":{"base":"10","height":5}}',   # string for int
                    '{"tool":"calculate_triangle_area","parameters":{"base":10}}',                # missing required
                    '{"tool":"calculate_triangle_area","parameters":{"base":10,"height":5,"extra":1}}',  # extra key
                    '{"tool":"calculate_triangle_area","parameters":{"base":10,"height":5,}}',    # trailing comma
                    '{"tool":"calculate_triangle_area","parameters":{"base":05,"height":5}}'):    # leading zero
            self.assertFalse(ok(bad, [function]), bad)

    def test_nested_objects_arrays_floats_and_empty_arrays(self):
        function = {"name": "database.query", "parameters": NESTED}
        good = ('{"tool":"database.query","parameters":{"conditions":[{"field":"age","operation":">","value":"25"}],'
                '"cash_flows":[-50000,10000,15000],"discount_rate":1e-09,"years":[]}}')
        self.assertTrue(ok(good, [function]))
        self.assertTrue(ok('{"tool":"database.query","parameters":{"conditions":[],"cash_flows":[0.0],"discount_rate":0.08}}',
                           [function]))
        # a nested object that fails its own required field
        bad = '{"tool":"database.query","parameters":{"conditions":[{"field":"age"}],"cash_flows":[]}}'
        self.assertFalse(ok(bad, [function]))

    def test_dependent_parameters_select_the_right_sub_schema(self):
        functions = [
            {"name": "add", "parameters": {"type": "dict", "properties": {"a": {"type": "integer"},
                                                                          "b": {"type": "integer"}}, "required": ["a", "b"]}},
            {"name": "concat", "parameters": {"type": "dict", "properties": {"x": {"type": "string"},
                                                                             "y": {"type": "string"}}, "required": ["x", "y"]}},
        ]
        self.assertTrue(ok('{"tool":"add","parameters":{"a":1,"b":2}}', functions))
        self.assertTrue(ok('{"tool":"concat","parameters":{"x":"p","y":"q"}}', functions))
        self.assertFalse(ok('{"tool":"add","parameters":{"x":"p","y":"q"}}', functions))   # concat's params under add
        self.assertFalse(ok('{"tool":"add","parameters":{"a":"x","b":2}}', functions))     # string for int

    def test_scientific_notation(self):
        function = {"name": "f", "parameters": {"type": "dict", "properties": {"x": {"type": "float"}}, "required": ["x"]}}
        for text in ('{"tool":"f","parameters":{"x":1e-09}}', '{"tool":"f","parameters":{"x":0.0}}',
                     '{"tool":"f","parameters":{"x":-0.5}}', '{"tool":"f","parameters":{"x":2.5E3}}'):
            self.assertTrue(ok(text, [function]), text)
        for bad in ('{"tool":"f","parameters":{"x":1.}}', '{"tool":"f","parameters":{"x":.5}}',
                    '{"tool":"f","parameters":{"x":1e}}'):
            self.assertFalse(ok(bad, [function]), bad)


class FakeTokenizer:
    KEY_PARTS = ['{', '}', '"', ':', ',', '[', ']', '-', '_', '.']
    WORDS = ['true', 'false', 'tool', 'parameters']

    def __init__(self):
        texts = self.KEY_PARTS + [chr(code) for code in range(ord('a'), ord('z') + 1)] + \
            [str(digit) for digit in range(10)] + self.WORDS
        self.id_to_text = {index: text for index, text in enumerate(texts)}
        self.eos_token_id = len(texts)
        self.vocab_size = len(texts) + 1
        self.all_special_ids = [self.eos_token_id]

    def decode(self, tokens):
        return self.id_to_text.get(int(tokens[0]), '')

    def __len__(self):
        return self.vocab_size

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


class SchemaProcessorTests(unittest.TestCase):
    def _generate(self, grammar, steps=400):
        tokenizer = FakeTokenizer()
        processor = SchemaGrammarProcessor(tokenizer, len(tokenizer), grammar)
        text_to_id = {text: token_id for token_id, text in tokenizer.id_to_text.items()}
        rng = np.random.RandomState(0)
        logits = rng.normal(0, 0.01, len(tokenizer)).astype(np.float32)
        # Bias closing tokens so a fixed-logit model terminates, with the quote strictly
        # above the others: `,`/`}`/`]` are legal string characters, so they must not
        # out-rank the closing quote inside a string, or values fill with punctuation.
        for character, bonus in (('"', 30.0), (',', 20.0), ('}', 20.0), (']', 20.0)):
            if character in text_to_id:
                logits[text_to_id[character]] += bonus
        logits = mx.array(logits)
        prompt, generated = [1, 2, 3], []
        for _ in range(steps):
            token = int(mx.argmax(processor(mx.array(prompt + generated), logits)).item())
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
        text = ''.join(tokenizer.decode([token_id]) for token_id in generated
                       if token_id != tokenizer.eos_token_id)
        return text, processor

    def test_round_trip_stays_in_sync_and_produces_a_valid_call(self):
        grammar = envelope([{"name": "calculate_triangle_area", "parameters": FLAT}])
        text, processor = self._generate(grammar)
        self.assertTrue(processor.grammar.complete, text)
        self.assertEqual(processor.invalid_tokens, 0, text)
        parsed = json.loads(text)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(sorted(parsed), ["parameters", "tool"])   # structural, not semantic

    def test_nested_round_trip_stays_valid(self):
        grammar = envelope([{"name": "database.query", "parameters": NESTED}])
        text, processor = self._generate(grammar, steps=600)
        self.assertTrue(processor.grammar.complete, text)
        self.assertEqual(processor.invalid_tokens, 0, text)
        json.loads(text)

    def test_every_bfcl_completion_is_accepted(self):
        records, _ = load_bfcl('/tmp/bfcl_simple.json', '/tmp/bfcl_simple_ans.json')
        self.assertGreater(len(records), 300)
        for record in records:
            grammar = envelope([record['meta']['function']])
            with self.subTest(id=record['id']):
                self.assertTrue(grammar.feed(record['messages'][-1]['content']), record['id'])
                self.assertTrue(grammar.complete, record['id'])


if __name__ == '__main__':
    unittest.main()
