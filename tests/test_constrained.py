"""Tests for grammar-constrained decoding: the prefix validator, the token mask, and the loop.

The model is faked with a tiny character-level vocabulary and uniform logits, so the
assertions are about the grammar and the mask — not about a checkpoint. The end-to-end
test still goes through the real mask → argmax → grammar cycle, which is what catches
desynchronization bugs.
"""
import json
import unittest

import mlx.core as mx
import numpy as np

from src.constrained import GrammarProcessor, ToolCallGrammar, required_fields, tokenizer_vocab_size
from src.dataset import DATA_DIR, load_samples
from src.schema import PARAM_MODEL_MAP, parse_and_validate

KEY_PARTS = ['{', '}', '"', ':', ',', '[', ']', '-', '_', '.']
ALPHABET = [chr(code) for code in range(ord('a'), ord('z') + 1)]
DIGITS = [str(digit) for digit in range(10)]
WORDS = ['true', 'false', 'tool', 'parameters']
EOS_TEXT = ''


class FakeTokenizer:
    """Character-level tokenizer with one special token, enough to spell any JSON value."""

    def __init__(self):
        texts = KEY_PARTS + ALPHABET + DIGITS + WORDS
        self.id_to_text = {index: text for index, text in enumerate(texts)}
        self.eos_token_id = len(texts)
        self.id_to_text[self.eos_token_id] = EOS_TEXT
        self.vocab_size = len(self.id_to_text)
        self.all_special_ids = [self.eos_token_id]

    def decode(self, tokens):
        return self.id_to_text.get(int(tokens[0]), '')

    def __len__(self):
        return self.vocab_size

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


def complete(text, grammar=None):
    grammar = grammar or ToolCallGrammar()
    return grammar.feed(text) and grammar.complete


class GrammarTests(unittest.TestCase):
    def test_every_reference_completion_is_accepted_and_complete(self):
        grammar = ToolCallGrammar()
        records = []
        for name in ('test', 'challenge_defaults', 'challenge_abstain'):
            records += load_samples(DATA_DIR / f'{name}.jsonl')
        for record in records:
            grammar.reset()
            text = json.dumps(record['expected'], separators=(',', ':'))
            self.assertTrue(grammar.feed(text), text)
            self.assertTrue(grammar.complete, text)

    def test_valid_prefixes_are_accepted_without_being_complete(self):
        for text in ('{"tool": "deploy_service"'.replace(' ', ''),
                     '{"tool":"deploy_service","parameters":{"service":"auth',
                     '{"tool":"scale_cluster","parameters":{"cluster_name":"c","node_count":1',
                     '{"tool":"deploy_service","parameters":{"notify_channels":["a"'): 
            grammar = ToolCallGrammar()
            self.assertTrue(grammar.feed(text), text)
            self.assertFalse(grammar.complete, text)

    def test_invalid_text_is_rejected(self):
        cases = {
            'prose': 'Here is the call: {"tool"',
            'unknown tool': '{"tool":"delete_everything"',
            'wrong type (string for int)': '{"tool":"scale_cluster","parameters":{"node_count":"3"',
            'out of range': '{"tool":"scale_cluster","parameters":{"cluster_name":"c","node_count":501}}',
            'below minimum': '{"tool":"deploy_service","parameters":{"service":"a","version":"v","environment":"production","replicas":0}}',
            'trailing comma': '{"tool":"deploy_service","parameters":{"service":"a","version":"v","environment":"production",}}',
            'duplicate key': '{"tool":"deploy_service","tool":"no_action"',
            'parameters before tool': '{"parameters":{}',
            'missing required field': '{"tool":"restart_pod","parameters":{"region":"r","reason":"x"}}',
            'bad enum': '{"tool":"no_action","parameters":{"reason":"because"}}',
            'chatter after the object': '{"tool":"no_action","parameters":{"reason":"unsupported_request"}} and then',
            'trailing comma in array': '{"tool":"deploy_service","parameters":{"service":"a","version":"v","environment":"production","notify_channels":["x",]}}',
        }
        for label, text in cases.items():
            with self.subTest(label=label):
                self.assertFalse(complete(text), label)

    def test_complete_calls_from_every_tool(self):
        samples = {
            'deploy_service': '{"tool":"deploy_service","parameters":{"service":"a","version":"v1","environment":"production","replicas":3,"notify_channels":["#x"]}}',
            'restart_pod': '{"tool":"restart_pod","parameters":{"pod_name":"p","region":"r","force":true,"reason":"oom"}}',
            'rollback_deployment': '{"tool":"rollback_deployment","parameters":{"deployment_id":"dep-1","target_tag":"v1","drain_traffic":false}}',
            'scale_cluster': '{"tool":"scale_cluster","parameters":{"cluster_name":"c","node_count":5,"auto_scale":true,"instance_type":"m6i.large"}}',
            'no_action': '{"tool":"no_action","parameters":{"reason":"missing_required_parameter"}}',
        }
        self.assertEqual(set(samples), set(PARAM_MODEL_MAP))
        for tool, text in samples.items():
            with self.subTest(tool=tool):
                self.assertTrue(complete(text))
                self.assertTrue(parse_and_validate(text)['is_schema_valid'])

    def test_required_fields_come_from_the_pydantic_models(self):
        self.assertEqual(required_fields('no_action'), ('reason',))
        self.assertIn('service', required_fields('deploy_service'))
        self.assertNotIn('replicas', required_fields('deploy_service'))


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.vocab = len(self.tokenizer)

    def processor(self, **kwargs):
        return GrammarProcessor(self.tokenizer, self.vocab, **kwargs)

    def logits(self):
        return mx.array(np.zeros(self.vocab, dtype=np.float32))

    def allowed_texts(self, processor):
        return sorted(processor.tokenizer.decode([token_id]) for token_id in processor.allowed_token_ids())

    def test_tokenizer_vocab_size_includes_special_tokens(self):
        class Small:
            vocab_size = 10
            all_special_ids = [12]
            def __len__(self): return 11
        self.assertEqual(tokenizer_vocab_size(Small()), 13)

    def test_first_step_offers_only_an_object_start(self):
        processor = self.processor()
        self.assertEqual(self.allowed_texts(processor), ['{'])

    def test_mask_size_follows_the_model_logits(self):
        processor = self.processor()
        logits = mx.array(np.zeros(self.vocab + 7, dtype=np.float32))
        masked = processor(mx.array([1]), logits)
        self.assertEqual(masked.shape[-1], self.vocab + 7)
        self.assertEqual(processor.mask_size, self.vocab + 7)

    def test_key_mask_requires_tool_before_parameters(self):
        processor = self.processor()
        processor.grammar.feed('{"')
        texts = self.allowed_texts(processor)
        self.assertIn('t', texts)          # "tool" can start
        self.assertNotIn('p', texts)       # "parameters" cannot come first
        processor.grammar.feed('tool":"deploy_service",')
        self.assertEqual(self.allowed_texts(processor), ['"'])   # a key must start here
        processor.grammar.feed('"')
        texts = self.allowed_texts(processor)
        self.assertTrue(texts, 'a key prefix must remain available')
        self.assertTrue(all('parameters'.startswith(text) for text in texts), texts)
        self.assertIn('p', texts)

    def test_completion_masks_everything_but_eos(self):
        processor = self.processor()
        processor.grammar.feed('{"tool":"no_action","parameters":{"reason":"unsupported_request"}}')
        self.assertTrue(processor.grammar.complete)
        self.assertEqual(processor.allowed_token_ids(), [self.tokenizer.eos_token_id])
        masked = np.array(processor(mx.array([1]), self.logits()))
        self.assertEqual(list(np.flatnonzero(np.isfinite(masked))), [self.tokenizer.eos_token_id])

    def test_special_tokens_never_desync_the_grammar(self):
        processor = self.processor()
        processor.grammar.feed('{"tool":"no_action","parameters":{"reason":"unsupported_request"}}')
        processor.advance([self.tokenizer.eos_token_id])
        self.assertEqual(processor.invalid_tokens, 0)
        self.assertTrue(processor.grammar.complete)

    def test_uniform_logits_still_produce_a_valid_call(self):
        """The whole loop: mask -> argmax -> feed the grammar -> mask, exactly as MLX-LM drives it."""
        processor = self.processor()
        logits = self.logits()
        generated = []
        for _ in range(600):
            token = int(mx.argmax(processor(mx.array([1] + generated), logits)).item())
            generated.append(token)
            if token == self.tokenizer.eos_token_id:
                break
        text = ''.join(self.tokenizer.decode([token_id]) for token_id in generated
                       if token_id != self.tokenizer.eos_token_id)
        self.assertTrue(processor.grammar.complete, text)
        self.assertEqual(processor.invalid_tokens, 0, text)
        parsed = parse_and_validate(text)
        self.assertTrue(parsed['is_pure_json'], text)
        self.assertTrue(parsed['is_schema_valid'], text)

    def test_string_cap_closes_runaway_strings(self):
        processor = self.processor(max_string_tokens=3)
        processor.grammar.feed('{"tool":"deploy_service","parameters":{"service":"')
        for _ in range(3):
            processor(mx.array([1]), self.logits())
        self.assertEqual(self.allowed_texts(processor), ['"'])


if __name__ == '__main__':
    unittest.main()
