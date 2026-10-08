"""Tests for the token-level decision inspector (src/inspect.py).

The model is faked: `stream_generate` is replaced with crafted log-probability
vectors, so the assertions are about the trace arithmetic (probabilities, ranks,
divergence), not about a real checkpoint.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from src.inspect import (
    align_to_reference,
    build_messages,
    explain_trace,
    greedy_trace,
    reference_tokens,
    run_inspection,
)

VOCAB = 4


class FakeTokenizer:
    """Minimal stand-in: 3-message chat is 5 tokens, shorter chats are 2."""

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3, 4, 5] if len(messages) == 3 else [1, 2]

    def decode(self, tokens):
        return f"<{tokens[0]}>"


def record(target=None):
    target = target or {'tool': 'no_action', 'parameters': {'reason': 'unsupported_request'}}
    return {'id': 'test:1', 'prompt': 'request', 'expected': target, 'meta': None, 'messages': [
        {'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'request'},
        {'role': 'assistant', 'content': json.dumps(target)},
    ]}


def response(token, probabilities, finish_reason=None):
    return SimpleNamespace(
        text=f"<{token}>", token=token, logprobs=mx.log(mx.array(probabilities)),
        prompt_tokens=5, finish_reason=finish_reason,
    )


class TraceTests(unittest.TestCase):
    def test_greedy_trace_records_probability_and_ranked_alternatives(self):
        records = [
            response(3, [0.05, 0.10, 0.15, 0.70]),
            response(1, [0.10, 0.50, 0.30, 0.10], finish_reason='stop'),
        ]
        with patch('src.inspect.mlx_lm.stream_generate', return_value=iter(records)):
            trace = greedy_trace(object(), FakeTokenizer(), record()['messages'][:-1], max_tokens=10, top_k=3)
        self.assertEqual([step.token_id for step in trace['steps']], [3, 1])
        self.assertAlmostEqual(trace['steps'][0].probability, 0.70, places=6)
        self.assertEqual([alt['token_id'] for alt in trace['steps'][0].alternatives], [2, 1, 0])
        self.assertAlmostEqual(trace['steps'][0].alternatives[0]['probability'], 0.15, places=6)
        self.assertEqual(trace['raw_output'], '<3><1>')
        self.assertEqual(trace['prompt_tokens'], 5)
        self.assertEqual(trace['finish_reason'], 'stop')

    def test_reference_tokens_are_the_masked_assistant_span(self):
        self.assertEqual(reference_tokens(FakeTokenizer(), record()), [3, 4, 5])

    def test_alignment_finds_first_divergence_and_ranks_the_reference_token(self):
        records = [response(3, [0.05, 0.10, 0.15, 0.70]), response(1, [0.10, 0.50, 0.30, 0.10])]
        with patch('src.inspect.mlx_lm.stream_generate', return_value=iter(records)):
            trace = greedy_trace(object(), FakeTokenizer(), record()['messages'][:-1], top_k=3)
        alignment = align_to_reference(trace['steps'], [3, 2], FakeTokenizer())
        self.assertEqual(alignment['step'], 1)
        self.assertEqual(alignment['chose']['token_id'], 1)
        self.assertEqual(alignment['reference']['token_id'], 2)
        self.assertAlmostEqual(alignment['reference']['probability'], 0.30, places=6)
        self.assertEqual(alignment['reference']['rank'], 2)
        self.assertIn('Diverges at step 1', explain_trace(trace, alignment))

    def test_alignment_reports_match_and_overrun(self):
        records = [response(3, [0.1, 0.1, 0.1, 0.7]), response(1, [0.1, 0.6, 0.2, 0.1])]
        tokenizer = FakeTokenizer()
        with patch('src.inspect.mlx_lm.stream_generate', return_value=iter(records)):
            trace = greedy_trace(object(), tokenizer, record()['messages'][:-1], top_k=2)
        self.assertIsNone(align_to_reference(trace['steps'], [3, 1], tokenizer))
        self.assertIn('matches the reference answer token for token', explain_trace(trace, None))
        # A one-token reference is satisfied by step 0, so step 1 runs past its end.
        overrun = align_to_reference(trace['steps'], [3], tokenizer)
        self.assertEqual(overrun['step'], 1)
        self.assertIsNone(overrun['reference']['token_id'])
        self.assertIn('ran past the end', explain_trace(trace, overrun))

    def test_fewshot_variant_inserts_demonstrations(self):
        base = build_messages(record(), 'base')
        fewshot = build_messages(record(), 'fewshot', shots=2)
        self.assertEqual(len(base), 2)
        self.assertEqual(len(fewshot), 2 + 2 * 2)
        self.assertEqual(fewshot[-1]['content'], 'request')


class InspectionRunTests(unittest.TestCase):
    def test_bad_index_fails_before_loading_a_model(self):
        for index in (10 ** 6, -1):
            with self.subTest(index=index), patch('src.inspect.mlx_lm.load') as load:
                with self.assertRaises(ValueError):
                    run_inspection('test', index, output_dir='/tmp/never')
                load.assert_not_called()
        with self.assertRaises(ValueError):
            run_inspection('test', 0, variant='nope', output_dir='/tmp/never')

    def test_base_run_writes_a_trace_and_explains_the_divergence(self):
        records = [response(3, [0.05, 0.10, 0.15, 0.70]), response(1, [0.10, 0.50, 0.30, 0.10])]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.inspect.resolve_source', return_value=('base-snapshot', {'revision': 'abc'})), \
                patch('src.inspect.mlx_lm.load', return_value=(object(), FakeTokenizer())) as load, \
                patch('src.inspect.mlx_lm.stream_generate', return_value=iter(records)), \
                patch('src.inspect.reference_tokens', return_value=[3, 2]):
            result = run_inspection('test', 0, variant='base', output_dir=tmp, quiet=True)
            trace_path = Path(result['run_dir']) / 'inspect_trace.json'
            saved = json.loads(trace_path.read_text())
            manifest = json.loads((Path(result['run_dir']) / 'manifest.json').read_text())
        self.assertEqual([step['token_id'] for step in saved['steps']], [3, 1])
        self.assertEqual(saved['alignment']['step'], 1)
        self.assertIn('Diverges at step 1', saved['explanation'])
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['variant'], 'base')
        self.assertEqual(result['raw_output'], '<3><1>')
        load.assert_called_once()

    def test_lora_run_requires_and_passes_an_adapter(self):
        records = [response(3, [0.1, 0.1, 0.1, 0.7])]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.inspect.resolve_adapter_source', return_value=('base-snapshot', {'revision': 'abc'})), \
                patch('src.inspect.mlx_lm.load', return_value=(object(), FakeTokenizer())) as load, \
                patch('src.inspect.mlx_lm.stream_generate', return_value=iter(records)), \
                patch('src.inspect.reference_tokens', return_value=[3]):
            run_inspection('test', 0, variant='lora', adapter_path=tmp, output_dir=tmp, quiet=True)
        self.assertEqual(load.call_args.kwargs['adapter_path'], tmp)


if __name__ == '__main__':
    unittest.main()
