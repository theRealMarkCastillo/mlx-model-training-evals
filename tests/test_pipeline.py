"""Regression tests for data validity, scoring, provenance, and real MLX training."""
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import yaml
from pydantic import ValidationError

from data.prepare_dataset import build_splits, to_chat_format, CHANNELS
from src import benchmark, evaluate, train, inference, fuse
from src.dataset import load_samples, validate_splits
from src.runs import file_identity, latest_path, resolve_adapter_source
from src.schema import ToolCall, parse_and_validate


TARGET = {'tool': 'deploy_service', 'parameters': {
    'service': 'api', 'version': 'v1', 'environment': 'production',
    'replicas': 3, 'notify_channels': [],
}}


def sample(target=None):
    target = target or TARGET
    return {'id': 'test:1', 'prompt': 'request', 'expected': target, 'messages': [
        {'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'request'},
        {'role': 'assistant', 'content': json.dumps(target)},
    ]}


def generation(text):
    return {'raw_output': text, 'output_tokens': 2, 'latency_seconds': 0.1, 'finish_reason': 'stop'}


class SchemaTests(unittest.TestCase):
    def test_malformed_envelopes_never_crash(self):
        for value in ([], {}, 1, True, None, 'wrong'):
            result = parse_and_validate(json.dumps({'tool': value, 'parameters': {}}))
            self.assertFalse(result['is_schema_valid'])
        for text in ('[]', 'null', '1', 'nonsense', '{', '```json\n{}\n```'):
            self.assertFalse(parse_and_validate(text)['is_schema_valid'])

    def test_strict_types_and_extras(self):
        for value in ('3', True, 3.0):
            payload = copy.deepcopy(TARGET)
            payload['parameters']['replicas'] = value
            self.assertFalse(parse_and_validate(json.dumps(payload))['is_schema_valid'])
        for location in ('top', 'parameters'):
            payload = copy.deepcopy(TARGET)
            (payload if location == 'top' else payload['parameters'])['invented'] = 1
            self.assertFalse(parse_and_validate(json.dumps(payload))['is_schema_valid'])

    def test_discriminator_rejects_wrong_parameters(self):
        with self.assertRaises(ValidationError):
            ToolCall.model_validate({'tool': 'deploy_service', 'parameters': {'deployment_id': 'dep1', 'target_tag': 'v1'}})

    def test_raw_data_preserved_and_defaults_separate(self):
        payload = copy.deepcopy(TARGET)
        del payload['parameters']['notify_channels']
        result = parse_and_validate(json.dumps(payload))
        self.assertTrue(result['is_schema_valid'])
        self.assertEqual(result['parsed_data'], payload)
        self.assertEqual(result['normalized_data'], TARGET)

    def test_wrappers_duplicates_constants_and_literal_backticks(self):
        text = json.dumps(TARGET)
        result = parse_and_validate('Here is the call:\n```json\n' + text + '\n```')
        self.assertTrue(result['is_schema_valid'])
        self.assertFalse(result['is_pure_json'])
        for bad in ('{"tool":"a","tool":"b"}', '{"tool":NaN}', '{"tool":Infinity}', '{"tool":1e999}'):
            self.assertFalse(parse_and_validate(bad)['is_valid_json'])
        payload = {'tool': 'restart_pod', 'parameters': {'pod_name': 'p', 'region': 'r', 'reason': 'literal ```', 'force': False}}
        self.assertTrue(parse_and_validate(json.dumps(payload))['is_pure_json'])


class DatasetTests(unittest.TestCase):
    def test_regenerated_data_is_reproducible_disjoint_and_grounded(self):
        splits = build_splits()
        self.assertEqual(splits, build_splits())
        prompts = set()
        for name, records in splits.items():
            stored = [json.loads(line) for line in Path(f'data/{name}.jsonl').read_text().splitlines()]
            self.assertEqual(stored, to_chat_format(records))
            for record in records:
                self.assertNotIn(record['prompt'], prompts)
                prompts.add(record['prompt'])
                target = ToolCall.model_validate_json(record['completion']).model_dump()
                params = target['parameters']
                if target['tool'] == 'deploy_service':
                    self.assertEqual(set(params['notify_channels']), {ch for ch in CHANNELS if ch in record['prompt']})
                if target['tool'] == 'restart_pod':
                    self.assertIn(params['reason'], record['prompt'])
        self.assertEqual(json.loads(Path('data/raw_test_samples.json').read_text()), splits['test'])
        validate_splits('data')

    def test_empty_invalid_and_nonpositive_inputs(self):
        for count in (0, -1):
            with self.assertRaises(ValueError):
                load_samples('data/test.jsonl', count)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.jsonl'
            path.write_text('')
            with self.assertRaises(ValueError):
                load_samples(path)


class EvaluationTests(unittest.TestCase):
    def test_exact_match_does_not_repair_types_extras_defaults_or_wrappers(self):
        missing = copy.deepcopy(TARGET)
        del missing['parameters']['notify_channels']
        extra = copy.deepcopy(TARGET)
        extra['parameters']['invented'] = 'bad'
        wrong_type = copy.deepcopy(TARGET)
        wrong_type['parameters']['replicas'] = '3'
        outputs = [json.dumps(TARGET), json.dumps(missing), json.dumps(extra), json.dumps(wrong_type), '```json\n'+json.dumps(TARGET)+'\n```', '{"tool":[]}']
        with patch.object(evaluate, 'generate_response', side_effect=[generation(s) for s in outputs]):
            metrics = evaluate.run_deterministic_eval(object(), object(), [sample() for _ in outputs])
        self.assertAlmostEqual(metrics['exact_match_rate'], 1/6)
        self.assertAlmostEqual(metrics['schema_valid_rate'], 3/6)
        self.assertAlmostEqual(metrics['normalized_match_rate'], 3/6)
        self.assertEqual(metrics['sample_results'][1]['parsed_data'], missing)
        self.assertEqual(metrics['sample_results'][0]['expected'], TARGET)

    def test_loss_excludes_prompt_and_includes_first_assistant_token(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [0, 0, 0, 1, 1] if len(messages) == 3 else [0, 0, 0]
        class Model:
            def eval(self): pass
            def __call__(self, inputs):
                # Prompt predictions deliberately wrong; assistant predictions uniform.
                return mx.array([[[0., 10.], [0., 10.], [0., 0.], [0., 0.]]])
        metrics = evaluate.compute_perplexity(Model(), Tokenizer(), [sample()])
        self.assertAlmostEqual(metrics['loss'], math.log(2), places=6)
        self.assertEqual(metrics['loss_tokens'], 2)
        self.assertAlmostEqual(metrics['perplexity'], 2, places=6)

    def test_complete_reports_and_fused_evaluation_use_same_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / 'adapter'
            adapter.mkdir()
            (adapter / 'adapters.safetensors').write_bytes(b'test')
            (adapter / 'adapter_config.json').write_text('{}')
            metrics = {'pure_json_rate': 1., 'schema_valid_rate': 1., 'tool_accuracy': 1., 'exact_match_rate': 1., 'normalized_match_rate': 1., 'sample_results': [generation('{}') for _ in range(12)]}
            with patch.object(evaluate, 'resolve_adapter_source', return_value=('base-snapshot', {'revision': 'abc'})), patch.object(evaluate, 'resolve_source', return_value=('fused-snapshot', {'revision': 'def'})), patch.object(evaluate.mlx_lm, 'load', return_value=(object(), object())) as load, patch.object(evaluate, 'compute_perplexity', return_value={'loss': 1., 'perplexity': math.e}) as loss, patch.object(evaluate, 'run_deterministic_eval', return_value=metrics) as deterministic:
                report = evaluate.run_comprehensive_evaluation(adapter_path=str(adapter), output_dir=tmp, num_eval_samples=12, fused_path='fused')
            saved = json.loads((Path(report['run_dir']) / 'eval_results.json').read_text())
            self.assertEqual(len(saved['lora_metrics']['sample_results']), 12)
            self.assertIn('fused_metrics', saved)
            self.assertEqual(load.call_count, 3)
            self.assertIs(loss.call_args_list[0].args[2], deterministic.call_args_list[2].args[2])
            self.assertEqual(saved['manifest']['dataset']['sha256'], file_identity('data/test.jsonl')['sha256'])
            self.assertEqual(saved['manifest']['status'], 'complete')

    def test_invalid_counts_fail_before_model_load(self):
        with patch.object(evaluate.mlx_lm, 'load') as load:
            for count in (0, -1):
                with self.assertRaises(ValueError):
                    evaluate.run_comprehensive_evaluation(num_eval_samples=count)
            load.assert_not_called()


class StreamingTests(unittest.TestCase):
    def test_stream_metadata_and_chat_template(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return [10, 11, 12]
        tokenizer = Tokenizer()
        records = [SimpleNamespace(text=t, generation_tokens=n, prompt_tokens=3, prompt_tps=30., generation_tps=20., finish_reason=f) for t,n,f in [('a',1,None),('b',2,'stop')]]
        with patch.object(inference.mlx_lm, 'stream_generate', return_value=iter(records)) as stream, patch.object(inference.time, 'perf_counter', side_effect=[10., 10.1, 10.4]):
            result = inference.generate_response(object(), tokenizer, sample()['messages'][:-1])
        self.assertEqual(result['raw_output'], 'ab')
        self.assertEqual(result['output_tokens'], 2)
        self.assertAlmostEqual(result['ttft_seconds'], .1)
        self.assertAlmostEqual(result['latency_seconds'], .4)
        self.assertEqual(result['decode_tokens_per_sec'], 20.)
        self.assertEqual(stream.call_args.kwargs['prompt'], [10, 11, 12])
        self.assertEqual(tokenizer.messages, sample()['messages'][:-1])

    def test_benchmark_uses_system_prompt_and_separate_metrics(self):
        stats = {'ttft_seconds': .1, 'latency_seconds': 1., 'prefill_tokens_per_sec': 100., 'decode_tokens_per_sec': 20., 'end_to_end_tokens_per_sec': 10., 'output_tokens': 10, 'prompt_tokens': 100, 'raw_output': '{}'}
        with patch.object(benchmark, 'generate_response', return_value=stats) as generate:
            report = benchmark.benchmark_generation(object(), object(), 'request', runs=2, warmup=1)
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(generate.call_args.args[2][0]['role'], 'system')
        self.assertEqual(report['avg_decode_tokens_per_sec'], 20.)
        self.assertEqual(report['avg_end_to_end_tokens_per_sec'], 10.)
        self.assertEqual(len(report['runs']), 2)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
    def __call__(self, x): return self.proj(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 8)
        self.layers = [TinyBlock()]
    def __call__(self, x): return self.layers[0](self.embed(x))


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3, 4, 5] if len(messages) == 3 else [1, 2]


class TrainingTests(unittest.TestCase):
    def test_missing_and_invalid_config_fail_before_loading(self):
        with patch.object(train, 'resolve_source') as source:
            with self.assertRaises(FileNotFoundError):
                train.run_training(config_path='/no/such/config.yaml')
            for body in ('null', '[]', 'model: wrong', 'model: x\ndata: data\nadapter_path: a\niters: 0'):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'config.yaml'
                    path.write_text(body)
                    with self.assertRaises(ValueError):
                        train.run_training(config_path=str(path))
            source.assert_not_called()

    def test_real_training_callback_weights_and_repeat_run_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / 'base'
            model_path.mkdir()
            cfg = yaml.safe_load(Path('config/lora_config.yaml').read_text())
            cfg.update(model=str(model_path), adapter_path=str(root/'adapters'), batch_size=1, iters=1, num_layers=1, val_batches=1, steps_per_eval=1, steps_per_report=1, save_every=1, max_seq_length=16)
            cfg['lora_parameters'] = {'rank': 2, 'scale': 2., 'dropout': 0.}
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump(cfg))
            callbacks = []
            for _ in range(2):
                with patch.object(train.lora, 'load', return_value=(TinyModel(), TinyTokenizer())):
                    callbacks.append(train.run_training(config_path=str(config), output_dir=tmp))
            first, second = callbacks
            self.assertNotEqual(first.run_dir, second.run_dir)
            for callback in callbacks:
                self.assertTrue(callback.train_losses)
                self.assertTrue(callback.val_losses)
                self.assertGreater(next(r['tok_per_sec'] for r in callback.history if r['type']=='train'), 0)
                self.assertTrue((Path(callback.adapter_path)/'adapters.safetensors').is_file())
                self.assertTrue((callback.run_dir/'loss_curve.png').is_file())
            self.assertEqual(latest_path(root/'adapters'), second.adapter_path)
            source, identity = resolve_adapter_source(str(model_path), second.adapter_path)
            self.assertEqual(source, str(model_path))
            with self.assertRaises(ValueError):
                resolve_adapter_source('different-model', second.adapter_path)
            manifest = json.loads((second.run_dir/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['datasets']), 3)
            (model_path / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed after training'):
                resolve_adapter_source(str(model_path), second.adapter_path)


if __name__ == '__main__':
    unittest.main()
