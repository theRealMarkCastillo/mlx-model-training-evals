"""MLX-dependent tests: evaluation plumbing, streaming/benchmark metrics, loss masking, real training.

Pure-Python tests live in tests/test_core.py so they can run on Linux CI.
"""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import yaml
from mlx_lm.tuner.utils import linear_to_lora_layers

from src import benchmark, evaluate, forgetting, inference, train
from src.dataset import DATA_DIR
from src.explain import loss_mask_tokens, mask_counts, parameter_summary
from src.metrics import score_sample, summarize
from src.runs import file_identity, latest_path, resolve_adapter_source
from src.schema import parse_and_validate

TARGET = {'tool': 'deploy_service', 'parameters': {
    'service': 'api', 'version': 'v1', 'environment': 'production',
    'replicas': 3, 'notify_channels': [],
}}


def sample(target=None, meta=None):
    target = target or TARGET
    return {'id': 'test:1', 'prompt': 'request', 'expected': target, 'meta': meta, 'messages': [
        {'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'request'},
        {'role': 'assistant', 'content': json.dumps(target)},
    ]}


def generation(text):
    return {'raw_output': text, 'output_tokens': 2, 'latency_seconds': 0.1, 'finish_reason': 'stop'}


class EvaluationTests(unittest.TestCase):
    def test_exact_match_does_not_repair_types_extras_defaults_or_wrappers(self):
        missing = copy.deepcopy(TARGET)
        del missing['parameters']['notify_channels']
        extra = copy.deepcopy(TARGET)
        extra['parameters']['invented'] = 'bad'
        wrong_type = copy.deepcopy(TARGET)
        wrong_type['parameters']['replicas'] = '3'
        outputs = [json.dumps(TARGET), json.dumps(missing), json.dumps(extra), json.dumps(wrong_type), '```json\n' + json.dumps(TARGET) + '\n```', '{"tool":[]}']
        with patch.object(evaluate, 'generate_response', side_effect=[generation(s) for s in outputs]):
            metrics = evaluate.run_deterministic_eval(object(), object(), [sample() for _ in outputs])
        self.assertAlmostEqual(metrics['exact_match_rate'], 1 / 6)
        self.assertAlmostEqual(metrics['schema_valid_rate'], 3 / 6)
        self.assertAlmostEqual(metrics['normalized_match_rate'], 3 / 6)
        self.assertLess(metrics['intervals']['exact_match_rate'][0], 1 / 6)
        self.assertEqual(metrics['sample_results'][1]['parsed_data'], missing)
        self.assertEqual(metrics['sample_results'][1]['error_category'], 'omitted_default')

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

    def test_variants_share_samples_and_fewshot_adds_demonstrations(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / 'adapter'
            adapter.mkdir()
            (adapter / 'adapters.safetensors').write_bytes(b'test')
            (adapter / 'adapter_config.json').write_text('{}')
            seen_messages = []

            def fake_eval(model, tokenizer, samples, max_tokens, desc, generate=None, temperature=0.0):
                seen_messages.append((desc, len(samples[0]['messages'])))
                results = []
                for s in samples:
                    parsed = parse_and_validate(s['messages'][-1]['content'])
                    results.append({'id': s['id'], 'prompt': s['prompt'], 'expected': s['expected'], 'meta': s['meta'],
                                    'raw_output': '', 'output_tokens': 1, **parsed,
                                    **score_sample(s['expected'], s['normalized_expected'], parsed)})
                return {**summarize(results), 'sample_results': results, 'avg_output_tokens': 1}

            with patch.object(evaluate, 'resolve_adapter_source', return_value=('base-snapshot', {'revision': 'abc'})), \
                    patch.object(evaluate, 'resolve_source', return_value=('fused-snapshot', {'revision': 'def'})), \
                    patch.object(evaluate.mlx_lm, 'load', return_value=(object(), object())) as load, \
                    patch.object(evaluate, 'compute_perplexity', return_value={'loss': 1., 'perplexity': math.e}), \
                    patch.object(evaluate, 'run_deterministic_eval', side_effect=fake_eval):
                report = evaluate.run_comprehensive_evaluation(adapter_path=str(adapter), output_dir=tmp, num_eval_samples=10,
                                                               fused_path='fused', challenge=True, quiet=True)
            saved = json.loads((Path(report['run_dir']) / 'eval_results.json').read_text())
            self.assertEqual(saved['variants'], ['base', 'fewshot', 'lora', 'fused'])
            self.assertEqual(set(saved['datasets']), {'test', 'challenge_entities', 'challenge_defaults', 'challenge_abstain'})
            self.assertEqual(len(saved['datasets']['test']['lora']['sample_results']), 10)
            self.assertEqual(load.call_count, 4)
            lengths = dict(seen_messages)
            self.assertEqual(lengths['base/test'], 3)
            self.assertEqual(lengths['fewshot/test'], 3 + 2 * 5)
            self.assertEqual(saved['manifest']['datasets']['test']['sha256'], file_identity(DATA_DIR / 'test.jsonl')['sha256'])
            self.assertEqual(len(saved['manifest']['fewshot_ids']), 5)
            self.assertTrue((Path(report['run_dir']) / 'challenge_comparison.png').is_file())

    def test_baselines_need_no_adapter(self):
        with patch.object(evaluate, 'resolve_source', return_value=('base', {'revision': 'r'})), \
                patch.object(evaluate, 'resolve_model_paths', wraps=evaluate.resolve_model_paths) as resolve, \
                patch.object(evaluate.mlx_lm, 'load', side_effect=RuntimeError('stop')), tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                evaluate.run_comprehensive_evaluation(variants=('base',), output_dir=tmp)
            self.assertFalse(resolve.call_args.kwargs['need_adapter'])
            manifest = json.loads(next(Path(tmp).glob('runs/*/manifest.json')).read_text())
            self.assertEqual(manifest['status'], 'failed')

    def test_invalid_counts_fail_before_model_load(self):
        with patch.object(evaluate.mlx_lm, 'load') as load:
            for count in (0, -1):
                with self.assertRaises(ValueError):
                    evaluate.run_comprehensive_evaluation(num_eval_samples=count)
            with self.assertRaises(ValueError):
                evaluate.run_comprehensive_evaluation(variants=('nope',))
            load.assert_not_called()

    def test_constrained_flag_adds_a_grammar_variant_and_only_it_is_constrained(self):
        used_generate = []

        def fake_eval(model, tokenizer, samples, max_tokens, desc, generate=None, temperature=0.0):
            used_generate.append((desc.split('/')[0], generate is not None))
            results = []
            for s in samples:
                parsed = parse_and_validate(s['messages'][-1]['content'])
                results.append({'id': s['id'], 'prompt': s['prompt'], 'expected': s['expected'], 'meta': s['meta'],
                                'raw_output': '', 'output_tokens': 1, **parsed,
                                **score_sample(s['expected'], s['normalized_expected'], parsed)})
            return {**summarize(results), 'sample_results': results, 'avg_output_tokens': 1}

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(evaluate, 'resolve_source', return_value=('base', {'revision': 'r'})), \
                patch.object(evaluate.mlx_lm, 'load', return_value=(object(), object())), \
                patch.object(evaluate, 'compute_perplexity', return_value={'loss': 1., 'perplexity': math.e}), \
                patch.object(evaluate, 'run_deterministic_eval', side_effect=fake_eval):
            report = evaluate.run_comprehensive_evaluation(variants=('base',), output_dir=tmp,
                                                           num_eval_samples=5, constrained=True, quiet=True)
        self.assertEqual(report['variants'], ['base', 'grammar'])
        self.assertTrue(report['constrained'])
        constrained_variants = {variant for variant, is_constrained in used_generate if is_constrained}
        unconstrained_variants = {variant for variant, is_constrained in used_generate if not is_constrained}
        self.assertEqual(constrained_variants, {'grammar'})
        self.assertEqual(unconstrained_variants, {'base'})


class StreamingTests(unittest.TestCase):
    def test_stream_metadata_and_chat_template(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return [10, 11, 12]
        tokenizer = Tokenizer()
        records = [SimpleNamespace(text=t, generation_tokens=n, prompt_tokens=3, prompt_tps=30., generation_tps=20., finish_reason=f)
                   for t, n, f in [('a', 1, None), ('b', 2, 'stop')]]
        with patch.object(inference.mlx_lm, 'stream_generate', return_value=iter(records)) as stream, \
                patch.object(inference.time, 'perf_counter', side_effect=[10., 10.1, 10.4]):
            result = inference.generate_response(object(), tokenizer, sample()['messages'][:-1])
        self.assertEqual(result['raw_output'], 'ab')
        self.assertEqual(result['output_tokens'], 2)
        self.assertAlmostEqual(result['ttft_seconds'], .1)
        self.assertAlmostEqual(result['latency_seconds'], .4)
        self.assertEqual(result['decode_tokens_per_sec'], 20.)
        self.assertEqual(stream.call_args.kwargs['prompt'], [10, 11, 12])
        self.assertEqual(tokenizer.messages, sample()['messages'][:-1])

    def test_benchmark_uses_system_prompt_and_separate_metrics(self):
        stats = {'ttft_seconds': .1, 'latency_seconds': 1., 'prefill_tokens_per_sec': 100., 'decode_tokens_per_sec': 20.,
                 'end_to_end_tokens_per_sec': 10., 'output_tokens': 10, 'prompt_tokens': 100, 'raw_output': '{}'}
        with patch.object(benchmark, 'generate_response', return_value=stats) as generate:
            report = benchmark.benchmark_generation(object(), object(), 'request', runs=2, warmup=1)
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(generate.call_args.args[2][0]['role'], 'system')
        self.assertEqual(report['avg_decode_tokens_per_sec'], 20.)
        self.assertEqual(len(report['runs']), 2)
        # Spread is reported, not just the mean: Macs thermal-throttle mid-benchmark.
        for key in ('decode_tokens_per_sec', 'ttft_seconds', 'latency_seconds'):
            self.assertEqual(report[f'min_{key}'], report[f'avg_{key}'])
            self.assertEqual(report[f'max_{key}'], report[f'avg_{key}'])
            self.assertEqual(report[f'std_{key}'], 0.0)


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

    def decode(self, tokens):
        return f'<{tokens[0]}>'


class ExplainTests(unittest.TestCase):
    def test_loss_mask_matches_training_offsets(self):
        pairs = loss_mask_tokens(TinyTokenizer(), sample())
        self.assertEqual([scored for _, scored in pairs], [False, False, True, True, True])
        self.assertEqual(mask_counts(pairs), {'total_tokens': 5, 'scored_tokens': 3, 'masked_tokens': 2})

    def test_parameter_summary_counts_lora_and_unpacks_quantized_weights(self):
        model = TinyModel()
        model.layers[0].proj = nn.Linear(64, 64, bias=False)
        nn.quantize(model.layers[0], group_size=32, bits=4)
        model.freeze()
        linear_to_lora_layers(model, 1, {'rank': 2, 'scale': 1.0, 'dropout': 0.0})
        summary = parameter_summary(model)
        self.assertEqual(summary['adapter_parameters'], 2 * 64 + 2 * 64)
        self.assertEqual(summary['base_parameters'], 8 * 8 + 64 * 64)
        self.assertEqual(summary['adapted_projections']['proj']['rank'], 2)


class TrainingTests(unittest.TestCase):
    def test_overfitting_flag_ignores_noise_near_zero(self):
        def summary(vals):
            cb = SimpleNamespace(train_losses=[{'iteration': 1, 'loss': 1.0}],
                                 val_losses=[{'iteration': i, 'loss': v} for i, v in enumerate(vals)])
            return train.loss_summary(cb)
        noisy = summary([1.758, 0.015, 0.0023, 0.0025])
        self.assertEqual(noisy['best_val_iteration'], 2)
        self.assertFalse(noisy['overfitting_suspected'])
        self.assertTrue(summary([2.0, 0.5, 0.9])['overfitting_suspected'])

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

    def _config(self, root, **changes):
        model_path = root / 'base'
        model_path.mkdir(exist_ok=True)
        cfg = yaml.safe_load(Path('config/base.yaml').read_text())
        cfg.update(model=str(model_path), data=str(DATA_DIR), adapter_path=str(root / 'adapters'), batch_size=1, iters=1,
                   num_layers=1, val_batches=1, steps_per_eval=1, steps_per_report=1, save_every=1, max_seq_length=16, **changes)
        cfg['lora_parameters'] = {'rank': 2, 'scale': 2., 'dropout': 0.}
        config = root / 'config.yaml'
        config.write_text(yaml.safe_dump(cfg))
        return model_path, config

    def test_real_training_callback_weights_and_repeat_run_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path, config = self._config(root)
            callbacks = []
            for _ in range(2):
                with patch.object(train.lora, 'load', return_value=(TinyModel(), TinyTokenizer())):
                    callbacks.append(train.run_training(config_path=str(config), output_dir=tmp))
            first, second = callbacks
            self.assertNotEqual(first.run_dir, second.run_dir)
            for callback in callbacks:
                self.assertTrue(callback.train_losses)
                self.assertTrue(callback.val_losses)
                self.assertGreater(next(r['scored_tok_per_sec'] for r in callback.history if r['type'] == 'train'), 0)
                self.assertTrue((Path(callback.adapter_path) / 'adapters.safetensors').is_file())
                self.assertTrue((callback.run_dir / 'loss_curve.png').is_file())
                self.assertGreater(callback.parameters['adapter_parameters'], 0)
                self.assertIn('best_val_iteration', callback.losses)
            self.assertEqual(latest_path(root / 'adapters'), second.adapter_path)
            source, identity = resolve_adapter_source(str(model_path), second.adapter_path)
            self.assertEqual(source, str(model_path))
            with self.assertRaises(ValueError):
                resolve_adapter_source('different-model', second.adapter_path)
            manifest = json.loads((second.run_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['datasets']), 3)
            history = json.loads((second.run_dir / 'training_history.json').read_text())
            self.assertEqual(history['examples_seen'], 1)
            (model_path / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed after training'):
                resolve_adapter_source(str(model_path), second.adapter_path)

    def test_failed_training_is_recorded_and_publishes_no_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self._config(root)
            with patch.object(train.lora, 'load', side_effect=MemoryError('out of memory')), self.assertRaises(MemoryError):
                train.run_training(config_path=str(config), output_dir=tmp)
            manifest = json.loads(next(root.glob('runs/training-*/manifest.json')).read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertIn('out of memory', manifest['error'])
            self.assertFalse((root / 'adapters' / 'latest.json').exists())


class DataTokenStatsTests(unittest.TestCase):
    def test_token_stats_split_prompt_from_scored_answer(self):
        from src.show_data import token_stats

        stats = token_stats(DATA_DIR / 'test.jsonl', TinyTokenizer())
        # TinyTokenizer: 5 tokens for the full chat, 2 up to the answer prompt.
        self.assertEqual((stats['prompt_mean'], stats['prompt_min'], stats['prompt_max']), (2.0, 2, 2))
        self.assertEqual((stats['answer_mean'], stats['answer_min'], stats['answer_max']), (3.0, 3, 3))


class ForgettingTests(unittest.TestCase):
    def _adapter(self, root):
        adapter = Path(root) / 'adapters'
        adapter.mkdir()
        (adapter / 'adapters.safetensors').write_bytes(b'test')
        (adapter / 'adapter_config.json').write_text('{}')
        return str(adapter)

    def test_forgetting_check_pairs_base_and_lora_and_tests_the_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp)
            loaded = []

            def fake_load(source, **kwargs):
                variant = 'lora' if kwargs.get('adapter_path') else 'base'
                loaded.append(variant)
                return variant, object()   # (model, tokenizer); the tag is the model

            def fake_losses(model, tokenizer, samples):
                return {s['id']: (0.5 if model == 'base' else 0.9) for s in samples}

            with patch.object(forgetting, 'resolve_adapter_source', return_value=('base-snapshot', {'revision': 'abc'})), \
                    patch.object(forgetting.mlx_lm, 'load', side_effect=fake_load), \
                    patch.object(forgetting, 'per_record_losses', side_effect=fake_losses), \
                    patch.object(forgetting, 'plot_forgetting'):
                result = forgetting.run_forgetting_check(adapter_path=adapter, output_dir=tmp,
                                                        max_samples=8, quiet=True)
            self.assertEqual(loaded, ['base', 'lora'])
            self.assertAlmostEqual(result['base_loss'], 0.5)
            self.assertAlmostEqual(result['lora_loss'], 0.9)
            self.assertAlmostEqual(result['delta'], 0.4)
            self.assertEqual(result['sign_test']['increased'], 8)
            self.assertLess(result['sign_test']['p_value'], 0.02)
            self.assertIn('worse', result['verdict'])
            self.assertEqual(len(result['per_record']), 8)
            manifest = json.loads((Path(result['run_dir']) / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['kind'], 'forgetting')
            self.assertEqual(manifest['result']['sign_test']['increased'], 8)
            self.assertTrue((Path(result['run_dir']) / 'forgetting.json').is_file())

    def test_forgetting_check_verdict_stays_cautious_when_nothing_moved(self):
        verdict = forgetting.forgetting_verdict(1.0, 1.02, {'increased': 5, 'decreased': 4,
                                                            'unchanged': 15, 'p_value': 1.0})
        self.assertIn('No detectable damage', verdict)
        self.assertIn('never that it was preserved', verdict)


class TemperatureTests(unittest.TestCase):
    def test_inference_passes_temperature_to_the_sampler(self):
        records = [SimpleNamespace(text='a', generation_tokens=1, prompt_tokens=2, prompt_tps=30.,
                                   generation_tps=20., finish_reason='stop')]
        with patch('src.inference.make_sampler', return_value='sampler') as sampler, \
                patch.object(inference.mlx_lm, 'stream_generate', return_value=iter(records)):
            inference.generate_response(object(), TinyTokenizer(), sample()['messages'][:-1], temperature=0.7)
        sampler.assert_called_once_with(temp=0.7)
        for bad in (-0.1, 2.5):
            with self.assertRaises(ValueError):
                inference.generate_response(object(), TinyTokenizer(), sample()['messages'][:-1], temperature=bad)

    def test_evaluation_records_temperature_and_seed_and_forwards_them(self):
        seen = {}

        def fake_eval(model, tokenizer, samples, max_tokens, desc, generate=None, temperature=0.0):
            seen.setdefault('temperature', temperature)
            results = []
            for s in samples:
                parsed = parse_and_validate(s['messages'][-1]['content'])
                results.append({'id': s['id'], 'prompt': s['prompt'], 'expected': s['expected'], 'meta': s['meta'],
                                'raw_output': '', 'output_tokens': 1, **parsed,
                                **score_sample(s['expected'], s['normalized_expected'], parsed)})
            return {**summarize(results), 'sample_results': results, 'avg_output_tokens': 1}

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(evaluate, 'resolve_source', return_value=('base', {'revision': 'r'})), \
                patch.object(evaluate.mlx_lm, 'load', return_value=(object(), object())), \
                patch.object(evaluate, 'compute_perplexity', return_value={'loss': 1., 'perplexity': math.e}), \
                patch.object(evaluate, 'run_deterministic_eval', side_effect=fake_eval):
            report = evaluate.run_comprehensive_evaluation(variants=('base',), output_dir=tmp, num_eval_samples=4,
                                                           temperature=0.7, seed=7, quiet=True)
            manifest = json.loads((Path(report['run_dir']) / 'manifest.json').read_text())
        self.assertEqual(seen['temperature'], 0.7)
        self.assertEqual(manifest['generation']['temperature'], 0.7)
        self.assertEqual(manifest['generation']['seed'], 7)
        self.assertEqual(report['temperature'], 0.7)
        with self.assertRaises(ValueError):
            evaluate.run_comprehensive_evaluation(temperature=3.0)


if __name__ == '__main__':
    unittest.main()
