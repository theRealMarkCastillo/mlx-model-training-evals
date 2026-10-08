"""Pure-Python core tests: schema parsing, scoring, statistics, dataset design, offline data summary.

No MLX, no transformers, no model downloads. This file is the suite that can run
on any platform (see .github/workflows/ci.yml), so keep those imports out —
`CoreImportGuardTests` fails the build if a core module starts importing them.
"""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console

from src.dataset import DATA_DIR, fewshot_messages, load_general_samples, load_samples, select_shots, validate_splits
from src.generate_data import (
    ACTION_TOOLS,
    GENERAL_PROMPTS,
    GENERAL_SYSTEM_PROMPT,
    SEEN,
    UNSEEN,
    build_general_set,
    build_splits,
    to_chat_format,
)
from src.metrics import (
    failure_examples,
    paired_comparison,
    pass_at_k,
    score_sample,
    sign_test,
    summarize,
    sweep_summary,
    wilson_interval,
)
from src.runs import REPO_ROOT
from src.schema import PARAM_MODEL_MAP, ToolCall, parse_and_validate
from src.show_data import print_dataset_summary, split_order, split_summary

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


class SchemaTests(unittest.TestCase):
    def test_malformed_envelopes_never_crash(self):
        for value in ([], {}, 1, True, None, 'wrong'):
            result = parse_and_validate(json.dumps({'tool': value, 'parameters': {}}))
            self.assertFalse(result['is_schema_valid'])
        for text in ('[]', 'null', '1', 'nonsense', '{', '```json\n{}\n```'):
            self.assertFalse(parse_and_validate(text)['is_schema_valid'])

    def test_valid_json_means_an_object_was_recovered(self):
        for text in ('[]', 'null', '1', '42', 'true', '"just a string"'):
            result = parse_and_validate(text)
            self.assertFalse(result['is_valid_json'], text)
            self.assertFalse(result['is_pure_json'], text)
        wrapped = 'Here it is: ' + json.dumps(TARGET)
        self.assertTrue(parse_and_validate(wrapped)['is_valid_json'])
        self.assertFalse(parse_and_validate(wrapped)['is_pure_json'])
        self.assertTrue(parse_and_validate(json.dumps(TARGET))['is_pure_json'])

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
        with self.assertRaises(ValidationError):
            ToolCall.model_validate({'tool': 'no_action', 'parameters': {'reason': 'because'}})

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

    def test_stray_brace_before_json_is_skipped(self):
        result = parse_and_validate('I will use {tool} now: ' + json.dumps(TARGET))
        self.assertTrue(result['is_schema_valid'])
        self.assertFalse(result['is_pure_json'])


class DatasetTests(unittest.TestCase):
    def test_regenerated_data_is_reproducible_disjoint_and_grounded(self):
        splits = build_splits()
        self.assertEqual(splits, build_splits())
        prompts = set()
        for name, records in splits.items():
            stored = [json.loads(line) for line in (DATA_DIR / f'{name}.jsonl').read_text().splitlines()]
            self.assertEqual(stored, to_chat_format(records))
            for record in records:
                self.assertNotIn(record['prompt'], prompts)
                prompts.add(record['prompt'])
                target = ToolCall.model_validate_json(record['completion']).model_dump()
                params, meta = target['parameters'], record['meta']
                defaults = {f: info.get_default(call_default_factory=True)
                            for f, info in PARAM_MODEL_MAP[target['tool']].model_fields.items() if not info.is_required()}
                for field in meta['omitted']:
                    self.assertEqual(params[field], defaults[field], (name, record['prompt']))
                if target['tool'] == 'deploy_service':
                    channels = SEEN.channels + UNSEEN.channels
                    self.assertEqual(set(params['notify_channels']), {ch for ch in channels if ch in record['prompt']})
                if target['tool'] == 'restart_pod':
                    self.assertIn(params['reason'], record['prompt'])
                if target['tool'] != 'no_action':
                    for value in params.values():
                        if isinstance(value, str) and value not in defaults.values() and target['tool'] != 'deploy_service':
                            self.assertIn(value, record['prompt'])
        validate_splits()

    def test_split_design(self):
        splits = build_splits()
        self.assertEqual({len(splits[n]) for n in ('train', 'valid', 'test')}, {250, 50, 75})
        families = {name: {r['meta']['family'] for r in records} for name, records in splits.items()}
        self.assertEqual(families['train'], {0, 1})
        self.assertEqual((families['valid'], families['test']), ({2}, {3}))
        tools = [r['meta']['tool'] for r in splits['test']]
        self.assertEqual({tools.count(t) for t in set(tools)}, {15})
        self.assertTrue(any(r['meta']['omitted'] for r in splits['train']))
        # Unseen entities never occur in standard splits.
        train_text = ' '.join(r['prompt'] for r in splits['train'])
        for service in UNSEEN.services:
            self.assertNotIn(service, train_text)
        self.assertTrue(all(r['meta']['entities'] == 'unseen' for r in splits['challenge_entities']))
        self.assertTrue(all(r['meta']['tool'] in ACTION_TOOLS for r in splits['challenge_entities']))
        self.assertTrue(all(r['meta']['omitted'] for r in splits['challenge_defaults']))
        self.assertTrue(all(r['meta']['tool'] == 'no_action' for r in splits['challenge_abstain']))

    def test_balanced_subsets_and_shots(self):
        subset = load_samples(DATA_DIR / 'test.jsonl', 10)
        tools = [r['expected']['tool'] for r in subset]
        self.assertEqual({tools.count(t) for t in set(tools)}, {2})
        shots = select_shots(count=5)
        self.assertEqual(len({s['expected']['tool'] for s in shots}), 5)
        self.assertTrue(all(s['id'].startswith('train.jsonl') for s in shots))
        messages = fewshot_messages(sample()['messages'], shots[:2])
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant'])
        self.assertEqual(messages[-2]['content'], 'request')

    def test_empty_invalid_and_nonpositive_inputs(self):
        for count in (0, -1):
            with self.assertRaises(ValueError):
                load_samples(DATA_DIR / 'test.jsonl', count)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.jsonl'
            path.write_text('')
            with self.assertRaises(ValueError):
                load_samples(path)


class MetricsTests(unittest.TestCase):
    def test_wilson_interval(self):
        low, high = wilson_interval(12, 15)
        self.assertAlmostEqual(low, 0.548, places=2)
        self.assertAlmostEqual(high, 0.930, places=2)
        self.assertEqual(wilson_interval(0, 10)[0], 0.0)
        self.assertLess(wilson_interval(0, 10)[1], 0.35)
        self.assertEqual(wilson_interval(10, 10)[1], 1.0)

    def test_paired_comparison(self):
        a = [{'id': str(i), 'param_exact': i < 9} for i in range(10)]
        b = [{'id': str(i), 'param_exact': i < 1} for i in range(10)]
        result = paired_comparison(a, b)
        self.assertEqual((result['both_correct'], result['only_a'], result['only_b'], result['neither']), (1, 8, 0, 1))
        self.assertAlmostEqual(result['p_value'], 2 / 2 ** 8)
        self.assertEqual(paired_comparison(a, a)['p_value'], 1.0)
        with self.assertRaises(ValueError):
            paired_comparison(a, list(reversed(b)))

    def test_categories_and_fields(self):
        missing = copy.deepcopy(TARGET)
        del missing['parameters']['notify_channels']
        wrong = copy.deepcopy(TARGET)
        wrong['parameters']['replicas'] = 4
        cases = {json.dumps(TARGET): None, json.dumps(missing): 'omitted_default', json.dumps(wrong): 'parameters',
                 'x ' + json.dumps(TARGET): 'format', '{"tool":"no_action","parameters":{"reason":"unsupported_request"}}': 'tool',
                 '{"tool":"deploy_service"}': 'schema', 'nope': 'json'}
        normalized = ToolCall.model_validate(TARGET).model_dump()
        for text, category in cases.items():
            scored = score_sample(TARGET, normalized, parse_and_validate(text))
            self.assertEqual(scored['error_category'], category, text)
        scored = score_sample(TARGET, normalized, parse_and_validate(json.dumps(wrong)))
        self.assertEqual(scored['wrong_fields'], ['replicas'])

    def test_summary_breakdowns(self):
        normalized = ToolCall.model_validate(TARGET).model_dump()
        outputs = [json.dumps(TARGET), 'nope', json.dumps(TARGET)]
        results = []
        for i, text in enumerate(outputs):
            parsed = parse_and_validate(text)
            results.append({'id': str(i), 'prompt': 'p', 'expected': TARGET, 'raw_output': text,
                            'meta': {'omitted': ['replicas'] if i == 2 else []},
                            **parsed, **score_sample(TARGET, normalized, parsed)})
        summary = summarize(results)
        self.assertAlmostEqual(summary['exact_match_rate'], 2 / 3)
        self.assertEqual(summary['per_tool']['deploy_service']['n'], 3)
        self.assertAlmostEqual(summary['per_field']['deploy_service']['replicas'], 2 / 3)
        self.assertEqual(summary['error_categories']['json'], 1)
        self.assertEqual(summary['slices']['omitted_optional']['n'], 1)
        self.assertEqual([e['category'] for e in failure_examples(results)], ['json'])

    def test_sign_test_counts_directions_and_flags_lopsided_splits(self):
        flat = sign_test([0.0, 0.0, 0.0])
        self.assertEqual((flat['increased'], flat['decreased'], flat['unchanged'], flat['p_value']), (0, 0, 3, 1.0))
        balanced = sign_test([1.0, -1.0, 1.0, -1.0])
        self.assertEqual((balanced['increased'], balanced['decreased']), (2, 2))
        self.assertAlmostEqual(balanced['p_value'], 1.0)
        lopsided = sign_test([1.0] * 8 + [-1.0])
        self.assertEqual((lopsided['increased'], lopsided['decreased']), (8, 1))
        self.assertAlmostEqual(lopsided['p_value'], 2 * (1 + 9) / 2 ** 9)
        self.assertLess(lopsided['p_value'], 0.05)


    def test_sweep_summary_reports_mean_and_spread(self):
        spread = sweep_summary([0.5, 1.0, 0.75])
        self.assertEqual(spread['n'], 3)
        self.assertAlmostEqual(spread['mean'], 0.75)
        self.assertAlmostEqual(spread['stdev'], 0.2041241452319315, places=9)
        self.assertEqual((spread['min'], spread['max']), (0.5, 1.0))
        self.assertEqual(sweep_summary([0.8])['stdev'], 0.0)
        with self.assertRaises(ValueError):
            sweep_summary([])


    def test_pass_at_k_counts_what_retrying_recovers(self):
        def result(index, exact):
            return {'id': f'test.jsonl:{index}', 'param_exact': exact}

        repeat_one = [result(1, True), result(2, False), result(3, False)]
        repeat_two = [result(1, True), result(2, True), result(3, False)]
        metrics = pass_at_k([repeat_one, repeat_two])
        self.assertEqual(metrics['k'], 2)
        self.assertEqual(metrics['n'], 3)
        self.assertAlmostEqual(metrics['first_repeat_rate'], 1 / 3)
        self.assertEqual(metrics['per_repeat_rates'], [1 / 3, 2 / 3])
        self.assertAlmostEqual(metrics['pass_at_k_rate'], 2 / 3)
        self.assertEqual(metrics['records_all_failed'], 1)   # record 3 fails both draws
        self.assertEqual(pass_at_k([repeat_one])['pass_at_k_rate'], pass_at_k([repeat_one])['first_repeat_rate'])
        for bad in ([], [[]], [repeat_one, repeat_one[:2]], [repeat_one, [result(9, True)] * 3]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pass_at_k(bad)


class GeneralSetTests(unittest.TestCase):
    def test_general_data_is_committed_deterministic_and_not_tool_shaped(self):
        stored = [json.loads(line) for line in (DATA_DIR / 'general.jsonl').read_text().splitlines()]
        self.assertEqual(stored, build_general_set())
        self.assertEqual(len(stored), len(GENERAL_PROMPTS))
        records = load_general_samples(DATA_DIR / 'general.jsonl')
        prompts = set()
        for record in records:
            self.assertEqual(record['messages'][0]['content'], GENERAL_SYSTEM_PROMPT)
            self.assertNotIn(record['prompt'], prompts)
            prompts.add(record['prompt'])
            self.assertNotIn('tool', record['prompt'])
            # The assistant answers are ordinary prose, not tool-call envelopes.
            self.assertFalse(parse_and_validate(record['expected'])['is_valid_json'])
        # General prompts never collide with any training/evaluation prompt.
        tool_prompts = set()
        for name in ('train', 'valid', 'test'):
            tool_prompts.update(r['prompt'] for r in load_samples(DATA_DIR / f'{name}.jsonl'))
        self.assertFalse(prompts.intersection(tool_prompts))

    def test_general_loader_validates_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'general.jsonl'
            path.write_text(json.dumps({'messages': [
                {'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'q'},
                {'role': 'assistant', 'content': '   '}]}) + '\n')
            with self.assertRaises(ValueError):
                load_general_samples(path)
            path.write_text(json.dumps({'messages': [
                {'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'q'}]}) + '\n')
            with self.assertRaises(ValueError):
                load_general_samples(path)
            path.write_text('')
            with self.assertRaises(ValueError):
                load_general_samples(path)


class ShowDataTests(unittest.TestCase):
    def test_split_summaries_match_the_design(self):
        summaries = {name: split_summary(DATA_DIR / f'{name}.jsonl') for name in split_order()}
        self.assertEqual(set(summaries), {'train', 'valid', 'test', 'challenge_entities',
                                          'challenge_defaults', 'challenge_abstain'})
        self.assertEqual(summaries['test']['n'], 75)
        self.assertTrue(summaries['test']['balanced'])
        self.assertEqual(summaries['train']['families'], [0, 1])
        self.assertEqual(summaries['challenge_defaults']['with_omitted'], summaries['challenge_defaults']['n'])
        self.assertEqual(summaries['challenge_abstain']['tools'], {'no_action': 40})

    def test_printed_summary_is_offline_and_reports_every_split(self):
        import src.show_data as show_data

        buffer = StringIO()
        original = show_data.console
        show_data.console = Console(file=buffer, width=250)
        try:
            summaries = print_dataset_summary(examples=1)
        finally:
            show_data.console = original
        output = buffer.getvalue()
        for name in ('train', 'valid', 'test', 'challenge_entities', 'challenge_defaults', 'challenge_abstain'):
            self.assertIn(name, output)
        self.assertIn('general.jsonl', output)   # the forgetting check's set is explained, not listed as a split
        self.assertIn('no prompt appears twice', output)
        self.assertEqual(len(summaries), 6)


class ExportReferenceTests(unittest.TestCase):
    """The export script feeds the committed reference data, so its contract is tested."""

    def _write_pointer(self, preset_root, kind, run_dir, **extra):
        (preset_root / f'latest_{kind}.json').write_text(json.dumps({'run_dir': str(run_dir), **extra}))

    def test_build_summary_exports_evaluation_forgetting_and_provenance(self):
        from scripts import export_reference

        with tempfile.TemporaryDirectory() as tmp:
            preset_root, out = Path(tmp) / 'preset', Path(tmp) / 'out'
            # Training run
            training = preset_root / 'runs' / 'training-1'
            training.mkdir(parents=True)
            (training / 'loss_curve.png').write_bytes(b'png')
            (training / 'training_history.json').write_text(json.dumps({
                'model': 'm', 'training_time_seconds': 1.0, 'peak_memory_mb': 2.0, 'examples_seen': 8,
                'train_records': 4, 'parameters': {'adapter_parameters': 10}, 'loss_summary': {'best_val_loss': 0.1},
                'history': []}))
            self._write_pointer(preset_root, 'training', training, config={
                'lora_parameters': {'rank': 8}, 'learning_rate': 1e-4, 'batch_size': 4, 'iters': 2,
                'num_layers': 1, 'seed': 42}, versions={'mlx': '1'}, platform='test')
            # Evaluation run, including the fields added for the grammar variant
            evaluation = preset_root / 'runs' / 'evaluation-1'
            evaluation.mkdir(parents=True)
            metrics = {'num_samples': 1, 'exact_match_rate': 0.0, 'sample_results': [
                {'id': 'test.jsonl:1', 'prompt': 'p', 'expected': {}, 'raw_output': 'x',
                 'error_category': 'json', 'wrong_fields': [], 'prompt_tokens': 10}]}
            (evaluation / 'eval_results.json').write_text(json.dumps({
                'variants': ['base', 'grammar'], 'constrained': True, 'temperature': 0.0,
                'paired': {'reference': 'grammar', 'comparisons': {}},
                'paired_schema': {'reference': 'grammar', 'comparisons': {'base': {'only_a': 1, 'only_b': 0}}},
                'datasets': {'test': {'base': metrics, 'grammar': metrics}}}))
            (evaluation / 'eval_comparison.png').write_bytes(b'png')
            self._write_pointer(preset_root, 'evaluation', evaluation)
            # Forgetting run
            forgetting = preset_root / 'runs' / 'forgetting-1'
            forgetting.mkdir(parents=True)
            (forgetting / 'forgetting.png').write_bytes(b'png')
            (forgetting / 'forgetting.json').write_text(json.dumps({
                'n': 24, 'base_loss': 1.0, 'lora_loss': 1.1, 'delta': 0.1,
                'sign_test': {'increased': 15, 'decreased': 9, 'unchanged': 0, 'p_value': 0.3},
                'verdict': 'v', 'per_record': [{'id': 'x'}]}))
            self._write_pointer(preset_root, 'forgetting', forgetting)

            summary = export_reference.build_summary(preset_root, out)

            self.assertEqual(summary['evaluation']['variants'], ['base', 'grammar'])
            self.assertTrue(summary['evaluation']['constrained'])
            self.assertIn('paired_schema', summary['evaluation'])
            self.assertNotIn('sample_results', summary['evaluation']['datasets']['test']['base'])
            self.assertEqual(summary['evaluation']['prompt_tokens'], {'base': 10, 'grammar': 10})
            self.assertEqual(summary['evaluation']['example_failures']['test']['base'][0]['category'], 'json')
            self.assertEqual(summary['forgetting']['sign_test']['increased'], 15)
            self.assertNotIn('per_record', summary['forgetting'])   # keep the committed file small
            self.assertEqual(summary['provenance']['forgetting'], None)  # pointer has no run_id
            for name in ('summary.json', 'loss_curve.png', 'eval_comparison.png', 'forgetting.png'):
                self.assertTrue((out / name).is_file(), name)
            self.assertEqual(json.loads((out / 'summary.json').read_text())['forgetting']['n'], 24)

    def test_richest_ablation_wins_over_the_newest_pointer(self):
        """A per-seed sweep (one point per run) must not displace the iteration sweep."""
        from scripts import export_reference

        def ablation_run(preset_root, name, rows, param):
            run = preset_root / 'runs' / name
            run.mkdir(parents=True)
            (run / 'ablation.json').write_text(json.dumps({'param': param, 'rows': rows, 'run_dir': str(run)}))
            (run / 'ablation.png').write_bytes(b'png')
            (run / 'manifest.json').write_text(json.dumps({'run_id': name, 'kind': 'ablation', 'status': 'complete'}))
            return run

        with tempfile.TemporaryDirectory() as tmp:
            preset_root, out = Path(tmp) / 'preset', Path(tmp) / 'out'
            iteration_rows = [{'value': value, 'exact_match_rate': 0.5} for value in (25, 50, 100, 200)]
            ablation_run(preset_root, 'ablation-iters', iteration_rows, 'iters')
            seed_run = ablation_run(preset_root, 'ablation-seed-42', [{'value': 42, 'exact_match_rate': 1.0}], 'seed')
            (preset_root / 'latest_ablation.json').write_text(json.dumps({'run_dir': str(seed_run)}))
            summary = export_reference.build_summary(preset_root, out)
            self.assertTrue((out / 'ablation.png').is_file())
        self.assertEqual(summary['ablation']['param'], 'iters')
        self.assertEqual(len(summary['ablation']['rows']), 4)
        self.assertEqual(summary['provenance']['ablation'], 'ablation-iters')

    def test_missing_runs_are_skipped_without_crashing(self):
        from scripts import export_reference

        with tempfile.TemporaryDirectory() as tmp:
            summary = export_reference.build_summary(Path(tmp) / 'nothing-here', Path(tmp) / 'out')
        self.assertEqual(set(summary), {'provenance'})
        self.assertIsNone(summary['provenance']['training'])


class MergeSeedSweepTests(unittest.TestCase):
    def _ablation_run(self, preset_root, name, rows, param='seed'):
        run = preset_root / 'runs' / name
        run.mkdir(parents=True)
        (run / 'ablation.json').write_text(json.dumps({'param': param, 'rows': rows, 'run_dir': str(run)}))
        return run

    def _row(self, seed, rate):
        return {'value': seed, 'exact_match_rate': rate, 'ci95': [rate - 0.1, rate + 0.1],
                'schema_valid_rate': 1.0, 'test_loss': 0.01, 'best_val_loss': 0.02, 'adapter_parameters': 6651904}

    def test_merges_seed_runs_and_deduplicates_retries(self):
        from scripts import merge_seed_sweep

        with tempfile.TemporaryDirectory() as tmp:
            preset_root = Path(tmp) / 'preset'
            self._ablation_run(preset_root, 'ablation-iters', [self._row(25, 0.27)], param='iters')
            self._ablation_run(preset_root, 'ablation-seed-42-old', [self._row(42, 0.80)])
            self._ablation_run(preset_root, 'ablation-seed-42-retry', [self._row(42, 1.00)])
            self._ablation_run(preset_root, 'ablation-seed-43', [self._row(43, 0.90)])
            merged = merge_seed_sweep.merge_seed_rows(merge_seed_sweep.collect_seed_rows(preset_root))
        self.assertEqual([row['value'] for row in merged['rows']], [42, 43])
        self.assertEqual(merged['rows'][0]['exact_match_rate'], 1.00)   # the retry wins
        self.assertEqual(merged['rows'][0]['sweep_run'], 'ablation-seed-42-retry')
        self.assertEqual(merged['spread']['n'], 2)
        self.assertAlmostEqual(merged['spread']['mean'], 0.95)
        self.assertAlmostEqual(merged['spread']['stdev'], 0.05)

    def test_missing_or_partial_sweeps_fail_clearly(self):
        from scripts import merge_seed_sweep

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                merge_seed_sweep.merge_seed_rows([])
            # A failed sweep leaves no ablation.json, so nothing is collected from it.
            preset_root = Path(tmp) / 'preset'
            run = self._ablation_run(preset_root, 'ablation-seed-44', [self._row(44, 0.7)])
            (run / 'ablation.json').unlink()
            self.assertEqual(merge_seed_sweep.collect_seed_rows(preset_root), [])


class CoreImportGuardTests(unittest.TestCase):
    def test_core_modules_import_without_mlx_or_transformers(self):
        """The Linux CI job depends on this staying true."""
        code = ("import sys, src.metrics, src.dataset, src.schema, src.show_data, src.generate_data; "
                "assert 'mlx' not in sys.modules, 'core module imported mlx'; "
                "assert 'transformers' not in sys.modules, 'core module imported transformers'")
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
