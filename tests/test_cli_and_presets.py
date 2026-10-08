"""CLI routing, preset configuration, and artifact pointer contracts."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src import cli
from src.config import load_training_config
from src.models import PRESETS, resolve_model_paths
from src.runs import REPO_ROOT, latest_path, new_run, record_failure, write_json


class PresetTests(unittest.TestCase):
    def test_presets_merge_over_base_and_validate(self):
        roots = set()
        for size, profile in PRESETS.items():
            config = load_training_config(preset=size)
            for key, value in profile.overrides().items():
                self.assertEqual(config[key], value)
            self.assertTrue(config['mask_prompt'])
            self.assertEqual(config['lora_parameters']['rank'], 8)
            roots.add(profile.output_dir)
        self.assertEqual(len(roots), len(PRESETS))
        self.assertTrue(all(root.parent == REPO_ROOT / 'artifacts' for root in roots))

    def test_overrides_reach_nested_lora_parameters(self):
        config = load_training_config(preset='7b', overrides={'rank': 4, 'iters': 3})
        self.assertEqual((config['lora_parameters']['rank'], config['iters']), (4, 3))

    def test_invalid_overrides_fail(self):
        for bad in ({'iters': 0}, {'rank': True}, {'mask_prompt': False}, {'bogus': 1},
                    {'num_layers': 0}, {'dropout': 1.0}, {'fine_tune_type': 'dora'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_training_config(overrides=bad)

    def test_standalone_config_resolves_paths_against_repo_root(self):
        """`--config` files are repo-relative, like presets, from any CWD."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text(yaml.safe_dump({
                'model': 'some/model', 'data': 'data', 'adapter_path': 'artifacts/custom/adapters',
                'fine_tune_type': 'lora', 'mask_prompt': True,
                'lora_parameters': {'rank': 1, 'scale': 1.0, 'dropout': 0.0},
                'learning_rate': 1e-4, 'optimizer': 'adamw', 'iters': 1, 'batch_size': 1,
                'max_seq_length': 16, 'steps_per_report': 1, 'steps_per_eval': 1,
                'save_every': 1, 'num_layers': 1, 'val_batches': 1,
            }))
            config = load_training_config(config_path=str(path))
            self.assertEqual(config['data'], str(REPO_ROOT / 'data'))
            self.assertEqual(config['adapter_path'], str(REPO_ROOT / 'artifacts' / 'custom' / 'adapters'))


class PointerTests(unittest.TestCase):
    def test_explicit_paths_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, adapter, output = resolve_model_paths('14b', adapter=tmp, output_dir='custom/reports')
            self.assertEqual((model, adapter, output), (PRESETS['14b'].model, tmp, Path('custom/reports')))
        with self.assertRaises(ValueError):
            resolve_model_paths('14b', model=PRESETS['3b'].model)
        with self.assertRaises(ValueError):
            resolve_model_paths(model='other/model')

    def test_default_adapter_requires_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(PRESETS['3b'].__class__, 'adapter_path', property(lambda self: tmp)):
            # Stale weights without a pointer must not be used silently.
            (Path(tmp) / 'adapters.safetensors').write_bytes(b'old')
            with self.assertRaisesRegex(FileNotFoundError, 'No completed run'):
                resolve_model_paths('3b')
            self.assertIsNone(resolve_model_paths('3b', need_adapter=False)[1])

    def test_latest_pointer_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / 'run' / 'adapters'
            artifact.mkdir(parents=True)
            write_json(Path(tmp) / 'latest.json', {'path': str(artifact)})
            self.assertEqual(latest_path(tmp, required=True), str(artifact))
            self.assertEqual(resolve_model_paths(adapter=tmp)[1], str(artifact))
            artifact.rmdir()
            with self.assertRaises(FileNotFoundError):
                latest_path(tmp)

    def test_failed_runs_are_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, manifest = new_run(tmp, 'demo')
            with self.assertRaises(RuntimeError), record_failure(directory, manifest):
                raise RuntimeError('boom')
            saved = json.loads((directory / 'manifest.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('boom', saved['error'])
            self.assertIn('Traceback', saved['traceback'])
            self.assertFalse((Path(tmp) / 'latest_demo.json').exists())


class CliTests(unittest.TestCase):
    def test_commands_dispatch_in_process(self):
        with patch('src.train.run_training') as train:
            self.assertEqual(cli.main(['train', '--preset', '14b', '--iters', '5', '--rank', '4']), 0)
        self.assertEqual(train.call_args.kwargs, {'config_path': None, 'iters_override': 5, 'preset': '14b',
                                                  'output_dir': None, 'overrides': {'rank': 4}})
        with patch('src.evaluate.run_comprehensive_evaluation') as evaluate:
            cli.main(['eval', '--variants', 'base', 'fewshot', '--challenge'])
        self.assertEqual(evaluate.call_args.kwargs['variants'], ['base', 'fewshot'])
        self.assertTrue(evaluate.call_args.kwargs['challenge'])
        self.assertIsNone(evaluate.call_args.kwargs['num_eval_samples'])
        with patch('src.ablation.run_ablation') as ablate:
            cli.main(['ablate', 'rank', '4', '8', '--iters', '20'])
        self.assertEqual(ablate.call_args.args, ('rank', ['4', '8']))

    def test_train_seed_flag_reaches_overrides(self):
        with patch('src.train.run_training') as train:
            cli.main(['train', '--preset', '3b', '--seed', '7'])
        self.assertEqual(train.call_args.kwargs['overrides'], {'seed': 7})

    def test_inspect_and_show_data_dispatch(self):
        with patch('src.inspect.run_inspection') as inspect:
            cli.main(['inspect', '--split', 'valid', '--index', '2', '--variant', 'base', '--top-k', '3'])
        self.assertEqual(inspect.call_args.args, ('valid', 2))
        self.assertEqual(inspect.call_args.kwargs['variant'], 'base')
        self.assertEqual(inspect.call_args.kwargs['top_k'], 3)
        with patch('src.show_data.print_dataset_summary') as show:
            cli.main(['show-data', '--tokens', '--examples', '2'])
        show.assert_called_once_with(with_token_stats=True, examples=2, preset=None)

    def test_eval_constrained_flag_reaches_the_evaluator(self):
        with patch('src.evaluate.run_comprehensive_evaluation') as evaluate:
            cli.main(['eval', '--variants', 'base', '--constrained'])
        self.assertTrue(evaluate.call_args.kwargs['constrained'])

    def test_ablate_seed_is_supported(self):
        from src.ablation import ABLATABLE
        self.assertEqual(ABLATABLE['seed'], int)
        with patch('src.ablation.run_ablation') as ablate:
            cli.main(['ablate', 'seed', '42', '43', '--iters', '25'])
        self.assertEqual(ablate.call_args.args, ('seed', ['42', '43']))

    def test_ablation_rejects_conflicting_iters(self):
        from src.ablation import run_ablation
        with self.assertRaises(ValueError):
            run_ablation('iters', [10, 20], iters=5)
        with self.assertRaises(ValueError):
            run_ablation('optimizer', ['sgd'])

    def test_user_errors_exit_cleanly(self):
        with patch('src.evaluate.run_comprehensive_evaluation', side_effect=FileNotFoundError('missing')):
            self.assertEqual(cli.main(['eval']), 2)

    def test_memory_errors_print_a_friendly_message(self):
        with patch('src.evaluate.run_comprehensive_evaluation', side_effect=MemoryError('out of memory')):
            self.assertEqual(cli.main(['eval']), 1)
        with patch.dict(os.environ, {'MLX_EVALS_DEBUG': '1'}):
            with patch('src.evaluate.run_comprehensive_evaluation', side_effect=MemoryError('out of memory')):
                with self.assertRaises(MemoryError):
                    cli.main(['eval'])

    def test_show_mask_rejects_out_of_range_index_without_downloads(self):
        with patch('transformers.AutoTokenizer') as tokenizer:
            self.assertEqual(cli.main(['show-mask', '--split', 'test', '--index', '1000000']), 2)
            self.assertEqual(cli.main(['show-mask', '--split', 'test', '--index', '-1']), 2)
            tokenizer.from_pretrained.assert_not_called()

    def test_show_mask_renders_a_valid_record(self):
        with patch('transformers.AutoTokenizer'), \
                patch('src.explain.loss_mask_tokens', return_value=[('a', False), ('b', True)]) as mask, \
                patch('src.explain.mask_counts', return_value={'total_tokens': 2, 'scored_tokens': 1, 'masked_tokens': 1}), \
                patch('src.explain.render_mask_rich', return_value='tokens'):
            self.assertEqual(cli.main(['show-mask', '--split', 'test', '--index', '0']), 0)
            mask.assert_called_once()

    def test_serving(self):
        for size, profile in PRESETS.items():
            with patch.object(cli.subprocess, 'run') as run:
                run.return_value.returncode = 0
                self.assertEqual(cli.main(['serve', '--port', '9000', '--preset', size, '--base']), 0)
                argv = run.call_args.args[0]
                self.assertEqual(argv[argv.index('--model') + 1], profile.model)
                self.assertEqual(argv[argv.index('--port') + 1], '9000')

    def test_subprocess_failures_propagate(self):
        with patch.object(cli.subprocess, 'run') as run:
            run.return_value.returncode = 7
            self.assertEqual(cli.main(['notebook']), 7)
            self.assertEqual(cli.main(['serve', '--base']), 7)


if __name__ == '__main__':
    unittest.main()
