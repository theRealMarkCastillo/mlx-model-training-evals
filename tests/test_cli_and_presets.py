"""CLI routing, preset configuration, and artifact pointer contracts."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from src import cli
from src.config import load_training_config
from src.models import PRESETS, resolve_model_paths
from src.runs import REPO_ROOT, latest_path, record_failure, new_run, write_json


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

    def test_ablation_rejects_conflicting_iters(self):
        from src.ablation import run_ablation
        with self.assertRaises(ValueError):
            run_ablation('iters', [10, 20], iters=5)
        with self.assertRaises(ValueError):
            run_ablation('optimizer', ['sgd'])

    def test_user_errors_exit_cleanly(self):
        with patch('src.evaluate.run_comprehensive_evaluation', side_effect=FileNotFoundError('missing')):
            self.assertEqual(cli.main(['eval']), 2)

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
