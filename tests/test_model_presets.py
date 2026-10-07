"""CLI routing and preset configuration contracts."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import yaml

import main
from src.models import PRESETS, resolve_model_paths
from src.runs import write_json, latest_path


class ModelPresetTests(unittest.TestCase):
    def test_configs_match_presets(self):
        paths = set()
        for profile in PRESETS.values():
            config = yaml.safe_load(Path(profile.config_path).read_text())
            for key in ('model', 'adapter_path', 'batch_size', 'num_layers'):
                self.assertEqual(config[key], getattr(profile, key))
            self.assertEqual(config.get('grad_checkpoint', False), profile.grad_checkpoint)
            self.assertTrue(config['mask_prompt'])
            paths.add(profile.output_dir)
        self.assertEqual(len(paths), len(PRESETS))

    def test_explicit_paths_and_conflicts(self):
        model, adapter, output = resolve_model_paths('14b', adapter='custom/adapters', output_dir='custom/reports')
        self.assertEqual((model, adapter, output), (PRESETS['14b'].model, 'custom/adapters', Path('custom/reports')))
        with self.assertRaises(ValueError):
            resolve_model_paths('14b', model=PRESETS['3b'].model)

    def test_latest_adapter_and_fused_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / 'run' / 'adapters'
            artifact.mkdir(parents=True)
            write_json(Path(tmp) / 'latest.json', {'path': str(artifact)})
            self.assertEqual(latest_path(tmp), str(artifact))
            self.assertEqual(resolve_model_paths(adapter=tmp)[1], str(artifact))

    def test_subprocess_failures_propagate(self):
        for command in ('prepare', 'train', 'eval', 'benchmark', 'fuse', 'notebook'):
            with self.subTest(command=command), patch('sys.argv', ['main.py', command]), patch.object(main.subprocess, 'run') as run:
                run.return_value.returncode = 7
                self.assertEqual(main.main(), 7)

    def test_fusion_forwards_preset(self):
        with patch('sys.argv', ['main.py', 'fuse', '--preset', '14b']), patch.object(main.subprocess, 'run') as run:
            run.return_value.returncode = 0
            self.assertEqual(main.main(), 0)
            self.assertEqual(run.call_args.args[0][1:], ['src/fuse.py', '--preset', '14b'])

    def test_serving(self):
        for size, profile in PRESETS.items():
            with patch('sys.argv', ['main.py', 'serve', '9000', '--preset', size, '--base']), patch.object(main.subprocess, 'run') as run:
                main.main()
                argv = run.call_args.args[0]
                self.assertEqual(argv[argv.index('--model')+1], profile.model)
                self.assertEqual(argv[argv.index('--port')+1], '9000')
