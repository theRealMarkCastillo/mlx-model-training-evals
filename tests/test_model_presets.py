"""Exercise model selection without downloading weights or running training."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import main
from src import benchmark, evaluate, train
from src.models import PRESETS, resolve_model_paths


class ModelPresetTests(unittest.TestCase):
    def test_configs_match_presets_and_keep_outputs_separate(self):
        paths = set()
        for profile in PRESETS.values():
            config = yaml.safe_load(Path(profile.config_path).read_text())
            self.assertEqual(config["model"], profile.model)
            self.assertEqual(config["adapter_path"], profile.adapter_path)
            self.assertEqual(config["batch_size"], profile.batch_size)
            self.assertEqual(config["num_layers"], profile.num_layers)
            self.assertEqual(config.get("grad_checkpoint", False), profile.grad_checkpoint)
            self.assertTrue(config["mask_prompt"])
            paths.add(profile.output_dir)
        self.assertEqual(len(paths), len(PRESETS))
        self.assertEqual(resolve_model_paths(), (PRESETS["3b"].model, "artifacts/adapters", Path("artifacts")))

    def test_explicit_paths_and_conflicting_model(self):
        model, adapter, output = resolve_model_paths("14b", adapter="custom/adapters", output_dir="custom/reports")
        self.assertEqual(model, PRESETS["14b"].model)
        self.assertEqual(adapter, "custom/adapters")
        self.assertEqual(output, Path("custom/reports"))
        with self.assertRaises(ValueError):
            resolve_model_paths("14b", model=PRESETS["3b"].model)
        with self.assertRaises(ValueError):
            train.run_training(config_path="config/lora_config.yaml", preset="14b")

    def test_training_uses_preset_config_and_iteration_override(self):
        for size, profile in PRESETS.items():
            with self.subTest(size=size), tempfile.TemporaryDirectory() as output:
                with patch.object(train.lora, "run") as run, patch.object(train, "plot_loss_curve"):
                    train.run_training(preset=size, iters_override=7, output_dir=output)
                args = run.call_args.args[0]
                self.assertEqual(args.model, profile.model)
                self.assertEqual(args.adapter_path, profile.adapter_path)
                self.assertEqual(args.batch_size, profile.batch_size)
                self.assertEqual(args.grad_checkpoint, profile.grad_checkpoint)
                self.assertEqual(args.iters, 7)
                history = json.loads((Path(output) / "training_history.json").read_text())
                self.assertEqual(history["model"], profile.model)

    def test_fusion_and_serving_follow_preset(self):
        for size, profile in PRESETS.items():
            for command in ("fuse", "serve"):
                with self.subTest(size=size, command=command):
                    with patch("sys.argv", ["main.py", command, "--preset", size]), patch.object(main.subprocess, "run") as run:
                        run.return_value.returncode = 0
                        self.assertEqual(main.main(), 0)
                    argv = run.call_args.args[0]
                    model = argv[argv.index("--model") + 1]
                    self.assertEqual(model, profile.model if command == "fuse" else profile.fused_path)
                    if command == "fuse":
                        self.assertEqual(argv[argv.index("--adapter-path") + 1], profile.adapter_path)
                        self.assertEqual(argv[argv.index("--save-path") + 1], profile.fused_path)

    def test_base_serving_and_legacy_port(self):
        with patch("sys.argv", ["main.py", "serve", "9000", "--preset", "32b", "--base"]), patch.object(main.subprocess, "run") as run:
            main.main()
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--model") + 1], PRESETS["32b"].model)
        self.assertEqual(argv[argv.index("--port") + 1], "9000")

    def test_training_failure_propagates(self):
        with patch("sys.argv", ["main.py", "train", "--preset", "14b"]), patch.object(main.subprocess, "run") as run:
            run.return_value.returncode = 2
            self.assertEqual(main.main(), 2)
            self.assertEqual(run.call_args.args[0][-2:], ["--preset", "14b"])

    def test_evaluation_uses_selected_model_adapter_and_reports(self):
        metrics = {
            "pure_json_rate": 1.0, "schema_valid_rate": 1.0, "tool_accuracy": 1.0,
            "exact_match_rate": 1.0, "avg_output_tokens": 10,
            "sample_results": [{"raw_output": "{}"}],
        }
        with tempfile.TemporaryDirectory() as output:
            with patch.object(evaluate.mlx_lm, "load", return_value=(object(), object())) as load, \
                    patch.object(evaluate, "compute_perplexity", return_value=(1.0, 2.718)), \
                    patch.object(evaluate, "run_deterministic_eval", side_effect=[dict(metrics), dict(metrics)]), \
                    patch.object(evaluate, "plot_eval_metrics") as plot:
                evaluate.run_comprehensive_evaluation(preset="14b", output_dir=output, num_eval_samples=1)
            self.assertEqual(load.call_args_list[0].args, (PRESETS["14b"].model,))
            self.assertEqual(load.call_args_list[1].kwargs, {"adapter_path": PRESETS["14b"].adapter_path})
            self.assertEqual(plot.call_args.args[2], Path(output) / "eval_comparison.png")
            report = json.loads((Path(output) / "eval_results.json").read_text())
            self.assertEqual(report["model"], PRESETS["14b"].model)

    def test_benchmark_uses_selected_model_adapter_and_reports(self):
        stats = {"avg_tokens_per_sec": 1, "peak_metal_memory_mb": 1,
                 "active_metal_memory_mb": 1, "avg_tokens_generated": 1}
        with tempfile.TemporaryDirectory() as output:
            with patch.object(benchmark.mlx_lm, "load", return_value=(object(), object())) as load, \
                    patch.object(benchmark, "benchmark_generation", return_value=stats):
                benchmark.run_benchmark_suite(preset="32b", output_dir=output)
            self.assertEqual(load.call_args_list[0].args, (PRESETS["32b"].model,))
            self.assertEqual(load.call_args_list[1].kwargs, {"adapter_path": PRESETS["32b"].adapter_path})
            report = json.loads((Path(output) / "benchmark_results.json").read_text())
            self.assertEqual(report["model"], PRESETS["32b"].model)


if __name__ == "__main__":
    unittest.main()
