"""Mechanics tests for the from-scratch toy training loop (src/mini_train.py)."""
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from src.mini_train import (
    ANSWER,
    CONTEXT,
    D_HIDDEN,
    VOCAB,
    adamw_step,
    build_toy_data,
    forward,
    init_adapter,
    init_base,
    run_toy_training,
)


class ToyForwardTests(unittest.TestCase):
    def test_adapter_is_identity_at_step_zero(self):
        """B = 0 means the first forward pass equals the frozen base model."""
        train, _ = build_toy_data(1, n_train=3, n_val=1)
        x = mx.array(train)
        base = init_base(1)
        adapter = init_adapter(3, 1)
        with_lora = forward(base, adapter, x, scale=1.0)
        frozen_only = forward(base, {}, x, scale=1.0)
        self.assertTrue(mx.allclose(with_lora, frozen_only, atol=1e-6))

    def test_adapter_scales_linearly(self):
        train, _ = build_toy_data(1, n_train=3, n_val=1)
        x = mx.array(train)
        base = init_base(1)
        adapter = init_adapter(3, 1)
        twice = forward(base, adapter, x, scale=2.0)
        once = forward(base, adapter, x, scale=1.0)
        delta_2 = twice - forward(base, {}, x, scale=1.0)
        delta_1 = once - forward(base, {}, x, scale=1.0)
        self.assertTrue(mx.allclose(delta_2, 2 * delta_1, atol=1e-5))

    def test_only_answer_positions_produce_logits(self):
        train, _ = build_toy_data(1, n_train=3, n_val=1)
        logits = forward(init_base(1), init_adapter(2, 1), mx.array(train), scale=1.0)
        self.assertEqual(logits.shape, (3, ANSWER, VOCAB))


class ToyOptimizerTests(unittest.TestCase):
    def test_adamw_step_moves_parameters_downhill(self):
        base = init_base(2)
        train, _ = build_toy_data(2, n_train=4, n_val=2)
        x = mx.array(train)
        params = init_adapter(2, 2)

        def loss_fn(p):
            from src.mini_train import masked_cross_entropy
            return masked_cross_entropy(forward(base, p, x, 1.0), x)

        before = loss_fn(params)
        state = [{"m": mx.zeros_like(p), "v": mx.zeros_like(p)}
                 for _, p in __import__('mlx.utils', fromlist=['tree_flatten']).tree_flatten(params)]
        _, grads = mx.value_and_grad(loss_fn)(params)
        for t in range(1, 11):
            params = adamw_step(params, grads, state, t, 0.1)
        after = loss_fn(params)
        self.assertLess(float(after.item()), float(before.item()))


class ToyTrainingTests(unittest.TestCase):
    def test_lora_run_memorizes_train_but_not_val(self):
        """The toy task cannot generalize by construction; only memorization works."""
        with tempfile.TemporaryDirectory() as tmp:
            run = run_toy_training(iters=250, seed=3, output_root=tmp)
        s = run.summary()
        self.assertGreater(s["final_train_accuracy"], 0.8)
        self.assertEqual(s["final_val_accuracy"], 0.0)
        # Val loss stays near/above the guessing floor: nothing was learned to transfer.
        self.assertGreater(s["final_val_loss"], 2.5)
        self.assertLess(s["final_train_loss"], 1.0)

    def test_parameter_accounting_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = run_toy_training(iters=25, seed=4, rank=4, output_root=tmp)
            s = run.summary()
            self.assertEqual(s["adapter_parameters"], 4 * (D_HIDDEN + VOCAB))
            self.assertEqual(s["mode"], "lora")
            self.assertTrue((run.run_dir / "loss_curve.png").is_file())
            self.assertTrue((run.run_dir / "toy_history.json").is_file())
            manifest = json.loads((run.run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["config"]["rank"], 4)
            latest = json.loads((Path(tmp) / "latest_toy-training.json").read_text())
            self.assertEqual(latest["run_id"], manifest["run_id"])

    def test_full_mode_trains_all_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = run_toy_training(iters=25, seed=5, mode="full", output_root=tmp)
        self.assertEqual(run.summary()["trainable_parameters"], run.summary()["base_parameters"])

    def test_same_seed_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            a = run_toy_training(iters=100, seed=6, output_root=tmp_a)
            b = run_toy_training(iters=100, seed=6, output_root=tmp_b)
        self.assertAlmostEqual(a.summary()["final_train_loss"], b.summary()["final_train_loss"], places=6)
        self.assertAlmostEqual(a.summary()["final_val_loss"], b.summary()["final_val_loss"], places=6)

    def test_invalid_arguments_fail_fast(self):
        for bad in ({"iters": 0}, {"rank": -1}, {"lr": 0}, {"mode": "nope"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_toy_training(**bad, output_root='/tmp/never-created')

    def test_train_val_contexts_are_disjoint(self):
        train, val = build_toy_data(8)
        train_ctx = {tuple(row[:CONTEXT]) for row in train.tolist()}
        val_ctx = {tuple(row[:CONTEXT]) for row in val.tolist()}
        self.assertFalse(train_ctx.intersection(val_ctx))
        self.assertEqual(len(val), 8)


if __name__ == '__main__':
    unittest.main()
