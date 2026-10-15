"""Pure-function checks for src/grpo.py: advantages, the shared loss, and its SFT equivalence."""

import math
import unittest

import mlx.core as mx
import numpy as np

from src.dpo import sequence_logprobs
from src.grpo import group_advantages, weighted_nll_loss
from tests.test_dpo import TinyLM


class AdvantageTests(unittest.TestCase):
    def test_groups_are_standardised_and_constant_groups_give_zero(self):
        advantages = group_advantages([[1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
        self.assertAlmostEqual(advantages[0].sum(), 0.0, places=6)
        self.assertAlmostEqual(float(advantages[0][0]), 3 ** 0.5, places=2)  # (1 - .25) / std(.433)
        self.assertTrue(np.all(advantages[1] == 0))
        self.assertTrue(np.all(advantages[2] == 0))

    def test_better_than_average_gets_positive_weight(self):
        advantages = group_advantages([[1, 0, 1, 0, 0, 0]])[0]
        self.assertTrue(np.all(advantages[[0, 2]] > 0))
        self.assertTrue(np.all(advantages[[1, 3, 4, 5]] < 0))


class LossTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(3)
        self.model = TinyLM()
        self.tokens, self.offsets = [[1, 2, 3, 4], [1, 2, 5, 6, 7]], [2, 2]

    def test_unit_weights_are_per_token_cross_entropy(self):
        logps, counts = sequence_logprobs(self.model, self.tokens, self.offsets)
        loss = weighted_nll_loss(logps, counts, mx.ones(2)).item()
        expected = -float(np.mean([logps[0].item() / 2, logps[1].item() / 3]))
        self.assertAlmostEqual(loss, expected, places=5)
        self.assertGreater(loss, 0.0)

    def test_weights_scale_each_sequence_and_sign_flips_the_loss(self):
        logps, counts = sequence_logprobs(self.model, self.tokens, self.offsets)
        positive = weighted_nll_loss(logps, counts, mx.array([1.0, 1.0])).item()
        negative = weighted_nll_loss(logps, counts, mx.array([-1.0, -1.0])).item()
        self.assertAlmostEqual(positive, -negative, places=6)
        self.assertTrue(math.isfinite(weighted_nll_loss(logps, counts, mx.array([2.0, 0.0])).item()))


if __name__ == "__main__":
    unittest.main()
