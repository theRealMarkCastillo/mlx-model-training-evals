"""Mechanics tests for preference tuning (src/dpo.py) and adapter comparison (src/compare.py).

The DPO math is checked on a tiny stand-in language model, so no downloads are needed.
"""
import json
import math
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from src.compare import parse_named_adapters
from src.dataset import DATA_DIR
from src.dpo import (
    all_evaluation_prompts,
    build_mining_prompts,
    dpo_loss,
    freeze_all_but_lora,
    indistribution_pairs,
    make_loss_fn,
    sequence_logprobs,
    split_pairs,
)
from src.generate_data import MINING_POOLS, SEEN, UNSEEN

VOCAB = 11


class TinyLM(nn.Module):
    """Embedding + two causal-free linear layers: enough for log-prob bookkeeping."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 8)
        self.out = nn.Linear(8, VOCAB)

    def __call__(self, tokens):
        return self.out(nn.relu(self.embed(tokens)))


def manual_logprob(model, tokens, offset):
    logits = model(mx.array([tokens[:-1]]))[0]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return sum(float(logp[i - 1, tokens[i]].item()) for i in range(offset, len(tokens)))


class SequenceLogprobTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)
        self.model = TinyLM()
        self.a, self.b = [1, 2, 3, 4, 5, 6], [7, 8, 9, 3]

    def test_matches_a_hand_computed_sum_over_assistant_tokens(self):
        logps, counts = sequence_logprobs(self.model, [self.a], [3])
        self.assertAlmostEqual(logps[0].item(), manual_logprob(self.model, self.a, 3), places=4)
        self.assertEqual(counts[0].item(), 3)

    def test_padding_does_not_change_a_shorter_sequence(self):
        alone, _ = sequence_logprobs(self.model, [self.b], [2])
        batched, _ = sequence_logprobs(self.model, [self.a, self.b], [3, 2])
        self.assertAlmostEqual(alone[0].item(), batched[1].item(), places=4)

    def test_prompt_tokens_are_not_scored(self):
        early, _ = sequence_logprobs(self.model, [self.a], [1])
        late, _ = sequence_logprobs(self.model, [self.a], [5])
        self.assertGreater(late[0].item(), early[0].item())  # fewer terms, each a negative log-prob


class DpoLossTests(unittest.TestCase):
    def test_policy_equal_to_reference_gives_ln2(self):
        x = mx.array([-3.0, -5.0])
        loss, stats = dpo_loss(x, x - 1.0, x, x - 1.0, beta=0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=5)
        self.assertEqual(stats[0].item(), 0.0)  # margin is exactly 0, which is not "chosen preferred"

    def test_raising_chosen_over_rejected_lowers_the_loss_and_scores_accurate(self):
        ref = mx.array([-4.0, -4.0])
        better, stats = dpo_loss(ref + 2.0, ref - 2.0, ref, ref, beta=0.5)
        worse, bad_stats = dpo_loss(ref - 2.0, ref + 2.0, ref, ref, beta=0.5)
        self.assertLess(better.item(), math.log(2))
        self.assertGreater(worse.item(), math.log(2))
        self.assertEqual(stats[0].item(), 1.0)
        self.assertEqual(bad_stats[0].item(), 0.0)
        self.assertAlmostEqual(stats[2].item(), 2.0, places=5)
        self.assertAlmostEqual(stats[3].item(), -2.0, places=5)

    def test_closed_form_value(self):
        # margin = beta * ((c - rc) - (r - rr)) = 0.1 * (3 - 1) = 0.2; loss = -log sigmoid(0.2)
        loss, _ = dpo_loss(mx.array([-1.0]), mx.array([-5.0]), mx.array([-4.0]), mx.array([-6.0]), beta=0.1)
        self.assertAlmostEqual(loss.item(), -math.log(1 / (1 + math.exp(-0.2))), places=5)

    def test_gradient_step_on_the_loss_raises_the_chosen_margin(self):
        mx.random.seed(1)
        model = TinyLM()
        chosen, rejected = [1, 2, 3, 4], [1, 2, 5, 6]
        tokens, offsets = [chosen, rejected], [2, 2]
        ref, _ = sequence_logprobs(model, tokens, offsets)
        mx.eval(ref)
        loss_fn = make_loss_fn(beta=1.0, sft_alpha=0.0)
        before = loss_fn(model, tokens, offsets, ref[:1], ref[1:])[0].item()
        self.assertAlmostEqual(before, math.log(2), places=4)
        grad_fn = nn.value_and_grad(model, loss_fn)
        for _ in range(20):
            (_, _), grads = grad_fn(model, tokens, offsets, ref[:1], ref[1:])
            model.update(tree_map(lambda p, g: p - 0.1 * g, model.parameters(), grads))
            mx.eval(model.parameters())
        after = loss_fn(model, tokens, offsets, ref[:1], ref[1:])[0].item()
        self.assertLess(after, before)

    def test_sft_only_drops_the_preference_term_and_keeps_the_anchor(self):
        mx.random.seed(1)
        model = TinyLM()
        tokens, offsets = [[1, 2, 3, 4], [1, 2, 5, 6]], [2, 2]
        logps, counts = sequence_logprobs(model, tokens, offsets)
        ref_chosen, ref_rejected = logps[:1] - 3.0, logps[1:] + 3.0  # a non-trivial preference loss
        nll = (-(logps[:1] / counts[:1])).mean().item()
        both = make_loss_fn(0.5, 1.0)(model, tokens, offsets, ref_chosen, ref_rejected)[0].item()
        anchor_only = make_loss_fn(0.5, 1.0, preference=False)(model, tokens, offsets, ref_chosen, ref_rejected)[0].item()
        self.assertAlmostEqual(anchor_only, nll, places=5)
        self.assertGreater(both, anchor_only)


class PairDataTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_nonempty(self):
        pairs = [{"id": f"p{i}"} for i in range(20)]
        train, valid = split_pairs(pairs)
        self.assertEqual((train, valid), split_pairs(list(reversed(pairs))))
        self.assertFalse({p["id"] for p in train} & {p["id"] for p in valid})
        self.assertEqual(len(train) + len(valid), 20)
        self.assertGreater(len(valid), 0)

    def test_mining_prompts_avoid_every_evaluation_prompt_and_pool(self):
        samples = build_mining_prompts(12, 6)
        prompts = [s["prompt"] for s in samples]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertFalse(set(prompts) & all_evaluation_prompts(DATA_DIR))
        self.assertEqual(len(samples), 18)
        self.assertTrue(all(s["meta"]["entities"] == "mining" for s in samples))
        # Mined entities are new; none of them may appear in the SEEN or UNSEEN pools.
        known = set(SEEN.services) | set(UNSEEN.services)
        for pool in MINING_POOLS:
            self.assertFalse(set(pool.services) & known)
        self.assertEqual(build_mining_prompts(12, 6), samples)  # deterministic

    def test_deployment_id_formats_differ_from_training_and_challenge(self):
        formats = {SEEN.deployment_id, UNSEEN.deployment_id}
        self.assertFalse({pool.deployment_id for pool in MINING_POOLS} & formats)


class FreezeTests(unittest.TestCase):
    def build(self):
        from mlx_lm.tuner.lora import LoRALinear

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = LoRALinear.from_base(nn.Linear(8, 8, bias=True), r=2, scale=1.0, dropout=0.0)
                self.norm = nn.RMSNorm(8)

        return Block()

    def test_only_lora_matrices_stay_trainable_and_names_are_checked(self):
        import tempfile
        from pathlib import Path

        from mlx.utils import tree_flatten

        model = self.build()  # nothing frozen yet: the state mlx_lm.load leaves a model in
        self.assertGreater(len(tree_flatten(model.trainable_parameters())), 2)
        with tempfile.TemporaryDirectory() as directory:
            names = {"proj.lora_a": mx.zeros((8, 2)), "proj.lora_b": mx.zeros((2, 8))}
            mx.save_safetensors(str(Path(directory) / "adapters.safetensors"), names)
            trainable = freeze_all_but_lora(model, directory)
            self.assertEqual({n for n, _ in trainable}, {"proj.lora_a", "proj.lora_b"})
            mx.save_safetensors(str(Path(directory) / "adapters.safetensors"), {"other.lora_a": mx.zeros((8, 2))})
            with self.assertRaises(RuntimeError):
                freeze_all_but_lora(model, directory)


class CompareArgumentTests(unittest.TestCase):
    def test_named_adapters_parse_in_order(self):
        self.assertEqual(parse_named_adapters(["sft=a", "dpo=b"]), [("sft", "a"), ("dpo", "b")])

    def test_bad_specs_are_rejected(self):
        for specs in (["sft=a"], ["sft", "dpo=b"], ["sft=a", "sft=b"], ["base=a", "dpo=b"]):
            with self.assertRaises(ValueError):
                parse_named_adapters(specs)


if __name__ == "__main__":
    unittest.main()


class InDistributionPairTests(unittest.TestCase):
    def test_pairs_swap_only_the_id_into_a_mining_format(self):
        pairs = indistribution_pairs(12)
        self.assertEqual(len(pairs), 12)
        for pair in pairs:
            chosen, rejected = json.loads(pair["chosen"]), json.loads(pair["rejected"])
            self.assertTrue(chosen["parameters"]["deployment_id"].startswith("dep-"))
            self.assertFalse(rejected["parameters"]["deployment_id"].startswith("dep-"))
            chosen["parameters"].pop("deployment_id")
            rejected["parameters"].pop("deployment_id")
            self.assertEqual(chosen, rejected)

    def test_prompts_are_disjoint_from_evaluation_sets(self):
        seen = all_evaluation_prompts()
        for pair in indistribution_pairs(12):
            self.assertNotIn(pair["messages"][1]["content"], seen)
