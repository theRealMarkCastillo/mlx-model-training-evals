"""Reinforcement learning from a verifiable reward (GRPO-style), against SFT on the same prompts.

The experiment this module supports. Start from a deliberately under-trained SFT adapter, so
sampled outputs are right some of the time, and give two methods the same prompts in the same
order, the same optimiser and the same learning rate:

* `grpo`  Sample `group_size` completions per prompt, reward each one 1 if it matches the
          expected call exactly and 0 otherwise, and push up the completions that did better
          than their own group's average. The gold answer is used only inside the reward; the
          model never sees it as a target.
* `sft`   Cross-entropy on the gold call of each prompt (continued SFT).

Both minimise the same expression, so the only difference is where the per-sequence weight
comes from:

    loss = - mean_over_sequences( weight * (sum of token log-probs) / (number of tokens) )

with weight = group-relative advantage for GRPO and weight = 1 for SFT (which makes the loss
plain per-token cross-entropy).

GRPO in one line. For a group of rewards r_1..r_G on one prompt, the advantage of sample i is
(r_i - mean(r)) / (std(r) + eps). A group whose samples all got the same reward has zero
advantage and contributes nothing, so RL learns only from prompts where the model is sometimes
right and sometimes wrong. That is the practical limit of the method: if the correct answer is
never sampled there is no signal (the SFT adapter's `dep-NNNN` failure on unseen id formats is
like that; see the README).

Implementation choices worth knowing:

* Updates are on-policy (one gradient step per batch of rollouts), so the importance ratio is 1
  and no clipping or reference-model KL term is needed.
* Rollouts use `mlx_lm.generate.batch_generate`, which returns the sampled token ids. The loss is
  computed on those ids, not on re-tokenised text, because re-tokenising can change the tokens.
  The stop token is not returned by the generator, so it is appended when the sample ended early.
* Memory is dominated by the logits (sequences x tokens x vocabulary), so each step's sequences
  go through the model in micro-batches and the gradients are accumulated.
"""

import gc
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.datasets import ChatDataset
from rich.console import Console

from src.dataset import DATA_DIR, load_samples, positive_int
from src.dpo import freeze_all_but_lora, sequence_logprobs
from src.metrics import score_sample
from src.models import resolve_model_paths
from src.runs import (
    adapter_identity,
    file_identity,
    finish_run,
    new_run,
    record_failure,
    resolve_adapter_source,
    write_json,
)
from src.schema import parse_and_validate

console = Console()
METHODS = ("grpo", "sft")


def group_advantages(rewards, eps=1e-4):
    """Per-group standardised rewards: `rewards` is (prompts, group_size); a constant row gives zeros."""
    rewards = np.asarray(rewards, dtype=np.float64)
    centred = rewards - rewards.mean(axis=1, keepdims=True)
    return centred / (rewards.std(axis=1, keepdims=True) + eps)


def weighted_nll_loss(logps, counts, weights):
    """-mean(weight * per-token log-prob) over sequences; weight 1 everywhere is cross-entropy."""
    return -(weights * logps / counts).mean()


def _prompt_tokens(tokenizer, record):
    return tokenizer.apply_chat_template(record["messages"][:-1], tokenize=True, add_generation_prompt=True,
                                         return_dict=False)


def rollout(model, tokenizer, records, group_size, temperature, max_tokens):
    """`group_size` sampled completions per record: token sequences, prompt offsets and exact-match rewards."""
    prompts = [_prompt_tokens(tokenizer, r) for r in records]
    flat = [p for p in prompts for _ in range(group_size)]
    response = batch_generate(model, tokenizer, flat, max_tokens=max_tokens,
                              sampler=make_sampler(temp=temperature), return_token_ids=True)
    samples = []
    for i, (text, ids) in enumerate(zip(response.texts, response.token_ids, strict=True)):
        record = records[i // group_size]
        parsed = parse_and_validate(text)
        scored = score_sample(record["expected"], record["normalized_expected"], parsed)
        stopped = len(ids) < max_tokens  # the generator drops the stop token it ended on
        prompt = prompts[i // group_size]
        samples.append({
            "tokens": prompt + list(ids) + ([tokenizer.eos_token_id] if stopped else []),
            "offset": len(prompt), "reward": float(scored["param_exact"]), "valid_json": bool(parsed["is_valid_json"]),
        })
    return samples


def greedy_accuracy(model, tokenizer, records, max_tokens=150, batch=32):
    """Exact-match rate of greedy decoding, batched (fast enough to run during training)."""
    correct = 0
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        response = batch_generate(model, tokenizer, [_prompt_tokens(tokenizer, r) for r in chunk],
                                  max_tokens=max_tokens, sampler=make_sampler(temp=0.0))
        for record, text in zip(chunk, response.texts, strict=True):
            correct += score_sample(record["expected"], record["normalized_expected"], parse_and_validate(text))["param_exact"]
    return correct / len(records)


def gold_sequences(tokenizer, records):
    """SFT targets with the same template and prompt masking as `mlx_lm` training."""
    encoder = ChatDataset([], tokenizer, mask_prompt=True)
    rows = []
    for record in records:
        tokens, offset = encoder.process({"messages": record["messages"]})
        rows.append({"tokens": list(tokens), "offset": offset})
    return rows


def _loss_fn(model, tokens, offsets, weights, scale):
    logps, counts = sequence_logprobs(model, tokens, offsets)
    return weighted_nll_loss(logps, mx.maximum(counts, 1), weights) * scale


def accumulated_gradients(model, rows, weights, micro_batch):
    """Mean-over-sequences loss and gradients, computed `micro_batch` sequences at a time."""
    value_and_grad = nn.value_and_grad(model, _loss_fn)
    total, grads = 0.0, None
    for start in range(0, len(rows), micro_batch):
        chunk = rows[start:start + micro_batch]
        scale = len(chunk) / len(rows)
        loss, chunk_grads = value_and_grad(
            model, [r["tokens"] for r in chunk], [r["offset"] for r in chunk],
            mx.array(weights[start:start + micro_batch], dtype=mx.float32), scale,
        )
        grads = chunk_grads if grads is None else tree_map(lambda a, b: a + b, grads, chunk_grads)
        mx.eval(grads, loss)
        total += loss.item()
        mx.clear_cache()
    return total, grads


def plot_grpo(history, output_path, method):
    train = [h for h in history if h["type"] == "train"]
    valid = [h for h in history if h["type"] == "valid"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    axes[0].plot([h["iteration"] for h in valid], [100 * h["greedy_exact_match"] for h in valid], marker="s", color="#1b9e77")
    axes[0].set_title("Held-out greedy exact match, %", fontweight="bold")
    axes[0].set_ylim(0, 105)
    if method == "grpo":
        axes[1].plot([h["iteration"] for h in train], [h["mean_reward"] for h in train], color="#1f77b4", label="mean reward")
        axes[1].plot([h["iteration"] for h in train], [h["groups_with_signal"] for h in train], color="#d95f02",
                     label="share of groups with signal")
        axes[1].legend()
        axes[1].set_title("Rollouts", fontweight="bold")
    else:
        axes[1].plot([h["iteration"] for h in train], [h["loss"] for h in train], color="#1f77b4")
        axes[1].set_title("Train cross-entropy", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_grpo(*, preset=None, model_name=None, adapter_path=None, output_dir=None, method="grpo", data_dir=None,
             iters=50, prompts_per_step=4, group_size=8, learning_rate=2e-5, temperature=1.0, max_tokens=150,
             micro_batch=8, steps_per_eval=10, seed=42):
    """Train from `adapter_path` with `method`; returns the run directory (kind `grpo`, either method)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    for value in (iters, prompts_per_step, group_size, max_tokens, micro_batch, steps_per_eval):
        positive_int(value)
    if learning_rate <= 0 or not 0.0 < temperature <= 2.0:
        raise ValueError("learning_rate must be positive and temperature in (0, 2]")
    if method == "grpo" and group_size < 2:
        raise ValueError("GRPO needs a group of at least 2 samples per prompt")
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    model_name, adapter_path, root = resolve_model_paths(preset, model_name, adapter_path, output_dir)
    source, identity = resolve_adapter_source(model_name, adapter_path)
    train_records = load_samples(data_dir / "train.jsonl")
    valid_records = load_samples(data_dir / "valid.jsonl")
    config = {"method": method, "iters": iters, "prompts_per_step": prompts_per_step, "group_size": group_size,
              "learning_rate": learning_rate, "temperature": temperature, "max_tokens": max_tokens,
              "micro_batch": micro_batch, "steps_per_eval": steps_per_eval, "seed": seed, "data_dir": str(data_dir)}
    run_dir, manifest = new_run(
        root, "grpo", model=model_name, model_source=identity, start_adapter=adapter_identity(adapter_path),
        config=config, data={name: file_identity(data_dir / f"{name}.jsonl") for name in ("train", "valid")},
    )
    with record_failure(run_dir, manifest):
        mx.random.seed(seed)
        np.random.seed(seed)
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        model, tokenizer = mlx_lm.load(source, adapter_path=adapter_path)
        trainable = freeze_all_but_lora(model, adapter_path)
        model.update(tree_map(lambda p: p.astype(mx.float32), model.trainable_parameters()))  # see run_dpo
        model.eval()  # no dropout: the policy that generated a sample is the policy that is updated
        if len(train_records) < prompts_per_step:
            raise ValueError(f"{len(train_records)} training prompts is fewer than one step's {prompts_per_step}")
        console.print(f"[bold blue]{method.upper()}[/bold blue] {len(train_records)} train prompts, "
                      f"{sum(v.size for _, v in trainable):,} trainable parameters, lr {learning_rate}")
        sft_rows = gold_sequences(tokenizer, train_records) if method == "sft" else None
        optimizer = optim.AdamW(learning_rate=learning_rate)
        history = []

        def validate(model, tokenizer, iteration):
            accuracy = greedy_accuracy(model, tokenizer, valid_records, max_tokens)
            history.append({"type": "valid", "iteration": iteration, "greedy_exact_match": accuracy})
            console.print(f"  iter {iteration:4d}  valid greedy exact match {100 * accuracy:.1f}%")

        validate(model, tokenizer, 0)
        order, cursor = np.random.permutation(len(train_records)), 0
        for iteration in range(1, iters + 1):
            if cursor + prompts_per_step > len(order):
                order, cursor = np.random.permutation(len(train_records)), 0
            index = order[cursor:cursor + prompts_per_step]
            cursor += prompts_per_step
            entry = {"type": "train", "iteration": iteration}
            if method == "sft":
                rows, weights = [sft_rows[i] for i in index], [1.0] * len(index)
            else:
                samples = rollout(model, tokenizer, [train_records[i] for i in index], group_size, temperature, max_tokens)
                rewards = np.array([s["reward"] for s in samples]).reshape(len(index), group_size)
                advantages = group_advantages(rewards)
                keep = np.abs(advantages.reshape(-1)) > 1e-6
                rows = [s for s, k in zip(samples, keep, strict=True) if k]
                weights = advantages.reshape(-1)[keep].tolist()
                entry.update(mean_reward=float(rewards.mean()), valid_json=float(np.mean([s["valid_json"] for s in samples])),
                             groups_with_signal=float((rewards.std(axis=1) > 0).mean()))
            if rows:
                loss, grads = accumulated_gradients(model, rows, weights, micro_batch)
                grads, _ = optim.clip_grad_norm(grads, 1.0)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                del grads
            else:
                loss = 0.0  # every group had identical rewards: nothing to learn from this step
            mx.clear_cache()
            entry["loss"] = loss
            history.append(entry)
            if iteration % steps_per_eval == 0 or iteration == iters:
                recent = [h for h in history if h["type"] == "train"][-steps_per_eval:]
                note = f"  reward {np.mean([h['mean_reward'] for h in recent]):.2f}" if method == "grpo" else ""
                console.print(f"  iter {iteration:4d}  train loss {np.mean([h['loss'] for h in recent]):.4f}{note}  "
                              f"peak memory {mx.get_peak_memory() / 1e9:.1f} GB")
                validate(model, tokenizer, iteration)

        adapter_dir = run_dir / "adapters"
        adapter_dir.mkdir()
        mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        shutil.copy(Path(adapter_path) / "adapter_config.json", adapter_dir / "adapter_config.json")
        valid_history = [h for h in history if h["type"] == "valid"]
        summary = {
            "initial_valid_exact_match": valid_history[0]["greedy_exact_match"],
            "final_valid_exact_match": valid_history[-1]["greedy_exact_match"],
            "training_time_seconds": round(time.perf_counter() - started, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 ** 2), 2),
        }
        write_json(run_dir / "grpo_history.json", {"config": config, "summary": summary, "history": history})
        plot_grpo(history, run_dir / "grpo_curves.png", method)
        manifest["adapter"] = adapter_identity(adapter_dir)
        manifest["adapter_path"] = str(adapter_dir)
        manifest["result"] = summary
        console.print(f"Adapter: {adapter_dir}")
        finish_run(root, run_dir, manifest)
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
    return run_dir
