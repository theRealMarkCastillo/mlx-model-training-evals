"""Preference tuning (DPO) on top of the SFT adapter, targeting its measured failures.

The experiment this module supports:

1. `mine`  Run the SFT adapter over fresh prompts (new entities, new deployment-id formats,
           disjoint from every evaluation set). Wherever it is wrong, record a preference
           pair: chosen = the gold call, rejected = what the model actually wrote.
2. `dpo`   Continue training the same LoRA adapter with the DPO loss on those pairs.
3. control Plain SFT on the *same prompts' gold calls* (`sft_plus/` data written by `mine`),
           so "DPO vs more SFT data" is a fair comparison: same prompts, different loss.

DPO in one line. For a pair (prompt x, chosen y_w, rejected y_l) and a frozen reference
policy (here: the SFT adapter at step 0):

    loss = -log sigmoid( beta * [ (log pi(y_w|x) - log pi(y_l|x)) - (log ref(y_w|x) - log ref(y_l|x)) ] )

It pushes the policy to raise the chosen response's probability *relative to the reference*
by more than it raises the rejected one's. `beta` is the leash: a small beta lets the policy
drift far from the reference, a large beta keeps it close.

Two implementation choices worth knowing:

* The reference is the policy at step 0, so its log-probabilities are computed once before
  training and stored. No second model is held in memory. This requires dropout to be off
  during DPO (`model.eval()`), otherwise the policy at step 0 would not equal the reference.
  A consequence you can check: the loss at step 0 is exactly ln 2 = 0.6931.
* Sequence log-probability is summed over the assistant tokens only, with the same chat
  template and prompt masking as SFT (`ChatDataset.process`), so "what is scored" is the
  same thing `main.py show-mask` displays.
"""

import gc
import json
import math
import random
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
from mlx_lm.tuner.datasets import ChatDataset
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.dataset import DATA_DIR, challenge_files, load_samples, positive_int
from src.generate_data import (
    MINING_POOLS,
    TRAIN_FAMILIES,
    build_samples,
    save_jsonl,
    to_chat_format,
)
from src.inference import generate_response
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
MINING_SEED = 1234  # disjoint from the dataset seed (42) so mined prompts are a different draw
OTHER_TOOLS = ("deploy_service", "restart_pod", "scale_cluster", "no_action")


# --------------------------------------------------------------------------------------
# Pair mining
# --------------------------------------------------------------------------------------

def all_evaluation_prompts(data_dir=DATA_DIR):
    """Every prompt in every standard split and challenge set: mined prompts must avoid all of them."""
    paths = [Path(data_dir) / f"{name}.jsonl" for name in ("train", "valid", "test")]
    paths += list(challenge_files(data_dir).values())
    return {record["prompt"] for path in paths for record in load_samples(path)}


def build_mining_prompts(rollback_count, other_count, *, seed=MINING_SEED, data_dir=DATA_DIR):
    """Fresh prompts from the mining pools, rollback-heavy because that is where the SFT adapter fails.

    Wording families are the training ones (0, 1): the experiment varies *entities and id
    formats*, not phrasing, so the held-out-wording test set (family 3) stays untouched.
    """
    positive_int(rollback_count)
    positive_int(other_count)
    rng = random.Random(seed)
    seen = all_evaluation_prompts(data_dir)
    overlap_guard = len(seen)
    samples = []
    for index, pool in enumerate(MINING_POOLS):
        # Split each total across the pools; the first pools take the remainder.
        rollback_share = rollback_count // len(MINING_POOLS) + (index < rollback_count % len(MINING_POOLS))
        other_share = other_count // len(MINING_POOLS) + (index < other_count % len(MINING_POOLS))
        samples += build_samples(rng, rollback_share, seen, families=TRAIN_FAMILIES, pool=pool,
                                 tools=("rollback_deployment",))
        samples += build_samples(rng, other_share, seen, families=TRAIN_FAMILIES, pool=pool, tools=OTHER_TOOLS)
    for sample in samples:
        sample["meta"]["entities"] = "mining"
    # build_samples adds each new prompt to `seen`; nothing it generated may have been there already.
    if len(seen) != overlap_guard + len(samples):
        raise ValueError("Mined prompts collided with each other or with an evaluation set")
    rng.shuffle(samples)
    return samples


def mine_pairs(model, tokenizer, samples, *, max_tokens=150, extra_draws=0, temperature=0.8, seed=MINING_SEED):
    """Preference pairs from the model's own mistakes on `samples` (chat-format records).

    The first draw is greedy, so the rejected text is what the model would really answer.
    If it is right and `extra_draws > 0`, sampled draws look for a wrong alternative; this
    surfaces second-best behaviour at the cost of rejected text the model rarely emits.
    """
    pairs, attempts = [], []
    mx.random.seed(seed)
    for item in tqdm(samples, desc="mining", leave=False):
        messages = item["messages"][:-1]
        gold = item["messages"][-1]["content"]
        expected = item["expected"]
        draws = [("greedy", 0.0)] + [("sampled", temperature)] * extra_draws
        rejected = None
        for source, temp in draws:
            output = generate_response(model, tokenizer, messages, max_tokens, temperature=temp)["raw_output"]
            parsed = parse_and_validate(output)
            scored = score_sample(expected, item["normalized_expected"], parsed)
            if source == "greedy":
                attempts.append({"id": item["id"], "tool": expected["tool"], "greedy_correct": scored["param_exact"]})
            if not scored["param_exact"] and output.strip():
                rejected = (output, source, scored["error_category"], scored["wrong_fields"])
                break
        if rejected:
            output, source, category, wrong_fields = rejected
            pairs.append({
                "id": item["id"], "messages": messages, "chosen": gold, "rejected": output,
                "rejected_source": source, "error_category": category, "wrong_fields": wrong_fields,
                "meta": item["meta"],
            })
    return pairs, attempts


def split_pairs(pairs, valid_fraction=0.15, seed=MINING_SEED):
    """Deterministic train/valid split of pairs; valid pairs are never trained on."""
    ordered = sorted(pairs, key=lambda p: p["id"])
    random.Random(seed).shuffle(ordered)
    n_valid = max(1, round(len(ordered) * valid_fraction)) if len(ordered) >= 4 else 0
    return ordered[n_valid:], ordered[:n_valid]


def write_control_data(directory, train_pairs, data_dir=DATA_DIR):
    """SFT control: the original data plus the gold call for every DPO-train prompt.

    Same prompts DPO sees, trained with plain cross-entropy instead of a preference loss.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for path in [Path(data_dir) / f"{name}.jsonl" for name in ("valid", "test")] + list(challenge_files(data_dir).values()):
        shutil.copy(path, directory / path.name)
    extra = [{"messages": [*p["messages"], {"role": "assistant", "content": p["chosen"]}], "meta": p["meta"]}
             for p in train_pairs]
    original = [json.loads(line) for line in (Path(data_dir) / "train.jsonl").read_text().splitlines() if line.strip()]
    rng = random.Random(MINING_SEED)
    combined = original + extra
    rng.shuffle(combined)
    save_jsonl(combined, directory / "train.jsonl")
    return len(original), len(extra)


def run_mining(*, preset=None, model_name=None, adapter_path=None, output_dir=None, rollback_count=240,
               other_count=60, extra_draws=0, temperature=0.8, max_tokens=150, valid_fraction=0.15):
    """Mine pairs with the given SFT adapter and write them (plus the SFT control data) to a run directory."""
    model_name, adapter_path, root = resolve_model_paths(preset, model_name, adapter_path, output_dir)
    source, identity = resolve_adapter_source(model_name, adapter_path)
    samples = load_mined_records(build_mining_prompts(rollback_count, other_count))
    run_dir, manifest = new_run(
        root, "mine", model=model_name, model_source=identity, adapter=adapter_identity(adapter_path),
        settings={"rollback_count": rollback_count, "other_count": other_count, "extra_draws": extra_draws,
                  "temperature": temperature, "max_tokens": max_tokens, "seed": MINING_SEED},
        evaluation_sets={p.name: file_identity(p) for p in
                         [DATA_DIR / "train.jsonl", DATA_DIR / "test.jsonl", *challenge_files(DATA_DIR).values()]},
    )
    with record_failure(run_dir, manifest):
        model, tokenizer = mlx_lm.load(source, adapter_path=adapter_path)
        try:
            pairs, attempts = mine_pairs(model, tokenizer, samples, max_tokens=max_tokens,
                                         extra_draws=extra_draws, temperature=temperature)
        finally:
            del model, tokenizer
            gc.collect()
            mx.clear_cache()
        if len(pairs) < 8:
            raise ValueError(f"Only {len(pairs)} preference pairs mined from {len(samples)} prompts: "
                             "the adapter is too accurate here to train on. Mine more prompts or add --extra-draws.")
        train_pairs, valid_pairs = split_pairs(pairs, valid_fraction)
        save_jsonl(train_pairs, run_dir / "pairs_train.jsonl")
        save_jsonl(valid_pairs, run_dir / "pairs_valid.jsonl")
        original, extra = write_control_data(run_dir / "sft_plus", train_pairs)

        by_tool = {}
        for attempt in attempts:
            row = by_tool.setdefault(attempt["tool"], {"prompts": 0, "greedy_wrong": 0})
            row["prompts"] += 1
            row["greedy_wrong"] += not attempt["greedy_correct"]
        report = {
            "prompts": len(samples), "pairs": len(pairs), "train_pairs": len(train_pairs),
            "valid_pairs": len(valid_pairs), "greedy_wrong_by_tool": by_tool,
            "rejected_source": {s: sum(p["rejected_source"] == s for p in pairs) for s in ("greedy", "sampled")},
            "error_categories": {c: sum(p["error_category"] == c for p in pairs)
                                 for c in sorted({p["error_category"] for p in pairs})},
            "control_train_records": {"original": original, "added": extra},
            "examples": [{k: p[k] for k in ("messages", "chosen", "rejected", "wrong_fields")} for p in pairs[:3]],
        }
        write_json(run_dir / "mining_report.json", report)
        print_mining_report(report)
        manifest["result"] = {k: report[k] for k in ("prompts", "pairs", "train_pairs", "valid_pairs")}
        finish_run(root, run_dir, manifest)
    return run_dir


def indistribution_pairs(count, *, seed=MINING_SEED + 1, data_dir=DATA_DIR):
    """Synthetic pairs for rollback prompts in the *training* id format (`dep-NNNN`).

    The SFT adapter already answers these correctly, so there is no mistake to mine. The
    rejected text is the gold call with the id rewritten into a mining-pool format
    (`{svc}-build{n}`): the opposite error to the mined pairs, so preference tuning cannot
    reduce the problem to "never write dep-".
    """
    positive_int(count)
    rng = random.Random(seed)
    samples = build_samples(rng, count, all_evaluation_prompts(data_dir), families=TRAIN_FAMILIES,
                            tools=("rollback_deployment",))
    pairs = []
    for index, record in enumerate(to_chat_format(samples)):
        gold = json.loads(record["messages"][-1]["content"])
        wrong = json.loads(record["messages"][-1]["content"])
        pool = rng.choice(MINING_POOLS)
        wrong["parameters"]["deployment_id"] = pool.deployment_id.format(n=rng.randint(1000, 9999),
                                                                         svc=rng.choice(pool.services))
        if wrong == gold:
            continue
        pairs.append({
            "id": f"indist:{index}", "messages": record["messages"][:-1],
            "chosen": record["messages"][-1]["content"], "rejected": json.dumps(wrong, separators=(",", ":")),
            "rejected_source": "synthetic", "error_category": "wrong_id_format",
            "wrong_fields": ["deployment_id"], "meta": record["meta"],
        })
    return pairs


def run_mix(*, source=None, output_dir=None, preset=None, model_name=None, count=120, valid_fraction=0.15):
    """Copy a mining run and add in-distribution pairs (and their gold calls to the SFT control data)."""
    _, _, root = resolve_model_paths(preset, model_name, None, output_dir, need_adapter=False)
    source = resolve_pairs_dir(source, root)
    extra = indistribution_pairs(count)
    extra_train, extra_valid = split_pairs(extra, valid_fraction)
    train = read_jsonl(source / "pairs_train.jsonl") + extra_train
    valid = read_jsonl(source / "pairs_valid.jsonl") + extra_valid
    run_dir, manifest = new_run(root, "mine", derived_from=str(source), indistribution_pairs=len(extra))
    with record_failure(run_dir, manifest):
        save_jsonl(train, run_dir / "pairs_train.jsonl")
        save_jsonl(valid, run_dir / "pairs_valid.jsonl")
        original, added = write_control_data(run_dir / "sft_plus", train)
        report = {"derived_from": str(source), "train_pairs": len(train), "valid_pairs": len(valid),
                  "indistribution_pairs": len(extra), "control_train_records": {"original": original, "added": added}}
        write_json(run_dir / "mining_report.json", report)
        manifest["result"] = report
        finish_run(root, run_dir, manifest)
    console.print(f"Wrote {len(train)} train / {len(valid)} valid pairs ({len(extra)} in-distribution) to {run_dir}")
    return run_dir


def load_mined_records(samples):
    """Turn generator samples into the record shape `load_samples` produces (with parsed expectations)."""
    records = []
    for index, record in enumerate(to_chat_format(samples)):
        target = parse_and_validate(record["messages"][-1]["content"])
        records.append({
            "id": f"mining:{index}", "messages": record["messages"], "prompt": record["messages"][1]["content"],
            "expected": target["parsed_data"], "normalized_expected": target["normalized_data"],
            "meta": record["meta"],
        })
    return records


def print_mining_report(report):
    table = Table(title="Preference-pair mining", header_style="bold magenta")
    for column in ("Tool", "Prompts", "Greedy wrong", "Rate"):
        table.add_column(column, justify="right" if column != "Tool" else "left")
    for tool, row in sorted(report["greedy_wrong_by_tool"].items()):
        table.add_row(tool, str(row["prompts"]), str(row["greedy_wrong"]), f"{100 * row['greedy_wrong'] / row['prompts']:.0f}%")
    console.print(table)
    console.print(f"{report['pairs']} pairs from {report['prompts']} prompts "
                  f"({report['train_pairs']} train / {report['valid_pairs']} valid); "
                  f"rejected from greedy: {report['rejected_source']['greedy']}, sampled: {report['rejected_source']['sampled']}")
    console.print(f"Error categories: {report['error_categories']}")


# --------------------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------------------

def sequence_logprobs(model, token_lists, offsets):
    """Sum of log p(token) over each sequence's assistant tokens (positions offset .. len-1)."""
    lengths = mx.array([len(t) for t in token_lists])
    width = max(len(t) for t in token_lists)
    batch = mx.array([t + [0] * (width - len(t)) for t in token_lists])  # right padding is invisible to causal attention
    logits = model(batch[:, :-1]).astype(mx.float32)
    logp = -nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
    position = mx.arange(1, width)[None, :]  # index in `batch` of the token each column predicts
    scored = (position >= mx.array(offsets)[:, None]) & (position < lengths[:, None])
    return (logp * scored).sum(axis=1), scored.sum(axis=1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """Mean DPO loss and its diagnostics, from the four sequence log-probabilities.

    Returns (loss, stats) with stats = [reward accuracy, mean reward margin, mean change in the
    chosen log-prob vs the reference, same for rejected]. The "implicit reward" of a response
    is beta * (log pi - log ref).
    """
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -nn.log_sigmoid(margin).mean()
    stats = mx.stack([(margin > 0).astype(mx.float32).mean(), margin.mean(),
                      (policy_chosen - ref_chosen).mean(), (policy_rejected - ref_rejected).mean()])
    return loss, stats


def encode_pairs(tokenizer, pairs, max_seq_length):
    """Tokenize pairs with SFT's template and prompt masking. Over-long pairs are dropped and counted."""
    encoder = ChatDataset([], tokenizer, mask_prompt=True)
    encoded, dropped = [], 0
    for pair in pairs:
        row = {}
        for side in ("chosen", "rejected"):
            tokens, offset = encoder.process({"messages": [*pair["messages"], {"role": "assistant", "content": pair[side]}]})
            row[side] = (list(tokens), offset)
        if max(len(row["chosen"][0]), len(row["rejected"][0])) > max_seq_length or \
                min(len(row["chosen"][0]) - row["chosen"][1], len(row["rejected"][0]) - row["rejected"][1]) < 1:
            dropped += 1
            continue
        encoded.append(row)
    if not encoded:
        raise ValueError("No usable preference pairs after tokenization")
    return encoded, dropped


def _stack(rows):
    """One flat sequence list for a batch: all chosen first, then all rejected."""
    tokens = [r["chosen"][0] for r in rows] + [r["rejected"][0] for r in rows]
    offsets = [r["chosen"][1] for r in rows] + [r["rejected"][1] for r in rows]
    return tokens, offsets


def reference_logprobs(model, encoded, batch_size):
    """Frozen-reference (step-0 policy) log-probs for every pair, computed once."""
    model.eval()
    chosen, rejected = [], []
    for start in range(0, len(encoded), batch_size):
        rows = encoded[start:start + batch_size]
        logps, _ = sequence_logprobs(model, *_stack(rows))
        mx.eval(logps)
        chosen += logps[:len(rows)].tolist()
        rejected += logps[len(rows):].tolist()
    return mx.array(chosen), mx.array(rejected)


def make_loss_fn(beta, sft_alpha, preference=True):
    """DPO loss plus an optional NLL anchor; `preference=False` keeps only the anchor (the SFT ablation).

    The preference statistics are still computed so both variants log the same curves.
    """
    def loss_fn(model, tokens, offsets, ref_chosen, ref_rejected):
        logps, counts = sequence_logprobs(model, tokens, offsets)
        n = ref_chosen.shape[0]
        loss, stats = dpo_loss(logps[:n], logps[n:], ref_chosen, ref_rejected, beta)
        if not preference:
            loss = loss * 0.0
        if sft_alpha:
            # Optional NLL anchor on the chosen response (RPO-style) against likelihood displacement.
            loss = loss + sft_alpha * (-(logps[:n] / counts[:n])).mean()
        return loss, stats
    return loss_fn


def evaluate_pairs(model, encoded, ref_chosen, ref_rejected, loss_fn, batch_size):
    model.eval()
    total, stats, count = 0.0, np.zeros(4), 0
    for start in range(0, len(encoded), batch_size):
        rows = encoded[start:start + batch_size]
        loss, batch_stats = loss_fn(model, *_stack(rows), ref_chosen[start:start + len(rows)],
                                    ref_rejected[start:start + len(rows)])
        mx.eval(loss, batch_stats)
        total += float(loss.item()) * len(rows)
        stats += np.array(batch_stats.tolist()) * len(rows)
        count += len(rows)
    return total / count, (stats / count).tolist()


STAT_NAMES = ("reward_accuracy", "reward_margin", "chosen_logp_delta", "rejected_logp_delta")


def plot_dpo(history, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    train = [h for h in history if h["type"] == "train"]
    valid = [h for h in history if h["type"] == "valid"]
    axes[0].plot([h["iteration"] for h in train], [h["loss"] for h in train], label="train", color="#1f77b4")
    axes[0].plot([h["iteration"] for h in valid], [h["loss"] for h in valid], label="valid", color="#ff7f0e", marker="s")
    axes[0].axhline(math.log(2), color="gray", linestyle=":", label="ln 2 (no preference)")
    axes[0].set_title("DPO loss", fontweight="bold")
    axes[1].plot([h["iteration"] for h in valid], [h["reward_accuracy"] for h in valid], marker="s", color="#1b9e77")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Valid reward accuracy\n(chosen implicit reward > rejected)", fontweight="bold")
    axes[2].plot([h["iteration"] for h in valid], [h["chosen_logp_delta"] for h in valid], marker="o", label="chosen", color="#1b9e77")
    axes[2].plot([h["iteration"] for h in valid], [h["rejected_logp_delta"] for h in valid], marker="o", label="rejected", color="#d95f02")
    axes[2].axhline(0, color="gray", linestyle=":")
    axes[2].set_title("Valid change in log-prob vs reference", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(linestyle="--", alpha=0.5)
    axes[0].legend()
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def freeze_all_but_lora(model, adapter_path):
    """Make exactly the adapter's LoRA matrices trainable, and return them as (name, array) pairs.

    `mlx_lm.load(..., adapter_path=...)` leaves the base model's own biases and norm weights
    trainable (SFT calls `model.freeze()` before adding LoRA layers; loading does not). Training
    those would silently turn LoRA preference tuning into something else, and the saved file
    would no longer be a LoRA adapter. The key set is checked against the SFT adapter file.
    """
    model.freeze()
    for _, module in model.named_modules():
        if hasattr(module, "lora_a"):
            module.unfreeze(keys=["lora_a", "lora_b"], strict=False)
    trainable = tree_flatten(model.trainable_parameters())
    expected = set(mx.load(str(Path(adapter_path) / "adapters.safetensors")))
    if {name for name, _ in trainable} != expected:
        raise RuntimeError("Trainable parameters differ from the SFT adapter's: "
                           f"{len({n for n, _ in trainable} ^ expected)} names do not match")
    return trainable


def resolve_pairs_dir(pairs, root):
    """An explicit pairs directory, or the latest completed `mine` run under the report root."""
    if pairs:
        directory = Path(pairs)
    else:
        pointer = Path(root) / "latest_mine.json"
        if not pointer.is_file():
            raise FileNotFoundError(f"No completed mining run at {pointer}. Run `main.py dpo-mine` first or pass --pairs.")
        directory = Path(json.loads(pointer.read_text())["run_dir"])
    for name in ("pairs_train.jsonl", "pairs_valid.jsonl"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing {directory / name}")
    return directory


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_dpo(*, preset=None, model_name=None, adapter_path=None, output_dir=None, pairs=None, iters=100,
            batch_size=4, learning_rate=5e-6, beta=0.1, sft_alpha=0.0, steps_per_eval=10, seed=42, preference=True,
            max_seq_length=1024):
    """Continue training the SFT adapter with the DPO loss on mined pairs; returns the run directory."""
    for value in (iters, batch_size, steps_per_eval):
        positive_int(value)
    if not preference and not sft_alpha:
        raise ValueError("--sft-only needs --sft-alpha > 0: with no preference term there would be no loss")
    if beta <= 0 or learning_rate <= 0 or sft_alpha < 0:
        raise ValueError("beta and learning_rate must be positive and sft_alpha non-negative")
    model_name, adapter_path, root = resolve_model_paths(preset, model_name, adapter_path, output_dir)
    pairs_dir = resolve_pairs_dir(pairs, root)
    source, identity = resolve_adapter_source(model_name, adapter_path)
    train_pairs, valid_pairs = read_jsonl(pairs_dir / "pairs_train.jsonl"), read_jsonl(pairs_dir / "pairs_valid.jsonl")
    config = {"iters": iters, "batch_size": batch_size, "learning_rate": learning_rate, "beta": beta,
              "sft_alpha": sft_alpha, "preference": preference, "steps_per_eval": steps_per_eval, "seed": seed, "max_seq_length": max_seq_length}
    run_dir, manifest = new_run(
        root, "dpo", model=model_name, model_source=identity, sft_adapter=adapter_identity(adapter_path),
        config=config, pairs={name: file_identity(pairs_dir / f"{name}.jsonl") for name in ("pairs_train", "pairs_valid")},
    )
    with record_failure(run_dir, manifest):
        mx.random.seed(seed)
        np.random.seed(seed)
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        model, tokenizer = mlx_lm.load(source, adapter_path=adapter_path)
        trainable = freeze_all_but_lora(model, adapter_path)
        # A saved adapter loads back in float16. AdamW's epsilon (1e-8) underflows to 0 there, so an
        # update with a zero second moment is 0/0 = NaN. Training the LoRA matrices in float32 avoids it.
        model.update(tree_map(lambda p: p.astype(mx.float32), model.trainable_parameters()))
        train_set, dropped = encode_pairs(tokenizer, train_pairs, max_seq_length)
        valid_set, _ = encode_pairs(tokenizer, valid_pairs, max_seq_length)
        if len(train_set) < batch_size:
            raise ValueError(f"{len(train_set)} training pairs is fewer than one batch of {batch_size}")
        console.print(f"[bold blue]DPO[/bold blue] {len(train_set)} train / {len(valid_set)} valid pairs "
                      f"({dropped} dropped), {sum(v.size for _, v in trainable):,} trainable parameters, "
                      f"beta {beta}, lr {learning_rate}")

        ref_chosen, ref_rejected = reference_logprobs(model, train_set, batch_size)
        valid_ref_chosen, valid_ref_rejected = reference_logprobs(model, valid_set, batch_size)
        loss_fn = make_loss_fn(beta, sft_alpha, preference)
        value_and_grad = nn.value_and_grad(model, loss_fn)
        optimizer = optim.AdamW(learning_rate=learning_rate)

        history = []

        def validate(model, iteration):
            loss, stats = evaluate_pairs(model, valid_set, valid_ref_chosen, valid_ref_rejected, loss_fn, batch_size)
            record = {"type": "valid", "iteration": iteration, "loss": loss, **dict(zip(STAT_NAMES, stats, strict=True))}
            history.append(record)
            console.print(f"  iter {iteration:4d}  valid loss {loss:.4f}  reward acc {stats[0]:.2f}  "
                          f"margin {stats[1]:+.3f}  Δlogp chosen {stats[2]:+.2f} rejected {stats[3]:+.2f}")
            return loss

        initial = validate(model, 0)
        # With the policy equal to the reference and dropout off, every margin is zero and the loss is ln 2.
        if abs(initial - math.log(2)) > 0.05 and not sft_alpha:
            raise RuntimeError(f"Initial DPO loss {initial:.4f} is not ln 2 = {math.log(2):.4f}: "
                               "the reference and the policy disagree at step 0")

        order, cursor, window = np.random.permutation(len(train_set)), 0, []
        for iteration in range(1, iters + 1):
            if cursor + batch_size > len(order):
                order, cursor = np.random.permutation(len(train_set)), 0
            index = order[cursor:cursor + batch_size]
            cursor += batch_size
            rows = [train_set[i] for i in index]
            (loss, stats), grads = value_and_grad(model, *_stack(rows), ref_chosen[mx.array(index)], ref_rejected[mx.array(index)])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, stats)
            del grads
            mx.clear_cache()  # sequence lengths vary per step, so cached buffers are rarely reused
            window.append(float(loss.item()))
            history.append({"type": "train", "iteration": iteration, "loss": window[-1]})
            if iteration % steps_per_eval == 0 or iteration == iters:
                console.print(f"  iter {iteration:4d}  train loss {np.mean(window):.4f}  "
                              f"peak memory {mx.get_peak_memory() / 1e9:.1f} GB")
                window = []
                validate(model, iteration)

        adapter_dir = run_dir / "adapters"
        adapter_dir.mkdir()
        mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        shutil.copy(Path(adapter_path) / "adapter_config.json", adapter_dir / "adapter_config.json")
        valid_records = [h for h in history if h["type"] == "valid"]
        summary = {
            "initial_valid_loss": valid_records[0]["loss"], "final_valid_loss": valid_records[-1]["loss"],
            "final_reward_accuracy": valid_records[-1]["reward_accuracy"],
            "final_chosen_logp_delta": valid_records[-1]["chosen_logp_delta"],
            "final_rejected_logp_delta": valid_records[-1]["rejected_logp_delta"],
            "train_pairs": len(train_set), "valid_pairs": len(valid_set), "dropped_pairs": dropped,
            "training_time_seconds": round(time.perf_counter() - started, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 ** 2), 2),
        }
        write_json(run_dir / "dpo_history.json", {"config": config, "summary": summary, "history": history})
        plot_dpo(history, run_dir / "dpo_curves.png")
        manifest["adapter"] = adapter_identity(adapter_dir)
        manifest["adapter_path"] = str(adapter_dir)
        manifest["result"] = summary
        console.print(f"Adapter: {adapter_dir}")
        finish_run(root, run_dir, manifest)
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
    return run_dir
