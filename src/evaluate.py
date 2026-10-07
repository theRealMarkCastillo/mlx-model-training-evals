"""Compare model variants on the holdout and challenge sets with auditable per-sample results.

Variants:
  base     zero-shot base model with the system prompt
  fewshot  base model with worked examples from the training split (no training)
  lora     base model plus the trained adapter
  fused    adapter merged into the weights (optional)

The few-shot baseline answers "was fine-tuning necessary?": if it scores
close to LoRA, prompting may be the cheaper solution.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx_lm
from mlx_lm.tuner.datasets import ChatDataset
from mlx_lm.tuner.trainer import default_loss
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.dataset import DATA_DIR, challenge_files, fewshot_messages, load_samples, positive_int, select_shots
from src.inference import generate_response
from src.metrics import (CATEGORY_HELP, CATEGORY_ORDER, failure_examples, format_rate, paired_comparison,
                         score_sample, summarize)
from src.models import resolve_model_paths
from src.schema import parse_and_validate
from src.runs import (adapter_identity, file_identity, finish_run, latest_path, new_run, record_failure,
                      resolve_adapter_source, resolve_source, write_json)

console = Console()
DEFAULT_VARIANTS = ("base", "fewshot", "lora")
COLORS = {"base": "#d95f02", "fewshot": "#7570b3", "lora": "#1b9e77", "fused": "#66a61e"}
LABELS = {"base": "Base (zero-shot)", "fewshot": "Base (few-shot)", "lora": "LoRA", "fused": "Fused LoRA"}


def compute_perplexity(model, tokenizer, samples):
    """Use the trainer's chat offsets and shifted-token loss mask without truncation."""
    if not samples:
        raise ValueError("Cannot score an empty dataset")
    model.eval()
    dataset = ChatDataset(samples, tokenizer, mask_prompt=True)
    total_loss = 0.0
    total_tokens = 0
    for sample in samples:
        tokens, offset = dataset.process(sample)
        if offset < 1 or offset >= len(tokens):
            raise ValueError(f"No assistant tokens in sample {sample.get('id')}")
        loss, count = default_loss(model, mx.array([tokens]), mx.array([[offset, len(tokens) - 1]]))
        count = int(count.item())
        total_loss += float(loss.item()) * count
        total_tokens += count
    average = total_loss / total_tokens
    if not math.isfinite(average):
        raise ValueError("Non-finite assistant loss")
    return {
        "loss": average, "perplexity": math.exp(average) if average < 709 else None,
        "loss_scope": "assistant_tokens", "loss_tokens": total_tokens,
    }


def run_deterministic_eval(model, tokenizer, test_samples, max_tokens=150, desc="Evaluating"):
    """Greedy-decode each prompt, parse strictly, and score against the reference."""
    if not test_samples:
        raise ValueError("Cannot evaluate an empty dataset")
    positive_int(max_tokens)
    results = []
    for item in tqdm(test_samples, desc=desc, leave=False):
        generated = generate_response(model, tokenizer, item["messages"][:-1], max_tokens)
        parsed = parse_and_validate(generated["raw_output"])
        scored = score_sample(item["expected"], item.get("normalized_expected", item["expected"]), parsed)
        results.append({
            "id": item["id"], "prompt": item["prompt"], "meta": item.get("meta"),
            "messages": item["messages"][:-1], "expected": item["expected"],
            **generated, **parsed, **scored,
        })
    metrics = summarize(results)
    metrics["avg_output_tokens"] = sum(r["output_tokens"] for r in results) / len(results)
    metrics["sample_results"] = results
    return metrics


def plot_eval_metrics(metrics_by_variant, output_path, title="Holdout test set"):
    """Left: headline rates with 95% intervals. Right: what kind of mistakes each variant makes."""
    keys = [("pure_json_rate", "Pure JSON"), ("schema_valid_rate", "Schema valid"),
            ("tool_accuracy", "Tool accuracy"), ("exact_match_rate", "Exact match")]
    variants = list(metrics_by_variant)
    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(14, 5), dpi=150, gridspec_kw={"width_ratios": [3, 2]})
    width = 0.8 / len(variants)
    for i, name in enumerate(variants):
        m = metrics_by_variant[name]
        values = [100 * m[k] for k, _ in keys]
        lows = [100 * (m[k] - m["intervals"][k][0]) for k, _ in keys]
        highs = [100 * (m["intervals"][k][1] - m[k]) for k, _ in keys]
        xs = [j + (i - (len(variants) - 1) / 2) * width for j in range(len(keys))]
        bars = ax.bar(xs, values, width, yerr=[lows, highs], capsize=3, label=LABELS.get(name, name),
                      color=COLORS.get(name), alpha=0.85, error_kw={"elinewidth": 1, "alpha": 0.7})
        for bar, value, high in zip(bars, values, highs):
            ax.annotate(f"{value:.0f}", (bar.get_x() + bar.get_width() / 2, value + high + 1),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(keys)), [label for _, label in keys])
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of samples (error bars: 95% Wilson interval)")
    ax.set_title(f"{title}: n={metrics_by_variant[variants[0]]['num_samples']}", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(variants), fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    bottoms = [0] * len(variants)
    cmap = plt.get_cmap("Set2")
    for c, category in enumerate(CATEGORY_ORDER):
        counts = [metrics_by_variant[v]["error_categories"][category] for v in variants]
        if not any(counts):
            continue
        ax_err.bar([LABELS.get(v, v) for v in variants], counts, bottom=bottoms, label=category, color=cmap(c))
        bottoms = [b + n for b, n in zip(bottoms, counts)]
    ax_err.set_title("Failures by category", fontweight="bold")
    ax_err.set_ylabel("samples")
    ax_err.tick_params(axis="x", labelsize=8)
    if any(bottoms):
        ax_err.legend(fontsize=7, title="category", title_fontsize=7)
    else:
        ax_err.text(0.5, 0.5, "no failures", ha="center", transform=ax_err.transAxes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_challenges(datasets, output_path):
    names = list(datasets)
    variants = list(datasets[names[0]])
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    width = 0.8 / len(variants)
    for i, variant in enumerate(variants):
        ms = [datasets[n][variant] for n in names]
        values = [100 * m["exact_match_rate"] for m in ms]
        err = [[100 * (m["exact_match_rate"] - m["intervals"]["exact_match_rate"][0]) for m in ms],
               [100 * (m["intervals"]["exact_match_rate"][1] - m["exact_match_rate"]) for m in ms]]
        xs = [j + (i - (len(variants) - 1) / 2) * width for j in range(len(names))]
        ax.bar(xs, values, width, yerr=err, capsize=3, label=LABELS.get(variant, variant), color=COLORS.get(variant), alpha=0.85)
    ax.set_xticks(range(len(names)), [n.replace("challenge_", "") for n in names])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact match % (95% interval)")
    ax.set_title("Exact match by evaluation set", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(variants), fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def paired_report(datasets, reference="lora"):
    """McNemar comparisons of the reference variant against each other variant, per dataset."""
    return {
        name: {other: paired_comparison(by_variant[reference]["sample_results"], by_variant[other]["sample_results"])
               for other in by_variant if other != reference}
        for name, by_variant in datasets.items() if reference in by_variant
    }


def print_report(datasets, paired=None, focus="lora"):
    test = datasets["test"]
    variants = list(test)
    table = Table(title="Holdout scorecard: rate [95% interval]")
    table.add_column("Metric")
    for v in variants:
        table.add_column(LABELS.get(v, v), justify="right")
    table.add_row("assistant loss", *(f"{test[v]['loss']:.4f}" for v in variants))
    for key in ("pure_json_rate", "schema_valid_rate", "tool_accuracy", "exact_match_rate", "normalized_match_rate"):
        table.add_row(key, *(format_rate(test[v][key], test[v]["intervals"][key]) for v in variants))
    console.print(table)

    tools = Table(title="Exact match by expected tool")
    tools.add_column("Tool")
    tools.add_column("n", justify="right")
    for v in variants:
        tools.add_column(LABELS.get(v, v), justify="right")
    for tool, row in test[variants[0]]["per_tool"].items():
        tools.add_row(tool, str(row["n"]), *(format_rate(test[v]["per_tool"][tool]["exact_match_rate"],
                                                          test[v]["per_tool"][tool]["ci95"]) for v in variants))
    console.print(tools)

    errors = Table(title="Failure categories (count)")
    errors.add_column("Category")
    errors.add_column("Meaning", style="dim")
    for v in variants:
        errors.add_column(LABELS.get(v, v), justify="right")
    for category in CATEGORY_ORDER:
        errors.add_row(category, CATEGORY_HELP[category], *(str(test[v]["error_categories"][category]) for v in variants))
    console.print(errors)

    if len(datasets) > 1:
        challenge = Table(title="Exact match by evaluation set")
        challenge.add_column("Set")
        for v in variants:
            challenge.add_column(LABELS.get(v, v), justify="right")
        for name, by_variant in datasets.items():
            challenge.add_row(name, *(format_rate(by_variant[v]["exact_match_rate"], by_variant[v]["intervals"]["exact_match_rate"]) for v in variants))
        console.print(challenge)

    if paired and paired.get("test"):
        table = Table(title="Paired exact-match comparison on the holdout (same samples)")
        for column in ("LoRA vs", "only LoRA right", "only other right", "both right", "McNemar p"):
            table.add_column(column, justify="right")
        for other, c in paired["test"].items():
            table.add_row(LABELS.get(other, other), str(c["only_a"]), str(c["only_b"]), str(c["both_correct"]), f"{c['p_value']:.3g}")
        console.print(table)
        console.print("[dim]p < 0.05: the difference is unlikely to be sampling noise on this set.[/dim]")

    focus = focus if focus in test else variants[-1]
    examples = failure_examples(test[focus]["sample_results"])
    if examples:
        console.print(f"[bold]Example {LABELS.get(focus, focus)} failures[/bold]")
        for e in examples:
            console.print(f"[yellow]{e['category']}[/yellow] {e['prompt']}")
            console.print(f"  expected: {e['expected']}")
            console.print(f"  output:   {e['raw_output']!r}")
            if e["wrong_fields"]:
                console.print(f"  wrong fields: {', '.join(e['wrong_fields'])}")


def run_comprehensive_evaluation(
    model_name=None, adapter_path=None, test_jsonl=None, num_eval_samples=None,
    *, preset=None, output_dir=None, fused_path=None, max_tokens=150,
    variants=DEFAULT_VARIANTS, shots=5, challenge=False, quiet=False,
):
    """Evaluate the chosen variants on the test split (and challenge sets) in a new run directory."""
    if num_eval_samples is not None:
        positive_int(num_eval_samples)
    positive_int(max_tokens)
    variants = tuple(variants)
    unknown = set(variants) - set(DEFAULT_VARIANTS)
    if unknown or not variants:
        raise ValueError(f"variants must be chosen from {DEFAULT_VARIANTS}")
    if "fewshot" in variants:
        positive_int(shots)
    paths = {"test": Path(test_jsonl) if test_jsonl else DATA_DIR / "test.jsonl"}
    if challenge:
        paths.update(challenge_files(paths["test"].parent))
    samples = {name: load_samples(path, num_eval_samples) for name, path in paths.items()}
    demos = select_shots(paths["test"].parent / "train.jsonl", shots) if "fewshot" in variants else []

    needs_adapter = "lora" in variants
    model_name, adapter_path, root = resolve_model_paths(preset, model_name, adapter_path, output_dir, need_adapter=needs_adapter)
    adapter_files = adapter_identity(adapter_path) if needs_adapter else None
    if needs_adapter:
        model_source, source_identity = resolve_adapter_source(model_name, adapter_path)
    else:
        model_source, source_identity = resolve_source(model_name)
    run_dir, manifest = new_run(
        root, "evaluation", model=model_name, model_source=source_identity, adapter=adapter_files,
        datasets={name: file_identity(path) for name, path in paths.items()},
        sample_ids={name: [s["id"] for s in group] for name, group in samples.items()},
        fewshot_ids=[s["id"] for s in demos],
        generation={"temperature": 0.0, "max_tokens": max_tokens},
    )
    with record_failure(run_dir, manifest):
        plan = [(name, model_source, adapter_path if name == "lora" else None) for name in variants]
        if fused_path:
            fused_source, fused_identity = resolve_source(latest_path(fused_path))
            manifest["fused_model"] = fused_identity
            plan.append(("fused", fused_source, None))
        datasets = {name: {} for name in samples}
        for variant, source, adapter in plan:
            console.print(f"[bold]Evaluating {LABELS.get(variant, variant)}[/bold]")
            model, tokenizer = mlx_lm.load(source, **({"adapter_path": adapter} if adapter else {}))
            try:
                for name, group in samples.items():
                    if variant == "fewshot":
                        group = [{**s, "messages": fewshot_messages(s["messages"], demos)} for s in group]
                    intrinsic = compute_perplexity(model, tokenizer, group)
                    metrics = run_deterministic_eval(model, tokenizer, group, max_tokens, desc=f"{variant}/{name}")
                    datasets[name][variant] = {**metrics, **intrinsic}
            finally:
                del model
                mx.clear_cache()
        report = {"model": model_name, "adapter": adapter_path, "run_dir": str(run_dir), "run_id": manifest["run_id"],
                  "variants": [name for name, _, _ in plan], "datasets": datasets,
                  "paired": paired_report(datasets)}
        plot_eval_metrics(datasets["test"], run_dir / "eval_comparison.png")
        if challenge and len(datasets) > 1:
            plot_challenges(datasets, run_dir / "challenge_comparison.png")
        if not quiet:
            print_report(datasets, report["paired"])
        report["manifest"] = {**manifest, "status": "complete"}
        write_json(run_dir / "eval_results.json", report)
        finish_run(root, run_dir, manifest)
    console.print(f"Saved per-sample results to {run_dir / 'eval_results.json'}")
    return report
