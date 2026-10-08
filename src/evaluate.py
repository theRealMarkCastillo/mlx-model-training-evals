"""Compare model variants on the holdout and challenge sets with auditable per-sample results.

Variants:
  base     zero-shot base model with the system prompt
  fewshot  base model with worked examples from the training split (no training)
  lora     base model plus the trained adapter
  grammar  base model with grammar-constrained decoding (`--constrained`), no training
  fused    adapter merged into the weights (optional)

The few-shot baseline answers "was fine-tuning necessary?": if it scores
close to LoRA, prompting may be the cheaper solution. The grammar variant
answers the companion question: how much of the LoRA win is formatting, and how
much is task skill? Constrained decoding cannot produce invalid JSON, so whatever
it still gets wrong is a genuine tool-choice or parameter error.
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

from src.constrained import constrained_generate
from src.dataset import DATA_DIR, challenge_files, fewshot_messages, load_samples, positive_int, select_shots
from src.inference import generate_response
from src.metrics import (
    CATEGORY_HELP,
    CATEGORY_ORDER,
    failure_examples,
    format_rate,
    paired_comparison,
    score_sample,
    summarize,
)
from src.models import resolve_model_paths
from src.runs import (
    adapter_identity,
    file_identity,
    finish_run,
    latest_path,
    new_run,
    record_failure,
    resolve_adapter_source,
    resolve_source,
    write_json,
)
from src.schema import parse_and_validate

console = Console()
DEFAULT_VARIANTS = ("base", "fewshot", "lora")
COLORS = {"base": "#d95f02", "fewshot": "#7570b3", "lora": "#1b9e77", "fused": "#66a61e", "grammar": "#e7298a"}
LABELS = {"base": "Base (zero-shot)", "fewshot": "Base (few-shot)", "lora": "LoRA", "fused": "Fused LoRA",
          "grammar": "Base + JSON grammar"}


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


def run_deterministic_eval(model, tokenizer, test_samples, max_tokens=150, desc="Evaluating",
                           generate=None, temperature=0.0):
    """Greedy-decode each prompt, parse strictly, and score against the reference.

    `generate` is the decoding function; evaluation passes the grammar-constrained one for
    its `grammar` variant and leaves the default (plain greedy) everywhere else.
    `temperature > 0` samples instead of decoding greedily — the metrics are the same, but
    a run is then a stochastic draw rather than a fixed measurement (see the README).
    """
    if not test_samples:
        raise ValueError("Cannot evaluate an empty dataset")
    positive_int(max_tokens)
    generate = generate or generate_response
    results = []
    for item in tqdm(test_samples, desc=desc, leave=False):
        generated = generate(model, tokenizer, item["messages"][:-1], max_tokens, temperature=temperature)
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
        for bar, value, high in zip(bars, values, highs, strict=True):
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
        bottoms = [b + n for b, n in zip(bottoms, counts, strict=True)]
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


def paired_report(datasets, reference="lora", flag="param_exact"):
    """McNemar comparisons of the reference variant against each other variant, per dataset.

    The reference is LoRA when it is present, otherwise the last variant evaluated — so a
    baselines-only or constrained-only run still gets a paired test (`base` vs `grammar`,
    for example) instead of an empty table. `flag` selects the paired metric: exact match by
    default, or `is_schema_valid` for "did decoding fix the format?" questions.
    """
    reports = {}
    for name, by_variant in datasets.items():
        chosen = reference if reference in by_variant else (list(by_variant)[-1] if by_variant else None)
        if chosen is None or len(by_variant) < 2:
            continue
        reports[name] = {
            "reference": chosen,
            "comparisons": {
                other: paired_comparison(by_variant[chosen]["sample_results"],
                                         by_variant[other]["sample_results"], flag)
                for other in by_variant if other != chosen
            },
        }
    return reports


def print_report(datasets, paired=None, focus="lora", schema_paired=None):
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
        reference = paired["test"]["reference"]
        table = Table(title=f"Paired exact-match comparison on the holdout (same samples): "
                            f"{LABELS.get(reference, reference)} vs each other variant")
        for column in (f"{LABELS.get(reference, reference)} vs", "only reference right", "only other right",
                       "both right", "McNemar p"):
            table.add_column(column, justify="right")
        for other, c in paired["test"]["comparisons"].items():
            table.add_row(LABELS.get(other, other), str(c["only_a"]), str(c["only_b"]), str(c["both_correct"]),
                          f"{c['p_value']:.3g}")
        console.print(table)
        console.print("[dim]p < 0.05: the difference is unlikely to be sampling noise on this set.[/dim]")

    if schema_paired and schema_paired.get("test"):
        reference = schema_paired["test"]["reference"]
        rows = [(other, c) for other, c in schema_paired["test"]["comparisons"].items()
                if c["only_a"] or c["only_b"]]
        if rows:
            table = Table(title=f"Paired schema-validity comparison (won/lost structural validity): "
                                f"{LABELS.get(reference, reference)} vs each other variant")
            for column in (f"{LABELS.get(reference, reference)} vs", "only reference valid", "only other valid",
                           "both valid", "McNemar p"):
                table.add_column(column, justify="right")
            for other, c in rows:
                table.add_row(LABELS.get(other, other), str(c["only_a"]), str(c["only_b"]), str(c["both_correct"]),
                              f"{c['p_value']:.3g}")
            console.print(table)

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
    variants=DEFAULT_VARIANTS, shots=5, challenge=False, quiet=False, constrained=False,
    temperature=0.0, seed=42,
):
    """Evaluate the chosen variants on the test split (and challenge sets) in a new run directory.

    `constrained=True` adds a `grammar` variant: the same base model, decoded under the
    schema grammar from `src/constrained.py`, so the report separates formatting failures
    from task failures ("would valid JSON have been enough?"). `temperature > 0` samples
    instead of decoding greedily, which turns each variant into one stochastic draw: the
    metrics and the paired test still apply, but the numbers are no longer the model's
    single most likely output. The seed is recorded and re-applied per variant, so a
    sampled run is reproducible.
    """
    if num_eval_samples is not None:
        positive_int(num_eval_samples)
    positive_int(max_tokens)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")
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
        fewshot_ids=[s["id"] for s in demos], constrained=constrained,
        generation={"temperature": temperature, "max_tokens": max_tokens, "seed": seed},
    )
    with record_failure(run_dir, manifest):
        # plan entries: (variant name, model source, adapter, use grammar-constrained decoding)
        plan = [(name, model_source, adapter_path if name == "lora" else None, False) for name in variants]
        if constrained:
            plan.append(("grammar", model_source, None, True))
        if fused_path:
            fused_source, fused_identity = resolve_source(latest_path(fused_path))
            manifest["fused_model"] = fused_identity
            plan.append(("fused", fused_source, None, False))
        datasets = {name: {} for name in samples}
        for variant, source, adapter, use_grammar in plan:
            console.print(f"[bold]Evaluating {LABELS.get(variant, variant)}[/bold]")
            model, tokenizer = mlx_lm.load(source, **({"adapter_path": adapter} if adapter else {}))
            try:
                if temperature > 0:
                    mx.random.seed(seed)  # same sampling draw for every variant
                for name, group in samples.items():
                    if variant == "fewshot":
                        group = [{**s, "messages": fewshot_messages(s["messages"], demos)} for s in group]
                    intrinsic = compute_perplexity(model, tokenizer, group)
                    generate = constrained_generate if use_grammar else None
                    metrics = run_deterministic_eval(model, tokenizer, group, max_tokens,
                                                     desc=f"{variant}/{name}", generate=generate,
                                                     temperature=temperature)
                    datasets[name][variant] = {**metrics, **intrinsic}
            finally:
                del model
                mx.clear_cache()
        report = {"model": model_name, "adapter": adapter_path, "run_dir": str(run_dir), "run_id": manifest["run_id"],
                  "variants": [name for name, _, _, _ in plan], "constrained": constrained,
                  "temperature": temperature, "seed": seed, "datasets": datasets,
                  "paired": paired_report(datasets),
                  "paired_schema": paired_report(datasets, flag="is_schema_valid")}
        plot_eval_metrics(datasets["test"], run_dir / "eval_comparison.png")
        if challenge and len(datasets) > 1:
            plot_challenges(datasets, run_dir / "challenge_comparison.png")
        if not quiet:
            print_report(datasets, report["paired"], schema_paired=report["paired_schema"])
        report["manifest"] = {**manifest, "status": "complete"}
        write_json(run_dir / "eval_results.json", report)
        finish_run(root, run_dir, manifest)
    console.print(f"Saved per-sample results to {run_dir / 'eval_results.json'}")
    return report
