"""Assistant-only loss, strict task scoring, and auditable per-sample results."""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import mlx.core as mx
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx_lm
from mlx_lm.tuner.datasets import ChatDataset
from mlx_lm.tuner.trainer import default_loss
from src.schema import parse_and_validate
from src.dataset import load_samples, positive_int
from src.inference import generate_response
from src.models import add_preset_argument, resolve_model_paths
from src.runs import resolve_source, resolve_adapter_source, file_identity, adapter_identity, new_run, finish_run, write_json, latest_path

console = Console()


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
        loss, count = default_loss(
            model, mx.array([tokens]), mx.array([[offset, len(tokens) - 1]]),
        )
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


def run_deterministic_eval(model, tokenizer, test_samples, max_tokens=150):
    if not test_samples:
        raise ValueError("Cannot evaluate an empty dataset")
    positive_int(max_tokens)
    results = []
    for item in tqdm(test_samples, desc="Evaluating test set"):
        expected = item["expected"]
        generated = generate_response(model, tokenizer, item["messages"][:-1], max_tokens)
        parsed = parse_and_validate(generated["raw_output"])
        actual = parsed["parsed_data"]
        tool_match = isinstance(actual, dict) and actual.get("tool") == expected["tool"]
        exact = parsed["is_pure_json"] and parsed["is_schema_valid"] and actual == expected
        normalized_match = parsed["is_schema_valid"] and parsed["normalized_data"] == item.get("normalized_expected", expected)
        if not parsed["is_valid_json"]:
            category = "json"
        elif not parsed["is_schema_valid"]:
            category = "schema"
        elif not parsed["is_pure_json"]:
            category = "format"
        elif not tool_match:
            category = "tool"
        elif not exact:
            category = "parameters"
        else:
            category = None
        results.append({
            "id": item["id"], "prompt": item["prompt"], "messages": item["messages"][:-1],
            "expected": expected, **generated, **parsed,
            "tool_match": tool_match, "param_exact": exact,
            "normalized_match": normalized_match, "error_category": category,
        })
    n = len(results)
    metrics = {"num_samples": n, "sample_results": results}
    for metric, flag in {
        "pure_json_rate": "is_pure_json", "valid_json_rate": "is_valid_json",
        "schema_valid_rate": "is_schema_valid", "tool_accuracy": "tool_match",
        "exact_match_rate": "param_exact", "normalized_match_rate": "normalized_match",
    }.items():
        metrics[metric] = sum(r[flag] for r in results) / n
    metrics["avg_output_tokens"] = sum(r["output_tokens"] for r in results) / n
    return metrics


def plot_eval_metrics(base_metrics: Dict[str, Any], lora_metrics: Dict[str, Any], output_path: Path):
    """Generates comparison bar chart comparing Base vs LoRA."""
    labels = ["Pure JSON", "Schema Valid", "Tool Accuracy", "Exact Match"]
    base_scores = [
        base_metrics["pure_json_rate"] * 100,
        base_metrics["schema_valid_rate"] * 100,
        base_metrics["tool_accuracy"] * 100,
        base_metrics["exact_match_rate"] * 100,
    ]
    lora_scores = [
        lora_metrics["pure_json_rate"] * 100,
        lora_metrics["schema_valid_rate"] * 100,
        lora_metrics["tool_accuracy"] * 100,
        lora_metrics["exact_match_rate"] * 100,
    ]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    rects1 = ax.bar([p - width / 2 for p in x], base_scores, width, label="Base Model", color="#d95f02", alpha=0.85)
    rects2 = ax.bar([p + width / 2 for p in x], lora_scores, width, label="LoRA Fine-Tuned", color="#1b9e77", alpha=0.85)

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Comprehensive Evaluation: Base Model vs. MLX LoRA Fine-Tuned", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="white")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Attach labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    console.print(f"[green]✓[/green] Saved evaluation comparison plot to [bold]{output_path}[/bold]")


def run_comprehensive_evaluation(
    model_name=None, adapter_path=None, test_jsonl="data/test.jsonl", num_eval_samples=30,
    *, preset=None, output_dir=None, fused_path=None, max_tokens=150,
):
    positive_int(num_eval_samples)
    positive_int(max_tokens)
    samples = load_samples(test_jsonl, num_eval_samples)
    model_name, adapter_path, root = resolve_model_paths(preset, model_name, adapter_path, output_dir)
    adapter_files = adapter_identity(adapter_path)
    model_source, source_identity = resolve_adapter_source(model_name, adapter_path)
    run_dir, manifest = new_run(
        root, "evaluation", model=model_name, model_source=source_identity,
        adapter=adapter_files, dataset=file_identity(test_jsonl),
        sample_ids=[s["id"] for s in samples],
        generation={"temperature": 0.0, "max_tokens": max_tokens},
    )
    variants = [("base", model_source, None), ("lora", model_source, adapter_path)]
    if fused_path:
        fused_source, fused_identity = resolve_source(latest_path(fused_path))
        manifest["fused_model"] = fused_identity
        variants.append(("fused", fused_source, None))
    report = {"model": model_name, "adapter": adapter_path, "run_dir": str(run_dir), "run_id": manifest["run_id"]}
    for name, source, adapter in variants:
        console.print(f"Evaluating {name} model...")
        model, tokenizer = mlx_lm.load(source, **({"adapter_path": adapter} if adapter else {}))
        try:
            intrinsic = compute_perplexity(model, tokenizer, samples)
            metrics = run_deterministic_eval(model, tokenizer, samples, max_tokens)
            report[f"{name}_metrics"] = {**metrics, **intrinsic}
        finally:
            del model
            mx.clear_cache()
    plot_eval_metrics(report["base_metrics"], report["lora_metrics"], run_dir / "eval_comparison.png")
    table = Table(title="Evaluation scorecard (assistant-only loss; strict exact match)")
    table.add_column("Metric")
    for name, _, _ in variants:
        table.add_column(name)
    for key in ("loss", "perplexity", "pure_json_rate", "schema_valid_rate", "tool_accuracy", "exact_match_rate", "normalized_match_rate"):
        table.add_row(key, *(str(report[f"{name}_metrics"][key]) for name, _, _ in variants))
    console.print(table)
    report["manifest"] = {**manifest, "status": "complete"}
    write_json(run_dir / "eval_results.json", report)
    finish_run(root, run_dir, manifest)
    console.print(f"Saved all sample results to {run_dir / 'eval_results.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate base, LoRA, and optionally fused models")
    selection = parser.add_mutually_exclusive_group()
    add_preset_argument(selection)
    selection.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--output-dir")
    parser.add_argument("--test-file", default="data/test.jsonl")
    parser.add_argument("--fused", help="Also evaluate this fused model directory")
    parser.add_argument("--samples", type=positive_int, default=30)
    parser.add_argument("--max-tokens", type=positive_int, default=150)
    args = parser.parse_args()
    run_comprehensive_evaluation(
        model_name=args.model, adapter_path=args.adapter, test_jsonl=args.test_file,
        num_eval_samples=args.samples, preset=args.preset, output_dir=args.output_dir,
        fused_path=args.fused, max_tokens=args.max_tokens,
    )
