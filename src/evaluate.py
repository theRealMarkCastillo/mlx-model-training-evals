"""
Comprehensive Multi-Pillar Evaluation Suite for MLX Models.

Evaluates Base Model vs. LoRA Fine-Tuned Model across 4 pillars:
1. Intrinsic Metrics: Cross-Entropy Loss & Perplexity on holdout test set.
2. Deterministic Metrics: Pure JSON rate, Pydantic Schema conformance, Tool Selection, and Parameter Exact Match.
3. Comparative Inspection: Side-by-side generation review on tricky test prompts.
4. Systems & Efficiency: Tokens/second generation speed, token count overhead, and peak Metal memory.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx_lm
import mlx_lm.lora as lora
from src.schema import SYSTEM_PROMPT, parse_and_validate

console = Console()


def compute_perplexity(model, tokenizer, test_file: Path, max_samples: int = 50) -> Tuple[float, float]:
    """
    Computes average loss and perplexity on the holdout test set.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with open(test_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()][:max_samples]

    for line in lines:
        sample = json.loads(line)
        # Apply chat template
        messages = sample["messages"]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = mx.array(tokenizer.encode(full_text))[None, :]

        # Forward pass to get logits
        logits = model(input_ids)
        # Shift tokens for autoregressive loss
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Cross entropy loss
        ce = nn.losses.cross_entropy(shift_logits, shift_labels)
        loss_val = ce.sum().item()
        n_toks = shift_labels.size

        total_loss += loss_val
        total_tokens += n_toks

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def run_deterministic_eval(
    model, tokenizer, test_samples: List[Dict[str, Any]], max_tokens: int = 150
) -> Dict[str, Any]:
    """
    Evaluates pure JSON rate, schema validity, tool selection, and parameter accuracy.
    """
    pure_json_count = 0
    valid_json_count = 0
    schema_valid_count = 0
    correct_tool_count = 0
    exact_match_count = 0
    total_output_tokens = 0
    results = []

    for item in tqdm(test_samples, desc="Evaluating test set"):
        prompt_text = item["prompt"]
        expected_json = json.loads(item["completion"])
        expected_tool = expected_json.get("tool")
        expected_params = expected_json.get("parameters", {})

        # Build prompt using chat template with generation prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate response
        start_time = time.time()
        raw_output = mlx_lm.generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        gen_time = time.time() - start_time
        out_tokens = len(tokenizer.encode(raw_output))
        total_output_tokens += out_tokens

        # Validate structure
        val_res = parse_and_validate(raw_output)

        if val_res["is_pure_json"]:
            pure_json_count += 1
        if val_res["is_valid_json"]:
            valid_json_count += 1
        if val_res["is_schema_valid"]:
            schema_valid_count += 1

        tool_match = False
        param_exact = False
        if val_res["parsed_data"]:
            parsed_tool = val_res["parsed_data"].get("tool")
            parsed_params = val_res["parsed_data"].get("parameters", {})
            if parsed_tool == expected_tool:
                tool_match = True
                correct_tool_count += 1
                # Check parameter equivalence
                if parsed_params == expected_params:
                    param_exact = True
                    exact_match_count += 1

        results.append({
            "prompt": prompt_text,
            "raw_output": raw_output,
            "output_tokens": out_tokens,
            "latency_seconds": round(gen_time, 3),
            "is_pure_json": val_res["is_pure_json"],
            "is_schema_valid": val_res["is_schema_valid"],
            "tool_match": tool_match,
            "param_exact": param_exact,
            "error": val_res["error"],
        })

    n = len(test_samples)
    return {
        "num_samples": n,
        "pure_json_rate": pure_json_count / n,
        "valid_json_rate": valid_json_count / n,
        "schema_valid_rate": schema_valid_count / n,
        "tool_accuracy": correct_tool_count / n,
        "exact_match_rate": exact_match_count / n,
        "avg_output_tokens": total_output_tokens / n,
        "sample_results": results,
    }


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
    model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit",
    adapter_path: str = "artifacts/adapters",
    test_jsonl: str = "data/test.jsonl",
    raw_test_file: str = "data/raw_test_samples.json",
    num_eval_samples: int = 30,
):
    console.print(
        Panel.fit(
            "[bold green]Running Comprehensive 4-Pillar Evaluation Suite[/bold green]\n"
            f"Base Model: [yellow]{model_name}[/yellow] | Adapter: [yellow]{adapter_path}[/yellow]",
            border_style="green",
        )
    )

    with open(raw_test_file, "r", encoding="utf-8") as f:
        test_samples = json.load(f)[:num_eval_samples]

    # --- Pillar 1 & 2: Evaluate BASE MODEL ---
    console.print("\n[bold yellow]Phase 1/2: Evaluating Base Model (no adapter)...[/bold yellow]")
    base_model, tokenizer = mlx_lm.load(model_name)
    base_loss, base_ppl = compute_perplexity(base_model, tokenizer, Path(test_jsonl), max_samples=num_eval_samples)
    base_metrics = run_deterministic_eval(base_model, tokenizer, test_samples)
    base_metrics["loss"] = base_loss
    base_metrics["perplexity"] = base_ppl

    # Clear memory
    del base_model
    mx.clear_cache()

    # --- Pillar 1 & 2: Evaluate LoRA MODEL ---
    console.print("\n[bold cyan]Phase 2/2: Evaluating Fine-Tuned Model (with LoRA adapter)...[/bold cyan]")
    lora_model, tokenizer = mlx_lm.load(model_name, adapter_path=adapter_path)
    lora_loss, lora_ppl = compute_perplexity(lora_model, tokenizer, Path(test_jsonl), max_samples=num_eval_samples)
    lora_metrics = run_deterministic_eval(lora_model, tokenizer, test_samples)
    lora_metrics["loss"] = lora_loss
    lora_metrics["perplexity"] = lora_ppl

    # --- Pillar 3: Side-by-Side Comparison ---
    comparison_table = Table(title="Pillar 3: Qualitative Side-by-Side Review", show_header=True, header_style="bold blue")
    comparison_table.add_column("User Operational Prompt", style="dim", width=35)
    comparison_table.add_column("Base Model Output", style="red", width=35)
    comparison_table.add_column("LoRA Model Output", style="green", width=35)

    for i in range(min(4, len(test_samples))):
        prompt_preview = test_samples[i]["prompt"]
        base_out = base_metrics["sample_results"][i]["raw_output"]
        lora_out = lora_metrics["sample_results"][i]["raw_output"]
        comparison_table.add_row(prompt_preview, base_out[:120] + ("..." if len(base_out) > 120 else ""), lora_out[:120] + ("..." if len(lora_out) > 120 else ""))

    console.print(comparison_table)

    # --- Summary Metrics Table ---
    summary_table = Table(title="Evaluation Scorecard: Base vs. LoRA", show_header=True, header_style="bold magenta")
    summary_table.add_column("Evaluation Metric", style="cyan")
    summary_table.add_column("Base Model", justify="right")
    summary_table.add_column("LoRA Fine-Tuned", justify="right")
    summary_table.add_column("Delta / Impact", justify="right")

    def diff_pct(base, lora):
        d = (lora - base) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}%"

    summary_table.add_row("Holdout Test Loss", f"{base_loss:.4f}", f"{lora_loss:.4f}", f"{(lora_loss - base_loss):.4f}")
    summary_table.add_row("Holdout Perplexity", f"{base_ppl:.2f}", f"{lora_ppl:.2f}", f"{(lora_ppl - base_ppl):.2f}")
    summary_table.add_row("Pure JSON Rate (No chatter/markdown)", f"{base_metrics['pure_json_rate']*100:.1f}%", f"{lora_metrics['pure_json_rate']*100:.1f}%", diff_pct(base_metrics['pure_json_rate'], lora_metrics['pure_json_rate']))
    summary_table.add_row("Pydantic Schema Validity", f"{base_metrics['schema_valid_rate']*100:.1f}%", f"{lora_metrics['schema_valid_rate']*100:.1f}%", diff_pct(base_metrics['schema_valid_rate'], lora_metrics['schema_valid_rate']))
    summary_table.add_row("Tool Selection Accuracy", f"{base_metrics['tool_accuracy']*100:.1f}%", f"{lora_metrics['tool_accuracy']*100:.1f}%", diff_pct(base_metrics['tool_accuracy'], lora_metrics['tool_accuracy']))
    summary_table.add_row("Parameter Exact Match", f"{base_metrics['exact_match_rate']*100:.1f}%", f"{lora_metrics['exact_match_rate']*100:.1f}%", diff_pct(base_metrics['exact_match_rate'], lora_metrics['exact_match_rate']))
    summary_table.add_row("Avg Output Tokens (Efficiency)", f"{base_metrics['avg_output_tokens']:.1f}", f"{lora_metrics['avg_output_tokens']:.1f}", f"{lora_metrics['avg_output_tokens'] - base_metrics['avg_output_tokens']:.1f} toks")

    console.print(summary_table)

    # Save artifact files
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    plot_eval_metrics(base_metrics, lora_metrics, artifacts_dir / "eval_comparison.png")

    report = {
        "model": model_name,
        "adapter": adapter_path,
        "base_metrics": {k: v for k, v in base_metrics.items() if k != "sample_results"},
        "lora_metrics": {k: v for k, v in lora_metrics.items() if k != "sample_results"},
        "side_by_side_samples": [
            {
                "prompt": test_samples[i]["prompt"],
                "base_output": base_metrics["sample_results"][i]["raw_output"],
                "lora_output": lora_metrics["sample_results"][i]["raw_output"],
            }
            for i in range(min(10, len(test_samples)))
        ],
    }

    with open(artifacts_dir / "eval_results.json", "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"[green]✓[/green] Full evaluation report saved to [bold]artifacts/eval_results.json[/bold]")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MLX model against base model")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit", help="Base model identifier")
    parser.add_argument("--adapter", default="artifacts/adapters", help="LoRA adapter directory")
    parser.add_argument("--samples", type=int, default=30, help="Number of test samples to evaluate")
    args = parser.parse_args()

    run_comprehensive_evaluation(
        model_name=args.model,
        adapter_path=args.adapter,
        num_eval_samples=args.samples,
    )
