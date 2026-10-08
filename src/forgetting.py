"""Catastrophic-forgetting check: did fine-tuning on the tool task hurt general ability?

Every other metric in this repo measures the task. This one measures the *cost* of the
task: it compares the base model and the trained adapter on a small set of ordinary
general-knowledge and reasoning requests (`data/general.jsonl`), using assistant loss —
the same masked cross-entropy the trainer optimizes, with no exact-match component
because free-form answers have no single correct string.

Two design choices matter:

* The general records use their own plain system prompt, not the tool-calling one. That is
  what makes drift visible: an adapter that has overfit the ops contract tends to produce
  tool-call-shaped output even when asked to do arithmetic, and the loss on the ordinary
  answer rises.
* The comparison is **paired per record**, and tested with an exact sign test
  (`src/metrics.py:sign_test`) rather than a difference of means, because a handful of
  records should not be able to produce a confident verdict by accident.

Answering this on a 24-record synthetic set can never prove that general ability is
preserved. It can show that it was not *obviously* destroyed, which is the honest claim.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx_lm
from rich.console import Console
from rich.table import Table

from src.dataset import DATA_DIR, load_general_samples, positive_int
from src.evaluate import compute_perplexity
from src.metrics import sign_test
from src.models import resolve_model_paths
from src.runs import (
    adapter_identity,
    finish_run,
    new_run,
    record_failure,
    resolve_adapter_source,
    write_json,
)

console = Console()
GENERAL_PATH = DATA_DIR / "general.jsonl"


def plot_forgetting(per_record, output_path):
    """Per-record loss before and after fine-tuning, sorted by the change."""
    ordered = sorted(per_record, key=lambda row: row["delta"])
    positions = range(len(ordered))
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.plot(positions, [row["base_loss"] for row in ordered], marker="o", linestyle="-",
            color="#d95f02", label="base", markersize=4)
    ax.plot(positions, [row["lora_loss"] for row in ordered], marker="s", linestyle="-",
            color="#1b9e77", label="LoRA", markersize=4)
    for position, row in zip(positions, ordered, strict=True):
        colour = "#d62728" if row["delta"] > 0 else "#2ca02c"
        ax.plot([position, position], [row["base_loss"], row["lora_loss"]], color=colour, alpha=0.5, linewidth=1)
    ax.set_title("General-capability loss per record (red: worse after fine-tuning)", fontweight="bold")
    ax.set_xlabel("general records, sorted by change")
    ax.set_ylabel("Assistant loss (nats/token)")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_forgetting_check(*, preset=None, model_name=None, adapter_path=None, output_dir=None,
                         general_jsonl=None, max_samples=None, quiet=False):
    """Score the base model and the trained adapter on the general set and compare them."""
    if max_samples is not None:
        positive_int(max_samples)
    path = Path(general_jsonl) if general_jsonl else GENERAL_PATH
    samples = load_general_samples(path, max_samples)
    if len(samples) < 8:
        console.print("[yellow]Very few general records: treat the sign test as decoration, not evidence.[/yellow]")

    model_name, adapter_path, root = resolve_model_paths(preset, model_name, adapter_path, output_dir)
    adapter_files = adapter_identity(adapter_path)
    source, identity = resolve_adapter_source(model_name, adapter_path)
    run_dir, manifest = new_run(
        root, "forgetting", model=model_name, model_source=identity, adapter=adapter_files,
        dataset={"path": str(path), "n": len(samples)},
        metric="assistant loss (masked cross-entropy on the general answers)",
    )
    with record_failure(run_dir, manifest):
        losses = {}
        for variant, load_adapter in (("base", False), ("lora", True)):
            console.print(f"[bold]Scoring {variant} on {len(samples)} general records[/bold]")
            model, tokenizer = mlx_lm.load(source, **({"adapter_path": adapter_path} if load_adapter else {}))
            try:
                record_losses = per_record_losses(model, tokenizer, samples)
            finally:
                del model
                mx.clear_cache()
            losses[variant] = record_losses

        per_record = [
            {"id": sample["id"], "prompt": sample["prompt"],
             "base_loss": losses["base"][sample["id"]], "lora_loss": losses["lora"][sample["id"]],
             "delta": losses["lora"][sample["id"]] - losses["base"][sample["id"]]}
            for sample in samples
        ]
        base_mean = sum(row["base_loss"] for row in per_record) / len(per_record)
        lora_mean = sum(row["lora_loss"] for row in per_record) / len(per_record)
        test = sign_test([row["delta"] for row in per_record])
        result = {
            "model": model_name, "adapter": adapter_path, "run_dir": str(run_dir), "run_id": manifest["run_id"],
            "n": len(per_record), "base_loss": base_mean, "lora_loss": lora_mean,
            "delta": lora_mean - base_mean, "sign_test": test, "per_record": per_record,
            "verdict": forgetting_verdict(base_mean, lora_mean, test),
        }
        plot_forgetting(per_record, run_dir / "forgetting.png")
        table = Table(title="Catastrophic-forgetting check (assistant loss on general requests)")
        for column in ("Model", "Mean loss", "vs base", "Records worse / better", "Sign-test p"):
            table.add_column(column, justify="right" if column != "Model" else "left")
        table.add_row("Base", f"{base_mean:.4f}", "-", "-", "-")
        table.add_row("LoRA", f"{lora_mean:.4f}", f"{result['delta']:+.4f}",
                      f"{test['increased']} / {test['decreased']}", f"{test['p_value']:.3g}")
        console.print(table)
        if not quiet:
            console.print(result["verdict"])
        manifest["result"] = {"base_loss": base_mean, "lora_loss": lora_mean, "sign_test": test}
        write_json(run_dir / "forgetting.json", result)
        finish_run(root, run_dir, manifest)
    return result


def per_record_losses(model, tokenizer, samples):
    """Assistant loss for each record separately, reusing the evaluator's masking."""
    losses = {}
    for sample in samples:
        metrics = compute_perplexity(model, tokenizer, [sample])
        losses[sample["id"]] = metrics["loss"]
    return losses


def forgetting_verdict(base_loss, lora_loss, test):
    """Plain-language reading of the numbers, including what they cannot show."""
    if test["p_value"] >= 0.05:
        headline = (f"No detectable damage to general ability on this set: mean loss {lora_loss:.3f} vs "
                    f"{base_loss:.3f} ({test['increased']} records worse, {test['decreased']} better, "
                    f"sign-test p = {test['p_value']:.2f}).")
    elif lora_loss > base_loss:
        headline = (f"General ability got worse: mean loss {lora_loss:.3f} vs {base_loss:.3f} "
                    f"({test['increased']} records worse, {test['decreased']} better, "
                    f"sign-test p = {test['p_value']:.3g}).")
    else:
        headline = (f"General ability got *better* ({lora_loss:.3f} vs {base_loss:.3f}, "
                    f"sign-test p = {test['p_value']:.3g}) — check that the general set is not leaking "
                    "the task's phrasing before believing it.")
    return (headline + " This is a 24-record synthetic set scored by loss, not a benchmark: it can show "
                       "that general ability was not obviously destroyed, never that it was preserved.")
