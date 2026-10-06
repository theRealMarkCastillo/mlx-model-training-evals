"""
MLX LoRA Fine-Tuning Runner.

Executes LoRA/QLoRA training on Apple Silicon with Metal acceleration,
real-time metrics logging, loss tracking, memory profiling, and curve plotting.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import mlx.core as mx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx_lm.lora as lora
from mlx_lm.tuner.callbacks import TrainingCallback

console = Console()


class MetricsLoggerCallback(TrainingCallback):
    """Logs training and validation loss progression for analysis and plotting."""

    def __init__(self):
        super().__init__()
        self.history: List[Dict[str, Any]] = []
        self.train_losses: List[Dict[str, float]] = []
        self.val_losses: List[Dict[str, float]] = []
        self.start_time = time.time()

    def on_train_loss_report(self, train_info: dict):
        it = train_info.get("iteration", 0)
        loss = train_info.get("train_loss", 0.0)
        tok_s = train_info.get("tok/s", 0.0)
        elapsed = time.time() - self.start_time
        record = {
            "type": "train",
            "iteration": it,
            "loss": float(loss),
            "tok_per_sec": float(tok_s),
            "elapsed_seconds": round(elapsed, 2),
        }
        self.history.append(record)
        self.train_losses.append({"iteration": it, "loss": float(loss)})

    def on_val_loss_report(self, val_info: dict):
        it = val_info.get("iteration", 0)
        loss = val_info.get("val_loss", 0.0)
        elapsed = time.time() - self.start_time
        record = {
            "type": "val",
            "iteration": it,
            "loss": float(loss),
            "elapsed_seconds": round(elapsed, 2),
        }
        self.history.append(record)
        self.val_losses.append({"iteration": it, "loss": float(loss)})


def plot_loss_curve(callback: MetricsLoggerCallback, output_path: Path):
    """Generates and saves a training/validation loss curve plot."""
    if not callback.train_losses:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    train_iters = [x["iteration"] for x in callback.train_losses]
    train_vals = [x["loss"] for x in callback.train_losses]
    ax.plot(train_iters, train_vals, label="Train Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)

    if callback.val_losses:
        val_iters = [x["iteration"] for x in callback.val_losses]
        val_vals = [x["loss"] for x in callback.val_losses]
        ax.plot(val_iters, val_vals, label="Validation Loss", color="#ff7f0e", linewidth=2.5, marker="s", markersize=6)

    ax.set_title("MLX LoRA Fine-Tuning Loss Curve (Apple Silicon Metal)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    console.print(f"[green]✓[/green] Saved loss curve plot to [bold]{output_path}[/bold]")


def run_training(config_path: str = "config/lora_config.yaml", iters_override: int = None):
    console.print(
        Panel.fit(
            "[bold cyan]Apple Silicon MLX LoRA Fine-Tuning Pipeline[/bold cyan]\n"
            f"Config: [yellow]{config_path}[/yellow] | Metal acceleration: [green]{mx.metal.is_available()}[/green]",
            border_style="cyan",
        )
    )

    # Initial memory benchmark
    mx.reset_peak_memory()
    start_mem_mb = mx.get_active_memory() / (1024**2)

    parser = lora.build_parser()
    args_list = ["-c", config_path, "--train"]
    if iters_override is not None:
        args_list.extend(["--iters", str(iters_override)])

    parsed_args = parser.parse_args(args_list)
    args_dict = vars(parsed_args)

    # Load YAML configuration
    if parsed_args.config and os.path.exists(parsed_args.config):
        with open(parsed_args.config, "r") as f:
            cfg = lora.yaml.load(f, lora.yaml_loader)
        for k, v in cfg.items():
            if args_dict.get(k) is None:
                args_dict[k] = v

    # Apply defaults
    for k, v in lora.CONFIG_DEFAULTS.items():
        if args_dict.get(k) is None:
            args_dict[k] = v

    import types
    args = types.SimpleNamespace(**args_dict)

    # Setup custom metrics callback
    callback = MetricsLoggerCallback()
    start_time = time.time()

    console.print("[bold blue]Starting LoRA training run...[/bold blue]")
    lora.run(args, training_callback=callback)

    total_time = time.time() - start_time
    peak_mem_mb = mx.get_peak_memory() / (1024**2)
    final_active_mem_mb = mx.get_active_memory() / (1024**2)

    # Save artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    history_file = artifacts_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(
            {
                "training_time_seconds": round(total_time, 2),
                "peak_memory_mb": round(peak_mem_mb, 2),
                "final_active_memory_mb": round(final_active_mem_mb, 2),
                "history": callback.history,
            },
            f,
            indent=2,
        )

    plot_loss_curve(callback, artifacts_dir / "loss_curve.png")

    # Summary table
    table = Table(title="Training Run Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    initial_loss = callback.train_losses[0]["loss"] if callback.train_losses else "N/A"
    final_loss = callback.train_losses[-1]["loss"] if callback.train_losses else "N/A"
    best_val_loss = min([x["loss"] for x in callback.val_losses]) if callback.val_losses else "N/A"

    table.add_row("Total Training Time", f"{total_time:.2f} s")
    table.add_row("Initial Train Loss", f"{initial_loss:.4f}" if isinstance(initial_loss, float) else "N/A")
    table.add_row("Final Train Loss", f"{final_loss:.4f}" if isinstance(final_loss, float) else "N/A")
    table.add_row("Best Validation Loss", f"{best_val_loss:.4f}" if isinstance(best_val_loss, float) else "N/A")
    table.add_row("Peak Metal Memory", f"{peak_mem_mb:.1f} MB")
    table.add_row("Adapters Directory", str(args.adapter_path))

    console.print(table)
    return callback


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Train MLX LoRA model")
    cli_parser.add_argument("-c", "--config", default="config/lora_config.yaml", help="Path to config file")
    cli_parser.add_argument("--iters", type=int, default=None, help="Override training iterations")
    parsed = cli_parser.parse_args()

    run_training(config_path=parsed.config, iters_override=parsed.iters)
