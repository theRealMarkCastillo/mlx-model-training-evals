"""
MLX LoRA Fine-Tuning Runner.

Executes LoRA/QLoRA training on Apple Silicon with Metal acceleration,
real-time metrics logging, loss tracking, memory profiling, and curve plotting.
"""

import argparse
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
from src.models import PRESETS, add_preset_argument
from src.dataset import positive_int, validate_splits
from src.runs import resolve_source, file_identity, directory_identity, new_run, finish_run, write_json

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
        tok_s = train_info.get("tokens_per_second", 0.0)
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


def run_training(config_path: str = None, iters_override: int = None, *, preset: str = None, output_dir: str = None):
    if config_path is not None and preset is not None:
        raise ValueError("Choose a config file or a preset, not both.")
    profile = PRESETS[preset or "3b"]
    config_path = config_path or profile.config_path
    artifacts_dir = Path(output_dir) if output_dir is not None else profile.output_dir
    console.print(
        Panel.fit(
            "[bold cyan]Apple Silicon MLX LoRA Fine-Tuning Pipeline[/bold cyan]\n"
            f"Config: [yellow]{config_path}[/yellow] | Metal acceleration: [green]{mx.metal.is_available()}[/green]",
            border_style="cyan",
        )
    )

    # Initial memory benchmark
    mx.reset_peak_memory()

    parser = lora.build_parser()
    args_list = ["-c", config_path, "--train"]
    if iters_override is not None:
        args_list.extend(["--iters", str(iters_override)])

    parsed_args = parser.parse_args(args_list)
    args_dict = vars(parsed_args)

    # Fail before any model loading if the requested configuration is absent or invalid.
    with open(config_path, "r") as source:
        cfg = lora.yaml.load(source, lora.yaml_loader)
    if not isinstance(cfg, dict):
        raise ValueError("Training config must be a YAML mapping")
    for key in ("model", "data", "adapter_path"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"Config requires a nonempty {key}")
    unknown = set(cfg) - set(lora.CONFIG_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
    args_dict.update(cfg)
    args_dict["train"] = True
    if iters_override is not None:
        args_dict["iters"] = iters_override

    # Apply defaults
    for k, v in lora.CONFIG_DEFAULTS.items():
        if args_dict.get(k) is None:
            args_dict[k] = v

    import types
    args = types.SimpleNamespace(**args_dict)

    for key in ("iters", "batch_size", "max_seq_length", "steps_per_report", "steps_per_eval", "save_every"):
        value = getattr(args, key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if args.fine_tune_type != "lora" or not args.mask_prompt:
        raise ValueError("This workflow requires fine_tune_type=lora and mask_prompt=true")
    if (type(args.learning_rate) not in (int, float) or args.learning_rate <= 0
            or type(args.num_layers) is not int or args.num_layers == 0 or args.num_layers < -1):
        raise ValueError("Invalid learning_rate or num_layers")
    if args.optimizer not in ("adam", "adamw", "muon", "sgd", "adafactor"):
        raise ValueError("Unsupported optimizer")
    params = args.lora_parameters
    if (not isinstance(params, dict) or type(params.get("rank")) is not int or params["rank"] <= 0
            or type(params.get("scale")) not in (int, float) or params["scale"] <= 0
            or type(params.get("dropout")) not in (int, float) or not 0 <= params["dropout"] < 1):
        raise ValueError("Invalid LoRA rank, scale, or dropout")
    if type(args.val_batches) is not int or args.val_batches == 0 or args.val_batches < -1:
        raise ValueError("val_batches must be positive or -1")
    validate_splits(args.data)
    configured_adapter = Path(args.adapter_path)
    if output_dir is None and preset is None and config_path != PRESETS["3b"].config_path:
        artifacts_dir = configured_adapter.parent
    requested_model = args.model
    model_source, source_identity = resolve_source(requested_model)
    run_dir, manifest = new_run(
        artifacts_dir, "training", model=requested_model, model_source=source_identity,
        config=dict(vars(args)),
        datasets=[file_identity(Path(args.data) / f"{split}.jsonl") for split in ("train", "valid", "test")],
    )
    # Store weights in the run itself. Publish the latest pointer only after success.
    args.adapter_path = str(run_dir / "adapters")
    args.model = model_source
    callback = MetricsLoggerCallback()
    callback.run_dir = run_dir
    callback.adapter_path = args.adapter_path
    start_time = time.perf_counter()

    console.print("[bold blue]Starting LoRA training run...[/bold blue]")
    # MLX-LM 0.32.0 lora.run discards user callbacks; train_model preserves them.
    model, tokenizer = lora.load(
        model_source, tokenizer_config={"trust_remote_code": args.trust_remote_code},
        trust_remote_code=args.trust_remote_code,
    )
    train_set, valid_set, _ = lora.load_dataset(args, tokenizer)
    for split in (train_set, valid_set):
        if len(split) < args.batch_size:
            raise ValueError("Each training/validation split must contain at least one full batch")
        for record in split:
            tokens, offset = split.process(record)
            if len(tokens) > args.max_seq_length or offset >= len(tokens):
                raise ValueError("max_seq_length must preserve every training/validation response")
    import numpy as np
    np.random.seed(args.seed)
    reporting = lora.get_reporting_callbacks(
        args.report_to, project_name=args.project_name, log_dir=args.adapter_path, config=vars(args),
    )
    class CombinedCallback(TrainingCallback):
        def on_train_loss_report(self, info):
            callback.on_train_loss_report(info)
            if reporting:
                reporting.on_train_loss_report(info)

        def on_val_loss_report(self, info):
            callback.on_val_loss_report(info)
            if reporting:
                reporting.on_val_loss_report(info)

    lora.train_model(args, model, train_set, valid_set, CombinedCallback())
    if not callback.train_losses or not callback.val_losses:
        raise RuntimeError("Training completed without required loss telemetry")
    total_time = time.perf_counter() - start_time
    peak_mem_mb = mx.get_peak_memory() / (1024**2)
    final_active_mem_mb = mx.get_active_memory() / (1024**2)
    history_file = run_dir / "training_history.json"
    write_json(history_file, {
        "model": requested_model, "adapter": args.adapter_path,
        "run_id": manifest["run_id"], "training_time_seconds": round(total_time, 2),
        "peak_memory_mb": round(peak_mem_mb, 2),
        "final_active_memory_mb": round(final_active_mem_mb, 2), "history": callback.history,
    })
    manifest["adapter"] = directory_identity(args.adapter_path)
    manifest["adapter_path"] = args.adapter_path
    artifacts_dir = run_dir
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
    finish_run(run_dir.parent.parent, run_dir, manifest)
    write_json(configured_adapter / "latest.json", {
        "path": args.adapter_path, "model": requested_model, "run_id": manifest["run_id"],
    })
    return callback


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Train MLX LoRA model")
    selection = cli_parser.add_mutually_exclusive_group()
    selection.add_argument("-c", "--config", help="Path to config file")
    add_preset_argument(selection)
    cli_parser.add_argument("--output-dir", help="Directory for training history and plots")
    cli_parser.add_argument("--iters", type=positive_int, default=None, help="Override training iterations")
    parsed = cli_parser.parse_args()

    run_training(config_path=parsed.config, iters_override=parsed.iters, preset=parsed.preset, output_dir=parsed.output_dir)
