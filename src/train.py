"""MLX LoRA training with loss telemetry, memory measurement, and run provenance."""

from pathlib import Path
import time
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx_lm.lora as lora
from mlx_lm.tuner.callbacks import TrainingCallback
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import load_training_config
from src.dataset import validate_splits
from src.explain import parameter_summary
from src.models import PRESETS, DEFAULT_PRESET
from src.runs import resolve_source, file_identity, directory_identity, new_run, finish_run, record_failure, write_json

console = Console()


class MetricsLoggerCallback(TrainingCallback):
    """Collects train/validation loss reports for the history file and plot."""

    def __init__(self):
        super().__init__()
        self.history = []
        self.train_losses = []
        self.val_losses = []
        self.start_time = time.time()

    def on_train_loss_report(self, train_info):
        record = {
            "type": "train", "iteration": train_info.get("iteration", 0),
            "loss": float(train_info.get("train_loss", 0.0)),
            # MLX counts only loss-scored (assistant) tokens here, not prompt tokens.
            "scored_tok_per_sec": float(train_info.get("tokens_per_second", 0.0)),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
        }
        self.history.append(record)
        self.train_losses.append({"iteration": record["iteration"], "loss": record["loss"]})

    def on_val_loss_report(self, val_info):
        record = {
            "type": "val", "iteration": val_info.get("iteration", 0),
            "loss": float(val_info.get("val_loss", 0.0)),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
        }
        self.history.append(record)
        self.val_losses.append({"iteration": record["iteration"], "loss": record["loss"]})


def loss_summary(callback):
    """Numbers that answer "did it learn, and did it overfit?"."""
    best = min(callback.val_losses, key=lambda r: r["loss"])
    final_val = callback.val_losses[-1]["loss"]
    return {
        "initial_train_loss": callback.train_losses[0]["loss"],
        "final_train_loss": callback.train_losses[-1]["loss"],
        "initial_val_loss": callback.val_losses[0]["loss"],
        "final_val_loss": final_val,
        "best_val_loss": best["loss"],
        "best_val_iteration": best["iteration"],
        "val_rise_after_best": final_val - best["loss"],
        # Validation loss rising after its minimum is the classic overfitting signature. Judge the rise
        # against the total improvement: near zero loss, tiny absolute wiggles are large relative ones.
        "overfitting_suspected": final_val - best["loss"] > 0.1 * (callback.val_losses[0]["loss"] - best["loss"]),
    }


def plot_loss_curve(callback, output_path):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot([x["iteration"] for x in callback.train_losses], [x["loss"] for x in callback.train_losses],
            label="Train loss (recent batches)", color="#1f77b4", linewidth=2, marker="o", markersize=3)
    if callback.val_losses:
        ax.plot([x["iteration"] for x in callback.val_losses], [x["loss"] for x in callback.val_losses],
                label="Validation loss (held-out wording)", color="#ff7f0e", linewidth=2.5, marker="s", markersize=5)
        best = min(callback.val_losses, key=lambda r: r["loss"])
        ax.axvline(best["iteration"], color="#ff7f0e", linestyle=":", alpha=0.6)
        ax.annotate(f"best val {best['loss']:.3f}", (best["iteration"], best["loss"]),
                    xytext=(6, 12), textcoords="offset points", fontsize=8, color="#ff7f0e")
    ax.set_title("LoRA training loss (assistant tokens only)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy (nats/token)")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _check_sequences(args, splits):
    for split in splits:
        if len(split) < args.batch_size:
            raise ValueError("Each training/validation split must contain at least one full batch")
        for record in split:
            tokens, offset = split.process(record)
            if len(tokens) > args.max_seq_length or offset >= len(tokens):
                raise ValueError("max_seq_length must preserve every training/validation response")


def run_training(config_path=None, iters_override=None, *, preset=None, output_dir=None, overrides=None):
    """Train one adapter into a new run directory and point adapter_path/latest.json at it."""
    overrides = dict(overrides or {})
    if iters_override is not None:
        overrides["iters"] = iters_override
    # Validation reads only YAML and local data; nothing is downloaded yet.
    config = load_training_config(config_path, preset, overrides)
    configured_adapter = Path(config["adapter_path"])
    if output_dir is not None:
        artifacts_dir = Path(output_dir)
    elif config_path is not None:
        artifacts_dir = configured_adapter.parent
    else:
        artifacts_dir = PRESETS[preset or DEFAULT_PRESET].output_dir
    validate_splits(config["data"])

    console.print(Panel.fit(
        f"[bold cyan]MLX LoRA training[/bold cyan]  {config['model']}\n"
        f"rank {config['lora_parameters']['rank']} · scale {config['lora_parameters']['scale']} · "
        f"lr {config['learning_rate']} · batch {config['batch_size']} · iters {config['iters']} · "
        f"layers {config['num_layers']} · Metal {mx.metal.is_available()}",
        border_style="cyan",
    ))
    args = types.SimpleNamespace(**{**lora.CONFIG_DEFAULTS, **config, "train": True})
    requested_model = args.model
    model_source, source_identity = resolve_source(requested_model)
    run_dir, manifest = new_run(
        artifacts_dir, "training", model=requested_model, model_source=source_identity,
        config=dict(vars(args)),
        datasets=[file_identity(Path(args.data) / f"{split}.jsonl") for split in ("train", "valid", "test")],
    )
    with record_failure(run_dir, manifest):
        # Weights live in the run itself; the latest pointer is published only after success.
        args.adapter_path = str(run_dir / "adapters")
        args.model = model_source
        callback = MetricsLoggerCallback()
        callback.run_dir = run_dir
        callback.adapter_path = args.adapter_path

        mx.reset_peak_memory()
        start_time = time.perf_counter()
        model, tokenizer = lora.load(
            model_source, tokenizer_config={"trust_remote_code": args.trust_remote_code},
            trust_remote_code=args.trust_remote_code,
        )
        train_set, valid_set, _ = lora.load_dataset(args, tokenizer)
        _check_sequences(args, (train_set, valid_set))
        np.random.seed(args.seed)  # batch order; train_model seeds mx.random for LoRA init and dropout
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

        # MLX-LM 0.32's lora.run discards user callbacks; train_model preserves them.
        console.print("[bold blue]Training...[/bold blue]")
        lora.train_model(args, model, train_set, valid_set, CombinedCallback())
        if not callback.train_losses or not callback.val_losses:
            raise RuntimeError("Training completed without required loss telemetry")
        total_time = time.perf_counter() - start_time
        peak_mem_mb = mx.get_peak_memory() / (1024**2)
        params = parameter_summary(model)
        losses = loss_summary(callback)
        callback.parameters, callback.losses = params, losses

        write_json(run_dir / "training_history.json", {
            "model": requested_model, "adapter": args.adapter_path, "run_id": manifest["run_id"],
            "training_time_seconds": round(total_time, 2), "peak_memory_mb": round(peak_mem_mb, 2),
            "final_active_memory_mb": round(mx.get_active_memory() / (1024**2), 2),
            "examples_seen": args.iters * args.batch_size, "train_records": len(train_set),
            "parameters": params, "loss_summary": losses, "history": callback.history,
        })
        manifest["adapter"] = directory_identity(args.adapter_path)
        manifest["adapter_path"] = args.adapter_path
        plot_loss_curve(callback, run_dir / "loss_curve.png")

        table = Table(title="Training run summary", header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_row("Training time", f"{total_time:.1f} s")
        table.add_row("Examples seen / epochs", f"{args.iters * args.batch_size} / {args.iters * args.batch_size / len(train_set):.1f}")
        table.add_row("Train loss  first → last", f"{losses['initial_train_loss']:.4f} → {losses['final_train_loss']:.4f}")
        table.add_row("Val loss    first → last", f"{losses['initial_val_loss']:.4f} → {losses['final_val_loss']:.4f}")
        table.add_row("Best val loss (iteration)", f"{losses['best_val_loss']:.4f} ({losses['best_val_iteration']})")
        table.add_row("Trainable LoRA parameters", f"{params['adapter_parameters']:,} ({params['adapter_percent']:.3f}% of {params['base_parameters'] / 1e9:.2f}B)")
        table.add_row("Peak Metal memory", f"{peak_mem_mb:,.0f} MB")
        table.add_row("Adapter directory", str(args.adapter_path))
        console.print(table)
        if losses["overfitting_suspected"]:
            console.print("[yellow]Validation loss rose after its minimum: likely overfitting. "
                          "Consider fewer iterations or the checkpoint near the best iteration.[/yellow]")

        finish_run(artifacts_dir, run_dir, manifest)
        write_json(configured_adapter / "latest.json", {
            "path": args.adapter_path, "model": requested_model, "run_id": manifest["run_id"],
        })
    return callback
