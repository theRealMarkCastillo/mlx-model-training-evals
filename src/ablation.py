"""Train and evaluate one adapter per value of a single hyperparameter.

Changing one knob at a time while holding data, seed, and everything else fixed
is the simplest way to build intuition for what each setting does. Each point
is a full training run, so start with a few values and a short schedule.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from src.evaluate import run_comprehensive_evaluation
from src.metrics import format_rate
from src.models import PRESETS, DEFAULT_PRESET
from src.runs import finish_run, new_run, record_failure, write_json
from src.train import run_training

console = Console()
ABLATABLE = {"rank": int, "scale": float, "dropout": float, "learning_rate": float,
             "iters": int, "num_layers": int, "batch_size": int}


def plot_ablation(param, rows, output_path):
    xs = [str(r["value"]) for r in rows]
    fig, (ax, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    em = [100 * r["exact_match_rate"] for r in rows]
    err = [[100 * (r["exact_match_rate"] - r["ci95"][0]) for r in rows],
           [100 * (r["ci95"][1] - r["exact_match_rate"]) for r in rows]]
    ax.errorbar(xs, em, yerr=err, marker="o", capsize=4, color="#1b9e77", label="exact match")
    ax.plot(xs, [100 * r["schema_valid_rate"] for r in rows], marker="s", linestyle="--", color="#7570b3", label="schema valid")
    ax.set_ylim(0, 105)
    ax.set_xlabel(param)
    ax.set_ylabel("% of test samples (95% interval)")
    ax.set_title(f"Holdout quality vs {param}", fontweight="bold")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    ax_loss.plot(xs, [r["final_train_loss"] for r in rows], marker="o", label="final train loss")
    ax_loss.plot(xs, [r["best_val_loss"] for r in rows], marker="s", label="best val loss")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel(param)
    ax_loss.set_title("Loss vs " + param, fontweight="bold")
    ax_loss.legend()
    ax_loss.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_ablation(param, values, *, preset=None, iters=None, samples=None, output_dir=None):
    if param not in ABLATABLE:
        raise ValueError(f"param must be one of {sorted(ABLATABLE)}")
    if not values:
        raise ValueError("Provide at least one value")
    if param == "iters" and iters is not None:
        raise ValueError("When sweeping iters, omit --iters")
    values = [ABLATABLE[param](v) for v in values]
    profile = PRESETS[preset or DEFAULT_PRESET]
    root = Path(output_dir) if output_dir else profile.output_dir
    directory, manifest = new_run(root, "ablation", model=profile.model, param=param, values=values,
                                  iters=iters, samples=samples)
    rows = []
    with record_failure(directory, manifest):
        for value in values:
            console.rule(f"{param} = {value}")
            point = directory / f"{param}-{value}"
            # A private adapter_path keeps ablation runs from replacing the preset's latest adapter.
            training = run_training(preset=preset or DEFAULT_PRESET, iters_override=iters, output_dir=str(point),
                                    overrides={param: value, "adapter_path": str(point / "adapters")})
            report = run_comprehensive_evaluation(
                preset=preset, adapter_path=training.adapter_path, output_dir=str(point),
                num_eval_samples=samples, variants=("lora",), quiet=True,
            )
            m = report["datasets"]["test"]["lora"]
            rows.append({
                "value": value, "exact_match_rate": m["exact_match_rate"], "ci95": m["intervals"]["exact_match_rate"],
                "schema_valid_rate": m["schema_valid_rate"], "test_loss": m["loss"],
                "final_train_loss": training.losses["final_train_loss"], "best_val_loss": training.losses["best_val_loss"],
                "adapter_parameters": training.parameters["adapter_parameters"],
                "training_run": str(training.run_dir), "evaluation_run": report["run_dir"],
            })
        plot_ablation(param, rows, directory / "ablation.png")
        table = Table(title=f"Ablation over {param}")
        for column in (param, "exact match [95%]", "test loss", "best val loss", "LoRA params"):
            table.add_column(column, justify="right")
        for r in rows:
            table.add_row(str(r["value"]), format_rate(r["exact_match_rate"], r["ci95"]), f"{r['test_loss']:.4f}",
                          f"{r['best_val_loss']:.4f}", f"{r['adapter_parameters']:,}")
        console.print(table)
        manifest["results"] = rows
        write_json(directory / "ablation.json", {"param": param, "rows": rows, "run_dir": str(directory)})
        finish_run(root, directory, manifest)
    console.print(f"Ablation plot: {directory / 'ablation.png'}")
    return {"param": param, "rows": rows, "run_dir": str(directory)}
