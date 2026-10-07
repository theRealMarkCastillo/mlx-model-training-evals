"""
Unified CLI Entry Point for MLX Model Training & Evals Tutorial.
"""

import sys
import subprocess
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.runs import latest_path
from src.models import PRESETS, add_preset_argument

console = Console()


def show_menu():
    console.print(
        Panel.fit(
            "[bold cyan]MLX Local Model Fine-Tuning & Comprehensive Evals[/bold cyan]\n"
            "Apple Silicon Metal Accelerated Framework",
            border_style="cyan",
        )
    )
    console.print("Available commands:")
    console.print("  [bold yellow]models[/bold yellow]    : List Qwen model presets and training settings")
    console.print("  [bold yellow]prepare[/bold yellow]   : Generate synthetic JSON tool-calling dataset splits")
    console.print("  [bold yellow]train[/bold yellow]     : Run MLX LoRA fine-tuning on Metal")
    console.print("  [bold yellow]eval[/bold yellow]      : Execute 4-pillar evaluation suite (PPL, Schema, Exact Match)")
    console.print("  [bold yellow]benchmark[/bold yellow] : Profile generation speed (tok/s) and Metal peak memory")
    console.print("  [bold yellow]fuse[/bold yellow]      : Fuse LoRA adapter into a standalone model")
    console.print("  [bold yellow]serve[/bold yellow]     : Launch local OpenAI-compatible API server via mlx_lm.server")
    console.print("  [bold yellow]notebook[/bold yellow]  : Open the interactive Jupyter Notebook (tutorial.ipynb)\n")


def main():
    if len(sys.argv) < 2:
        show_menu()
        console.print("[dim]Usage: uv run python main.py <command>[/dim]")
        return

    cmd = sys.argv[1].lower()
    if cmd == "models":
        table = Table(title="Qwen2.5 Instruct 4-bit presets")
        for column in ("Preset", "Batch", "LoRA layers", "Checkpointing", "Artifacts"):
            table.add_column(column)
        for size, profile in PRESETS.items():
            table.add_row(size, str(profile.batch_size), str(profile.num_layers),
                          str(profile.grad_checkpoint), str(profile.output_dir))
        console.print(table)
        console.print("Models: mlx-community/Qwen2.5-<SIZE>-Instruct-4bit (uppercase size, e.g. 14B).")
        console.print("Use --preset SIZE with train, eval, benchmark, fuse, or serve.")
        console.print("Larger presets trade speed for lower memory. Measure peak memory with a short training run first.")
    elif cmd == "prepare":
        return subprocess.run([sys.executable, "data/prepare_dataset.py"]).returncode
    elif cmd == "train":
        args = [sys.executable, "src/train.py"] + sys.argv[2:]
        return subprocess.run(args).returncode
    elif cmd == "eval":
        args = [sys.executable, "src/evaluate.py"] + sys.argv[2:]
        return subprocess.run(args).returncode
    elif cmd == "benchmark":
        args = [sys.executable, "src/benchmark.py"] + sys.argv[2:]
        return subprocess.run(args).returncode
    elif cmd == "fuse":
        return subprocess.run([sys.executable, "src/fuse.py", *sys.argv[2:]]).returncode
    elif cmd == "serve":
        parser = argparse.ArgumentParser(description="Serve a fused model or a base model")
        add_preset_argument(parser)
        parser.add_argument("port", nargs="?", help="Legacy positional port")
        parser.add_argument("--port", dest="named_port")
        parser.add_argument("--model", help="Explicit model path or Hugging Face repository")
        parser.add_argument("--base", action="store_true", help="Try the preset's base model before training")
        args = parser.parse_args(sys.argv[2:])
        port = args.named_port or args.port or "8080"
        profile = PRESETS[args.preset or "3b"]
        model_path = args.model or (profile.model if args.base else latest_path(profile.fused_path))
        console.print(f"[bold green]Starting MLX Server on http://localhost:{port}...[/bold green]")
        return subprocess.run([sys.executable, "-m", "mlx_lm.server", "--model", model_path, "--port", port]).returncode
    elif cmd == "notebook":
        return subprocess.run([sys.executable, "-m", "jupyter", "lab", "tutorial.ipynb"]).returncode
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        show_menu()
        return 2


if __name__ == "__main__":
    sys.exit(main())
