"""
Unified CLI Entry Point for MLX Model Training & Evals Tutorial.
"""

import sys
import subprocess
from rich.console import Console
from rich.panel import Panel

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
    console.print("  [bold yellow]prepare[/bold yellow]   : Generate synthetic JSON tool-calling dataset splits")
    console.print("  [bold yellow]train[/bold yellow]     : Run MLX LoRA fine-tuning on Metal")
    console.print("  [bold yellow]eval[/bold yellow]      : Execute 4-pillar evaluation suite (PPL, Schema, Exact Match)")
    console.print("  [bold yellow]benchmark[/bold yellow] : Profile generation speed (tok/s) and Metal peak memory")
    console.print("  [bold yellow]fuse[/bold yellow]      : Fuse LoRA adapter into base model weights for zero-overhead")
    console.print("  [bold yellow]serve[/bold yellow]     : Launch local OpenAI-compatible API server via mlx_lm.server")
    console.print("  [bold yellow]notebook[/bold yellow]  : Open the interactive Jupyter Notebook (tutorial.ipynb)\n")


def main():
    if len(sys.argv) < 2:
        show_menu()
        console.print("[dim]Usage: uv run python main.py <command>[/dim]")
        return

    cmd = sys.argv[1].lower()
    if cmd == "prepare":
        subprocess.run(["uv", "run", "python", "data/prepare_dataset.py"])
    elif cmd == "train":
        args = ["uv", "run", "python", "src/train.py"] + sys.argv[2:]
        subprocess.run(args)
    elif cmd == "eval":
        args = ["uv", "run", "python", "src/evaluate.py"] + sys.argv[2:]
        subprocess.run(args)
    elif cmd == "benchmark":
        args = ["uv", "run", "python", "src/benchmark.py"] + sys.argv[2:]
        subprocess.run(args)
    elif cmd == "fuse":
        subprocess.run([
            "uv", "run", "python", "-m", "mlx_lm.fuse",
            "--model", "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "--adapter-path", "artifacts/adapters",
            "--save-path", "artifacts/fused_model"
        ])
    elif cmd == "serve":
        port = sys.argv[2] if len(sys.argv) > 2 else "8080"
        model_path = "artifacts/fused_model"
        console.print(f"[bold green]Starting MLX Server on http://localhost:{port}...[/bold green]")
        subprocess.run(["uv", "run", "python", "-m", "mlx_lm.server", "--model", model_path, "--port", port])
    elif cmd == "notebook":
        subprocess.run(["uv", "run", "jupyter", "lab", "tutorial.ipynb"])
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        show_menu()


if __name__ == "__main__":
    main()
