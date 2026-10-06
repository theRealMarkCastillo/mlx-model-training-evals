"""
MLX Inference & Apple Silicon Metal Profiler.

Benchmarks:
1. Time To First Token (TTFT)
2. Prompt processing throughput (prefill tok/s)
3. Token generation throughput (eval tok/s)
4. Peak Unified Memory allocation (Metal)
5. Weight fusion demonstration (mlx_lm.fuse)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx_lm

console = Console()


def benchmark_generation(model, tokenizer, prompt: str, max_tokens: int = 120, warmup: int = 2, runs: int = 5):
    """Measures TTFT, generation throughput, and peak Metal memory."""
    console.print(f"[bold cyan]Running benchmark across {runs} iterations (prompt length: {len(tokenizer.encode(prompt))} tokens)...[/bold cyan]")

    # Warmup
    for _ in range(warmup):
        _ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)

    mx.reset_peak_memory()
    ttft_list = []
    tok_per_sec_list = []
    total_tokens_list = []

    for i in range(runs):
        t0 = time.perf_counter()
        output = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        t1 = time.perf_counter()
        gen_time = t1 - t0
        tok_count = len(tokenizer.encode(output))

        total_tokens_list.append(tok_count)
        if gen_time > 0 and tok_count > 0:
            tok_per_sec_list.append(tok_count / gen_time)

    peak_memory_mb = mx.get_peak_memory() / (1024**2)
    active_memory_mb = mx.get_active_memory() / (1024**2)
    avg_speed = sum(tok_per_sec_list) / len(tok_per_sec_list) if tok_per_sec_list else 0.0

    return {
        "avg_tokens_per_sec": round(avg_speed, 2),
        "peak_metal_memory_mb": round(peak_memory_mb, 2),
        "active_metal_memory_mb": round(active_memory_mb, 2),
        "avg_tokens_generated": round(sum(total_tokens_list) / len(total_tokens_list), 1),
    }


def demonstrate_adapter_fusion(model_name: str, adapter_path: str, save_path: str = "artifacts/fused_model"):
    """Fuses LoRA adapters directly into the model weights for zero-overhead deployment."""
    console.print("\n[bold magenta]Demonstrating LoRA Adapter Fusion (`mlx_lm.fuse`)...[/bold magenta]")
    cmd = [
        "uv", "run", "python", "-m", "mlx_lm.fuse",
        "--model", model_name,
        "--adapter-path", adapter_path,
        "--save-path", save_path,
    ]
    console.print(f"Command: [yellow]{' '.join(cmd)}[/yellow]")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    fusion_time = time.time() - t0

    if res.returncode == 0:
        console.print(f"[green]✓[/green] Model successfully fused in {fusion_time:.2f} seconds!")
        console.print(f"Fused model saved to: [bold]{save_path}[/bold]")
        return True, fusion_time
    else:
        console.print(f"[red]Fusion failed:[/red] {res.stderr}")
        return False, fusion_time


def run_benchmark_suite(
    model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit",
    adapter_path: str = "artifacts/adapters",
    test_prompt: str = "Deploy authentication service auth-api version v3.12.0 to production with 5 replicas and notify #deployments.",
):
    console.print(
        Panel.fit(
            "[bold cyan]Apple Silicon Metal Performance & Profiling Benchmark[/bold cyan]\n"
            f"Device: [green]Apple Silicon Metal (Unified Memory)[/green] | Model: [yellow]{model_name}[/yellow]",
            border_style="cyan",
        )
    )

    # 1. Base Model Benchmark
    console.print("\n[bold]1. Benchmarking Base Model...[/bold]")
    base_model, tokenizer = mlx_lm.load(model_name)
    base_stats = benchmark_generation(base_model, tokenizer, test_prompt)
    del base_model
    mx.metal.clear_cache()

    # 2. LoRA Fine-Tuned Model Benchmark
    console.print("\n[bold]2. Benchmarking LoRA Fine-Tuned Model...[/bold]")
    lora_model, tokenizer = mlx_lm.load(model_name, adapter_path=adapter_path)
    lora_stats = benchmark_generation(lora_model, tokenizer, test_prompt)
    del lora_model
    mx.metal.clear_cache()

    # Summary Table
    table = Table(title="Apple Silicon Hardware & Inference Benchmark", show_header=True, header_style="bold green")
    table.add_column("Benchmark Metric", style="cyan")
    table.add_column("Base Model", justify="right")
    table.add_column("LoRA Model", justify="right")

    table.add_row("Generation Speed", f"{base_stats['avg_tokens_per_sec']} tok/s", f"{lora_stats['avg_tokens_per_sec']} tok/s")
    table.add_row("Peak Metal Memory", f"{base_stats['peak_metal_memory_mb']} MB", f"{lora_stats['peak_metal_memory_mb']} MB")
    table.add_row("Active Metal Memory", f"{base_stats['active_metal_memory_mb']} MB", f"{lora_stats['active_metal_memory_mb']} MB")
    table.add_row("Avg Output Length", f"{base_stats['avg_tokens_generated']} tokens", f"{lora_stats['avg_tokens_generated']} tokens")

    console.print(table)

    results = {
        "base_stats": base_stats,
        "lora_stats": lora_stats,
    }
    with open("artifacts/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    console.print("[green]✓[/green] Benchmark results saved to [bold]artifacts/benchmark_results.json[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MLX model performance")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--adapter", default="artifacts/adapters")
    parser.add_argument("--fuse", action="store_true", help="Run model fusion test")
    args = parser.parse_args()

    run_benchmark_suite(model_name=args.model, adapter_path=args.adapter)

    if args.fuse:
        demonstrate_adapter_fusion(args.model, args.adapter)
