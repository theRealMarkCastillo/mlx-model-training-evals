"""Command-line interface. Each subcommand calls the same functions the notebook uses."""

import argparse
import os
import subprocess
import sys

from rich.console import Console
from rich.table import Table

from src.dataset import positive_int
from src.models import DEFAULT_PRESET, PRESETS, add_preset_argument
from src.runs import REPO_ROOT, latest_path

console = Console()


def _model_selection(parser):
    group = parser.add_mutually_exclusive_group()
    add_preset_argument(group)
    group.add_argument("--model", help="Custom base model path or Hugging Face repository")
    parser.add_argument("--adapter", help="Adapter directory or a directory containing latest.json")
    parser.add_argument("--output-dir", help="Report root (default: artifacts/qwen2.5-<size>)")


def cmd_models(args):
    table = Table(title="Qwen2.5 Instruct 4-bit presets (overrides applied to config/base.yaml)")
    for column in ("Preset", "Model", "Batch", "LoRA layers", "Checkpointing", "Artifacts"):
        table.add_column(column)
    for size, p in PRESETS.items():
        table.add_row(size, p.model, str(p.batch_size), str(p.num_layers), str(p.grad_checkpoint),
                      str(p.output_dir.relative_to(REPO_ROOT)))
    console.print(table)
    console.print("Larger presets trade speed for memory. Measure peak memory with a short run first (--iters 10).")


def cmd_prepare(args):
    from src.dataset import validate_splits
    from src.generate_data import main
    main()
    validate_splits()


def cmd_train(args):
    from src.train import run_training
    overrides = {k: v for k, v in (("rank", args.rank), ("learning_rate", args.learning_rate),
                                   ("num_layers", args.num_layers), ("seed", args.seed)) if v is not None}
    run_training(config_path=args.config, iters_override=args.iters, preset=args.preset,
                 output_dir=args.output_dir, overrides=overrides)


def cmd_eval(args):
    from src.evaluate import run_comprehensive_evaluation
    run_comprehensive_evaluation(
        model_name=args.model, adapter_path=args.adapter, test_jsonl=args.test_file,
        num_eval_samples=args.samples, preset=args.preset, output_dir=args.output_dir,
        fused_path=args.fused, max_tokens=args.max_tokens, variants=args.variants,
        shots=args.shots, challenge=args.challenge, constrained=args.constrained,
        temperature=args.temperature, seed=args.seed, repeats=args.repeats,
    )


def cmd_bfcl(args):
    from src.bfcl_eval import run_bfcl_eval, run_bfcl_irrelevance
    runner = run_bfcl_irrelevance if args.irrelevance else run_bfcl_eval
    runner(
        model_name=args.model, preset=args.preset, records_path=args.records,
        max_tokens=args.max_tokens, max_records=args.samples, output_dir=args.output_dir,
    )


def cmd_forgetting(args):
    from src.forgetting import run_forgetting_check
    run_forgetting_check(
        preset=args.preset, model_name=args.model, adapter_path=args.adapter,
        output_dir=args.output_dir, general_jsonl=args.general_file, max_samples=args.samples,
    )


def cmd_benchmark(args):
    from src.benchmark import run_benchmark_suite
    run_benchmark_suite(
        args.model, args.adapter, preset=args.preset, output_dir=args.output_dir,
        fused_path=args.fused, fuse=args.fuse, runs=args.runs, warmup=args.warmup,
        max_tokens=args.max_tokens, quality_samples=args.samples,
    )


def cmd_fuse(args):
    from src.fuse import fuse_model
    from src.models import resolve_model_paths
    model, adapter, root = resolve_model_paths(args.preset, args.model, args.adapter, args.output_dir)
    console.print(fuse_model(model, adapter, args.save_path or str(root / "fused_model")))


def cmd_ablate(args):
    from src.ablation import run_ablation
    run_ablation(args.param, args.values, preset=args.preset, iters=args.iters,
                 samples=args.samples, output_dir=args.output_dir)


def cmd_show_mask(args):
    from src.dataset import DATA_DIR, load_samples
    records = load_samples(DATA_DIR / f"{args.split}.jsonl")
    if not 0 <= args.index < len(records):
        raise ValueError(f"--index {args.index} is out of range for '{args.split}' ({len(records)} records)")
    from transformers import AutoTokenizer

    from src.explain import loss_mask_tokens, mask_counts, render_mask_rich
    profile = PRESETS[args.preset or DEFAULT_PRESET]
    tokenizer = AutoTokenizer.from_pretrained(profile.model)
    record = records[args.index]
    pairs = loss_mask_tokens(tokenizer, record)
    console.print(render_mask_rich(pairs))
    counts = mask_counts(pairs)
    console.print(f"\n[green]{counts['scored_tokens']}[/green] of {counts['total_tokens']} tokens are scored "
                  f"(highlighted); the other {counts['masked_tokens']} are context only.")


def cmd_show_params(args):
    import mlx_lm

    from src.explain import parameter_summary
    from src.models import resolve_model_paths
    from src.runs import resolve_adapter_source
    model_name, adapter, _ = resolve_model_paths(args.preset, args.model, args.adapter)
    source, _ = resolve_adapter_source(model_name, adapter)
    model, _ = mlx_lm.load(source, adapter_path=adapter)
    s = parameter_summary(model)
    table = Table(title=f"LoRA parameters: {adapter}")
    for column in ("Projection", "Layers", "Rank", "Parameters"):
        table.add_column(column, justify="right")
    for name, entry in s["adapted_projections"].items():
        table.add_row(name, str(entry["layers"]), str(entry["rank"]), f"{entry['parameters']:,}")
    console.print(table)
    console.print(f"Adapter: {s['adapter_parameters']:,} parameters ({s['adapter_megabytes_fp16']:.1f} MB in fp16) = "
                  f"{s['adapter_percent']:.3f}% of {s['base_parameters']:,} base parameters.")


def cmd_inspect(args):
    from src.inspect import run_inspection
    run_inspection(
        args.split, args.index, preset=args.preset, model_name=args.model, adapter_path=args.adapter,
        output_dir=args.output_dir, variant=args.variant, max_tokens=args.max_tokens,
        top_k=args.top_k, shots=args.shots,
    )


def cmd_show_data(args):
    from src.show_data import print_dataset_summary
    print_dataset_summary(with_token_stats=args.tokens, examples=args.examples, preset=args.preset)


def cmd_toy_train(args):
    from src.mini_train import run_toy_training
    overrides = {k: v for k, v in (("iters", args.iters), ("seed", args.seed),
                                   ("rank", args.rank), ("lr", args.lr)) if v is not None}
    run_toy_training(mode="full" if args.full else "lora", output_root=args.output_root, **overrides)


def cmd_serve(args):
    profile = PRESETS[args.preset or DEFAULT_PRESET]
    model_path = args.model or (profile.model if args.base else latest_path(profile.fused_path, required=True))
    console.print(f"[bold green]Starting MLX server on http://localhost:{args.port} with {model_path}[/bold green]")
    console.print("Send SYSTEM_PROMPT from src/schema.py with each chat request; the server does not add it.")
    return subprocess.run([sys.executable, "-m", "mlx_lm.server", "--model", model_path, "--port", str(args.port)]).returncode


def cmd_notebook(args):
    return subprocess.run([sys.executable, "-m", "jupyter", "lab", str(REPO_ROOT / "tutorial.ipynb")]).returncode


def build_parser():
    parser = argparse.ArgumentParser(prog="main.py", description="Local LoRA training and tool-call evaluation with Apple MLX")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("models", help="List model presets").set_defaults(func=cmd_models)
    sub.add_parser("prepare", help="Generate dataset splits and challenge sets").set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="Train a LoRA adapter")
    group = p.add_mutually_exclusive_group()
    group.add_argument("-c", "--config", help="Standalone MLX-LM YAML config (instead of a preset)")
    add_preset_argument(group)
    p.add_argument("--output-dir", help="Report root")
    p.add_argument("--iters", type=positive_int)
    p.add_argument("--rank", type=positive_int, help="Override LoRA rank")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--seed", type=int, help="Random seed for batch order and LoRA init (default: base config)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="Compare base, few-shot, LoRA (and grammar-constrained or fused) models")
    _model_selection(p)
    p.add_argument("--test-file", help="Chat JSONL holdout (default: data/test.jsonl)")
    p.add_argument("--fused", help="Also evaluate this fused model directory")
    p.add_argument("--samples", type=positive_int, help="Tool-balanced subset size per set (default: all)")
    p.add_argument("--max-tokens", type=positive_int, default=150)
    p.add_argument("--variants", nargs="+", default=["base", "fewshot", "lora"], choices=["base", "fewshot", "lora"])
    p.add_argument("--shots", type=positive_int, default=5, help="Few-shot demonstrations (one per tool at 5)")
    p.add_argument("--challenge", action="store_true", help="Also evaluate data/challenge_*.jsonl")
    p.add_argument("--constrained", action="store_true",
                   help="Also evaluate the base model with grammar-constrained decoding (valid JSON by construction)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy (default). >0 samples: one stochastic draw per record, seeded")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed when --temperature > 0")
    p.add_argument("--repeats", type=positive_int, default=1,
                   help="Draws per record; >1 adds a pass@k summary of what retrying recovers")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("bfcl", help="BFCL pilot: base model vs grammar-constrained decoding on real schemas")
    add_preset_argument(p)
    p.add_argument("--model", help="Custom base model path or Hugging Face repository (default: the preset)")
    p.add_argument("--records", help="Converted BFCL JSONL (default: data/bfcl_simple.jsonl, or data/bfcl_irrelevance.jsonl)")
    p.add_argument("--samples", type=positive_int, help="Subset size for a quick smoke run (default: all)")
    p.add_argument("--max-tokens", type=positive_int, default=200)
    p.add_argument("--irrelevance", action="store_true",
                   help="Measure hallucinated-call rate on the irrelevance category (no ground truth)")
    p.add_argument("--output-dir", help="Directory for the JSON report (default: no file)")
    p.set_defaults(func=cmd_bfcl)

    p = sub.add_parser("forgetting", help="Compare base vs LoRA on general requests (catastrophic forgetting)")
    _model_selection(p)
    p.add_argument("--general-file", help="Chat JSONL with free-form answers (default: data/general.jsonl)")
    p.add_argument("--samples", type=positive_int, help="Subset of the general set (default: all)")
    p.set_defaults(func=cmd_forgetting)

    p = sub.add_parser("benchmark", help="Measure latency, throughput, and memory")
    _model_selection(p)
    fusion = p.add_mutually_exclusive_group()
    fusion.add_argument("--fuse", action="store_true", help="Fuse, benchmark, and evaluate all three variants")
    fusion.add_argument("--fused", help="Benchmark and evaluate an existing fused model")
    p.add_argument("--runs", type=positive_int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-tokens", type=positive_int, default=120)
    p.add_argument("--samples", type=positive_int, help="Samples for the linked quality evaluation (default: all)")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("fuse", help="Merge the adapter into a standalone model")
    _model_selection(p)
    p.add_argument("--save-path", help="Fusion root (default: <report root>/fused_model)")
    p.set_defaults(func=cmd_fuse)

    p = sub.add_parser("ablate", help="Sweep one hyperparameter: train + evaluate per value")
    add_preset_argument(p)
    from src.ablation import ABLATABLE
    p.add_argument("param", choices=sorted(ABLATABLE))
    p.add_argument("values", nargs="+")
    p.add_argument("--iters", type=positive_int, help="Training iterations per point (default: base config)")
    p.add_argument("--samples", type=positive_int)
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("show-mask", help="Show which tokens the training loss scores")
    add_preset_argument(p)
    p.add_argument("--split", default="train", choices=["train", "valid", "test"])
    p.add_argument("--index", type=int, default=0)
    p.set_defaults(func=cmd_show_mask)

    p = sub.add_parser("show-params", help="Count trainable LoRA parameters in the latest adapter")
    _model_selection(p)
    p.set_defaults(func=cmd_show_params)

    p = sub.add_parser("inspect", help="Trace one greedy decode: per-token probabilities and where it diverged")
    _model_selection(p)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--variant", choices=["base", "fewshot", "lora"], default="lora")
    p.add_argument("--max-tokens", type=positive_int, default=150)
    p.add_argument("--top-k", type=positive_int, default=5, help="Alternatives listed per step")
    p.add_argument("--shots", type=positive_int, default=5, help="Few-shot demonstrations when --variant fewshot")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("show-data", help="Offline summary of every dataset split (counts, balance, examples)")
    add_preset_argument(p)
    p.add_argument("--tokens", action="store_true", help="Also count tokens per split (downloads the tokenizer)")
    p.add_argument("--examples", type=int, default=1, help="Example records printed per tool (default: 1)")
    p.set_defaults(func=cmd_show_data)

    p = sub.add_parser("toy-train", help="From-scratch LoRA loop on a toy model (seconds, no downloads)")
    p.add_argument("--iters", type=positive_int, help="Training steps (default: 300)")
    p.add_argument("--seed", type=int, help="Random seed (default: 7)")
    p.add_argument("--rank", type=positive_int, help="LoRA rank (default: 4)")
    p.add_argument("--lr", type=float, help="Learning rate (default: 0.1)")
    p.add_argument("--full", action="store_true", help="Full fine-tuning instead of LoRA")
    p.add_argument("--output-root", help="Artifact root (default: artifacts/toy)")
    p.set_defaults(func=cmd_toy_train)

    p = sub.add_parser("serve", help="OpenAI-compatible server via mlx_lm.server")
    add_preset_argument(p)
    p.add_argument("--port", type=int, default=8080)
    which = p.add_mutually_exclusive_group()
    which.add_argument("--model", help="Explicit model path or Hugging Face repository")
    which.add_argument("--base", action="store_true", help="Serve the preset's base model")
    p.set_defaults(func=cmd_serve)

    sub.add_parser("notebook", help="Open tutorial.ipynb in JupyterLab").set_defaults(func=cmd_notebook)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        result = args.func(args)
    except MemoryError as exc:
        if os.environ.get("MLX_EVALS_DEBUG"):
            raise
        console.print(f"[red]Out of memory:[/red] {exc}")
        console.print("Try a smaller preset, fewer iterations (`--iters 10`), or closing other GPU-heavy apps.")
        console.print("The failed run was recorded under artifacts/ (see its manifest.json for the full traceback).")
        console.print("[dim]Set MLX_EVALS_DEBUG=1 for the full traceback.[/dim]")
        return 1
    except (FileNotFoundError, ValueError) as exc:
        if os.environ.get("MLX_EVALS_DEBUG"):
            raise
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        console.print("[dim]Set MLX_EVALS_DEBUG=1 for the full traceback.[/dim]")
        return 2
    return result if isinstance(result, int) else 0
