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
    from src.models import RAM_PROFILES, get_system_ram_gb, ram_to_tier

    ram_gb = get_system_ram_gb()
    tier = getattr(args, "ram", None) or ram_to_tier(ram_gb)
    if tier == "auto":
        tier = ram_to_tier(ram_gb)

    table = Table(title=f"Qwen2.5 Instruct 4-bit presets (Base defaults · Detected system RAM: {ram_gb} GB)")
    for column in ("Preset", "Model", "Batch", "LoRA layers", "Checkpointing", "LR", "Artifacts"):
        table.add_column(column)
    for size, p in PRESETS.items():
        lr_str = f"{p.learning_rate:.0e}".replace("-0", "-") if p.learning_rate is not None else "1e-4"
        table.add_row(size, p.model, str(p.batch_size), str(p.num_layers), str(p.grad_checkpoint), lr_str,
                      str(p.output_dir.relative_to(REPO_ROOT)))
    console.print(table)

    opt_table = Table(title=f"Optimal parameters for {tier.upper()} tier (use `main.py train --optimal` to apply)")
    for column in ("Preset", "Optimal Batch", "Optimal LoRA layers", "Checkpointing", "Fit Status"):
        opt_table.add_column(column)
    for size in PRESETS:
        opt = RAM_PROFILES[tier].get(size)
        if opt:
            opt_table.add_row(
                size, str(opt["batch_size"]), str(opt["num_layers"]), str(opt["grad_checkpoint"]),
                "[green]Fits in RAM[/green]",
            )
        else:
            opt_table.add_row(size, "-", "-", "-", f"[yellow]Exceeds {tier} RAM[/yellow]")
    console.print(opt_table)
    console.print("Pass `--optimal` to `main.py train` to use the tuned parameters for your hardware.")


def cmd_prepare(args):
    from src.dataset import validate_splits
    from src.generate_data import main
    main()
    validate_splits()


def cmd_train(args):
    from src.train import run_training
    overrides = {}
    if getattr(args, "optimal", False) or getattr(args, "ram", None):
        from src.models import optimal_overrides
        tier = args.ram if getattr(args, "ram", None) and args.ram != "auto" else None
        overrides.update(optimal_overrides(args.preset or DEFAULT_PRESET, tier))
    cli_overrides = {k: v for k, v in (("rank", args.rank), ("learning_rate", args.learning_rate),
                                       ("num_layers", args.num_layers), ("seed", args.seed), ("data", args.data),
                                       ("batch_size", args.batch_size), ("grad_checkpoint", args.grad_checkpoint)) if v is not None}
    overrides.update(cli_overrides)
    run_training(config_path=args.config, iters_override=args.iters, preset=args.preset,
                 output_dir=args.output_dir, overrides=overrides, validate=not args.no_validate)


def cmd_dpo_mine(args):
    from src.dpo import run_mining
    run_mining(preset=args.preset, model_name=args.model, adapter_path=args.adapter, output_dir=args.output_dir,
               rollback_count=args.rollback, other_count=args.other, extra_draws=args.extra_draws,
               temperature=args.temperature, max_tokens=args.max_tokens)


def cmd_dpo_mix(args):
    from src.dpo import run_mix
    run_mix(source=args.pairs, output_dir=args.output_dir, preset=args.preset, model_name=args.model, count=args.count)


def cmd_dpo(args):
    from src.dpo import run_dpo
    run_dpo(preset=args.preset, model_name=args.model, adapter_path=args.adapter, output_dir=args.output_dir,
            pairs=args.pairs, iters=args.iters, batch_size=args.batch_size, learning_rate=args.learning_rate,
            beta=args.beta, sft_alpha=args.sft_alpha, steps_per_eval=args.steps_per_eval, seed=args.seed)


def cmd_compare(args):
    from src.compare import parse_named_adapters, run_comparison
    run_comparison(parse_named_adapters(args.adapters), preset=args.preset, model_name=args.model,
                   output_dir=args.output_dir, max_tokens=args.max_tokens, challenge=not args.no_challenge)


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
    if args.irrelevance:
        run_bfcl_irrelevance(
            model_name=args.model, preset=args.preset, records_path=args.records,
            max_tokens=args.max_tokens, max_records=args.samples, output_dir=args.output_dir,
        )
    else:
        run_bfcl_eval(
            model_name=args.model, preset=args.preset, records_path=args.records,
            variants=args.variants, max_tokens=args.max_tokens, max_records=args.samples,
            output_dir=args.output_dir, prompt_style=args.prompt, adapter_path=args.adapter,
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

    m = sub.add_parser("models", help="List model presets and optimal RAM configurations")
    m.add_argument("--ram", choices=["16gb", "32gb", "64gb", "96gb", "auto"], default=None,
                   help="Show optimal parameters for a specific RAM tier (16gb, 32gb, 64gb, 96gb)")
    m.set_defaults(func=cmd_models)
    sub.add_parser("prepare", help="Generate dataset splits and challenge sets").set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="Train a LoRA adapter")
    group = p.add_mutually_exclusive_group()
    group.add_argument("-c", "--config", help="Standalone MLX-LM YAML config (instead of a preset)")
    add_preset_argument(group)
    p.add_argument("--output-dir", help="Report root")
    p.add_argument("--data", help="Directory with train/valid/test.jsonl (default: data/), e.g. a `dpo-mine` sft_plus control")
    p.add_argument("--iters", type=positive_int)
    p.add_argument("--batch-size", type=positive_int, help="Override training batch size")
    p.add_argument("--rank", type=positive_int, help="Override LoRA rank")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--seed", type=int, help="Random seed for batch order and LoRA init (default: base config)")
    checkpoint_group = p.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", dest="grad_checkpoint", action="store_true", default=None,
                                  help="Enable gradient checkpointing (trade compute for memory)")
    checkpoint_group.add_argument("--no-checkpoint", dest="grad_checkpoint", action="store_false", default=None,
                                  help="Disable gradient checkpointing (faster training on high-memory Macs)")
    p.add_argument("--optimal", action="store_true",
                   help="Auto-tune batch size, layers, and checkpointing for detected system RAM")
    p.add_argument("--ram", choices=["16gb", "32gb", "64gb", "96gb", "auto"], default=None,
                   help="Tune parameters for a specific RAM tier (16gb, 32gb, 64gb, 96gb, auto)")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the 5-tool schema pre-flight (for non-synthetic data, e.g. the BFCL split)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("dpo-mine", help="Mine preference pairs from the SFT adapter's own mistakes on fresh prompts")
    _model_selection(p)
    p.add_argument("--rollback", type=positive_int, default=240, help="Fresh rollback prompts to try (default: 240)")
    p.add_argument("--other", type=positive_int, default=60, help="Fresh prompts for the other tools (default: 60)")
    p.add_argument("--extra-draws", type=int, default=0,
                   help="Sampled draws to look for a wrong answer where greedy was right (default: 0)")
    p.add_argument("--temperature", type=float, default=0.8, help="Temperature for --extra-draws")
    p.add_argument("--max-tokens", type=positive_int, default=150)
    p.set_defaults(func=cmd_dpo_mine)

    p = sub.add_parser("dpo-mix", help="Add synthetic in-distribution (dep-NNNN) rollback pairs to a mining run")
    _model_selection(p)
    p.add_argument("--pairs", help="Mining run to extend (default: the latest)")
    p.add_argument("--count", type=positive_int, default=120, help="In-distribution prompts to add (default: 120)")
    p.set_defaults(func=cmd_dpo_mix)

    p = sub.add_parser("dpo", help="Continue the SFT adapter with the DPO loss on mined pairs")
    _model_selection(p)
    p.add_argument("--pairs", help="Directory from `dpo-mine` (default: the latest mining run)")
    p.add_argument("--iters", type=positive_int, default=100)
    p.add_argument("--batch-size", type=positive_int, default=4, help="Pairs per step (each pair is two sequences)")
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1, help="KL leash: higher stays closer to the SFT adapter")
    p.add_argument("--sft-alpha", type=float, default=0.0, help="Weight of an NLL anchor on the chosen response")
    p.add_argument("--steps-per-eval", type=positive_int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_dpo)

    p = sub.add_parser("compare", help="Compare adapters (NAME=DIR ...; the first is the reference) with paired tests")
    add_preset_argument(p)
    p.add_argument("adapters", nargs="+", metavar="NAME=ADAPTER_DIR")
    p.add_argument("--model", help="Custom base model path or Hugging Face repository")
    p.add_argument("--max-tokens", type=positive_int, default=150)
    p.add_argument("--no-challenge", action="store_true", help="Evaluate the holdout only")
    p.add_argument("--output-dir", help="Report root (default: artifacts/qwen2.5-<size>)")
    p.set_defaults(func=cmd_compare)

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
    p.add_argument("--variants", nargs="+", default=["base", "grammar"], choices=["base", "grammar", "lora"])
    p.add_argument("--adapter", help="Adapter directory for the 'lora' variant (e.g. artifacts/qwen2.5-3b/bfcl-adapters)")
    p.add_argument("--prompt", choices=["full", "minimal"], default="full",
                   help="System prompt style: full (types + envelope example) or minimal (names only)")
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
