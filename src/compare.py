"""Compare several adapters of one base model side by side.

`eval` compares *variants* (base, few-shot, one adapter). This compares *adapters*: the SFT
adapter against the same adapter after DPO, against an SFT control trained on the same
prompts. Each adapter is scored on the holdout and every challenge set, with:

* exact match and schema validity, with Wilson intervals (the JSON envelope rate is reported
  separately because over-optimised preference tuning can break the format while improving a
  field);
* the targeted field's accuracy (`rollback_deployment.deployment_id`), pulled from the
  per-field breakdown;
* a paired McNemar test of every adapter against the first one listed (the reference), per set;
* the forgetting loss on `data/general.jsonl` relative to the bare base model.
"""

import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx_lm
from rich.console import Console
from rich.table import Table

from src.dataset import DATA_DIR, challenge_files, load_general_samples, load_samples
from src.evaluate import run_deterministic_eval
from src.forgetting import GENERAL_PATH, per_record_losses
from src.metrics import format_rate, paired_comparison, sign_test
from src.models import resolve_model_paths
from src.runs import (
    adapter_identity,
    finish_run,
    latest_path,
    new_run,
    record_failure,
    resolve_adapter_source,
    write_json,
)

console = Console()
TARGET = ("rollback_deployment", "deployment_id")


def parse_named_adapters(specs):
    """['sft=path', 'dpo=path'] -> [('sft', path), ('dpo', path)], order preserved, names unique."""
    named = []
    for spec in specs:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Expected NAME=ADAPTER_DIR, got {spec!r}")
        if name in dict(named) or name == "base":
            raise ValueError(f"Adapter name {name!r} is duplicated or reserved")
        named.append((name, latest_path(path)))
    if len(named) < 2:
        raise ValueError("Give at least two adapters to compare; the first is the reference")
    return named


def target_accuracy(metrics):
    tool, field = TARGET
    per_field = metrics.get("per_field", {}).get(tool, {})
    return per_field.get(field)


def plot_comparison(names, results, output_path):
    sets = list(results)
    fig, ax = plt.subplots(figsize=(1.8 * len(sets) + 3, 4.6), dpi=150)
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        rates = [100 * results[s][name]["exact_match_rate"] for s in sets]
        lows = [100 * (results[s][name]["exact_match_rate"] - results[s][name]["intervals"]["exact_match_rate"][0]) for s in sets]
        highs = [100 * (results[s][name]["intervals"]["exact_match_rate"][1] - results[s][name]["exact_match_rate"]) for s in sets]
        ax.bar([j + i * width for j in range(len(sets))], rates, width, yerr=[lows, highs], capsize=3, label=name)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(sets))], sets, rotation=15)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact match, % (95% interval)")
    ax.set_title("Adapter comparison", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_comparison(adapters, *, preset=None, model_name=None, output_dir=None, max_tokens=150, challenge=True,
                   general_jsonl=None):
    """Evaluate each `(name, adapter_dir)` and print/save the comparison; returns the report."""
    names = [name for name, _ in adapters]
    model_name, _, root = resolve_model_paths(preset, model_name, adapters[0][1], output_dir)
    paths = {"test": DATA_DIR / "test.jsonl"}
    if challenge:
        paths.update(challenge_files(DATA_DIR))
    samples = {name: load_samples(path) for name, path in paths.items()}
    general = load_general_samples(Path(general_jsonl) if general_jsonl else GENERAL_PATH)
    source, identity = resolve_adapter_source(model_name, adapters[0][1])
    for name, path in adapters:
        resolved, _ = resolve_adapter_source(model_name, path)
        if resolved != source:
            raise ValueError(f"Adapter {name!r} was built on a different base snapshot than {names[0]!r}")
    run_dir, manifest = new_run(
        root, "compare", model=model_name, model_source=identity, reference=names[0],
        adapters={name: adapter_identity(path) for name, path in adapters},
        sample_ids={name: [s["id"] for s in group] for name, group in samples.items()},
        generation={"temperature": 0.0, "max_tokens": max_tokens},
    )
    with record_failure(run_dir, manifest):
        results = {s: {} for s in samples}
        forgetting = {}
        for name, path in [("base", None), *adapters]:
            console.print(f"[bold]{name}[/bold]")
            model, tokenizer = mlx_lm.load(source, **({"adapter_path": path} if path else {}))
            try:
                forgetting[name] = per_record_losses(model, tokenizer, general)
                if path is None:
                    continue
                for set_name, group in samples.items():
                    results[set_name][name] = run_deterministic_eval(model, tokenizer, group, max_tokens, desc=f"{name}/{set_name}")
            finally:
                del model, tokenizer
                gc.collect()
                mx.clear_cache()

        reference = names[0]
        paired = {
            set_name: {name: paired_comparison(by_name[name]["sample_results"], by_name[reference]["sample_results"])
                       for name in names[1:]}
            for set_name, by_name in results.items()
        }
        base_losses = forgetting["base"]
        forget = {}
        for name in names:
            deltas = [forgetting[name][s["id"]] - base_losses[s["id"]] for s in general]
            forget[name] = {"mean_loss": sum(forgetting[name].values()) / len(general),
                            "mean_delta_vs_base": sum(deltas) / len(deltas), "sign_test": sign_test(deltas)}
        forget["base"] = {"mean_loss": sum(base_losses.values()) / len(general)}

        report = {"model": model_name, "reference": reference, "run_dir": str(run_dir), "adapters": dict(adapters),
                  "datasets": {s: {n: {k: v for k, v in m.items() if k != "sample_results"} for n, m in by.items()}
                               for s, by in results.items()},
                  "paired_vs_reference": paired, "forgetting": forget,
                  "per_sample": {s: {n: [{k: r[k] for k in ("id", "raw_output", "param_exact", "error_category", "wrong_fields")}
                                         for r in m["sample_results"]] for n, m in by.items()} for s, by in results.items()}}
        plot_comparison(names, results, run_dir / "comparison.png")
        print_comparison(names, results, paired, forget, reference)
        write_json(run_dir / "comparison.json", report)
        manifest["result"] = {"exact_match": {s: {n: m["exact_match_rate"] for n, m in by.items()} for s, by in results.items()}}
        finish_run(root, run_dir, manifest)
    console.print(f"Saved comparison to {run_dir / 'comparison.json'}")
    return report


def print_comparison(names, results, paired, forget, reference):
    table = Table(title=f"Exact match [95% interval]; McNemar p vs {reference}", header_style="bold magenta")
    table.add_column("Set (n)")
    for name in names:
        table.add_column(name, justify="right")
    for set_name, by_name in results.items():
        row = [f"{set_name} ({by_name[reference]['num_samples']})"]
        for name in names:
            m = by_name[name]
            cell = format_rate(m["exact_match_rate"], m["intervals"]["exact_match_rate"])
            if name != reference:
                p = paired[set_name][name]
                cell += f"  +{p['only_a']}/-{p['only_b']} p={p['p_value']:.2g}"
            row.append(cell)
        table.add_row(*row)
    console.print(table)

    table = Table(title="Envelope and targeted field", header_style="bold magenta")
    table.add_column("Set / metric")
    for name in names:
        table.add_column(name, justify="right")
    for set_name, by_name in results.items():
        table.add_row(f"{set_name}: schema-valid", *[f"{100 * by_name[n]['schema_valid_rate']:.1f}%" for n in names])
        accuracy = [target_accuracy(by_name[n]) for n in names]
        if any(a is not None for a in accuracy):
            table.add_row(f"{set_name}: {TARGET[0]}.{TARGET[1]}", *["-" if a is None else f"{100 * a:.1f}%" for a in accuracy])
    console.print(table)

    table = Table(title="Forgetting: mean assistant loss on general requests", header_style="bold magenta")
    for column in ("Model", "Mean loss", "vs base", "Records worse / better", "Sign-test p"):
        table.add_column(column, justify="right" if column != "Model" else "left")
    table.add_row("base", f"{forget['base']['mean_loss']:.4f}", "-", "-", "-")
    for name in names:
        f = forget[name]
        table.add_row(name, f"{f['mean_loss']:.4f}", f"{f['mean_delta_vs_base']:+.4f}",
                      f"{f['sign_test']['increased']} / {f['sign_test']['decreased']}", f"{f['sign_test']['p_value']:.3g}")
    console.print(table)
