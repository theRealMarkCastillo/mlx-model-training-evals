"""Offline summary of every dataset split: sizes, balance, omitted defaults, token lengths, examples.

The data is the part of an experiment that is easiest to take on faith, so this
command makes it inspectable without training anything. Offline by default;
`--tokens` downloads the preset's tokenizer and reports the numbers training
actually sees (prompt tokens vs loss-scored answer tokens, per split).
"""

import json
from collections import Counter

from rich.console import Console
from rich.table import Table

from src.dataset import DATA_DIR, challenge_files, load_samples, validate_splits
from src.models import DEFAULT_PRESET, PRESETS

console = Console()

SPLIT_NOTES = {
    "train": "training (wording families 0–1, seen entities)",
    "valid": "validation loss during training (family 2)",
    "test": "holdout: new wording, seen entities (family 3)",
    "challenge_entities": "unseen services/pods/regions/clusters",
    "challenge_defaults": "every optional parameter omitted (defaults must be written)",
    "challenge_abstain": "unsupported request kinds never seen in training",
}


def split_order(data_dir=DATA_DIR):
    """Standard splits first, then challenge sets, in a stable order."""
    names = ["train", "valid", "test"]
    names += [path.stem for path in sorted(challenge_files(data_dir).values())]
    return [name for name in names if (data_dir / f"{name}.jsonl").is_file()]


def split_summary(path):
    """Counts and shape of one split, from the file alone (no model, no tokenizer)."""
    records = load_samples(path)
    tools = Counter(record["expected"]["tool"] for record in records)
    metals = [record.get("meta") or {} for record in records]
    omitted_fields = Counter(field for meta in metals for field in meta.get("omitted", []))
    prompt_chars = [len(record["prompt"]) for record in records]
    return {
        "name": path.stem,
        "n": len(records),
        "tools": dict(sorted(tools.items())),
        "balanced": len(set(tools.values())) == 1,
        "families": sorted({meta.get("family") for meta in metals if meta.get("family") is not None}),
        "with_omitted": sum(bool(meta.get("omitted")) for meta in metals),
        "omitted_fields": dict(sorted(omitted_fields.items())),
        "mean_prompt_chars": sum(prompt_chars) / len(prompt_chars),
    }


def token_stats(path, tokenizer):
    """Prompt vs answer tokens, tokenized exactly as training does (ChatDataset + mask_prompt)."""
    from mlx_lm.tuner.datasets import ChatDataset

    records = load_samples(path)
    dataset = ChatDataset(records, tokenizer, mask_prompt=True)
    prompts, answers = [], []
    for record in records:
        tokens, offset = dataset.process(record)
        prompts.append(offset)
        answers.append(len(tokens) - offset)
    return {
        "name": path.stem,
        "prompt_mean": sum(prompts) / len(prompts), "prompt_min": min(prompts), "prompt_max": max(prompts),
        "answer_mean": sum(answers) / len(answers), "answer_min": min(answers), "answer_max": max(answers),
    }


def _summary_table(summaries):
    table = Table(title="Dataset splits (offline)")
    for column, justify in (("Split", "left"), ("n", "right"), ("Tools", "left"), ("Wording families", "left"),
                            ("Omitted defaults", "right"), ("Mean prompt chars", "right")):
        table.add_column(column, justify=justify)
    for summary in summaries:
        counts = list(summary["tools"].values())
        tools = (f"{len(summary['tools'])} × {counts[0]}" if summary["balanced"]
                 else ", ".join(f"{t}:{c}" for t, c in summary["tools"].items()))
        omitted = (f"{summary['with_omitted']} ({100 * summary['with_omitted'] / summary['n']:.0f}%)"
                   if summary["with_omitted"] else "0")
        table.add_row(
            summary["name"], str(summary["n"]), tools,
            ",".join(str(f) for f in summary["families"]) or "-", omitted,
            f"{summary['mean_prompt_chars']:.0f}",
        )
    console.print(table)
    console.print("[bold]What each split tests[/bold]")
    for summary in summaries:
        console.print(f"  [cyan]{summary['name']:20}[/cyan] {SPLIT_NOTES.get(summary['name'], '')}")


def _token_table(stats):
    table = Table(title="Tokens per record (training tokenization, mask_prompt=True)")
    for column, justify in (("Split", "left"), ("Prompt mean", "right"), ("Prompt min–max", "right"),
                            ("Answer mean", "right"), ("Answer min–max", "right"), ("Scored share", "right")):
        table.add_column(column, justify=justify)
    for row in stats:
        total = row["prompt_mean"] + row["answer_mean"]
        table.add_row(row["name"], f"{row['prompt_mean']:.0f}", f"{row['prompt_min']}–{row['prompt_max']}",
                      f"{row['answer_mean']:.0f}", f"{row['answer_min']}–{row['answer_max']}",
                      f"{100 * row['answer_mean'] / total:.1f}%")
    console.print(table)
    console.print("[dim]Only the answer tokens are scored by the loss; the prompt is context "
                  "(see `main.py show-mask`).[/dim]")


def print_examples(summaries, examples=1, data_dir=DATA_DIR):
    """One worked record per tool from the first split, so the task is visible without JSONL files."""
    if examples <= 0:
        return
    name = next((s["name"] for s in summaries if s["name"] == "test"), summaries[0]["name"])
    records = load_samples(data_dir / f"{name}.jsonl")
    by_tool = {}
    for record in records:
        by_tool.setdefault(record["expected"]["tool"], []).append(record)
    console.print(f"[bold]Examples from {name}.jsonl[/bold]")
    for tool, group in sorted(by_tool.items()):
        for record in group[:examples]:
            console.print(f"  [cyan]{record['prompt']}[/cyan]")
            console.print(f"    → [green]{json.dumps(record['expected'], separators=(',', ':'))}[/green]"
                          f"  [dim]{tool}[/dim]")


def print_dataset_summary(with_token_stats=False, examples=1, preset=None, data_dir=DATA_DIR):
    names = split_order(data_dir)
    summaries = [split_summary(data_dir / f"{name}.jsonl") for name in names]
    _summary_table(summaries)
    general = data_dir / "general.jsonl"
    if general.is_file():
        from src.dataset import load_general_samples

        console.print(f"[dim]{general.name} ({len(load_general_samples(general))} records) is deliberately separate: "
                      "ordinary requests with a plain assistant prompt, scored by loss in `main.py forgetting`.[/dim]")
    print_examples(summaries, examples, data_dir)
    if with_token_stats:
        from transformers import AutoTokenizer

        profile = PRESETS[preset or DEFAULT_PRESET]
        tokenizer = AutoTokenizer.from_pretrained(profile.model)
        _token_table([token_stats(data_dir / f"{name}.jsonl", tokenizer) for name in names])
    validate_splits(data_dir)
    console.print("Every split loads, and no prompt appears twice across splits. "
                  "Regenerate with `main.py prepare` (seeded, deterministic).")
    return summaries
