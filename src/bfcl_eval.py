"""The BFCL pilot: base model vs grammar-constrained decoding on real function schemas.

The synthetic task's headline comparison — does grammar-constrained decoding recover the
format failures that training fixes? — is re-run here against BFCL v3 `simple`, where the
function schemas are real (nested objects, arrays, floats) and were never seen by the
five-tool pipeline. The same per-record system prompt and `{"tool","parameters"}` envelope
are used, so the numbers stay comparable.

Two prompting-only variants are measured on the committed `data/bfcl_simple.jsonl`:

* `base`    — plain greedy decoding with the per-record system prompt.
* `grammar` — the same, decoded under a per-record `JsonSchemaGrammar` built from that
  record's function schema.

The metric is BFCL-style rather than exact match: a call is right when the function name
matches and every provided argument's value is in the ground-truth acceptable set (and
every required argument is present), because BFCL's answers are sets of alternatives, not
single values.
"""

import json
from collections import Counter
from pathlib import Path

import mlx_lm
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.bfcl import build_minimal_system_prompt, parse_bfcl_call, score_bfcl_call
from src.constrained import constrained_generate, tokenizer_vocab_size
from src.inference import generate_response
from src.metrics import format_rate, paired_comparison, wilson_interval
from src.models import PRESETS
from src.schema_grammar import JsonSchemaGrammar, SchemaGrammarProcessor, bfcl_envelope

console = Console()
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bfcl_simple.jsonl"
IRRELEVANCE_PATH = Path(__file__).resolve().parent.parent / "data" / "bfcl_irrelevance.jsonl"
RATE_FLAGS = (("schema_valid_rate", "is_schema_valid"), ("tool_accuracy", "tool_correct"),
              ("arg_accuracy", "args_correct"))
LABELS = {"base": "Base (zero-shot)", "grammar": "Base + JSON grammar", "lora": "LoRA (BFCL split)"}


def load_bfcl_records(path=DATA_PATH, max_records=None):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if max_records is not None:
        records = records[: int(max_records)]
    return records


def summarize_bfcl(results):
    n = len(results)
    if not n:
        raise ValueError("Cannot summarize zero results")
    summary = {"num_samples": n, "intervals": {}}
    for metric, flag in RATE_FLAGS:
        successes = sum(bool(result[flag]) for result in results)
        summary[metric] = successes / n
        summary["intervals"][metric] = wilson_interval(successes, n)
    summary["error_categories"] = dict(Counter(result["error_category"] for result in results
                                               if result["error_category"]))
    summary["sample_results"] = results
    return summary


def run_bfcl_eval(model_name=None, *, preset="3b", records_path=None, variants=("base", "grammar"),
                  max_tokens=200, max_records=None, output_dir=None, quiet=False, prompt_style="full",
                  adapter_path=None):
    records = load_bfcl_records(records_path or DATA_PATH, max_records)
    preset = preset or "3b"
    model_name = model_name or PRESETS[preset].model
    console.print(f"[bold]BFCL simple: {len(records)} records, model {model_name}, prompt={prompt_style}[/bold]")
    datasets = {"bfcl_simple": {}}
    for variant in variants:
        console.print(f"[bold]Evaluating {LABELS.get(variant, variant)}[/bold]")
        adapter = adapter_path if variant == "lora" else None
        model, tokenizer = mlx_lm.load(model_name, **({"adapter_path": adapter} if adapter else {}))
        vocab_size = getattr(getattr(model, "args", None), "vocab_size", None) or tokenizer_vocab_size(tokenizer)
        results = []
        try:
            for record in tqdm(records, desc=variant, leave=False):
                messages = [dict(message) for message in record["messages"][:-1]]   # system + user
                if prompt_style == "minimal" and messages and messages[0]["role"] == "system":
                    messages[0]["content"] = build_minimal_system_prompt(
                        record["meta"].get("functions") or [record["meta"]["function"]])
                if variant == "grammar":
                    grammar = JsonSchemaGrammar(bfcl_envelope(record["meta"].get("functions") or [record["meta"]["function"]]))
                    processor = SchemaGrammarProcessor(tokenizer, vocab_size, grammar)
                    generated = constrained_generate(model, tokenizer, messages, max_tokens, processor=processor)
                else:
                    generated = generate_response(model, tokenizer, messages, max_tokens)
                parsed = parse_bfcl_call(generated["raw_output"])
                results.append({
                    "id": record["id"], "prompt": record["prompt"], "expected": record["expected"],
                    "meta": record["meta"], "raw_output": generated["raw_output"], "parsed": parsed,
                    **score_bfcl_call(parsed, record["meta"]),
                    "grammar_complete": generated.get("grammar_complete"),
                })
        finally:
            del model
            import mlx.core as mx
            mx.clear_cache()
        datasets["bfcl_simple"][variant] = summarize_bfcl(results)

    report = {"model": model_name, "preset": preset, "records_path": str(records_path or DATA_PATH),
              "variants": list(variants), "max_tokens": max_tokens, "prompt_style": prompt_style,
              "adapter_path": adapter_path, "datasets": datasets}
    if len(variants) >= 2:
        report["paired"] = {flag: paired_comparison(datasets["bfcl_simple"][variants[0]]["sample_results"],
                                                    datasets["bfcl_simple"][variants[1]]["sample_results"], flag)
                            for _, flag in RATE_FLAGS}
    if output_dir:
        suffix = "" if prompt_style == "full" else f"_{prompt_style}"
        path = Path(output_dir) / f"{Path(records_path or DATA_PATH).stem}{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        compact = {key: value for key, value in report.items() if key != "datasets"}
        compact["datasets"] = {
            name: {variant: {key: value for key, value in summary.items() if key != "sample_results"}
                   for variant, summary in by_variant.items()}
            for name, by_variant in report["datasets"].items()
        }
        path.write_text(json.dumps(compact, indent=2) + "\n")
        console.print(f"Saved {path}")
    if not quiet:
        print_bfcl_report(datasets["bfcl_simple"], report.get("paired"))
    return report


def print_bfcl_report(by_variant, paired=None):
    variants = list(by_variant)
    table = Table(title="BFCL v3 simple: base vs grammar-constrained decoding")
    table.add_column("Metric", justify="left")
    for variant in variants:
        table.add_column(LABELS.get(variant, variant), justify="right")
    for metric, _ in RATE_FLAGS:
        table.add_row(metric, *[format_rate(by_variant[v][metric], by_variant[v]["intervals"][metric])
                                for v in variants])
    for variant, summary in by_variant.items():
        table.add_row(f"{variant} error categories", str(summary["error_categories"]), "")
    console.print(table)
    if paired:
        console.print("McNemar paired test (base vs grammar):")
        for metric, flag in RATE_FLAGS:
            if flag in paired:
                console.print(f"  {metric}: p={paired[flag]['p_value']:.3f}")


def run_bfcl_irrelevance(model_name=None, *, preset="3b", records_path=None, max_tokens=200,
                         max_records=None, output_dir=None, quiet=False):
    """Measure the hallucination rate on the irrelevance category (the model must not call).

    The correct behaviour is to refuse (`no_action` in this repo's envelope) rather than call
    the provided — but irrelevant — function. The grammar variant is given a two-function
    envelope (`no_action` + the record's function) so it *can* express a refusal; the metric
    is the fraction of records where the model nevertheless calls the irrelevant function.
    """
    from src.bfcl import NO_ACTION_SCHEMA
    records = load_bfcl_records(records_path or IRRELEVANCE_PATH, max_records)
    preset = preset or "3b"
    model_name = model_name or PRESETS[preset].model
    console.print(f"[bold]BFCL irrelevance: {len(records)} records, model {model_name}[/bold]")
    model, tokenizer = mlx_lm.load(model_name)
    vocab_size = getattr(getattr(model, "args", None), "vocab_size", None) or tokenizer_vocab_size(tokenizer)
    datasets = {"bfcl_irrelevance": {}}
    try:
        for variant in ("base", "grammar"):
            console.print(f"[bold]Evaluating {LABELS.get(variant, variant)}[/bold]")
            results = []
            for record in tqdm(records, desc=variant, leave=False):
                if variant == "grammar":
                    functions = [{"name": "no_action", "parameters": NO_ACTION_SCHEMA},
                                 {"name": record["meta"]["function"]["name"],
                                  "parameters": record["meta"]["function"].get("parameters") or {}}]
                    grammar = JsonSchemaGrammar(bfcl_envelope(functions))
                    processor = SchemaGrammarProcessor(tokenizer, vocab_size, grammar)
                    generated = constrained_generate(model, tokenizer, record["messages"], max_tokens,
                                                     processor=processor)
                else:
                    generated = generate_response(model, tokenizer, record["messages"], max_tokens)
                parsed = parse_bfcl_call(generated["raw_output"])
                hallucinated = bool(parsed and parsed.get("tool") == record["meta"]["function"]["name"])
                results.append({"id": record["id"], "prompt": record["prompt"], "meta": record["meta"],
                                "raw_output": generated["raw_output"], "hallucinated": hallucinated})
            datasets["bfcl_irrelevance"][variant] = summarize_hallucination(results)
    finally:
        del model
        import mlx.core as mx
        mx.clear_cache()

    report = {"model": model_name, "preset": preset, "records_path": str(records_path or IRRELEVANCE_PATH),
              "variants": ["base", "grammar"], "max_tokens": max_tokens, "datasets": datasets}
    if output_dir:
        path = Path(output_dir) / f"{Path(records_path or IRRELEVANCE_PATH).stem}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        compact = {key: value for key, value in report.items() if key != "datasets"}
        compact["datasets"] = {name: {v: {k: x for k, x in s.items() if k != "sample_results"}
                                      for v, s in by_variant.items()}
                               for name, by_variant in report["datasets"].items()}
        path.write_text(json.dumps(compact, indent=2) + "\n")
        console.print(f"Saved {path}")
    if not quiet:
        table = Table(title="BFCL irrelevance: hallucinated-call rate (lower is better)")
        table.add_column("Metric", justify="left")
        for variant in ("base", "grammar"):
            table.add_column(LABELS.get(variant, variant), justify="right")
        table.add_row("hallucination_rate", *[
            format_rate(datasets["bfcl_irrelevance"][variant]["hallucination_rate"],
                        datasets["bfcl_irrelevance"][variant]["intervals"]["hallucination_rate"])
            for variant in ("base", "grammar")])
        console.print(table)
    return report


def summarize_hallucination(results):
    n = len(results)
    if not n:
        raise ValueError("Cannot summarize zero results")
    hallucinated = sum(bool(result["hallucinated"]) for result in results)
    return {"num_samples": n, "hallucination_rate": hallucinated / n,
            "intervals": {"hallucination_rate": wilson_interval(hallucinated, n)},
            "abstained": n - hallucinated, "sample_results": results}
