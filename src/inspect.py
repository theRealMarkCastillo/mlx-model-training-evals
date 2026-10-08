"""Trace one greedy decode token by token: what the model chose, how close the runner-up was, and where it left the reference answer.

Ordinary evaluation tells you *what* failed. This tells you *where the decision
happened*: every generated token comes with its probability and the alternatives
the model weighed, computed from MLX-LM's own log-probabilities (the same
distribution `main.py eval` samples from at temperature 0).

The money shot is the **divergence point**: the first token where the greedy
output stops matching the reference answer, with the reference token's rank and
probability at that exact step. That is how "it wrote dep-9821 instead of the
identifier in the request" becomes "at step 14 the model put 41% on 'dep' and 2%
on the token that names the service".
"""

from dataclasses import dataclass, field

import mlx.core as mx
import mlx_lm
from mlx_lm.sample_utils import make_sampler
from rich.console import Console
from rich.table import Table

from src.dataset import DATA_DIR, fewshot_messages, load_samples, positive_int, select_shots
from src.models import resolve_model_paths
from src.runs import finish_run, new_run, record_failure, resolve_adapter_source, resolve_source, write_json

console = Console()
VARIANTS = ("base", "fewshot", "lora")


@dataclass
class Step:
    """One generated token and the distribution it was drawn from."""

    index: int
    token_id: int
    text: str
    probability: float
    alternatives: list = field(default_factory=list)   # [(token_id, text, probability)], best first, chosen excluded
    finish_reason: str | None = None


def _top_k(logprobs, k, chosen_id):
    """Top-k tokens by probability from a full-vocabulary log-probability vector.

    Only the head of the sorted order is materialized as Python ints: sorting a
    150k-token vocabulary takes long enough to notice at every generated token.
    """
    probs = mx.exp(logprobs)
    order = mx.argsort(-probs)[: k + 2].tolist()
    if chosen_id not in order:  # non-greedy sampler: fall back to the full order
        order = mx.argsort(-probs).tolist()
    picked = []
    for token_id in order:
        if token_id == chosen_id:
            continue
        picked.append((token_id, float(probs[token_id].item())))
        if len(picked) == k:
            break
    return picked


def greedy_trace(model, tokenizer, messages, max_tokens=150, top_k=5):
    """Greedy-decode `messages`, recording per-token probabilities and alternatives.

    Uses the same sampler as evaluation (temperature 0), so the trace is the
    exact sequence `main.py eval` would score.
    """
    positive_int(max_tokens)
    positive_int(top_k)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    steps = []
    chunks = []
    meta = {}
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0),
    ):
        logprobs = response.logprobs
        probability = float(mx.exp(logprobs[response.token]).item())
        alternatives = []
        for token_id, alt_probability in _top_k(logprobs, top_k, response.token):
            alternatives.append({
                "token_id": token_id, "text": tokenizer.decode([token_id]), "probability": alt_probability,
            })
        steps.append(Step(
            index=len(steps), token_id=response.token, text=response.text, probability=probability,
            alternatives=alternatives, finish_reason=response.finish_reason,
        ))
        chunks.append(response.text)
        meta = {"prompt_tokens": response.prompt_tokens, "finish_reason": response.finish_reason}
    return {"steps": steps, "raw_output": "".join(chunks), **meta}


def reference_tokens(tokenizer, record):
    """The assistant segment of the training sequence, exactly as the trainer tokenizes it."""
    from mlx_lm.tuner.datasets import ChatDataset

    tokens, offset = ChatDataset([record], tokenizer, mask_prompt=True).process(record)
    return tokens[offset:]


def align_to_reference(steps, expected_ids, tokenizer):
    """First index where the greedy output leaves the reference token sequence.

    Comparison is by token id and only meaningful up to that first mismatch:
    after it, the model is predicting from a different context than the
    reference would have produced.
    """
    for position, step in enumerate(steps):
        if position >= len(expected_ids) or step.token_id != expected_ids[position]:
            reference_id = expected_ids[position] if position < len(expected_ids) else None
            ranked = sorted(
                [{"token_id": step.token_id, "text": step.text, "probability": step.probability}, *step.alternatives],
                key=lambda item: -item["probability"],
            )
            rank = None
            reference_probability = None
            if reference_id is not None:
                reference_probability = next(
                    (c["probability"] for c in ranked if c["token_id"] == reference_id), 0.0,
                )
                rank = next((i + 1 for i, c in enumerate(ranked) if c["token_id"] == reference_id), None)
            return {
                "step": position,
                "chose": {"token_id": step.token_id, "text": step.text, "probability": step.probability},
                "reference": {
                    "token_id": reference_id,
                    "text": tokenizer.decode([reference_id]) if reference_id is not None else None,
                    "probability": reference_probability,
                    "rank": rank,
                },
                "alternatives": step.alternatives,
            }
    return None


def explain_trace(trace, alignment):
    """One-sentence, human-readable summary of where and how the output diverged."""
    if alignment is None:
        return "Greedy output matches the reference answer token for token."
    reference = alignment["reference"]
    if reference["token_id"] is None:
        return (f"Output ran past the end of the reference answer after step {alignment['step']} "
                f"(reference had {alignment['step']} tokens).")
    rank = reference["rank"]
    rank_text = "outside the top-k" if rank is None else f"rank {rank}"
    return (f"Diverges at step {alignment['step']}: model chose {alignment['chose']['text']!r} "
            f"(p={alignment['chose']['probability']:.3f}) while the reference token "
            f"{reference['text']!r} had p={reference['probability']:.3f} ({rank_text}).")


def print_trace(trace, alignment):
    steps = trace["steps"]
    table = Table(title=f"Greedy decode trace ({len(steps)} tokens)")
    for column, justify in (("#", "right"), ("token", "left"), ("p", "right"), ("Δ next", "right"), ("note", "left")):
        table.add_column(column, justify=justify)
    divergence_step = alignment["step"] if alignment else None
    for step in steps:
        runner_up = step.alternatives[0]["probability"] if step.alternatives else 0.0
        note = ""
        if divergence_step is not None and step.index == divergence_step:
            note = "[red]leaves reference here[/red]"
        elif step.finish_reason == "stop":
            note = "[dim]end of turn[/dim]"
        table.add_row(str(step.index), repr(step.text), f"{step.probability:.3f}",
                      f"{step.probability - runner_up:.3f}", note)
    console.print(table)

    if alignment and alignment["reference"]["token_id"] is not None:
        alternatives = Table(title=f"Model's distribution at step {alignment['step']}")
        for column, justify in (("rank", "right"), ("token", "left"), ("p", "right"), ("", "left")):
            alternatives.add_column(column, justify=justify)
        candidates = sorted(
            [{"token_id": alignment["chose"]["token_id"], "text": alignment["chose"]["text"],
              "probability": alignment["chose"]["probability"]},
             *alignment["alternatives"]],
            key=lambda item: -item["probability"],
        )
        for index, candidate in enumerate(candidates, start=1):
            is_chosen = candidate["token_id"] == alignment["chose"]["token_id"]
            is_reference = candidate["token_id"] == alignment["reference"]["token_id"]
            marker = " ".join(filter(None, ["[bold]chosen[/bold]" if is_chosen else "",
                                            "[green]← reference[/green]" if is_reference else ""]))
            alternatives.add_row(str(index), repr(candidate["text"]), f"{candidate['probability']:.3f}", marker)
        console.print(alternatives)
    console.print(explain_trace(trace, alignment))


def build_messages(record, variant, shots=5, data_dir=DATA_DIR):
    """Chat turns the model sees for this variant, assistant turn stripped for generation."""
    if variant == "fewshot":
        demos = select_shots(data_dir / "train.jsonl", shots)
        messages = fewshot_messages(record["messages"], demos)
    else:
        messages = record["messages"]
    return messages[:-1]


def run_inspection(split="test", index=0, *, preset=None, model_name=None, adapter_path=None,
                   output_dir=None, variant="lora", max_tokens=150, top_k=5, shots=5, quiet=False):
    """Trace one greedy decode and record it as an inspection run."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    records = load_samples(DATA_DIR / f"{split}.jsonl")
    if not 0 <= index < len(records):
        raise ValueError(f"--index {index} is out of range for '{split}' ({len(records)} records)")
    record = records[index]
    needs_adapter = variant == "lora"
    model_name, adapter_path, root = resolve_model_paths(
        preset, model_name, adapter_path, output_dir, need_adapter=needs_adapter,
    )
    if needs_adapter:
        source, identity = resolve_adapter_source(model_name, adapter_path)
    else:
        source, identity = resolve_source(model_name)

    directory, manifest = new_run(
        root, "inspection", model=model_name, model_source=identity,
        dataset=f"{split}.jsonl", sample_id=record["id"], variant=variant,
        generation={"temperature": 0.0, "max_tokens": max_tokens, "top_k": top_k},
    )
    with record_failure(directory, manifest):
        model, tokenizer = mlx_lm.load(source, **({"adapter_path": adapter_path} if needs_adapter else {}))
        try:
            messages = build_messages(record, variant, shots)
            mx.reset_peak_memory()
            trace = greedy_trace(model, tokenizer, messages, max_tokens, top_k)
            expected_ids = reference_tokens(tokenizer, record)
            alignment = align_to_reference(trace["steps"], expected_ids, tokenizer)
            result = {
                "id": record["id"], "variant": variant, "prompt": record["prompt"],
                "expected": record["expected"], "raw_output": trace["raw_output"],
                "prompt_tokens": trace.get("prompt_tokens"), "finish_reason": trace.get("finish_reason"),
                "peak_memory_mb": round(mx.get_peak_memory() / (1024 ** 2), 2),
                "run_dir": str(directory), "run_id": manifest["run_id"],
                "steps": [vars(step) for step in trace["steps"]],
                "expected_token_ids": expected_ids,
                "alignment": alignment, "explanation": explain_trace(trace, alignment),
            }
            if not quiet:
                console.print(f"[bold]{record['id']}[/bold]  variant={variant}")
                console.print(f"request: [cyan]{record['prompt']}[/cyan]")
                console.print(f"reference: [green]{record['messages'][-1]['content']}[/green]")
                console.print(f"model:     [yellow]{trace['raw_output']!r}[/yellow]")
                print_trace(trace, alignment)
            manifest["alignment"] = alignment
            write_json(directory / "inspect_trace.json", result)
            finish_run(root, directory, manifest)
        finally:
            del model
            mx.clear_cache()
    if quiet:
        console.print(f"Trace saved to {directory / 'inspect_trace.json'}")
    return result
