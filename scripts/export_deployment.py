"""Export a deployment trade-off table: the trained 3B adapter against prompted 14B and 32B models.

`export_capacity.py` answers "which variant is most accurate"; this answers "what does each
cost to serve". Every candidate is scored on the same holdout and challenge sets, and each
request's latency is the measured one from that run (single stream, greedy, one machine):

    uv run python scripts/export_deployment.py

Candidates (the newest evaluation run per size that contains the variant):

* 3b-lora, 3b-lora-fused                 the first SFT adapter, loaded dynamically and merged into the weights
* 3b-continued-lora, 3b-continued-fused  the same adapter after continuing SFT on the mined pairs
* 14b-fewshot, 32b-fewshot   the untrained base model with five in-context demonstrations

It also reads the newest `benchmark` run for the 3B model's controlled speed and peak memory,
and sizes each model by the bytes of its weight files in the Hugging Face cache.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import wilson_interval  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "docs" / "reference-run" / "deployment.json"
SETS = ("test", "challenge_abstain", "challenge_defaults", "challenge_entities")
CACHE = Path.home() / ".cache" / "huggingface" / "hub"
MODELS = {"3b": "Qwen2.5-3B-Instruct-4bit", "14b": "Qwen2.5-14B-Instruct-4bit", "32b": "Qwen2.5-32B-Instruct-4bit"}


def evaluation_runs(size):
    """Current-format evaluation reports for one size, newest first."""
    reports = []
    for path in sorted((ARTIFACTS / f"qwen2.5-{size}" / "runs").glob("evaluation-*/eval_results.json"), reverse=True):
        data = json.loads(path.read_text())
        if isinstance(data.get("variants"), list) and "datasets" in data:
            reports.append(data)
    return reports


def newest_with(size, variant, adapter=None):
    """Newest report containing `variant`; `adapter` (a run-id fragment) pins which trained adapter it used."""
    for report in evaluation_runs(size):
        if adapter and adapter not in str(report.get("adapter")):
            continue
        if variant in report["variants"] and "test" in report["datasets"] and variant in report["datasets"]["test"]:
            return report
    return None


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def candidate(report, variant):
    sets, results = {}, []
    for name in SETS:
        metrics = report["datasets"].get(name, {}).get(variant)
        if not metrics:
            continue
        low, high = wilson_interval(round(metrics["exact_match_rate"] * metrics["num_samples"]), metrics["num_samples"])
        sets[name] = {"n": metrics["num_samples"], "exact_match": metrics["exact_match_rate"], "ci95": [low, high]}
        results += metrics["sample_results"]
    latency = [r["latency_seconds"] for r in results]
    return {
        "run_id": report["run_id"], "requests": len(results), "exact_match": sets,
        "latency_seconds": {"median": statistics.median(latency), "p95": percentile(latency, 0.95)},
        "ttft_seconds_median": statistics.median(r["ttft_seconds"] for r in results),
        "decode_tokens_per_sec_median": statistics.median(r["decode_tokens_per_sec"] for r in results),
        "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in results),
        "output_tokens_mean": statistics.mean(r["output_tokens"] for r in results),
    }


def weights_gb(size):
    folder = CACHE / f"models--mlx-community--{MODELS[size]}"
    files = [p for p in folder.glob("snapshots/*/*.safetensors")]
    return round(sum(p.stat().st_size for p in files) / 1e9, 2) if files else None


def newest_benchmark(size):
    paths = sorted((ARTIFACTS / f"qwen2.5-{size}" / "runs").glob("benchmark-*/benchmark_results.json"), reverse=True)
    for path in paths:
        data = json.loads(path.read_text())
        if "fused_stats" in data:
            return {k: data[k] for k in ("base_stats", "lora_stats", "fused_stats")} | {"run": path.parent.name}
    return None


def main():
    rows = {}
    # (label, size, variant, adapter run-id fragment): the first SFT adapter, and its continuation on the mined pairs.
    for label, size, variant, adapter in (
            ("3b-lora", "3b", "lora", "7ad75c0f"), ("3b-lora-fused", "3b", "fused", "7ad75c0f"),
            ("3b-continued-lora", "3b", "lora", "6dd5213a"), ("3b-continued-fused", "3b", "fused", "6dd5213a"),
            ("14b-fewshot", "14b", "fewshot", None), ("32b-fewshot", "32b", "fewshot", None)):
        report = newest_with(size, variant, adapter)
        if report is None:
            print(f"skipping {label}: no evaluation run contains `{variant}` on the holdout")
            continue
        rows[label] = candidate(report, variant) | {"weights_gb": weights_gb(size)}
    summary = {"hardware_note": "single stream, greedy, one Apple-silicon Mac; latencies are per request including prefill",
               "candidates": rows, "benchmark_3b": newest_benchmark("3b")}
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{'candidate':20}{'test':>8}{'abstain':>9}{'defaults':>10}{'entities':>10}{'lat med/p95 s':>16}{'prompt tok':>11}{'GB':>7}")
    for label, row in rows.items():
        em = row["exact_match"]
        cells = "".join(f"{100 * em[s]['exact_match']:>{w}.1f}" if s in em else f"{'-':>{w}}"
                        for s, w in zip(SETS, (8, 9, 10, 10), strict=True))
        lat = row["latency_seconds"]
        print(f"{label:20}{cells}{lat['median']:>9.2f}/{lat['p95']:<6.2f}{row['prompt_tokens_mean']:>11.0f}{row['weights_gb'] or 0:>7.1f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
