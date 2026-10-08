"""Export a cross-model-size comparison: base / few-shot / grammar (and LoRA where it exists).

Answers the question the 3B reference table cannot: does a larger base model, decoded under
the schema grammar, close the gap to a small trained adapter? Reads the newest completed
evaluation run per preset and writes `docs/reference-run/capacity.json`, so the comparison is
regenerable rather than hand-copied.

    uv run python scripts/export_capacity.py

Sizes with no evaluation are skipped, so the file grows as runs land. Training is not
required: `base`, `fewshot` and `grammar` are prompting-only variants.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_OUT = ROOT / "docs" / "reference-run"
PRESETS = ("3b", "7b", "14b", "32b", "72b")
METRICS = ("exact_match_rate", "schema_valid_rate", "tool_accuracy")


def newest_evaluation(preset_root):
    """The most recently written *current-format* evaluation report under one preset.

    Older runs (pre-rewrite, or an evaluation from an earlier pipeline) can sit in the same
    directory with a different schema, so reports are validated by shape and skipped —
    with their paths returned — rather than crashing the export.
    """
    reports = sorted(Path(preset_root).glob("runs/evaluation-*/eval_results.json"),
                     key=lambda path: path.stat().st_mtime)
    skipped = []
    for path in reversed(reports):
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            skipped.append(str(path))
            continue
        if isinstance(report.get("variants"), list) and isinstance(report.get("datasets"), dict):
            return report, path, list(reversed(skipped))
        skipped.append(str(path))
    return None, None, list(reversed(skipped))


def compact_metrics(metrics):
    """Just what a cross-size table needs, with the interval on exact match."""
    compact = {"n": metrics["num_samples"]}
    for key in METRICS:
        compact[key] = metrics[key]
    compact["ci95"] = metrics["intervals"]["exact_match_rate"]
    return compact


def build_capacity(artifacts_root=DEFAULT_ARTIFACTS, presets=PRESETS):
    """{size: {dataset: {variant: metrics}}} plus provenance, for sizes that have a report."""
    capacity = {}
    for size in presets:
        report, path, skipped = newest_evaluation(Path(artifacts_root) / f"qwen2.5-{size}")
        if report is None:
            if skipped:
                print(f"skipped {len(skipped)} report(s) for {size} with an older schema")
            continue
        capacity[size] = {
            "model": report["model"],
            "run_id": report["run_id"],
            "variants": report["variants"],
            "constrained": report.get("constrained", False),
            "datasets": {
                name: {variant: compact_metrics(metrics) for variant, metrics in by_variant.items()}
                for name, by_variant in report["datasets"].items()
            },
        }
    return capacity


def print_table(capacity):
    variants = []
    for entry in capacity.values():
        for variant in entry["variants"]:
            if variant not in variants:
                variants.append(variant)
    print(f"{'size':>5} {'dataset':<20} " + " ".join(f"{v:>18}" for v in variants))
    for size, entry in capacity.items():
        for dataset, by_variant in entry["datasets"].items():
            cells = []
            for variant in variants:
                metrics = by_variant.get(variant)
                if metrics is None:
                    cells.append(f"{'-':>18}")
                else:
                    low, high = metrics["ci95"]
                    cells.append(f"{100 * metrics['exact_match_rate']:5.1f}% [{100 * low:.0f}-{100 * high:.0f}]".rjust(18))
            print(f"{size:>5} {dataset:<20} " + " ".join(cells))


def main(artifacts_root=DEFAULT_ARTIFACTS, out=DEFAULT_OUT):
    capacity = build_capacity(artifacts_root)
    if not capacity:
        raise ValueError(f"No evaluation runs found under {artifacts_root}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"metric": "exact match (95% Wilson interval), greedy decoding",
               "variants": "base, fewshot and grammar are prompting-only; lora needs an adapter",
               "sizes": capacity}
    (out / "capacity.json").write_text(json.dumps(payload, indent=2) + "\n")
    print_table(capacity)
    print(f"Wrote {out / 'capacity.json'} ({', '.join(capacity)})")
    return payload


if __name__ == "__main__":
    main()
