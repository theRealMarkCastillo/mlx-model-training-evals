"""Merge per-seed ablation runs into one mean ± spread table.

`main.py ablate seed 42 43 44 --iters 200` is the clean way to run a seed sweep, but a
sweep of full training runs is exactly the kind of long GPU job that macOS can abort
(`[METAL] Command buffer execution failed: Impacting Interactivity`, see the README's
troubleshooting). Running one seed per invocation keeps each seed's work independent:
an interrupted seed is retried alone, and finished seeds are never recomputed.

    uv run python main.py ablate seed 42 --iters 200
    uv run python main.py ablate seed 43 --iters 200
    uv run python scripts/merge_seed_sweep.py      # writes docs/reference-run/seed_sweep.json

Rows are deduplicated by seed (the newest run wins), so retrying a seed does not double
count it.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import sweep_summary  # noqa: E402

DEFAULT_PRESET_ROOT = ROOT / "artifacts" / "qwen2.5-3b"
DEFAULT_OUT = ROOT / "docs" / "reference-run"
METRIC_KEYS = ("exact_match_rate", "schema_valid_rate", "test_loss", "best_val_loss", "adapter_parameters")


def collect_seed_rows(preset_root, param="seed"):
    """Every completed `ablate <param>` row across the preset's ablation runs, newest run last."""
    rows = []
    for run in sorted(Path(preset_root).glob("runs/ablation-*"), key=lambda path: path.stat().st_mtime):
        report = run / "ablation.json"
        if not report.is_file():
            continue  # a failed sweep writes no ablation.json
        data = json.loads(report.read_text())
        if data.get("param") != param:
            continue
        for row in data.get("rows", []):
            rows.append({**{k: row[k] for k in METRIC_KEYS if k in row},
                         "value": row["value"], "ci95": row.get("ci95"),
                         "training_run": row.get("training_run"), "evaluation_run": row.get("evaluation_run"),
                         "sweep_run": run.name})
    return rows


def merge_seed_rows(rows):
    """Deduplicate by seed (newest run wins) and add the mean ± spread summary."""
    by_seed = {}
    for row in rows:
        by_seed[row["value"]] = row   # rows arrive oldest-run-first, so later runs overwrite
    merged = [by_seed[value] for value in sorted(by_seed)]
    if not merged:
        raise ValueError("No completed seed runs found: run `main.py ablate seed <N> --iters 200` first")
    return {"param": "seed", "rows": merged,
            "spread": sweep_summary([row["exact_match_rate"] for row in merged])}


def main(preset_root=DEFAULT_PRESET_ROOT, out=DEFAULT_OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    merged = merge_seed_rows(collect_seed_rows(preset_root))
    (out / "seed_sweep.json").write_text(json.dumps(merged, indent=2) + "\n")
    spread = merged["spread"]
    print(f"seeds merged: {[row['value'] for row in merged['rows']]}")
    for row in merged["rows"]:
        low, high = row["ci95"]
        print(f"  seed {row['value']}: exact match {100 * row['exact_match_rate']:.1f}% "
              f"[{100 * low:.0f}-{100 * high:.0f}]  test loss {row['test_loss']:.4f}")
    print(f"across seeds: {100 * spread['mean']:.1f}% ± {100 * spread['stdev']:.1f} "
          f"(n={spread['n']}, min {100 * spread['min']:.0f}%, max {100 * spread['max']:.0f}%)")
    print(f"Wrote {out / 'seed_sweep.json'}")
    return merged


if __name__ == "__main__":
    main()
