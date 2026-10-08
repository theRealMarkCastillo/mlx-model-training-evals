"""Copy the latest preset runs into docs/reference-run/ as a small, committable summary.

Full eval_results.json files hold every prompt and output (several MB), so only
the aggregate metrics, a few example failures, and the charts are exported.
Run it after a pipeline run:  uv run python scripts/export_reference.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import failure_examples  # noqa: E402

DEFAULT_PRESET_ROOT = ROOT / "artifacts" / "qwen2.5-3b"
DEFAULT_OUT = ROOT / "docs" / "reference-run"


def latest(preset_root, kind):
    path = Path(preset_root) / f"latest_{kind}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def best_ablation(preset_root):
    """The ablation run with the most points (ties: newest).

    `latest_ablation.json` follows the newest run, and a per-seed sweep is run one seed at a
    time (so an interrupted seed is cheap to retry) — which would leave the committed
    `ablation.png` showing a single point and hide the iteration sweep. Scanning for the
    richest sweep keeps the curated artifact meaningful; seed results live in
    `seed_sweep.json` regardless.
    """
    candidates = []
    for report in Path(preset_root).glob("runs/ablation-*/ablation.json"):
        data = json.loads(report.read_text())
        candidates.append((len(data.get("rows", [])), report.stat().st_mtime, report, data))
    if not candidates:
        return None, None
    _, _, report, data = max(candidates, key=lambda item: (item[0], item[1]))
    manifest = json.loads((report.parent / "manifest.json").read_text())
    return {"run_dir": str(report.parent), **manifest}, data


def strip(metrics):
    return {k: v for k, v in metrics.items() if k != "sample_results"}


def build_summary(preset_root=DEFAULT_PRESET_ROOT, out=DEFAULT_OUT):
    """Export the newest run of each kind and return the summary that was written."""
    preset_root, out = Path(preset_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    training = latest(preset_root, "training")
    if training:
        run = Path(training["run_dir"])
        history = json.loads((run / "training_history.json").read_text())
        summary["training"] = {k: history[k] for k in (
            "model", "training_time_seconds", "peak_memory_mb", "examples_seen", "train_records",
            "parameters", "loss_summary")}
        summary["training"]["config"] = {k: training["config"][k] for k in (
            "lora_parameters", "learning_rate", "batch_size", "iters", "num_layers", "seed")}
        summary["training"]["history"] = history["history"]
        shutil.copy(run / "loss_curve.png", out / "loss_curve.png")
    evaluation = latest(preset_root, "evaluation")
    if evaluation:
        run = Path(evaluation["run_dir"])
        report = json.loads((run / "eval_results.json").read_text())
        summary["evaluation"] = {
            "variants": report["variants"],
            "constrained": report.get("constrained", False),
            "temperature": report.get("temperature", 0.0),
            "paired": report.get("paired"),
            "paired_schema": report.get("paired_schema"),
            "datasets": {name: {v: strip(m) for v, m in by.items()} for name, by in report["datasets"].items()},
            "example_failures": {name: {v: failure_examples(m["sample_results"], 3) for v, m in by.items()}
                                 for name, by in report["datasets"].items()},
            "prompt_tokens": {v: report["datasets"]["test"][v]["sample_results"][0]["prompt_tokens"]
                              for v in report["variants"]},
        }
        for png in run.glob("*.png"):
            shutil.copy(png, out / png.name)
    benchmark = latest(preset_root, "benchmark")
    if benchmark:
        report = json.loads((Path(benchmark["run_dir"]) / "benchmark_results.json").read_text())
        summary["benchmark"] = {k: {m: v for m, v in s.items() if m != "runs"}
                                for k, s in report.items() if k.endswith("_stats")}
    ablation, ablation_data = best_ablation(preset_root)
    if ablation:
        run = Path(ablation["run_dir"])
        summary["ablation"] = ablation_data
        shutil.copy(run / "ablation.png", out / "ablation.png")
    forgetting = latest(preset_root, "forgetting")
    if forgetting:
        run = Path(forgetting["run_dir"])
        result = json.loads((run / "forgetting.json").read_text())
        summary["forgetting"] = {k: result[k] for k in ("n", "base_loss", "lora_loss", "delta", "sign_test", "verdict")}
        shutil.copy(run / "forgetting.png", out / "forgetting.png")
    summary["provenance"] = {kind: (manifest or {}).get("run_id") for kind, manifest in
                             (("training", training), ("evaluation", evaluation), ("benchmark", benchmark),
                              ("ablation", ablation), ("forgetting", forgetting))}
    summary["provenance"]["versions"] = (training or {}).get("versions")
    summary["provenance"]["platform"] = (training or {}).get("platform")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(preset_root=DEFAULT_PRESET_ROOT, out=DEFAULT_OUT):
    summary = build_summary(preset_root, out)
    print(f"Exported to {out} ({', '.join(sorted(summary))})")
    return summary


if __name__ == "__main__":
    main()
