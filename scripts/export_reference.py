"""Copy the latest 3B runs into docs/reference-run/ as a small, committable summary.

Full eval_results.json files hold every prompt and output (several MB), so only
the aggregate metrics, a few example failures, and the charts are exported.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import failure_examples  # noqa: E402

PRESET_ROOT = ROOT / "artifacts" / "qwen2.5-3b"
OUT = ROOT / "docs" / "reference-run"


def latest(kind):
    path = PRESET_ROOT / f"latest_{kind}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def strip(metrics):
    return {k: v for k, v in metrics.items() if k != "sample_results"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    training = latest("training")
    if training:
        run = Path(training["run_dir"])
        history = json.loads((run / "training_history.json").read_text())
        summary["training"] = {k: history[k] for k in (
            "model", "training_time_seconds", "peak_memory_mb", "examples_seen", "train_records",
            "parameters", "loss_summary")}
        summary["training"]["config"] = {k: training["config"][k] for k in (
            "lora_parameters", "learning_rate", "batch_size", "iters", "num_layers", "seed")}
        summary["training"]["history"] = history["history"]
        shutil.copy(run / "loss_curve.png", OUT / "loss_curve.png")
    evaluation = latest("evaluation")
    if evaluation:
        run = Path(evaluation["run_dir"])
        report = json.loads((run / "eval_results.json").read_text())
        summary["evaluation"] = {
            "variants": report["variants"], "paired": report["paired"],
            "datasets": {name: {v: strip(m) for v, m in by.items()} for name, by in report["datasets"].items()},
            "example_failures": {name: {v: failure_examples(m["sample_results"], 3) for v, m in by.items()}
                                 for name, by in report["datasets"].items()},
            "prompt_tokens": {v: report["datasets"]["test"][v]["sample_results"][0]["prompt_tokens"] for v in report["variants"]},
        }
        for png in run.glob("*.png"):
            shutil.copy(png, OUT / png.name)
    benchmark = latest("benchmark")
    if benchmark:
        report = json.loads((Path(benchmark["run_dir"]) / "benchmark_results.json").read_text())
        summary["benchmark"] = {k: {m: v for m, v in s.items() if m != "runs"} for k, s in report.items() if k.endswith("_stats")}
    ablation = latest("ablation")
    if ablation:
        run = Path(ablation["run_dir"])
        summary["ablation"] = json.loads((run / "ablation.json").read_text())
        shutil.copy(run / "ablation.png", OUT / "ablation.png")
    summary["provenance"] = {kind: (m or {}).get("run_id") for kind, m in
                             (("training", training), ("evaluation", evaluation), ("benchmark", benchmark), ("ablation", ablation))}
    summary["provenance"]["versions"] = (training or {}).get("versions")
    summary["provenance"]["platform"] = (training or {}).get("platform")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Exported to {OUT}")


if __name__ == "__main__":
    main()
