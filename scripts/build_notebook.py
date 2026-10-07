"""Build the tutorial from stable cells using the same modules as the CLI."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.metadata.language_info = {'name': 'python', 'version': '3.12'}
cells = []


def markdown(identifier, text):
    cells.append(nbf.v4.new_markdown_cell(text, id=identifier))


def code(identifier, text):
    cells.append(nbf.v4.new_code_cell(text, id=identifier))


markdown('intro', '''# Local LoRA training and auditable evaluation with Apple MLX

This notebook runs the same workflow as the CLI: strict tool-call schemas, grounded synthetic data, LoRA training, assistant-only loss, exact-match evaluation, and streaming inference measurements.

**Historical result migration:** the original top-level reports and chart in `artifacts/` used the earlier dataset and evaluator. Retrain before making current quality claims. This notebook writes new run directories and preserves those historical files.

Run `uv sync --locked` from the repository root, then open this notebook with `uv run jupyter lab tutorial.ipynb`. Larger presets require more memory; their memory fit has not been established by this tutorial.''')
code('hardware', '''import json
import sys
from pathlib import Path
import mlx.core as mx
from IPython.display import Image, display

print(f"Python: {sys.version.split()[0]}; MLX: {mx.__version__}")
print(f"Metal available: {mx.metal.is_available()}")''')
markdown('schema-notes', '''## 1. Strict tool-call contracts

A discriminated union binds each tool to its own parameters. Schema validation rejects wrong types and unknown fields. Raw parsed output remains unchanged; default-filled normalization is reported separately. Exact match additionally requires pure JSON and the complete reference object.''')
code('schema-example', '''from src.schema import SYSTEM_PROMPT, parse_and_validate

examples = [
    '{"tool":"deploy_service","parameters":{"service":"api","version":"v1","environment":"production","replicas":3,"notify_channels":[]}}',
    '{"tool":"deploy_service","parameters":{"service":"api","version":"v1","environment":"production","replicas":"3","invented":true}}',
    '{"tool":[],"parameters":{}}',
]
for output in examples:
    result = parse_and_validate(output)
    print({key: result[key] for key in ("is_pure_json", "is_schema_valid", "error")})''')
markdown('data-notes', '''## 2. Dataset engineering

The generator uses a fixed seed, balanced tool counts, unique prompts across splits, and separate wording templates for training, validation, and test. Notification targets appear in their requests; restart reasons copy the request phrase exactly. The canonical chat records supply both generation prompts and reference loss targets.

This is an unseen-wording synthetic holdout, not a production reliability benchmark.''')
code('prepare-data', '''from data.prepare_dataset import main as prepare_data
from src.dataset import load_samples, validate_splits

prepare_data()
validate_splits("data")
for split in ("train", "valid", "test"):
    print(split, len(load_samples(f"data/{split}.jsonl")))
print(json.dumps(load_samples("data/test.jsonl", 1)[0], indent=2))''')
markdown('training-notes', '''## 3. LoRA training and provenance

MLX applies the direct multiplier `scale` to its low-rank update: `W_effective = W_base + scale * B A`. It does not divide `scale` by rank.

With `mask_prompt: true`, training scores the assistant response. Each invocation creates new weights, loss history, a plot, and a manifest. The manifest connects model revision, configuration, dataset hashes, dependencies, and adapter identity. The configured adapter directory points to the latest completed run.

Start with a short run to measure memory. The training cell below performs a new run each time it is executed.''')
code('train', '''from src.models import PRESETS
from src.train import run_training

PRESET = "3b"  # 3b, 7b, 14b, 32b, or 72b
ITERS = 10     # Increase after measuring memory and reviewing loss
profile = PRESETS[PRESET]
training = run_training(preset=PRESET, iters_override=ITERS)
print(f"Weights: {training.adapter_path}")
display(Image(filename=str(training.run_dir / "loss_curve.png")))''')
markdown('eval-notes', '''## 4. Quality evaluation

Evaluation uses the same assistant-token loss mask as training. Generation uses each stored record's system and user messages. Exact match does not coerce values, remove extra fields, or insert defaults. Normalized match is an additional, explicitly weaker metric.

All sample outputs, expected answers, parsed objects, verdicts, errors, token counts, and timings are saved, including samples beyond the first ten.''')
code('evaluate', '''from src.evaluate import run_comprehensive_evaluation

report = run_comprehensive_evaluation(preset=PRESET, num_eval_samples=60)
print(f"Report: {report['run_dir']}")
display(Image(filename=str(Path(report["run_dir"]) / "eval_comparison.png")))''')
markdown('benchmark-notes', '''## 5. Performance and fusion

The benchmark formats the same system prompt and chat template, then measures time to first token, prefill throughput, decode throughput, end-to-end throughput, total latency, and peak Metal allocation. It retains every measured run and uses MLX token metadata rather than retokenizing output.

Fusion removes the dynamic adapter branch, but quantized fusion can alter outputs. Speed and quality must be measured. The optional command below fuses, benchmarks all three variants, and runs a linked holdout evaluation; it can take additional time and disk space.''')
code('benchmark', '''from src.benchmark import run_benchmark_suite

benchmark_report = run_benchmark_suite(preset=PRESET)
print(f"Benchmark: {benchmark_report['run_dir']}")
print(f"Fuse, benchmark, and evaluate: uv run python main.py benchmark --preset {PRESET} --fuse --samples 60")
print(f"Serve the latest fused model: uv run python main.py serve --preset {PRESET} --port 8080")
print("Supply SYSTEM_PROMPT in chat requests; the server does not inject it automatically.")''')
nb.cells = cells
nbf.validate(nb)
with Path('tutorial.ipynb').open('w') as output:
    nbf.write(nb, output)
print('Generated tutorial.ipynb')
