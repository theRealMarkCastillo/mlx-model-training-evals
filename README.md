# Local LoRA training and tool-call evaluation with Apple MLX

A tutorial and reference implementation for Qwen2.5 LoRA/QLoRA training on Apple Silicon. The task converts operational requests into JSON tool calls for deployment, pod restart, rollback, and cluster scaling.

Repository: [theRealMarkCastillo/mlx-model-training-evals](https://github.com/theRealMarkCastillo/mlx-model-training-evals).

**Result migration:** the JSON reports and chart originally checked into the top level of `artifacts/` are historical outputs from the earlier dataset and evaluator. Their accuracy and performance figures are not validated results for the corrected pipeline. Existing weights must be retrained on the regenerated data before making new quality claims. No replacement accuracy or fused-speed claim is published here.

## Quickstart

Use an Apple Silicon Mac with Metal support and Python 3.12 or later:

```bash
uv sync --locked
uv run python main.py prepare
uv run python main.py train --preset 3b --iters 80
uv run python main.py eval --preset 3b --samples 60
uv run python main.py benchmark --preset 3b
```

Downloads occur on first model use. Training and inference need memory beyond quantized model weights. Start with a short run and inspect measured peak memory before attempting a larger preset.

The interactive equivalent is:

```bash
uv run jupyter lab tutorial.ipynb
```

The notebook uses the same Python modules. Change `PRESET` in its training cell and rerun the following cells. It starts a new training run explicitly; it does not silently reuse historical adapters.

## Model presets

```bash
uv run python main.py models
uv run python main.py serve --preset 14b --base --port 8080
uv run python main.py train --preset 14b --iters 10
uv run python main.py eval --preset 14b --samples 60
```

| Preset | Qwen2.5 Instruct 4-bit model size | Batch | Adapted layers | Gradient checkpointing |
| --- | --- | ---: | ---: | --- |
| `3b` | 3B | 4 | 16 | Off |
| `7b` | 7B | 2 | 16 | On |
| `14b` | 14B | 1 | 16 | On |
| `32b` | 32B | 1 | 8 | On |
| `72b` | 72B | 1 | 4 | On |

These are starting configurations, not measured memory-fit guarantees. Equal iteration counts do not mean equal training exposure when batch sizes differ. The 72B preset is exploratory and requires a high-memory machine.

Preset configs are in `config/`. The scripts accept `--preset` directly. Training accepts `--config` instead of `--preset`; a missing or invalid config fails before loading weights. Configs must describe local chat JSONL splits, LoRA training, and prompt masking.

## Data and schema contract

`src/schema.py` defines strict parameter models and a discriminated union binding each tool to its parameters. The system prompt is generated from those definitions. Unknown fields, incorrect JSON types, duplicate keys, and nonstandard numeric constants are rejected. Defaults may be filled in a separate normalized representation; raw parsed output is preserved.

`data/prepare_dataset.py` produces balanced splits with a fixed seed:

- 200 training records, using template families 0 and 1.
- 40 validation records, using template family 2.
- 60 test records, using template family 3.

Prompts are unique within and across all splits. Every target notification channel appears in its request. Restart reasons copy the request's exact reason phrase instead of relying on an undocumented canonical vocabulary. Other omitted optional values follow the documented schema defaults.

The canonical record format is:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"{\"tool\":\"...\",\"parameters\":{...}}"}]}
```

Training validation, perplexity, and generation evaluation consume the same chat records. `raw_test_samples.json` is a generated inspection export; the evaluator does not read it.

The holdout tests unseen wording within this synthetic task. It does not establish reliability on arbitrary operational requests, ambiguous instructions, unsupported tools, or production traffic.

## Training and experiment identity

```bash
uv run python src/train.py --preset 3b --iters 80
uv run python src/train.py --config config/lora_config.yaml --iters 10 --output-dir /tmp/mlx-reports
```

The runner uses MLX-LM's callback-preserving `train_model` entry point. It records train loss, validation loss, throughput, timing, and Metal memory, and writes a loss curve. A run without expected loss telemetry fails rather than publishing an empty success report.

`mask_prompt: true` trains on assistant response tokens. The evaluator uses the same chat-template offsets and shifted-token loss mask, without truncating the holdout records. Training's `max_seq_length` still controls its sequence limit; keep it large enough for complete prompts and responses.

MLX uses `scale` as the direct multiplier for the LoRA update:

```text
W_effective = W_base + scale * B A
```

For `rank: 8, scale: 16.0`, the multiplier is 16. It is not an `alpha` value that MLX divides by rank. Some other LoRA implementations express their multiplier as `alpha / rank`.

Every run gets a new directory and manifest. The 3B report root is `artifacts/`; larger presets use `artifacts/qwen2.5-<size>/`.

```text
artifacts/
  runs/training-<id>/
    manifest.json
    adapters/adapter_config.json
    adapters/adapters.safetensors
    training_history.json
    loss_curve.png
  runs/evaluation-<id>/
    manifest.json
    eval_results.json
    eval_comparison.png
  runs/benchmark-<id>/
    manifest.json
    benchmark_results.json
  latest_training.json
  latest_evaluation.json
  latest_benchmark.json
  adapters/latest.json
```

Manifests record dependency versions, source hashes, configuration, dataset hashes, model snapshot revision (or local model file hashes), and adapter hashes. Evaluation and fusion reuse the base snapshot recorded by the adapter's training run. Completed runs retain their own weights and reports; small latest pointers select the most recent successful run.

`--output-dir` chooses the report root. Training weights live inside the training run, and the YAML's `adapter_path` becomes the directory containing its `latest.json` pointer. Evaluation, fusion, and serving resolve these pointers automatically. An explicit `--adapter` can select an older run's actual adapter directory. External/native MLX commands require that actual directory; they do not understand this project's pointer files.

## Quality evaluation

```bash
uv run python main.py eval --preset 3b --samples 60
uv run python src/evaluate.py --preset 3b --test-file data/test.jsonl --samples 60 --max-tokens 150
```

The report contains:

| Metric | Definition |
| --- | --- |
| Assistant loss / perplexity | Token-weighted cross entropy over the reference assistant response, using the training mask |
| Pure JSON rate | Entire response is a JSON object with no surrounding chatter |
| Schema validity | Recovered JSON satisfies the strict envelope and matching tool parameter schema |
| Tool accuracy | Decoded tool name matches the reference, independently of parameter validity |
| Exact match | Pure JSON, valid schema, and raw decoded object equals the reference; no added defaults or discarded fields |
| Normalized match | Valid schema and explicitly default-filled object equals the reference; reported separately and can include wrapped JSON |

Object key order and JSON whitespace do not affect exact match. Array order and parameter string contents do. A missing default can pass schema validation and normalized match while failing exact match. Malformed model outputs count as failures without aborting the run.

Every evaluated sample retains its input messages, expected answer, raw output, parsed and normalized objects, token counts, timings, finish reason, metric decisions, and error category. The generation and loss paths share the same selected records. Use the full 60-record split for comparisons; smaller counts select a prefix and can change tool balance.

Reports are saved in the printed run directory. `latest_evaluation.json` points to that completed run. The new metrics are not directly comparable to the historical full-conversation loss or permissive exact-match scores.

## Benchmarking and fusion

```bash
# Base and dynamic adapter, using the same system prompt and chat template as evaluation
uv run python main.py benchmark --preset 3b --runs 5 --warmup 2

# Fuse, benchmark all three variants, and evaluate their quality on the same holdout
uv run python main.py benchmark --preset 3b --fuse --samples 60

# Or create a fused model separately
uv run python main.py fuse --preset 3b
uv run python main.py eval --preset 3b --fused artifacts/fused_model --samples 60
```

Benchmarks record time to first token, prompt/prefill throughput, decode throughput, end-to-end throughput, total latency, and peak Metal allocation. Streaming metadata supplies token counts; generated text is not retokenized to estimate them. The MLX generation count includes a terminal EOS token when generated. Decode/prefill rates use MLX's measurements; TTFT and total latency use wall-clock time. Warmup runs are excluded, and every measured run is retained.

The benchmark uses one operational prompt, so its results describe that workload and its generated lengths. Metal allocation is not total process or operating-system memory. Compare model variants using the same machine and runtime conditions.

Fusion removes the dynamic adapter branch. Quantized fusion can alter outputs, and the speedup is workload-dependent. Fused speed is reported only after loading and measuring the fused model. `benchmark --fuse` and `benchmark --fused PATH` also create a linked quality evaluation of base, dynamic adapter, and fused model. A standalone fusion manifest records that quality has not yet been evaluated.

Fused runs live under `artifacts/fused_model/runs/` (or the larger preset's corresponding directory), with a latest pointer. `--save-path` on `main.py fuse` changes that fusion root.

## Serving

```bash
uv run python main.py serve --preset 3b --port 8080
```

This resolves the latest fused model. Use `--base` for an unadapted preset or `--model /path/to/actual/model` for a specific model. Supply the same system prompt used during training when making chat requests; the server does not inject this project's task prompt automatically.

## Validation and maintenance

```bash
uv run python -m unittest discover -s tests -v
uv run python scripts/build_notebook.py
```

Tests cover strict parsing, raw-versus-normalized scoring, grounded labels, reproducible and disjoint data, assistant loss masking, streaming measurements, CLI failures, full result persistence, and a real tiny-model MLX LoRA training run that checks callback delivery and artifact isolation. They do not establish memory fit or quality for the large pretrained models.
