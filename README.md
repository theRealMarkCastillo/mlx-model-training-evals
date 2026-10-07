# Fine-tuning and honestly evaluating a local LLM with Apple MLX

A hands-on tutorial: train a LoRA adapter for Qwen2.5 on your Mac so it turns operational requests into strict JSON tool calls, then measure whether it actually worked.

> *"Ship v2.3.1 of auth-api into prod with 3 instances."*
> → `{"tool":"deploy_service","parameters":{"service":"auth-api","version":"v2.3.1","environment":"production","replicas":3,"notify_channels":[]}}`

The training loop is the easy part. This repo is mostly about the questions to ask of *any* fine-tune:

| Question | Where it's answered |
|---|---|
| What exactly is the model trained on? | `main.py show-mask` (which tokens the loss scores) |
| Is fine-tuning even necessary? | zero-shot and **few-shot baselines** in every evaluation |
| How big is the change? | `main.py show-params` (LoRA parameter accounting) |
| Did it learn, or memorize? | loss curves, plus **challenge sets** with unseen entities, omitted defaults, and new kinds of unsupported requests |
| Is the improvement real, or noise? | **95% confidence intervals** and **paired McNemar tests** |
| What does it still get wrong? | failure categories, per-field accuracy, example failures |
| What do the knobs do? | `main.py ablate` (one hyperparameter at a time) |

Repository: [theRealMarkCastillo/mlx-model-training-evals](https://github.com/theRealMarkCastillo/mlx-model-training-evals).

## Quickstart

Apple Silicon Mac, Python 3.12+, [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python main.py prepare                    # generate data (deterministic)
uv run python main.py show-mask                  # see what training scores
uv run python main.py train --preset 3b          # ~20 min on an M2 Max
uv run python main.py eval --preset 3b --challenge
```

Or follow the guided notebook, which runs the same code with explanations at each step:

```bash
uv run python main.py notebook
```

The model downloads on first use (3B ≈ 2 GB). Before trying a larger preset, run a short job (`train --preset 14b --iters 10`) and check the peak-memory line.

## Reference results (Qwen2.5-3B, M2 Max)

One complete run on an M2 Max (96 GB), default settings: 200 iterations, batch 4, rank 8, seed 42. Training took about 19 minutes, peaked at 15.9 GB Metal memory, and trained 6.65M parameters (0.216% of 3.09B). Charts and the full numbers are in [`docs/reference-run/`](docs/reference-run/). Your numbers will differ in the last digits; Metal kernels are not bit-reproducible.

**Exact match, % [95% interval], greedy decoding:**

| Set | n | Base zero-shot | Base few-shot (5 shots) | LoRA |
|---|---:|---:|---:|---:|
| `test` (new wording) | 75 | 0 [0–5] | 68 [57–77] | **100** [95–100] |
| `challenge_defaults` (all optionals omitted) | 40 | 0 [0–9] | 58 [42–71] | **100** [91–100] |
| `challenge_abstain` (new unsupported kinds) | 40 | 0 [0–9] | **100** [91–100] | 95 [83–99] |
| `challenge_entities` (unseen names) | 40 | 0 [0–9] | 60 [45–74] | 75 [60–86] |

What the run shows:

* **Fine-tuning earned its keep here.** LoRA beats few-shot on the holdout (24 samples only LoRA gets right, 0 the other way; McNemar p ≈ 1e-7). Few-shot is not free either: its prompt is 985 tokens against 666, paid on every request.
* **Zero-shot's 0% is mostly a format result, not a capability one.** 95% of outputs are valid JSON, but the model writes `{"deploy_service": {...}}` or bare parameters instead of the `{"tool": ..., "parameters": ...}` envelope. The system prompt describes that envelope in words and never shows one. I haven't tested a zero-shot prompt that includes an example envelope, and the adapter was trained against this exact prompt, so it stays unchanged; few-shot is the fairer prompted baseline.
* **100% on the holdout overstates what was learned.** On unseen entities LoRA drops to 75%, and all 10 failures are `rollback_deployment`. The model wrote `dep-9821` (the example in the system prompt) or `dep-9148` for a request naming `email-dispatcher-r9148`: it learned the `dep-NNNN` shape rather than copying the identifier from the request. The other three tools score 100% on unseen names. The unseen identifier format is a deliberately large shift, so read this as one specific failure, not as a general 75%.
* **Abstaining generalizes, mostly.** The two LoRA misses answered `missing_required_parameter` for "increase the memory limit of X to 4Gi", a kind of request never seen in training.
* **Validation loss falls from 1.76 to 0.002 by iteration 150**, and the best checkpoint (iteration 149) is not the last. Most of the learning happens in the first 50 iterations, which the ablation below quantifies.
* **Dynamic adapters cost decode speed:** 41 tokens/s against 102 for the base model on this prompt. Fusing the adapter into the weights should recover most of that speed, but this reference run did not measure it: run `benchmark --fuse` to see the speed and the effect of requantization on quality.

**Ablation over training iterations** (`main.py ablate iters 25 50 100 200`, one seed, `test` set, n = 75):

| Iterations | Exact match [95%] | Schema valid | Test loss | Main failure |
|---:|---:|---:|---:|---|
| 25 | 27% [18–38] | 36% | 0.134 | invalid JSON/schema (19 + 29 of 75) |
| 50 | 75% [64–83] | 88% | 0.041 | 9 no JSON; 8 wrong tool; weak on `deploy_service` and `no_action` |
| 100 | 61% [50–72] | 100% | 0.036 | 22 wrong tool: answers `no_action` to real restart and scale requests |
| 200 | 100% [95–100] | 100% | 0.001 | none |

The point to take from it: format is learned first (schema validity reaches 100% by iteration 100), correct tool choice and parameters come later, and progress is not monotonic. At iteration 100 the model over-abstains, and its validation loss bumped up (0.039, from 0.015 at iteration 50) at the same time. The 50 and 100 intervals overlap and this is a single seed, so treat the dip as suggestive, not established. The four runs share a seed and follow the same trajectory (their validation losses at iterations 25 and 50 are identical), so this sweep is closer to "checkpoints of one run" than to four independent experiments.



## The task and the data

`src/schema.py` defines five tools as strict Pydantic models: `deploy_service`, `restart_pod`, `rollback_deployment`, `scale_cluster`, and `no_action`. The model should answer `no_action` when no tool fits or a required value is missing, instead of guessing. The system prompt is generated from the models, so instructions and validation can't drift apart.

`src/generate_data.py` builds a seeded, synthetic dataset with two generalization axes controlled separately:

* **Wording.** Each tool has four phrasing families: train uses 0–1, validation 2, test 3.
* **Entities.** Standard splits share one pool of service/pod/region/cluster names; `challenge_entities` uses names never seen in training.

About 30% of requests omit optional values, which must then be written with their documented defaults.

| File | n | Wording | Entities | Purpose |
|---|---:|---|---|---|
| `train.jsonl` | 250 | families 0–1 | seen | training (50 per tool) |
| `valid.jsonl` | 50 | family 2 | seen | validation loss during training |
| `test.jsonl` | 75 | family 3 | seen | holdout (15 per tool) |
| `challenge_entities.jsonl` | 40 | families 0–1 | **unseen** | do new names and values transfer? |
| `challenge_defaults.jsonl` | 40 | family 3 | seen | **every** optional value omitted |
| `challenge_abstain.jsonl` | 40 | all | seen | **kinds** of unsupported request not seen in training |

No prompt appears in more than one file. Each record carries a `meta` field (tool, family, omitted fields) that the evaluator uses to slice results. This is still a synthetic task: good scores here say nothing about messy production traffic.

## Training

```bash
uv run python main.py train --preset 3b
uv run python main.py train --preset 3b --iters 50 --rank 4      # quick overrides
uv run python main.py train --config my_config.yaml              # standalone MLX-LM config
```

LoRA freezes the base weights and learns a low-rank update per adapted projection:

```text
W_effective = W + scale · (B @ A)      A: r × d_in,  B: d_out × r,  B initialised to 0
```

MLX applies `scale` directly; it is not `alpha / r` as in some other libraries. With `mask_prompt: true`, the loss covers only the assistant's answer. In this task that is about 16–30 of ~650 tokens per record (`show-mask` highlights them).

All shared settings live in `config/base.yaml`. A preset changes only what must scale with model size:

| Preset | Model (4-bit) | Batch | LoRA layers | Grad checkpointing |
|---|---|---:|---:|---|
| `3b` | Qwen2.5-3B-Instruct | 4 | 16 | off |
| `7b` | Qwen2.5-7B-Instruct | 2 | 16 | on |
| `14b` | Qwen2.5-14B-Instruct | 1 | 16 | on |
| `32b` | Qwen2.5-32B-Instruct | 1 | 8 | on |
| `72b` | Qwen2.5-72B-Instruct | 1 | 4 | on |

These are starting points, not guaranteed memory fits. Equal `iters` with different batch sizes means different numbers of examples seen. The training summary reports examples seen, epochs, the best validation iteration, LoRA parameter count, and peak Metal memory, and warns if validation loss rises after its minimum. The config is validated by a Pydantic model (`src/config.py`) before anything downloads.

## Evaluation

```bash
uv run python main.py eval --preset 3b                       # test set: base, few-shot, LoRA
uv run python main.py eval --preset 3b --challenge           # plus the three challenge sets
uv run python main.py eval --preset 3b --variants base fewshot   # baselines only, no adapter needed
```

| Variant | What it is |
|---|---|
| `base` | base model, system prompt only (zero-shot) |
| `fewshot` | base model plus 5 worked examples from the training split (one per tool) |
| `lora` | base model plus the latest trained adapter |
| `fused` | adapter merged into weights (`--fused PATH` or `benchmark --fuse`) |

Decoding is greedy, and every variant sees the same records. Metrics:

| Metric | Definition |
|---|---|
| Assistant loss | Token-weighted cross-entropy on the reference answer, with the training mask |
| Pure JSON | The whole response is one JSON object, with no prose or fences |
| Schema valid | Recovered JSON satisfies the matching tool's strict schema |
| Tool accuracy | Correct tool name |
| **Exact match** | Pure JSON, schema valid, and identical to the reference, including defaults and types |
| Normalized match | Equal after filling omitted defaults (weaker; reported separately) |

Each rate has a **95% Wilson interval**. With 15 samples per tool, a measured 80% is compatible with anything from about 55% to 93%. The report also includes:

* **Paired comparisons:** McNemar's exact test of LoRA against each other variant on the same samples. Only discordant samples (one right, one wrong) count as evidence.
* **Failure categories:** `json`, `schema`, `format` (wrapped in prose), `tool`, `omitted_default` (correct values but a default left out), `parameters`.
* **Per-tool and per-field accuracy**, and slices with and without omitted values.
* **Example failures**, with the wrong fields listed.

`--samples N` takes a tool-balanced subset. Use the full sets for any comparison you intend to report. Every sample's prompt, output, parse result, verdicts, token counts, and timings are saved in `eval_results.json`.

Loss and exact match answer different questions. Loss measures how much probability the model puts on the reference answer; exact match judges one greedy output, all or nothing. Use loss to watch training and task metrics to decide.

## Teaching tools

```bash
uv run python main.py show-mask --split test --index 3   # highlight loss-scored tokens
uv run python main.py show-params --preset 3b            # adapter size per projection
uv run python main.py ablate iters 25 50 100 200         # train + evaluate per value
uv run python main.py ablate rank 2 8 32 --iters 100
```

Ablations write `ablation.png` (exact match ± CI and losses vs the parameter) and keep their adapters inside the ablation run, so your latest adapter is never replaced.

## Benchmarking, fusion, and serving

```bash
uv run python main.py benchmark --preset 3b              # base vs dynamic adapter
uv run python main.py benchmark --preset 3b --fuse       # also fuse, benchmark, and re-evaluate quality
uv run python main.py serve --preset 3b                  # latest fused model, OpenAI-compatible API
uv run python main.py serve --preset 3b --base
```

The benchmark reports time to first token, prefill/decode/end-to-end throughput, latency, and peak Metal allocation over measured runs (warmup excluded) for one fixed prompt. Fusion removes the adapter's extra matmuls. With a quantized base it re-quantizes merged weights, which can change outputs, so `--fuse` re-runs the quality evaluation. The server does not add this project's system prompt; send `SYSTEM_PROMPT` from `src/schema.py` with each request.

## Artifacts and provenance

Each preset has one root, `artifacts/qwen2.5-<size>/`, and every run gets its own immutable directory:

```text
artifacts/qwen2.5-3b/
  runs/training-<id>/      manifest.json, adapters/, training_history.json, loss_curve.png
  runs/evaluation-<id>/    manifest.json, eval_results.json, eval_comparison.png, challenge_comparison.png
  runs/benchmark-<id>/     manifest.json, benchmark_results.json
  runs/ablation-<id>/      manifest.json, ablation.json, ablation.png, <param>-<value>/...
  adapters/latest.json     → newest successful training run's adapter
  fused_model/latest.json  → newest successful fusion
  latest_<kind>.json       copy of the newest successful manifest per kind
```

Manifests record dependency versions, git commit and dirty state, source file hashes, the full config, dataset hashes, the model snapshot revision, and adapter hashes. If a run fails, its manifest is marked `failed` with the traceback, and no latest pointer is published.

Defaults resolve **only** through `latest.json` pointers. If no training run has completed, `eval`/`fuse`/`serve` stop with a clear error instead of picking up stale weights. Pass `--adapter` (a run's `adapters/` directory or any directory containing `latest.json`) to choose an older run. Outputs from the pre-rewrite pipeline are in `artifacts/historical/` and are not comparable.

## Code map

| File | Role |
|---|---|
| `main.py`, `src/cli.py` | CLI; every subcommand calls the same functions as the notebook |
| `src/schema.py` | tool schemas, generated system prompt, strict parser |
| `src/generate_data.py` | dataset and challenge-set generator |
| `src/dataset.py` | loading, validation, balanced subsets, few-shot construction |
| `src/config.py`, `src/models.py` | base config + preset overrides, Pydantic validation |
| `src/train.py` | MLX-LM training with telemetry and provenance |
| `src/metrics.py` | per-sample scoring, Wilson intervals, McNemar, breakdowns |
| `src/evaluate.py` | variants × datasets evaluation, plots, report |
| `src/explain.py` | loss-mask view, LoRA parameter accounting |
| `src/ablation.py`, `src/benchmark.py`, `src/fuse.py` | sweeps, speed, fusion |
| `src/runs.py` | run directories, manifests, latest pointers |

## Development

```bash
uv run python -m unittest discover -s tests -v
uv run python scripts/build_notebook.py     # tutorial.ipynb is generated; edit the script
```

The tests cover strict parsing, scoring and statistics, dataset design and grounding, loss masking, config validation, pointer and failure handling, CLI routing, and real MLX LoRA training on tiny models, including an offline end-to-end run with a miniature Qwen model. They don't establish memory fit or quality for the large pretrained models.

## Troubleshooting

* **`[METAL] Command buffer execution failed: Impacting Interactivity`.** macOS can abort a long GPU job when something else needs the GPU, such as the display or another GPU-using app. It showed up twice during the reference run (a 200-iteration training run and an ablation point), and the likely cause was other applications competing for the GPU. It is not a data or code error. The run is recorded as `failed` and publishes no pointer, so rerun it. Closing other GPU-heavy apps (browsers with video or WebGL, other model runners, screen recorders) and not running tests during training should help.
* **`No completed run recorded at .../latest.json`.** Nothing has been trained for that preset yet (or the last run failed). Run `main.py train --preset <size>`, or pass `--adapter` explicitly.

## Limitations

* The task is synthetic and narrow. Wording, entities, and request types are drawn from small templates, which is useful for controlled experiments but is not real traffic.
* Test sets are small (40–75 records). Read the intervals; differences of a few points are usually noise.
* Greedy decoding with no constrained/grammar-guided generation. Constrained decoding is a natural next comparison (it would remove most `json`/`format` failures without training) and is not implemented here.
