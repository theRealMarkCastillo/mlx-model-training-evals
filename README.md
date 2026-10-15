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
| What is the training loop actually doing? | `main.py toy-train` (the same loop by hand, no downloads, ~1 s) |

Repository: [theRealMarkCastillo/mlx-model-training-evals](https://github.com/theRealMarkCastillo/mlx-model-training-evals).

Theory behind every number printed here, one page per concept: [`docs/concepts.md`](docs/concepts.md).

```mermaid
flowchart LR
    A[prepare<br/>seeded data + challenge sets] --> B[train<br/>LoRA adapter, run manifest]
    B --> C[eval<br/>base / few-shot / LoRA vs holdout + challenges]
    B --> D[ablate<br/>one knob at a time]
    B --> E[benchmark<br/>speed + memory]
    E --> F[fuse<br/>merge adapter, re-check quality]
    F --> G[serve<br/>OpenAI-compatible API]
    C --> H[report<br/>Wilson intervals, McNemar, failure taxonomy]
    D --> H
```

## Quickstart

Apple Silicon Mac, Python 3.12+, [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python main.py toy-train                  # optional first step: the loop in miniature, no downloads
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

Re-running the same evaluation later with `--constrained` (same adapter, same seed) reproduced the three columns below exactly — a small reproducibility check on top of the intervals.

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

**How much of that is the seed?** The table above is seed 42. Repeating the identical setup for two more seeds (`main.py ablate seed 42 --iters 200`, one seed per invocation so an interrupted run costs only that seed, merged by `scripts/merge_seed_sweep.py`) gives:

| Seed | Exact match (test, n=75) | Test loss | Best val loss |
|---:|---:|---:|---:|
| 42 | 100.0% [95–100] | 0.0011 | 0.0023 |
| 43 | 97.3% [91–99] | 0.0010 | **0.0001** |
| 44 | 96.0% [89–99] | 0.0032 | 0.0012 |
| **across seeds** | **97.8% ± 1.7** | | |

Two things follow. The honest headline is **97.8% ± 1.7**, not 100%: the intervals overlap heavily, so the runs agree, and the last digit of any single run means nothing. And seed 43 had the *best* validation loss of the three while scoring *below* seed 42 on exact match — a small, concrete instance of the point the metrics section makes: loss and task accuracy answer different questions. The merged numbers live in [`docs/reference-run/seed_sweep.json`](docs/reference-run/seed_sweep.json).

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
| `general.jsonl` | 24 | — | — | ordinary requests (arithmetic, facts, rewriting) for the forgetting check, with a plain assistant system prompt |

No prompt appears in more than one file. Each record carries a `meta` field (tool, family, omitted fields) that the evaluator uses to slice results. This is still a synthetic task: good scores here say nothing about messy production traffic.

## Training

```bash
uv run python main.py train --preset 3b
uv run python main.py train --preset 3b --iters 50 --rank 4      # quick overrides
uv run python main.py train --preset 7b --batch-size 4 --no-checkpoint   # more memory: bigger batch, no recompute
uv run python main.py train --preset 32b --optimal               # auto-tune batch/layers/checkpointing for your RAM
uv run python main.py train --config my_config.yaml              # standalone MLX-LM config
```

LoRA freezes the base weights and learns a low-rank update per adapted projection:

```text
W_effective = W + scale · (B @ A)      A: r × d_in,  B: d_out × r,  B initialised to 0
```

MLX applies `scale` directly; it is not `alpha / r` as in some other libraries. With `mask_prompt: true`, the loss covers only the assistant's answer. In this task that is about 16–30 of ~650 tokens per record (`show-mask` highlights them).

All shared settings live in `config/base.yaml`. A preset changes only what must scale with model size:

| Preset | Model (4-bit) | Batch | LoRA layers | Grad checkpointing | Learning rate |
|---|---|---:|---:|---|---|
| `3b` | Qwen2.5-3B-Instruct | 4 | 16 | off | 1e-4 |
| `7b` | Qwen2.5-7B-Instruct | 2 | 16 | on | 1e-4 |
| `14b` | Qwen2.5-14B-Instruct | 1 | 16 | on | 1e-4 |
| `32b` | Qwen2.5-32B-Instruct | 1 | 8 | on | 1e-4 |
| `72b` | Qwen2.5-72B-Instruct | 1 | 4 | on | 1e-5 |

These are starting points, not guaranteed memory fits. `72b` drops the learning rate to `1e-5` (overriding `config/base.yaml`'s `1e-4`) to prevent float16 residual overflow in its deep 80-layer architecture. Equal `iters` with different batch sizes means different numbers of examples seen. The training summary reports examples seen, epochs, the best validation iteration, LoRA parameter count, and peak Metal memory, and warns if validation loss rises after its minimum. The config is validated by a Pydantic model (`src/config.py`) before anything downloads.

## Evaluation

```bash
uv run python main.py eval --preset 3b                       # test set: base, few-shot, LoRA
uv run python main.py eval --preset 3b --challenge           # plus the three challenge sets
uv run python main.py eval --preset 3b --variants base fewshot   # baselines only, no adapter needed
uv run python main.py eval --preset 3b --variants base --constrained   # base + grammar-constrained decoding
```

| Variant | What it is |
|---|---|
| `base` | base model, system prompt only (zero-shot) |
| `fewshot` | base model plus 5 worked examples from the training split (one per tool) |
| `lora` | base model plus the latest trained adapter |
| `grammar` | base model with **grammar-constrained decoding** (`--constrained`): valid JSON by construction |
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

* **Paired comparisons:** McNemar's exact test of the reference variant (LoRA when present, otherwise the last variant) against each other variant on the same samples, for both exact match and schema validity. Only discordant samples (one right, one wrong) count as evidence.
* **Failure categories:** `json`, `schema`, `format` (wrapped in prose), `tool`, `omitted_default` (correct values but a default left out), `parameters`.
* **Per-tool and per-field accuracy**, and slices with and without omitted values.
* **Example failures**, with the wrong fields listed.

### The constrained-decoding comparison

`--constrained` adds the `grammar` variant: the same base model, decoded under a JSON grammar derived from the tool schemas (`src/constrained.py`). Every step masks tokens that would leave the language of schema-valid calls, so prose, markdown fences, and the wrong envelope are impossible — **without any training**.

This separates formatting failures from task failures. On the 3B reference preset (full sets, greedy, seed 42), adding the grammar to the **base** model produces:

| Set | Metric | Base | **Base + grammar** | Few-shot | LoRA |
|---|---|---:|---:|---:|---:|
| `test` (n=75) | schema valid | 0% | **100%** | 85% | 100% |
| `test` (n=75) | tool accuracy | 0% | 89% | 92% | 100% |
| `test` (n=75) | exact match | 0% | 43% [32–54] | 68% [57–77] | 100% [95–100] |
| `challenge_abstain` (n=40) | exact match | 0% | **100%** | 100% | 95% |
| `challenge_defaults` (n=40) | exact match | 0% | 42% | 58% | 100% |
| `challenge_entities` (n=40) | exact match | 0% | 40% | 60% | 75% |

What it shows:

* **Grammar-constrained decoding matches LoRA's schema validity exactly** on the holdout — 75 of 75 samples valid for both, no discordant pair (McNemar p = 1.0) — and lifts the base model from 0% to 89% tool accuracy and 43% exact match, with no training at all.
* What remains is squarely task error: the grammar variant's holdout failures are 8 wrong tools, 11 omitted defaults and 24 wrong parameter values. It never fails on format.
* On `challenge_abstain` it is the **best** variant (100%, against LoRA's 95%): once the envelope is guaranteed, the base model abstains correctly on kinds of request it never saw in training. LoRA's two misses there are parameter/format slips, not failures to abstain.
* The two fixes are complementary — grammar buys the contract, training buys the decisions — and few-shot prompting sits between them (68% exact match) at 985 prompt tokens per request against 666. (That sentence is about the 3B model; the next section shows it needs a size qualifier.)

The 0.5B smoke test this section used to quote (`base` 0/8 schema-valid vs `grammar` 8/8, paired p = 0.008) showed the same mechanism more cheaply; the 3B numbers above are the ones to cite.

`--samples N` takes a tool-balanced subset. Use the full sets for any comparison you intend to report. Every sample's prompt, output, parse result, verdicts, token counts, and timings are saved in `eval_results.json`.

Loss and exact match answer different questions. Loss measures how much probability the model puts on the reference answer; exact match judges one greedy output, all or nothing. Use loss to watch training and task metrics to decide.

### Sampling instead of greedy decoding

`--temperature T` (with `--seed`) samples instead of decoding greedily. The metrics and the paired tests still apply, but each variant becomes one stochastic draw rather than the model's single most likely output, which is the point: format validity usually gets *worse* and less predictable when you sample, and that is a cost of deployment worth measuring. The seed is recorded and re-applied per variant, so the run is reproducible.

```bash
uv run python main.py eval --preset 3b --temperature 0.7 --seed 7
uv run python main.py eval --preset 3b --temperature 0.7 --seed 7 --repeats 5   # plus pass@k
```

`--repeats k` draws k samples per record and adds a **pass@k** table: the first draw's rate (what the flat metrics report), every draw's rate, the fraction of records that *any* draw got right, and how many records no draw got. That is how you price sampling honestly — retrying recovers part of what one draw loses, and the never-right residue is what more attempts will not fix. With `--temperature 0` the same command measures run-to-run flakiness instead of sampling, which is a useful sanity check of the Metal non-determinism the reproducibility notes mention.

### Did fine-tuning hurt anything else?

Every other metric here measures the task. `main.py forgetting` measures the *cost* of the task: it scores the base model and the trained adapter on 24 ordinary requests (`data/general.jsonl` — arithmetic, facts, rewriting, sentiment) using assistant loss, with their own plain assistant system prompt rather than the ops one. That prompt switch is what makes drift visible: an adapter that has overfit the tool contract tends to answer arithmetic with JSON, and its loss on the ordinary answer rises.

```bash
uv run python main.py forgetting --preset 3b
```

The comparison is paired per record and tested with an exact **sign test** (not a difference of means), so a handful of records cannot produce a confident verdict by accident. The report prints base and LoRA mean loss, how many records got worse versus better, and the p-value, plus `forgetting.png` (per-record losses, red where fine-tuning hurt). On the 3B reference preset:

| Model | Mean loss | vs base | Records worse / better | Sign-test p |
|---|---:|---:|---:|---:|
| Base | 1.823 | — | — | — |
| LoRA | 1.215 | **−0.609** | 7 / 17 | 0.064 |

There is **no detectable forgetting**: the (non-significant) direction of the effect is that the adapter got *better* at ordinary requests. That is plausible for LoRA on a narrow task — the base weights never move, and practice at producing strict structured output helps general instruction-following — but read it as "not damaged on these 24 requests", never as "general ability preserved". The honest reading is asymmetric: a 24-record synthetic set can show that general ability was **not obviously destroyed**; it can never show that it was preserved.

### Preference tuning (DPO) against more SFT data

The reference adapter's one measured failure is `rollback_deployment` on unseen entities: it writes `dep-9821` instead of copying the identifier from the request. Two ways to fix it, compared on equal footing:

```bash
SFT=artifacts/qwen2.5-3b/runs/training-<id>/adapters         # the adapter being repaired (pass it explicitly)
uv run python main.py dpo-mine --adapter $SFT                # fresh prompts, new entities and id formats → (chosen, rejected) pairs
uv run python main.py dpo-mix --pairs <mine-run>               # add synthetic in-distribution (dep-NNNN) rollback pairs
uv run python main.py grpo --adapter $WEAK --method grpo     # RL from the exact-match reward (or --method sft for the matched control)
uv run python main.py dpo --adapter $SFT                     # continue the adapter with the DPO loss
uv run python main.py train --config my-control.yaml         # control: SFT on the *same prompts'* gold calls (data in the mining run's sft_plus/)
uv run python main.py compare sft=$SFT control=<dir> dpo=<dir>   # paired tests, targeted field, envelope validity, forgetting
```

`dpo-mine` generates prompts from two entity pools and two deployment-id formats that appear in no evaluation set (`generate_data.MINING_POOLS`, checked disjoint from every split and challenge prompt), greedy-decodes the SFT adapter, and keeps every wrong answer as the *rejected* side of a pair against the gold call. `src/dpo.py` documents the loss; the reference is the adapter at step 0, so no second model is held in memory and the loss at step 0 is exactly ln 2 (the run aborts if it is not).

Results on the 3B reference adapter, exact match %, greedy, one seed (full numbers in [`docs/reference-run/dpo/`](docs/reference-run/dpo/)):

| Set (n) | SFT | **Control** (SFT, same prompts) | DPO, 100 it | DPO, 10 it | DPO + NLL anchor |
|---|---:|---:|---:|---:|---:|
| `test` (75) | **100** | 96.0 | 78.7 | 82.7 | 78.7 |
| `challenge_defaults` (40) | **100** | 97.5 | 57.5 | 92.5 | 67.5 |
| `challenge_abstain` (40) | 95.0 | 92.5 | 95.0 | 95.0 | 95.0 |
| `challenge_entities` (40) | 75.0 | **97.5** | 72.5 | 82.5 | **97.5** |
| rollback `deployment_id`, `challenge_entities` | 0 | **100** | 0 | 40 | **100** |
| rollback `deployment_id`, `test` | **100** | **100** | 0 | 20 | 0 |
| schema-valid, `challenge_entities` | 100 | 100 | 75 | 85 | 100 |

What it shows:

* **More SFT data fixed it; DPO did not, as configured.** Control: `challenge_entities` 75 → 97.5 (McNemar +10/−1, p = 0.012), nothing else significantly worse.
* **DPO with an NLL anchor fixes the target equally well, then breaks something it was never shown.** `challenge_entities` reaches 97.5 and `deployment_id` 100, but `deployment_id` on the standard `test` falls from 100 to 0. Every mined pair has a non-`dep-` identifier on the chosen side and a `dep-NNNN` on the rejected side, so "never write `dep-`" separates all of them, and nothing in the pairs says `dep-` is right when the request contains it. The control does not have this problem because its data still contains the original `dep-NNNN` training records.
* **Unanchored DPO over-optimises.** By iteration 100 the rejected log-probability has fallen by ~117 nats while the chosen one barely moved (−0.14), validation reward accuracy has been 1.00 since iteration 10, and 25% of `challenge_entities` outputs are no longer schema-valid. Stopping at 10 iterations lands in between and is not better than SFT on anything.
* **Forgetting is not the issue.** All five adapters have lower general-request loss than the base model (1.82 → 1.21–1.38); none is significantly worse than it. DPO adapters sit slightly above SFT (1.36–1.38 vs 1.21), but the sign tests are against the base model, and none of the five shows more records worse than better.

Read it with the caveats it needs: one seed; 40-record challenge sets; mined pairs are 100% `rollback_deployment` (240 of 240 prompts failed, none of the 60 other-tool prompts did), so this tests one failure mode; `learning_rate` (5e-6) and `beta` (0.1) were not swept. **Follow-up: the missing `dep-NNNN` pairs were the cause.** `dpo-mix` adds 120 synthetic in-distribution rollback pairs (102 train): the prompt uses the training id format, chosen is the gold call, and rejected is the same call with the id rewritten into a mining-pool format (the opposite error to the mined pairs). Anchored DPO on that mixed set (306 train pairs, same hyperparameters) against the SFT adapter:

| Set (n) | SFT | Control (original) | Anchored DPO | **Anchored DPO, mixed pairs** | Control, mixed data |
|---|---:|---:|---:|---:|---:|
| `test` (75) | 100 | 96.0 | 78.7 | **100** | 70.7 (p < 1e-6) |
| `challenge_abstain` (40) | 95.0 | 92.5 | 95.0 | **95.0** | 40.0 (p = 1e-5) |
| `challenge_defaults` (40) | 100 | 97.5 | 67.5 | **100** | 75.0 |
| `challenge_entities` (40) | 75.0 | 97.5 | 97.5 | **97.5** (p = 0.012) | 70.0 |
| rollback `deployment_id`, `test` | 100 | 100 | 0 | **100** | 100 |

Anchored DPO on the mixed pairs fixes the unseen-entity failure (+10/−1 vs SFT) and no longer regresses anything: no paired difference from SFT on any other set, `dep-NNNN` rollbacks back at 100%. The pairs' coverage, not the DPO loss, caused the earlier regression. The mixed SFT control got worse, not better: it adds 306 rollback-only records to 250 originals, which skews the tool mix (abstention falls to 40%). So "control" in this table is not a like-for-like baseline for the mixed pairs; DPO's reference-relative loss is less sensitive to that skew, and the original control (96 / 92.5 / 97.5 / 97.5) remains the better SFT baseline. One seed and 40-record sets still apply, and the added pairs are synthetic.

**Three seeds change the comparison.** Repeating anchored DPO on the mixed pairs and the original SFT control with seeds 42, 43 and 44 (exact match %, mean [min–max]):

| Set (n) | SFT | Control, 3 seeds | **Anchored DPO, mixed pairs, 3 seeds** |
|---|---:|---:|---:|
| `test` (75) | 100 | 97.8 [96.0–100] | **100 [100–100]** |
| `challenge_abstain` (40) | 95.0 | 88.3 [82.5–92.5] | **95.0 [95.0–95.0]** |
| `challenge_defaults` (40) | 100 | 88.3 [80.0–97.5] | **100 [100–100]** |
| `challenge_entities` (40) | 75.0 | 93.3 [90.0–97.5] | **98.3 [97.5–100]** |

The seed-42 control looked competitive because it was the best of its three. The control varies by up to 17 points per set between seeds, and seed 43 raised general-request loss to 2.39, worse than the base model's 1.82. DPO is nearly identical across seeds and does not regress. Two reasons to be careful before crediting the loss itself: the control retrains from scratch with a new LoRA initialisation, while DPO only continues the 100% SFT adapter and varies batch order, so its lower variance is partly built in; and the fairer baseline, continuing SFT from the SFT adapter on the same gold calls, has not been run. What the data does support is that, as configured, the mixed-pair DPO procedure is a safe way to fix this failure, and that a single control seed was misleading.

**The fair baseline ties DPO.** `dpo --sft-only` continues the SFT adapter on the same 306 pairs' chosen responses with the same optimiser, learning rate, batches and NLL term, and drops only the preference term (3 seeds, exact match %, mean [min–max]):

| Set (n) | SFT | Continued SFT | Anchored DPO |
|---|---:|---:|---:|
| `test` (75) | 100 | 100 [100–100] | 100 [100–100] |
| `challenge_abstain` (40) | 95.0 | 95.0 [95–95] | 95.0 [95–95] |
| `challenge_defaults` (40) | 100 | 100 [100–100] | 100 [100–100] |
| `challenge_entities` (40) | 75.0 | 99.2 [97.5–100] | 98.3 [97.5–100] |
| general-request loss (base 1.82) | 1.215 | 1.263–1.271 | 1.373–1.385 |

Continued SFT reaches the same accuracy with a smaller forgetting cost, and each seed's gain on `challenge_entities` is significant against SFT (+10/−1 or +10/−0, p ≤ 0.012). So the preference term adds nothing measurable on this task. The earlier control's variance came from retraining from scratch, not from SFT as such, and the "three seeds" table above overstated DPO's advantage. This task has one correct answer per prompt, which is what SFT is built for.

Two things nearly invalidated the experiment, and both are checked in code now: `mlx_lm.load(..., adapter_path=...)` leaves the base model's biases and norm weights trainable (`freeze_all_but_lora` freezes them and verifies the trainable names equal the SFT adapter's), and a loaded adapter's float16 LoRA matrices turn AdamW's `eps` into zero (NaN loss within ten steps), so they are trained in float32. Separately, `latest.json` points at whichever run *finished last*, which here was a 5-iteration smoke test; pass `--adapter` explicitly to anything that matters.

### RL from a verifiable reward (GRPO) against SFT

DPO needs pairs; RL needs only a reward. Here the reward is the exact-match check the evaluation already uses (1 if the sampled call equals the expected call, else 0), and the gold answer is used inside the reward but never as a training target. `src/grpo.py` implements GRPO from scratch on `mlx_lm.generate.batch_generate`: sample 8 completions per prompt, standardise rewards within each group, and push up the completions that beat their own group's average. Because the update is on-policy there is no clipping or KL term.

```bash
uv run python main.py grpo --adapter $WEAK --method grpo --iters 50 --seed 42   # RL from the exact-match reward
uv run python main.py grpo --adapter $WEAK --method sft  --iters 50 --seed 42   # same prompts, order, optimiser, lr; cross-entropy on gold
```

`--method sft` minimises the same expression with weight 1 instead of the group advantage, so the two arms differ only in where the learning signal comes from.

**First finding: RL cannot start from the adapter whose failure we care about.** On 16 mined rollback prompts the 100%-accurate SFT adapter produced the exactly-right call in 0.8% of 128 samples at temperature 1.0 and 1.6% at 1.6 (where the JSON also starts to break). It is locked into writing `dep-NNNN`; sampling does not find the right answer, and groups whose samples all score 0 contribute nothing. SFT and DPO fix this failure because they *show* the answer; RL has to find it by exploration, and here it essentially cannot.

So the comparison starts from a deliberately under-trained adapter (5 SFT iterations; 53% exact at temperature 1.0, with reward variance on 9 of 16 prompts), where RL has signal. Both arms train 50 steps of 4 prompts from `data/train.jsonl`, learning rate 2e-5, 3 seeds. Exact match %, mean [min–max] over seeds:

| Set (n) | Start | SFT | GRPO |
|---|---:|---:|---:|
| `test` (75) | 77.3 | **92.4** [88.0–94.7] | 86.7 [86.7–86.7] |
| `challenge_abstain` (40) | 65.0 | **97.5** [92.5–100] | 92.5 [80–100] |
| `challenge_defaults` (40) | 72.5 | 98.3 [97.5–100] | 97.5 [97.5–97.5] |
| `challenge_entities` (40) | 85.0 | 93.3 [92.5–95] | **96.7** [95–97.5] |
| held-out greedy accuracy during training (50 prompts) | 76 | **96** [94–98] | 85.3 [84–86] |
| general-request loss (base 1.82) | 1.51 | 1.56–1.58 | 1.68–1.77 |
| time per run / peak memory | | 6 min / 16 GB | 21 min / 32–35 GB |

What it shows:

* **Both improve on the start, and SFT improves more, faster and cheaper** on in-distribution prompts (`test` +15 vs +9 points; every SFT seed is significantly better than the start on `test`, p ≤ 0.008, and so is every GRPO seed, p ≤ 0.039). GRPO sees 8x as many sequences per step and still lands lower.
* **GRPO is nominally ahead only on unseen entities** (96.7 vs 93.3), which is the pattern people cite for RL generalising where SFT memorises. The gap is one or two records out of 40 and neither arm's gain over the start is significant there (p ≥ 0.06), so this is a hint, not a finding.
* **GRPO forgets more.** Both stay below the base model's general-request loss (1.82), but GRPO moves further from the start (1.51) than SFT does.

Caveats: one learning rate and one temperature, no sweep; 50 steps (200 prompt visits); the start adapter is artificial; compute is not matched between arms (GRPO generates 32 samples per step); 3 seeds on 40-record challenge sets. With a verifiable reward and a gold answer for every prompt, SFT is the efficient tool; RL's case is tasks where a reward exists and gold answers do not.

### Does scale close the gap?

The 3B table leaves open whether the story is specific to one model size. Repeating the prompting-only variants (base, few-shot, grammar — no training) on the cached 14B and 32B models:

| Set | 3B base | 14B base | 32B base | 3B few-shot | 14B few-shot | 32B few-shot | 3B + gram | 14B + gram | 32B + gram | 3B LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test` | 0% | 65% [54–75] | 76% [65–84] | 68% | 92% [84–96] | 97% [91–99] | 43% | 57% [46–68] | 47% [36–58] | 100% |
| `challenge_abstain` | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% | 100% | 95% |
| `challenge_defaults` | 0% | 98% | 100% | 58% | 100% | 98% | 43% | 50% | 45% | 100% |
| `challenge_entities` | 0% | 93% | 93% | 60% | 100% | 100% | 40% | 50% | 50% | 75% |

Three things fall out of it:

* **Scale buys the format for free.** The base model reads the system prompt and produces the tool-call envelope on its own: 0% → 65% → 76% exact match on the holdout across 3B → 14B → 32B (schema validity 0% → 77% → 84%). The 3B result was not a capability gap; it was a size gap.
* **Few-shot scales to match the trained adapter.** Few-shot exact match on the holdout goes 68% → 92% → 97%, and the 32B few-shot number is within the noise of the 3B LoRA's 100% — no training, no adapter. This is the cheapest way to beat the headline result, and it is paid in prompt tokens (985 vs 666) on every request.
* **Constrained decoding is a scalpel, not a ladder.** It is the best variant on `challenge_abstain` at every size (0% → 100%: when the envelope is missing, the model fails to refuse; when it is guaranteed, the model refuses correctly). Everywhere else the base model already writes the contract, the grammar *lowers* it — on `test` 65% → 57% at 14B and 76% → 47% at 32B, on `defaults` 98–100% → 45–50%. The grammar guarantees validity but not completeness (it allows omitting optionals, and once it changes the decode path the model omits them: `omitted_default` failures jump from 1 to 14 at 14B), and at 32B it even produced two truncated-unfinished objects (schema validity 97%, not 100%), because the strict envelope is longer than the model's natural answer and the token budget runs out.

So the earlier sentence needs a qualifier: grammar buys the contract *when the model does not already have it*; training buys the decisions at every size, and so — at larger sizes — does few-shot prompting. Full per-size numbers are regenerable from [`docs/reference-run/capacity.json`](docs/reference-run/capacity.json) (`scripts/export_capacity.py`).

### Which one should you deploy?

Accuracy alone says the prompted 32B (97% on the holdout) is within noise of the trained 3B adapter (100%). Serving cost is where they differ. Every candidate below was scored on the same 195 requests (holdout plus the three challenge sets), single stream, greedy decoding, one Apple-silicon Mac; latency is per request, prefill included. "Continued" is the SFT adapter after continuing SFT on the mined pairs ([the DPO section](#preference-tuning-dpo-against-more-sft-data)), seed 42. Regenerate with `uv run python scripts/export_deployment.py` (output in [`docs/reference-run/deployment.json`](docs/reference-run/deployment.json)).

| Candidate | `test` | `abstain` | `defaults` | `entities` | Latency median / p95 | TTFT | Decode | Prompt tokens | Weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B SFT adapter, dynamic | **100** | 95 | **100** | 75 | 0.98 / 1.08 s | 0.47 s | 73 tok/s | 664 | 1.7 GB |
| 3B SFT adapter, fused | **100** | **100** | **100** | 75 | 0.67 / 0.72 s | 0.40 s | 144 tok/s | 664 | 1.7 GB |
| 3B continued adapter, dynamic | **100** | 95 | **100** | 97.5 | 0.98 / 1.08 s | 0.46 s | 72 tok/s | 664 | 1.7 GB |
| **3B continued adapter, fused** | **100** | **100** | **100** | **100** | **0.68 / 0.73 s** | 0.41 s | 143 tok/s | 664 | 1.7 GB |
| 14B few-shot | 92 | 100 | 100 | **100** | 3.73 / 3.96 s | 2.78 s | 39 tok/s | 983 | 8.3 GB |
| 32B few-shot | 97 | **100** | 97.5 | **100** | 9.13 / 9.55 s | 7.07 s | 18 tok/s | 983 | 18.4 GB |

* **Fusing is free here.** Merging the adapter takes median latency from 0.98 s to 0.67 s and decode from 73 to 144 tok/s (the controlled benchmark agrees: 1.04 s vs 0.70 s, peak memory 2.7 GB vs 2.2 GB). Quality did not move on the holdout or `defaults`; the dynamic and fused rows differ by two records on `abstain` (95 vs 100) and, for the continued adapter, one on `entities` (97.5 vs 100).
* **The fused continued adapter matches the larger models on every set** while being about 5.5x faster than 14B few-shot and 13x faster than 32B per request, in a fifth to a tenth of the weights memory, with a third fewer prompt tokens per request (664 vs 983) because it needs no demonstrations.
* **What the larger models buy** is robustness on unseen entities without any training. The first SFT adapter's 75% there is its known failure; continuing SFT on the mined pairs closes it (97.5 dynamic, 100 fused) at the same size and speed.

**Recommendation.** For this task and an Apple-silicon host, deploy the fused continued 3B adapter. Use few-shot 32B only if you cannot train at all, or as an offline checker or labeller for the 3B model's outputs. Prompted 14B is dominated: it is slower, less accurate on the holdout, and its one advantage (unseen entities) is closed by the continued adapter.

Caveats: the continued adapter is one seed here (its other two seeds scored 97.5–100 on `entities` unfused; see the DPO section), so treat 100 as the top of a 97.5–100 range; one machine and a single request at a time (batched serving changes the ratios, usually in favour of larger models' throughput per dollar, not per-request latency); weights size is the 4-bit safetensors on disk, not peak runtime memory for the 14B and 32B (peak was measured only for the 3B); the challenge sets have 40 records each; the task is synthetic.

### Real data: the BFCL pilot

The synthetic task could be protecting the conclusion, so the same comparison is re-run on real function schemas from [BFCL v3 `simple`](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (Apache-2.0): 395 records, one real function per record, evaluated with the same `{"tool","parameters"}` envelope and a per-record system prompt, but scored BFCL-style (name + arguments in the ground-truth acceptable sets, required present) rather than exact match. Run it with `python main.py bfcl`; details and provenance are in [`docs/real-data.md`](docs/real-data.md).

| Metric | Base (zero-shot) | Base + JSON grammar |
|---|---:|---:|
| schema-valid | 97.2% [95–98] | 93.9% [91–96] |
| tool accuracy | 98.2% [96–99] | 94.2% [91–96] |
| **argument accuracy** | **81.5%** [77–85] | **37.7%** [33–43] |

The 3B base model already writes the contract: a clear per-record prompt — function name, typed parameters, an envelope example — gets 97% of outputs schema-valid and 82% of arguments right, with no training and no grammar. So the grammar has nothing left to fix, and its constraints actively cost: argument accuracy *drops* to 37% (McNemar p ≈ 0). The failures are visible and consistent — sign flips (`base: 4` → `-4`) and the model's native `<tool_call>`/single-quote tokens leaking into string values.

The natural suspicion is that the grammar's whitespace-free, compact path causes this, so it was tested directly: allowing structural whitespace (and capping a whitespace run, like strings and arrays are already capped) changes almost nothing — argument accuracy stays 37.7%, and schema validity stays 94%. The degradation is not a tokenization artifact; it is the model fighting an envelope its native format does not produce, which the grammar (a structural constraint) cannot fix. `multiple` (195 records, 2–4 candidate functions per record, exercising the dependent-`parameters` node) confirms it on tool *choice*:

| Metric | Base (zero-shot) | Base + JSON grammar |
|---|---:|---:|
| schema-valid | 92.8% [88–96] | 89.7% [85–93] |
| tool accuracy | 94.9% [91–97] | 90.8% [86–94] |
| **argument accuracy** | **75.4%** [69–81] | **35.9%** [29–43] |

Tool choice survives the grammar (90.8%, p = 0.10), and argument accuracy collapses exactly as it did on `simple`.

The decisive question is then: where does the base model's 82% come from? Ablating the prompt — replacing the envelope example, type hints and "respond with only JSON" with just the function *names* — answers it:

| Argument accuracy | Full prompt | Minimal prompt (names only) |
|---|---:|---:|
| base (no grammar) | 81.5% | **0.0%** |
| base + grammar | 37.7% | **22.0%** |

Without the prompt's format guidance the base model emits **zero** structured calls — it explains the answer in prose, names the function in backticks, but never produces the envelope. The prompt and the grammar are two ways to supply the same *format contract*, and the prompt is by far the better carrier of it: it shapes the model's *intent* (what to output), so it gets the structure *and* the values right (82%); the grammar only constrains *structure* (what is allowed), so it forces the envelope while the model fills it with whatever its unshaped intent produces (22–37%). Grammar-constrained decoding is not a substitute for the prompt — it is a weaker, narrower version of it.

The repo's core claim is that LoRA fine-tuning is what finally teaches the contract, so that cell was measured too: a LoRA trained on a 270-record split of `simple` itself (in-distribution — BFCL ships no training data), evaluated on the held-out 80 records (`python main.py bfcl --records data/bfcl_split/test.jsonl --variants base grammar lora`):

| Metric (test split, n = 80) | Base | Grammar | LoRA |
|---|---:|---:|---:|
| schema-valid | 96.2% | 95.0% | 97.5% |
| tool accuracy | 97.5% | 95.0% | 100.0% |
| **argument accuracy** | **75.0%** | 33.8% | **72.5%** |

The adapter ties the base model (72.5% vs 75.0%, inside the ±10% interval of 80 records) rather than beating it — it does not buy back the decisions that prompting leaves on the table. That is the opposite of the synthetic task, where LoRA went 0% → 100%: there, the base model could not follow the prompt at all, so training *was* supplying the format. On real schemas the prompt already supplies it, and in-distribution fine-tuning adds nothing. The synthetic "training buys the decisions" conclusion was really "training buys the *format* when the prompt does not deliver it" — which is a property of the weak base model, not of fine-tuning in general.

Scale is the one lever that does move the number: the 14B base model reaches **88.9%** argument accuracy on the same 395 records (100% tool, 99.7% schema-valid), up from 81.5% at 3B — and the grammar still costs it (71.1%), so the grammar's harm is not a small-model artifact (`python main.py bfcl --preset 14b`; report at [`docs/reference-run/bfcl_simple_14b.json`](docs/reference-run/bfcl_simple_14b.json)). Quantization is a second, smaller lever: the 8-bit 3B base reaches 86.8% (vs 81.5% at 4-bit) — nearly the 4-bit 14B at about half the memory — while its grammar variant stays collapsed (33.9%), so 4-bit costs the small model a few real points but is not what breaks the grammar.

The abstention axis is where the synthetic task's grammar win was cleanest (0% → 100%), so it was re-run too. On BFCL `irrelevance` (237 records where the one provided function is a deliberate mismatch), the metric is the hallucinated-call rate — and the grammar does **not** help:

| Metric | Base (zero-shot) | Base + JSON grammar |
|---|---:|---:|
| hallucinated-call rate | 37.6% [32–44] | 45.1% [39–52] |

The base model already refuses 62% of the time (the synthetic base refused ~0% because it could not format at all), and forcing a valid envelope — with `no_action` offered as an alternative — leaves the model calling the irrelevant function slightly *more* often. Both of the synthetic grammar's wins, format and abstention, were the same artifact: they fixed a base model that could not follow a prompt. On real schemas the prompt already does the job, and the grammar adds nothing. Grammar-constrained decoding is a fix for *missing* format — and format is only missing when the prompt does not supply it.

## Teaching tools

```bash
uv run python main.py toy-train                          # from-scratch loop: forward, mask, mx.grad, AdamW, no downloads
uv run python main.py toy-train --full                   # the same toy task with all parameters trained
uv run python main.py show-data                          # offline: sizes, balance, omitted defaults, examples
uv run python main.py show-data --tokens                 # plus token counts exactly as training sees them
uv run python main.py show-mask --split test --index 3   # highlight loss-scored tokens
uv run python main.py show-params --preset 3b            # adapter size per projection
uv run python main.py inspect --variant base --index 0    # token-by-token probabilities for one decode
uv run python main.py eval --preset 3b --constrained       # base model + valid-JSON-by-construction decoding
uv run python main.py forgetting --preset 3b             # did fine-tuning hurt general requests?
uv run python main.py dpo-mine --adapter $SFT            # preference pairs from the adapter's own mistakes
uv run python main.py dpo --adapter $SFT                 # DPO on those pairs (see "Preference tuning" above)
uv run python main.py compare sft=$SFT dpo=<dir> ...     # several adapters, paired tests, forgetting
uv run python main.py eval --preset 3b --temperature 0.7 --seed 7   # sampled draw instead of greedy
uv run python main.py ablate iters 25 50 100 200         # train + evaluate per value
uv run python main.py ablate rank 2 8 32 --iters 100
uv run python main.py ablate seed 42 43 44 --iters 200   # honest headline: mean ± spread (resumable; see below)
```

`toy-train` is the decoder ring for everything else: `src/mini_train.py` rebuilds the training loop by hand on a toy task whose validation answers **cannot** be learned from the training set, so it shows the memorization signature (train loss → 0, val loss climbing above the guessing floor) in about a second, then prints how each piece maps to a `main.py train` setting.

`inspect` answers "why did it get that one wrong?" instead of just "it got that one wrong". It greedy-decodes one record and prints every token with its probability, the runner-up's probability, and — at the first token where the output leaves the reference answer — the model's full top-k distribution with the reference token's rank highlighted. On the base model over a `restart_pod` request it typically shows step 0 choosing `restart` (p≈0.69) while the reference `{"` has p≈0.00: the zero-shot failure is a format decision, not a capability one. Traces are saved to `runs/inspection-<id>/inspect_trace.json`.

Ablations write `ablation.png` (exact match ± CI and losses vs the parameter) and keep their adapters inside the ablation run, so your latest adapter is never replaced. Every sweep also prints its **across-point mean ± spread** (`ablation.json` has it under `spread`), which is the number to quote when the swept parameter is `seed`: one seed is one draw, and the spread is how much of the headline is luck.

A three-seed sweep is three full trainings, so it is written to be resumable — run one seed per invocation and merge, and the Metal abort described in Troubleshooting never costs you the seeds that already finished:

```bash
uv run python main.py ablate seed 42 --iters 200
uv run python main.py ablate seed 43 --iters 200
uv run python main.py ablate seed 44 --iters 200
uv run python scripts/merge_seed_sweep.py   # mean ± spread → docs/reference-run/seed_sweep.json
```

If a seed's run aborts, its manifest is marked `failed` with the traceback and no pointer is published, so simply rerunning that one seed is safe; the merge step deduplicates retries by seed.

## Glossary

| Term | One-line meaning |
|---|---|
| Adapter (LoRA) | A small set of trained matrices added to frozen base weights; here ~6.65M of 3.09B parameters |
| Rank (`r`) | How many independent directions the adapter can push a projection; higher = more capacity |
| Scale | The multiplier on the LoRA update. MLX applies it directly; it is **not** `alpha / r` |
| `mask_prompt` | Score the loss only on assistant tokens; system prompt and request are context |
| Cross-entropy | `-log P(correct token)` in nats; the number the optimizer minimizes |
| Perplexity | `e^loss`: the effective number of equally likely choices the model behaves as if it faced |
| Exact match | Pure JSON, schema-valid, and identical to the reference including defaults and types |
| Normalized match | Equal after omitted defaults are filled in; the weaker view |
| Wilson interval | The range of true rates compatible with a measured rate at a given n; used on every rate here |
| McNemar test | Paired test over discordant samples; answers "is the improvement real, or noise?" |
| Few-shot | Worked examples inserted as earlier chat turns; the baseline that asks "was training needed?" |
| Challenge set | A holdout that isolates one difficulty: unseen entities, all defaults omitted, or new request kinds |
| 4-bit quantization | Weights stored as 4-bit codes plus group scales; the adapter percentage is against the *logical* count |
| Prefill / decode | Prompt processed in parallel / output generated token by token; decode is usually slower |
| TTFT | Time to first token: prefill plus one decode step — the latency a user feels |
| Fusion | Merging `scale · B@A` into the weights, removing the adapter's per-token matmuls |
| Gradient checkpointing | Recompute activations during backprop to trade speed for memory on larger presets |

Longer explanations, with formulas and pointers to the code: [`docs/concepts.md`](docs/concepts.md).

## Benchmarking, fusion, and serving

```bash
uv run python main.py benchmark --preset 3b              # base vs dynamic adapter
uv run python main.py benchmark --preset 3b --fuse       # also fuse, benchmark, and re-evaluate quality
uv run python main.py serve --preset 3b                  # latest fused model, OpenAI-compatible API
uv run python main.py serve --preset 3b --base
```

The benchmark reports time to first token, prefill/decode/end-to-end throughput, latency, and peak Metal allocation over measured runs (warmup excluded) for one fixed prompt, as mean ± spread (Macs thermal-throttle mid-benchmark). Fusion removes the adapter's extra matmuls. With a quantized base it re-quantizes merged weights, which can change outputs, so `--fuse` re-runs the quality evaluation. The server does not add this project's system prompt; send `SYSTEM_PROMPT` from `src/schema.py` with each request — `scripts/chat.py` is a dependency-free example that does exactly that:

```bash
uv run python main.py serve --preset 3b                  # terminal 1
uv run python scripts/chat.py "Ship v2.3.1 of auth-api into prod with 3 instances"   # terminal 2
```

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
| `src/mini_train.py` | from-scratch toy loop: forward, masked loss, `mx.value_and_grad`, hand-written AdamW |
| `src/inspect.py` | token-level decode trace: per-step probabilities, top-k alternatives, divergence point |
| `src/show_data.py` | offline dataset summary: sizes, balance, omitted defaults, token counts, examples |
| `src/constrained.py` | schema-derived JSON grammar and the logits mask for grammar-constrained decoding |
| `src/forgetting.py` | general-capability loss comparison (base vs LoRA) with an exact sign test |
| `src/dpo.py` | preference-pair mining, DPO loss and trainer (reference = step-0 policy), SFT control data |
| `src/grpo.py` | GRPO from a verifiable reward and its matched SFT arm, on `mlx_lm` batched generation |
| `scripts/export_deployment.py` | deployment trade-off table: accuracy, latency, prompt cost and weights size per candidate |
| `src/compare.py` | side-by-side adapter comparison: exact match, paired McNemar, targeted field, forgetting |
| `src/ablation.py`, `src/benchmark.py`, `src/fuse.py` | sweeps, speed, fusion |
| `src/runs.py` | run directories, manifests, latest pointers |

Docs: [`docs/concepts.md`](docs/concepts.md) explains every number this repo prints; [`docs/exercise-solutions.md`](docs/exercise-solutions.md) works through the notebook exercises (masking, starved data, no abstention training, overfitting, scaling up, grammar decoding, seed spread, forgetting, sampling); [`docs/extending.md`](docs/extending.md) adds a sixth tool end to end; [`docs/real-data.md`](docs/real-data.md) scopes putting real function-calling data (BFCL) behind the same measurements — planned, not implemented; [`docs/REVIEW.md`](docs/REVIEW.md) is the review and roadmap behind the current shape of the project.

## Development

```bash
uv run python -m unittest discover -s tests -v          # full suite (needs MLX)
uv run python -m unittest discover -s tests -p "test_core.py" -v   # pure-Python core, no MLX
uv run python scripts/build_notebook.py     # tutorial.ipynb is generated; edit the script
uv run ruff check .                         # lint config lives in pyproject.toml
```

`tests/test_core.py` holds the pure-Python tests (schema parsing, scoring, statistics, dataset design, the offline data summary) and imports neither MLX nor transformers; a guard test enforces that. That split is what lets CI run the core suite on Linux and the full suite on an Apple Silicon runner (`.github/workflows/ci.yml`). The remaining files cover evaluation plumbing, loss masking, the toy loop's mechanics (LoRA identity at init, memorization-vs-generalization, determinism), real MLX LoRA training on tiny models, and an offline end-to-end run with a miniature Qwen model. They don't establish memory fit or quality for the large pretrained models.

## Troubleshooting

* **`[METAL] Command buffer execution failed: Impacting Interactivity`.** macOS can abort a long GPU job when something else needs the GPU, such as the display or another GPU-using app. It showed up twice during the reference run (a 200-iteration training run and an ablation point), and the likely cause was other applications competing for the GPU. It is not a data or code error. The run is recorded as `failed` and publishes no pointer, so rerun it. Closing other GPU-heavy apps (browsers with video or WebGL, other model runners, screen recorders) and not running tests during training should help.
* **`No completed run recorded at .../latest.json`.** Nothing has been trained for that preset yet (or the last run failed). Run `main.py train --preset <size>`, or pass `--adapter` explicitly.

## Limitations

* The task is synthetic and narrow. Wording, entities, and request types are drawn from small templates, which is useful for controlled experiments but is not real traffic. The BFCL pilot (see "Real data: the BFCL pilot") re-runs the headline comparison on real function schemas and finds the synthetic grammar/training wins do not transfer.
* Test sets are small (40–75 records). Read the intervals; differences of a few points are usually noise.
* Greedy decoding is the default and what the reference numbers use. `--temperature` samples and `--repeats k` reports pass@k, so the stochastic cost of structured output and the value of retrying are both measurable; what is still missing is a study of *which* records retrying fixes (the per-record data is in `eval_results.json`, the analysis is not written).
* The grammar variant of the synthetic eval is specialized to this envelope, not a general JSON-Schema compiler: whitespace outside strings, escapes inside them, and strings/arrays longer than the caps are excluded by construction. It answers "would valid JSON have been enough?" for this task, which is the question the repo is about. (`src/schema_grammar.py`, used by the BFCL pilot, generalizes this — whitespace, nested objects, typed arrays — but is not wired into the synthetic eval.)
* The constrained-decoding numbers come from one seed on the full evaluation sets. The mechanism (format failures vanish, task errors remain) is robust; the exact percentages in that table are point estimates with the intervals shown.

## Local configuration and generated files

The following files are local-only and ignored by Git:

- `.agents/hooks.json`
- `.claude/settings.json`
- `.codex/hooks.json`
- `.entire/settings.json`

Before pulling the commit that untracks them, back up any configured copies
outside the checkout. Git may remove the formerly tracked files during the
update; restore your copies afterward. Do not force-add them.
