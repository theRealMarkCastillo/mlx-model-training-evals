# Code & Design Review — mlx-model-training-evals

Reviewed: 2026-10-07, against commit `4096f91`. All 40 tests pass (`unittest discover -s tests -v`, ~3.5 s including real MLX LoRA training on tiny models and an offline end-to-end run). Notebook is byte-identical to `scripts/build_notebook.py` output. Git hygiene is good: 46 tracked files, ~2.3 MB total, no weights committed.

**Verdict: this is already one of the better educational ML repos out there. The review below is about the last 20%, split into (1) code findings, (2) design observations, (3) how to make it more educational, (4) how to make it more complete, and (5) a prioritized plan.**

> **Status: P0, P1 and most of P2 are implemented.** P0: bug fixes B1/B2/M1/M2/M4, the `seed` ablation, the from-scratch toy loop (`src/mini_train.py` + `main.py toy-train` + notebook §5b + tests), `docs/concepts.md` with the README glossary and pipeline diagram, and the packaging cleanup (MIT `LICENSE`, pyproject metadata, removed unused deps, ruff config). P1: the token-level inspector (`main.py inspect`), the dataset explorer (`main.py show-data`), the pure-Python core suite plus GitHub Actions CI, the `docs/exercise-solutions.md` and `docs/extending.md` guides, and the constrained-decoding comparison (`src/constrained.py`, `eval --constrained`). P2 partially: benchmark spread (mean ± sd) and `scripts/chat.py`. Still open: the multi-seed reference re-run, the forgetting check (C4), and the `--temperature` sampling study. Test suite: 85 tests, `ruff check .` clean, notebook regenerated and lint-clean.
>
> The constrained-decoding result is worth calling out. On the **3B reference preset** (full sets, greedy), grammar-masked decoding of the *base* model reaches **100% schema validity — identical to the trained LoRA adapter, with no discordant sample (McNemar p = 1.0)** — and lifts tool accuracy from 0% to 89% and exact match from 0% to 43% without training. Everything that remains is task error (8 wrong tools, 11 omitted defaults, 24 wrong values on the holdout), and on `challenge_abstain` the grammar variant is the best of all four (100% vs LoRA's 95%). Grammar buys the contract; training buys the decisions.
>
> Two later additions close the remaining P2 items: `main.py forgetting` (base vs LoRA assistant loss on 24 ordinary requests with a plain assistant prompt, compared with an exact sign test) answers "what did the task cost?" — on the 3B preset LoRA's general loss is *lower* than the base model's (1.215 vs 1.823, 7 records worse / 17 better, p = 0.064), i.e. no detectable forgetting, with the caveat that 24 synthetic records can never prove general ability is preserved. And `eval --temperature` samples instead of decoding greedily (seeded and recorded, and composable with `--constrained`: masked sampling still emits 100% valid JSON), with `--repeats k` adding a **pass@k** table so the value of retrying is measurable rather than asserted. 101 tests, ruff clean, `docs/reference-run/` refreshed with the grammar, forgetting, and seed-sweep results.

---

## 1. What is genuinely excellent (keep doing this)

- **The core pedagogical thesis** — the training loop is the easy part; the questions around it are the lesson. The README's question table (`What exactly is the model trained on? → show-mask` …) is the best summary of the repo's intent.
- **Challenge-set design** ([`src/generate_data.py`](../src/generate_data.py)) — wording families and entity pools as two independently-controlled generalization axes, plus three single-difficulty challenge sets (`entities`, `defaults`, `abstain`). This is real experimental design, and it's *tested* (grounding, disjointness, determinism — `tests/test_pipeline.py::DatasetTests`).
- **Statistical rigor** ([`src/metrics.py`](../src/metrics.py)) — Wilson intervals on every rate, exact McNemar on paired discordant samples, failure taxonomy with a strict precedence order, per-field and per-slice breakdowns. The docstring ("a measured 80% could plausibly be 55–93% at n=15") teaches the point in one sentence.
- **Strict parsing** ([`src/schema.py`](../src/schema.py)) — duplicate JSON keys rejected, `NaN`/`Infinity`/`1e999` rejected, strict Pydantic types (`"3"` ≠ `3`), embedded-object recovery distinguished from pure JSON. The parser is its own teaching tool, and every edge case is tested.
- **Provenance discipline** ([`src/runs.py`](../src/runs.py)) — immutable run dirs, manifests with dependency versions, git commit/dirty state, source-file hashes, dataset hashes, model snapshot revision, adapter hashes; failures recorded with tracebacks and *no* latest-pointer published. `resolve_adapter_source` even re-verifies the base model hasn't changed since training.
- **Loss-mask visibility** ([`src/explain.py`](../src/explain.py)) — `show-mask` reuses MLX-LM's own `ChatDataset`, so the picture *is* the training reality, not a reimplementation.
- **Notebook generated from a script** — no drift between `tutorial.ipynb` and its source; a real (and rare) engineering win for a tutorial repo.
- **Test quality** — including the offline end-to-end `test_workflow_integration.py` that builds a miniature Qwen model on disk and runs train → fuse → benchmark → evaluate without a network.

---

## 2. Code review findings

### 2.1 Real bugs (low severity, easy fixes)

**B1. Standalone `--config` resolves `data`/`adapter_path` against the CWD, presets against the repo.**
[`src/config.py:96-106`](../src/config.py#L96-L106) applies `repo_path()` only in the preset branch. [`src/train.py:116-122`](../src/train.py#L116-L122) then derives the artifacts root from `adapter_path`. Running `uv run python main.py train --config config/base.yaml` from anywhere but the repo root silently writes runs and `latest.json` next to your CWD, and training data lookup fails or picks the wrong directory. Fix: `repo_path()` the `data` and `adapter_path` values in the standalone branch too (absolute paths pass through untouched).

**B2. `show-mask --index N` crashes with a raw traceback on out-of-range input.**
[`src/cli.py:90`](../src/cli.py#L90) indexes the loaded split directly, and `IndexError` is not in the catch list at [`src/cli.py:213`](../src/cli.py#L213). Negative indices also silently wrap (Python semantics), which is confusing in a teaching tool. Fix: validate `0 <= index < len(records)` and raise `ValueError` with the split size.

### 2.2 Minor bugs / polish

**M1. MemoryError and other runtime failures print raw tracebacks despite being recorded.**
The run manifest *is* correctly marked `failed` with the traceback (`record_failure`), but [`src/cli.py:211-218`](../src/cli.py#L211-L218) only catches `ValueError`/`FileNotFoundError`, so an OOM during training dumps a full traceback at the learner. Fix: also catch `MemoryError` (suggest a smaller preset / `--iters 10`) and `RuntimeError`, and print the failed run's directory so the learner can inspect the manifest.

**M2. `parse_and_validate` counts any top-level JSON scalar as "valid JSON".**
[`src/schema.py:179-184`](../src/schema.py#L179-L184): the output `"hello"` or `42` gets `is_valid_json=True, is_pure_json=False` — it only fails at schema validation. "Valid JSON" then means something different from what the README's metrics table implies ("one JSON object"). Fix: set `is_valid_json` only when a dict is recovered (fold the `isinstance` check in), or rename the rate to `json_recoverable_rate` and document it.

**M3. `show-mask` downloads with `AutoTokenizer` but `show-params` needs the full model — worth one line in the README.**
Minor, but a learner running `show-params` first will download the whole 3B model (≈2 GB) when they expected parameter arithmetic. A "this command loads the model" note per teaching command costs nothing.

**M4. `run_training` seeds NumPy batch order but the CLI exposes no `--seed`.**
`seed` is validated in the config and honored ([`src/train.py:156`](../src/train.py#L156)), but there is no CLI override, unlike `--rank`/`--iters`/`--learning-rate`/`--num-layers`. This matters directly for the completeness item C1 below.

**M5. Benchmark reports means only.**
[`src/benchmark.py:33-41`](../src/benchmark.py#L33-L41) averages over measured runs but drops min/max/std. Macs thermal-throttle; a decode rate of 41 tok/s with min 38/max 44 is a different lesson from min 25/max 55. Report min/max/std alongside the means.

**M6. Local (non-snapshot) base models are fully re-hashed on every eval/benchmark.**
[`src/runs.py:141`](../src/runs.py#L141) calls `directory_identity` to verify a local base model unchanged — over a 2 GB 4-bit model that is minutes of hashing per run. The HF-snapshot path (the common one) already avoids this via revision identity. Fix: cheap (mtime, size) pre-check before re-hashing, or hash only config/tokenizer files plus a stored manifest.

### 2.3 Verified-correct things reviewers might otherwise flag

- `compute_perplexity` passes `lengths = [[offset, len(tokens)-1]]` to `default_loss` — verified against the installed MLX-LM 0.32 source: the mask is `steps >= start and steps <= end`, so it covers the first assistant token through the final EOS, inclusive. Matches the training mask. ✔
- Few-shot loss **is** comparable to base/LoRA loss: the few-shot messages end with the original assistant turn, so `ChatDataset.process`'s mask covers only that final answer — the five demo turns are masked out. ✔ (Worth a one-line comment in [`src/evaluate.py:276`](../src/evaluate.py#L276) because it *looks* wrong.)
- `LoRA scale` semantics, `lora.CONFIG_DEFAULTS` extras validation, the ablation's private `adapter_path` so sweeps never clobber the preset's latest pointer — all correct.
- The few-shot demonstrations come from the **training** split, and prompt uniqueness across *all* files is enforced — no evaluation prompt can leak into the shots. ✔

---

## 3. Design review

| Area | Assessment |
|---|---|
| Architecture | Clean layered CLI → pipeline modules; every subcommand calls the same functions the notebook calls. `schema / generate_data / dataset / train / metrics / evaluate / explain` is a teachable dependency order. |
| Data design | Exceptional. Deterministic, seeded, committed; two generalization axes; per-record `meta` drives slices. The grounding tests are the real proof. |
| Statistics | Right tools (Wilson, McNemar) with the right caveats (single seed, shared trajectory disclosed honestly in the README). |
| Provenance | Above the bar for a tutorial; manifests rival research-tracker tooling. |
| Reproducibility | Good: single-seed caveats are now backed by an actual sweep (97.8% ± 1.7 across three seeds, `docs/reference-run/seed_sweep.json`). MLX Metal non-determinism is disclosed; see C1. |
| Packaging | `pyproject.toml` declares **three unused dependencies** — `tabulate`, `jsonschema`, and `datasets` (verified: nothing in `src/`, `tests/`, or `scripts/` imports them; `datasets` is only lazily imported by MLX-LM for HF-hosted data, which this repo never uses). No `license`/`authors` metadata, no `[dependency-groups]`, no linter config. |
| Error handling | `record_failure` + no-stale-pointer is excellent; CLI-level friendliness (B2, M1) is the weak spot. |

Design gaps worth naming:

- **The training loop is a black box.** The repo's own thesis says "the training loop is the easy part," but `train_model` from MLX-LM does it all, so a learner never sees backprop, the optimizer step, gradient accumulation, or how the loss mask actually reaches the loss function. See E1.
- **Single seed by default.** Every headline number is one trajectory; the README is admirably honest about it, but the tooling makes running the honest version (multiple seeds) awkward. See C1.
- **No token-level diagnostics.** When the model writes `dep-9821` instead of `dep-9148`, the repo reports *what* failed but not *why* — there is no way to see which token decision went wrong and what the alternatives were. See E3.
- **No check for catastrophic forgetting.** The challenge sets measure task generalization only; "did fine-tuning make the model worse at everything else?" is unanswered. See C4.
- **Constrained decoding is explicitly unimplemented** (README's Limitations) — the single most instructive *next* comparison for this exact task (format errors vanish without training). Keep it as the marquee roadmap item; see C5.

---

## 4. How to make it more educational

The repo already teaches via structure (loss mask, baselines, challenge sets, intervals, ablations). These five additions would cover the remaining pedagogical gaps, roughly in value order:

**E1. A from-scratch mini training loop (~100 lines + a notebook section).**
Implement `src/mini_train.py` (or §5b of the notebook): a tiny two-layer transformer, forward pass, masked cross-entropy with the *same* `(tokens, offset)` convention, `mx.grad`, a hand-written AdamW step, an eval pass — training to visibly falling loss on a toy 2-token task. Then show the one-to-one mapping to MLX-LM's `train_model` arguments (`iters`, `batch_size`, `steps_per_report`, `save_every`, `val_batches`). This single addition teaches more than any other: gradients, the mask, the optimizer, and why the CLI flags exist. It also gives the tests a place to assert *mechanics* (loss decreases, B starts at zero so `W_eff == W` at step 0) that the current tests can't, because they go through the real `train_model`.

**E2. `docs/concepts.md` — one page per number the repo prints.**
Cross-entropy → perplexity (why `log(2)` means "coin-flip confidence"), why loss curves are plotted log-scale, the LoRA equation with concrete arithmetic for the 3B preset (why rank 8 on 16 layers = 6.65M params), why B is initialized to zero, `scale` vs `alpha/r`, Wilson interval intuition ("with 15 samples, 80% could be 55–93%"), McNemar ("only discordant pairs count"), 4-bit quantization, prefill vs decode and TTFT, peak Metal memory. Each concept ends with "see it in this repo: <command>".

**E3. A token-level decision inspector: `main.py inspect --split test --index N`.**
For one generated example, show the greedy token sequence with, at each position, the top-5 candidate tokens and their probabilities (a single forward pass over the generated prefix, softmax over the last logits). Overlay the expected answer so the learner *sees* the fork where `rollback_deployment` beat the correct tool and how close it was. This turns the README's error analysis ("it learned the `dep-NNNN` shape") from an assertion into something observable. It is also the single best debugging tool for the exercise set.

**E4. A dataset explorer: `main.py show-data`.**
Offline: per-split counts, tool balance, omitted-default rates, family histograms, token-length distribution (needs the tokenizer), and 2–3 random records per tool rendered nicely. Right now, the only way to see the data is opening JSONL in an editor.

**E5. Supporting teaching material.**
- A **glossary** in the README (adapter, rank, scale, cross-entropy, perplexity, exact vs normalized match, Wilson interval, McNemar, quantization, gradient checkpointing, fusion, TTFT — one line each, hyperlinked to `concepts.md`).
- A **mermaid pipeline diagram** in the README (data → train → eval → challenge sets → ablation → benchmark → fuse → serve).
- **Exercise solutions** (`docs/exercise-solutions.md`) — the five notebook exercises are good; learners stall without somewhere to check.
- **"Add your own tool" walkthrough** (`docs/extending.md`): the exact five touchpoints for a sixth tool (schema model + map, generator + phrase families, `TOOL_GENERATORS`, counts, tests) — learners learn this codebase fastest by extending it.
- A "what to read next" list (LoRA paper, Wilson 1927, MLX-LM docs, lm-eval).

---

## 5. How to make it more complete

**C1. Multi-seed support (one-line change, real payoff).**
Add `"seed"` to `ABLATABLE` in [`src/ablation.py:23-24`](../src/ablation.py#L23-L24) (`int` type; the config already accepts `seed` overrides). Then `main.py ablate seed 42 43 44 --iters 200` produces the honest headline: mean ± spread across seeds instead of one trajectory. Update the README's reference table to a seed-sweep once run. This closes the largest honesty gap in the current results.

**C2. CI + test split.**
MLX is Apple-only, so free CI is tricky — but ~half the tests are pure Python. Split `tests/test_pipeline.py` into `tests/test_core.py` (schema, metrics, data generation, dataset — no `mlx`/`transformers` imports) and keep the MLX tests separate. Then a GitHub Actions workflow can run the core suite on Linux for free, plus an optional macOS-arm64 job for the MLX suite. This is the difference between "tutorial that bit-rots" and "project that stays green."

**C3. Tooling metadata.**
Add a LICENSE (MIT fits the educational intent), `license`/`authors`/`[project.urls]` to `pyproject.toml`, a `[dependency-groups] dev` group with `ruff`, `[tool.ruff]` config, and remove the three unused dependencies (`tabulate`, `jsonschema`, `datasets`). Add `[tool.pyright]`/type hints as budget allows — for an educational repo, annotations on the public pipeline functions (`score_sample`, `summarize`, `run_training`, `run_comprehensive_evaluation`) teach the data contracts.

**C4. Forgetting check (small version).**
Reuse the eval machinery with a tiny general-capability set (e.g. 20–30 diverse instruction/QA prompts in `data/general.jsonl`, scored by assistant loss only, or reuse `mlx_lm.evaluate` on a small lm-eval task). Add a `--general` flag to `eval` so every experiment reports "task ↑, general ↓?" — the classic question the repo currently can't answer.

**C5. Constrained decoding comparison (the marquee roadmap item).**
The README already names it. A JSON-schema-constrained decoder for this envelope (either via a grammar library or a hand-rolled mask over the 5-tool vocabulary — it's a tiny JSON language) would answer "would formatting have failed without fine-tuning?" — the single most instructive comparison the repo is missing. Effort: moderate; keep it as the P1 headline.

**C6. Small completeness wins.**
- `benchmark` min/max/std (M5), `--seed` CLI flag (M4).
- `eval --temperature N` (default 0, validated 0 < t ≤ 2) to open the sampling study; with greedy as the pinned default, nothing else changes.
- A `README.md` for `docs/reference-run/` noting it is regenerated by `scripts/export_reference.py`.
- `serve` already warns that the system prompt isn't added; a 5-line `scripts/chat.py` (OpenAI client against the local server) would close the loop for learners.

---

## 6. Prioritized plan

| Status | Priority | Item | Effort | Where |
|---|---|---|---|---|
| ✅ done | **P0** | B1 config-path fix, B2 index bounds, M1 friendly errors, M2 JSON semantics, M4 `--seed` | hours | `src/config.py`, `src/cli.py`, `src/schema.py` |
| ✅ done | **P0** | E1 mini training loop + notebook §5b + mechanics tests | 1–2 days | new `src/mini_train.py`, `scripts/build_notebook.py`, `tests/test_mini_train.py` |
| ✅ done | **P0** | E2 `docs/concepts.md`, glossary + mermaid in README | 1 day | `docs/concepts.md`, README |
| ✅ done | **P0** | C1 `seed` in `ABLATABLE` + the sweep itself: 97.8% ± 1.7 exact match across seeds 42/43/44 (`docs/reference-run/seed_sweep.json`) | hours + GPU-night | `src/ablation.py`, `scripts/merge_seed_sweep.py`, README |
| ✅ done | **P0** | C3 LICENSE, pyproject metadata, ruff, drop unused deps | hours | repo root |
| ✅ done | **P1** | E3 `main.py inspect` (token-level top-k) | 1–2 days | `src/inspect.py`, `tests/test_inspect.py`, CLI |
| ✅ done | **P1** | E4 `main.py show-data` | hours | `src/show_data.py`, `tests/test_core.py`, CLI |
| ✅ done | **P1** | C2 core-test split + GitHub Actions | hours | `tests/test_core.py`, `.github/workflows/ci.yml` |
| ✅ done | **P1** | E5 solutions, add-a-tool walkthrough, reading list | 1 day | `docs/exercise-solutions.md`, `docs/extending.md` |
| ✅ done | **P1** | C5 constrained decoding comparison | days | `src/constrained.py`, `tests/test_constrained.py`, `eval --constrained` |
| ✅ done | **P2** | C6 benchmark spread (mean ± sd) and `scripts/chat.py` | hours | `src/benchmark.py`, `scripts/chat.py` |
| ✅ done | **P2** | C4 forgetting check (`main.py forgetting`, exact sign test) + `--temperature` sampling option | days | `src/forgetting.py`, `data/general.jsonl`, `src/evaluate.py`, `src/inference.py` |

All of it has now been run. The three wall-clock items landed: the 3B evaluation with `--constrained` (grammar matches LoRA's schema validity exactly, 75/75), the 3B forgetting check (LoRA general loss 1.215 vs base 1.823, no detectable forgetting), and the three-seed sweep (**97.8% ± 1.7** exact match, seeds 100%/97.3%/96.0%). The first sweep attempt died with the documented `Impacting Interactivity` Metal abort at iteration 136 — the run was marked `failed` with its traceback and published no pointer, and the sweep was rerun one seed per invocation so only that seed was lost. `docs/reference-run/` now carries the constrained evaluation, the forgetting plot, and `seed_sweep.json`.
