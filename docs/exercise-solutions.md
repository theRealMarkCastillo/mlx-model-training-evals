# Exercise solutions

Solutions and expected outcomes for the five notebook exercises (§10). These are *guides*, not transcripts: the numbers depend on your Mac, seed, and MLX version, and the honest way to report any of them is with the confidence intervals the pipeline already prints.

Run the notebook first (`uv run python main.py notebook`); each exercise below names the file to change and what to look for.

---

## 1. Remove the loss mask

**Change.** `src/config.py` pins `mask_prompt: Literal[True]`. Widen it to `bool` (and, if you want to be strict, make the trainer refuse `mask_prompt: false` for the *evaluation* path), then run:

```bash
uv run python main.py train --preset 3b --iters 50        # masked baseline
# edit config/base.yaml to mask_prompt: false, then:
uv run python main.py train --preset 3b --iters 50
uv run python main.py eval --preset 3b --variants lora
```

**What happens.** With the mask off, every token in the record is scored: the ~600-token system prompt (identical in all 250 training records) plus the ~20-token answer. Two effects show up immediately:

1. The training loss becomes much *lower* and much flatter, because most of the loss now comes from boilerplate the model memorizes in a few steps. A low loss here does **not** mean a better model — it means the average is dominated by easy tokens.
2. The evaluation's "assistant loss" still masks the prompt (`src/evaluate.py::compute_perplexity` always uses `mask_prompt=True`), so training loss and evaluation loss are no longer the same quantity. That mismatch is the lesson: the mask is what keeps the two comparable.

Task accuracy at equal iterations is usually worse without masking, because the answer tokens receive a smaller share of the gradient.

**Report it as:** same iters, both settings, `test` exact match with intervals, plus the two training-loss curves side by side. Do not compare the two losses numerically — they score different token sets.

---

## 2. Starve the data

**Change.** In `src/generate_data.py::build_splits`, drop the training count from 250 to 50 (keep 10 per tool so the split stays balanced). Then:

```bash
uv run python main.py prepare
uv run python main.py train --preset 3b
uv run python main.py eval --preset 3b --challenge
```

**What to expect.** The three challenge sets fail in a predictable order of difficulty:

* `challenge_defaults` degrades first. Writing omitted optionals with their documented defaults is a *format habit*; few examples means the habit is weak, and the model starts leaving fields out (the `omitted_default` failure category grows).
* `test` (unseen wording) degrades next: the wording families in training are the only evidence for how requests can be phrased.
* `challenge_abstain` and `challenge_entities` are the most interesting to compare. Abstention needs enough `no_action` examples to learn *when* to stop acting; entity transfer needs enough examples to learn *copy the identifier from the request*, which the reference run shows is already the weak spot at 250 records.

**Falsification:** if `test` holds up while `challenge_defaults` collapses, that is evidence the model learned the default-writing habit as a schema regularity rather than as wording-following — a result worth reporting, not a bug.

---

## 3. Remove abstention training

**Change.** In `build_splits`, build the `train` split with `tools=ACTION_TOOLS` (the four action tools, no `no_action`). Keep the evaluation sets unchanged, so `no_action` still appears in `test` and `challenge_abstain`.

**What to expect.** The model never *chooses* to abstain: with no `no_action` examples it has no representation of "stop", so unsupported requests get force-fitted into the nearest action tool. That usually means hallucinated required parameters: a version for `deploy_service` that was never in the request, or a region for `restart_pod`. This is the cleanest demonstration in the repo that **abstention is a behaviour you train, not a behaviour you get for free**, and it shows up in the per-tool table as `no_action` at or near 0% while the four action tools look healthy.

**Extension worth running:** keep `no_action` but remove only the *missing-required-parameter* half (`no_action(..., missing=False)`), so the model must abstain on request kinds it has seen but never on incomplete requests. The `challenge_defaults`-style misses will move into the `parameters` category instead of `tool`.

---

## 4. Find the overfitting point

**Change.** Push the iteration sweep far past the reference run:

```bash
uv run python main.py ablate iters 200 400 600 800 --samples 75
```

**What to expect.** On this narrow task the model mostly memorizes the *format* early (schema validity reaches 100% by iteration 100 in the reference run) and then continues to sharpen the mapping. Watch the two curves in `ablation.png` separately:

* **Best validation loss** typically bottoms out well before the last point — the reference run's minimum is at iteration 149 of 200.
* **Exact match** tends to plateau rather than fall, because the holdout tests new *wording*, not new *tasks*; once the format is learned, extra iterations rarely hurt the score even as validation loss creeps up.

The lesson is that "overfitting" on a narrow, templated task looks like a rising validation loss with flat task accuracy, not like a collapsing score. With `--samples 75` the intervals are ±10 points or so: only call a difference real if the intervals separate *and* a second seed reproduces it (`ablate seed ...`).

**Falsification:** if exact match clearly drops while validation loss rises across several seeds, that is real overfitting on this task — report it, and consider evaluating the checkpoint near the best validation iteration instead of the final one.

---

## 5. Scale up to 14B

**Change.** The cheap half of this question needs no training at all: the baselines are just prompting.

```bash
uv run python main.py train --preset 14b --iters 10    # memory check first: read the peak line
uv run python main.py eval --preset 14b --variants base fewshot    # no adapter needed
uv run python main.py train --preset 14b               # then the LoRA run, if memory allows
uv run python main.py eval --preset 14b --challenge
```

**What to expect.** A larger base model usually improves the *few-shot* baseline more than it improves zero-shot: more capacity means in-context learning works better, while the zero-shot failure mode (no example of the envelope) stays a format failure at any size. The interesting comparison is therefore:

| Question | Comparison |
|---|---|
| Does the bigger model fix zero-shot format? | 3B base vs 14B base on `test` |
| Does it close the few-shot gap? | 3B few-shot vs 14B few-shot |
| Is training still worth it? | 14B few-shot vs 14B LoRA |

Equal `iters` across presets means different numbers of examples seen (batch sizes differ), so compare *scores*, not training curves.

**Memory note:** the 14B preset uses batch 1 with gradient checkpointing. If the 10-iteration check reports a peak near your machine's limit, reduce `--num-layers` or stay at 7B; the memory line is the whole point of the short run.

---

## 6. Separate formatting from task skill

**Run.** `uv run python main.py eval --preset 3b --constrained` (no training needed for the `grammar` variant; it decodes the base model under a JSON grammar derived from the tool schemas).

**What to expect.** On the 3B reference preset the split is sharp:

| Metric (holdout, n=75) | base | base + grammar | few-shot | LoRA |
|---|---:|---:|---:|---:|
| schema valid | 0% | **100%** | 85% | 100% |
| tool accuracy | 0% | 89% | 92% | 100% |
| exact match | 0% | 43% | 68% | 100% |

The grammar variant's schema validity is *identical* to LoRA's — 75 of 75 samples valid for both, no discordant pair — so the entire difference between 43% and 100% exact match is task skill: wrong tool choice, omitted defaults, wrong values. Two further details are worth noticing: on `challenge_abstain` the grammar variant scores 100%, better than LoRA's 95%, so abstention on unseen request kinds was never the hard part once the envelope is guaranteed; and few-shot prompting lands in between while paying ~985 prompt tokens per request against ~666.

**Report it as:** schema validity (format) and exact match (task) side by side, with the paired schema-validity test. "Decoding fixes the contract; training fixes the decisions" is the sentence the numbers support — anything stronger is not in the data.

---

## 7. How much of the headline is luck?

**Run.** One seed per invocation, then merge:

```bash
uv run python main.py ablate seed 42 --iters 200
uv run python main.py ablate seed 43 --iters 200
uv run python main.py ablate seed 44 --iters 200
uv run python scripts/merge_seed_sweep.py
```

**Why one per invocation.** A full sweep is three 20-minute GPU jobs, and macOS can abort a long one with `Impacting Interactivity` (see the README's troubleshooting). Running one seed per process means an abort costs only that seed: its run is marked `failed`, no pointer is published, and rerunning it is safe. The merge step deduplicates retries by seed.

**What to expect.** Seed 42 alone gives 100% exact match on the holdout — which, with 75 samples, is compatible with anything from about 95% to 100%. The sweep's mean ± spread is the number to quote, and the interesting quantity is the spread, not the mean: if the seeds disagree by more than the Wilson interval suggests, the task is seed-sensitive and every single-run claim in the repo needs that asterisk.

**Report it as:** per-seed exact match with intervals, plus mean ± spread, and say explicitly which seeds contributed. Do not report the best seed.

---

## 8. What did the task cost?

**Run.** `uv run python main.py forgetting --preset 3b` (scoring only, no training; a few minutes on the 3B preset).

**What to expect.** On the 3B reference preset the adapter's general-capability loss is *lower* than the base model's:

| Model | Mean assistant loss | Records worse / better | Sign-test p |
|---|---:|---:|---:|
| Base | 1.823 | — | — |
| LoRA | 1.215 | 7 / 17 | 0.064 |

No detectable forgetting, and the (non-significant) direction favours the adapter — plausible for LoRA, where the base weights never move and practice at strict structured output also sharpens ordinary instruction-following. The failure mode to look for instead is *drift*: an answer that turns into a tool-call envelope. `forgetting.png` shades each record red when the loss rose, so a few bad records are visible even when the mean improves; open the ones that got worse and read what the model produced.

**Report it as:** mean loss for both models, the direction counts, the sign-test p, and the explicit limitation — 24 synthetic records scored by loss can show that general ability was **not obviously destroyed**, never that it was preserved. If you want the stronger claim, you need a real benchmark (for example `mlx_lm.evaluate` on a small lm-eval task) and enough samples to power it.

---

## 9. Sample instead of decoding greedily

**Run.** `uv run python main.py eval --preset 3b --temperature 0.7 --seed 7` (add `--samples 30` while exploring).

**What to expect.** Greedy decoding hides the model's uncertainty: it always emits the single most likely token, so a 51%/49% fork is scored as a certainty. Sampling exposes the cost — format validity (`pure_json_rate`, `schema_valid_rate`) usually falls and varies between runs, and the exact-match rate moves in the same direction. Because the seed is recorded and re-applied per variant, a seeded run is reproducible; two different seeds are two different draws, which is itself the lesson. Pair it with `--constrained` to see that the grammar removes the format variance while leaving the task variance.

**Report it as:** greedy versus sampled with intervals, the seed used, and — if you run `--repeats k` — the pass@k table, which prices the retry policy: sampling lowers the single-draw score, and pass@k shows how much of it a deployment recovers by asking again. The `records no draw got` column is the honest ceiling on retrying. Running `--repeats` at temperature 0 instead measures Metal run-to-run flakiness: if two greedy draws of the same record disagree, that is the noise floor under every number in this repo.
