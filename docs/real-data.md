# Scoping: running this pipeline on real function-calling data

Every honest section of this repo says the same thing: the task is synthetic. Wording, entities and request kinds come from small templates, so the numbers say something about *this* pipeline, not about production traffic. This document scopes what it would take to put a real benchmark behind the same measurements, and what each option costs.

**Status (2026-10-07):** Option A is implemented — the loader (`src/bfcl.py`), the general JSON-Schema grammar (`src/schema_grammar.py`), and the eval harness (`src/bfcl_eval.py`, run with `python main.py bfcl`) — and the pilot has been run on BFCL v3 `simple`. See the *Results* section and `docs/reference-run/bfcl_simple.json`.

---

## What the real data would buy

The repo currently supports four claims with numbers: format is a decoding problem (`grammar` matches LoRA's schema validity with no training), training buys task decisions, seed spread is real (97.8% ± 1.7), and the task did not damage general ability. A real dataset tests whether those claims survive contact with:

* function schemas nobody wrote for this pipeline (nested objects, unions, many optional fields, non-string types),
* prompts that are not four phrasing families,
* tool sets of 5–20 functions rather than exactly five, chosen per request.

## Candidate: BFCL (Berkeley Function Calling Leaderboard)

Facts verified from the dataset card and repository listing on 2026-10-07:

* Dataset: [`gorilla-llm/Berkeley-Function-Calling-Leaderboard`](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard), **Apache-2.0**.
* Format: **one JSON object per line, one file per category** — questions in `BFCL_v3_<category>.json`, ground truth in `possible_answer/BFCL_v3_<category>.json`. The card explicitly says *not* to use Hugging Face `load_dataset`, which suits this repo: `datasets` is not a dependency here.
* Relevant categories, with sizes from the card: `simple` (400), `multiple` (200), `parallel` (200), `parallel_multiple` (200), `irrelevance` (875 in V2 Live), V3 multi-turn base (200) and augmented (800).
* Scoring: AST match against the possible answers, or executable checks, in the [gorilla repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).

The mapping onto this repo's three axes is unusually clean:

| This repo | BFCL category |
|---|---|
| `test` (tool choice + parameters) | `simple`, `multiple` |
| `challenge_abstain` (refuse unsupported) | `irrelevance` (875 records — 20× our challenge set) |
| `grammar` experiment (format vs task) | any category; the question is whether valid-by-construction decoding still leaves only task error |
| `challenge_defaults` (write omitted defaults) | not directly; BFCL tolerates omitted optional arguments, so this axis would be dropped, not faked |

That last row is the honest cost of moving to real data: some of the axes this repo was designed around do not exist in BFCL, and its scoring tolerances are looser than our exact match.

---

## Option A — eval-only pilot on BFCL v3 Simple (recommended first step)

**Goal:** answer "does the grammar-vs-training conclusion hold on real schemas?" without building a training pipeline for a new task.

Work items (all done):

1. ✅ **Loader** (`src/bfcl.py`): reads the JSONL, builds one record per line with `{system, user, assistant}` in this repo's envelope, taking the function list from the record and the reference call from `possible_answer/`. A 395-record subset is committed under `data/bfcl_simple.jsonl` (5 of 400 skipped: 3 unsupported schema features, 1 schema/value mismatch, 1 function-name mismatch — each counted, never guessed).
2. ✅ **Per-record system prompt** (`build_system_prompt`): the pipeline's one fixed `SYSTEM_PROMPT` becomes a function of the record's function list.
3. ✅ **Generalize the grammar** (`src/schema_grammar.py`): a frame-stack `JsonSchemaGrammar` supporting nested objects, typed arrays, floats/scientific notation, negative numbers, empty arrays, and a `dep` node so a multi-function record's `parameters` schema resolves on the written `tool`. `SchemaGrammarProcessor` subclasses the token-mask machinery; only the mask decision and cache key change.
4. ✅ **Metrics** (`score_bfcl_call`): BFCL-style scoring — function name equality plus "every provided argument's value is in the ground-truth acceptable set, and every required argument is present" — with the report keeping Wilson intervals and a McNemar paired test.
5. ✅ **Run it** (`python main.py bfcl`): base vs grammar on Simple; `multiple` and `irrelevance` are the next categories.

**Cost:** roughly 2–4 days of work, no training compute (all three variants are prompting-only). Inference is cheap: 400 records × 3 variants ≈ 1,200 generations, under an hour on the 3B preset.

**Risks and honest limits:** 400 records per category means ±5-point intervals; skipping unsupported schema features will bias the surviving subset and must be reported; our envelope differs from BFCL's canonical `[{"name": ..., "arguments": {...}}]`, so results are *internally* comparable to the synthetic task and only qualitatively comparable to the leaderboard.

## Results (BFCL v3 simple, 395 records, 3B base)

Run with `python main.py bfcl`; the full per-sample report is `docs/reference-run/bfcl_simple.json`.

| Metric | Base (zero-shot) | Base + JSON grammar |
|---|---:|---:|
| schema-valid | 97.2% [95–98] | 93.9% [91–96] |
| tool accuracy | 98.2% [96–99] | 94.2% [91–96] |
| argument accuracy | **81.5%** [77–85] | **37.7%** [33–43] |

**The conclusion does not generalize as stated — it generalizes as a warning.** The 3B base model already writes the contract on real schemas: a clear per-record prompt gets 97% of outputs schema-valid and 82% of arguments right with no training and no grammar. So grammar-constrained decoding has nothing to fix, and its constraints cost — argument accuracy drops to 37% (McNemar p ≈ 0). The failures are visible and consistent: sign flips (`base: 4` → `-4`) and the model's native `<tool_call>`/single-quote tokens leaking into string values.

### Follow-up: whitespace and tool choice

The natural suspicion was that the grammar's whitespace-free, compact path causes the degradation, so both documented follow-ups were done:

* **Whitespace does not help.** Allowing structural whitespace (and capping a whitespace run, like strings and arrays are already capped) leaves argument accuracy at 37.7% and schema validity at 94%. The degradation is not a tokenization artifact — it is the model fighting an envelope its native format does not produce, which a structural constraint cannot fix. (Along the way this surfaced and fixed two real bugs: the whitespace mask matched Unicode spaces via `str.strip()` and desynchronized from the grammar, and an uncapped whitespace run let a model that had leaked prose loop to the token budget.)
* **Tool choice survives the grammar, argument filling does not.** On `multiple` (195 records, 2–4 candidate functions, exercising the dependent-`parameters` node): tool accuracy 94.9% → 90.8% (p = 0.10), argument accuracy **75.4% → 35.9%** (p ≈ 0). The grammar preserves *which* function to call and still destroys *what* to pass it.

### The prompt is doing the format work

Where does the base model's 82% come from? Ablating the prompt — replacing the envelope example, type hints and "respond with only JSON" with just the function *names* — answers it:

| Argument accuracy | Full prompt | Minimal prompt (names only) |
|---|---:|---:|
| base (no grammar) | 81.5% | **0.0%** |
| base + grammar | 37.7% | **22.0%** |

Without the prompt's format guidance the base model emits **zero** structured calls — it explains the answer in prose, names the function in backticks, but never produces the envelope. The prompt and the grammar are two ways to supply the same *format contract*, and the prompt is by far the better carrier of it: it shapes the model's *intent* (what to output), so it gets the structure *and* the values right (82%); the grammar only constrains *structure* (what is allowed), so it forces the envelope while the model fills it with whatever its unshaped intent produces (22–37%). Grammar-constrained decoding is a weaker, narrower version of the prompt, not a substitute for it.

### Training does not beat prompting either

The repo's core claim is that LoRA fine-tuning is what finally teaches the contract, so that cell was measured too. A LoRA was trained on a 270-record split of `simple` itself (in-distribution — BFCL ships no training data), evaluated on the held-out 80 records:

| Metric (test split, n = 80) | Base | Grammar | LoRA |
|---|---:|---:|---:|
| schema-valid | 96.2% | 95.0% | 97.5% |
| tool accuracy | 97.5% | 95.0% | 100.0% |
| **argument accuracy** | **75.0%** | 33.8% | **72.5%** |

The adapter ties the base model (72.5% vs 75.0%, inside the ±10% interval) rather than beating it. That is the opposite of the synthetic task, where LoRA went 0% → 100%: there the base model could not follow the prompt, so training *was* supplying the format. On real schemas the prompt already supplies it, and in-distribution fine-tuning adds nothing. "Training buys the decisions" was really "training buys the *format* when the prompt does not deliver it" — a property of the weak base model, not of fine-tuning in general. (Training details: `config/bfcl.yaml`, 200 iters, val loss 1.22 → 0.067; the split and report are `scripts/split_bfcl.py` and `docs/reference-run/bfcl_split_test.json`.)

### Scale is the one lever that moves the number

The conclusion named a larger base model as the way past ~75%, so it was measured: the 14B base reaches **88.9%** argument accuracy on the same 395 records (100% tool, 99.7% schema-valid), up from 81.5% at 3B — while the grammar still costs it (71.1%), so the grammar's harm is not a small-model artifact. `python main.py bfcl --preset 14b`; report at `docs/reference-run/bfcl_simple_14b.json`.

### Abstention (BFCL irrelevance, 237 records)

The synthetic task's cleanest grammar win was abstention (0% → 100%), so it was re-run on real data. BFCL `irrelevance` ships no answer file — the one provided function is a deliberate mismatch, and the correct behaviour is to refuse — so the metric is the hallucinated-call rate (`python main.py bfcl --irrelevance`):

| Metric | Base (zero-shot) | Base + JSON grammar |
|---|---:|---:|
| hallucinated-call rate | 37.6% [32–44] | 45.1% [39–52] |

The grammar does **not** help. The base model already refuses 62% of the time (the synthetic base refused ~0% because it could not format at all), and forcing a valid envelope — with `no_action` offered as an alternative — leaves the model calling the irrelevant function slightly more often. Both synthetic grammar wins, format and abstention, were the same artifact: they fixed a base model that could not follow a prompt. On real data the prompt already does the job.

## Option B — BFCL (or ToolBench) as a full second task

Train, evaluate, ablate and forgetting-check on real schemas: splits, a validator, per-record prompts everywhere, plus a grammar general enough for the whole schema subset.

**Cost:** 1–2 weeks. One trap to avoid: **BFCL is an evaluation benchmark and ships no training split** — training data would have to come from a different source (Gorilla's APIBench/OpenFunctions releases, or ToolBench), which brings its own license and preprocessing. Doing this properly means two datasets, not one.

## Option C — a harder synthetic domain

Multi-tool requests, nested parameters, ambiguous instructions, contradictions — all within the existing machinery.

**Cost:** 2–3 days, and it keeps the controlled experiment (known ground truth, clean axes, no license questions). It improves the *task*; it does not improve the *evidence*, because the generator and the evaluator would still share an author.

---

## Recommendation

**Option A is done, including the prompt ablation, an in-distribution training run, and a scale check** (see *Results*). The full picture on real schemas: the 3B base writes the contract *because the prompt supplies it* (82% args with the full prompt, 0% with names alone); grammar-constrained decoding is a weaker carrier of the same contract (22–37%); abstention gains nothing; and an in-distribution LoRA ties the base model (72.5% vs 75.0%) rather than beating it. The synthetic task stays intact as the controlled experiment it was designed to be.

The honest, useful conclusion is now threefold: **format is supplied by the prompt; grammar-constrained decoding is a weaker substitute for it, not an addition; and fine-tuning on real schemas buys nothing the prompt does not already deliver.** The synthetic task's "training buys the decisions" was really "training buys the format when a weak base model cannot follow a prompt" — a property of that base model, not of the methods.

The one lever that *does* move real-schema accuracy is scale: the 14B base reaches 88.9% (vs 81.5% at 3B), and the grammar still costs it there (71.1%). If the goal is to push argument accuracy further, a larger base model — not grammar and not in-distribution LoRA — is what has been shown to work; a stronger instruction-tuned model or a genuinely harder generalization target are the other, unmeasured options.
