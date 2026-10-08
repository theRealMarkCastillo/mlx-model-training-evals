# Scoping: running this pipeline on real function-calling data

Every honest section of this repo says the same thing: the task is synthetic. Wording, entities and request kinds come from small templates, so the numbers say something about *this* pipeline, not about production traffic. This document scopes what it would take to put a real benchmark behind the same measurements, and what each option costs.

**Status (2026-10-07):** Option A is implemented — the loader (`src/bfcl.py`), the general JSON-Schema grammar (`src/schema_grammar.py`), and the eval harness (`src/bfcl_eval.py`, run with `python main.py bfcl`) — and the pilot has been run on BFCL v3 `simple`. See the *Results* section and `docs/reference-run/bfcl_eval.json`.

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

Run with `python main.py bfcl`; the full per-sample report is `docs/reference-run/bfcl_eval.json`.

| Metric | Base (zero-shot) | Base + JSON grammar |
|---|---:|---:|
| schema-valid | 97.2% [95–98] | 94.4% [92–96] |
| tool accuracy | 98.2% [96–99] | 94.7% [92–96] |
| argument accuracy | **81.5%** [77–85] | **37.0%** [32–42] |

**The conclusion does not generalize as stated — it generalizes as a warning.** The 3B base model already writes the contract on real schemas: a clear per-record prompt gets 97% of outputs schema-valid and 82% of arguments right with no training and no grammar. So grammar-constrained decoding has nothing to fix, and its constraints cost — argument accuracy drops to 37% (McNemar p ≈ 0). The failures are visible and consistent: sign flips (`base: 4` → `-4`) and the model's native `<tool_call>`/single-quote tokens leaking into string values, because the grammar forbids whitespace and forces a compact, enum-pinned path whose tokenization differs from the model's natural one.

Honest limits: this grammar is a minimal one (no whitespace, strings capped at 24 tokens, tool forced to the record's function name), so a production whitespace-allowing grammar would degrade *less* — the 21/395 outputs that hit the 200-token budget are largely this artifact. And `simple` is the easiest category; `multiple` (tool choice) and `irrelevance` (abstention — where the grammar won at every size on the synthetic task) are the natural next steps. The direction is the finding: grammar buys the contract *only when the model lacks it*, and on real schemas the format was never missing.

## Option B — BFCL (or ToolBench) as a full second task

Train, evaluate, ablate and forgetting-check on real schemas: splits, a validator, per-record prompts everywhere, plus a grammar general enough for the whole schema subset.

**Cost:** 1–2 weeks. One trap to avoid: **BFCL is an evaluation benchmark and ships no training split** — training data would have to come from a different source (Gorilla's APIBench/OpenFunctions releases, or ToolBench), which brings its own license and preprocessing. Doing this properly means two datasets, not one.

## Option C — a harder synthetic domain

Multi-tool requests, nested parameters, ambiguous instructions, contradictions — all within the existing machinery.

**Cost:** 2–3 days, and it keeps the controlled experiment (known ground truth, clean axes, no license questions). It improves the *task*; it does not improve the *evidence*, because the generator and the evaluator would still share an author.

---

## Recommendation

**Option A is done** (see *Results*). It settled the direction — on real schemas the 3B base already writes the contract, so grammar-constrained decoding has nothing to fix and its constraints cost — while leaving the synthetic task intact as the controlled experiment.

The next step, if this is worth pursuing, is to close the two obvious gaps before believing the "grammar hurts" number is general:

1. **Allow whitespace in the grammar.** The largest suspected artifact is the compact, whitespace-free path; a production grammar allows `{ "tool": ... }` spacing and would degrade less.
2. **Run `irrelevance` (and `multiple`).** Abstention is where the grammar was the surprise winner at every size on the synthetic task; if it wins there on real data too, the honest story is "grammar buys *refusal*, not *correctness*" — which would be a cleaner, more useful conclusion than a single average.
