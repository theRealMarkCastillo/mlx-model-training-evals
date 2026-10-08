# Scoping: running this pipeline on real function-calling data

Every honest section of this repo says the same thing: the task is synthetic. Wording, entities and request kinds come from small templates, so the numbers say something about *this* pipeline, not about production traffic. This document scopes what it would take to put a real benchmark behind the same measurements, and what each option costs.

Nothing here is implemented yet. It exists so the decision is concrete rather than aspirational.

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

Work items, in order:

1. **Loader** (`src/bfcl.py`): read the JSONL, build one record per line with `{system, user, assistant}` in this repo's envelope (`{"tool": ..., "parameters": ...}`), taking the function list from the record and the reference call from `possible_answer/`. Cache a small subset in `data/` so runs stay reproducible offline, exactly like the synthetic sets.
2. **Per-record system prompt.** This is the real structural change: the pipeline assumes one fixed `SYSTEM_PROMPT`, while BFCL gives each record its own function documents. The prompt generator in `src/schema.py` becomes a function of the record's schemas; `show-mask`, training and the few-shot builder all need to read it from the record instead of importing a constant.
3. **Generalize the grammar** (`src/json_grammar.py`): the token-masking machinery in `src/constrained.py` is already schema-agnostic — it asks the grammar for allowed keys, value kinds and completion. What is task-specific is `ToolCallGrammar`: fixed `TOP_KEYS`, `depth <= 2`, `outer_keys` as a single slot. A general version needs a frame stack for nesting, a per-record function table, and a JSON-Schema subset: `object`/`array`/`string`/`integer`/`number`/`boolean`, `enum`, `required`, `additionalProperties: false`. Records using `$ref`, `anyOf`/`oneOf` or exotic types get skipped and counted.
4. **Metrics.** Port BFCL's AST check faithfully or, to start, implement the documented approximation: function name equality plus argument-dict equality with BFCL's tolerance for omitted optional arguments. Either way the report keeps this repo's shape (Wilson intervals, per-category breakdowns, failure taxonomy) so numbers stay comparable to the synthetic results.
5. **Run it:** base / few-shot / grammar on Simple, then `multiple` (tool choice) and `irrelevance` (abstention).

**Cost:** roughly 2–4 days of work, no training compute (all three variants are prompting-only). Inference is cheap: 400 records × 3 variants ≈ 1,200 generations, under an hour on the 3B preset.

**Risks and honest limits:** 400 records per category means ±5-point intervals; skipping unsupported schema features will bias the surviving subset and must be reported; our envelope differs from BFCL's canonical `[{"name": ..., "arguments": {...}}]`, so results are *internally* comparable to the synthetic task and only qualitatively comparable to the leaderboard.

## Option B — BFCL (or ToolBench) as a full second task

Train, evaluate, ablate and forgetting-check on real schemas: splits, a validator, per-record prompts everywhere, plus a grammar general enough for the whole schema subset.

**Cost:** 1–2 weeks. One trap to avoid: **BFCL is an evaluation benchmark and ships no training split** — training data would have to come from a different source (Gorilla's APIBench/OpenFunctions releases, or ToolBench), which brings its own license and preprocessing. Doing this properly means two datasets, not one.

## Option C — a harder synthetic domain

Multi-tool requests, nested parameters, ambiguous instructions, contradictions — all within the existing machinery.

**Cost:** 2–3 days, and it keeps the controlled experiment (known ground truth, clean axes, no license questions). It improves the *task*; it does not improve the *evidence*, because the generator and the evaluator would still share an author.

---

## Recommendation

Do **Option A** first. It is the cheapest way to falsify or confirm the repo's headline claim — "decoding buys the contract, training buys the decisions" — on schemas this repo did not write, and it leaves the synthetic task intact as the controlled experiment it was designed to be.

Decisions needed before starting:

1. **Option A, B or C** (or A now, B later).
2. **Envelope:** keep `{"tool": ..., "parameters": ...}` for comparability with the synthetic results (recommended), or adopt BFCL's list-of-calls shape for leaderboard-adjacent numbers.
3. **Scope:** the eval-only pilot (recommended), or commit to a training run on real schemas too.
