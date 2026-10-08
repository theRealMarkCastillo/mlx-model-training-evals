"""Build tutorial.ipynb from stable cells that call the same modules as the CLI."""
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


markdown('intro', '''# Fine-tuning a small LLM for tool calling, and measuring it honestly

You will teach Qwen2.5 to turn requests like *"Ship v2.3.1 of auth-api into prod with 3 instances"* into strict JSON tool calls, on your Mac, with Apple MLX. The model is not the main subject. The main subject is the questions you should ask about any fine-tune:

1. **What exactly is the model trained on?** (§3: the loss mask)
2. **Is fine-tuning even necessary?** (§4: zero-shot and few-shot baselines)
3. **How big is the change?** (§5: LoRA parameter accounting)
4. **Did it learn, or memorize?** (§5: loss curves; §6: challenge sets)
5. **Is the improvement real, or noise?** (§6–7: confidence intervals, paired tests)
6. **What does it still get wrong, and why?** (§7: error analysis)

Each step calls the same functions as `main.py`, and every run writes a self-describing directory under `artifacts/qwen2.5-<size>/runs/`.

**Time and memory (3B preset, M-series Mac):** training ≈ 20 min, full evaluation with challenge sets ≈ 20 min. Larger presets need much more memory. Run `uv run python main.py train --preset 14b --iters 10` first and read the peak-memory line.''')

code('setup', '''import json
from pathlib import Path

import mlx.core as mx
from IPython.display import HTML, Image, Markdown, display

PRESET = "3b"  # 3b, 7b, 14b, 32b, or 72b
print(f"MLX {mx.__version__}; Metal available: {mx.metal.is_available()}")''')

markdown('schema-notes', '''## 1. The task is a contract

`src/schema.py` defines five tools with Pydantic models: four actions plus `no_action`, which the model should choose when no tool fits or a required value is missing. The system prompt is generated from those models, so the instructions and the validator cannot drift apart.

Scoring is strict on purpose. **Exact match** requires a bare JSON object, the correct tool, and every parameter, including defaults, with exactly the right value and type. Weaker views are kept separately, so you can see *how* an output fails:

| Check | Fails when |
|---|---|
| valid JSON | no JSON object can be recovered at all |
| pure JSON | the object is wrapped in prose or markdown fences |
| schema valid | wrong type (`"3"` for 3), unknown field, missing required field |
| normalized match | values are right once omitted defaults are filled in |''')

code('schema-example', '''from src.schema import SYSTEM_PROMPT, parse_and_validate

print(SYSTEM_PROMPT[:600], "...\\n")
examples = {
    "exact": '{"tool":"deploy_service","parameters":{"service":"api","version":"v1","environment":"production","replicas":3,"notify_channels":[]}}',
    "wrapped in prose": 'Sure! {"tool":"rollback_deployment","parameters":{"deployment_id":"dep-1","target_tag":"v1.0.0","drain_traffic":true}}',
    "string instead of int": '{"tool":"deploy_service","parameters":{"service":"api","version":"v1","environment":"production","replicas":"3"}}',
    "abstain": '{"tool":"no_action","parameters":{"reason":"missing_required_parameter"}}',
}
for label, output in examples.items():
    r = parse_and_validate(output)
    print(f"{label:22} pure={r['is_pure_json']!s:5} schema_valid={r['is_schema_valid']!s:5} {r['error'] or ''}"[:160])''')

markdown('data-notes', '''## 2. Data designed to test one thing at a time

The generator (`src/generate_data.py`) is synthetic and seeded, so every number below is reproducible. Two kinds of generalization are controlled separately:

* **Wording.** Each tool has four phrasing *families*. Train sees families 0–1, validation 2, test 3. A good test score means the model handles **new phrasings**.
* **Entities.** Service, pod, region, and cluster names come from one pool for the standard splits. `challenge_entities` uses names that never appear in training.

About 30% of requests leave optional values out (e.g. no replica count), so the model must write the documented default. Three challenge sets each isolate one difficulty:

| Set | Wording | Entities | Tests |
|---|---|---|---|
| `test` | unseen (family 3) | seen | standard holdout |
| `challenge_entities` | seen (families 0–1) | **unseen** | names and values never seen in training |
| `challenge_defaults` | unseen | seen | **every** optional value omitted |
| `challenge_abstain` | mixed | seen | **kinds** of unsupported request never seen in training |

This is still a synthetic task. High scores here say nothing about messy real traffic.''')

code('prepare-data', '''from collections import Counter

from src.dataset import DATA_DIR, load_samples, validate_splits
from src.generate_data import main as generate

generate()
validate_splits()
for name in ("train", "valid", "test", "challenge_entities", "challenge_defaults", "challenge_abstain"):
    records = load_samples(DATA_DIR / f"{name}.jsonl")
    tools = Counter(r["expected"]["tool"] for r in records)
    omitted = sum(bool(r["meta"]["omitted"]) for r in records)
    print(f"{name:20} n={len(records):3}  with omitted defaults={omitted:3}  {dict(tools)}")
example = load_samples(DATA_DIR / "challenge_defaults.jsonl")[0]
print("\\nExample:", example["prompt"])
print("Target: ", json.dumps(example["expected"]))''')

markdown('mask-notes', '''## 3. What the model is actually trained on

Each record becomes one token sequence: system prompt, user request, assistant answer. With `mask_prompt: true`, the loss counts **only the answer tokens**. Below, highlighted tokens are scored; grey tokens are context the model reads but is never trained to reproduce.

Without the mask, most of the gradient would go to memorizing the ~600-token system prompt, which is identical in every example. The evaluator applies this same mask when it reports "assistant loss", so training and evaluation losses are comparable.''')

code('mask', '''from transformers import AutoTokenizer

from src.explain import loss_mask_tokens, render_mask_html
from src.models import PRESETS

tokenizer = AutoTokenizer.from_pretrained(PRESETS[PRESET].model)
record = load_samples(DATA_DIR / "train.jsonl")[0]
display(HTML(render_mask_html(loss_mask_tokens(tokenizer, record))))''')

markdown('baseline-notes', '''## 4. Before training: is fine-tuning necessary?

Always measure the cheap alternatives first:

* **Zero-shot:** the base model with only the system prompt.
* **Few-shot:** the same, plus five worked examples (one per tool) from the *training* split, inserted as earlier chat turns.

If few-shot prompting already gets close to what you need, it may beat fine-tuning: nothing to train, store, or redeploy. The cost is a longer prompt on every request (compare `prompt_tokens` later). The cell below uses a 20-sample subset to stay quick; §6 repeats this on the full sets.''')

code('baseline', '''from src.evaluate import run_comprehensive_evaluation
from src.metrics import failure_examples, format_rate

baseline = run_comprehensive_evaluation(preset=PRESET, variants=("base", "fewshot"), num_eval_samples=20)
for variant in ("base", "fewshot"):
    m = baseline["datasets"]["test"][variant]
    print(f"{variant:8} exact match {format_rate(m['exact_match_rate'], m['intervals']['exact_match_rate'])}   errors: {m['error_categories']}")
print("\\nA typical zero-shot failure:")
for e in failure_examples(baseline["datasets"]["test"]["base"]["sample_results"], limit=2):
    print(f"[{e['category']}] {e['prompt']}\\n   -> {e['raw_output'][:200]!r}")''')

markdown('training-notes', '''## 5. LoRA training

LoRA freezes the base weights `W` and learns a low-rank update for selected projections:

```
W_effective = W + scale · (B @ A)        A: r × d_in,  B: d_out × r
```

For a 2048 × 2048 attention projection at rank `r = 8`, that is 8 · (2048 + 2048) = 32,768 trainable numbers instead of 4.2 million. `B` starts at zero, so training starts from the unchanged base model. MLX applies `scale` directly; it is **not** `alpha / r` as in some other libraries.

Settings come from `config/base.yaml`, plus four per-preset overrides (`model`, `batch_size`, `num_layers`, `grad_checkpoint`). Each call creates a new run directory. `artifacts/qwen2.5-<size>/adapters/latest.json` is updated only after the run succeeds.

**Reading the loss curve:**
* Both curves fall and flatten: learning, then saturating.
* Train keeps falling while validation turns upward: **overfitting**. The model is memorizing training wording. The best validation iteration is marked.
* Train loss is noisy because each point averages only the last few batches.''')

code('train', '''from src.train import run_training

ITERS = None  # None = base config (200). Try 25 first to measure time and memory.
training = run_training(preset=PRESET, iters_override=ITERS)
display(Image(filename=str(training.run_dir / "loss_curve.png")))
p = training.parameters
print(f"Trainable: {p['adapter_parameters']:,} of {p['base_parameters']:,} ({p['adapter_percent']:.3f}%), "
      f"{p['adapter_megabytes_fp16']:.1f} MB")
for projection, entry in p["adapted_projections"].items():
    print(f"  {projection:10} rank {entry['rank']} in {entry['layers']} layers: {entry['parameters']:,} parameters")
print(json.dumps(training.losses, indent=2))''')

markdown('toy-notes', '''## 5b. What `train_model` actually does

`run_training` delegates its loop to MLX-LM's `train_model`, so the mechanics above stay invisible. `src/mini_train.py` rebuilds the same loop by hand on a toy task that trains in about a second on any Mac, with no downloads:

* A fixed table maps random 4-token "requests" to 2-token "answers". Training and validation contexts are **disjoint**, so the only thing the model can do with the training set is memorize it.
* The forward pass computes logits *only* for answer positions — that construction is exactly what `mask_prompt: true` guarantees for the real trainer.
* LoRA enters as `W2 + scale · (B @ A)` on the output head, with `B = 0` at step 0, so the first forward pass equals the frozen base model.
* The gradient comes from `mx.value_and_grad`, and the update is a hand-written AdamW step (`src/mini_train.py:adamw_step`).

Watch the validation loss: it cannot improve, because the validation answers do not exist in training. Once the adapter memorizes, val loss climbs *above* the ln(48) guessing line — the model becomes confidently wrong on unseen contexts. That is the overfitting signature you are looking for on real loss curves, in miniature.''')

code('toy', '''from src.mini_train import run_toy_training

toy = run_toy_training()   # 300 iterations, LoRA rank 4, ~1 second
display(Image(filename=str(toy.run_dir / "loss_curve.png")))
print(json.dumps(toy.summary(), indent=2))
# Compare with full fine-tuning (every parameter, no LoRA):
# toy = run_toy_training(mode="full")''')

markdown('eval-notes', '''## 6. Full evaluation: baselines vs LoRA, holdout and challenge sets

Every variant sees the same records with greedy decoding (temperature 0), so differences come from the model, not the dice. Read the error bars: they are 95% Wilson intervals. With n = 75, an exact-match rate of 90% means "probably between about 82% and 95%".

**Loss and accuracy measure different things.** Assistant loss scores the reference answer token by token, rewarding the model for putting probability on it. Exact match scores the single greedy output, all or nothing. A model can lower its loss a lot and still produce the same wrong field, or improve accuracy with little change in loss. Use loss to watch training; use task metrics to decide.

**One more baseline worth running:** `main.py eval --constrained` adds a `grammar` variant that decodes the base model under a JSON grammar derived from the tool schemas, so invalid JSON is impossible. It answers "was this a formatting problem or a task problem?" without spending a single training step — and on the small base model it turns 0% schema validity into 100%, leaving the wrong-tool and wrong-value errors behind for training to fix.

**And the cost side:** every metric so far measures the task. `main.py forgetting` measures what the task cost — it scores the base model and the adapter on 24 ordinary requests (arithmetic, facts, rewriting) with a plain assistant system prompt, and compares assistant loss per record with an exact sign test. An adapter that answers arithmetic with JSON has overfit the contract; the per-record loss plot shows exactly which requests got worse.''')

code('evaluate', '''report = run_comprehensive_evaluation(preset=PRESET, challenge=True)
run_dir = Path(report["run_dir"])
display(Image(filename=str(run_dir / "eval_comparison.png")))
display(Image(filename=str(run_dir / "challenge_comparison.png")))''')

markdown('paired-notes', '''### Is LoRA really better? Use a paired test

Two variants answered the *same* questions, so compare them question by question. Samples both get right, or both get wrong, carry no information about which is better; only the **discordant** samples do. McNemar's exact test asks how surprising the split of discordant samples would be if the two variants were equally good. Overlapping confidence bars do not settle this; the paired test is more sensitive.''')

code('paired', '''for name, comparisons in report["paired"].items():
    for other, c in comparisons.items():
        print(f"{name:20} LoRA vs {other:8}: only LoRA right {c['only_a']:3}, only {other} right {c['only_b']:3}, p = {c['p_value']:.3g}")''')

markdown('errors-notes', '''## 7. Error analysis: where does it still fail?

Aggregate scores hide the story. Look at per-field accuracy (which parameter is wrong?), slices (does omitting a value hurt?), and actual outputs. Typical findings for this task:

* The base model fails on **format** (prose, markdown fences) and on **omitted defaults**.
* LoRA fixes format almost completely; remaining errors concentrate in specific fields or in the challenge sets.
* Abstaining on *new kinds* of unsupported requests is harder than abstaining on the kinds seen in training.

When an aggregate is not enough, `main.py inspect` opens up a single decision: it greedy-decodes one record and prints every token with its probability, the runner-up's probability, and the divergence point — the first token where the output leaves the reference answer, with the reference token's rank and probability at that step.

```bash
uv run python main.py inspect --variant base --split test --index 0   # no adapter needed
uv run python main.py inspect --variant lora --split test --index 0   # the trained adapter
```

That is how "the base model scores 0%" becomes "at step 0 it put 69% on `restart` and ~0% on the reference `{\\"` — the envelope is a format decision, not a capability one".''')

code('errors', '''from src.metrics import CATEGORY_HELP

for variant in report["variants"]:
    m = report["datasets"]["test"][variant]
    print(f"\\n== {variant}: slices", {k: f"{v['exact_match_rate']:.0%} (n={v['n']})" for k, v in m["slices"].items()})
    for tool, fields in m["per_field"].items():
        weak = {f: f"{acc:.0%}" for f, acc in fields.items() if acc < 1}
        if weak:
            print(f"   {tool:20} fields below 100%: {weak}")

for dataset in report["datasets"]:
    examples = failure_examples(report["datasets"][dataset]["lora"]["sample_results"], limit=3)
    if examples:
        display(Markdown(f"**LoRA failures on `{dataset}`**"))
    for e in examples:
        print(f"[{e['category']}: {CATEGORY_HELP[e['category']]}]\\n  request:  {e['prompt']}\\n  expected: {json.dumps(e['expected'])}\\n  output:   {e['raw_output']!r}\\n  wrong fields: {e['wrong_fields']}")''')

markdown('ablation-notes', '''## 8. Ablation: one knob at a time (optional, slow)

Intuition for hyperparameters comes from changing one while holding everything else fixed. `run_ablation` trains and evaluates one adapter per value. Good first experiments:

* `iters`: 25, 50, 100, 200. How quickly does task accuracy saturate, and does validation loss start to climb?
* `rank`: 2, 8, 32. On a narrow task, does a bigger adapter help at all?
* `num_layers`: 2, 8, 16. How much of the network needs adapting?

Each point is a full training run plus a LoRA-only evaluation (about 3–8 min each at 3B). Ablation adapters stay inside the ablation's own directory and never replace your latest adapter.''')

code('ablation', '''RUN_ABLATION = False
if RUN_ABLATION:
    from src.ablation import run_ablation
    ablation = run_ablation("iters", [25, 50, 100, 200], preset=PRESET)
    display(Image(filename=str(Path(ablation["run_dir"]) / "ablation.png")))''')

markdown('benchmark-notes', '''## 9. Speed, fusion, and serving

The benchmark measures time to first token, prefill and decode throughput, latency, and peak Metal memory for the base model and the base model with the adapter applied dynamically. **Fusion** merges `scale · B @ A` into the weights, which removes the adapter's extra matmuls. With a quantized base, merging re-quantizes the weights and can change outputs slightly, so `--fuse` also re-evaluates quality.''')

code('benchmark', '''from src.benchmark import run_benchmark_suite

bench = run_benchmark_suite(preset=PRESET)
print(f"Prompt tokens: zero-shot {report['datasets']['test']['base']['sample_results'][0]['prompt_tokens']}, "
      f"few-shot {report['datasets']['test']['fewshot']['sample_results'][0]['prompt_tokens']}: few-shot pays this on every request.")
print(f"Fuse, benchmark, and evaluate: uv run python main.py benchmark --preset {PRESET} --fuse")
print(f"Serve the latest fused model:  uv run python main.py serve --preset {PRESET}")''')

markdown('exercises', '''## 10. Exercises

1. **Remove the mask.** `TrainingConfig` in `src/config.py` requires `mask_prompt: true`. Allow `false`, train for 50 iterations, and compare the training loss with a masked run. Which tokens now dominate it, and why does the evaluator's assistant-only loss no longer match training?
2. **Starve the data.** Regenerate with 50 training records instead of 250. Which challenge set suffers first?
3. **Remove abstention training.** Drop `no_action` from the training tools. What does the model do with unsupported requests?
4. **Find the overfitting point.** Run the `iters` ablation up to 800. Does exact match fall when validation loss rises?
5. **Scale up.** Repeat §4–6 with `PRESET = "14b"`. Does the bigger base model close the few-shot gap without training?
6. **Separate formatting from task skill.** Run `main.py eval --preset 3b --constrained` and compare the `grammar` variant against LoRA and few-shot. When invalid JSON is impossible, which failures survive — and what does that say about what training actually bought?
7. **How much of the headline is luck?** Run the sweep one seed per invocation (`main.py ablate seed 42 --iters 200`, then 43, then 44) and merge with `scripts/merge_seed_sweep.py`. Compare the spread across seeds with the Wilson intervals on a single run.
8. **What did the task cost?** Run `main.py forgetting` and read the per-record plot: which ordinary requests got worse, which got better, and does the sign test agree with the visual impression?
9. **Sample instead of decoding greedily.** `main.py eval --preset 3b --temperature 0.7 --seed 7`. Does format validity drop, and by how much? This is the cost you would pay for the diversity greedy decoding does not give you.

Solutions and expected outcomes for all of these are in [`docs/exercise-solutions.md`](docs/exercise-solutions.md).''')

nb.cells = cells
nbf.validate(nb)
with (Path(__file__).resolve().parent.parent / 'tutorial.ipynb').open('w') as output:
    nbf.write(nb, output)
print('Generated tutorial.ipynb')
