# Concepts: one page per number this repo prints

A companion to the tutorial. Every entry answers three questions: **what is it, why does it matter here, and where do you see it in this repo**. Terminal examples assume you are at the repo root with `uv` installed.

---

## Cross-entropy and perplexity

The training loss is **cross-entropy** in nats per token:

```
loss = -mean(log P_model(correct token))        # averaged over scored tokens
```

If the model puts probability `p` on the right token, that token contributes `-log p` nats. A confident correct answer costs ~0; a coin-flip (p = 1/2) costs 0.69 nats (that is why `log(2)` appears in the tests); a uniform guess over a 48-token vocabulary costs `ln 48 ≈ 3.87` nats.

**Perplexity** is `e^loss`: the effective number of equally likely choices the model behaves as if it faced. Loss 0 → perplexity 1 ("certain"), loss 0.69 → 2 ("a fair coin"), loss 3.87 → 48 ("guessing"). Loss is the more honest unit: it is what the optimizer minimizes, it behaves linearly, and two models' losses are directly comparable even when one is "certain but wrong" — a model that is confidently wrong has *higher* loss than one that hesitates.

- **See it:** every `main.py train` run prints train/val loss; `eval` reports "assistant loss". `src/mini_train.py` prints the `ln(vocab)` guessing floor on the toy loss plot, where the overfitting model's val loss climbs *above* the floor.
- **Read:** [Bits-per-character intuitions](https://en.wikipedia.org/wiki/Perplexity).

## The loss mask (`mask_prompt: true`)

A chat record becomes one token sequence: system prompt, user request, assistant answer. Without masking, most of the gradient would go into reproducing the ~600-token system prompt, which is identical in every example. With `mask_prompt: true`, the loss scores **only assistant tokens** — the positions the model is trained to produce.

- **See it:** `main.py show-mask --split train --index 3` highlights exactly which tokens are scored, using MLX-LM's own `ChatDataset`, so the picture *is* the training reality. `src/mini_train.py` makes the same idea visible by construction: its forward pass computes logits only for answer positions.
- **Catch:** the evaluator applies the same mask when reporting "assistant loss", so training loss and evaluation loss are comparable.

## LoRA: the math and the parameter count

LoRA freezes the base weights `W` and learns a low-rank update:

```
W_effective = W + scale · (B @ A)        A: r × d_in,   B: d_out × r
```

Rank `r` is the number of independent directions the adapter can push the projection. `B` is initialized to zero, so training starts from the unchanged base model — step 0 cannot be worse than the base. For a `d_in = d_out = 2048` projection at rank 8, the update needs `8 · (2048 + 2048) = 32,768` numbers instead of 4.2 million.

Which projections get adapters is `num_layers` (the last N transformer blocks) — in MLX-LM, the attention `q/k/v/o` and MLP `gate/up/down` projections.

- **See it:** `main.py show-params --preset 3b` lists each adapted projection, its rank, and its parameter count; the training summary prints the total. `src/mini_train.py` applies the same formula to a toy head and prints the adapter-vs-full comparison (e.g. 288 of 2376 parameters).
- **Catch:** equal `iters` with different `batch_size` means different numbers of examples seen.

## `scale` is not `alpha / r`

Some libraries (Hugging Face PEFT) parameterize LoRA as `W + (alpha/r) · B@A` and tune `alpha`. MLX applies `scale` directly: `W + scale · B@A`. Concretely, the default config here uses `rank: 8, scale: 16.0`, which is *not* "alpha=16 over r=8"; it multiplies the update by 16 outright.

- **See it:** `config/base.yaml`, and the `scale` ablation (`main.py ablate scale 4 16 64 --iters 100`).
- **Catch:** moving between MLX and PEFT recipes requires converting this number, or you silently train with a 8×-strength adapter.

## 4-bit quantization and "logical" parameters

The presets load `Qwen2.5-*-Instruct-4bit`: each weight is stored as 4-bit codes plus per-group scales instead of fp16. The *logical* parameter count (what a full-precision copy would hold) is unchanged, so `show-params` reports the adapter as a percentage of the logical 3.09B, not of the ~1.7 GB actually on disk. The helper that undoes the packing is `src/explain.py:_logical_size`.

- **See it:** `main.py show-params` prints "6,651,904 parameters (0.216% of 3.09B)".
- **Catch:** fusion re-quantizes merged weights, which can change outputs slightly — that is why `benchmark --fuse` re-runs the quality evaluation.

## Learning curves: reading loss

- Both curves fall and flatten → learning, then saturating.
- Train keeps falling while validation turns upward → **overfitting** (memorizing training wording). The trainer flags this automatically (`overfitting_suspected`) and marks the best validation iteration on the plot.
- A tiny absolute rise near zero loss is usually noise: the trainer compares the rise against the total improvement, not against zero.
- The toy loop (`main.py toy-train`) shows the pure form: train loss → ~0 with 100% accuracy while val loss climbs above the guessing floor, because the val answers *cannot* be derived from training.

- **See it:** `training_history.json` (`loss_summary`) and `loss_curve.png` in every training run.

## Wilson confidence intervals

A rate measured on `n` samples is an estimate of a true probability, not the probability itself. The **95% Wilson score interval** is the range of true rates compatible with what you measured, and it behaves sanely at 0% and 100% (unlike the naive `± 1.96 √(p(1-p)/n)`).

With 15 samples per tool, a measured 80% is compatible with anything from about 55% to 93%. That is why the README's reference tables carry intervals everywhere, and why differences of a few points on these sets are usually noise.

- **See it:** every rate in the eval report and the `[low–high]` error bars on the charts; `src/metrics.py:wilson_interval`.

## McNemar's paired test

Two variants answered the *same* samples, so compare them sample by sample. Samples both get right — or both wrong — say nothing about which variant is better; only **discordant pairs** (one right, one wrong) carry evidence. McNemar's exact test asks: if the two variants were equally good, how surprising is a split this lopsided? Answer: `p`.

Overlapping confidence intervals do not settle the question; the paired test is more sensitive because it removes between-sample variance.

- **See it:** the "Paired exact-match comparison" table in every `eval` report, and `src/metrics.py:paired_comparison`.
- **Catch:** `p < 0.05` means "unlikely to be sampling noise on this set", not "definitely better on all traffic".

## Greedy decoding vs sampling

Evaluation uses `temperature = 0` (greedy): every variant always picks the single most likely next token. This makes comparisons deterministic — differences come from the model, not the dice. Greedy also *understates* what a model can do on structured outputs: a token-level mistake (e.g. `rollback_deployment` over `scale_cluster`) is locked in forever, while sampling might recover. Temperature `t > 0` flattens the probability distribution (`t = 1` unchanged, `t → 0` greedy); top-p (nucleus) sampling truncates the tail.

- **See it:** `src/inference.py:generate_response` (`make_sampler(temp=temperature)`); the temperature and seed are recorded in the run manifest's `generation` block.
- **Constrained decoding:** instead of sampling the model's distribution directly, mask every token that would leave the grammar of valid calls, so the output is valid JSON *by construction*. `src/constrained.py` derives that grammar from the same Pydantic models that build the system prompt, and `src/evaluate.py` exposes it as the `grammar` variant (`eval --constrained`). It is the honest way to ask "how much of this task is formatting?" — on the 3B reference preset it lifts the base model's schema validity from 0% to **100%, identical to the trained adapter** (75/75 samples, no discordant pair), and its exact match from 0% to 43%, with no training at all; the rest is tool-choice and parameter error. See the README's constrained-decoding section for the full scorecard.
- **Sampling:** `eval --temperature 0.7 --seed 7` samples instead of decoding greedily. The seed makes a run reproducible, but each variant is then one stochastic draw: expect format validity to fall and to vary between runs, which is exactly the cost you would pay in production.

## Catastrophic forgetting

Fine-tuning on a narrow task can degrade everything else the model knew. The usual story is **catastrophic forgetting**: gradient descent on the task's distribution moves shared weights away from the general-purpose solution, and the general loss rises even though task accuracy improves.

This repo measures it the cheap way (`main.py forgetting`): 24 ordinary requests with a *plain* assistant system prompt, scored by assistant loss — the same masked cross-entropy as training, but with no exact-match component, because free-form answers have no single correct string. Using a different system prompt is deliberate: an adapter that has overfit the ops contract will answer "What is 17 times 24?" with a tool-call envelope, and that shows up as a loss increase on the ordinary answer.

The paired comparison is an exact **sign test** over per-record loss changes: it counts records that got worse versus better and ignores ties, so a couple of unlucky records cannot manufacture significance. What it cannot do is prove general ability was preserved — 24 records of loss say "not obviously destroyed", nothing stronger.

- **See it:** `main.py forgetting --preset 3b`, `src/forgetting.py`, `src/metrics.py:sign_test`.
- **Related:** LoRA is inherently gentler than full fine-tuning here because the base weights never move; the `forgetting` check is how you would confirm that for your own data.

## Prefill, decode, TTFT

- **Prefill:** the prompt is processed in parallel (one pass, all tokens at once). Throughput is measured in prompt tokens/s.
- **Decode:** output tokens are generated one at a time, each needing a forward pass. Throughput is decode tokens/s — usually 2–4× slower than prefill per token.
- **TTFT (time to first token):** prefill + first decode step — the latency a user feels before the answer starts.

Few-shot prompting pays prefill cost on *every* request: the reference run's few-shot prompt is 985 tokens against 666 for zero-shot.

- **See it:** `main.py benchmark` (base vs LoRA vs fused), and `prompt_tokens` in `eval_results.json`.

## Peak Metal memory

`mx.get_peak_memory()` reports the high-water mark of the Metal allocator during a run. It is the number that decides whether a preset fits on your Mac — and it includes activations, gradients, and optimizer state, not just weights. Gradient checkpointing (`grad_checkpoint: true` on larger presets) trades recomputation for memory.

- **See it:** the "Peak Metal memory" line in every training summary; `main.py benchmark` reports the same for inference.
- **Catch:** before trying a larger preset, run `main.py train --preset 14b --iters 10` and read that line. macOS can also abort long GPU jobs when the display or another app needs the GPU (`[METAL] Command buffer execution failed` in Troubleshooting).

## Fusion

A dynamic adapter costs an extra pair of matmuls per adapted projection per token. **Fusion** merges `scale · B@A` into the weights, producing a standalone model directory with no per-token LoRA overhead. On the reference run, the dynamic adapter decoded at 41 tokens/s against 102 for the base model; fusion should recover most of that. With a quantized base, fusion re-quantizes the merged weights, which can change outputs — hence the linked quality re-evaluation.

- **See it:** `main.py benchmark --preset 3b --fuse` (benchmark + quality for base/LoRA/fused in one command), and `src/fuse.py`.
- **Serve:** `main.py serve --preset 3b` serves the latest fused model via an OpenAI-compatible API. The server does not add this project's system prompt; send `SYSTEM_PROMPT` from `src/schema.py` with each request.

## Exact match vs normalized match

**Exact match** requires: pure JSON, schema-valid, and byte-identical to the reference — including optional parameters written with their documented defaults, and exact types (`3` not `"3"`). **Normalized match** is the weaker view: equal after omitted defaults are filled in. Reporting both separates "wrong values" from "right values, wrong style" — the `omitted_default` failure category exists precisely for that gap.

- **See it:** the metric definitions table in the README, `src/metrics.py:score_sample`, and the failure-category chart in every evaluation.

---

## Where each concept lives in code

| Concept | Module |
|---|---|
| Cross-entropy loss, masking | `mlx_lm.tuner.trainer.default_loss` (used identically in `src/evaluate.py:compute_perplexity`) |
| LoRA math, size, packing | `src/explain.py` |
| Wilson interval, McNemar | `src/metrics.py` |
| Greedy decoding, timing | `src/inference.py` |
| Grammar-constrained decoding | `src/constrained.py` |
| From-scratch loop (everything above, toy-sized) | `src/mini_train.py` |
