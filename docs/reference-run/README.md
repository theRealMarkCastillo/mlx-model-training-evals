# Reference run

Curated outputs from one complete pipeline run on the 3B preset (Qwen2.5-3B-Instruct-4bit, M2 Max): training, evaluation with challenge sets and the grammar-constrained variant, benchmark, forgetting check, and an iteration ablation. `summary.json` holds the aggregate numbers (the full per-sample `eval_results.json` files stay in `artifacts/`, which is gitignored — they are several megabytes of prompts and outputs).

Regenerate it from the newest 3B runs:

```bash
uv run python scripts/export_reference.py
```

What it copies and why:

| File | Source |
|---|---|
| `summary.json` | aggregate metrics from `latest_training.json`, `latest_evaluation.json`, `latest_benchmark.json`, `latest_ablation.json`, `latest_forgetting.json`, plus provenance (run ids, versions, platform) |
| `seed_sweep.json` | merged per-seed runs (`scripts/merge_seed_sweep.py`): exact match per seed with the mean ± spread |
| `loss_curve.png` | the training run's loss plot |
| `eval_comparison.png`, `challenge_comparison.png` | the evaluation run's charts (four variants, because the run used `--constrained`) |
| `forgetting.png` | per-record general-capability loss, base vs LoRA, red where fine-tuning hurt |
| `ablation.png` | the iteration sweep |

**Caveats worth keeping in mind when citing these numbers.**

* The individual tables come from a **single seed** (42). `seed_sweep.json` is the mean ± spread version; check whether the seeds agree before reading anything into a few points of difference.
* Long GPU runs on macOS can abort with `Impacting Interactivity` (see the README's troubleshooting). When that happens the run's manifest is marked `failed`, no pointer is published, and only the affected seed needs rerunning — which is why the sweep is run one seed per invocation.
* Metal kernels are not bit-reproducible, so your own reruns will differ in the last digits even with the same seed.
* Ablations over `iters` share a seed and follow one trajectory (their validation losses at iterations 25 and 50 are identical), so that sweep is closer to "checkpoints of one run" than to four independent experiments.
* The `ablation` section of `summary.json` is the **iteration** sweep. Seed results live in `seed_sweep.json`.
* Outputs from the pre-rewrite pipeline are in `artifacts/historical/` and are **not** comparable to anything here.

## `dpo/`

Curated outputs of the preference-tuning experiment described in the main README ("Preference tuning (DPO) against more SFT data"): `comparison_summary.json` (all five adapters on the holdout and challenge sets, with paired tests against the SFT reference and the forgetting losses; per-sample outputs stay in `artifacts/`), `comparison.png`, `mining_report.json` (which prompts the SFT adapter failed), and the loss, reward-accuracy and log-probability-shift history of the 100-iteration plain and NLL-anchored DPO runs. These are exported by hand from the runs named in `comparison_summary.json`; `scripts/export_reference.py` does not regenerate them.

Follow-up (`dpo-mix`): `comparison_mixed_summary.json` / `comparison_mixed.png` compare SFT, both earlier arms, anchored DPO on mixed pairs, and an SFT control on the mixed data; `dpo_anchor_mixed_*` are that run's curves and history.

## `grpo/`

`comparison_summary.json` / `comparison.png`: the under-trained start adapter, three seeds of matched SFT and three of GRPO, on the holdout and challenge sets, with forgetting. `*_seed42_curves.png` / `*_seed42_history.json`: held-out greedy accuracy and rollout statistics during training for seed 42.

## `deployment.json`

Per-candidate accuracy on the holdout and challenge sets, per-request latency, TTFT, decode speed, prompt tokens and weights size for the 3B adapter (dynamic and fused) against prompted 14B and 32B, plus the controlled 3B benchmark. Regenerate with `scripts/export_deployment.py`.
