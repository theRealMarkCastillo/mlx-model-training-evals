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
