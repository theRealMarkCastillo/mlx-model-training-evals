# Historical artifacts

These reports, charts, and adapter configs come from an earlier version of the dataset and evaluator, before strict exact match, assistant-only loss, the `no_action` tool, and the challenge sets existed. They are kept only for inspection. **Do not cite them as results of the current pipeline**: the task, data, and metrics have all changed.

Local weights that were next to them (`*.safetensors`, `fused_model/`, and the 14B/32B folders) were moved here as well and are gitignored. Nothing in the current code reads from this directory: preset defaults resolve only through `artifacts/qwen2.5-<size>/adapters/latest.json`, which training writes after a successful run.

For current reference numbers, see `docs/reference-run/`.
