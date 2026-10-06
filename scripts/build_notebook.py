"""
Constructs tutorial.ipynb with clean cells, rich markdown, and executable code.
"""

from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Header
cells.append(nbf.v4.new_markdown_cell("""# 🚀 Modern Local LLM Fine-Tuning & Comprehensive Evals with Apple MLX
### *A Production-Grade Guide to LoRA/QLoRA and Multi-Pillar Evaluations on Apple Silicon*

Welcome! In this tutorial, you will learn how to:
1. **Leverage Apple Silicon's Unified Memory Architecture** using Apple's native **MLX** framework.
2. **Fine-tune an Instruct Model (Qwen 2.5 3B)** for high-reliability **JSON Tool Calling** using LoRA / QLoRA.
3. **Apply Prompt Masking** so loss backpropagation only updates assistant response tokens.
4. **Implement a 4-Pillar Evaluation Suite**:
   - **Pillar 1: Intrinsic Metrics** (Cross-Entropy Loss & Test Perplexity)
   - **Pillar 2: Deterministic Metrics** (Strict JSON parseability, Pydantic schema adherence, parameter exact match)
   - **Pillar 3: Qualitative Analysis** (Base vs. LoRA side-by-side behavioral review)
   - **Pillar 4: Systems Profiling** (Metal unified memory & token throughput)
5. **Fuse LoRA Adapters** into standalone weights for zero-overhead local deployment.

---"""))

# Cell 2: System & Hardware Verification
cells.append(nbf.v4.new_code_cell("""import sys
import os
import json
import time
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

console = Console()

# Verify Apple Silicon Hardware & Metal Acceleration
print(f"Python Version    : {sys.version.split()[0]}")
print(f"MLX Version       : {mx.__version__}")
print(f"Default Device    : {mx.default_device()}")
print(f"Metal Acceleration: {mx.metal.is_available()}")

# Peak memory check
mx.reset_peak_memory()
print(f"Initial Memory    : {mx.get_active_memory() / (1024**2):.2f} MB")"""))

# Cell 3: Markdown on Schema & Deterministic Contracts
cells.append(nbf.v4.new_markdown_cell("""## 1. Defining the Operational Tool Schema (Pydantic v2)

Base instruction models frequently struggle with strict JSON generation:
- They wrap outputs in Markdown code blocks (` ```json ... ``` `).
- They hallucinate parameters or use invalid enum variants (e.g. `"prod"` instead of `"production"`).
- They add conversational filler before or after the JSON.

To test this deterministically, we define 4 infrastructure operations using **Pydantic**:
1. `deploy_service` (service, version, environment, replicas, channels)
2. `restart_pod` (pod_name, region, force, reason)
3. `rollback_deployment` (deployment_id, target_tag, drain_traffic)
4. `scale_cluster` (cluster_name, node_count, auto_scale, instance_type)"""))

# Cell 4: Schema Code
cells.append(nbf.v4.new_code_cell("""from src.schema import (
    SYSTEM_PROMPT,
    DeployServiceParams,
    RestartPodParams,
    RollbackDeploymentParams,
    ScaleClusterParams,
    parse_and_validate
)

# Test our parser against typical LLM output variations
test_output_clean = '{"tool": "deploy_service", "parameters": {"service": "auth-api", "version": "v2.1.0", "environment": "production", "replicas": 3}}'
test_output_markdown = 'Sure! Here is the JSON:\\n```json\\n{"tool": "restart_pod", "parameters": {"pod_name": "postgres-0", "region": "us-west-2", "force": true, "reason": "OOM"}}\\n```'

res_clean = parse_and_validate(test_output_clean)
res_md = parse_and_validate(test_output_markdown)

print("Clean JSON Parse Result:")
print(f"  Pure JSON: {res_clean['is_pure_json']} | Schema Valid: {res_clean['is_schema_valid']}")

print("\\nMarkdown JSON Parse Result:")
print(f"  Pure JSON: {res_md['is_pure_json']} (Fails strict pure check) | Schema Valid: {res_md['is_schema_valid']}")"""))

# Cell 5: Markdown on Data Preparation & Prompt Masking
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Engineering & Prompt Masking

When fine-tuning instruction models, **Prompt Masking** is essential:
- Without prompt masking, the model calculates loss over the system prompt and user instructions, wasting gradient updates memorizing the input format.
- With `mask_prompt: true`, loss is **only** computed on the target tokens (the assistant's JSON output).

MLX expects datasets in standard ChatML/OpenAI JSONL format:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```"""))

# Cell 6: Inspecting Dataset
cells.append(nbf.v4.new_code_cell("""# Inspect generated dataset splits
with open("data/train.jsonl", "r") as f:
    train_lines = f.readlines()
with open("data/valid.jsonl", "r") as f:
    valid_lines = f.readlines()
with open("data/test.jsonl", "r") as f:
    test_lines = f.readlines()

print(f"Training Samples  : {len(train_lines)}")
print(f"Validation Samples: {len(valid_lines)}")
print(f"Holdout Test Samples: {len(test_lines)}")

# Preview one sample
sample = json.loads(train_lines[0])
print("\\n--- User Request Sample ---")
print(sample["messages"][1]["content"])
print("\\n--- Expected Assistant JSON ---")
print(sample["messages"][2]["content"])"""))

# Cell 7: Markdown on LoRA / QLoRA
cells.append(nbf.v4.new_markdown_cell("""## 3. LoRA & QLoRA Fine-Tuning on Apple Silicon

### Why LoRA?
Instead of modifying all 3 billion weights, **Low-Rank Adaptation (LoRA)** freezes the base model and injects trainable rank decomposition matrices:
$$W' = W + \\frac{\\alpha}{r} (B \\times A)$$

For `Qwen2.5-3B-Instruct-4bit`:
- Base parameters: **3.08 Billion** (quantized to 4 bits)
- Trainable parameters: **6.65 Million** (~0.21% of model)
- Adapter size on disk: **~25 MB** (instead of 6 GB)
- Memory required: **< 10 GB** during backpropagation!"""))

# Cell 8: Training / Config Execution
cells.append(nbf.v4.new_code_cell("""# You can run training directly from Python or via the MLX CLI:
# CLI: uv run python -m mlx_lm lora -c config/lora_config.yaml --train --iters 80

from src.train import run_training

# Run training (or inspect existing artifacts if already run)
if not Path("artifacts/adapters/adapters.safetensors").exists():
    print("Executing fine-tuning...")
    run_training(config_path="config/lora_config.yaml", iters_override=80)
else:
    print("Found existing fine-tuned LoRA adapters in 'artifacts/adapters'!")
    adapter_size_mb = Path("artifacts/adapters/adapters.safetensors").stat().st_size / (1024**2)
    print(f"Adapter weight file size: {adapter_size_mb:.2f} MB")"""))

# Cell 9: Markdown on 4-Pillar Evaluation Suite
cells.append(nbf.v4.new_markdown_cell("""## 4. The 4-Pillar Comprehensive Evaluation Suite

A modern MLX evaluation is not a single prompt inspection. We execute a rigorous 4-pillar benchmark:

| Pillar | Focus | What It Measures |
| :--- | :--- | :--- |
| **1. Intrinsic Metrics** | Statistical Fit | Cross-Entropy Loss & Perplexity on unseen holdout test set |
| **2. Deterministic Evals** | Functional Correctness | Pure JSON rate, Pydantic schema validity, Tool accuracy, Parameter Exact Match |
| **3. Qualitative Review** | Error Taxonomy | Side-by-side comparison of Base vs. LoRA completions |
| **4. Systems & Hardware** | Metal Efficiency | Generation throughput (tok/s), Peak Unified Memory (MB) |"""))

# Cell 10: Running Evaluation & Visualizing Scorecard
cells.append(nbf.v4.new_code_cell("""from src.evaluate import run_comprehensive_evaluation

# Run the 4-Pillar evaluation on holdout test samples
report = run_comprehensive_evaluation(
    model_name="mlx-community/Qwen2.5-3B-Instruct-4bit",
    adapter_path="artifacts/adapters",
    num_eval_samples=20
)"""))

# Cell 11: Display Inline Plot of Evaluation Results
cells.append(nbf.v4.new_code_cell("""# Display Evaluation Comparison Chart
from IPython.display import Image, display

if Path("artifacts/eval_comparison.png").exists():
    display(Image(filename="artifacts/eval_comparison.png"))
else:
    print("eval_comparison.png not found. Run evaluation first.")"""))

# Cell 12: Markdown on Benchmarking & Adapter Fusion
cells.append(nbf.v4.new_markdown_cell("""## 5. Inference Benchmarking & Model Weight Fusion

### Dynamic LoRA vs. Weight Fusion
- **Dynamic LoRA**: At inference time, the model executes the base layer, computes the adapter branch, and adds them. This incurs a small latency penalty.
- **Model Fusion (`mlx_lm.fuse`)**: Fuses $W + \\frac{\\alpha}{r}(B \\times A)$ directly into a new standalone model weights file.
  - Zero runtime overhead: runs at full base-model speed (~140 tok/s).
  - Standalone: No adapter files needed at runtime.
  - Ready for local API serving via `mlx_lm.server`."""))

# Cell 13: Benchmark Code & Serving Instructions
cells.append(nbf.v4.new_code_cell("""from src.benchmark import run_benchmark_suite

# Run benchmark across Base vs LoRA
run_benchmark_suite(
    model_name="mlx-community/Qwen2.5-3B-Instruct-4bit",
    adapter_path="artifacts/adapters"
)

print("\\nTo serve your fine-tuned model as an OpenAI-compatible API on Apple Silicon:")
print("  uv run python -m mlx_lm.server --model artifacts/fused_model --port 8080")"""))

nb.cells = cells

with open("tutorial.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Successfully generated tutorial.ipynb!")
