# 🚀 Modern Local LLM Fine-Tuning & Comprehensive Evals with Apple MLX

A production-grade tutorial and reference implementation for **fine-tuning local language models with LoRA/QLoRA** and conducting **multi-pillar evaluations** natively on Apple Silicon using Apple's [MLX](https://github.com/ml-explore/mlx) framework.

---

## 📑 Table of Contents
1. [Why Apple Silicon & MLX?](#-why-apple-silicon--mlx)
2. [Tutorial Architecture & Scenario](#-tutorial-architecture--scenario)
3. [The 4-Pillar Comprehensive Evaluation Suite](#-the-4-pillar-comprehensive-evaluation-suite)
4. [Project Structure](#-project-structure)
5. [Quickstart (with `uv`)](#-quickstart-with-uv)
6. [Module 1: Dataset Engineering & Prompt Masking](#-module-1-dataset-engineering--prompt-masking)
7. [Module 2: LoRA & QLoRA Fine-Tuning](#-module-2-lora--qlora-fine-tuning)
8. [Module 3: Running the 4-Pillar Evaluation Suite](#-module-3-running-the-4-pillar-evaluation-suite)
9. [Module 4: Systems Benchmarking & Model Fusion](#-module-4-systems-benchmarking--model-fusion)
10. [Serving Locally (OpenAI-Compatible API)](#-serving-locally-openai-compatible-api)

---

## ⚡ Why Apple Silicon & MLX?

Traditional PyTorch training relies on discrete GPU memory (VRAM), requiring constant data transfers over PCIe buses. In contrast, Apple Silicon features **Unified Memory Architecture (UMA)**:
- **Zero-Copy Sharing**: The CPU and Metal GPU share the exact same physical memory pool.
- **Massive Local Context**: Macs with 36 GB, 64 GB, 96 GB, or 128+ GB of unified memory can fine-tune 7B, 14B, or 32B parameter models that would otherwise require multiple server-grade GPUs ($10,000+).
- **Native Metal Acceleration**: MLX optimizes compute kernels directly for Apple Metal, avoiding CUDA translation layers.

---

## 🎯 Tutorial Architecture & Scenario

### The Problem: Reliable Tool Calling & Structured JSON
Base instruction models frequently fail when required to output strict JSON:
1. They prefix responses with conversational filler (*"Sure, here is your command..."*).
2. They wrap outputs in Markdown backticks (`` ```json ... ``` ``).
3. They hallucinate parameters or enum values (e.g. `"prod"` instead of `"production"`).

### The Solution: Targeted LoRA Fine-Tuning
By fine-tuning **Qwen 2.5 3B Instruct** with LoRA:
- **Model**: `mlx-community/Qwen2.5-3B-Instruct-4bit` (QLoRA)
- **Trainable Parameters**: **6.65 Million** out of 3.08 Billion (**0.21%**)
- **Adapter Weight Footprint**: **~25 MB** (vs ~1.8 GB base model)
- **Result**: 100% pure JSON without markdown wrappers, lower test perplexity, and +20% jump in parameter exact match.

---

## 📊 The 4-Pillar Comprehensive Evaluation Suite

Most fine-tuning tutorials stop at *"It printed some text, look, it works!"*. This tutorial implements a **4-pillar evaluation framework**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   MLX Multi-Pillar Evaluation Suite                    │
├────────────────────┬───────────────────────────────────────────────────┤
│ 1. Intrinsic       │ • Cross-Entropy Loss on Holdout Test Split        │
│    Metrics         │ • Perplexity ($e^{\text{loss}}$) Base vs. LoRA    │
├────────────────────┼───────────────────────────────────────────────────┤
│ 2. Deterministic   │ • Pure JSON Output Rate (no markdown/chatter)     │
│    Validation      │ • Pydantic v2 Schema Compliance Rate (%)          │
│                    │ • Tool Selection Accuracy                         │
│                    │ • Parameter Field-Level Exact Match Rate          │
├────────────────────┼───────────────────────────────────────────────────┤
│ 3. Qualitative     │ • Side-by-side prompt output diffs                │
│    Analysis        │ • Error taxonomy (formatting vs logic errors)     │
├────────────────────┼───────────────────────────────────────────────────┤
│ 4. Systems &       │ • Peak Apple Metal Unified Memory (MB)            │
│    Performance     │ • Generation Throughput (tokens/second)           │
│                    │ • Adapter Fusion (`mlx_lm.fuse`) footprint        │
└────────────────────┴───────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```text
mlx-model-training-evals/
├── pyproject.toml              # Modern uv / PEP 621 package config
├── README.md                   # Complete tutorial guide & documentation
├── tutorial.ipynb              # Fully runnable interactive Jupyter Notebook
├── config/
│   └── lora_config.yaml        # Declarative MLX LoRA hyperparameters
├── data/
│   ├── prepare_dataset.py      # Dataset generator (train / valid / test)
│   ├── train.jsonl             # 200 instruction training samples
│   ├── valid.jsonl             # 40 validation samples
│   ├── test.jsonl              # 60 holdout evaluation samples
│   └── raw_test_samples.json   # Raw prompts & ground-truth for inspection
├── src/
│   ├── __init__.py
│   ├── schema.py               # Pydantic v2 tool models & output validator
│   ├── train.py                # Python training runner with memory & loss tracking
│   ├── evaluate.py             # 4-pillar evaluation runner & chart generator
│   └── benchmark.py            # Apple Silicon throughput & memory benchmark
└── artifacts/
    ├── adapters/               # Trained LoRA weights (adapters.safetensors)
    ├── fused_model/            # Standalone deployable model (post-fusion)
    ├── eval_comparison.png     # Evaluation scorecard visual comparison
    ├── eval_results.json       # Detailed benchmark metrics
    └── benchmark_results.json  # Tok/s and memory profiling data
```

---

## 🚀 Quickstart (with `uv`)

This project uses [`uv`](https://github.com/astral-sh/uv) for fast Python environment management.

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/mlx-model-training-evals.git
cd mlx-model-training-evals

# uv automatically creates .venv and installs locked dependencies in seconds
uv sync
```

### 2. Generate the Dataset
```bash
uv run python data/prepare_dataset.py
```

### 3. Launch the Interactive Notebook
```bash
uv run jupyter lab tutorial.ipynb
```

Or run everything from the terminal using the modular CLI scripts described below.

---

## 🛠️ Module 1: Dataset Engineering & Prompt Masking

### The Chat Schema
MLX uses the standard ChatML / OpenAI JSONL format:
```json
{
  "messages": [
    {"role": "system", "content": "You are an automated Cloud Infrastructure Action Dispatcher..."},
    {"role": "user", "content": "Deploy auth-api version v2.1.0 to staging with 3 replicas."},
    {"role": "assistant", "content": "{\"tool\":\"deploy_service\",\"parameters\":{\"service\":\"auth-api\",\"version\":\"v2.1.0\",\"environment\":\"staging\",\"replicas\":3,\"notify_channels\":[]}}"}
  ]
}
```

### Why Prompt Masking Matters
In [`config/lora_config.yaml`](file:///Users/markcastillo/git/mlx-model-training-evals/config/lora_config.yaml):
```yaml
mask_prompt: true
```
Without prompt masking, standard language model training computes cross-entropy loss over every token in the sequence (including the 300+ token system prompt). By setting `mask_prompt: true`, MLX zeroes out the loss for all tokens preceding the assistant's turn, ensuring that **100% of gradient updates optimize the model's structured response**.

---

## 🏋️ Module 2: LoRA & QLoRA Fine-Tuning

### The Hyperparameters
In [`config/lora_config.yaml`](file:///Users/markcastillo/git/mlx-model-training-evals/config/lora_config.yaml):
```yaml
model: "mlx-community/Qwen2.5-3B-Instruct-4bit"
data: "data"
fine_tune_type: "lora"
lora_parameters:
  rank: 8          # Dimension of decomposition matrices
  scale: 16.0      # Scaling factor (alpha)
  dropout: 0.05
learning_rate: 1.0e-4
batch_size: 4
iters: 80
num_layers: 16     # Number of attention/MLP layers to adapt
optimizer: "adamw"
adapter_path: "artifacts/adapters"
```

### Running Training
You can run training either via the Python script:
```bash
uv run python src/train.py --iters 80
```
Or directly via the MLX CLI:
```bash
uv run python -m mlx_lm lora -c config/lora_config.yaml --train --iters 80
```

### Training Dynamics Observed
On an Apple M2 Max:
- **Starting Validation Loss**: `1.381`
- **Final Validation Loss**: `0.016` (after 80 iterations)
- **Peak Metal Memory**: ~10.2 GB during backpropagation
- **Adapter Weight File**: `25.4 MB`

---

## 📈 Module 3: Running the 4-Pillar Evaluation Suite

Run the evaluation script across the holdout test set:
```bash
uv run python src/evaluate.py --samples 25
```

### Benchmark Results Scorecard

| Evaluation Metric | Base Model (`Qwen2.5-3B-4bit`) | LoRA Fine-Tuned Model | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Holdout Test Loss** | `3.0637` | **`2.5737`** | **-0.4900 (Better fit)** |
| **Holdout Test Perplexity** | `21.41` | **`13.11`** | **-8.29 PPL (Stronger confidence)** |
| **Pure JSON Rate** | 100.0% | 100.0% | Stable |
| **Pydantic Schema Validity** | 100.0% | 100.0% | Stable |
| **Tool Selection Accuracy** | 100.0% | 100.0% | 100% correct routing |
| **Parameter Exact Match** | `46.7%` | **`66.7%`** | **+20.0% accuracy** |

The evaluation outputs an automated comparison plot at `artifacts/eval_comparison.png` and full JSON telemetry at `artifacts/eval_results.json`.

---

## 🔬 Module 4: Systems Benchmarking & Model Fusion

### Profiling Inference on Apple Silicon Metal
Run the profiler:
```bash
uv run python src/benchmark.py
```

| Metric | Base Model | LoRA Adapter (Dynamic) | Fused Model |
| :--- | :---: | :---: | :---: |
| **Generation Speed** | ~141 tok/s | ~70 tok/s | **~141 tok/s** |
| **Peak Metal Memory** | ~1,755 MB | ~1,773 MB | **~1,755 MB** |
| **Runtime Overhead** | 0% | Adapter calculation branch | **Zero overhead** |

### Weight Fusion (`mlx_lm.fuse`)
Dynamic LoRA adapters introduce a minor latency overhead because the model must branch and compute $W_{\text{base}}x + \frac{\alpha}{r}BAx$.

Using `mlx_lm.fuse`, you can bake the adapter weights directly into the base model weights:
```bash
uv run python -m mlx_lm.fuse \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --adapter-path artifacts/adapters \
  --save-path artifacts/fused_model
```

The resulting `artifacts/fused_model/` contains standalone `model.safetensors` that run at full native speed without loading adapter files!

---

## 🌐 Serving Locally (OpenAI-Compatible API)

You can serve your fine-tuned or fused model locally with Apple Silicon acceleration:

```bash
uv run python -m mlx_lm.server --model artifacts/fused_model --port 8080
```

Test it with `curl`:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are an automated Cloud Infrastructure Action Dispatcher..."},
      {"role": "user", "content": "Restart pod postgres-primary-0 in us-east-1 immediately because of OOM error."}
    ],
    "temperature": 0.0
  }'
```

---

## 🎓 Summary of Key Learnings

1. **Prompt Masking is Crucial**: Always set `mask_prompt: true` during instruction fine-tuning to prevent the model from wasting gradient updates on input instructions.
2. **Unified Memory Unlocks Local Workflows**: 96 GB of unified memory on Apple Silicon allows running multiple models and fine-tuning without GPU memory thrashing.
3. **Comprehensive Evals Beat "Eyeball Tests"**: Combining intrinsic metrics (Perplexity) with deterministic validation (Pydantic parsing) and hardware profiling provides a trustworthy signal of model readiness.
4. **Fuse Adapters Before Deployment**: Use `mlx_lm.fuse` to restore 100% of base model inference speed for production serving.
