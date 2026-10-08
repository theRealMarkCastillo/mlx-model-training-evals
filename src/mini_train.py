"""A from-scratch LoRA training loop on a toy model, for teaching.

The real pipeline delegates its loop to `mlx_lm.train_model`, which hides the
mechanics this repo is about. This module builds the same mechanics by hand on
a toy next-token task small enough to train in seconds on any Mac:

    forward pass  ->  masked cross-entropy  ->  mx.value_and_grad  ->  AdamW step

The toy task is *designed* to fail to generalize: each training "request" is a
random context of C tokens whose fixed 2-token "answer" comes from a table the
model can only memorize. The validation contexts share no answers with the
training contexts, so training loss falls toward zero while validation loss
stays at ln(vocab_size) -- the loss curve that means "memorized, not learned".

Every knob below has a one-to-one mapping to a `main.py train` setting, printed
at the end of a run, so the toy loop is a decoder ring for the real training log.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from rich.console import Console
from rich.table import Table

from src.models import ARTIFACTS
from src.runs import finish_run, new_run, record_failure, write_json

console = Console()

# Toy-task shape. Kept tiny on purpose: the model is a lookup table with an
# embedding layer, not an LLM, so the loss curves are readable in one glance.
VOCAB = 48            # distinct tokens
CONTEXT = 4           # "prompt" tokens (never scored)
ANSWER = 2            # "assistant" tokens (always scored)
D_EMBED = 8           # embedding width
D_HIDDEN = 24         # hidden width of the two-layer head
DEFAULT_ITERS = 300
DEFAULT_SEED = 7
DEFAULT_RANK = 4
DEFAULT_LR = 0.1


def build_toy_data(seed=DEFAULT_SEED, n_train=24, n_val=8,
                   vocab=VOCAB, context=CONTEXT, answer=ANSWER):
    """Fixed table of random contexts -> random 2-token answers.

    Contexts are unique across train and val, so val answers cannot be looked
    up from training data: the only possible strategy is memorization, which
    is exactly what the val loss reveals.
    """
    rng = np.random.RandomState(seed)
    total = n_train + n_val
    contexts = set()
    while len(contexts) < total:
        contexts.add(tuple(rng.randint(0, vocab, size=context)))
    contexts = sorted(contexts)
    rng.shuffle(contexts)
    answers = rng.randint(0, vocab, size=(total, answer))
    sequences = [list(c) + list(a) for c, a in zip(contexts, answers, strict=True)]
    train = np.array(sequences[:n_train], dtype=np.int32)
    val = np.array(sequences[n_train:], dtype=np.int32)
    return train, val


def init_base(seed=DEFAULT_SEED):
    """Frozen base parameters: embedding, one hidden layer, output head.

    Returns plain mx arrays. Nothing here is differentiated in LoRA mode.
    """
    mx.random.seed(seed)

    def w(*shape):
        return mx.random.normal(shape) * 0.1

    return {
        "emb": w(VOCAB, D_EMBED),
        "W1": w(D_HIDDEN, CONTEXT * D_EMBED),
        "b1": mx.zeros((D_HIDDEN,)),
        "W2": w(VOCAB, D_HIDDEN),
        "b2": mx.zeros((VOCAB,)),
    }


def init_adapter(rank, seed=DEFAULT_SEED):
    """LoRA pair for the output head: A ~ N(0, .05), B = 0.

    B starts at zero so the first forward pass is numerically identical to
    the frozen base model -- training cannot make things worse at step 0.
    """
    mx.random.seed(seed + 1)
    return {
        "A": mx.random.normal((rank, D_HIDDEN)) * 0.05,
        "B": mx.zeros((VOCAB, rank)),
    }


def forward(base, params, x, scale):
    """Predict the next token after each C-token window: the whole model.

    For every position i in [C, T): flatten the embeddings of the C previous
    tokens, apply the hidden layer, then the head, then the LoRA update:

        W2_effective = W2 + scale * (B @ A)     A: r x d_in,  B: d_out x r

    Only answer positions get logits at all, so the "loss mask" here is not a
    mask but the *construction* of the forward pass -- the same guarantee
    `mask_prompt: true` gives the real trainer, made visible.
    """
    seq_len = x.shape[1]
    if params.get("A") is not None and params.get("B") is not None:
        lora = scale * (params["B"] @ params["A"])
    else:
        lora = None
    emb = base["emb"] if "W1" not in params else params["emb"]
    W1 = base["W1"] if "W1" not in params else params["W1"]
    b1 = base["b1"] if "b1" not in params else params["b1"]
    W2 = base["W2"] if "W2" not in params else params["W2"]
    b2 = base["b2"] if "b2" not in params else params["b2"]
    logits = []
    for i in range(CONTEXT, seq_len):
        window = mx.take(emb, x[:, i - CONTEXT:i], axis=0)   # (n, C, d)
        h = window.reshape(x.shape[0], CONTEXT * D_EMBED)    # (n, C*d)
        h = mx.maximum(h @ W1.T + b1, 0)                     # relu hidden layer
        out = h @ W2.T + b2                                  # frozen head
        if lora is not None:
            out = out + h @ lora.T                           # low-rank update
        logits.append(out)
    return mx.stack(logits, axis=1)                          # (n, A, V)


def masked_cross_entropy(logits, x):
    """Average cross-entropy over answer positions only.

    logits[i] (for i in [C, T)) predicts token x[i] from the C-token window
    before it. Prompt positions are never predicted, so no explicit mask is
    needed -- see forward().
    """
    targets = x[:, CONTEXT:]
    ce = nn.losses.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none")
    return ce.mean()


def adamw_step(params, grads, state, t, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
    """AdamW, written out so the optimizer is not a black box either.

    m tracks the running mean of the gradient (momentum), v its running
    second moment (scale). Both are bias-corrected by (1 - beta**t) because
    they start at zero. Weight decay is applied to the parameters directly
    (that is what makes it Adam*W* rather than L2 regularization).
    """
    pairs = tree_flatten(params)
    updated = []
    for (key, p), (_, g), s in zip(pairs, tree_flatten(grads), state, strict=True):
        s["m"] = b1 * s["m"] + (1 - b1) * g
        s["v"] = b2 * s["v"] + (1 - b2) * g * g
        m_hat = s["m"] / (1 - b1 ** t)
        v_hat = s["v"] / (1 - b2 ** t)
        updated.append((key, p - lr * (m_hat / (mx.sqrt(v_hat) + eps) + weight_decay * p)))
    return tree_unflatten(updated)


def evaluate(base, params, x, scale):
    """Full-batch loss and greedy-decoded answer accuracy for one split."""
    logits = forward(base, params, x, scale)
    loss = masked_cross_entropy(logits, x)
    predicted = mx.argmax(logits, axis=-1)
    targets = mx.array(x[:, CONTEXT:])
    correct = (predicted == targets).all(axis=-1)
    return float(loss.item()), float(correct.mean().item())


def plot_toy_losses(history, output_path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot([h["iteration"] for h in history["train"]], [h["loss"] for h in history["train"]],
            label="train loss", color="#1f77b4", marker="o", markersize=3)
    ax.plot([h["iteration"] for h in history["val"]], [h["loss"] for h in history["val"]],
            label="val loss (unseen contexts)", color="#ff7f0e", marker="s", markersize=5)
    ax.axhline(np.log(VOCAB), color="grey", linestyle=":", alpha=0.7)
    ax.annotate(f"ln({VOCAB}) = guessing (uniform)", xy=(0.02, np.log(VOCAB) + 0.15),
                xycoords=("axes fraction", "data"), fontsize=8, color="grey")
    ax.set_title("Toy LoRA training: memorize train, cannot touch val", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy (nats/token)")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


class ToyTraining:
    """Result object: history, parameter accounting, and the run directory."""

    def __init__(self, run_dir, history, summary, manifest):
        self.run_dir = run_dir
        self.history = history
        self._summary = summary
        self.manifest = manifest

    @property
    def train_losses(self):
        return self.history["train"]

    @property
    def val_losses(self):
        return self.history["val"]

    def summary(self):
        return self._summary


def run_toy_training(iters=DEFAULT_ITERS, seed=DEFAULT_SEED, rank=DEFAULT_RANK, batch_size=8,
                     *, mode="lora", lr=DEFAULT_LR, steps_per_eval=25, output_root=None):
    """Train the toy model with the from-scratch loop and record it like a real run."""
    if mode not in ("lora", "full"):
        raise ValueError("mode must be 'lora' or 'full'")
    for name, value in (("iters", iters), ("rank", rank), ("batch_size", batch_size),
                        ("steps_per_eval", steps_per_eval)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if lr <= 0:
        raise ValueError("lr must be positive")

    train, val = build_toy_data(seed)
    base = init_base(seed)
    adapter = init_adapter(rank, seed)
    params = adapter if mode == "lora" else base
    scale = 1.0

    # Parameter accounting: what does LoRA buy us?
    adapter_count = sum(a.size for a in adapter.values())
    full_count = sum(p.size for p in base.values())
    trainable_count = adapter_count if mode == "lora" else full_count

    state = [{"m": mx.zeros_like(p), "v": mx.zeros_like(p)}
             for _, p in tree_flatten(params)]
    np_rng = np.random.RandomState(seed + 2)
    history = {"train": [], "val": []}
    start = time.perf_counter()

    def loss_fn(p):
        return masked_cross_entropy(forward(base, p, mx.array(batch), scale), mx.array(batch))

    root = Path(output_root) if output_root else ARTIFACTS / "toy"
    directory, manifest = new_run(
        root, "toy-training", model="toy-lm", config={
            "mode": mode, "iters": iters, "seed": seed, "rank": rank,
            "batch_size": batch_size, "lr": lr, "steps_per_eval": steps_per_eval,
            "vocab": VOCAB, "context": CONTEXT, "answer": ANSWER,
        },
    )
    with record_failure(directory, manifest):
        console.print(intro_panel(mode, rank, adapter_count, full_count, trainable_count))
        for it in range(1, iters + 1):
            # One training step: sample a batch, differentiate, step.
            idx = np_rng.randint(0, len(train), size=batch_size)
            batch = train[idx]
            loss, grads = mx.value_and_grad(loss_fn)(params)
            params = adamw_step(params, grads, state, it, lr)
            if it % steps_per_eval == 0 or it == iters:
                train_loss, train_acc = evaluate(base, params, mx.array(train), scale)
                val_loss, val_acc = evaluate(base, params, mx.array(val), scale)
                history["train"].append({"iteration": it, "loss": train_loss})
                history["val"].append({"iteration": it, "loss": val_loss})
                console.print(f"iter {it:4d}  loss {float(loss.item()):7.4f}  "
                              f"train acc {train_acc:5.1%}  val acc {val_acc:5.1%}")
        summary = {
            "mode": mode, "iters": iters, "seed": seed, "rank": rank,
            "final_train_loss": history["train"][-1]["loss"],
            "final_val_loss": history["val"][-1]["loss"],
            "final_train_accuracy": train_acc, "final_val_accuracy": val_acc,
            "adapter_parameters": adapter_count,
            "base_parameters": full_count,
            "trainable_parameters": trainable_count,
            "adapter_percent_of_trainable_full": 100 * adapter_count / full_count if full_count else 0.0,
            "val_floor_at_uniform_guessing": float(np.log(VOCAB)),
            "seconds": round(time.perf_counter() - start, 2),
        }
        write_json(directory / "toy_history.json", {"summary": summary, "history": history})
        plot_toy_losses(history, directory / "loss_curve.png")
        manifest["summary"] = summary
        print_mapping_table()
        finish_run(root, directory, manifest)
    return ToyTraining(directory, history, summary, manifest)


def intro_panel(mode, rank, adapter_count, full_count, trainable_count):
    from rich.panel import Panel
    if mode == "lora":
        line = (f"[bold cyan]Toy LoRA training[/bold cyan]  rank {rank}\n"
                f"Trainable: {adapter_count} adapter parameters of {full_count} full-finetune "
                f"parameters ({100 * adapter_count / full_count:.1f}%). "
                f"Base weights are frozen and never differentiated.")
    else:
        line = (f"[bold cyan]Toy full fine-tuning[/bold cyan]\n"
                f"Trainable: all {trainable_count} parameters. Compare with the LoRA run: "
                f"same task, {adapter_count} vs {full_count} numbers to update.")
    return Panel(line, border_style="cyan")


def print_mapping_table():
    table = Table(title="Toy loop -> main.py train settings", header_style="bold magenta")
    for column in ("This toy loop", "Real training flag", "What it controls"):
        table.add_column(column)
    for toy, flag, meaning in (
        ("logits only for answer positions", "mask_prompt: true", "which tokens the loss scores"),
        ("A: r x d_in, B: d_out x r, B starts at 0", "lora_parameters.rank / .scale", "the low-rank update"),
        ("adamw_step(...)", "optimizer: adamw, learning_rate", "how gradients become updates"),
        ("sample `batch_size` rows per step", "batch_size", "examples seen per step"),
        ("loop over `iters` steps", "iters", "examples seen = iters x batch_size"),
        ("evaluate every `steps_per_eval`", "steps_per_eval", "how often val loss is scored"),
        ("val set shares no answers with train", "valid.jsonl (held-out wording)", "what generalization means"),
    ):
        table.add_row(toy, flag, meaning)
    console.print(table)


if __name__ == "__main__":
    run_toy_training()
