"""Make two invisible parts of training visible: the loss mask and LoRA's size.

`loss_mask_tokens` shows exactly which tokens `mask_prompt: true` scores, using
MLX-LM's own ChatDataset so the picture matches training. `parameter_summary`
counts adapter parameters against the logical size of the (quantized) base model.
"""

import html

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.datasets import ChatDataset
from mlx_lm.tuner.lora import LoRALinear


def loss_mask_tokens(tokenizer, record):
    """[(token_text, scored)] for one chat record, exactly as training sees it.

    A token is scored when the model is trained to predict it: positions from
    the first assistant token through the end, including the end-of-turn marker.
    """
    tokens, offset = ChatDataset([record], tokenizer, mask_prompt=True).process(record)
    return [(tokenizer.decode([token]), index >= offset) for index, token in enumerate(tokens)]


def mask_counts(pairs):
    scored = sum(scored for _, scored in pairs)
    return {"total_tokens": len(pairs), "scored_tokens": scored, "masked_tokens": len(pairs) - scored}


def render_mask_html(pairs):
    """Notebook rendering: masked tokens grey, scored tokens highlighted."""
    spans = []
    for text, scored in pairs:
        style = ("background:#1b9e7733;border-bottom:2px solid #1b9e77" if scored
                 else "color:#888")
        spans.append(f'<span style="{style}">{html.escape(text)}</span>')
    counts = mask_counts(pairs)
    legend = (f"<p><b>{counts['scored_tokens']}</b> of {counts['total_tokens']} tokens contribute to the loss "
              f"(highlighted). The system prompt and request are context only.</p>")
    return legend + '<pre style="white-space:pre-wrap;font-size:12px;line-height:1.6">' + "".join(spans) + "</pre>"


def render_mask_rich(pairs):
    from rich.text import Text
    text = Text()
    for token, scored in pairs:
        text.append(token, style="bold black on green" if scored else "dim")
    return text


def _logical_size(module):
    """Parameters a module represents, undoing 4-bit packing of quantized weights."""
    if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        rows, packed = module.weight.shape
        size = rows * packed * 32 // module.bits
        return size + (module.bias.size if "bias" in module else 0)
    if isinstance(module, LoRALinear):
        return 0  # its frozen `linear` child and LoRA matrices are counted separately
    return sum(v.size for _, v in tree_flatten(module.parameters()))


def parameter_summary(model):
    """Count base and LoRA parameters and list the adapted projections.

    Base size is the logical parameter count (what a full-precision copy would
    hold), so the percentage reflects how small the adapter is relative to the model.
    """
    base = adapter = 0
    adapted = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            count = module.lora_a.size + module.lora_b.size
            adapter += count
            projection = name.rsplit(".", 1)[-1]
            entry = adapted.setdefault(projection, {"layers": 0, "rank": module.lora_a.shape[1], "parameters": 0})
            entry["layers"] += 1
            entry["parameters"] += count
        elif not module.children() or isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding, nn.Linear, nn.Embedding)):
            base += _logical_size(module)
    return {
        "base_parameters": base,
        "adapter_parameters": adapter,
        "adapter_percent": 100 * adapter / base if base else 0.0,
        "adapter_megabytes_fp16": adapter * 2 / 1024**2,
        "adapted_projections": adapted,
    }
