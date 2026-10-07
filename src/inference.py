"""Shared chat formatting and measured greedy generation."""

import time
import mlx_lm
from mlx_lm.sample_utils import make_sampler


def generate_response(model, tokenizer, messages, max_tokens=150):
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    start = time.perf_counter()
    first_token = None
    chunks = []
    last = None
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0),
    ):
        if first_token is None:
            first_token = time.perf_counter() - start
        chunks.append(response.text)
        last = response
    elapsed = time.perf_counter() - start
    if last is None:
        raise RuntimeError("Generation yielded no token metadata")
    return {
        "raw_output": ''.join(chunks),
        "output_tokens": last.generation_tokens,
        "prompt_tokens": last.prompt_tokens,
        "ttft_seconds": first_token,
        "latency_seconds": elapsed,
        "prefill_tokens_per_sec": last.prompt_tps,
        "decode_tokens_per_sec": last.generation_tps,
        "end_to_end_tokens_per_sec": last.generation_tokens / elapsed if elapsed else 0.0,
        "finish_reason": last.finish_reason,
    }
