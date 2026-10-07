"""Measure the chat workload using streaming token metadata and wall-clock latency."""

from pathlib import Path
from statistics import mean

import mlx.core as mx
import mlx_lm
from rich.console import Console
from rich.table import Table

from src.dataset import positive_int
from src.fuse import fuse_model
from src.inference import generate_response
from src.models import resolve_model_paths
from src.runs import (adapter_identity, finish_run, latest_path, new_run, record_failure,
                      resolve_adapter_source, resolve_source, write_json)
from src.schema import SYSTEM_PROMPT

console = Console()
DEFAULT_PROMPT = 'Deploy auth-api version v3.12.0 to production with 5 replicas and notify #deployments.'


def benchmark_generation(model, tokenizer, prompt, max_tokens=120, warmup=2, runs=5):
    positive_int(runs)
    positive_int(max_tokens)
    if warmup < 0:
        raise ValueError('warmup must be nonnegative')
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
    for _ in range(warmup):
        generate_response(model, tokenizer, messages, max_tokens)
    mx.reset_peak_memory()
    records = [generate_response(model, tokenizer, messages, max_tokens) for _ in range(runs)]
    return {
        **{f'avg_{key}': mean(r[key] for r in records) for key in (
            'ttft_seconds', 'latency_seconds', 'prefill_tokens_per_sec',
            'decode_tokens_per_sec', 'end_to_end_tokens_per_sec', 'output_tokens', 'prompt_tokens',
        )},
        'peak_metal_memory_mb': mx.get_peak_memory() / (1024**2),
        'active_metal_memory_mb': mx.get_active_memory() / (1024**2),
        'runs': records,
    }


def run_benchmark_suite(
    model_name=None, adapter_path=None, test_prompt=DEFAULT_PROMPT,
    *, preset=None, output_dir=None, fused_path=None, fuse=False, runs=5, warmup=2, max_tokens=120,
    quality_samples=None,
):
    positive_int(runs)
    positive_int(max_tokens)
    if quality_samples is not None:
        positive_int(quality_samples)
    if warmup < 0:
        raise ValueError('warmup must be nonnegative')
    if fuse and fused_path:
        raise ValueError('Choose either fuse or fused_path')
    model_name, adapter_path, root = resolve_model_paths(preset, model_name, adapter_path, output_dir)
    adapter_files = adapter_identity(adapter_path)
    source, identity = resolve_adapter_source(model_name, adapter_path)
    if fuse:
        fused_path = fuse_model(model_name, adapter_path, str(root / 'fused_model'))
    directory, manifest = new_run(
        root, 'benchmark', model=model_name, model_source=identity, adapter=adapter_files,
        messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': test_prompt}],
        generation={'temperature': 0.0, 'max_tokens': max_tokens, 'warmup': warmup, 'runs': runs},
        token_count_convention='MLX generation_tokens, including terminal EOS when generated',
    )
    with record_failure(directory, manifest):
        variants = [('base', source, None), ('lora', source, adapter_path)]
        if fused_path:
            fused_source, fused_identity = resolve_source(latest_path(fused_path))
            manifest['fused_model'] = fused_identity
            variants.append(('fused', fused_source, None))
        report = {'model': model_name, 'adapter': adapter_path, 'run_dir': str(directory)}
        for name, path, adapter in variants:
            model, tokenizer = mlx_lm.load(path, **({'adapter_path': adapter} if adapter else {}))
            try:
                report[f'{name}_stats'] = benchmark_generation(model, tokenizer, test_prompt, max_tokens, warmup, runs)
            finally:
                del model
                mx.clear_cache()
        table = Table(title='Chat inference benchmark (means over measured runs)')
        table.add_column('Metric')
        for name, _, _ in variants:
            table.add_column(name, justify='right')
        for key in ('avg_ttft_seconds', 'avg_latency_seconds', 'avg_prefill_tokens_per_sec',
                    'avg_decode_tokens_per_sec', 'avg_end_to_end_tokens_per_sec', 'avg_output_tokens', 'peak_metal_memory_mb'):
            table.add_row(key, *(f"{report[f'{name}_stats'][key]:.3f}" for name, _, _ in variants))
        console.print(table)
        if fused_path:
            from src.evaluate import run_comprehensive_evaluation
            quality = run_comprehensive_evaluation(
                model_name, adapter_path, num_eval_samples=quality_samples,
                output_dir=str(root), fused_path=fused_path, variants=('base', 'lora'),
            )
            manifest['quality_evaluation'] = str(Path(quality['run_dir']) / 'eval_results.json')
        report['manifest'] = {**manifest, 'status': 'complete'}
        write_json(directory / 'benchmark_results.json', report)
        finish_run(root, directory, manifest)
    return report
