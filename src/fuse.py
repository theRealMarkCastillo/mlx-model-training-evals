"""Fuse an adapter into a new model directory and record its provenance."""

import subprocess
import sys
from pathlib import Path

from src.runs import (
    adapter_identity,
    directory_identity,
    finish_run,
    new_run,
    record_failure,
    resolve_adapter_source,
    write_json,
)


def fuse_model(model_name, adapter_path, save_path):
    adapter_files = adapter_identity(adapter_path)
    source, identity = resolve_adapter_source(model_name, adapter_path)
    root = Path(save_path)
    directory, manifest = new_run(root, 'fusion', model=model_name, model_source=identity, adapter=adapter_files)
    with record_failure(directory, manifest):
        destination = directory / 'model'
        subprocess.run([
            sys.executable, '-m', 'mlx_lm', 'fuse', '--model', source,
            '--adapter-path', adapter_path, '--save-path', str(destination),
        ], check=True)
        manifest['fused_model'] = directory_identity(destination)
        manifest['model_path'] = str(destination)
        manifest['quality_evaluated'] = False
        finish_run(root, directory, manifest)
        write_json(root / 'latest.json', {'path': str(destination), 'run_id': manifest['run_id']})
    return str(destination)
