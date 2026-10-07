"""Fuse an adapter into a new model directory and record its provenance."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import add_preset_argument, resolve_model_paths
from src.runs import resolve_adapter_source, directory_identity, adapter_identity, new_run, finish_run, write_json


def fuse_model(model_name, adapter_path, save_path):
    adapter_files = adapter_identity(adapter_path)
    source, identity = resolve_adapter_source(model_name, adapter_path)
    root = Path(save_path)
    directory, manifest = new_run(root, 'fusion', model=model_name, model_source=identity, adapter=adapter_files)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    add_preset_argument(selection)
    selection.add_argument('--model')
    parser.add_argument('--adapter')
    parser.add_argument('--save-path')
    args = parser.parse_args()
    model, adapter, output = resolve_model_paths(args.preset, args.model, args.adapter)
    print(fuse_model(model, adapter, args.save_path or str(output / 'fused_model')))
