"""Run provenance and immutable artifact directories with small latest pointers."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def directory_identity(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return [file_identity(p) for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in ('.json', '.safetensors', '.model', '.jinja', '.txt', '.tiktoken', '.py')]


def resolve_source(model):
    """Resolve a remote repository once, then load all variants from that snapshot."""
    path = Path(model)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(model, allow_patterns=[
            '*.json', '*.safetensors', '*.model', '*.jinja', '*.txt', '*.tiktoken', '*.py',
        ]))
    path = path.absolute()
    if path.parent.name == 'snapshots':
        identity = {"requested": model, "path": str(path), "revision": path.name}
    else:
        identity = {"requested": model, "path": str(path), "files": directory_identity(path)}
    return str(path), identity


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def new_run(root, kind, **inputs):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
    directory = Path(root).resolve() / 'runs' / f'{kind}-{run_id}'
    directory.mkdir(parents=True)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    manifest = {
        'schema_version': 2, 'run_id': run_id, 'kind': kind, 'run_dir': str(directory),
        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'started',
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {p: version(p) for p in ('mlx', 'mlx-lm', 'pydantic', 'transformers')},
        'git_commit': git.stdout.strip() if git.returncode == 0 else None,
        'git_dirty': bool(status.stdout.strip()),
        'source_files': [file_identity(Path('main.py')), file_identity(Path('scripts/build_notebook.py'))] + [file_identity(p) for folder in ('src', 'data', 'config')
                         for p in sorted(Path(folder).glob('*.py' if folder != 'config' else '*.yaml'))],
        **inputs,
    }
    write_json(directory / 'manifest.json', manifest)
    return directory, manifest


def finish_run(root, directory, manifest):
    manifest['status'] = 'complete'
    write_json(Path(directory) / 'manifest.json', manifest)
    write_json(Path(root) / f"latest_{manifest['kind']}.json", manifest)


def latest_path(path):
    """A configured adapter/fusion directory can point at its latest completed run."""
    pointer = Path(path) / 'latest.json'
    if pointer.is_file():
        target = Path(json.loads(pointer.read_text())['path'])
        if not target.is_dir():
            raise FileNotFoundError(f'Latest artifact no longer exists: {target}')
        return str(target)
    return str(path)


def resolve_adapter_source(model, adapter):
    """Reuse the exact base snapshot recorded by our training runner, if available."""
    manifest_path = Path(adapter).parent / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('kind') == 'training':
            if manifest['model'] != model:
                raise ValueError(f"Adapter was trained for {manifest['model']}, not {model}")
            source = manifest['model_source']
            if not Path(source['path']).is_dir():
                raise FileNotFoundError(f"Recorded base snapshot is missing: {source['path']}")
            if 'files' in source and directory_identity(source['path']) != source['files']:
                raise ValueError('Recorded local base model files changed after training')
            return source['path'], source
    config_path = Path(adapter) / 'adapter_config.json'
    if config_path.is_file():
        configured_model = json.loads(config_path.read_text()).get('model')
        if configured_model and configured_model != model:
            raise ValueError(f'Adapter config specifies {configured_model}; supply that exact --model')
    return resolve_source(model)


def adapter_identity(path):
    path = Path(path)
    for name in ('adapter_config.json', 'adapters.safetensors'):
        if not (path / name).is_file():
            raise FileNotFoundError(f'Missing trained adapter file: {path / name}')
    return directory_identity(path)
