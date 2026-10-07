"""Run provenance and immutable artifact directories with small latest pointers."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import traceback
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE_SUFFIXES = ('.json', '.safetensors', '.model', '.jinja', '.txt', '.tiktoken', '.py')


def repo_path(path):
    """Resolve repository-relative defaults independently of the current directory."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


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
    return [file_identity(p) for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in MODEL_FILE_SUFFIXES]


def resolve_source(model):
    """Resolve a remote repository once, then load all variants from that snapshot."""
    path = Path(model)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(model, allow_patterns=[f'*{s}' for s in MODEL_FILE_SUFFIXES]))
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


def _source_files():
    files = [REPO_ROOT / 'main.py', REPO_ROOT / 'scripts' / 'build_notebook.py']
    files += sorted((REPO_ROOT / 'src').glob('*.py')) + sorted((REPO_ROOT / 'config').glob('*.yaml'))
    return [file_identity(p) for p in files if p.is_file()]


def new_run(root, kind, **inputs):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
    directory = Path(root).resolve() / 'runs' / f'{kind}-{run_id}'
    directory.mkdir(parents=True)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO_ROOT)
    status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd=REPO_ROOT)
    manifest = {
        'schema_version': 3, 'run_id': run_id, 'kind': kind, 'run_dir': str(directory),
        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'started',
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {p: version(p) for p in ('mlx', 'mlx-lm', 'pydantic', 'transformers')},
        'git_commit': git.stdout.strip() if git.returncode == 0 else None,
        'git_dirty': bool(status.stdout.strip()),
        'source_files': _source_files(),
        **inputs,
    }
    write_json(directory / 'manifest.json', manifest)
    return directory, manifest


@contextmanager
def record_failure(directory, manifest):
    """Mark the run failed, with its traceback, if the body raises."""
    try:
        yield
    except BaseException as exc:
        manifest.update(
            status='failed', error=f'{type(exc).__name__}: {exc}',
            traceback=traceback.format_exc(),
            failed_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json(Path(directory) / 'manifest.json', manifest)
        raise


def finish_run(root, directory, manifest):
    manifest['status'] = 'complete'
    write_json(Path(directory) / 'manifest.json', manifest)
    write_json(Path(root) / f"latest_{manifest['kind']}.json", manifest)


def latest_path(path, *, required=False):
    """Follow `path/latest.json` to the most recent completed run.

    With required=True, a missing pointer is an error instead of falling back to
    `path` itself. Preset defaults use this so stale weights left in a configured
    directory are never picked up silently.
    """
    pointer = Path(path) / 'latest.json'
    if pointer.is_file():
        target = Path(json.loads(pointer.read_text())['path'])
        if not target.is_dir():
            raise FileNotFoundError(f'Latest artifact no longer exists: {target}')
        return str(target)
    if required:
        raise FileNotFoundError(
            f'No completed run recorded at {pointer}. Run the producing step first '
            '(train or fuse), or pass an explicit path.'
        )
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
