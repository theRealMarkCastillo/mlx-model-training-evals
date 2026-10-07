"""Qwen2.5 presets: the only settings that change with model size."""

from dataclasses import dataclass
from pathlib import Path

from src.runs import REPO_ROOT, latest_path

ARTIFACTS = REPO_ROOT / "artifacts"


@dataclass(frozen=True)
class ModelPreset:
    size: str
    batch_size: int
    num_layers: int
    grad_checkpoint: bool

    @property
    def model(self) -> str:
        return f"mlx-community/Qwen2.5-{self.size.upper()}-Instruct-4bit"

    @property
    def output_dir(self) -> Path:
        return ARTIFACTS / f"qwen2.5-{self.size}"

    @property
    def adapter_path(self) -> str:
        return str(self.output_dir / "adapters")

    @property
    def fused_path(self) -> str:
        return str(self.output_dir / "fused_model")

    def overrides(self) -> dict:
        """Merged over config/base.yaml to form this preset's training config."""
        return {
            "model": self.model, "batch_size": self.batch_size, "num_layers": self.num_layers,
            "grad_checkpoint": self.grad_checkpoint, "adapter_path": self.adapter_path,
        }


PRESETS = {
    size: ModelPreset(size, batch, layers, checkpoint)
    for size, batch, layers, checkpoint in (
        ("3b", 4, 16, False),
        ("7b", 2, 16, True),
        ("14b", 1, 16, True),
        ("32b", 1, 8, True),
        ("72b", 1, 4, True),
    )
}
DEFAULT_PRESET = "3b"


def add_preset_argument(parser):
    parser.add_argument("--preset", choices=PRESETS, help=f"Qwen2.5 Instruct 4-bit model size (default: {DEFAULT_PRESET})")


def resolve_model_paths(preset=None, model=None, adapter=None, output_dir=None, *, need_adapter=True):
    """Return (model, adapter directory or None, report root).

    A preset's default adapter must come from a completed training run's
    latest.json pointer. An explicit --adapter may be a pointer directory or an
    actual adapter directory.
    """
    profile = PRESETS[preset or DEFAULT_PRESET]
    custom = model is not None and model != profile.model
    if preset and custom:
        raise ValueError("Use either --preset or a different --model, not both.")
    if custom and need_adapter and adapter is None:
        raise ValueError("An explicit --adapter is required for a custom --model")
    if adapter is not None:
        adapter = latest_path(adapter)
    elif need_adapter:
        adapter = latest_path(profile.adapter_path, required=True)
    if output_dir is not None:
        root = Path(output_dir)
    else:
        root = ARTIFACTS / "custom" if custom else profile.output_dir
    return model or profile.model, adapter, root
