"""Shared Qwen presets and artifact paths for the tutorial workflow."""

from dataclasses import dataclass
from pathlib import Path
from src.runs import latest_path


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
        return Path("artifacts") if self.size == "3b" else Path("artifacts") / f"qwen2.5-{self.size}"

    @property
    def adapter_path(self) -> str:
        return str(self.output_dir / "adapters")

    @property
    def fused_path(self) -> str:
        return str(self.output_dir / "fused_model")

    @property
    def config_path(self) -> str:
        return "config/lora_config.yaml" if self.size == "3b" else f"config/qwen2.5-{self.size}.yaml"


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


def add_preset_argument(parser):
    parser.add_argument("--preset", choices=PRESETS, help="Qwen2.5 Instruct 4-bit model size (default: 3b)")


def resolve_model_paths(preset=None, model=None, adapter=None, output_dir=None):
    profile = PRESETS[preset or "3b"]
    if preset and model and model != profile.model:
        raise ValueError("Use either --preset or a different --model, not both.")
    if model and model != profile.model and adapter is None:
        raise ValueError("An explicit --adapter is required for a custom --model")
    return (
        model or profile.model,
        latest_path(adapter or profile.adapter_path),
        Path(output_dir) if output_dir is not None else profile.output_dir,
    )
