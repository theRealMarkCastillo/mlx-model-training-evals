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
    learning_rate: float | None = None

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
        cfg = {
            "model": self.model, "batch_size": self.batch_size, "num_layers": self.num_layers,
            "grad_checkpoint": self.grad_checkpoint, "adapter_path": self.adapter_path,
        }
        if self.learning_rate is not None:
            cfg["learning_rate"] = self.learning_rate
        return cfg


PRESETS = {
    size: ModelPreset(size, batch, layers, checkpoint, lr)
    for size, batch, layers, checkpoint, lr in (
        ("3b", 4, 16, False, None),
        ("7b", 2, 16, True, None),
        ("14b", 1, 16, True, None),
        ("32b", 1, 8, True, None),
        ("72b", 1, 4, True, 1e-5),
    )
}
DEFAULT_PRESET = "3b"

RAM_TIERS = ("16gb", "32gb", "64gb", "96gb")

MIN_RAM_REQUIRED = {
    "3b": 8,
    "7b": 16,
    "14b": 24,
    "32b": 36,
    "72b": 64,
}

# Optimal (batch_size, num_layers, grad_checkpoint) per (preset, ram_tier).
RAM_PROFILES: dict[str, dict[str, dict[str, int | bool]]] = {
    "16gb": {
        "3b": {"batch_size": 4, "num_layers": 16, "grad_checkpoint": True},
        "7b": {"batch_size": 1, "num_layers": 8, "grad_checkpoint": True},
    },
    "32gb": {
        "3b": {"batch_size": 4, "num_layers": 16, "grad_checkpoint": False},
        "7b": {"batch_size": 2, "num_layers": 16, "grad_checkpoint": True},
        "14b": {"batch_size": 1, "num_layers": 16, "grad_checkpoint": True},
        "32b": {"batch_size": 1, "num_layers": 8, "grad_checkpoint": True},
    },
    "64gb": {
        "3b": {"batch_size": 8, "num_layers": -1, "grad_checkpoint": False},
        "7b": {"batch_size": 4, "num_layers": 28, "grad_checkpoint": False},
        "14b": {"batch_size": 2, "num_layers": 24, "grad_checkpoint": True},
        "32b": {"batch_size": 1, "num_layers": 16, "grad_checkpoint": True},
        "72b": {"batch_size": 1, "num_layers": 4, "grad_checkpoint": True},
    },
    "96gb": {
        "3b": {"batch_size": 8, "num_layers": -1, "grad_checkpoint": False},
        "7b": {"batch_size": 4, "num_layers": -1, "grad_checkpoint": False},
        "14b": {"batch_size": 2, "num_layers": 32, "grad_checkpoint": True},
        "32b": {"batch_size": 1, "num_layers": 32, "grad_checkpoint": True},
        "72b": {"batch_size": 1, "num_layers": 8, "grad_checkpoint": True},
    },
}


def get_system_ram_gb() -> int:
    """Return total physical system RAM in GiB (e.g. 16, 32, 64, 96, 128)."""
    try:
        import os
        return int(round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)))
    except Exception:
        return 32


def ram_to_tier(ram_gb: int) -> str:
    """Map physical RAM to the closest supported tier."""
    if ram_gb < 24:
        return "16gb"
    if ram_gb < 54:
        return "32gb"
    if ram_gb < 80:
        return "64gb"
    return "96gb"


def optimal_overrides(preset: str, tier: str | None = None) -> dict[str, int | bool]:
    """Return optimal (batch_size, num_layers, grad_checkpoint) for preset on the requested or detected RAM tier."""
    preset = preset.lower()
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Expected one of {list(PRESETS)}")
    if tier is None or tier == "auto":
        tier = ram_to_tier(get_system_ram_gb())
    else:
        tier = tier.lower()
        if not tier.endswith("gb"):
            tier = f"{tier}gb"
        if tier not in RAM_PROFILES:
            raise ValueError(f"Unknown RAM tier {tier!r}. Expected one of {list(RAM_PROFILES)}")

    tier_config = RAM_PROFILES[tier].get(preset)
    if tier_config is None:
        min_ram = MIN_RAM_REQUIRED.get(preset, 32)
        raise ValueError(
            f"Preset '{preset}' requires at least {min_ram}GB RAM, but tier '{tier}' was selected. "
            f"Use a larger RAM tier or a smaller preset."
        )
    return dict(tier_config)


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
