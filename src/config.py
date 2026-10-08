"""Training configuration: config/base.yaml + preset overrides, validated by Pydantic.

Validation happens before any model is downloaded or loaded, so a typo costs
milliseconds rather than a multi-gigabyte download.
"""

from typing import Literal

import mlx_lm.lora as lora
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, field_validator, model_validator

from src.models import DEFAULT_PRESET, PRESETS
from src.runs import repo_path

BASE_CONFIG = "config/base.yaml"
NESTED_LORA_KEYS = ("rank", "scale", "dropout")


class LoraParameters(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    rank: PositiveInt
    scale: PositiveFloat
    dropout: float = Field(ge=0, lt=1)
    keys: list[str] | None = None


class TrainingConfig(BaseModel):
    """The subset of MLX-LM options this workflow depends on, with strict types.

    Other MLX-LM options are allowed if MLX-LM knows them; unknown keys fail.
    """

    model_config = ConfigDict(strict=True, extra="allow")

    model: str = Field(min_length=1)
    data: str = Field(min_length=1)
    adapter_path: str = Field(min_length=1)
    fine_tune_type: Literal["lora"]
    mask_prompt: Literal[True]
    lora_parameters: LoraParameters
    learning_rate: PositiveFloat
    optimizer: Literal["adam", "adamw", "muon", "sgd", "adafactor"]
    iters: PositiveInt
    batch_size: PositiveInt
    max_seq_length: PositiveInt
    steps_per_report: PositiveInt
    steps_per_eval: PositiveInt
    save_every: PositiveInt
    num_layers: int
    val_batches: int
    grad_checkpoint: bool = False
    seed: int = 0

    @field_validator("model", "data", "adapter_path")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("must be nonempty")
        return value

    @field_validator("num_layers", "val_batches")
    @classmethod
    def positive_or_all(cls, value):
        if value == 0 or value < -1:
            raise ValueError("must be positive, or -1 for all")
        return value

    @model_validator(mode="after")
    def known_extras(self):
        unknown = set(self.model_extra or {}) - set(lora.CONFIG_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
        return self


def _load_yaml(path):
    with open(path) as source:
        config = lora.yaml.load(source, lora.yaml_loader)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")
    return config


def apply_overrides(config, overrides):
    """Apply flat overrides; rank/scale/dropout go into lora_parameters."""
    config = {**config, "lora_parameters": dict(config.get("lora_parameters") or {})}
    for key, value in (overrides or {}).items():
        if key in NESTED_LORA_KEYS:
            config["lora_parameters"][key] = value
        else:
            config[key] = value
    return config


def load_training_config(config_path=None, preset=None, overrides=None):
    """A standalone --config file, or base.yaml merged with a preset.

    Relative `data` and `adapter_path` values are resolved against the
    repository root (absolute paths pass through), in both branches, so a
    standalone config behaves the same no matter which directory you run from.
    """
    if config_path is not None and preset is not None:
        raise ValueError("Choose a config file or a preset, not both.")
    if config_path is not None:
        config = _load_yaml(config_path)
    else:
        config = {**_load_yaml(repo_path(BASE_CONFIG)), **PRESETS[preset or DEFAULT_PRESET].overrides()}
    # Resolve only the keys that exist; missing ones are reported by Pydantic.
    for key in ("data", "adapter_path"):
        if key in config:
            config[key] = str(repo_path(config[key]))
    config = apply_overrides(config, overrides)
    return TrainingConfig.model_validate(config).model_dump(exclude_none=True)
