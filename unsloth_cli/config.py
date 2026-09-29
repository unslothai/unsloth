# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
from typing import Literal, Optional, List

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class DataConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    dataset: Optional[str] = None
    local_dataset: Optional[List[str]] = None
    format_type: Literal["auto", "alpaca", "chatml", "sharegpt"] = "auto"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    training_type: Literal["lora", "full"] = "lora"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    output_dir: Path = Path("./outputs")
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 0
    save_steps: int = 0
    weight_decay: float = 0.01
    random_seed: int = 3407
    packing: bool = False
    train_on_completions: bool = False
    gradient_checkpointing: Literal["unsloth", "true", "none"] = "unsloth"


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    vision_all_linear: bool = False
    use_rslora: bool = False
    use_loftq: bool = False
    use_dora: bool = False
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enable_wandb: bool = False
    wandb_project: str = "unsloth-training"
    wandb_token: Optional[str] = None
    enable_tensorboard: bool = False
    tensorboard_dir: str = "runs"
    hf_token: Optional[str] = None


class Config(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    model: Optional[str] = None
    data: DataConfig = Field(default_factory = DataConfig)
    training: TrainingConfig = Field(default_factory = TrainingConfig)
    lora: LoraConfig = Field(default_factory = LoraConfig)
    logging: LoggingConfig = Field(default_factory = LoggingConfig)

    def apply_overrides(self, **kwargs):
        """Apply CLI overrides by matching arg names to config fields."""
        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                for section in (self.data, self.training, self.lora, self.logging):
                    if hasattr(section, key):
                        setattr(section, key, value)
                        break

    def model_kwargs(self, use_lora: bool, is_vision: bool) -> dict:
        """Return kwargs for trainer.prepare_model_for_training()."""
        if use_lora and is_vision:
            # Vision models expect a string (e.g. "all-linear"); None uses trainer defaults
            target_modules = "all-linear" if self.lora.vision_all_linear else None
        else:
            parsed = [
                m.strip() for m in str(self.lora.target_modules).split(",") if m and m.strip()
            ]
            target_modules = parsed or None

        return {
            "use_lora": use_lora,
            "finetune_vision_layers": self.lora.finetune_vision_layers,
            "finetune_language_layers": self.lora.finetune_language_layers,
            "finetune_attention_modules": self.lora.finetune_attention_modules,
            "finetune_mlp_modules": self.lora.finetune_mlp_modules,
            "target_modules": target_modules,
            "lora_r": self.lora.lora_r,
            "lora_alpha": self.lora.lora_alpha,
            "lora_dropout": self.lora.lora_dropout,
            "use_gradient_checkpointing": self.training.gradient_checkpointing,
            "use_rslora": self.lora.use_rslora,
            "use_loftq": self.lora.use_loftq,
            "use_dora": self.lora.use_dora,
        }

    def training_kwargs(self) -> dict:
        """Return kwargs for trainer.start_training()."""
        return {
            "output_dir": str(self.training.output_dir),
            "num_epochs": self.training.num_epochs,
            "learning_rate": self.training.learning_rate,
            "batch_size": self.training.batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "warmup_steps": self.training.warmup_steps,
            "max_steps": self.training.max_steps,
            "save_steps": self.training.save_steps,
            "weight_decay": self.training.weight_decay,
            "random_seed": self.training.random_seed,
            "packing": self.training.packing,
            "train_on_completions": self.training.train_on_completions,
            "max_seq_length": self.training.max_seq_length,
            "enable_wandb": self.logging.enable_wandb,
            "wandb_project": self.logging.wandb_project,
            "wandb_token": self.logging.wandb_token,
            "enable_tensorboard": self.logging.enable_tensorboard,
            "tensorboard_dir": self.logging.tensorboard_dir,
        }


class ConfigError(ValueError):
    pass


def _section_for_field(name: str) -> Optional[str]:
    for section, field_info in Config.model_fields.items():
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if name in annotation.model_fields:
                return section
    return None


def _describe_unknown_key(loc: tuple) -> str:
    key = str(loc[-1])
    parent = str(loc[-2]) if len(loc) > 1 else None
    where = f"in section '{parent}'" if parent else "at the top level"
    canonical = key.replace("-", "_")
    section = _section_for_field(canonical)
    subject = "it" if canonical == key else f"'{canonical}'"

    if section is not None and section == parent:
        return f"unknown key '{key}' {where}: did you mean '{canonical}'?"
    if section is not None:
        return f"unknown key '{key}' {where}: {subject} belongs under '{section}:'"
    if canonical in Config.model_fields:
        if parent is None:
            return f"unknown key '{key}' {where}: did you mean '{canonical}'?"
        return f"unknown key '{key}' {where}: {subject} belongs at the top level"
    return f"unknown key '{key}' {where}"


def _config_error_message(path: Path, error: ValidationError) -> str:
    lines = [f"Invalid config file: {path}"]
    for err in error.errors():
        loc = tuple(err.get("loc") or ())
        if err.get("type") == "extra_forbidden" and loc:
            lines.append(f"  - {_describe_unknown_key(loc)}")
        elif not loc:
            got = type(err.get("input")).__name__
            article = "an" if got[:1].lower() in "aeiou" else "a"
            lines.append(
                f"  - the top level must be a mapping of keys and sections, not {article} {got}"
            )
        else:
            field = ".".join(str(part) for part in loc) or "config"
            lines.append(f"  - {field}: {err.get('msg', 'invalid value')}")
    return "\n".join(lines)


def load_config(path: Optional[Path]) -> Config:
    """Load config from YAML/JSON file, or return defaults if no path given."""
    if not path:
        return Config()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # utf-8-sig: drops a Notepad BOM, identical to utf-8 when there is none.
    try:
        text = path.read_text(encoding = "utf-8-sig")
    except UnicodeDecodeError as error:
        raise ConfigError(
            f"Could not read config file: {path}\n"
            f"  - {error}\n"
            f"  - config files must be UTF-8; re-save it as UTF-8 and try again"
        ) from None
    except OSError as error:
        raise ConfigError(f"Could not read config file: {path}\n  - {error}") from None

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ConfigError(f"Could not parse config file: {path}\n  - {error}") from None
    else:
        import json
        try:
            data = json.loads(text.strip() or "{}")
        except json.JSONDecodeError as error:
            hint = (
                ""
                if path.suffix.lower() == ".json"
                else (
                    f"\n  - parsed as JSON because of the '{path.suffix}' extension; "
                    f"name it .yaml or .yml for YAML"
                )
            )
            raise ConfigError(f"Could not parse config file: {path}\n  - {error}{hint}") from None

    if data is None:
        data = {}

    try:
        return Config.model_validate(data)
    except ValidationError as error:
        raise ConfigError(_config_error_message(path, error)) from None
