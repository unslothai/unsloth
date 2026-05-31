# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
from typing import Literal, Optional, List

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset: Optional[str] = None
    local_dataset: Optional[List[str]] = None
    format_type: Literal["auto", "alpaca", "chatml", "sharegpt"] = "auto"


class TrainingConfig(BaseModel):
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
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    vision_all_linear: bool = False
    use_rslora: bool = False
    use_loftq: bool = False
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


class LoggingConfig(BaseModel):
    enable_wandb: bool = False
    wandb_project: str = "unsloth-training"
    wandb_token: Optional[str] = None
    enable_tensorboard: bool = False
    tensorboard_dir: str = "runs"
    hf_token: Optional[str] = None


class Config(BaseModel):
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
        # Determine target modules based on model type
        if use_lora and is_vision:
            # Vision models expect a string (e.g., "all-linear"); fall back to None to use trainer defaults
            target_modules = "all-linear" if self.lora.vision_all_linear else None
        else:
            parsed = [
                m.strip()
                for m in str(self.lora.target_modules).split(",")
                if m and m.strip()
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


def load_config(path: Optional[Path]) -> Config:
    """Load config from YAML/JSON file, or return defaults if no path given."""
    if not path:
        return Config()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding = "utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        import json

        data = json.loads(text or "{}")

    return Config(**data)
