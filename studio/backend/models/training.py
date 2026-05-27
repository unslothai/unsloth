# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Training API
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Any, Optional, List, Dict, Literal


_MAX_BATCH_SIZE = 4096
_MAX_GRAD_ACCUM = 4096
_MAX_STEPS = 1_000_000
_MAX_EPOCHS = 1000
# 2M is a sanity cap; host RAM runs out long before this.
_MAX_SEQ_LENGTH = 2_000_000
_MAX_LR_VALUE = 1.0
_MAX_LORA_R = 16_384
_MAX_LORA_ALPHA = 32_768


def _parse_lr(v: Any) -> float:
    """Parse learning_rate as a positive float strictly below _MAX_LR_VALUE."""
    if v is None:
        raise ValueError("learning_rate is required")
    if isinstance(v, bool):
        raise ValueError("learning_rate must be a number, not a bool")
    try:
        lr = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"learning_rate must be parseable as float (got {v!r})")
    if not (lr > 0.0):
        raise ValueError(
            f"learning_rate must be > 0 (got {lr!r}); " "typical range is 1e-6 .. 1e-3"
        )
    if lr >= _MAX_LR_VALUE:
        raise ValueError(
            f"learning_rate must be < 1.0 (got {lr!r}); "
            "values that large always diverge training"
        )
    return lr


class TrainingStartRequest(BaseModel):
    """Request schema for starting training"""

    # Model parameters
    model_name: str = Field(
        ..., description = "Model identifier (e.g., 'unsloth/llama-3-8b-bnb-4bit')"
    )
    training_type: Literal["LoRA/QLoRA", "Full Finetuning", "Continued Pretraining"] = (
        Field(
            ...,
            description = "Training type: 'LoRA/QLoRA', 'Full Finetuning', or 'Continued Pretraining'",
        )
    )
    hf_token: Optional[str] = Field(None, description = "HuggingFace token")
    load_in_4bit: bool = Field(True, description = "Load model in 4-bit quantization")
    max_seq_length: int = Field(2048, description = "Maximum sequence length")
    trust_remote_code: bool = Field(
        False,
        description = "Allow loading models with custom code (e.g. NVIDIA Nemotron). Only enable for repos you trust.",
    )

    # Dataset parameters
    hf_dataset: Optional[str] = Field(
        None, description = "HuggingFace dataset identifier"
    )
    local_datasets: List[str] = Field(
        default_factory = list, description = "List of local dataset paths"
    )
    local_eval_datasets: List[str] = Field(
        default_factory = list, description = "List of local eval dataset paths"
    )
    format_type: str = Field(..., description = "Dataset format type")
    subset: Optional[str] = None
    train_split: Optional[str] = Field("train", description = "Training split name")
    eval_split: Optional[str] = Field(
        None, description = "Eval split name. None = auto-detect"
    )
    eval_steps: float = Field(
        0.00, description = "Fraction of total steps between evals (0-1)"
    )
    dataset_slice_start: Optional[int] = Field(
        None, description = "Inclusive start row index for dataset slicing"
    )
    dataset_slice_end: Optional[int] = Field(
        None, description = "Inclusive end row index for dataset slicing"
    )

    @model_validator(mode = "before")
    @classmethod
    def _compat_split(cls, values: Any) -> Any:
        """Accept legacy 'split' field as alias for 'train_split'."""
        if isinstance(values, dict) and "split" in values:
            values.setdefault("train_split", values.pop("split"))
        return values

    @field_validator("learning_rate", mode = "before")
    @classmethod
    def _check_learning_rate(cls, v):
        # Stringify because downstream call sites float() it themselves.
        lr = _parse_lr(v)
        return str(lr)

    @field_validator("batch_size")
    @classmethod
    def _check_batch_size(cls, v: int) -> int:
        if v is None:
            raise ValueError("batch_size is required")
        if v < 1 or v > _MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be in [1, {_MAX_BATCH_SIZE}] (got {v!r})"
            )
        return v

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def _check_grad_accum(cls, v: int) -> int:
        if v is None:
            return 1
        if v < 1 or v > _MAX_GRAD_ACCUM:
            raise ValueError(
                f"gradient_accumulation_steps must be in [1, {_MAX_GRAD_ACCUM}] "
                f"(got {v!r})"
            )
        return v

    @field_validator("num_epochs")
    @classmethod
    def _check_num_epochs(cls, v: int) -> int:
        # 0 is a sentinel meaning "use max_steps instead"; the frontend's
        # steps-vs-epochs toggle sends it.
        if v is None:
            return 1
        if v < 0 or v > _MAX_EPOCHS:
            raise ValueError(f"num_epochs must be in [0, {_MAX_EPOCHS}] (got {v!r})")
        return v

    @field_validator("max_steps")
    @classmethod
    def _check_max_steps(cls, v: Optional[int]) -> Optional[int]:
        # 0 is the frontend's sentinel for "use num_epochs instead".
        if v is None:
            return v
        if not isinstance(v, int) or v < 0 or v > _MAX_STEPS:
            raise ValueError(
                f"max_steps must be a non-negative int <= {_MAX_STEPS} (got {v!r})"
            )
        return v

    @field_validator("max_seq_length")
    @classmethod
    def _check_max_seq_length(cls, v: int) -> int:
        if v is None or v < 1 or v > _MAX_SEQ_LENGTH:
            raise ValueError(
                f"max_seq_length must be in [1, {_MAX_SEQ_LENGTH}] (got {v!r})"
            )
        return v

    @field_validator("warmup_steps")
    @classmethod
    def _check_warmup_steps(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if not isinstance(v, int) or v < 0 or v > _MAX_STEPS:
            raise ValueError(
                f"warmup_steps must be a non-negative int <= {_MAX_STEPS} "
                f"(got {v!r})"
            )
        return v

    @field_validator("warmup_ratio")
    @classmethod
    def _check_warmup_ratio(cls, v):
        if v is None:
            return v
        try:
            r = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"warmup_ratio must be a number (got {v!r})")
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0.0, 1.0] (got {r!r})")
        return r

    @field_validator("save_steps")
    @classmethod
    def _check_save_steps(cls, v: int) -> int:
        if v is None:
            return 100
        if v < 0 or v > _MAX_STEPS:
            raise ValueError(f"save_steps must be in [0, {_MAX_STEPS}] (got {v!r})")
        return v

    @field_validator("weight_decay")
    @classmethod
    def _check_weight_decay(cls, v: float) -> float:
        if v is None:
            return 0.0
        try:
            wd = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"weight_decay must be a number (got {v!r})")
        if wd < 0 or wd > 10.0:
            raise ValueError(
                f"weight_decay must be in [0, 10] (got {wd!r}); typical 0..0.1"
            )
        return wd

    @field_validator("lora_r")
    @classmethod
    def _check_lora_r(cls, v: int) -> int:
        if v is None:
            return 16
        if v < 1 or v > _MAX_LORA_R:
            raise ValueError(f"lora_r must be in [1, {_MAX_LORA_R}] (got {v!r})")
        return v

    @field_validator("lora_alpha")
    @classmethod
    def _check_lora_alpha(cls, v: int) -> int:
        if v is None:
            return 16
        if v < 1 or v > _MAX_LORA_ALPHA:
            raise ValueError(
                f"lora_alpha must be in [1, {_MAX_LORA_ALPHA}] (got {v!r})"
            )
        return v

    @field_validator("lora_dropout")
    @classmethod
    def _check_lora_dropout(cls, v: float) -> float:
        if v is None:
            return 0.0
        try:
            d = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"lora_dropout must be a number (got {v!r})")
        if not (0.0 <= d < 1.0):
            raise ValueError(f"lora_dropout must be in [0.0, 1.0) (got {d!r})")
        return d

    custom_format_mapping: Optional[Dict[str, Any]] = Field(
        None,
        description = (
            "User-provided column-to-role mapping, e.g. {'image': 'image', 'caption': 'text'} "
            "for VLM or {'instruction': 'user', 'output': 'assistant'} for LLM. "
            "Enhanced format includes __system_prompt, __user_template, "
            "__assistant_template, __label_mapping metadata keys."
        ),
    )
    # Training parameters
    num_epochs: int = Field(1, description = "Number of training epochs")
    learning_rate: str = Field("2e-4", description = "Learning rate")
    batch_size: int = Field(1, description = "Batch size")
    gradient_accumulation_steps: int = Field(
        1, description = "Gradient accumulation steps"
    )
    warmup_steps: Optional[int] = Field(None, description = "Warmup steps")
    warmup_ratio: Optional[float] = Field(None, description = "Warmup ratio")
    max_steps: Optional[int] = Field(None, description = "Maximum training steps")
    save_steps: int = Field(100, description = "Steps between checkpoints")
    weight_decay: float = Field(0.001, description = "Weight decay")
    max_grad_norm: float = Field(
        0.0,
        ge = 0,
        description = "Global gradient norm clipping threshold. Set 0 to disable.",
    )
    random_seed: int = Field(42, description = "Random seed")
    packing: bool = Field(False, description = "Enable sequence packing")
    optim: str = Field("adamw_8bit", description = "Optimizer")
    lr_scheduler_type: str = Field("linear", description = "Learning rate scheduler type")
    embedding_learning_rate: Optional[float] = Field(
        None,
        gt = 0,
        lt = 1.0,
        description = "Separate learning rate for embedding matrices (CPT). "
        "Must be in (0, 1). Should be 2-10x smaller than the main learning rate.",
    )

    # LoRA parameters
    use_lora: bool = Field(True, description = "Use LoRA (derived from training_type)")
    lora_r: int = Field(16, description = "LoRA rank")
    lora_alpha: int = Field(16, description = "LoRA alpha")
    lora_dropout: float = Field(0.0, description = "LoRA dropout")
    target_modules: List[str] = Field(
        default_factory = list, description = "Target modules for LoRA"
    )
    gradient_checkpointing: str = Field(
        "", description = "Gradient checkpointing setting"
    )
    use_rslora: bool = Field(False, description = "Use RSLoRA")
    use_loftq: bool = Field(False, description = "Use LoftQ")
    train_on_completions: bool = Field(False, description = "Train on completions only")

    # Vision-specific LoRA parameters
    finetune_vision_layers: bool = Field(False, description = "Finetune vision layers")
    finetune_language_layers: bool = Field(
        False, description = "Finetune language layers"
    )
    finetune_attention_modules: bool = Field(
        False, description = "Finetune attention modules"
    )
    finetune_mlp_modules: bool = Field(False, description = "Finetune MLP modules")
    is_dataset_image: bool = Field(
        False, description = "Whether the dataset contains image data"
    )
    is_dataset_audio: bool = Field(
        False, description = "Whether the dataset contains audio data"
    )
    is_embedding: bool = Field(
        False, description = "Whether model is an embedding/sentence-transformer model"
    )

    # Logging parameters
    enable_wandb: bool = Field(False, description = "Enable Weights & Biases logging")
    wandb_token: Optional[str] = Field(None, description = "W&B token")
    wandb_project: Optional[str] = Field(None, description = "W&B project name")
    enable_tensorboard: bool = Field(False, description = "Enable TensorBoard logging")
    tensorboard_dir: Optional[str] = Field(None, description = "TensorBoard directory")
    resume_from_checkpoint: Optional[str] = Field(
        None, description = "Saved training output directory to resume from"
    )

    # GPU selection
    gpu_ids: Optional[List[int]] = Field(
        None,
        description = "Physical GPU indices to use, for example [0, 1]. Omit or pass [] to use automatic selection. Explicit gpu_ids are unsupported when the parent CUDA_VISIBLE_DEVICES uses UUID/MIG entries.",
    )

    @model_validator(mode = "after")
    def _check_steps_or_epochs(self) -> "TrainingStartRequest":
        # num_epochs and max_steps each accept 0 as a "use the other one"
        # sentinel. If both resolve to 0 there's nothing to train against.
        if (self.max_steps is None or self.max_steps == 0) and self.num_epochs == 0:
            raise ValueError(
                "Either num_epochs or max_steps must be > 0; both cannot be 0."
            )
        return self


class TrainingJobResponse(BaseModel):
    """Immediate response when training is initiated"""

    job_id: str = Field(..., description = "Unique training job identifier")
    status: Literal["queued", "error"] = Field(..., description = "Initial job status")
    message: str = Field(..., description = "Human-readable status message")
    error: Optional[str] = Field(None, description = "Error details if status is 'error'")


class TrainingStatus(BaseModel):
    """Current training job status - works for streaming or polling"""

    job_id: str = Field(..., description = "Training job identifier")
    phase: Literal[
        "idle",
        "loading_model",
        "loading_dataset",
        "configuring",
        "training",
        "completed",
        "error",
        "stopped",
    ] = Field(..., description = "Current phase of training pipeline")
    is_training_running: bool = Field(
        ..., description = "True if training loop is actively running"
    )
    eval_enabled: bool = Field(
        False,
        description = "True if evaluation dataset is configured for this training run",
    )
    message: str = Field(..., description = "Human-readable status message")
    error: Optional[str] = Field(None, description = "Error details if phase is 'error'")
    details: Optional[dict] = Field(
        None, description = "Phase-specific info, e.g. {'model_size': '8B'}"
    )
    metric_history: Optional[dict] = Field(
        None,
        description = "Full metric history arrays for chart recovery after SSE reconnection. "
        "Keys: 'steps', 'loss', 'lr', 'grad_norm', 'grad_norm_steps' — each a list of numeric values.",
    )


class TrainingProgress(BaseModel):
    """Training progress metrics - for streaming or polling"""

    job_id: str = Field(..., description = "Training job identifier")
    step: int = Field(..., description = "Current training step")
    total_steps: int = Field(..., description = "Total training steps")
    loss: Optional[float] = Field(None, description = "Current loss value")
    learning_rate: Optional[float] = Field(None, description = "Current learning rate")
    progress_percent: float = Field(
        ..., description = "Progress percentage (0.0 to 100.0)"
    )
    epoch: Optional[float] = Field(None, description = "Current epoch")
    elapsed_seconds: Optional[float] = Field(
        None, description = "Time elapsed since training started"
    )
    eta_seconds: Optional[float] = Field(None, description = "Estimated time remaining")
    grad_norm: Optional[float] = Field(
        None, description = "L2 norm of gradients, computed before gradient clipping"
    )
    num_tokens: Optional[int] = Field(
        None, description = "Total number of tokens processed so far"
    )
    eval_loss: Optional[float] = Field(
        None, description = "Eval loss from the most recent evaluation step"
    )


class TrainingRunSummary(BaseModel):
    """Summary of a training run for list views."""

    id: str
    status: Literal["running", "completed", "stopped", "error"]
    model_name: str
    dataset_name: str
    display_name: Optional[str] = None
    started_at: str
    ended_at: Optional[str] = None
    total_steps: Optional[int] = None
    final_step: Optional[int] = None
    final_loss: Optional[float] = None
    output_dir: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    loss_sparkline: Optional[List[float]] = None
    can_resume: bool = False
    resumed_later: bool = False


class TrainingRunUpdateRequest(BaseModel):
    """Mutable fields on a training run."""

    model_config = ConfigDict(extra = "forbid")

    display_name: Optional[str] = Field(None, max_length = 120)


class TrainingRunListResponse(BaseModel):
    """Response for listing training runs."""

    runs: List[TrainingRunSummary]
    total: int


class TrainingRunMetrics(BaseModel):
    """Metrics arrays for a training run, using paired step arrays per metric."""

    step_history: List[int] = Field(default_factory = list)
    loss_history: List[float] = Field(default_factory = list)
    loss_step_history: List[int] = Field(default_factory = list)
    lr_history: List[float] = Field(default_factory = list)
    lr_step_history: List[int] = Field(default_factory = list)
    grad_norm_history: List[float] = Field(default_factory = list)
    grad_norm_step_history: List[int] = Field(default_factory = list)
    eval_loss_history: List[float] = Field(default_factory = list)
    eval_step_history: List[int] = Field(default_factory = list)
    final_epoch: Optional[float] = None
    final_num_tokens: Optional[int] = None


class TrainingRunDetailResponse(BaseModel):
    """Response for a single training run with config and metrics."""

    run: TrainingRunSummary
    config: dict
    metrics: TrainingRunMetrics


class TrainingRunDeleteResponse(BaseModel):
    """Response for deleting a training run."""

    status: str
    message: str
