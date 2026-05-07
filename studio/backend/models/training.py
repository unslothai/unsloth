# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Training API
"""

from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, List, Dict, Literal


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
