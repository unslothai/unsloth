# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Training API
"""

import re
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Any, Optional, List, Dict, Literal

from utils.training_runs import normalize_project_name


# ASCII integer, optional single sign. Rejects "++512" and Unicode digits
# ("５１２") that slip through str.isdigit() + int().
_INT_RE = re.compile(r"[+-]?[0-9]+")


_MAX_BATCH_SIZE = 4096
_MAX_GRAD_ACCUM = 4096
_MAX_STEPS = 1_000_000
_MAX_EPOCHS = 1000
# 2M is a sanity cap; host RAM runs out long before this.
_MAX_SEQ_LENGTH = 2_000_000
_MAX_LR_VALUE = 1.0
_MAX_LORA_R = 16_384
_MAX_LORA_ALPHA = 32_768
_MIN_VISION_IMAGE_SIZE = 256
# 2048 is the highest most llms stay stable at
_MAX_VISION_IMAGE_SIZE = 2048
# Upper bound for dataset slice indices. Caps `.skip(n)` on streaming datasets so
# an absurd index can't make the loader iterate effectively forever (DoS guard).
# 1e9 is far beyond any realistic fine-tuning dataset row count.
_MAX_DATASET_SLICE_INDEX = 1_000_000_000


class S3Config(BaseModel):
    """S3 bucket configuration for loading datasets from AWS S3"""

    # Accept both snake_case and the frontend's camelCase field names.
    model_config = ConfigDict(populate_by_name = True)

    bucket: str = Field(..., description = "S3 bucket name")
    region: str = Field("us-east-1", description = "AWS region")
    prefix: Optional[str] = Field(
        None, description = "Optional path prefix within bucket"
    )
    access_key_id: Optional[str] = Field(
        None,
        alias = "accessKeyId",
        description = "AWS access key ID (optional if using IAM role)",
    )
    secret_access_key: Optional[str] = Field(
        None,
        alias = "secretAccessKey",
        description = "AWS secret access key (optional if using IAM role)",
    )
    use_iam_role: bool = Field(
        False,
        alias = "useIamRole",
        description = "Use IAM role credentials instead of access keys",
    )

    @model_validator(mode = "after")
    def _check_credentials(self) -> "S3Config":
        # Require either IAM role auth or a full key pair so credentials are
        # never half-configured.
        if not self.use_iam_role and not (
            self.access_key_id and self.secret_access_key
        ):
            raise ValueError(
                "s3_config requires either use_iam_role=True or both "
                "access_key_id and secret_access_key"
            )
        return self


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
            f"learning_rate must be > 0 (got {lr!r}); typical range is 1e-6 .. 1e-3"
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
    project_name: Optional[str] = Field(
        None,
        max_length = 80,
        description = "Optional user-defined project name appended to run folders and shown in history",
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
    vision_image_size: Optional[int] = Field(
        None,
        description = "Optional maximum image side length for VLM training. Null uses model default.",
    )
    trust_remote_code: bool = Field(
        False,
        description = "Allow loading models with custom code (e.g. NVIDIA Nemotron). Only enable for repos you trust.",
    )
    approved_remote_code_fingerprint: Optional[str] = Field(
        None,
        description = "sha256 fingerprint from the remote-code scan, pinning user approval of this exact custom-code version.",
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
    dataset_streaming: bool = Field(
        False,
        description = "Whether to load the Hugging Face dataset in streaming mode",
    )
    eval_steps: float = Field(
        0.00, description = "Fraction of total steps between evals (0-1)"
    )
    dataset_slice_start: Optional[int] = Field(
        None,
        ge = 0,
        le = _MAX_DATASET_SLICE_INDEX,
        description = "Inclusive start row index for dataset slicing",
    )
    dataset_slice_end: Optional[int] = Field(
        None,
        ge = 0,
        le = _MAX_DATASET_SLICE_INDEX,
        description = "Inclusive end row index for dataset slicing",
    )

    @model_validator(mode = "before")
    @classmethod
    def _compat_split(cls, values: Any) -> Any:
        """Accept legacy 'split' field as alias for 'train_split'."""
        if isinstance(values, dict) and "split" in values:
            values.setdefault("train_split", values.pop("split"))
        return values

    @field_validator("project_name")
    @classmethod
    def _normalize_project_name(cls, value: Optional[str]) -> Optional[str]:
        return normalize_project_name(value)

    # NOTE: pydantic runs all `mode="after"` validators in definition order. A
    # second one, `_check_steps_or_epochs`, is defined lower in this class; keep
    # these cross-field checks order-independent so the two stay decoupled.
    @model_validator(mode = "after")
    def _validate_dataset_slice(self) -> "TrainingStartRequest":
        # Only the ordering is validated here. No upper bound is enforced on the
        # indices: the trainer slices via datasets `.take()` / `.select()`, which
        # clamp gracefully when the end index exceeds the dataset length.
        # start == end is intentionally allowed (deliberate single-row slice,
        # e.g. for debugging); the trainer logs a warning for that 1-row case.
        if (
            self.dataset_slice_start is not None
            and self.dataset_slice_end is not None
            and self.dataset_slice_end < self.dataset_slice_start
        ):
            raise ValueError(
                "dataset_slice_end must be greater than or equal to dataset_slice_start"
            )
        return self

    @field_validator("hf_dataset")
    @classmethod
    def _check_hf_dataset(cls, v: Optional[str]) -> Optional[str]:
        # Constrain the HF dataset id to a safe charset + length to shrink the
        # path-traversal / SSRF surface of `load_dataset(<id>, ...)`.
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if len(v) > 256:
            raise ValueError("hf_dataset is too long (max 256 chars)")
        if ".." in v:
            raise ValueError("hf_dataset must not contain '..'")
        if not re.fullmatch(r"[A-Za-z0-9._\-/]+", v):
            raise ValueError(
                "hf_dataset may only contain letters, digits, '_', '-', '.', '/'"
            )
        return v

    @field_validator("subset")
    @classmethod
    def _check_subset(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if len(v) > 128:
            raise ValueError("subset is too long (max 128 chars)")
        if not re.fullmatch(r"[A-Za-z0-9._\-]*", v):
            raise ValueError("subset may only contain letters, digits, '_', '-', '.'")
        return v

    @field_validator("train_split", "eval_split")
    @classmethod
    def _check_split_name(cls, v: Optional[str]) -> Optional[str]:
        # Split names feed HF slice syntax (e.g. "train[:80%]"), so allow that
        # charset but cap length and block path-traversal / NUL bytes.
        if v is None:
            return v
        if len(v) > 128:
            raise ValueError("split name is too long (max 128 chars)")
        if "\x00" in v or ".." in v or "/" in v or "\\" in v:
            raise ValueError("split name contains invalid characters")
        if not re.fullmatch(r"[A-Za-z0-9_\-\[\]:%.+ ]*", v):
            raise ValueError("split name contains invalid characters")
        return v

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
        # 0 is a sentinel for "use max_steps instead" (frontend toggle).
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

    @field_validator("vision_image_size", mode = "before")
    @classmethod
    def _check_vision_image_size(cls, v: Any) -> Optional[int]:
        # mode="before" sees True/False as bool (not 1/0) for a precise error.
        if v is None:
            return v
        if isinstance(v, bool):
            raise ValueError("vision_image_size must be an integer or null")
        if isinstance(v, int):
            coerced = v
        elif isinstance(v, str) and _INT_RE.fullmatch(v.strip()):
            coerced = int(v.strip())
        elif isinstance(v, float) and v.is_integer():
            coerced = int(v)
        else:
            # numpy ints / Integral subclasses, without a hard numpy import.
            try:
                import numbers
                if isinstance(v, numbers.Integral):
                    coerced = int(v)
                elif isinstance(v, numbers.Real) and float(v).is_integer():
                    coerced = int(v)
                else:
                    raise TypeError
            except Exception:
                raise ValueError("vision_image_size must be an integer or null")
        if coerced < _MIN_VISION_IMAGE_SIZE or coerced > _MAX_VISION_IMAGE_SIZE:
            raise ValueError(
                f"vision_image_size must be in [{_MIN_VISION_IMAGE_SIZE}, "
                f"{_MAX_VISION_IMAGE_SIZE}] (got {coerced!r})"
            )
        return coerced

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
    max_grad_value: Optional[float] = Field(
        None,
        ge = 0,
        description = (
            "MLX-only elementwise gradient value clipping threshold. "
            "If unset, MLX uses its runtime default."
        ),
    )
    max_grad_leaf_norm: Optional[float] = Field(
        None,
        ge = 0,
        description = (
            "MLX-only proportional per-parameter gradient norm cap. "
            "Preserves each tensor's gradient direction without global norm "
            "clipping's memory overhead."
        ),
    )
    cast_norm_output_to_input_dtype: bool = Field(
        True,
        description = (
            "MLX-only: keep norm parameters in fp32 but cast norm outputs "
            "back to the incoming activation dtype."
        ),
    )
    random_seed: int = Field(
        3407,
        description = (
            "Random seed; matches the Unsloth backend / MLX worker default "
            "and unsloth's historical recommended value."
        ),
    )
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

    # S3 dataset source configuration
    s3_config: Optional[S3Config] = Field(
        None,
        description = "S3 bucket configuration for loading datasets from AWS S3. Requires boto3 to be installed.",
    )

    @model_validator(mode = "after")
    def _validate_streaming_splits(self) -> "TrainingStartRequest":
        # Streaming load_dataset does not accept HF slice syntax (e.g. "train[:50%]"
        # or "train[:20]"). Probe-confirmed: raises ValueError: Bad split. Reject
        # early with a clear message so the user knows to use a plain split name.
        if self.dataset_streaming:
            for field_name, split_val in (
                ("train_split", self.train_split),
                ("eval_split", self.eval_split),
            ):
                if split_val is not None and "[" in split_val:
                    raise ValueError(
                        f"dataset_streaming does not support HF slice syntax in {field_name} "
                        f"(got {split_val!r}); streaming load_dataset raises 'Bad split' on "
                        "bracket expressions. Use a plain split name (e.g. 'train', 'validation')."
                    )
        return self

    @model_validator(mode = "after")
    def _check_steps_or_epochs(self) -> "TrainingStartRequest":
        # Each accepts 0 as "use the other"; both 0 means nothing to train.
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
    project_name: Optional[str] = None
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
    has_preview_model: bool = False
    preview_ref: Optional[str] = None
    # HMAC capability token for the `/p/{preview_ref}` share link; None when not
    # previewable. The frontend appends it as `?k=` so a guessed ref can't be used.
    preview_sig: Optional[str] = None


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
