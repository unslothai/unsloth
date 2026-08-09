# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Training API
"""

import re
from pathlib import Path, PureWindowsPath
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Any, Optional, List, Dict, Literal, Union

from hub.schemas.inventory import ModelFormat
from utils.hf_dataset_options import (
    MAX_HF_DATASET_OPTION_LENGTH,
    valid_hf_dataset_config_name,
    valid_hf_dataset_split_name,
    valid_hf_dataset_split_instruction,
)
from utils.training_runs import normalize_project_name


# ASCII integer, optional single sign. Rejects "++512" and Unicode digits that pass str.isdigit().
_INT_RE = re.compile(r"[+-]?[0-9]+")
_HF_DATASET_ID_SEGMENT_RE = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?")
TRAINING_REQUEST_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"


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
# Upper bound for dataset slice indices, capping `.skip(n)` on streaming datasets so an absurd
# index can't iterate effectively forever. 1e9 is far beyond any realistic dataset row count.
_MAX_DATASET_SLICE_INDEX = 1_000_000_000


class S3Config(BaseModel):
    """S3 bucket configuration for loading datasets from AWS S3"""

    # Accept both snake_case and the frontend's camelCase field names.
    model_config = ConfigDict(populate_by_name = True)

    bucket: str = Field(..., description = "S3 bucket name")
    region: str = Field("us-east-1", description = "AWS region")
    prefix: Optional[str] = Field(None, description = "Optional path prefix within bucket")
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
        # Require either IAM role auth or a full key pair so credentials are never half-configured.
        if not self.use_iam_role and not (self.access_key_id and self.secret_access_key):
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
        raise ValueError(f"learning_rate must be > 0 (got {lr!r}); typical range is 1e-6 .. 1e-3")
    if lr >= _MAX_LR_VALUE:
        raise ValueError(
            f"learning_rate must be < 1.0 (got {lr!r}); values that large always diverge training"
        )
    return lr


class TrainingStartRequest(BaseModel):
    """Request schema for starting training"""

    model_name: str = Field(
        ..., description = "Model identifier (e.g., 'unsloth/llama-3-8b-bnb-4bit')"
    )
    project_name: Optional[str] = Field(
        None,
        max_length = 80,
        description = "Optional user-defined project name appended to run folders and shown in history",
    )
    start_request_id: Optional[str] = Field(
        None,
        min_length = 1,
        max_length = 128,
        pattern = TRAINING_REQUEST_ID_PATTERN,
        description = "Opaque client-generated identifier used to reconcile an ambiguous start response",
    )
    training_type: Literal["LoRA/QLoRA", "Full Finetuning", "Continued Pretraining"] = Field(
        ...,
        description = "Training type: 'LoRA/QLoRA', 'Full Finetuning', or 'Continued Pretraining'",
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
    model_known_cached: bool = Field(
        False,
        description = "Whether the selected model is already present in the local HF cache",
    )
    model_local_path: Optional[str] = Field(
        None,
        description = "Local HF cache path for the selected model, when known",
    )
    model_format: Optional[ModelFormat] = Field(
        None,
        description = "On-disk format of the selected model, when known",
    )
    model_snapshot_path: Optional[str] = Field(
        None,
        description = "Server-verified model snapshot directory pinned for this run",
    )

    hf_dataset: Optional[str] = Field(None, description = "HuggingFace dataset identifier")
    dataset_known_cached: bool = Field(
        False,
        description = "Whether the selected HF dataset is already present in the local cache",
    )
    dataset_local_path: Optional[str] = Field(
        None,
        description = "Local HF cache path for the selected dataset, when known",
    )
    dataset_snapshot_path: Optional[str] = Field(
        None,
        description = "Server-verified dataset snapshot directory pinned for this run",
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
    eval_split: Optional[str] = Field(None, description = "Eval split name. None = auto-detect")
    dataset_streaming: bool = Field(
        False,
        description = "Whether to load the Hugging Face dataset in streaming mode",
    )
    eval_steps: float = Field(0.00, description = "Fraction of total steps between evals (0-1)")
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

    # NOTE: pydantic runs all `mode="after"` validators in definition order, and
    # `_check_steps_or_epochs` is lower in this class; keep these checks order-independent.
    @model_validator(mode = "after")
    def _validate_dataset_slice(self) -> "TrainingStartRequest":
        # start == end is intentionally allowed (a deliberate single-row slice); the trainer warns.
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
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if len(v) > 256:
            raise ValueError("hf_dataset is too long (max 256 chars)")
        if ".." in v:
            raise ValueError("hf_dataset must not contain '..'")
        if any(_HF_DATASET_ID_SEGMENT_RE.fullmatch(segment) is None for segment in v.split("/")):
            raise ValueError("hf_dataset contains invalid characters or path segments")
        return v

    @field_validator("subset")
    @classmethod
    def _check_subset(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if len(v) > MAX_HF_DATASET_OPTION_LENGTH:
            raise ValueError(f"subset is too long (max {MAX_HF_DATASET_OPTION_LENGTH} chars)")
        if not valid_hf_dataset_config_name(v):
            raise ValueError("subset contains invalid characters")
        return v

    @field_validator(
        "model_local_path",
        "dataset_local_path",
        "model_snapshot_path",
        "dataset_snapshot_path",
    )
    @classmethod
    def _check_cache_local_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if len(v) > 4096:
            raise ValueError("local cache path is too long (max 4096 chars)")
        if "\x00" in v:
            raise ValueError("local cache path contains invalid characters")
        if ".." in Path(v).parts or ".." in PureWindowsPath(v).parts:
            raise ValueError("local cache path must not contain '..' segments")
        return v

    @field_validator("train_split", "eval_split")
    @classmethod
    def _check_split_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if len(v) > MAX_HF_DATASET_OPTION_LENGTH:
            raise ValueError(f"split name is too long (max {MAX_HF_DATASET_OPTION_LENGTH} chars)")
        if not valid_hf_dataset_split_instruction(v):
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
            raise ValueError(f"batch_size must be in [1, {_MAX_BATCH_SIZE}] (got {v!r})")
        return v

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def _check_grad_accum(cls, v: int) -> int:
        if v is None:
            return 1
        if v < 1 or v > _MAX_GRAD_ACCUM:
            raise ValueError(
                f"gradient_accumulation_steps must be in [1, {_MAX_GRAD_ACCUM}] (got {v!r})"
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
            raise ValueError(f"max_steps must be a non-negative int <= {_MAX_STEPS} (got {v!r})")
        return v

    @field_validator("max_seq_length")
    @classmethod
    def _check_max_seq_length(cls, v: int) -> int:
        if v is None or v < 1 or v > _MAX_SEQ_LENGTH:
            raise ValueError(f"max_seq_length must be in [1, {_MAX_SEQ_LENGTH}] (got {v!r})")
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
            raise ValueError(f"warmup_steps must be a non-negative int <= {_MAX_STEPS} (got {v!r})")
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
            raise ValueError(f"weight_decay must be in [0, 10] (got {wd!r}); typical 0..0.1")
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
            raise ValueError(f"lora_alpha must be in [1, {_MAX_LORA_ALPHA}] (got {v!r})")
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
    num_epochs: int = Field(1, description = "Number of training epochs")
    learning_rate: str = Field("2e-4", description = "Learning rate")
    batch_size: int = Field(1, description = "Batch size")
    gradient_accumulation_steps: int = Field(1, description = "Gradient accumulation steps")
    warmup_steps: Optional[int] = Field(None, description = "Warmup steps")
    warmup_ratio: Optional[float] = Field(None, description = "Warmup ratio")
    max_steps: Optional[int] = Field(None, description = "Maximum training steps")
    save_steps: int = Field(100, description = "Steps between checkpoints")
    weight_decay: float = Field(0.001, description = "Weight decay")
    # All three clip knobs are finite as well as non-negative: JSON 1e309 (and
    # FastAPI's Infinity literal) floats to inf, which clears ge=0 but never
    # binds, so the run would train unclipped while reporting a threshold.
    max_grad_norm: Optional[float] = Field(
        None,
        ge = 0,
        allow_inf_nan = False,
        description = (
            "Global gradient norm clipping threshold. Unset keeps the training "
            "backend's own default; 0 turns global-norm clipping off, which on "
            "MLX leaves per-parameter clipping in force unless max_grad_leaf_norm "
            "is also 0. Honored on MLX; the CUDA path trains with its own fixed "
            "thresholds."
        ),
    )
    max_grad_value: Optional[float] = Field(
        None,
        ge = 0,
        allow_inf_nan = False,
        description = (
            "MLX-only elementwise gradient value clipping threshold. "
            "If unset, MLX uses its runtime default."
        ),
    )
    max_grad_leaf_norm: Optional[float] = Field(
        None,
        ge = 0,
        allow_inf_nan = False,
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

    use_lora: bool = Field(True, description = "Use LoRA (derived from training_type)")
    lora_r: int = Field(16, description = "LoRA rank")
    lora_alpha: int = Field(16, description = "LoRA alpha")
    lora_dropout: float = Field(0.0, description = "LoRA dropout")
    target_modules: List[str] = Field(default_factory = list, description = "Target modules for LoRA")
    gradient_checkpointing: str = Field("", description = "Gradient checkpointing setting")
    use_rslora: bool = Field(False, description = "Use RSLoRA")
    use_loftq: bool = Field(False, description = "Use LoftQ")
    use_dora: bool = Field(False, description = "Use DoRA")
    train_on_completions: bool = Field(False, description = "Train on completions only")

    # Vision-specific LoRA parameters
    finetune_vision_layers: bool = Field(False, description = "Finetune vision layers")
    finetune_language_layers: bool = Field(False, description = "Finetune language layers")
    finetune_attention_modules: bool = Field(False, description = "Finetune attention modules")
    finetune_mlp_modules: bool = Field(False, description = "Finetune MLP modules")
    is_dataset_image: bool = Field(False, description = "Whether the dataset contains image data")
    is_dataset_audio: bool = Field(False, description = "Whether the dataset contains audio data")
    is_embedding: bool = Field(
        False, description = "Whether model is an embedding/sentence-transformer model"
    )

    enable_wandb: bool = Field(False, description = "Enable Weights & Biases logging")
    wandb_token: Optional[str] = Field(None, description = "W&B token")
    wandb_project: Optional[str] = Field(None, description = "W&B project name")
    enable_tensorboard: bool = Field(False, description = "Enable TensorBoard logging")
    tensorboard_dir: Optional[str] = Field(None, description = "TensorBoard directory")
    resume_from_checkpoint: Optional[str] = Field(
        None, description = "Saved training output directory to resume from"
    )

    gpu_ids: Optional[List[int]] = Field(
        None,
        description = (
            "Physical GPU indices to use, for example [0, 1]. Omit or pass "
            "[] to use automatic selection. Explicit gpu_ids are unsupported "
            "when the parent visibility mask uses non-numeric or subdevice "
            "entries -- this includes CUDA_VISIBLE_DEVICES with UUID/MIG "
            "entries on NVIDIA, and ZE_AFFINITY_MASK with subdevice tokens "
            "(e.g. '0.0,0.1') or FLAT-hierarchy (default) tile handles on "
            "Intel XPU."
        ),
    )

    s3_config: Optional[S3Config] = Field(
        None,
        description = "S3 bucket configuration for loading datasets from AWS S3. Requires boto3 to be installed.",
    )

    @field_validator("target_modules", mode = "before")
    @classmethod
    def _normalize_target_modules(cls, value: Any) -> Any:
        # Sanitized non-LoRA history stores the unused value as null; treat it as an omitted list.
        return [] if value is None else value

    @model_validator(mode = "after")
    def _validate_streaming_splits(self) -> "TrainingStartRequest":
        # Streaming load_dataset does not accept HF slice syntax (probe-confirmed: ValueError: Bad
        # split). Reject early with a clear message so the user knows to use a plain split name.
        if self.dataset_streaming:
            for field_name, split_val in (
                ("train_split", self.train_split),
                ("eval_split", self.eval_split),
            ):
                if split_val is not None and not valid_hf_dataset_split_name(split_val):
                    raise ValueError(
                        f"dataset_streaming requires a plain split name in {field_name} "
                        f"(got {split_val!r}); use a name such as 'train' or 'validation'."
                    )
        return self

    @model_validator(mode = "after")
    def _check_steps_or_epochs(self) -> "TrainingStartRequest":
        # Each accepts 0 as "use the other"; both 0 means nothing to train.
        if (self.max_steps is None or self.max_steps == 0) and self.num_epochs == 0:
            raise ValueError("Either num_epochs or max_steps must be > 0; both cannot be 0.")
        return self

    @model_validator(mode = "after")
    def _validate_lora_variant_flags(self) -> "TrainingStartRequest":
        # The frontend only ever sends one of these and never under Full Finetuning, but a direct
        # API/YAML/CLI caller can bypass that. Nothing downstream breaks, but reject early for a
        # clear error instead of a silently-ignored flag.
        active = [
            name
            for name, enabled in (
                ("use_rslora", self.use_rslora),
                ("use_loftq", self.use_loftq),
                ("use_dora", self.use_dora),
            )
            if enabled
        ]
        if len(active) > 1:
            raise ValueError(
                f"Only one LoRA variant may be enabled at a time; got {active}. "
                "use_rslora, use_loftq, and use_dora are mutually exclusive."
            )
        # getattr, not self.training_type: model_construct() (used by single-field tests) leaves
        # required fields unset, and this mode="after" validator still runs on that partial instance.
        if getattr(self, "training_type", None) == "Full Finetuning" and active:
            raise ValueError(
                f"{active[0]} requires an adapter method (LoRA/QLoRA or "
                "Continued Pretraining); it has no effect under Full Finetuning."
            )
        return self


class TrainingJobResponse(BaseModel):
    """Immediate response when training is initiated"""

    job_id: str = Field(..., description = "Unique training job identifier")
    status: Literal["pending", "queued", "error"] = Field(..., description = "Initial job status")
    message: str = Field(..., description = "Human-readable status message")
    error: Optional[str] = Field(None, description = "Error details if status is 'error'")
    error_code: Optional[str] = Field(None, description = "Stable error code if status is 'error'")


class TrainingStartRequestStatus(BaseModel):
    start_request_id: str
    job_id: str
    state: Literal["pending", "accepted", "rejected"]
    message: str
    error: Optional[str] = None
    error_code: Optional[str] = None


class TrainingStatus(BaseModel):
    """Current training job status - works for streaming or polling"""

    job_id: str = Field(..., description = "Training job identifier")
    start_request_id: Optional[str] = Field(
        None, description = "Client-generated identifier for the current training start request"
    )
    start_request_state: Optional[Literal["pending", "accepted", "rejected"]] = Field(
        None, description = "Lifecycle state of the current training start request"
    )
    phase: Literal[
        "idle",
        "loading_model",
        "loading_dataset",
        "configuring",
        "training",
        "finalizing",
        "completed",
        "error",
        "stopped",
    ] = Field(..., description = "Current phase of training pipeline")
    is_training_running: bool = Field(..., description = "True if training loop is actively running")
    eval_enabled: bool = Field(
        False,
        description = "True if evaluation dataset is configured for this training run",
    )
    message: str = Field(..., description = "Human-readable status message")
    error: Optional[str] = Field(None, description = "Error details if phase is 'error'")
    warnings: List[str] = Field(
        default_factory = list,
        description = "Non-fatal warnings retained for the current training run",
    )
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
    progress_percent: float = Field(..., description = "Progress percentage (0.0 to 100.0)")
    epoch: Optional[float] = Field(None, description = "Current epoch")
    elapsed_seconds: Optional[float] = Field(
        None, description = "Time elapsed since training started"
    )
    eta_seconds: Optional[float] = Field(None, description = "Estimated time remaining")
    grad_norm: Optional[float] = Field(
        None, description = "L2 norm of gradients, computed before gradient clipping"
    )
    num_tokens: Optional[int] = Field(None, description = "Total number of tokens processed so far")
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
    # Why resume is unavailable when the reason is the recorded resource provenance rather than
    # the checkpoint. None when resumable or when the checkpoint itself is the problem.
    resume_blocked_reason: Optional[str] = None
    resumed_later: bool = False
    artifacts_available: bool = False
    has_preview_model: bool = False
    preview_ref: Optional[str] = None
    # HMAC capability token for the `/p/{preview_ref}` share link, appended as `?k=`; None when not previewable.
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
    artifacts_deleted: bool = False
    artifacts_kept_reason: Optional[Literal["shared_output_dir", "purge_failed"]] = None


class DiffusionTrainingStartRequest(BaseModel):
    """Request to start a diffusion (SDXL) LoRA training job.

    Field names mirror ``core.training.diffusion_lora_trainer.DiffusionLoraConfig`` so the
    service can pass ``model_dump()`` straight through. Only the paths are required; the
    rest carry the trainer's defaults.
    """

    model_config = ConfigDict(protected_namespaces = ())

    base_model: str = Field(..., description = "HF repo id or local path to a trainable base")
    data_dir: str = Field(..., description = "Folder of training images (+ captions)")
    output_dir: str = Field(..., description = "Directory to write the LoRA .safetensors into")
    model_family: Optional[str] = Field(
        None,
        description = "Explicit trainer family (sdxl / flux.1 / ...); omitted = detect from base_model",
    )
    instance_prompt: Optional[str] = Field(
        None, description = "Dreambooth caption applied to images without their own caption"
    )
    resolution: int = Field(
        1024, ge = 64, le = 2048, description = "Square training resolution (multiple of 8)"
    )
    train_steps: int = Field(500, ge = 1, le = 100000)
    num_epochs: int = Field(
        0,
        ge = 0,
        le = 1000,
        description = (
            "0 = use train_steps; > 0 overrides train_steps with epochs x "
            "ceil(N / (batch x grad_accum)) optimizer steps over the N-image dataset"
        ),
    )
    # Upper bound as well as positive, matching the LLM schema: JSON accepts 1e309, which floats to inf and satisfies a
    # gt-only constraint, so the route would evict residents and start AdamW at an infinite rate. Values >= 1.0 diverge.
    learning_rate: float = Field(1e-4, gt = 0, lt = 1.0)
    train_batch_size: int = Field(1, ge = 1, le = 64)
    gradient_accumulation_steps: int = Field(1, ge = 1, le = 256)
    lora_rank: int = Field(16, ge = 1, le = 320)
    lora_alpha: Optional[int] = Field(None, ge = 1, le = 640, description = "Defaults to lora_rank")
    # Strictly below 1: PEFT turns lora_dropout into nn.Dropout(p=...), so 1.0 zeroes the LoRA branch and the run saves an
    # untrained adapter while reporting normal progress. Matches TrainingStartRequest's validator.
    lora_dropout: float = Field(0.0, ge = 0.0, lt = 1.0)
    # Mirror the remaining training-affecting knobs of DiffusionLoraConfig so a client that sets them is not silently trained with defaults. Defaults to the SDXL projections.
    lora_target_modules: List[str] = Field(
        default_factory = lambda: ["to_k", "to_q", "to_v", "to_out.0"],
        description = "U-Net modules to attach LoRA to",
    )
    # Finite as well as non-negative: JSON accepts 1e309 (and FastAPI's parser the Infinity literal), which floats to inf and satisfies a ge-only
    # constraint. clip_grad_norm_ then clamps its coefficient to 1.0, so the run trains completely unclipped while the config reports clipping.
    max_grad_norm: float = Field(
        1.0,
        ge = 0,
        allow_inf_nan = False,
        description = "Gradient clipping max-norm; 0 disables clipping",
    )
    # Bounded to what torch.manual_seed unpacks (int64 low / uint64 high): an out-of-range value otherwise passes every
    # preflight, evicts the resident models, spawns the trainer, and only then dies unpacking long long.
    seed: int = Field(42, ge = -(2**63), le = 2**64 - 1)
    mixed_precision: Literal["bf16", "fp16", "no"] = Field("bf16")
    # Finite for the same reason as max_grad_norm: an inf gamma passes gt here and in normalized(), then collapses every
    # min-SNR weight to 1.0, silently training on plain unweighted MSE. null is the documented disable.
    snr_gamma: Optional[float] = Field(
        5.0, gt = 0, allow_inf_nan = False, description = "Min-SNR loss weighting; null disables"
    )
    gradient_checkpointing: bool = Field(True)
    lr_scheduler: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = Field("constant")
    lr_warmup_steps: int = Field(0, ge = 0)
    center_crop: bool = Field(False)
    random_flip: bool = Field(True)
    caption_column: str = Field("text")
    hf_token: Optional[str] = Field(None)
    cache_latents: bool = Field(
        True, description = "Precompute VAE latents once and free the VAE for the run"
    )
    cache_variants: int = Field(
        4, ge = 1, le = 16, description = "Frozen crop/flip variants per image in the latent cache"
    )
    cond_cache_dir: Optional[str] = Field(
        None,
        description = (
            "Directory for the PERSISTENT conditioning cache (latents + text embeddings), reused "
            "across runs: a rerun whose images, captions and resolution are unchanged skips "
            "loading the VAE and the multi-GB text encoders entirely. Studio-relative names are "
            "resolved under the Studio outputs root and absolute paths must stay inside it. "
            "null or blank keeps the in-memory cache, which is rebuilt every run."
        ),
    )
    compile_transformer: Literal["off", "on", "auto"] = Field(
        "auto", description = "Regional torch.compile of the transformer blocks"
    )
    enable_tf32: bool = Field(
        True, description = "TF32 matmuls + cudnn autotuning (near-lossless speedup)"
    )
    base_precision: Literal["nf4", "bf16", "int8", "fp8", "mxfp8", "auto"] = Field(
        "nf4",
        description = (
            "DiT base transformer precision: nf4 QLoRA (memory floor, default), bf16 dense, "
            "int8 torchao weight-only, fp8 float8 training compute (Ada/Hopper/Blackwell), "
            "mxfp8 block-scaled float8 compute (Blackwell, best at high resolution/batch), "
            "or auto (pick by free VRAM + GPU class). Dense modes need a non-prequant base."
        ),
    )
    # DiT-only levers the trainer implements. Undeclared, they were silently dropped by model_dump(); the defaults match DiffusionLoraConfig.
    ema_decay: float = Field(
        0.0, ge = 0.0, lt = 1.0, description = "EMA of the LoRA weights; 0 disables it"
    )
    cfg_dropout: float = Field(
        0.0, ge = 0.0, le = 1.0, description = "Chance of dropping the caption to an empty prompt"
    )
    weighting_scheme: Literal["none", "bell"] = Field(
        "none",
        description = (
            "Per-sample loss weighting over the drawn timestep: none (unweighted MSE) or bell "
            "(Gaussian bell centered mid-schedule). Timesteps are always logit-normal sampled."
        ),
    )
    flow_shift: Optional[Union[float, Literal["auto"]]] = Field(
        None,
        description = (
            "Flow-matching timestep shift. null uses the family default "
            "(auto for qwen-image, 1.0 otherwise)."
        ),
    )


class DiffusionTrainingStopRequest(BaseModel):
    """Optional body for stopping a diffusion training job. ``save`` mirrors the LLM
    trainer's stop: True (default) exports the partial adapter, False cancels without
    leaving one behind."""

    save: bool = Field(True)


class DiffusionTrainingStartResponse(BaseModel):
    """Response for starting a diffusion training job."""

    job_id: str
    status: str


class DiffusionMetricHistory(BaseModel):
    """Paired step-indexed history arrays for the live training charts. ``lr`` and
    ``grad_norm`` entries may be null so those sparse series still align with ``steps``
    by index."""

    steps: List[int] = Field(default_factory = list)
    loss: List[float] = Field(default_factory = list)
    lr: List[Optional[float]] = Field(default_factory = list)
    grad_norm: List[Optional[float]] = Field(default_factory = list)


class DiffusionTrainingStatusResponse(BaseModel):
    """A snapshot of the current diffusion training job (or idle)."""

    active: bool
    job_id: Optional[str] = None
    status: str
    message: str = ""
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    avg_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    # Total pre-clip gradient norm from the last optimizer step (health signal the UI charts).
    grad_norm: Optional[float] = None
    num_images: Optional[int] = None
    in_model_load: bool = False
    output_dir: Optional[str] = None
    lora_path: Optional[str] = None
    # The second, EMA-averaged adapter written in the run's ema subdir when ema_decay was enabled.
    ema_path: Optional[str] = None
    # Where the adapter was mirrored into the Studio LoRA catalog, and what family / base it trained from, so the UI can deploy it.
    catalog_path: Optional[str] = None
    family: Optional[str] = None
    base_model: Optional[str] = None
    # Live throughput + peak VRAM (from the trainer's progress events).
    samples_per_second: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    # Bounded step/loss/lr history for the live loss + LR charts.
    metric_history: Optional[DiffusionMetricHistory] = None


class DiffusionTrainingRunSummary(BaseModel):
    """One persisted diffusion training run (terminal), as listed in the Train tab's
    previous-runs history. The heavy payload (config + metric logs) lives in the detail."""

    job_id: str
    status: str
    message: str = ""
    adapter: Optional[str] = None
    family: Optional[str] = None
    base_model: Optional[str] = None
    step: int = 0
    total_steps: int = 0
    avg_loss: Optional[float] = None
    # Whether this run left an adapter on disk (full completion or stop-and-save).
    saved: bool = False
    catalog_path: Optional[str] = None
    instance_prompt: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


class DiffusionTrainingRunDetail(DiffusionTrainingRunSummary):
    """The full persisted record: summary + scrubbed start config + metric logs."""

    loss: Optional[float] = None
    samples_per_second: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    num_images: Optional[int] = None
    lora_path: Optional[str] = None
    ema_path: Optional[str] = None
    config: Optional[dict] = None
    metric_history: Optional[DiffusionMetricHistory] = None


class DiffusionTrainingRunsResponse(BaseModel):
    runs: List[DiffusionTrainingRunSummary] = Field(default_factory = list)


class DiffusionDatasetSummary(BaseModel):
    """One image-dataset folder under the Studio datasets root."""

    name: str
    path: str
    image_count: int
    caption_count: int


class DiffusionTrainableFamily(BaseModel):
    """A base-model family the diffusion trainer supports, with UI-facing metadata."""

    name: str
    label: str
    default_base: str
    base_repos: List[str] = Field(default_factory = list)
    defaults: dict = Field(default_factory = dict)
    vram_note: str = ""
    # vram_note's facts as fields. Empty when this host cannot train the family.
    params: str = ""
    qlora_vram_gb: Optional[int] = None
    gated: bool = False
    note: str = ""
    # base_precision modes this machine supports for the family (empty = no selector, e.g. SDXL), the recommended pick, and
    # whether regional torch.compile applies. Defaults keep older backends' payloads valid.
    precision_modes: List[str] = Field(default_factory = list)
    recommended_precision: str = "nf4"
    supports_compile: bool = False
    # When set, a LoRA trained on this family previews on this repo instead of the training base (Krea trains on Raw, runs on Turbo).
    deploy_base: Optional[str] = None


class DiffusionTrainingInfoResponse(BaseModel):
    """Where diffusion training reads/writes on this Studio, plus usable datasets and the
    trainable model families (so the UI can offer a base picker with realistic guidance)."""

    datasets_root: str
    outputs_root: str
    datasets: List[DiffusionDatasetSummary]
    families: List[DiffusionTrainableFamily] = Field(default_factory = list)


class DiffusionDatasetUploadResponse(BaseModel):
    """Result of uploading images/captions into a named dataset folder. Counts are
    for the whole folder after the upload, so repeat uploads show the running total."""

    name: str
    path: str
    image_count: int
    caption_count: int
    uploaded: int


class DiffusionDatasetImageRecord(BaseModel):
    """One image in a training dataset folder, with its resolved caption. ``caption`` is
    null when no caption exists from any source; ``caption_source`` records where the
    shown caption came from (``metadata`` beats a per-image ``sidecar``; ``none`` when
    uncaptioned) so the labeling UI can highlight images that still need a caption."""

    filename: str
    caption: Optional[str] = None
    caption_source: Literal["sidecar", "metadata", "none"] = "none"
    width: int
    height: int
    size_bytes: int


class DiffusionDatasetImagesResponse(BaseModel):
    """Every image in a dataset folder (including uncaptioned ones), for the labeling grid."""

    name: str
    path: str
    images: List[DiffusionDatasetImageRecord]


class DiffusionCaptionUpdateRequest(BaseModel):
    """Write (or, when blank, clear) the per-image ``.txt`` caption sidecar."""

    caption: str = ""


class DiffusionDatasetExample(BaseModel):
    """A curated, one-click-importable example image dataset. ``image_cap`` bounds how many
    images are materialized; ``license`` is shown verbatim so users see the terms before
    importing; ``suggested_trigger`` seeds the trigger prompt for uncaptioned subject sets."""

    id: str
    label: str
    repo: str
    description: str
    license: str
    image_cap: int
    suggested_trigger: Optional[str] = None


class DiffusionDatasetExamplesResponse(BaseModel):
    """The curated example-dataset registry the Train tab offers for one-click import."""

    examples: List[DiffusionDatasetExample]


class DiffusionDatasetImportRequest(BaseModel):
    """Import a curated example (``id``) into a dataset folder (``name``; defaults to the
    example id)."""

    id: str
    name: Optional[str] = None


class DiffusionDatasetImportResponse(BaseModel):
    """Result of a one-click example import: folder counts plus provenance so the UI can
    show what was fetched and under what license."""

    name: str
    path: str
    image_count: int
    caption_count: int
    imported: int
    license: str
    source_repo: str
