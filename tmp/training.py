"""
Pydantic schemas for Training API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class TrainingStartRequest(BaseModel):
    """Request schema for starting training"""
    # Model parameters
    model_name: str = Field(..., description="Model identifier (e.g., 'unsloth/llama-3-8b-bnb-4bit')")
    training_type: str = Field(..., description="Training type: 'LoRA/QLoRA' or 'Full Finetuning'")
    hf_token: Optional[str] = Field(None, description="HuggingFace token")
    load_in_4bit: bool = Field(True, description="Load model in 4-bit quantization")
    max_seq_length: int = Field(2048, description="Maximum sequence length")

    # Dataset parameters
    hf_dataset: Optional[str] = Field(None, description="HuggingFace dataset identifier")
    local_datasets: List[str] = Field(default_factory=list, description="List of local dataset paths")
    format_type: str = Field(..., description="Dataset format type")

    # Training parameters
    num_epochs: int = Field(1, description="Number of training epochs")
    learning_rate: str = Field("2e-4", description="Learning rate")
    batch_size: int = Field(1, description="Batch size")
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps")
    warmup_steps: Optional[int] = Field(None, description="Warmup steps")
    warmup_ratio: Optional[float] = Field(None, description="Warmup ratio")
    max_steps: Optional[int] = Field(None, description="Maximum training steps")
    save_steps: int = Field(100, description="Steps between checkpoints")
    weight_decay: float = Field(0.01, description="Weight decay")
    random_seed: int = Field(42, description="Random seed")
    packing: bool = Field(False, description="Enable sequence packing")
    optim: str = Field("adamw_8bit", description="Optimizer")
    lr_scheduler_type: str = Field("linear", description="Learning rate scheduler type")

    # LoRA parameters
    use_lora: bool = Field(True, description="Use LoRA (derived from training_type)")
    lora_r: int = Field(16, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    lora_dropout: float = Field(0.0, description="LoRA dropout")
    target_modules: List[str] = Field(default_factory=list, description="Target modules for LoRA")
    gradient_checkpointing: str = Field("", description="Gradient checkpointing setting")
    use_rslora: bool = Field(False, description="Use RSLoRA")
    use_loftq: bool = Field(False, description="Use LoftQ")
    train_on_completions: bool = Field(False, description="Train on completions only")

    # Vision-specific LoRA parameters
    finetune_vision_layers: bool = Field(False, description="Finetune vision layers")
    finetune_language_layers: bool = Field(False, description="Finetune language layers")
    finetune_attention_modules: bool = Field(False, description="Finetune attention modules")
    finetune_mlp_modules: bool = Field(False, description="Finetune MLP modules")

    # Logging parameters
    enable_wandb: bool = Field(False, description="Enable Weights & Biases logging")
    wandb_token: Optional[str] = Field(None, description="W&B token")
    wandb_project: Optional[str] = Field(None, description="W&B project name")
    enable_tensorboard: bool = Field(False, description="Enable TensorBoard logging")
    tensorboard_dir: Optional[str] = Field(None, description="TensorBoard directory")


class TrainingJobResponse(BaseModel):
    """Immediate response when training is initiated"""
    job_id: str = Field(..., description="Unique training job identifier")
    status: Literal["queued", "error"] = Field(..., description="Initial job status")
    message: str = Field(..., description="Human-readable status message")
    error: Optional[str] = Field(None, description="Error details if status is 'error'")


class TrainingStatus(BaseModel):
    """Current training job status - works for streaming or polling"""
    job_id: str = Field(..., description="Training job identifier")
    phase: Literal[
        "idle",
        "loading_model",
        "loading_dataset",
        "configuring",
        "training",
        "completed",
        "error",
        "stopped"
    ] = Field(..., description="Current phase of training pipeline")
    is_training_running: bool = Field(..., description="True if training loop is actively running")
    message: str = Field(..., description="Human-readable status message")
    error: Optional[str] = Field(None, description="Error details if phase is 'error'")
    details: Optional[dict] = Field(None, description="Phase-specific info, e.g. {'model_size': '8B'}")


class TrainingProgress(BaseModel):
    """Training progress metrics - for streaming or polling"""
    job_id: str = Field(..., description="Training job identifier")
    step: int = Field(..., description="Current training step")
    total_steps: int = Field(..., description="Total training steps")
    loss: float = Field(..., description="Current loss value")
    learning_rate: float = Field(..., description="Current learning rate")
    progress_percent: float = Field(..., description="Progress percentage (0.0 to 100.0)")
    epoch: Optional[int] = Field(None, description="Current epoch")
    elapsed_seconds: Optional[float] = Field(None, description="Time elapsed since training started")
    eta_seconds: Optional[float] = Field(None, description="Estimated time remaining")
