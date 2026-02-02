"""
Unsloth Training Backend
Integrates Unsloth training capabilities with the Gradio UI
"""
import torch
torch._dynamo.config.recompile_limit = 64
from unsloth import FastLanguageModel, FastVisionModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

import os
import json
import threading
import math
import logging
from typing import Optional, Callable
from dataclasses import dataclass
import pandas as pd
from datasets import Dataset, load_dataset

# Add the parent directory to sys.path to import unsloth modules
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.models import is_vision_model
from utils.datasets.dataset_utils import format_and_template_dataset
from utils.datasets.dataset_utils import MODEL_TO_TEMPLATE_MAPPER, TEMPLATE_TO_RESPONSES_MAPPER
from trl import SFTTrainer, SFTConfig

# Import Unsloth trainers
#from unsloth_compiled_cache.UnslothSFTTrainer import _UnslothSFTTrainer as SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingProgress:
    """Training progress tracking"""
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    is_training: bool = False
    is_completed: bool = False
    error: Optional[str] = None
    status_message: str = "Ready to train"  # Current stage message

class UnslothTrainer:
    """
    Unsloth Training Backend for Gradio UI Integration
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_thread = None
        self.training_progress = TrainingProgress()
        self.progress_callbacks = []
        self.is_training = False
        self.should_stop = False

        # Model state tracking
        self.is_vlm = False
        self.model_name = None

        # Thread safety
        self._lock = threading.Lock()

        # Store training context for later transfer
        self.training_context = {
            'base_model_name': None,
            'output_dir': None,
            'is_lora': True,  # Default to LoRA
        }

    def add_progress_callback(self, callback: Callable[[TrainingProgress], None]):
        """Add callback for training progress updates"""
        self.progress_callbacks.append(callback)

    def _update_progress(self, **kwargs):
        """Update training progress and notify callbacks"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.training_progress, key):
                    setattr(self.training_progress, key, value)

            # Notify all callbacks
            for callback in self.progress_callbacks:
                try:
                    callback(self.training_progress)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

    def load_model(self,
                   model_name: str,
                   max_seq_length: int = 2048,
                   load_in_4bit: bool = True,
                   hf_token: Optional[str] = None) -> bool:
        """Load model for training (supports both text and vision models)"""
        try:
            print("\nClearing GPU memory before training...")
            torch.cuda.empty_cache()
            import gc
            gc.collect()

            # Detect if this is a vision model first
            self.is_vlm = is_vision_model(model_name)
            self.model_name = model_name

            logger.info(f"Model type detected: {'Vision' if self.is_vlm else 'Text'}")

            # Reset training state for new run
            self._update_progress(
                is_training=True,
                is_completed=False,
                error=None,
                step=0,
                loss=0.0,
                epoch=0
            )

            # Update UI immediately with loading message
            model_display = model_name.split('/')[-1] if '/' in model_name else model_name
            self._update_progress(
                status_message=f"Loading {'vision' if self.is_vlm else 'text'} model... {model_display}"
            )

            print(f"\nLoading {'vision' if self.is_vlm else 'text'} model: {model_name}")

            # Set HF token if provided
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token


            # Branch based on model type
            if self.is_vlm:
                # Load vision model - returns (model, tokenizer)
                self.model, self.tokenizer = FastVisionModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    dtype=None,  # Auto-detect
                    load_in_4bit=load_in_4bit,
                    token=hf_token,
                )
                logger.info("Loaded vision model")
            else:
                # Load text model - returns (model, tokenizer)
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    dtype=None,  # Auto-detect
                    load_in_4bit=load_in_4bit,
                    token=hf_token,
                )
                logger.info("Loaded text model")

            if self.should_stop:
                return False

            self._update_progress(status_message="Model loaded successfully")
            print("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._update_progress(error=str(e), is_training=False)
            return False

    def prepare_model_for_training(self,
                                   use_lora: bool = True,
                                   # Vision-specific LoRA parameters (only used if is_vlm=True)
                                   finetune_vision_layers: bool = True,
                                   finetune_language_layers: bool = True,
                                   finetune_attention_modules: bool = True,
                                   finetune_mlp_modules: bool = True,
                                   # Standard LoRA parameters
                                   target_modules: list = None,
                                   lora_r: int = 16,
                                   lora_alpha: int = 16,
                                   lora_dropout: float = 0.0,
                                   use_gradient_checkpointing: str = "unsloth",
                                   use_rslora: bool = False,
                                   use_loftq: bool = False) -> bool:
        """
        Prepare model for training (with optional LoRA).
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")


            # Full finetuning mode - skip PEFT entirely
            if not use_lora:
                self._update_progress(status_message="Full finetuning mode - no LoRA adapters")
                print("Full finetuning mode - training all parameters\n")
                return True

            # LoRA/QLoRA mode - apply PEFT
            if target_modules is None or (isinstance(target_modules, list) and len(target_modules) == 0):
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]

            # Validate and normalize gradient_checkpointing
            # Must be one of: True, False, or "unsloth"
            if isinstance(use_gradient_checkpointing, str):
                use_gradient_checkpointing = use_gradient_checkpointing.strip().lower()
                if use_gradient_checkpointing == "" or use_gradient_checkpointing == "unsloth":
                    use_gradient_checkpointing = "unsloth"
                elif use_gradient_checkpointing in ("true", "1", "yes"):
                    use_gradient_checkpointing = True
                elif use_gradient_checkpointing in ("false", "0", "no"):
                    use_gradient_checkpointing = False
                else:
                    # Invalid value, default to "unsloth"
                    logger.warning(f"Invalid gradient_checkpointing value: {use_gradient_checkpointing}, defaulting to 'unsloth'")
                    use_gradient_checkpointing = "unsloth"
            elif use_gradient_checkpointing not in (True, False, "unsloth"):
                # Invalid type or value, default to "unsloth"
                logger.warning(f"Invalid gradient_checkpointing type/value: {use_gradient_checkpointing}, defaulting to 'unsloth'")
                use_gradient_checkpointing = "unsloth"

            # Verify model is loaded
            if self.model is None:
                error_msg = "Model is None - model was not loaded properly"
                logger.error(error_msg)
                self._update_progress(error=error_msg)
                return False

            # Check if model has the expected attributes
            if not hasattr(self.model, 'config'):
                error_msg = "Model does not have config attribute - model may not be loaded correctly"
                logger.error(error_msg)
                self._update_progress(error=error_msg)
                return False

            print(f"Configuring LoRA adapters (r={lora_r}, alpha={lora_alpha})...\n")
            print(f"Gradient checkpointing: {use_gradient_checkpointing} (type: {type(use_gradient_checkpointing).__name__})\n")

            # Branch based on vision vs text
            if self.is_vlm:
                # Vision model LoRA
                print(f"Vision model LoRA configuration:")
                print(f"  - Finetune vision layers: {finetune_vision_layers}")
                print(f"  - Finetune language layers: {finetune_language_layers}")
                print(f"  - Finetune attention modules: {finetune_attention_modules}")
                print(f"  - Finetune MLP modules: {finetune_mlp_modules}\n")

                self.model = FastVisionModel.get_peft_model(
                    self.model,
                    finetune_vision_layers=finetune_vision_layers,
                    finetune_language_layers=finetune_language_layers,
                    finetune_attention_modules=finetune_attention_modules,
                    finetune_mlp_modules=finetune_mlp_modules,
                    r=lora_r,
                    target_modules=target_modules,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    random_state=3407,
                    use_rslora=use_rslora,
                    loftq_config={"loftq_bits": 4, "loftq_iter": 1} if use_loftq else None,
                )
            else:
                # Text model LoRA
                print(f"Text model LoRA configuration:")
                print(f"  - Target modules: {target_modules}\n")

                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=lora_r,
                    target_modules=target_modules,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    random_state=3407,
                    use_rslora=use_rslora,
                    loftq_config={"loftq_bits": 4, "loftq_iter": 1} if use_loftq else None,
                )

            # Check if stopped during LoRA preparation
            if self.should_stop:
                print("Stopped during LoRA configuration\n")
                return False

            self._update_progress(status_message="LoRA adapters configured")
            print("LoRA adapters configured successfully\n")
            return True

        except Exception as e:
            import traceback
            import sys
            error_details = f"{type(e).__name__}: {str(e)}" if str(e) else f"{type(e).__name__} (no message)"
            full_traceback = traceback.format_exc()
            logger.error(f"Error preparing model: {error_details}")
            logger.error(f"Full traceback:\n{full_traceback}")
            print(f"\n[ERROR] Error preparing model: {error_details}", file=sys.stderr, flush=True)
            print(f"[ERROR] Full traceback:\n{full_traceback}", file=sys.stderr, flush=True)
            self._update_progress(error=error_details)
            return False

    def load_and_format_dataset(self,
                     dataset_source: str,
                     format_type: str = "auto",
                     local_datasets: list = None) -> Optional[Dataset]:
        """
        Load and prepare dataset for training
        """
        try:
            dataset = None

            if local_datasets:
                # Load local datasets
                all_data = []
                for dataset_file in local_datasets:
                    # dataset_file may already be an absolute path from routes/training.py
                    if os.path.isabs(dataset_file):
                        file_path = dataset_file
                    else:
                        # Fallback: try relative to assets/datasets
                        script_dir = Path(__file__).parent.parent
                        assets_datasets_dir = script_dir / "assets" / "datasets"
                        file_path = assets_datasets_dir / dataset_file
                    
                    if str(file_path).endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_data.extend(data)
                            else:
                                all_data.append(data)
                    elif str(file_path).endswith('.csv'):
                        df = pd.read_csv(file_path)
                        all_data.extend(df.to_dict('records'))

                if all_data:
                    dataset = Dataset.from_list(all_data)

                    # Check if stopped during dataset loading
                    if self.should_stop:
                        print("Stopped during dataset loading\n")
                        return None

                    self._update_progress(status_message=f"Loaded {len(all_data)} samples from local files")
                    print(f"Loaded {len(all_data)} samples from local files\n")

            elif dataset_source:
                # Load from Hugging Face
                dataset = load_dataset(dataset_source, split="train")

                # Check if stopped during dataset loading
                if self.should_stop:
                    print("Stopped during dataset loading\n")
                    return None

                self._update_progress(status_message=f"Loaded dataset from HuggingFace: {dataset_source}")
                print(f"Loaded dataset from Hugging Face: {dataset_source}\n")

            if dataset is None:
                raise ValueError("No dataset provided")

            # Check if stopped before applying template
            if self.should_stop:
                print("Stopped before applying chat template\n")
                return None

            # NEW: Use unified format_and_template_dataset
            print(f"Formatting dataset with format_type='{format_type}'...\n")

            #breakpoint()
            dataset_info = format_and_template_dataset(
                dataset,
                model_name=self.model_name,
                tokenizer=self.tokenizer,  # Works for both text and vision models
                is_vlm=self.is_vlm,
                format_type=format_type,  # "auto", "alpaca", "chatml", "sharegpt"
                dataset_name=dataset_source,
            )

            # Check if stopped during formatting
            if self.should_stop:
                print("Stopped during dataset formatting\n")
                return None

            self._update_progress(status_message=f"Dataset formatted and ready for training")
            print(f"Dataset formatted successfully\n")
            return dataset_info

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self._update_progress(error=str(e))
            return None

    def start_training(self,
                       dataset: Dataset,
                       output_dir: str = "./outputs",
                       num_epochs: int = 3,
                       learning_rate: float = 5e-5,
                       batch_size: int = 2,
                       gradient_accumulation_steps: int = 4,
                       warmup_steps: int = None,
                       warmup_ratio: float = None,
                       max_steps: int = 0,
                       save_steps: int = 0,
                       weight_decay: float = 0.01,
                       random_seed: int = 3407,
                       packing: bool = False,
                       train_on_completions: bool = False,
                       enable_wandb: bool = False,
                       wandb_project: str = "unsloth-training",
                       wandb_token: str = None,
                       enable_tensorboard: bool = False,
                       tensorboard_dir: str = "runs",
                       **kwargs) -> bool:
        """Start training in a separate thread"""

        if self.is_training:
            logger.warning("Training already in progress")
            return False


        if self.model is None or self.tokenizer is None:
            self._update_progress(error="Model not loaded")
            return False

        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self._train_worker,
            args=(dataset,),
            kwargs={
                'output_dir': output_dir,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'warmup_steps': warmup_steps,
                'warmup_ratio': warmup_ratio,
                'max_steps': max_steps,
                'save_steps': save_steps,
                'weight_decay': weight_decay,
                'random_seed': random_seed,
                'packing': packing,
                'train_on_completions': train_on_completions,
                'enable_wandb': enable_wandb,
                'wandb_project': wandb_project,
                'wandb_token': wandb_token,
                'enable_tensorboard': enable_tensorboard,
                'tensorboard_dir': tensorboard_dir,
                **kwargs
            }
        )

        self.should_stop = False
        self.training_thread.start()
        return True

    def _train_worker(self, dataset: Dataset, **training_args):
        """Worker function for training (runs in separate thread)"""
        try:
            self._update_progress(is_training=True, error=None)

            # Setup logging
            if training_args.get('enable_wandb', False) and training_args.get('wandb_token'):
                os.environ["WANDB_API_KEY"] = training_args['wandb_token']
                import wandb
                wandb.init(project=training_args.get('wandb_project', 'unsloth-training'))

            # Create output directory
            output_dir = training_args.get('output_dir', './outputs')
            os.makedirs(output_dir, exist_ok=True)

            # ========== DATA COLLATOR SELECTION ==========
            # Detect special model types
            model_name_lower = self.model_name.lower()
            is_deepseek_ocr = "deepseek" in model_name_lower and "ocr" in model_name_lower

            print("Configuring data collator...\n")

            data_collator = None  # Default to built-in data collator
            if is_deepseek_ocr:
                # Special DeepSeek OCR collator - auto-install if needed
                print("Detected DeepSeek OCR model\n")
                # Ensure DeepSeek OCR module is installed
                if not _ensure_deepseek_ocr_installed():
                    error_msg = (
                        "Failed to install DeepSeek OCR module. "
                        "Please install manually: "
                        "from huggingface_hub import snapshot_download; "
                        "snapshot_download('unsloth/DeepSeek-OCR', local_dir='deepseek_ocr')"
                    )
                    logger.error(error_msg)
                    self._update_progress(error=error_msg, is_training=False)
                    return

                try:
                    from backend.data_utils import DeepSeekOCRDataCollator

                    print("Configuring DeepSeek OCR data collator...\n")
                    FastVisionModel.for_training(self.model)
                    data_collator = DeepSeekOCRDataCollator(
                        tokenizer=self.tokenizer,
                        model=self.model,
                        image_size=640,
                        base_size=1024,
                        crop_mode=True,
                        train_on_responses_only=training_args.get('train_on_completions', False),
                    )
                    print("DeepSeek OCR data collator configured successfully\n")

                except Exception as e:
                    logger.error(f"Failed to configure DeepSeek OCR collator: {e}")
                    error_msg = f"Error configuring DeepSeek OCR: {str(e)}"
                    self._update_progress(error=error_msg, is_training=False)
                    return

            elif self.is_vlm:
                # Standard VLM collator
                print("Using UnslothVisionDataCollator for vision model\n")
                from unsloth.trainer import UnslothVisionDataCollator

                FastVisionModel.for_training(self.model)
                data_collator = UnslothVisionDataCollator(self.model, self.tokenizer)
                print("Vision data collator configured\n")

            # ========== TRAINING CONFIGURATION ==========
            # Handle epochs vs max_steps properly
            max_steps_val = training_args.get('max_steps', 0)
            num_epochs_val = training_args.get('num_epochs', 3)

            # Handle warmup_steps vs warmup_ratio
            warmup_steps_val = training_args.get('warmup_steps', None)
            warmup_ratio_val = training_args.get('warmup_ratio', None)
            
            config_args = {
                "per_device_train_batch_size": training_args.get('batch_size', 2),
                "gradient_accumulation_steps": training_args.get('gradient_accumulation_steps', 4),
                "num_train_epochs": training_args.get('num_epochs', 3),  # Default to epochs
                "learning_rate": training_args.get('learning_rate', 2e-4),
                "fp16": not is_bfloat16_supported(),
                "bf16": is_bfloat16_supported(),
                "logging_steps": 1,
                "weight_decay": training_args.get('weight_decay', 0.01),
                "seed": training_args.get('random_seed', 3407),
                "output_dir": output_dir,
                "report_to": ["wandb"] if training_args.get('enable_wandb', False) else "none",
            }
            
            # Add warmup parameter - use warmup_ratio if provided, otherwise warmup_steps
            if warmup_ratio_val is not None:
                config_args["warmup_ratio"] = warmup_ratio_val
                print(f"Using warmup_ratio: {warmup_ratio_val}\n")
            elif warmup_steps_val is not None:
                config_args["warmup_steps"] = warmup_steps_val
                print(f"Using warmup_steps: {warmup_steps_val}\n")
            else:
                # Default to warmup_steps if neither provided
                config_args["warmup_steps"] = 5
                print(f"Using default warmup_steps: 5\n")

            #  If max_steps is specified, use it instead of epochs
            max_steps_val = training_args.get('max_steps', 0)
            if max_steps_val and max_steps_val > 0:
                del config_args["num_train_epochs"]  # Remove epochs
                config_args["max_steps"] = max_steps_val  # Use steps instead
                print(f"Training for {max_steps_val} steps\n")
            else:
                print(f"Training for {config_args['num_train_epochs']} epochs\n")

            # Add model-specific parameters
            # Use optim and lr_scheduler_type from training_args if provided, otherwise use defaults
            optim_value = training_args.get('optim', "adamw_8bit")
            lr_scheduler_type_value = training_args.get('lr_scheduler_type', "linear")
            
            if self.is_vlm:
                # Vision-specific config
                print("Configuring vision model training parameters\n")
                # Use provided values or defaults for vision models
                optim_value = training_args.get('optim', "adamw_torch_fused")
                lr_scheduler_type_value = training_args.get('lr_scheduler_type', "cosine")
                config_args.update({
                    "optim": optim_value,
                    "lr_scheduler_type": lr_scheduler_type_value,
                    "gradient_checkpointing": True,
                    "gradient_checkpointing_kwargs": {"use_reentrant": False},
                    "max_grad_norm": 0.3,  # Recommended for vision models
                    "remove_unused_columns": False,
                    "dataset_text_field": "",
                    "dataset_kwargs": {"skip_prepare_dataset": True},
                    "max_length": training_args.get('max_seq_length', 2048),
                })
            else:
                print("Configuring text model training parameters\n")
                config_args.update({
                    "optim": optim_value,
                    "lr_scheduler_type": lr_scheduler_type_value,
                    "dataset_text_field": "text",
                })

                # Only add packing for text models (not DeepSeek OCR which is VLM)
                if not is_deepseek_ocr:
                    packing_enabled = training_args.get('packing', False)
                    config_args["packing"] = packing_enabled
                    print(f"Sequence packing: {'enabled' if packing_enabled else 'disabled'}\n")

            print(f"The configuration is: {config_args}")

            print("Training configuration prepared\n")
            # ========== TRAINER INITIALIZATION ==========
            if self.is_vlm:
                self.trainer = SFTTrainer(
                    model=self.model,
                    train_dataset=dataset['dataset'],
                    processing_class = self.tokenizer.tokenizer,
                    data_collator=data_collator,
                    args=SFTConfig(**config_args),
                )
            else:
                self.trainer = SFTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    train_dataset=dataset['dataset'],
                    data_collator=data_collator,
                    args=SFTConfig(**config_args),
                )
            print("Trainer initialized\n")

            # ========== TRAIN ON RESPONSES ONLY ==========
            # Determine if we should train on responses only
            instruction_part = None
            response_part = None
            train_on_responses_enabled = training_args.get('train_on_completions', False)

            # DeepSeek OCR handles this internally in its collator, so skip
            if train_on_responses_enabled and not (is_deepseek_ocr or dataset["final_format"].lower() == 'alpaca'):
                try:
                    print("Configuring train on responses only...\n")

                    # Get the template mapping for this model
                    model_name_lower = self.model_name.lower()

                    if model_name_lower in MODEL_TO_TEMPLATE_MAPPER:
                        template_name = MODEL_TO_TEMPLATE_MAPPER[model_name_lower]
                        print(f"Detected template: {template_name}\n")

                        if template_name in TEMPLATE_TO_RESPONSES_MAPPER:
                            instruction_part = TEMPLATE_TO_RESPONSES_MAPPER[template_name]["instruction"]
                            response_part = TEMPLATE_TO_RESPONSES_MAPPER[template_name]["response"]

                            print(f"Instruction marker: {instruction_part[:50]}...\n")
                            print(f"Response marker: {response_part[:50]}...\n")
                        else:
                            print(f"No response mapping found for template: {template_name}\n")
                            train_on_responses_enabled = False
                    else:
                        print(f"No template mapping found for model: {self.model_name}\n")
                        train_on_responses_enabled = False

                except Exception as e:
                    logger.warning(f"Could not configure train on responses: {e}")
                    train_on_responses_enabled = False

            # Apply train on responses only if we have valid parts
            if train_on_responses_enabled and instruction_part and response_part and not (is_deepseek_ocr or dataset["final_format"].lower() == 'alpaca'):
                try:
                    from unsloth.chat_templates import train_on_responses_only

                    self.trainer = train_on_responses_only(
                        self.trainer,
                        instruction_part=instruction_part,
                        response_part=response_part,
                    )
                    print("Train on responses only configured successfully\n")
                except Exception as e:
                    logger.warning(f"Failed to apply train on responses only: {e}")
                    train_on_responses_enabled = False
            else:
                if train_on_responses_enabled and is_deepseek_ocr:
                    print("Train on responses handled by DeepSeek OCR collator\n")
                else:
                    print("Training on full sequences (including prompts)\n")

            # Add custom callback for progress tracking
            from transformers import TrainerCallback

            class ProgressCallback(TrainerCallback):
                def __init__(self, trainer_instance):
                    self.trainer_instance = trainer_instance

                def on_train_begin(self, args, state, control, **kwargs):
                    """Called at the beginning of training"""
                    pass

                def on_log(self, args, state, control, logs=None, **kwargs):
                    """Called when logging occurs"""
                    if logs:
                        # Get loss from either 'loss' or 'train_loss' key
                        loss_value = logs.get('loss', logs.get('train_loss', 0.0))
                        self.trainer_instance._update_progress(
                            step=state.global_step,
                            epoch=round(state.epoch, 2) if state.epoch else 0,  # Round epoch to 2 decimals
                            loss=loss_value,
                            learning_rate=logs.get('learning_rate', 0.0),
                            status_message=""  # Clear status message so metrics show
                        )

                def on_epoch_end(self, args, state, control, **kwargs):
                    """Called at the end of each epoch"""
                    self.trainer_instance._update_progress(
                        epoch=state.epoch,
                        step=state.global_step
                    )

                def on_step_end(self, args, state, control, **kwargs):
                    """Called at the end of each step"""
                    # Check if we should stop training
                    if self.trainer_instance.should_stop:
                        print(f"Stop detected at step {state.global_step}\n")
                        control.should_training_stop = True
                        return control

            # ========== PROGRESS TRACKING ==========
            progress_callback = ProgressCallback(self)
            self.trainer.add_callback(progress_callback)

            num_samples = len(dataset["dataset"])
            batch_size = training_args.get('batch_size', 2)
            grad_accum = training_args.get('gradient_accumulation_steps', 4)
            num_epochs = training_args.get('num_epochs', 3)
            max_steps_val = training_args.get('max_steps', 0)

            # Step 1: Calculate dataloader length (number of batches)
            len_dataloader = math.ceil(num_samples / batch_size)

            # Step 2: Calculate steps per epoch (following transformers logic)
            num_update_steps_per_epoch = max(
                len_dataloader // grad_accum + int(len_dataloader % grad_accum > 0),
                1
            )

            # Step 3: Determine total steps based on max_steps or epochs
            if max_steps_val and max_steps_val > 0:
                # Use max_steps if specified
                total_steps = max_steps_val
                print(f"Progress tracking: {total_steps} steps (max_steps)\n")
            else:
                # Calculate from epochs
                total_steps = num_update_steps_per_epoch * num_epochs
                print(f"Progress tracking: {total_steps} steps ({num_epochs} epochs × {num_update_steps_per_epoch} steps/epoch)\n")

            self._update_progress(total_steps=total_steps)

            # ========== START TRAINING ==========
            self._update_progress(status_message="Starting training...")
            print("Starting training...\n")
            self.trainer.train()

            # ========== SAVE MODEL ==========
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            print(f"\nTraining completed! Model saved to {output_dir}\n")

            self._update_progress(
                is_training=False,
                is_completed=True,
                #status_message=status_msg
                status_message=f"Training completed! Model saved to {output_dir}",
            )

        except Exception as e:
            logger.error(f"Training error: {e}")
            self._update_progress(is_training=False, error=str(e))

        finally:
            self.is_training = False

    def stop_training(self):
        """Stop ongoing training"""
        print("\nStopping training...")
        self.should_stop = True
        self.is_training = False
        # Clear the status message so timer doesn't show stale status
        self._update_progress(is_training=False, status_message="")

        # If trainer exists, try to stop it gracefully
        if self.trainer:
            try:
                # The callback will catch should_stop flag and stop the training loop
                print("Training will stop at next step...\n")
            except Exception as e:
                logger.error(f"Error stopping trainer: {e}")

    def get_training_progress(self) -> TrainingProgress:
        """Get current training progress"""
        with self._lock:
            return self.training_progress

    def cleanup(self):
        """Cleanup resources"""
        if self.trainer:
            self.trainer = None
        if self.model:
            self.model = None
        if self.tokenizer:
            self.tokenizer = None

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ensure_deepseek_ocr_installed():
    """
    Auto-install DeepSeek OCR module if not available.
    Downloads from HuggingFace hub as a local module.

    Returns:
        bool: True if available (either already installed or just installed)
    """
    try:
        # Try importing to see if already available
        from deepseek_ocr.modeling_deepseekocr import format_messages
        logger.info("DeepSeek OCR module already available")
        return True
    except ImportError:
        pass

    try:
        logger.info("DeepSeek OCR module not found. Auto-installing from HuggingFace...")
        print("\n Downloading DeepSeek OCR module from HuggingFace...\n")

        from huggingface_hub import snapshot_download
        import sys
        import os

        # Get the script directory to install locally
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)  # Go up to project root

        # Download to project root as 'deepseek_ocr' folder
        local_dir = os.path.join(parent_dir, "deepseek_ocr")

        snapshot_download(
            "unsloth/DeepSeek-OCR",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

        # Add to sys.path if not already there
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Try importing again
        from deepseek_ocr.modeling_deepseekocr import format_messages

        logger.info("DeepSeek OCR module installed successfully")
        print("DeepSeek OCR module installed successfully!\n")
        return True

    except Exception as e:
        logger.error(f"Failed to install DeepSeek OCR module: {e}")
        print(f"\n❌ Failed to install DeepSeek OCR module: {e}\n")
        return False

# Global trainer instance
_trainer_instance = None

def get_trainer() -> UnslothTrainer:
    """Get global trainer instance"""
    global _trainer_instance
    if _trainer_instance is None:
        _trainer_instance = UnslothTrainer()
    return _trainer_instance
