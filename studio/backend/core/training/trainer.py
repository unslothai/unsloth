"""
Unsloth Training Backend
Integrates Unsloth training capabilities with the FastAPI backend
"""
import os
# Prevent tokenizer parallelism deadlocks when datasets uses multiprocessing fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from utils.hardware import clear_gpu_cache
torch._dynamo.config.recompile_limit = 64
from unsloth import FastLanguageModel, FastVisionModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

import json
import threading
import math
import logging
import time
from typing import Optional, Callable
from dataclasses import dataclass
import pandas as pd
from datasets import Dataset, load_dataset

# Add the parent directory to sys.path to import unsloth modules
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.models import is_vision_model
from utils.datasets import format_and_template_dataset
from utils.datasets import MODEL_TO_TEMPLATE_MAPPER, TEMPLATE_TO_RESPONSES_MAPPER
from trl import SFTTrainer, SFTConfig

# Import Unsloth trainers
#from unsloth_compiled_cache.UnslothSFTTrainer import _UnslothSFTTrainer as SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingProgress:
    """Training progress tracking"""
    epoch: float = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    is_training: bool = False
    is_completed: bool = False
    error: Optional[str] = None
    status_message: str = "Ready to train"  # Current stage message
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    grad_norm: Optional[float] = None
    num_tokens: Optional[int] = None
    eval_loss: Optional[float] = None

class UnslothTrainer:
    """
    Unsloth Training Backend
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
        self.save_on_stop = True

        # Model state tracking
        self.is_vlm = False
        self.model_name = None

        # Training metrics tracking
        self.training_start_time: Optional[float] = None
        self.batch_size: Optional[int] = None
        self.max_seq_length: Optional[int] = None
        self.gradient_accumulation_steps: Optional[int] = None

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
                   hf_token: Optional[str] = None,
                   is_dataset_multimodal: bool = False) -> bool:
        """Load model for training (supports both text and vision models)"""
        try:
            print("\nClearing GPU memory before training...")
            clear_gpu_cache()

            # Detect if this is a vision model AND dataset is multimodal
            # A vision-capable model with a text-only dataset should use FastLanguageModel
            self.is_vlm = is_vision_model(model_name) and is_dataset_multimodal
            self.model_name = model_name

            logger.info(f"Model architecture is vision: {is_vision_model(model_name)}")
            logger.info(f"Dataset is multimodal: {is_dataset_multimodal}")
            logger.info(f"Using VLM path: {self.is_vlm}")

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
            # "all-linear" is a PEFT keyword that targets every linear layer
            if isinstance(target_modules, list) and "all-linear" in target_modules:
                if len(target_modules) == 1:
                    target_modules = "all-linear"
                else:
                    target_modules = [m for m in target_modules if m != "all-linear"]
            elif target_modules is None or (isinstance(target_modules, list) and len(target_modules) == 0):
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
                     local_datasets: list = None,
                     custom_format_mapping: dict = None,
                     subset: str = None,
                     train_split: str = "train",
                     eval_split: str = None) -> Optional[tuple]:
        """
        Load and prepare dataset for training.

        Strategy: format first, then split — ensures both train and eval
        portions are properly formatted and templated.

        Returns:
            Tuple of (dataset_info, eval_dataset) or None on error.
            eval_dataset may be None if no eval split is available.
        """
        try:
            dataset = None
            eval_dataset = None
            has_separate_eval_source = False  # True if eval comes from a separate HF split

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
                load_kwargs = {"path": dataset_source, "split": train_split or "train"}
                if subset:
                    load_kwargs["name"] = subset
                dataset = load_dataset(**load_kwargs)

                # Check if stopped during dataset loading
                if self.should_stop:
                    print("Stopped during dataset loading\n")
                    return None

                self._update_progress(status_message=f"Loaded dataset from HuggingFace: {dataset_source}")
                print(f"Loaded dataset from Hugging Face: {dataset_source}\n")

                # Resolve eval split from a separate HF split (explicit or auto-detected)
                if eval_split:
                    # Explicit eval split provided - load it directly
                    print(f"Loading explicit eval split: '{eval_split}'\n")
                    eval_load_kwargs = {"path": dataset_source, "split": eval_split}
                    if subset:
                        eval_load_kwargs["name"] = subset
                    eval_dataset = load_dataset(**eval_load_kwargs)
                    has_separate_eval_source = True
                    print(f"Loaded eval split '{eval_split}' with {len(eval_dataset)} rows\n")
                else:
                    # Auto-detect eval split from HF (returns a separate dataset, or None)
                    eval_dataset = self._auto_detect_eval_split_from_hf(
                        dataset_source=dataset_source,
                        subset=subset,
                    )
                    if eval_dataset is not None:
                        has_separate_eval_source = True

            if dataset is None:
                raise ValueError("No dataset provided")

            # Check if stopped before applying template
            if self.should_stop:
                print("Stopped before applying chat template\n")
                return None

            # ========== FORMAT FIRST ==========
            print(f"Formatting dataset with format_type='{format_type}'...\n")

            dataset_info = format_and_template_dataset(
                dataset,
                model_name=self.model_name,
                tokenizer=self.tokenizer,
                is_vlm=self.is_vlm,
                format_type=format_type,
                dataset_name=dataset_source,
                custom_format_mapping=custom_format_mapping,
            )

            # Check if stopped during formatting
            if self.should_stop:
                print("Stopped during dataset formatting\n")
                return None

            self._update_progress(status_message=f"Dataset formatted and ready for training")
            print(f"Dataset formatted successfully\n")

            # ========== THEN SPLIT ==========
            if has_separate_eval_source and eval_dataset is not None:
                # Eval came from a separate HF split — format it too
                print(f"Formatting eval dataset ({len(eval_dataset)} rows)...\n")
                eval_info = format_and_template_dataset(
                    eval_dataset,
                    model_name=self.model_name,
                    tokenizer=self.tokenizer,
                    is_vlm=self.is_vlm,
                    format_type=format_type,
                    dataset_name=dataset_source,
                    custom_format_mapping=custom_format_mapping,
                )
                eval_dataset = eval_info["dataset"]
                print(f"Eval dataset formatted successfully\n")
            elif not has_separate_eval_source:
                # No separate eval source — split the already-formatted dataset
                formatted_dataset = dataset_info["dataset"]
                split_result = self._resolve_eval_split_from_dataset(formatted_dataset)
                if split_result is not None:
                    train_portion, eval_dataset = split_result
                    dataset_info["dataset"] = train_portion

            return (dataset_info, eval_dataset)

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self._update_progress(error=str(e))
            return None

    def _auto_detect_eval_split_from_hf(self, dataset_source: str,
                                         subset: str) -> Optional[Dataset]:
        """Auto-detect an eval split from HF dataset (separate named split only)."""
        try:
            from datasets import get_dataset_split_names
            load_kwargs = {"path": dataset_source}
            if subset:
                load_kwargs["name"] = subset
            available_splits = get_dataset_split_names(**load_kwargs)
            print(f"Available splits: {available_splits}\n")

            # Check for common eval split names
            for candidate in ["eval", "validation", "valid", "val", "test"]:
                if candidate in available_splits:
                    eval_load_kwargs = {"path": dataset_source, "split": candidate}
                    if subset:
                        eval_load_kwargs["name"] = subset
                    candidate_ds = load_dataset(**eval_load_kwargs)
                    if len(candidate_ds) >= 16:
                        print(f"Auto-detected eval split '{candidate}' with {len(candidate_ds)} rows\n")
                        return candidate_ds
                    else:
                        print(f"Found eval split '{candidate}' but only {len(candidate_ds)} rows (< 16), skipping\n")

        except Exception as e:
            logger.warning(f"Could not check dataset splits: {e}")

        # No separate HF eval split found — caller will handle programmatic splitting
        return None

    def _resolve_eval_split_from_dataset(self, dataset) -> Optional[tuple]:
        """Split a dataset into train and eval portions.

        Returns:
            Tuple of (train_dataset, eval_dataset), or None if dataset too small.
        """
        MIN_EVAL_ROWS = 16
        MIN_TOTAL_ROWS = 32  # Need at least 16 train + 16 eval

        n = len(dataset)
        if n < MIN_TOTAL_ROWS:
            print(f"Dataset too small ({n} rows) for eval split, skipping eval\n")
            return None

        eval_size = max(MIN_EVAL_ROWS, min(128, int(0.05 * n)))
        # Ensure we don't take more than half the dataset
        eval_size = min(eval_size, n // 2)

        print(f"Auto-splitting: {eval_size} rows for eval from {n} total\n")
        split_result = dataset.train_test_split(test_size=eval_size, seed=3407)
        print(f"Split complete: {len(split_result['train'])} train, {len(split_result['test'])} eval\n")
        return (split_result['train'], split_result['test'])

    def start_training(self,
                       dataset: Dataset,
                       eval_dataset: Dataset = None,
                       eval_steps: float = 0.01,
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
                'eval_dataset': eval_dataset,
                'eval_steps': eval_steps,
                **kwargs
            }
        )

        self.should_stop = False
        self.is_training = True
        try:
            self.training_thread.start()
            return True
        except Exception as e:
            self.is_training = False
            logger.error(f"Failed to start training thread: {e}")
            return False

    def _train_worker(self, dataset: Dataset, **training_args):
        """Worker function for training (runs in separate thread)"""
        try:
            # Store training parameters for metrics calculation
            self.batch_size = training_args.get('batch_size', 2)
            self.max_seq_length = training_args.get('max_seq_length', 2048)
            self.gradient_accumulation_steps = training_args.get('gradient_accumulation_steps', 4)
            
            # Set training start time
            self.training_start_time = time.time()
            
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
                "include_num_input_tokens_seen": True,  # Enable token counting
                "dataset_num_proc": max(1, os.cpu_count() // 3),
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

            # Add save_steps if specified
            save_steps_val = training_args.get('save_steps', 0)
            if save_steps_val and save_steps_val > 0:
                config_args["save_steps"] = save_steps_val
                config_args["save_strategy"] = "steps"

            #  If max_steps is specified, use it instead of epochs
            max_steps_val = training_args.get('max_steps', 0)
            if max_steps_val and max_steps_val > 0:
                del config_args["num_train_epochs"]  # Remove epochs
                config_args["max_steps"] = max_steps_val  # Use steps instead
                print(f"Training for {max_steps_val} steps\n")
            else:
                print(f"Training for {config_args['num_train_epochs']} epochs\n")

            # ========== EVAL CONFIGURATION ==========
            eval_dataset = training_args.get('eval_dataset', None)
            eval_steps_val = training_args.get('eval_steps', 0.01)
            if eval_dataset is not None:
                config_args["eval_strategy"] = "steps"
                config_args["eval_steps"] = eval_steps_val
                print(f"Evaluation enabled: eval_steps={eval_steps_val} (fraction of total steps)\n")
                print(f"Eval dataset: {len(eval_dataset)} rows\n")
            else:
                print("No eval dataset — evaluation disabled\n")

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
                trainer_kwargs = {
                    "model": self.model,
                    "train_dataset": dataset['dataset'],
                    "processing_class": self.tokenizer.tokenizer,
                    "data_collator": data_collator,
                    "args": SFTConfig(**config_args),
                }
                if eval_dataset is not None:
                    trainer_kwargs["eval_dataset"] = eval_dataset
                self.trainer = SFTTrainer(**trainer_kwargs)
            else:
                trainer_kwargs = {
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "train_dataset": dataset['dataset'],
                    "data_collator": data_collator,
                    "args": SFTConfig(**config_args),
                }
                if eval_dataset is not None:
                    trainer_kwargs["eval_dataset"] = eval_dataset
                self.trainer = SFTTrainer(**trainer_kwargs)
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
                        current_step = state.global_step
                        
                        # Extract grad_norm from logs (available when gradient clipping is enabled)
                        grad_norm = logs.get('grad_norm', None)
                        
                        # Calculate elapsed_seconds
                        elapsed_seconds = None
                        if self.trainer_instance.training_start_time is not None:
                            elapsed_seconds = time.time() - self.trainer_instance.training_start_time
                        
                        # Calculate eta_seconds
                        eta_seconds = None
                        if elapsed_seconds is not None and current_step > 0:
                            total_steps = self.trainer_instance.training_progress.total_steps
                            if total_steps > 0:
                                steps_remaining = total_steps - current_step
                                if steps_remaining > 0:
                                    time_per_step = elapsed_seconds / current_step
                                    eta_seconds = time_per_step * steps_remaining
                        
                        # Extract num_tokens from TRL SFTTrainer state (real counter)
                        # Requires include_num_input_tokens_seen=True in SFTConfig
                        num_tokens = getattr(state, "num_input_tokens_seen", None)
                        
                        self.trainer_instance._update_progress(
                            step=current_step,
                            epoch=round(state.epoch, 2) if state.epoch else 0,
                            loss=loss_value,
                            learning_rate=logs.get('learning_rate', 0.0),
                            elapsed_seconds=elapsed_seconds,
                            eta_seconds=eta_seconds,
                            grad_norm=grad_norm,
                            num_tokens=num_tokens,
                            eval_loss=logs.get('eval_loss', None),
                            status_message=""
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
            if self.should_stop and self.save_on_stop:
                # Stopped by user — save model at current checkpoint
                self.trainer.save_model()
                self.tokenizer.save_pretrained(output_dir)
                print(f"\nTraining stopped. Model saved to {output_dir}\n")
                self._update_progress(
                    is_training=False,
                    status_message=f"Training stopped. Model saved to {output_dir}",
                )
            elif self.should_stop:
                # Cancelled by user — don't save
                print("\nTraining cancelled.\n")
                self._update_progress(
                    is_training=False,
                    status_message="Training cancelled.",
                )
            else:
                # Normal completion
                self.trainer.save_model()
                self.tokenizer.save_pretrained(output_dir)
                print(f"\nTraining completed! Model saved to {output_dir}\n")
                self._update_progress(
                    is_training=False,
                    is_completed=True,
                    status_message=f"Training completed! Model saved to {output_dir}",
                )

        except Exception as e:
            logger.error(f"Training error: {e}")
            self._update_progress(is_training=False, error=str(e))

        finally:
            self.is_training = False

    def stop_training(self, save: bool = True):
        """Stop ongoing training"""
        print(f"\nStopping training (save={save})...")
        self.should_stop = True
        self.save_on_stop = save
        stop_msg = (
            "Stopping training and saving checkpoint..."
            if save
            else "Cancelling training..."
        )
        self._update_progress(status_message=stop_msg)

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
        clear_gpu_cache()


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
