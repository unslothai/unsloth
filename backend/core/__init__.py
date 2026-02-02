"""
Unified core module for Unsloth backend
"""

# Inference
from .inference import InferenceBackend, get_inference_backend

# Training
from .training import UnslothTrainer, get_trainer, TrainingBackend, get_training_backend, create_training_handlers, TrainingProgress

# Configuration (from utils)
from utils.models import is_vision_model, ModelConfig, scan_trained_loras, load_model_defaults, get_base_model_from_lora

# Utilities (from utils)
from utils.paths import normalize_path, is_local_path, is_model_cached
from utils.utils import without_hf_auth, format_error_message, get_gpu_memory_info, search_hf_models
from utils.datasets.dataset_utils import format_and_template_dataset

__all__ = [
    # Inference
    'InferenceBackend',
    'get_inference_backend',

    # Training
    'UnslothTrainer',
    'get_trainer',
    'get_training_backend',
    'TrainingBackend',
    'create_training_handlers',
    'TrainingProgress',

    # Config
    'ModelConfig',
    'is_vision_model',
    'scan_trained_loras',
    'load_model_defaults',
    'get_base_model_from_lora',

    # Utils
    'search_hf_models',
    'format_and_template_dataset',
    'normalize_path',
    'is_local_path',
    'is_model_cached',
    'without_hf_auth',
    'format_error_message',
    'get_gpu_memory_info',
]
