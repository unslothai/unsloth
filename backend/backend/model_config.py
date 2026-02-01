"""
Model and LoRA configuration handling
"""
from transformers import AutoConfig
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .path_utils import normalize_path, is_local_path, is_model_cached
from .utils import without_hf_auth
import logging
from pathlib import Path
from typing import List, Tuple
import json
import yaml


logger = logging.getLogger(__name__)

# Model name mapping: maps all equivalent model names to their canonical YAML config file
# Format: "canonical_model_name.yaml": [list of all equivalent model names]
# Based on the model mapper provided - canonical filename is based on the first model name in the mapper
MODEL_NAME_MAPPING = {
    "unsloth_answerdotai_ModernBERT-large.yaml": [
        "answerdotai/ModernBERT-large",
    ],
    "unsloth_Qwen2.5-Coder-7B-Instruct-bnb-4bit.yaml": [
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ],
    "unsloth_codegemma-7b-bnb-4bit.yaml": [
        "unsloth/codegemma-7b-bnb-4bit",
        "unsloth/codegemma-7b",
        "google/codegemma-7b",
    ],
    "unsloth_ERNIE-4.5-21B-A3B-PT.yaml": [
        "unsloth/ERNIE-4.5-21B-A3B-PT",
    ],
    "unsloth_ERNIE-4.5-VL-28B-A3B-PT.yaml": [
        "unsloth/ERNIE-4.5-VL-28B-A3B-PT",
    ],
   
    "tiiuae_Falcon-H1-0.5B-Instruct.yaml": [
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "unsloth/Falcon-H1-0.5B-Instruct",
    ],
    "unsloth_functiongemma-270m-it.yaml": [
        "unsloth/functiongemma-270m-it-unsloth-bnb-4bit",
        "google/functiongemma-270m-it",
        "unsloth/functiongemma-270m-it-unsloth-bnb-4bit",
    ],
    "unsloth_gemma-2-2b.yaml": [
        "unsloth/gemma-2-2b-bnb-4bit",
        "google/gemma-2-2b",
    ],
    "unsloth_gemma-2-27b-bnb-4bit.yaml": [
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-9b",
        "google/gemma-2-9b",
        "unsloth/gemma-2-27b",
        "google/gemma-2-27b",
    ],
    "unsloth_gemma-3-4b-pt.yaml": [
        "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit",
        "google/gemma-3-4b-pt",
        "unsloth/gemma-3-4b-pt-bnb-4bit",
    ],
    "unsloth_gemma-3-4b-it.yaml": [
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "google/gemma-3-4b-it",
        "unsloth/gemma-3-4b-it-bnb-4bit",
    ],
    "unsloth_gemma-3-27b-it.yaml": [
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        "google/gemma-3-27b-it",
        "unsloth/gemma-3-27b-it-bnb-4bit",
    ],
    "unsloth_gemma-3-270m-it.yaml": [
        "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
        "google/gemma-3-270m-it",
        "unsloth/gemma-3-270m-it-bnb-4bit",
    ],
    "unsloth_gemma-3n-E4B-it.yaml": [
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "google/gemma-3n-E4B-it",
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    ],
    "unsloth_gemma-3n-E4B.yaml": [
        "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
        "google/gemma-3n-E4B",
    ],
    "unsloth_gpt-oss-20b.yaml": [
        "openai/gpt-oss-20b",
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-20b-BF16",
    ],
    
    "unsloth_gpt-oss-120b.yaml": [
        "openai/gpt-oss-120b",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    ],
    "unsloth_granite-4.0-350m-unsloth-bnb-4bit.yaml": [
        "unsloth/granite-4.0-350m",
        "ibm-granite/granite-4.0-350m",
        "unsloth/granite-4.0-350m-bnb-4bit",
    ],
    "unsloth_granite-4.0-h-micro.yaml": [
        "ibm-granite/granite-4.0-h-micro",
        "unsloth/granite-4.0-h-micro-bnb-4bit",
        "unsloth/granite-4.0-h-micro-unsloth-bnb-4bit",
    ],
    "unsloth_LFM2-1.2B.yaml": [
        "unsloth/LFM2-1.2B",
    ],
    "unsloth_llama-3-8b-bnb-4bit.yaml": [
        "unsloth/llama-3-8b",
        "meta-llama/Meta-Llama-3-8B",
    ],
    "unsloth_llama-3-8b-Instruct-bnb-4bit.yaml": [
        "unsloth/llama-3-8b-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    "unsloth_Meta-Llama-3.1-70B-bnb-4bit.yaml": [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-unsloth-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-8B",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B",
        "unsloth/Meta-Llama-3.1-70B",
        "meta-llama/Meta-Llama-3.1-70B",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-405B",
    ],
    
    "unsloth_Meta-Llama-3.1-8B-Instruct-bnb-4bit.yaml": [
        "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        "RedHatAI/Llama-3.1-8B-Instruct-FP8",
        "unsloth/Llama-3.1-8B-Instruct-FP8-Block",
        "unsloth/Llama-3.1-8B-Instruct-FP8-Dynamic",
    ],
    "unsloth_Llama-3.2-3B-Instruct.yaml": [
        "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
        "meta-llama/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "RedHatAI/Llama-3.2-3B-Instruct-FP8",
        "unsloth/Llama-3.2-3B-Instruct-FP8-Block",
        "unsloth/Llama-3.2-3B-Instruct-FP8-Dynamic",
    ],
    "unsloth_Llama-3.2-1B-Instruct.yaml": [
        "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
        "meta-llama/Llama-3.2-1B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "RedHatAI/Llama-3.2-1B-Instruct-FP8",
        "unsloth/Llama-3.2-1B-Instruct-FP8-Block",
        "unsloth/Llama-3.2-1B-Instruct-FP8-Dynamic",
    ],
    "unsloth_Llama-3.2-11B-Vision-Instruct.yaml": [
        "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    ],
    "unsloth_Llama-3.3-70B-Instruct.yaml": [
        "unsloth/Llama-3.3-70B-Instruct-unsloth-bnb-4bit",
        "meta-llama/Llama-3.3-70B-Instruct",
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "RedHatAI/Llama-3.3-70B-Instruct-FP8",
        "unsloth/Llama-3.3-70B-Instruct-FP8-Block",
        "unsloth/Llama-3.3-70B-Instruct-FP8-Dynamic",
    ],
    "unsloth_Llasa-3B.yaml": [
        "HKUSTAudio/Llasa-1B",
        "unsloth/Llasa-3B",
    ],
    "unsloth_Magistral-Small-2509-unsloth-bnb-4bit.yaml": [
        "unsloth/Magistral-Small-2509",
        "mistralai/Magistral-Small-2509",
        "unsloth/Magistral-Small-2509-bnb-4bit",
    ],
    "unsloth_Ministral-3-3B-Instruct-2512.yaml": [
        "unsloth/Ministral-3-3B-Instruct-2512",
    ],
    "unsloth_mistral-7b-v0.3-bnb-4bit.yaml": [
        "unsloth/mistral-7b-v0.3-bnb-4bit"
        "unsloth/mistral-7b-v0.3",
        "mistralai/Mistral-7B-v0.3",
    ],
    "unsloth_Mistral-Nemo-Base-2407-bnb-4bit.yaml": [
        "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        "unsloth/Mistral-Nemo-Base-2407",
        "mistralai/Mistral-Nemo-Base-2407",
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit", 
        "unsloth/Mistral-Nemo-Instruct-2407", 
        "mistralai/Mistral-Nemo-Instruct-2407",

    ],
    "unsloth_Mistral-Small-Instruct-2409.yaml": [
        "unsloth/Mistral-Small-Instruct-2409-bnb-4bit",
        "mistralai/Mistral-Small-Instruct-2409",
    ],
    "unsloth_mistral-7b-instruct-v0.3-bnb-4bit.yaml": [
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "unsloth_Qwen2.5-1.5B-Instruct.yaml": [
        "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    ],
    "unsloth_Nemotron-3-Nano-30B-A3B.yaml": [
        "unsloth/Nemotron-3-Nano-30B-A3B",
    ],
    "unsloth_orpheus-3b-0.1-ft.yaml": [
        "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit",
        "canopylabs/orpheus-3b-0.1-ft",
        "unsloth/orpheus-3b-0.1-ft-bnb-4bit",
    ],
    "OuteAI_Llama-OuteTTS-1.0-1B.yaml": [
        "OuteAI/Llama-OuteTTS-1.0-1B",
    ],
    "unsloth_PaddleOCR-VL.yaml": [
        "unsloth/PaddleOCR-VL",
    ],
    "unsloth_Phi-3-medium-4k-instruct.yaml": [
        "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
        "microsoft/Phi-3-medium-4k-instruct",
    ],
    "unsloth_Phi-3.5-mini-instruct.yaml": [
        "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "microsoft/Phi-3.5-mini-instruct",
    ],
    "unsloth_Phi-4.yaml": [
        "unsloth/phi-4-unsloth-bnb-4bit",
        "microsoft/phi-4",
        "unsloth/phi-4-bnb-4bit",
    ],
    "unsloth_Pixtral-12B-2409.yaml": [
        "unsloth/Pixtral-12B-2409-unsloth-bnb-4bit",
        "mistralai/Pixtral-12B-2409",
        "unsloth/Pixtral-12B-2409-bnb-4bit",
    ],
    "unsloth_Qwen2-7B.yaml": [
        "unsloth/Qwen2-7B-bnb-4bit",
        "Qwen/Qwen2-7B",
    ],
    "unsloth_Qwen2-VL-7B-Instruct.yaml": [
        "unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit",
        "Qwen/Qwen2-VL-7B-Instruct",
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    ],
    "unsloth_Qwen2.5-7B.yaml": [
        "unsloth/Qwen2.5-7B-unsloth-bnb-4bit",
        "Qwen/Qwen2.5-7B",
        "unsloth/Qwen2.5-7B-bnb-4bit",
    ],
    "unsloth_Qwen2.5-Coder-1.5B-Instruct.yaml": [
        "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    ],
    "unsloth_Qwen2.5-Coder-14B-Instruct.yaml": [
        "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ],
    "unsloth_Qwen2.5-VL-7B-Instruct-bnb-4bit.yaml": [
        "unsloth/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
    ],
    "unsloth_Qwen3-0.6B.yaml": [
        "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        "Qwen/Qwen3-0.6B",
        "unsloth/Qwen3-0.6B-bnb-4bit",
        "Qwen/Qwen3-0.6B-FP8",
        "unsloth/Qwen3-0.6B-FP8",
    ],
    "unsloth_Qwen3-4B-Instruct-2507.yaml": [
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "Qwen/Qwen3-4B-Instruct-2507",
        "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "unsloth/Qwen3-4B-Instruct-2507-FP8",
    ],
    "unsloth_Qwen3-4B-Thinking-2507.yaml": [
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        "Qwen/Qwen3-4B-Thinking-2507",
        "unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
        "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "unsloth/Qwen3-4B-Thinking-2507-FP8",
    ],
    "unsloth_Qwen3-14B-Base-unsloth-bnb-4bit.yaml": [
        "unsloth/Qwen3-14B-Base",
        "Qwen/Qwen3-14B-Base",
        "unsloth/Qwen3-14B-Base-bnb-4bit",
    ],
    "unsloth_Qwen3-14B.yaml": [
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "Qwen/Qwen3-14B",
        "unsloth/Qwen3-14B-bnb-4bit",
        "Qwen/Qwen3-14B-FP8",
        "unsloth/Qwen3-14B-FP8",
    ],
    "unsloth_Qwen3-32B.yaml": [
        "unsloth/Qwen3-32B-unsloth-bnb-4bit",
        "Qwen/Qwen3-32B",
        "unsloth/Qwen3-32B-bnb-4bit",
        "Qwen/Qwen3-32B-FP8",
        "unsloth/Qwen3-32B-FP8",
    ],
    "unsloth_Qwen3-VL-8B-Instruct-unsloth-bnb-4bit.yaml": [
        "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "unsloth/Qwen3-VL-8B-Instruct-FP8",
        "unsloth/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
    ],
    "sesame_csm-1b.yaml": [
        "sesame/csm-1b",
    ],
    "Spark-TTS-0.5B_LLM.yaml": [
        "Spark-TTS-0.5B/LLM",
    ],
    "unsloth_tinyllama-bnb-4bit.yaml": [
        "unsloth/tinyllama",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    ],
    "unsloth_whisper-large-v3.yaml": [
        "unsloth/whisper-large-v3",
        "openai/whisper-large-v3",
    ],
}

# Reverse mapping for quick lookup: model_name -> canonical_filename
_REVERSE_MODEL_MAPPING = {}
for canonical_file, model_names in MODEL_NAME_MAPPING.items():
    for model_name in model_names:
        _REVERSE_MODEL_MAPPING[model_name] = canonical_file

def load_model_config(model_name: str, use_auth: bool = False, token: Optional[str] = None):
    """
    Load model config with optional authentication control.
    """

    if token:
        # Explicit token provided - use it
        return AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=token
        )

    if not use_auth:
        # Load without any authentication (for public model checks)
        with without_hf_auth():
            return AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=None
            )

    # Use default authentication (cached tokens)
    return AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )
pass


def is_vision_model(model_name: str, hf_token: Optional[str] = None) -> bool:
    """
    Detect vision models by checking architecture in config.
    Works for fine-tuned models since they inherit the base architecture.

    Args:
        model_name: Model identifier (HF repo or local path)
        hf_token: Optional HF token for accessing gated/private models
    """
    try:
        config = load_model_config(model_name, token=hf_token)

        # Check vision arch
        if hasattr(config, 'architectures'):
            is_vlm = any(
                x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
                for x in config.architectures
            )
            if is_vlm:
                logger.info(f"Model {model_name} detected as vision model: architecture {config.architectures}")
                return True

        # Quick check for vision config as backup
        if hasattr(config, 'vision_config'):
            logger.info(f"Model {model_name} detected as vision model: has vision_config")
            return True

        return False

    except Exception as e:
        logger.warning(f"Could not determine if {model_name} is vision model: {e}")
        return False
pass


def scan_trained_loras(outputs_dir: str = "./outputs") -> List[Tuple[str, str]]:
    """
    Scan outputs folder for trained LoRA adapters.

    Returns:
        List of tuples: [(display_name, adapter_path), ...]

    Example:
        [
            ("unsloth_Meta-Llama-3.1_...", "./outputs/unsloth_Meta-Llama-3.1_.../"),
            ("my_finetuned_model", "./outputs/my_finetuned_model/"),
        ]
    """
    trained_loras = []
    outputs_path = Path(outputs_dir)

    if not outputs_path.exists():
        logger.warning(f"Outputs directory not found: {outputs_dir}")
        return trained_loras

    try:
        for item in outputs_path.iterdir():
            if item.is_dir():
                # Check if this directory contains a LoRA adapter
                adapter_config = item / "adapter_config.json"
                adapter_model = item / "adapter_model.safetensors"

                if adapter_config.exists() or adapter_model.exists():
                    display_name = item.name
                    adapter_path = str(item)
                    trained_loras.append((display_name, adapter_path))
                    logger.debug(f"Found trained LoRA: {display_name}")

        # Sort by modification time (newest first)
        trained_loras.sort(key=lambda x: Path(x[1]).stat().st_mtime, reverse=True)

        logger.info(f"Found {len(trained_loras)} trained LoRA adapters in {outputs_dir}")
        return trained_loras

    except Exception as e:
        logger.error(f"Error scanning outputs folder: {e}")
        return []

def get_base_model_from_lora(lora_path: str) -> Optional[str]:
    """
    Read the base model name from a LoRA adapter's config.

    Args:
        lora_path: Path to the LoRA adapter directory

    Returns:
        Base model identifier (e.g., "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
        or None if not found

    Example:
        >>> get_base_model_from_lora("./outputs/unsloth_Meta-Llama-3.1_.../")
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    """
    try:
        lora_path_obj = Path(lora_path)

        # Try adapter_config.json first
        adapter_config_path = lora_path_obj / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path")
                if base_model:
                    logger.info(f"Detected base model from adapter_config.json: {base_model}")
                    return base_model

        # Fallback: try training_args.bin (requires torch)
        training_args_path = lora_path_obj / "training_args.bin"
        if training_args_path.exists():
            try:
                import torch
                training_args = torch.load(training_args_path)
                if hasattr(training_args, 'model_name_or_path'):
                    base_model = training_args.model_name_or_path
                    logger.info(f"Detected base model from training_args.bin: {base_model}")
                    return base_model
            except Exception as e:
                logger.warning(f"Could not load training_args.bin: {e}")

        # Last resort: parse from directory name
        # Format: unsloth_Meta-Llama-3.1-8B-Instruct-bnb-4bit_timestamp
        dir_name = lora_path_obj.name
        if dir_name.startswith("unsloth_"):
            # Remove timestamp suffix (usually _1234567890)
            parts = dir_name.split("_")
            # Reconstruct model name
            if len(parts) >= 2:
                model_parts = parts[1:-1]  # Skip "unsloth" and timestamp
                base_model = "unsloth/" + "_".join(model_parts)
                logger.info(f"Detected base model from directory name: {base_model}")
                return base_model

        logger.warning(f"Could not detect base model for LoRA: {lora_path}")
        return None

    except Exception as e:
        logger.error(f"Error reading base model from LoRA config: {e}")
        return None
pass

# Status indicators that appear in UI dropdowns
UI_STATUS_INDICATORS = [" (Ready)", " (Loading...)", " (Active)", "↓ "]

def load_model_defaults(model_name: str) -> Dict[str, Any]:
    """
    Load default training parameters for a model from YAML file.
    
    Args:
        model_name: Model identifier (e.g., "unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    
    Returns:
        Dictionary with default parameters from YAML file, or empty dict if not found
    
    The function looks for a YAML file in configs/model_defaults/ (including subfolders) 
    based on the model name or its aliases from MODEL_NAME_MAPPING.
    If no specific file exists, it falls back to default.yaml.
    """
    try:
        # Get the script directory to locate configs
        script_dir = Path(__file__).parent.parent
        defaults_dir = script_dir / "configs" / "model_defaults"
        
        # First, check if model is in the mapping
        if model_name in _REVERSE_MODEL_MAPPING:
            canonical_file = _REVERSE_MODEL_MAPPING[model_name]
            # Search in subfolders and root
            for config_path in defaults_dir.rglob(canonical_file):
                if config_path.is_file():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                        logger.info(f"Loaded model defaults from {config_path} (via mapping)")
                        return config
        
        # Try exact model name match (for backward compatibility)
        model_filename = model_name.replace("/", "_") + ".yaml"
        # Search in subfolders and root
        for config_path in defaults_dir.rglob(model_filename):
            if config_path.is_file():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                    logger.info(f"Loaded model defaults from {config_path}")
                    return config
        
        # Fall back to default.yaml
        default_config_path = defaults_dir / "default.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                logger.info(f"Loaded default model defaults from {default_config_path}")
                return config
        
        logger.warning(f"No default config found for model {model_name}")
        return {}
        
    except Exception as e:
        logger.error(f"Error loading model defaults for {model_name}: {e}")
        return {}

@dataclass
class ModelConfig:
    """Configuration for a model to load"""
    identifier: str      # Clean model identifier (org/name or path)
    display_name: str    # Original UI display name
    path: str           # Normalized filesystem path
    is_local: bool      # Is this a local file vs HF model?
    is_cached: bool     # Is this already in HF cache?
    is_vision: bool     # Is this a vision model?
    is_lora: bool       # Is this a lora adapter?
    base_model: Optional[str] = None  # Base model (for LoRAs)

    @classmethod
    def from_lora_path(cls, lora_path: str, hf_token: Optional[str] = None) -> Optional['ModelConfig']:
        """
        Create ModelConfig from a local LoRA adapter path.

        Automatically detects the base model from adapter config.

        Args:
            lora_path: Path to LoRA adapter (e.g., "./outputs/unsloth_Meta-Llama-3.1_.../")
            hf_token: HF token for vision detection

        Returns:
            ModelConfig for the LoRA adapter
        """
        try:
            lora_path_obj = Path(lora_path)

            if not lora_path_obj.exists():
                logger.error(f"LoRA path does not exist: {lora_path}")
                return None

            # Get base model
            base_model = get_base_model_from_lora(lora_path)
            if not base_model:
                logger.error(f"Could not determine base model for LoRA: {lora_path}")
                return None

            # Check if base model is vision
            is_vision = is_vision_model(base_model, hf_token=hf_token)

            display_name = lora_path_obj.name
            identifier = lora_path  # Use path as identifier for local LoRAs

            return cls(
                identifier=identifier,
                display_name=display_name,
                path=lora_path,
                is_local=True,
                is_cached=True,  # Local LoRAs are always "cached"
                is_vision=is_vision,
                is_lora=True,
                base_model=base_model,
            )

        except Exception as e:
            logger.error(f"Error creating ModelConfig from LoRA path: {e}")
            return None


    @classmethod
    def from_ui_selection(cls,
                          dropdown_value: Optional[str],
                          search_value: Optional[str],
                          local_models: list = None,
                          hf_token: Optional[str] = None,
                          is_lora: bool = False) -> Optional['ModelConfig']:
        """
        Create a universal ModelConfig from UI dropdown/search selections.
        Handles base models and LoRA adapters.
        """
        selected = None
        if search_value and search_value.strip():
            selected = search_value.strip()
        elif dropdown_value:
            selected = dropdown_value

        if not selected:
            return None

        display_name = selected

        #  Use the correct 'local_models' parameter to resolve display names
        if " (Active)" in selected or " (Ready)" in selected:
            clean_display_name = selected.replace(" (Active)", "").replace(" (Ready)", "")
            if local_models:
                for local_display, local_path in local_models:
                    if local_display == clean_display_name:
                        selected = local_path
                        break

        # Clean all UI status indicators to get the final identifier
        identifier = selected
        for status in UI_STATUS_INDICATORS:
            identifier = identifier.replace(status, "")
        identifier = identifier.strip()

        is_local = is_local_path(identifier)
        path = normalize_path(identifier) if is_local else identifier

        # Add unsloth/ prefix for shorthand HF models
        if not is_local and "/" not in identifier:
            identifier = f"unsloth/{identifier}"
            path = identifier

        # --- Logic for Base Model and Vision Detection ---
        base_model = None
        is_vision = False

        if is_lora:
            # For a LoRA, we MUST find its base model.
            base_model = get_base_model_from_lora(path)
            if not base_model:
                logger.warning(f"Could not determine base model for LoRA '{path}'. Cannot create config.")
                return None # Cannot proceed without a base model

            # A LoRA's vision capability is determined by its base model.
            is_vision = is_vision_model(base_model, hf_token=hf_token)
        else:
            # For a base model, just check its own vision status.
            is_vision = is_vision_model(identifier, hf_token=hf_token)

        from .path_utils import is_model_cached
        is_cached = is_model_cached(identifier) if not is_local else True

        return cls(
            identifier=identifier,
            display_name=display_name,
            path=path,
            is_local=is_local,
            is_cached=is_cached,
            is_vision=is_vision,
            is_lora=is_lora,
            base_model=base_model, # This will be None for base models, and populated for LoRAs
        )
    pass
