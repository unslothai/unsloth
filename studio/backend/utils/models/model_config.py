"""
Model and LoRA configuration handling
"""
from transformers import AutoConfig
from dataclasses import dataclass
from typing import Optional, Dict, Any
from utils.paths import normalize_path, is_local_path, is_model_cached
from utils.utils import without_hf_auth
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
    Detect vision-language models (VLMs) by checking architecture in config.
    Works for fine-tuned models since they inherit the base architecture.

    Args:
        model_name: Model identifier (HF repo or local path)
        hf_token: Optional HF token for accessing gated/private models
    """
    try:
        config = load_model_config(model_name, use_auth=True, token=hf_token)

        # Check 1: Architecture class name patterns
        if hasattr(config, 'architectures'):
            is_vlm = any(
                x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
                for x in config.architectures
            )
            if is_vlm:
                logger.info(f"Model {model_name} detected as VLM: architecture {config.architectures}")
                return True

        # Check 2: Has vision_config (most VLMs: LLaVA, Gemma-3, Qwen2-VL, etc.)
        if hasattr(config, 'vision_config'):
            logger.info(f"Model {model_name} detected as VLM: has vision_config")
            return True

        # Check 3: Has img_processor (Phi-3.5 Vision uses this instead of vision_config)
        if hasattr(config, 'img_processor'):
            logger.info(f"Model {model_name} detected as VLM: has img_processor")
            return True

        # Check 4: Has image_token_index (common in VLMs for image placeholder tokens)
        if hasattr(config, 'image_token_index'):
            logger.info(f"Model {model_name} detected as VLM: has image_token_index")
            return True

        # Check 5: Known VLM model_type values that may not match above checks
        if hasattr(config, 'model_type'):
            vlm_model_types = {
                'phi3_v', 'llava', 'llava_next', 'llava_onevision',
                'internvl_chat', 'cogvlm2', 'minicpmv',
            }
            if config.model_type in vlm_model_types:
                logger.info(f"Model {model_name} detected as VLM: model_type={config.model_type}")
                return True

        return False

    except Exception as e:
        logger.warning(f"Could not determine if {model_name} is vision model: {e}")
        return False
pass


def detect_gguf_model(path: str) -> Optional[str]:
    """
    Check if the given local path is or contains a GGUF model file.

    Handles two cases:
    1. path is a direct .gguf file path
    2. path is a directory containing .gguf files

    Returns the full path to the .gguf file if found, None otherwise.
    For HuggingFace repo detection, use detect_gguf_model_remote() instead.
    """
    p = Path(path)

    # Case 1: direct .gguf file
    if p.suffix == ".gguf" and p.is_file():
        return str(p.resolve())

    # Case 2: directory containing .gguf files
    if p.is_dir():
        gguf_files = sorted(p.glob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
        if gguf_files:
            return str(gguf_files[0].resolve())

    return None


# Preferred GGUF quantization levels, in descending priority.
# Q4_K_M is a good default: small, fast, acceptable quality.
_GGUF_QUANT_PREFERENCE = [
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S",
    "Q6_K", "Q8_0", "Q3_K_M", "Q3_K_L", "Q2_K",
    "F16", "BF16", "F32",
]


def _pick_best_gguf(filenames: list[str]) -> Optional[str]:
    """
    Pick the best GGUF file from a list of filenames.

    Prefers quantization levels in _GGUF_QUANT_PREFERENCE order.
    Falls back to the first .gguf file found.
    """
    gguf_files = [f for f in filenames if f.endswith(".gguf")]
    if not gguf_files:
        return None

    # Try preferred quantization levels
    for quant in _GGUF_QUANT_PREFERENCE:
        for f in gguf_files:
            if quant in f:
                return f

    # Fallback: first GGUF file
    return gguf_files[0]


@dataclass
class GgufVariantInfo:
    """A single GGUF quantization variant from a HuggingFace repo."""
    filename: str       # e.g., "gemma-3-4b-it-Q4_K_M.gguf"
    quant: str          # e.g., "Q4_K_M" (extracted from filename)
    size_bytes: int     # file size


def _extract_quant_label(filename: str) -> str:
    """
    Extract quantization label like Q4_K_M, IQ4_XS, BF16 from a GGUF filename.

    Examples:
        "gemma-3-4b-it-Q4_K_M.gguf" → "Q4_K_M"
        "model-IQ4_NL.gguf"          → "IQ4_NL"
        "model-BF16.gguf"            → "BF16"
        "model-UD-IQ1_S.gguf"        → "UD-IQ1_S"
    """
    import re
    stem = filename.rsplit(".", 1)[0]  # Remove .gguf
    # Match known quantization patterns (UD- prefix, IQ, Q, BF/F variants)
    match = re.search(
        r'(UD-)?'  # Optional UD- prefix (Ultra Discrete)
        r'(IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?'  # IQ variants: IQ4_XS, IQ4_NL, IQ1_S
        r'|Q[0-9]+_K_[A-Z]+'                  # K-quant: Q4_K_M, Q3_K_S
        r'|Q[0-9]+_[0-9]+'                    # Standard: Q8_0, Q5_1
        r'|Q[0-9]+_K'                          # Short K-quant: Q6_K
        r'|BF16|F16|F32)',                     # Full precision
        stem, re.IGNORECASE,
    )
    if match:
        prefix = match.group(1) or ""
        return f"{prefix}{match.group(2)}"
    # Fallback: last segment after hyphen
    return stem.split("-")[-1]


def list_gguf_variants(
    repo_id: str,
    hf_token: Optional[str] = None,
) -> tuple[list[GgufVariantInfo], bool]:
    """
    List all GGUF quantization variants in a HuggingFace repo.

    Separates main model files from mmproj (vision projection) files.
    The presence of mmproj files indicates a vision-capable model.

    Returns:
        (variants, has_vision): list of non-mmproj GGUF variants + vision flag.
    """
    from huggingface_hub import model_info as hf_model_info

    info = hf_model_info(repo_id, token=hf_token)
    variants: list[GgufVariantInfo] = []
    has_vision = False

    for sibling in info.siblings:
        fname = sibling.rfilename
        if not fname.endswith(".gguf"):
            continue
        size = sibling.size or 0

        # mmproj files are vision projection models, not main model files
        if "mmproj" in fname.lower():
            has_vision = True
            continue

        quant = _extract_quant_label(fname)
        variants.append(GgufVariantInfo(
            filename=fname,
            quant=quant,
            size_bytes=size,
        ))

    return variants, has_vision


def detect_gguf_model_remote(
    repo_id: str,
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """
    Check if a HuggingFace repo contains GGUF files.

    Returns the filename of the best GGUF file in the repo, or None.
    """
    try:
        from huggingface_hub import model_info as hf_model_info

        info = hf_model_info(repo_id, token=hf_token)
        repo_files = [s.rfilename for s in info.siblings]
        return _pick_best_gguf(repo_files)
    except Exception as e:
        logger.debug(f"Could not check GGUF files for '{repo_id}': {e}")
        return None


def download_gguf_file(
    repo_id: str,
    filename: str,
    hf_token: Optional[str] = None,
) -> str:
    """
    Download a specific GGUF file from a HuggingFace repo.

    Returns the local path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=hf_token,
    )
    return local_path


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

def scan_exported_models(exports_dir: str = "./exports") -> List[Tuple[str, str, str, Optional[str]]]:
    """
    Scan exports folder for exported models (merged, LoRA, base).
    Skips GGUF-only exports (not loadable by Unsloth inference backend).

    The exports directory is two levels deep: {run}/{checkpoint}/

    Returns:
        List of tuples: [(display_name, model_path, export_type, base_model), ...]
        export_type: "lora" | "merged"
    """
    results = []
    exports_path = Path(exports_dir)

    if not exports_path.exists():
        return results

    try:
        for run_dir in exports_path.iterdir():
            if not run_dir.is_dir():
                continue
            for checkpoint_dir in run_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue

                adapter_config = checkpoint_dir / "adapter_config.json"
                config_file = checkpoint_dir / "config.json"
                has_weights = (
                    any(checkpoint_dir.glob("*.safetensors"))
                    or any(checkpoint_dir.glob("*.bin"))
                )
                has_gguf = any(checkpoint_dir.glob("*.gguf"))

                base_model = None
                export_type = None

                if adapter_config.exists():
                    export_type = "lora"
                    try:
                        cfg = json.loads(adapter_config.read_text())
                        base_model = cfg.get("base_model_name_or_path")
                    except Exception:
                        pass
                elif config_file.exists() and has_weights:
                    export_type = "merged"
                    # Read base model from export_metadata.json (written at export time)
                    export_meta = checkpoint_dir / "export_metadata.json"
                    try:
                        if export_meta.exists():
                            meta = json.loads(export_meta.read_text())
                            base_model = meta.get("base_model")
                    except Exception:
                        pass
                elif has_gguf:
                    # GGUF-only — not loadable by current inference backend
                    continue
                else:
                    continue

                # Fallback: read base model from the original training run's
                # adapter_config.json in ./outputs/{run_name}/
                if not base_model:
                    outputs_adapter_cfg = Path("./outputs") / run_dir.name / "adapter_config.json"
                    try:
                        if outputs_adapter_cfg.exists():
                            cfg = json.loads(outputs_adapter_cfg.read_text())
                            base_model = cfg.get("base_model_name_or_path")
                    except Exception:
                        pass

                display_name = f"{run_dir.name} / {checkpoint_dir.name}"
                model_path = str(checkpoint_dir)
                results.append((display_name, model_path, export_type, base_model))
                logger.debug(f"Found exported model: {display_name} ({export_type})")

        results.sort(key=lambda x: Path(x[1]).stat().st_mtime, reverse=True)
        logger.info(f"Found {len(results)} exported models in {exports_dir}")
        return results

    except Exception as e:
        logger.error(f"Error scanning exports folder: {e}")
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
        script_dir = Path(__file__).parent.parent.parent
        defaults_dir = script_dir / "assets" / "configs" / "model_defaults"
        
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
    is_gguf: bool = False       # Is this a GGUF model?
    gguf_file: Optional[str] = None  # Full path to the .gguf file (local mode)
    gguf_hf_repo: Optional[str] = None  # HF repo ID for -hf mode (e.g. "unsloth/gemma-3-4b-it-GGUF")
    gguf_variant: Optional[str] = None  # Quantization variant (e.g. "Q4_K_M")
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
    def from_identifier(
        cls,
        model_id: str,
        hf_token: Optional[str] = None,
        is_lora: bool = False,
        gguf_variant: Optional[str] = None,
    ) -> Optional['ModelConfig']:
        """
        Create ModelConfig from a clean model identifier.

        For FastAPI routes where the frontend sends sanitized model paths.
        No Gradio dropdown parsing - expects clean identifiers like:
        - "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        - "./outputs/my_lora_adapter"
        - "/absolute/path/to/model"

        Args:
            model_id: Clean model identifier (HF repo name or local path)
            hf_token: Optional HF token for vision detection on gated models
            is_lora: Whether this is a LoRA adapter
            gguf_variant: Optional GGUF quantization variant (e.g. "Q4_K_M").
                For remote GGUF repos, specifies which quant to load via -hf.
                If None, auto-selects using _pick_best_gguf().

        Returns:
            ModelConfig or None if configuration cannot be created
        """
        if not model_id or not model_id.strip():
            return None

        identifier = model_id.strip()
        is_local = is_local_path(identifier)
        path = normalize_path(identifier) if is_local else identifier

        # Add unsloth/ prefix for shorthand HF models
        if not is_local and "/" not in identifier:
            identifier = f"unsloth/{identifier}"
            path = identifier

        # Auto-detect GGUF models (check before LoRA/vision detection)
        if is_local:
            gguf_file = detect_gguf_model(path)
            if gguf_file:
                display_name = Path(gguf_file).stem
                logger.info(f"Detected local GGUF model: {gguf_file}")
                return cls(
                    identifier=identifier,
                    display_name=display_name,
                    path=path,
                    is_local=True,
                    is_cached=True,
                    is_vision=False,
                    is_lora=False,
                    is_gguf=True,
                    gguf_file=gguf_file,
                )
        else:
            # Check if the HF repo contains GGUF files
            gguf_filename = detect_gguf_model_remote(identifier, hf_token=hf_token)
            if gguf_filename:
                # Preflight: verify llama-server binary exists BEFORE user waits
                # for a multi-GB download that llama-server handles natively
                from core.inference.llama_cpp import LlamaCppBackend
                if not LlamaCppBackend._find_llama_server_binary():
                    raise RuntimeError(
                        "llama-server binary not found — cannot load GGUF models. "
                        "Run setup.sh to build it, or set LLAMA_SERVER_PATH."
                    )

                # Use list_gguf_variants() to detect vision & resolve variant
                variants, has_vision = list_gguf_variants(identifier, hf_token=hf_token)
                variant = gguf_variant
                if not variant:
                    # Auto-select best quantization
                    variant_filenames = [v.filename for v in variants]
                    best = _pick_best_gguf(variant_filenames)
                    if best:
                        variant = _extract_quant_label(best)
                    else:
                        variant = "Q4_K_M"  # Fallback — llama-server's own default

                display_name = f"{identifier.split('/')[-1]} ({variant})"
                logger.info(
                    f"Detected remote GGUF repo '{identifier}', "
                    f"variant={variant}, vision={has_vision}"
                )
                return cls(
                    identifier=identifier,
                    display_name=display_name,
                    path=identifier,
                    is_local=False,
                    is_cached=False,
                    is_vision=has_vision,
                    is_lora=False,
                    is_gguf=True,
                    gguf_file=None,
                    gguf_hf_repo=identifier,
                    gguf_variant=variant,
                )

        # Auto-detect LoRA for local paths (check adapter_config.json on disk)
        if not is_lora and is_local:
            detected_base = get_base_model_from_lora(path)
            if detected_base:
                is_lora = True
                logger.info(f"Auto-detected local LoRA adapter at '{path}' (base: {detected_base})")
        
        # Auto-detect LoRA for remote HF models (check repo file listing)
        if not is_lora and not is_local:
            try:
                from huggingface_hub import model_info as hf_model_info
                info = hf_model_info(identifier, token=hf_token)
                repo_files = [s.rfilename for s in info.siblings]
                if "adapter_config.json" in repo_files:
                    is_lora = True
                    logger.info(f"Auto-detected remote LoRA adapter: '{identifier}'")
            except Exception as e:
                logger.debug(f"Could not check remote LoRA status for '{identifier}': {e}")
        
        # Handle LoRA adapters
        base_model = None
        if is_lora:
            if is_local:
                # Local LoRA: read adapter_config.json from disk
                base_model = get_base_model_from_lora(path)
            else:
                # Remote LoRA: download adapter_config.json from HF
                try:
                    from huggingface_hub import hf_hub_download
                    config_path = hf_hub_download(identifier, "adapter_config.json", token=hf_token)
                    with open(config_path, 'r') as f:
                        adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")
                    if base_model:
                        logger.info(f"Resolved remote LoRA base model: '{base_model}'")
                except Exception as e:
                    logger.warning(f"Could not download adapter_config.json for '{identifier}': {e}")
            
            if not base_model:
                logger.warning(f"Could not determine base model for LoRA '{path}'")
                return None
            vision = is_vision_model(base_model, hf_token=hf_token)
        else:
            vision = is_vision_model(identifier, hf_token=hf_token)
        
        display_name = Path(path).name if is_local else identifier.split("/")[-1]
        
        return cls(
            identifier=identifier,
            display_name=display_name,
            path=path,
            is_local=is_local,
            is_cached=is_model_cached(identifier) if not is_local else True,
            is_vision=vision,
            is_lora=is_lora,
            base_model=base_model,
        )


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

        from utils.paths import is_model_cached
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
