# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Inference configuration loading utilities.

This module provides functions to load inference parameters (temperature, top_p, top_k, min_p)
from model YAML configuration files, with fallback to default.yaml.
"""

from pathlib import Path
from typing import Dict, Any
import yaml
import structlog
from loggers import get_logger

from utils.models.model_config import load_model_defaults

logger = get_logger(__name__)


def load_inference_config(model_identifier: str) -> Dict[str, Any]:
    """
    Load inference configuration parameters for a model.

    This function loads inference parameters (temperature, top_p, top_k, min_p) from the
    model's YAML configuration file using the same mapping logic as the /config endpoint.
    If a parameter is missing from the model's config, it falls back to the value in
    default.yaml.

    Args:
        model_identifier: Model identifier (e.g., "unsloth/llama-3-8b-bnb-4bit")

    Returns:
        Dictionary containing inference parameters:
        {
            "temperature": float,
            "top_p": float,
            "top_k": int,
            "min_p": float
        }
    """
    # Load model defaults to get inference parameters
    model_defaults = load_model_defaults(model_identifier)

    # Load default.yaml for fallback values
    script_dir = Path(__file__).parent.parent.parent
    defaults_dir = script_dir / "assets" / "configs" / "model_defaults"
    default_config_path = defaults_dir / "default.yaml"

    default_inference = {}
    if default_config_path.exists():
        try:
            with open(default_config_path, "r", encoding = "utf-8") as f:
                default_config = yaml.safe_load(f) or {}
                default_inference = default_config.get("inference", {})
        except Exception as e:
            logger.warning(f"Failed to load default.yaml: {e}")

    # Extract inference parameters from model config, fallback to defaults
    model_inference = model_defaults.get("inference", {})
    inference_config = {
        "temperature": model_inference.get(
            "temperature", default_inference.get("temperature", 0.7)
        ),
        "top_p": model_inference.get("top_p", default_inference.get("top_p", 0.95)),
        "top_k": model_inference.get("top_k", default_inference.get("top_k", -1)),
        "min_p": model_inference.get("min_p", default_inference.get("min_p", 0.01)),
        "trust_remote_code": model_inference.get(
            "trust_remote_code", default_inference.get("trust_remote_code", False)
        ),
    }

    return inference_config
