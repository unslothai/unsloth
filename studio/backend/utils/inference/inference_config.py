# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference configuration loading utilities.

This module provides functions to load inference parameters (temperature, top_p, top_k, min_p)
from model YAML configuration files, with fallback to default.yaml.
Includes family-based lookup from inference_defaults.json for GGUF models.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
import structlog
from loggers import get_logger

from utils.models.model_config import load_model_defaults
from utils.paths import is_local_path, normalize_path

logger = get_logger(__name__)

# ── Family-based inference defaults (loaded once, cached) ──────────────

_FAMILY_DEFAULTS: Optional[Dict[str, Any]] = None
_FAMILY_PATTERNS: Optional[list] = None


def _load_family_defaults():
    """Load and cache inference_defaults.json."""
    global _FAMILY_DEFAULTS, _FAMILY_PATTERNS
    if _FAMILY_DEFAULTS is not None:
        return

    json_path = (
        Path(__file__).parent.parent.parent
        / "assets"
        / "configs"
        / "inference_defaults.json"
    )
    try:
        with open(json_path, "r", encoding = "utf-8") as f:
            data = json.load(f)
        _FAMILY_DEFAULTS = data.get("families", {})
        _FAMILY_PATTERNS = data.get("patterns", [])
    except Exception as e:
        logger.warning(f"Failed to load inference_defaults.json: {e}")
        _FAMILY_DEFAULTS = {}
        _FAMILY_PATTERNS = []


def get_family_inference_params(model_id: str) -> Dict[str, Any]:
    """
    Look up recommended inference parameters by model family.

    Extracts the model family from the identifier (e.g. "unsloth/Qwen3.5-9B-GGUF" -> "qwen3.5")
    and returns the matching parameters from inference_defaults.json.

    Args:
        model_id: Model identifier (e.g. "unsloth/Qwen3.5-9B-GGUF")

    Returns:
        Dict with inference params, or empty dict if no family match.
    """
    _load_family_defaults()

    if not _FAMILY_PATTERNS or not _FAMILY_DEFAULTS:
        return {}

    # Normalize: lowercase, strip org prefix
    normalized = model_id.lower()
    if "/" in normalized:
        normalized = normalized.split("/", 1)[1]

    # Match against patterns (ordered longest-match-first in the JSON)
    for pattern in _FAMILY_PATTERNS:
        if pattern in normalized:
            params = _FAMILY_DEFAULTS.get(pattern, {})
            if params:
                return dict(params)

    return {}


def _has_specific_yaml(model_identifier: str) -> bool:
    """Check if a model has its own YAML config (not just default.yaml)."""
    from utils.models.model_config import _REVERSE_MODEL_MAPPING

    script_dir = Path(__file__).parent.parent.parent
    defaults_dir = script_dir / "assets" / "configs" / "model_defaults"

    # Check the mapping
    if model_identifier.lower() in _REVERSE_MODEL_MAPPING:
        return True

    # For local filesystem paths (e.g. C:\Users\...\model on Windows),
    # normalize backslashes so Path().parts splits correctly on POSIX/WSL,
    # then try matching the last 1-2 path components against the registry
    # (mirrors the logic in load_model_defaults).
    _is_local = is_local_path(model_identifier)
    _normalized = normalize_path(model_identifier) if _is_local else model_identifier

    if _is_local:
        parts = Path(_normalized).parts
        for depth in (2, 1):
            if len(parts) >= depth:
                suffix = "/".join(parts[-depth:])
                if suffix.lower() in _REVERSE_MODEL_MAPPING:
                    return True
        _lookup = Path(_normalized).name
    else:
        _lookup = model_identifier

    # Check for exact filename match (basename for local paths to avoid
    # passing absolute paths into rglob which raises
    # "Non-relative patterns are unsupported" on Windows).
    model_filename = _lookup.replace("/", "_") + ".yaml"
    for config_path in defaults_dir.rglob(model_filename):
        if config_path.is_file():
            return True

    return False


def load_inference_config(model_identifier: str) -> Dict[str, Any]:
    """
    Load inference configuration parameters for a model.

    Priority chain:
    1. Model-specific YAML (if it exists and has inference params)
    2. Family-based defaults from inference_defaults.json
    3. default.yaml fallback

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

    # Family-based defaults from inference_defaults.json
    family_params = get_family_inference_params(model_identifier)

    model_inference = model_defaults.get("inference", {})

    # If the model has its own YAML config, those values take priority over family defaults.
    # If it only fell back to default.yaml, family defaults take priority.
    has_own_yaml = _has_specific_yaml(model_identifier)

    def _get_param(key, hardcoded_default):
        if has_own_yaml:
            # Model-specific YAML wins, then family fills gaps, then default.yaml
            val = model_inference.get(key)
            if val is not None and isinstance(val, (int, float)):
                return val
            if key in family_params:
                return family_params[key]
            return default_inference.get(key, hardcoded_default)
        else:
            # No model-specific YAML: family wins, then default.yaml
            if key in family_params:
                return family_params[key]
            return default_inference.get(key, hardcoded_default)

    inference_config = {
        "temperature": _get_param("temperature", 0.7),
        "top_p": _get_param("top_p", 0.95),
        "top_k": _get_param("top_k", -1),
        "min_p": _get_param("min_p", 0.01),
        "presence_penalty": _get_param("presence_penalty", 0.0),
        "trust_remote_code": model_inference.get(
            "trust_remote_code", default_inference.get("trust_remote_code", False)
        ),
    }

    return inference_config
