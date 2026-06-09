# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load inference params (temperature, top_p, top_k, min_p) from model YAML, family defaults, or default.yaml."""

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
        Path(__file__).parent.parent.parent / "assets" / "configs" / "inference_defaults.json"
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
    """Look up recommended inference params by model family.

    Extracts the family from the identifier (e.g. "unsloth/Qwen3.5-9B-GGUF" ->
    "qwen3.5") and returns matching params from inference_defaults.json, or {}.
    """
    _load_family_defaults()

    if not _FAMILY_PATTERNS or not _FAMILY_DEFAULTS:
        return {}

    # Normalize: lowercase, strip org prefix.
    normalized = model_id.lower()
    if "/" in normalized:
        normalized = normalized.split("/", 1)[1]

    # Match patterns (ordered longest-match-first in the JSON).
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

    if model_identifier.lower() in _REVERSE_MODEL_MAPPING:
        return True

    # For local paths, normalize backslashes so Path().parts splits correctly,
    # then match the last 1-2 components against the registry (mirrors load_model_defaults).
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

    # Exact filename match (basename for local paths; absolute paths break rglob on Windows).
    model_filename = _lookup.replace("/", "_") + ".yaml"
    for config_path in defaults_dir.rglob(model_filename):
        if config_path.is_file():
            return True

    return False


def load_inference_config(model_identifier: str) -> Dict[str, Any]:
    """Load inference params for a model.

    Priority: model-specific YAML, then family defaults (inference_defaults.json),
    then default.yaml. Returns a dict of temperature/top_p/top_k/min_p/etc.
    """
    model_defaults = load_model_defaults(model_identifier)

    # default.yaml for fallback values.
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

    # Family-based defaults from inference_defaults.json.
    family_params = get_family_inference_params(model_identifier)

    model_inference = model_defaults.get("inference", {})

    # Model's own YAML beats family defaults; if it only fell back to
    # default.yaml, family defaults win.
    has_own_yaml = _has_specific_yaml(model_identifier)

    def _get_param(key, hardcoded_default):
        if has_own_yaml:
            # Model-specific YAML wins, then family fills gaps, then default.yaml.
            val = model_inference.get(key)
            if val is not None and isinstance(val, (int, float)):
                return val
            if key in family_params:
                return family_params[key]
            return default_inference.get(key, hardcoded_default)
        else:
            # No model-specific YAML: family wins, then default.yaml.
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
