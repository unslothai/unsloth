# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load inference params (temperature, top_p, top_k, min_p) from model YAML, family defaults, or default.yaml."""

from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache
import json
import math
import os
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
        # Family defaults (inference_defaults.json) carry repetition_penalty; surface it too so
        # a recommended value isn't silently dropped when resolving sampling for a request.
        "repetition_penalty": _get_param("repetition_penalty", 1.0),
        "trust_remote_code": model_inference.get(
            "trust_remote_code", default_inference.get("trust_remote_code", False)
        ),
    }

    return inference_config


# ── Effective sampling resolution for `unsloth run` / `unsloth start` ──────────
#
# Per-model recommended sampling is applied to a request only for the fields the
# client omitted; an operator can pin a field from the CLI via UNSLOTH_SAMPLING_*
# (a hard override that wins even over an explicit client value). Precedence per
# field: operator pin -> client explicit -> per-model recommendation -> the static
# schema default (mirroring ChatCompletionRequest, so behavior is unchanged when
# nothing is recommended or pinned).

# field -> (env var, static default, min, max, is_int)
_SAMPLING_FIELDS = {
    "temperature": ("UNSLOTH_SAMPLING_TEMPERATURE", 0.6, 0.0, 2.0, False),
    "top_p": ("UNSLOTH_SAMPLING_TOP_P", 0.95, 0.0, 1.0, False),
    "top_k": ("UNSLOTH_SAMPLING_TOP_K", 20, -1, 100, True),
    "min_p": ("UNSLOTH_SAMPLING_MIN_P", 0.01, 0.0, 1.0, False),
    "repetition_penalty": ("UNSLOTH_SAMPLING_REPETITION_PENALTY", 1.0, 1.0, 2.0, False),
    "presence_penalty": ("UNSLOTH_SAMPLING_PRESENCE_PENALTY", 0.0, 0.0, 2.0, False),
}

# Public, ordered tuple of the sampling fields callers resolve.
SAMPLING_FIELD_NAMES = tuple(_SAMPLING_FIELDS)


def _clean_sampling_value(field: str, val: Any):
    """Coerce ``val`` to the field's numeric type when it is a finite, in-range number, else None.

    Rejects bool, non-numeric, NaN/inf, and out-of-range values so neither a bad operator env
    var nor a malformed model recommendation can reach llama-server. NaN matters because
    ``nan < lo`` and ``nan > hi`` are both False, so a plain range check would let it through.
    """
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    if not math.isfinite(val):
        return None
    _env, _default, lo, hi, is_int = _SAMPLING_FIELDS[field]
    val = int(val) if is_int else float(val)
    if val < lo or val > hi:
        return None
    return val


def _operator_sampling_override(field: str):
    """Operator-pinned value for a sampling field from UNSLOTH_SAMPLING_*, or None.

    An unparseable, non-finite, or out-of-range value is ignored so a bad env var can never
    reach llama-server; the field then falls back to the client / recommended value.
    """
    _env, _default, _lo, _hi, is_int = _SAMPLING_FIELDS[field]
    raw = os.environ.get(_env)
    if raw is None or raw.strip() == "":
        return None
    try:
        val = int(raw) if is_int else float(raw)
    except (TypeError, ValueError):
        return None
    return _clean_sampling_value(field, val)


@lru_cache(maxsize = 128)
def _recommended_sampling(model_id: str) -> Dict[str, Any]:
    """Per-model recommended sampling values (model-specific YAML or family defaults only).

    The generic ``default.yaml`` tier that :func:`load_inference_config` falls back to is
    intentionally excluded: a model with no specific/family recommendation keeps the
    request's schema defaults instead of shifting every unknown model to the generic
    baseline. Model-specific YAML wins over the family default. Cached by model id.
    """
    if not model_id:
        return {}
    # load_model_defaults() falls back to default.yaml for an unknown model, so gate the
    # model-YAML tier on _has_specific_yaml (mirrors load_inference_config) to keep the
    # generic default.yaml out of "recommended".
    try:
        if _has_specific_yaml(model_id):
            model_inference = (load_model_defaults(model_id) or {}).get("inference", {}) or {}
        else:
            model_inference = {}
    except Exception as e:
        logger.debug(f"Could not load model defaults for '{model_id}': {e}")
        model_inference = {}
    try:
        family = get_family_inference_params(model_id) or {}
    except Exception as e:
        logger.debug(f"Could not load family sampling for '{model_id}': {e}")
        family = {}

    recommended: Dict[str, Any] = {}
    for field in _SAMPLING_FIELDS:
        for source in (model_inference, family):
            cleaned = _clean_sampling_value(field, source.get(field))
            if cleaned is not None:
                recommended[field] = cleaned
                break
    return recommended


def resolve_effective_sampling(model_id: Optional[str], explicit: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective sampling params for a request.

    ``explicit`` maps each field in :data:`SAMPLING_FIELD_NAMES` to the client-sent
    value, or ``None`` when the client omitted it. Precedence (highest first): an
    operator ``UNSLOTH_SAMPLING_*`` pin, then the client's explicit value, then the
    per-model recommendation, then the static schema default.
    """
    recommended = _recommended_sampling(model_id or "")
    effective: Dict[str, Any] = {}
    for field, (_env, default, _lo, _hi, _int) in _SAMPLING_FIELDS.items():
        override = _operator_sampling_override(field)
        if override is not None:
            effective[field] = override
        elif explicit.get(field) is not None:
            effective[field] = explicit[field]
        elif field in recommended:
            effective[field] = recommended[field]
        else:
            effective[field] = default
    return effective
