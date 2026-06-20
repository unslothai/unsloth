# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic transformers version switching.

Some newer model architectures (Ministral-3, GLM-4.7-Flash, Qwen3-30B-A3B MoE,
tiny_qwen3_moe) require transformers>=5.3.0, while Gemma 4 models require a
newer 5.x sidecar.  Everything else needs the default 4.57.x that ships with
Unsloth.

Two separate target directories are maintained:
  - .venv_t5_530/  — transformers 5.3.0 (Ministral-3, GLM, Qwen3 MoE, etc.)
  - .venv_t5_550/  — transformers 5.5.0 (Gemma 4)
  - .venv_t5_510/  — transformers 5.10.2 (Gemma 4 Unified / 12B)

When loading a LoRA adapter with a custom name, we resolve the base model from
``adapter_config.json`` and check *that* against the model list.

Strategy:
  Training and inference run in subprocesses that activate the correct version
  via sys.path (prepending the appropriate .venv_t5_*/ directory). See:
    - core/training/worker.py
    - core/inference/worker.py

  For export (still in-process), ensure_transformers_version() does a lightweight
  sys.path swap using the same directories pre-installed by setup.sh.
"""

import importlib
import json
import structlog
from loggers import get_logger
import os
import shutil
import subprocess
import sys
from pathlib import Path

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

logger = get_logger(__name__)


def _env_offline() -> bool:
    """True if HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set to a truthy value."""
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Lowercase substrings — any match in the lowered model name needs transformers 5.3.0.
TRANSFORMERS_5_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "ministral-3-",  # Ministral-3-{3,8,14}B-{Instruct,Reasoning,Base}-2512
    "glm-4.7-flash",  # GLM-4.7-Flash
    "qwen3-30b-a3b",  # Qwen3-30B-A3B-Instruct-2507 and variants
    "qwen3.5",  # Qwen3.5 family (35B-A3B, etc.)
    "qwen3-next",  # Qwen3-Next and variants
    "tiny_qwen3_moe",  # imdatta0/tiny_qwen3_moe_2.8B_0.7B
    "lfm2.5-vl-450m",  # LiquidAI/LFM2.5-VL-450M
)

# Lowercase substrings for models that require transformers 5.10.x (checked first).
TRANSFORMERS_510_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "gemma-4-12b",  # Gemma 4 Unified 12B
    "gemma4-12b",
)

# Lowercase substrings for models that require the Gemma 4 transformers 5.5 sidecar.
TRANSFORMERS_550_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "gemma-4",  # Gemma-4 (E2B-it, E4B-it, 31B-it, 26B-A4B-it)
    "gemma4",  # Gemma-4 alternate naming
    "qwen3.6",
)

# Architecture classes / model_type values that require transformers 5.10.x.
# Checked via config.json (local or HuggingFace).
_TRANSFORMERS_510_ARCHITECTURES: set[str] = {
    "Gemma4UnifiedForConditionalGeneration",
    "Gemma4AssistantForCausalLM",
    "Gemma4UnifiedAssistantForCausalLM",
}
_TRANSFORMERS_510_MODEL_TYPES: set[str] = {
    "gemma4_unified",
    "gemma4_assistant",
    "gemma4_unified_assistant",
}

# Architecture classes / model_type values that require transformers 5.5.0.
# Checked via config.json (local or HuggingFace).
_TRANSFORMERS_550_ARCHITECTURES: set[str] = {
    "Gemma4ForConditionalGeneration",
}
_TRANSFORMERS_550_MODEL_TYPES: set[str] = {
    "gemma4",
}

# Architecture classes / model_type values that require transformers 5.3.0.
# Checked via config.json (local or HuggingFace).
_TRANSFORMERS_530_ARCHITECTURES: set[str] = {
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",  # Qwen3.5 MoE (e.g. Qwen3.5-35B-A3B / 122B-A10B)
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",  # Qwen3-Next
    "Glm4MoeLiteForCausalLM",
    "Lfm2VlForConditionalGeneration",
}
_TRANSFORMERS_530_MODEL_TYPES: set[str] = {
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_moe",
    "qwen3_next",
    "glm4_moe_lite",
    "lfm2_vl",
}

# Tokenizer classes that only exist in transformers>=5.x.
_TRANSFORMERS_5_TOKENIZER_CLASSES: set[str] = {
    "TokenizersBackend",
}

# Cache for dynamic tokenizer_config.json lookups (avoids repeated fetches).
_tokenizer_class_cache: dict[str, bool] = {}

# config.json cache keyed on (model_name, token-hash) so authed/unauthed reads stay separate.
_config_json_cache: dict[tuple[str, str | None], dict | None] = {}
_config_needs_510_cache: dict[str, bool] = {}
_config_needs_550_cache: dict[str, bool] = {}
_config_needs_530_cache: dict[str, bool] = {}

# Versions
TRANSFORMERS_510_VERSION = "5.10.2"
TRANSFORMERS_550_VERSION = "5.5.0"
TRANSFORMERS_530_VERSION = "5.3.0"
TRANSFORMERS_DEFAULT_VERSION = "4.57.6"
# Backwards-compat alias — points to the highest 5.x tier.
# Consumers should prefer TRANSFORMERS_510_VERSION / TRANSFORMERS_550_VERSION /
# TRANSFORMERS_530_VERSION.
TRANSFORMERS_5_VERSION = TRANSFORMERS_510_VERSION

# Pre-installed directories — created by setup.sh / setup.ps1.
from utils.paths.storage_roots import studio_root as _studio_root  # noqa: E402

_VENV_T5_530_DIR = str(_studio_root() / ".venv_t5_530")
_VENV_T5_550_DIR = str(_studio_root() / ".venv_t5_550")
_VENV_T5_510_DIR = str(_studio_root() / ".venv_t5_510")
# Backwards-compat alias
_VENV_T5_DIR = _VENV_T5_550_DIR


def activate_transformers_for_subprocess(model_name: str) -> None:
    """Activate the correct transformers version in a subprocess worker.

    Call BEFORE any ML imports. Resolves LoRA adapters to their base model,
    determines the required tier, prepends the appropriate ``.venv_t5_*`` dir to
    ``sys.path``, and propagates it via ``PYTHONPATH`` for child processes
    (e.g. GGUF converter). Used by training, inference, and export workers.
    """
    # Only pre-resolve for LoRA adapter directories (adapter_config.json present
    # OR adapter_model-only weights). Full checkpoints go directly to
    # get_transformers_tier, which reads their local config.json for model_type —
    # more reliable than resolving to a private/offline HF ID that can't be probed
    # and may lack tier substrings.
    if _is_lora_adapter_dir(Path(model_name)):
        resolved = _resolve_base_model(model_name)
    else:
        resolved = model_name
    tier = get_transformers_tier(resolved)

    if tier == "510":
        if not _ensure_venv_t5_510_exists():
            raise RuntimeError(
                f"Cannot activate transformers {TRANSFORMERS_510_VERSION}: "
                f".venv_t5_510 missing at {_VENV_T5_510_DIR}"
            )
        if _VENV_T5_510_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_510_DIR)
        logger.info(
            "Prepended transformers %s venv to sys.path from %s "
            "(path only; the loaded version is confirmed later by "
            "'Subprocess loaded transformers ...' on first import)",
            TRANSFORMERS_510_VERSION,
            _VENV_T5_510_DIR,
        )
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_510_DIR + (os.pathsep + _pp if _pp else "")
    elif tier == "550":
        if not _ensure_venv_t5_550_exists():
            raise RuntimeError(
                f"Cannot activate transformers {TRANSFORMERS_550_VERSION}: "
                f".venv_t5_550 missing at {_VENV_T5_550_DIR}"
            )
        if _VENV_T5_550_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_550_DIR)
        logger.info(
            "Prepended transformers %s venv to sys.path from %s "
            "(path only; the loaded version is confirmed later by "
            "'Subprocess loaded transformers ...' on first import)",
            TRANSFORMERS_550_VERSION,
            _VENV_T5_550_DIR,
        )
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_550_DIR + (os.pathsep + _pp if _pp else "")
    elif tier == "530":
        if not _ensure_venv_t5_530_exists():
            raise RuntimeError(
                f"Cannot activate transformers 5.3.0: "
                f".venv_t5_530 missing at {_VENV_T5_530_DIR}"
            )
        if _VENV_T5_530_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_530_DIR)
        logger.info(
            "Prepended transformers %s venv to sys.path from %s "
            "(path only; the loaded version is confirmed later by "
            "'Subprocess loaded transformers ...' on first import)",
            TRANSFORMERS_530_VERSION,
            _VENV_T5_530_DIR,
        )
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_530_DIR + (os.pathsep + _pp if _pp else "")
    else:
        logger.info("Using default transformers (4.57.x) for %s", model_name)


def _has_adapter_weights(path: Path) -> bool:
    """True if *path* holds LoRA adapter weight files (``adapter_model.*``)."""
    try:
        return any(path.glob("adapter_model*.safetensors")) or any(path.glob("adapter_model*.bin"))
    except OSError:
        return False


def _is_lora_adapter_dir(path: Path) -> bool:
    """True if *path* is a local LoRA adapter directory.

    Mirrors ``utils.models._looks_like_lora_adapter`` but stays import-light so it
    can run during subprocess activation without dragging in transformers. Detects
    both ``adapter_config.json`` adapters and adapter_model-only LoRAs (weights
    present, config absent) that a config-only check would miss.
    """
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    return (path / "adapter_config.json").is_file() or _has_adapter_weights(path)


def _resolve_base_model(model_name: str) -> str:
    """If *model_name* points to a LoRA adapter, return its base model.

    Checks ``adapter_config.json`` locally first. Only calls the heavier
    ``get_base_model_from_lora`` for real local directories (avoids noisy
    warnings for plain HF model IDs). Returns *model_name* unchanged if not a
    LoRA adapter.
    """
    # --- Fast local check ---------------------------------------------------
    local_path = Path(model_name)
    adapter_cfg_path = local_path / "adapter_config.json"
    if adapter_cfg_path.is_file():
        try:
            with open(adapter_cfg_path) as f:
                cfg = json.load(f)
            base = cfg.get("base_model_name_or_path")
            if base:
                logger.info(
                    "Resolved LoRA adapter '%s' → base model '%s'",
                    model_name,
                    base,
                )
                return base
        except Exception as exc:
            logger.debug("Could not read %s: %s", adapter_cfg_path, exc)

    # --- config.json fallback (works for both LoRA and full fine-tune) ------
    config_json_path = local_path / "config.json"
    if config_json_path.is_file():
        try:
            with open(config_json_path) as f:
                cfg = json.load(f)
            # Unsloth writes "model_name"; HF writes "_name_or_path". Try both:
            # if "model_name" is self-referential (equals the local path), the
            # useful base id may still live in "_name_or_path".
            for _key in ("model_name", "_name_or_path"):
                base = cfg.get(_key)
                if base and base != str(local_path):
                    logger.info(
                        "Resolved checkpoint '%s' → base model '%s' (via config.json)",
                        model_name,
                        base,
                    )
                    return base
        except Exception as exc:
            logger.debug("Could not read %s: %s", config_json_path, exc)

    # --- Only try the heavier fallback for genuine LoRA adapters ------------
    # ``get_base_model_from_lora`` returns None for anything that isn't a LoRA
    # adapter, so the only effect of taking this branch for a full checkpoint is
    # the side effect of importing ``utils.models``, which eagerly imports
    # ``transformers``. During subprocess activation that pins the *default*
    # transformers into ``sys.modules`` BEFORE the correct sidecar venv is
    # prepended to ``sys.path``, so the worker then loads the wrong version.
    # Gate on a real adapter_config.json to keep activation import-clean.
    if adapter_cfg_path.is_file():
        try:
            from utils.models import get_base_model_from_lora
            base = get_base_model_from_lora(model_name)
            if base:
                logger.info(
                    "Resolved LoRA adapter '%s' → base model '%s' "
                    "(via get_base_model_from_lora)",
                    model_name,
                    base,
                )
                return base
        except Exception as exc:
            logger.debug(
                "get_base_model_from_lora failed for '%s': %s",
                model_name,
                exc,
            )

    # --- adapter_model-only LoRA (weights but no adapter_config.json) --------
    # These can't be resolved from a config, so fall back to the
    # ``unsloth_<model>_<timestamp>`` directory-name convention (matching
    # get_base_model_from_lora's last-resort branch). This is a pure string
    # parse — no transformers import — so subprocess activation ordering is
    # preserved even though the heavier resolver above is skipped.
    if local_path.name.startswith("unsloth_") and _has_adapter_weights(local_path):
        parts = local_path.name.split("_")
        if len(parts) >= 2:  # unsloth_<model...>_<timestamp>
            base = "unsloth/" + "_".join(parts[1:-1])
            logger.info(
                "Resolved adapter-only LoRA '%s' → base model '%s' (via directory name)",
                model_name,
                base,
            )
            return base

    return model_name


def _check_tokenizer_config_needs_v5(model_name: str) -> bool:
    """True if the model's tokenizer_class requires transformers 5.x.

    Checks local tokenizer_config.json, else fetches from HuggingFace. Cached in
    ``_tokenizer_class_cache``. Returns False on any network/parse error
    (fail-open to default version).
    """
    if model_name in _tokenizer_class_cache:
        return _tokenizer_class_cache[model_name]

    # --- Check local tokenizer_config.json first ---------------------------
    local_path = Path(model_name)
    local_tc = local_path / "tokenizer_config.json"
    if local_tc.is_file():
        try:
            with open(local_tc) as f:
                data = json.load(f)
            tokenizer_class = data.get("tokenizer_class", "")
            result = tokenizer_class in _TRANSFORMERS_5_TOKENIZER_CLASSES
            if result:
                logger.info(
                    "Local check: %s uses tokenizer_class=%s (requires transformers 5.x)",
                    model_name,
                    tokenizer_class,
                )
            _tokenizer_class_cache[model_name] = result
            return result
        except Exception as exc:
            logger.debug("Could not read %s: %s", local_tc, exc)

    # Offline: skip the 10s urllib fetch (fail-open to lower tier).
    if _env_offline():
        _tokenizer_class_cache[model_name] = False
        return False

    # --- Fall back to fetching from HuggingFace ----------------------------
    import urllib.request

    url = f"https://huggingface.co/{model_name}/raw/main/tokenizer_config.json"
    try:
        req = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio"})
        with urllib.request.urlopen(req, timeout = 10) as resp:
            data = json.loads(resp.read().decode())
        tokenizer_class = data.get("tokenizer_class", "")
        result = tokenizer_class in _TRANSFORMERS_5_TOKENIZER_CLASSES
        if result:
            logger.info(
                "Dynamic check: %s uses tokenizer_class=%s (requires transformers 5.x)",
                model_name,
                tokenizer_class,
            )
        _tokenizer_class_cache[model_name] = result
        return result
    except Exception as exc:
        logger.debug("Could not fetch tokenizer_config.json for '%s': %s", model_name, exc)
        _tokenizer_class_cache[model_name] = False
        return False


def _load_config_json(model_name: str, hf_token: str | None = None) -> dict | None:
    """Return parsed ``config.json`` for *model_name*, checking local files first.

    ``hf_token`` authenticates the raw fetch so gated/private repos resolve. The
    cache is keyed on the token so an unauthenticated miss never poisons a later
    authenticated read.
    """
    import hashlib

    tok = hashlib.sha256(hf_token.encode()).hexdigest()[:16] if hf_token else None
    cache_key = (model_name, tok)
    if cache_key in _config_json_cache:
        return _config_json_cache[cache_key]

    local_cfg = Path(model_name) / "config.json"
    if local_cfg.is_file():
        try:
            with open(local_cfg) as f:
                cfg = json.load(f)
            _config_json_cache[cache_key] = cfg
            return cfg
        except Exception as exc:
            logger.debug("Could not read %s: %s", local_cfg, exc)
            _config_json_cache[cache_key] = None
            return None

    if _env_offline():
        _config_json_cache[cache_key] = None
        return None

    import urllib.request

    url = f"https://huggingface.co/{model_name}/raw/main/config.json"
    headers = {"User-Agent": "unsloth-studio"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        req = urllib.request.Request(url, headers = headers)
        with urllib.request.urlopen(req, timeout = 10) as resp:
            cfg = json.loads(resp.read().decode())
        _config_json_cache[cache_key] = cfg
        return cfg
    except Exception as exc:
        logger.debug("Could not fetch config.json for '%s': %s", model_name, exc)
        _config_json_cache[cache_key] = None
        return None


def _config_matches_tier(cfg: dict, architectures: set[str], model_types: set[str]) -> bool:
    archs = cfg.get("architectures", [])
    if any(a in architectures for a in archs):
        return True
    if cfg.get("model_type") in model_types:
        return True
    return False


def _config_needs_550(cfg: dict) -> bool:
    return _config_matches_tier(
        cfg,
        _TRANSFORMERS_550_ARCHITECTURES,
        _TRANSFORMERS_550_MODEL_TYPES,
    )


def _config_needs_510(cfg: dict) -> bool:
    return _config_matches_tier(
        cfg,
        _TRANSFORMERS_510_ARCHITECTURES,
        _TRANSFORMERS_510_MODEL_TYPES,
    )


def _config_needs_530(cfg: dict) -> bool:
    return _config_matches_tier(
        cfg,
        _TRANSFORMERS_530_ARCHITECTURES,
        _TRANSFORMERS_530_MODEL_TYPES,
    )


def _check_config_needs_550(model_name: str) -> bool:
    """True if ``config.json`` has architectures/model_type needing transformers
    5.5.0 (e.g. Gemma 4).

    Checks locally first, else fetches from HuggingFace. Cached in
    ``_config_needs_550_cache``. Returns False on any error (fail-open to lower tier).
    """
    if model_name in _config_needs_550_cache:
        return _config_needs_550_cache[model_name]

    cfg = _load_config_json(model_name)
    if cfg is None:
        _config_needs_550_cache[model_name] = False
        return False

    result = _config_needs_550(cfg)
    if result:
        logger.info(
            "config.json check: %s needs transformers %s (architectures=%s, model_type=%s)",
            model_name,
            TRANSFORMERS_550_VERSION,
            cfg.get("architectures", []),
            cfg.get("model_type"),
        )
    _config_needs_550_cache[model_name] = result
    return result


def _check_config_needs_530(model_name: str) -> bool:
    """Check ``config.json`` for 5.3.0-only architectures (Qwen3.5, Qwen3 MoE, GLM-4.7, LFM2.5-VL).

    Used in the slow HF-ID path for private/renamed repos where name substrings
    aren't reliable.
    """
    if model_name in _config_needs_530_cache:
        return _config_needs_530_cache[model_name]

    cfg = _load_config_json(model_name)
    if cfg is None:
        _config_needs_530_cache[model_name] = False
        return False

    result = _config_needs_530(cfg)
    if result:
        logger.info(
            "config.json check: %s needs transformers %s (architectures=%s, model_type=%s)",
            model_name,
            TRANSFORMERS_530_VERSION,
            cfg.get("architectures", []),
            cfg.get("model_type"),
        )
    _config_needs_530_cache[model_name] = result
    return result


def _check_config_needs_510(model_name: str) -> bool:
    """Check ``config.json`` for Gemma 4 Unified / 12B architectures."""
    if model_name in _config_needs_510_cache:
        return _config_needs_510_cache[model_name]

    cfg = _load_config_json(model_name)
    if cfg is None:
        _config_needs_510_cache[model_name] = False
        return False

    result = _config_needs_510(cfg)
    if result:
        logger.info(
            "config.json check: %s needs transformers %s (architectures=%s, model_type=%s)",
            model_name,
            TRANSFORMERS_510_VERSION,
            cfg.get("architectures", []),
            cfg.get("model_type"),
        )
    _config_needs_510_cache[model_name] = result
    return result


def _norm_separators(s: str) -> str:
    """Collapse ``_`` and whitespace to ``-`` so underscore aliases (e.g.
    ``Qwen3_Next``) match the canonical hyphen substrings. ``.`` is left intact:
    version dots (``qwen3.5``) must not be conflated with size separators
    (``Qwen3-5B`` / ``Qwen3-6B``)."""
    return "".join("-" if ch in "_ \t" else ch for ch in s)


def _looks_like_hf_id(value: str) -> bool:
    """True if *value* looks like a Hub id (``org/name``) rather than a local
    filesystem path, so a stale/renamed checkpoint path isn't name-matched."""
    if os.path.isabs(value) or value.startswith((".", "~")) or "\\" in value:
        return False
    return value.count("/") <= 1


def _tier_from_name(name: str) -> tuple[str, str] | None:
    """Return ``(tier, matched_reason)`` from name-based substring rules, or
    ``None`` if nothing matches.

    Applies the same detection order used by :func:`get_transformers_tier`:
    510 before 550 before 530, with the Gemma-4 assistant special-case first.
    Used both for direct model-name checks and as a fallback when a local
    checkpoint's ``config.json`` architectures aren't yet enumerated in the
    config sets.

    Underscore aliases match (``Qwen3_5`` == ``Qwen3.5``), but a dot-version
    substring (``qwen3.5``/``qwen3.6``) matches only the dot or underscore form,
    never a hyphen, so ``Qwen3-5B``/``Qwen3-6B`` size names aren't promoted.
    """
    lowered = name.lower()
    norm = _norm_separators(lowered)
    dotted = lowered.replace("_", ".")
    if "assistant" in lowered and ("gemma-4" in norm or "gemma4" in norm):
        return "510", "gemma-4 assistant variant"
    for substrings, tier in (
        (TRANSFORMERS_510_MODEL_SUBSTRINGS, "510"),
        (TRANSFORMERS_550_MODEL_SUBSTRINGS, "550"),
        (TRANSFORMERS_5_MODEL_SUBSTRINGS, "530"),
    ):
        for s in substrings:
            if "." in s:
                if s in lowered or s in dotted:
                    return tier, s
            elif s in lowered or _norm_separators(s) in norm:
                return tier, s
    return None


def get_transformers_tier(model_name: str) -> str:
    """Return the transformers tier required for *model_name*.

    Returns ``"510"`` for models needing transformers 5.10.x (Gemma 4 Unified),
    ``"550"`` for models needing transformers 5.5.0 (Gemma 4),
    ``"530"`` for models needing transformers 5.3.0 (e.g. Ministral-3, Qwen3 MoE),
    or ``"default"`` for everything else (4.57.x).

    Higher 5.x tiers run first.  For local paths, ``config.json`` is checked
    before name heuristics to avoid false-positives from directory name fragments.
    """
    # --- Local checkpoint path ---
    # config.json acts as a positive-signal oracle: if it matches a known
    # sidecar architecture, return immediately.  If it doesn't, we fall back
    # to the HF ID embedded in the config rather than the filesystem path, so
    # renamed folders are handled correctly and parent-dir false-positives are
    # avoided.
    local_cfg = Path(model_name) / "config.json"
    if local_cfg.is_file():
        cfg = _load_config_json(model_name)
        if cfg is not None:
            if _config_needs_510(cfg):
                logger.info(
                    "Transformers tier 510 selected for %s (local config.json check)",
                    model_name,
                )
                return "510"
            if _config_needs_550(cfg):
                logger.info(
                    "Transformers tier 550 selected for %s (local config.json check)",
                    model_name,
                )
                return "550"
            if _config_needs_530(cfg):
                # Qwen3.6 reuses Qwen3.5 config ids (qwen3_5 / qwen3_5_moe) but is
                # a 5.5 model by name; let a higher-tier name match override 530.
                # Only trust the resolved value as a name hint when it's a real
                # Hub id — a stale/renamed local path saved in model_name/
                # _name_or_path (e.g. /old/run/qwen3.6-source) must not flip a
                # correct 530 config to 550 via arbitrary path substrings. Fall
                # back to the current folder's basename otherwise.
                base = _resolve_base_model(model_name)
                hint_src = (
                    base
                    if (base != model_name and _looks_like_hf_id(base))
                    else Path(model_name).name
                )
                hint = _tier_from_name(hint_src)
                if hint is not None and hint[0] in ("510", "550"):
                    logger.info(
                        "Transformers tier %s selected for %s (name overrides 530 config)",
                        hint[0],
                        model_name,
                    )
                    return hint[0]
                logger.info(
                    "Transformers tier 530 selected for %s (local config.json check)",
                    model_name,
                )
                return "530"
            # Architecture not in any config set — resolve the base model name
            # (_name_or_path / model_name in config) and detect tier from it.
            # Split on whether the resolved value is a local path or a HF Hub ID:
            #   local dir  → recurse (config.json check, no network I/O) so a
            #                 self-referencing absolute path doesn't false-positive
            #                 on directory-name substrings.
            #   HF Hub ID  → _tier_from_name only (no network probes).
            resolved = _resolve_base_model(model_name)
            if resolved != model_name:
                if Path(resolved).is_dir():
                    tier = get_transformers_tier(resolved)
                    if tier != "default":
                        logger.info(
                            "Transformers tier %s selected for %s (resolved local path: %s)",
                            tier,
                            model_name,
                            resolved,
                        )
                        return tier
                elif _looks_like_hf_id(resolved):
                    result = _tier_from_name(resolved)
                    if result is not None:
                        tier, match = result
                        logger.info(
                            "Transformers tier %s selected for %s (resolved HF ID: %s, match: %s)",
                            tier,
                            model_name,
                            resolved,
                            match,
                        )
                        return tier
            local_tc = Path(model_name) / "tokenizer_config.json"
            if local_tc.is_file() and _check_tokenizer_config_needs_v5(model_name):
                logger.info(
                    "Transformers tier 530 selected for %s (local tokenizer_config.json check)",
                    model_name,
                )
                return "530"
            logger.info(
                "Transformers tier default (4.57.x) selected for %s (local config.json no match)",
                model_name,
            )
            return "default"

    # --- Fast substring checks (no I/O) ------------------------------------
    result = _tier_from_name(model_name)
    if result is not None:
        tier, match = result
        logger.info(
            "Transformers tier %s selected for %s (substring match: %s)",
            tier,
            model_name,
            match,
        )
        return tier

    # --- Slow config fallbacks (network for HF IDs) ------------------------
    if _check_config_needs_510(model_name):
        logger.info("Transformers tier 510 selected for %s (config.json check)", model_name)
        return "510"
    if _check_config_needs_550(model_name):
        logger.info("Transformers tier 550 selected for %s (config.json check)", model_name)
        return "550"
    if _check_config_needs_530(model_name):
        logger.info("Transformers tier 530 selected for %s (config.json check)", model_name)
        return "530"
    if _check_tokenizer_config_needs_v5(model_name):
        logger.info(
            "Transformers tier 530 selected for %s (tokenizer_config.json check)",
            model_name,
        )
        return "530"

    logger.info("Transformers tier default (4.57.x) selected for %s (no match)", model_name)
    return "default"


def needs_transformers_5(model_name: str) -> bool:
    """Return True if *model_name* requires any transformers 5.x version.

    Convenience wrapper around :func:`get_transformers_tier`.
    """
    return get_transformers_tier(model_name) != "default"


# ---------------------------------------------------------------------------
# Version switching (in-process — used only by export)
# ---------------------------------------------------------------------------


def _get_in_memory_version() -> str | None:
    """Return the transformers version currently loaded in this process."""
    tf = sys.modules.get("transformers")
    if tf is not None:
        return getattr(tf, "__version__", None)
    return None


# All top-level prefixes that hold references to transformers internals.
_PURGE_PREFIXES = (
    "transformers",
    "huggingface_hub",
    "unsloth",
    "unsloth_zoo",
    "peft",
    "trl",
    "accelerate",
    "auto_gptq",
    # NOTE: bitsandbytes is intentionally EXCLUDED -- it registers torch custom
    # operators via torch.library.define() into torch's global registry, which
    # survives module purge; re-importing after purge -> duplicate registration
    # -> crash.
    # Our own modules that import from transformers at module level.
    "utils.models",
    "core.training",
    "core.inference",
    "core.export",
)


def _purge_modules() -> int:
    """Remove all cached modules for transformers and its dependents.

    Returns the number of modules purged.
    """
    importlib.invalidate_caches()
    to_remove = [
        k
        for k in list(sys.modules.keys())
        if any(k == p or k.startswith(p + ".") for p in _PURGE_PREFIXES)
    ]
    for key in to_remove:
        del sys.modules[key]
    return len(to_remove)


_VENV_T5_530_PACKAGES = (
    f"transformers=={TRANSFORMERS_530_VERSION}",
    "huggingface_hub==1.8.0",
    "hf_xet==1.4.2",
    "tiktoken",
)

_VENV_T5_510_PACKAGES = (
    f"transformers=={TRANSFORMERS_510_VERSION}",
    "huggingface_hub==1.8.0",
    "hf_xet==1.4.2",
    "tiktoken",
)

_VENV_T5_550_PACKAGES = (
    f"transformers=={TRANSFORMERS_550_VERSION}",
    "huggingface_hub==1.8.0",
    "hf_xet==1.4.2",
    "tiktoken",
)

# Backwards-compat alias
_VENV_T5_PACKAGES = _VENV_T5_550_PACKAGES


def _venv_dir_is_valid(venv_dir: str, packages: tuple[str, ...]) -> bool:
    """Return True if *venv_dir* has all *packages* at the correct versions."""
    if not os.path.isdir(venv_dir) or not os.listdir(venv_dir):
        return False
    for pkg_spec in packages:
        parts = pkg_spec.split("==")
        pkg_name = parts[0]
        pkg_version = parts[1] if len(parts) > 1 else None
        pkg_name_norm = pkg_name.replace("-", "_")
        # Directory must exist.
        if not any(
            (Path(venv_dir) / d).is_dir() for d in (pkg_name_norm, pkg_name_norm.replace("_", "-"))
        ):
            return False
        # Unpinned packages: existence is enough.
        if pkg_version is None:
            continue
        # Check version via .dist-info metadata.
        dist_info_found = False
        for di in Path(venv_dir).glob(f"{pkg_name_norm}-*.dist-info"):
            metadata = di / "METADATA"
            if not metadata.is_file():
                continue
            for line in metadata.read_text(errors = "replace").splitlines():
                if line.startswith("Version:"):
                    installed_ver = line.split(":", 1)[1].strip()
                    if installed_ver != pkg_version:
                        logger.warning(
                            "%s has %s==%s but need %s -- venv will be wiped and reinstalled",
                            venv_dir,
                            pkg_name,
                            installed_ver,
                            pkg_version,
                        )
                        return False
                    dist_info_found = True
                    break
            if dist_info_found:
                break
        if not dist_info_found:
            return False
    return True


def _venv_t5_is_valid() -> bool:
    """Backwards-compat: check the Gemma 4 sidecar venv."""
    return _venv_dir_is_valid(_VENV_T5_550_DIR, _VENV_T5_550_PACKAGES)


def _install_to_dir(pkg: str, target_dir: str) -> bool:
    """Install a single package into *target_dir*, preferring uv then pip."""
    # Try uv first (faster) if on PATH -- do NOT install uv at runtime.
    if shutil.which("uv"):
        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                target_dir,
                "--no-deps",
                "--upgrade",
                pkg,
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
        if result.returncode == 0:
            return True
        logger.warning("uv install of %s failed, falling back to pip", pkg)

    # Fallback to pip.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            target_dir,
            "--no-deps",
            "--upgrade",
            pkg,
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        env = child_env_without_native_path_secret(),
        **_windows_hidden_subprocess_kwargs(),
    )
    if result.returncode != 0:
        logger.error("install failed:\n%s", result.stdout)
        return False
    return True


def _ensure_venv_dir(venv_dir: str, packages: tuple[str, ...], label: str) -> bool:
    """Ensure *venv_dir* exists with all *packages*. Install if missing."""
    if _venv_dir_is_valid(venv_dir, packages):
        return True

    logger.warning("%s not found or incomplete at %s -- installing at runtime", label, venv_dir)
    shutil.rmtree(venv_dir, ignore_errors = True)
    os.makedirs(venv_dir, exist_ok = True)
    total = len(packages)
    for idx, pkg in enumerate(packages, start = 1):
        logger.info("Installing %s (%d/%d) into %s ...", pkg, idx, total, venv_dir)
        if not _install_to_dir(pkg, venv_dir):
            return False
    logger.info("Installed %s to %s", label, venv_dir)
    return True


def _ensure_venv_t5_530_exists() -> bool:
    """Ensure .venv_t5_530/ exists with transformers 5.3.0."""
    return _ensure_venv_dir(_VENV_T5_530_DIR, _VENV_T5_530_PACKAGES, "transformers 5.3.0")


def _ensure_venv_t5_550_exists() -> bool:
    """Ensure .venv_t5_550/ exists with transformers 5.5.0."""
    return _ensure_venv_dir(
        _VENV_T5_550_DIR,
        _VENV_T5_550_PACKAGES,
        f"transformers {TRANSFORMERS_550_VERSION}",
    )


def _ensure_venv_t5_510_exists() -> bool:
    """Ensure .venv_t5_510/ exists with transformers 5.10.x."""
    return _ensure_venv_dir(
        _VENV_T5_510_DIR,
        _VENV_T5_510_PACKAGES,
        f"transformers {TRANSFORMERS_510_VERSION}",
    )


def _ensure_venv_t5_exists() -> bool:
    """Backwards-compat: ensure the Gemma 4 5.5 sidecar venv exists."""
    return _ensure_venv_t5_550_exists()


def _activate_venv(venv_dir: str, label: str) -> None:
    """Prepend *venv_dir* to sys.path, purge stale modules, reimport."""
    if venv_dir not in sys.path:
        sys.path.insert(0, venv_dir)
        logger.info("Prepended %s to sys.path", venv_dir)

    count = _purge_modules()
    logger.info("Purged %d cached modules", count)

    import transformers

    logger.info("Loaded transformers %s (%s)", transformers.__version__, label)


def _deactivate_5x() -> None:
    """Remove all .venv_t5_*/ dirs from sys.path, purge stale modules, reimport."""
    for d in (_VENV_T5_530_DIR, _VENV_T5_550_DIR, _VENV_T5_510_DIR):
        while d in sys.path:
            sys.path.remove(d)
    logger.info("Removed venv_t5 dirs from sys.path")

    count = _purge_modules()
    logger.info("Purged %d cached modules", count)

    import transformers

    logger.info("Reverted to transformers %s", transformers.__version__)


def ensure_transformers_version(model_name: str) -> None:
    """Ensure the correct ``transformers`` version is active for *model_name*.

    Uses sys.path with .venv_t5_510/, .venv_t5_550/, or .venv_t5_530/
    (pre-installed by setup.sh):
      • Need 5.10.x → prepend .venv_t5_510/ to sys.path, purge modules.
      • Need 5.5.0 → prepend .venv_t5_550/ to sys.path, purge modules.
      • Need 5.3.0 → prepend .venv_t5_530/ to sys.path, purge modules.
      • Need 4.x  → remove all .venv_t5_*/ from sys.path, purge modules.

    For custom-named LoRA adapters, the base model is resolved before checking
    (from ``adapter_config.json`` or, for adapter_model-only LoRAs, the directory
    name).

    NOTE: Training and inference use subprocess isolation instead. Used only by
    the export path (routes/export.py).
    """
    # Only pre-resolve for LoRA adapter dirs; see activate_transformers_for_subprocess.
    if _is_lora_adapter_dir(Path(model_name)):
        resolved = _resolve_base_model(model_name)
    else:
        resolved = model_name
    tier = get_transformers_tier(resolved)

    if tier == "510":
        target_version = TRANSFORMERS_510_VERSION
        venv_dir = _VENV_T5_510_DIR
        ensure_fn = _ensure_venv_t5_510_exists
    elif tier == "550":
        target_version = TRANSFORMERS_550_VERSION
        venv_dir = _VENV_T5_550_DIR
        ensure_fn = _ensure_venv_t5_550_exists
    elif tier == "530":
        target_version = TRANSFORMERS_530_VERSION
        venv_dir = _VENV_T5_530_DIR
        ensure_fn = _ensure_venv_t5_530_exists
    else:
        target_version = TRANSFORMERS_DEFAULT_VERSION
        venv_dir = None
        ensure_fn = None

    target_major = int(target_version.split(".")[0])

    # Check what's actually loaded in memory
    in_memory = _get_in_memory_version()

    logger.info(
        "Version check for '%s' (resolved: '%s'): need=%s, in_memory=%s",
        model_name,
        resolved,
        target_version,
        in_memory,
    )

    # --- Already correct? ---------------------------------------------------
    if in_memory is not None:
        if in_memory == target_version:
            logger.info(
                "transformers %s already loaded — correct for '%s'",
                in_memory,
                model_name,
            )
            return
        # Different 5.x -> need to switch (e.g. 5.3.0 loaded but need 5.10.x).
        in_memory_major = int(in_memory.split(".")[0])
        if in_memory_major == target_major and venv_dir is None:
            # Both are default (4.x) — close enough.
            logger.info(
                "transformers %s already loaded — correct for '%s'",
                in_memory,
                model_name,
            )
            return

    # --- Switch version -----------------------------------------------------
    if venv_dir is not None:
        # First remove any other 5.x venv from sys.path.
        _deactivate_5x()
        if not ensure_fn():
            raise RuntimeError(
                f"Cannot activate transformers {target_version}: " f"venv missing at {venv_dir}"
            )
        logger.info("Activating transformers %s…", target_version)
        _activate_venv(venv_dir, f"transformers {target_version}")
    else:
        logger.info("Reverting to default transformers %s…", TRANSFORMERS_DEFAULT_VERSION)
        _deactivate_5x()

    final = _get_in_memory_version()
    logger.info("✓ transformers version is now %s", final)
