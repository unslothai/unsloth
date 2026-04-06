# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Automatic transformers version switching.

Some newer model architectures (Ministral-3, GLM-4.7-Flash, Qwen3-30B-A3B MoE,
tiny_qwen3_moe) require transformers>=5.3.0, while Gemma 4 models require
transformers>=5.5.0.  Everything else needs the default 4.57.x that ships
with Unsloth.

Two separate target directories are maintained:
  - .venv_t5_530/  — transformers 5.3.0 (Ministral-3, GLM, Qwen3 MoE, etc.)
  - .venv_t5_550/  — transformers 5.5.0 (Gemma 4)

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


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Lowercase substrings — if ANY appears anywhere in the lowered model name,
# we need transformers 5.3.0.
TRANSFORMERS_5_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "ministral-3-",  # Ministral-3-{3,8,14}B-{Instruct,Reasoning,Base}-2512
    "glm-4.7-flash",  # GLM-4.7-Flash
    "qwen3-30b-a3b",  # Qwen3-30B-A3B-Instruct-2507 and variants
    "qwen3.5",  # Qwen3.5 family (35B-A3B, etc.)
    "qwen3-next",  # Qwen3-Next and variants
    "tiny_qwen3_moe",  # imdatta0/tiny_qwen3_moe_2.8B_0.7B
    "lfm2.5-vl-450m",  # LiquidAI/LFM2.5-VL-450M
)

# Lowercase substrings for models that require transformers 5.5.0 (checked first).
TRANSFORMERS_550_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "gemma-4",  # Gemma-4 (E2B-it, E4B-it, 31B-it, 26B-A4B-it)
    "gemma4",  # Gemma-4 alternate naming
    "qwen3.6",
)

# Architecture classes that require transformers 5.5.0.
_TRANSFORMERS_550_ARCHITECTURES: set[str] = {
    "Gemma4ForConditionalGeneration",
}

# model_type values (from config.json) → tier mapping.
_TRANSFORMERS_550_MODEL_TYPES: set[str] = {
    "gemma4",
}
_TRANSFORMERS_530_MODEL_TYPES: set[str] = {
    "qwen3_moe",
    "qwen3_5_moe",
    "qwen3_vl_moe",
    "qwen3_next",
    "deepseek_v3_moe",
    "glm4_moe",
    "glm4_moe_lite",
    "ministral",
}

# Tokenizer classes that only exist in transformers>=5.x
_TRANSFORMERS_5_TOKENIZER_CLASSES: set[str] = {
    "TokenizersBackend",
}

# Cache for dynamic tokenizer_config.json lookups to avoid repeated fetches
_tokenizer_class_cache: dict[str, bool] = {}

# Cache for config.json lookups (returns the parsed dict or None)
_config_json_cache: dict[str, dict | None] = {}

# Versions
TRANSFORMERS_550_VERSION = "5.5.0"
TRANSFORMERS_530_VERSION = "5.3.0"
TRANSFORMERS_DEFAULT_VERSION = "4.57.6"
# Backwards-compat alias used by other modules
TRANSFORMERS_5_VERSION = TRANSFORMERS_550_VERSION

# Pre-installed directories — created by setup.sh / setup.ps1
_VENV_T5_530_DIR = str(Path.home() / ".unsloth" / "studio" / ".venv_t5_530")
_VENV_T5_550_DIR = str(Path.home() / ".unsloth" / "studio" / ".venv_t5_550")
# Backwards-compat alias
_VENV_T5_DIR = _VENV_T5_550_DIR


def _resolve_base_model(model_name: str) -> str:
    """If *model_name* points to a LoRA adapter, return its base model.

    Checks for ``adapter_config.json`` locally first.  Only calls the heavier
    ``get_base_model_from_lora`` for paths that are actual local directories
    (avoids noisy warnings for plain HF model IDs).

    Returns the original *model_name* unchanged if it is not a LoRA adapter.
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
            # Unsloth writes "model_name"; HF writes "_name_or_path"
            base = cfg.get("model_name") or cfg.get("_name_or_path")
            if base and base != str(local_path):
                logger.info(
                    "Resolved checkpoint '%s' → base model '%s' (via config.json)",
                    model_name,
                    base,
                )
                return base
        except Exception as exc:
            logger.debug("Could not read %s: %s", config_json_path, exc)

    # --- Only try the heavier fallback for local directories ----------------
    if local_path.is_dir():
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

    return model_name


def _check_tokenizer_config_needs_v5(model_name: str) -> bool:
    """Fetch tokenizer_config.json from HuggingFace and check if the
    tokenizer_class requires transformers 5.x.

    Results are cached in ``_tokenizer_class_cache`` to avoid repeated fetches.
    Returns False on any network/parse error (fail-open to default version).
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
        logger.debug(
            "Could not fetch tokenizer_config.json for '%s': %s", model_name, exc
        )
        _tokenizer_class_cache[model_name] = False
        return False


_SENTINEL = object()  # distinguishes "not cached" from "cached as None"


def _get_config_json(model_name: str) -> dict | None:
    """Read and cache ``config.json`` for *model_name*.

    Checks local path first, then fetches from HuggingFace.
    Returns the parsed dict, or ``None`` on any error (fail-open).
    The result is cached in ``_config_json_cache``.
    """
    cached = _config_json_cache.get(model_name, _SENTINEL)
    if cached is not _SENTINEL:
        return cached

    # --- Check local config.json first ------------------------------------
    local_cfg = Path(model_name) / "config.json"
    if local_cfg.is_file():
        try:
            with open(local_cfg) as f:
                cfg = json.load(f)
            _config_json_cache[model_name] = cfg
            return cfg
        except Exception as exc:
            logger.debug("Could not read %s: %s", local_cfg, exc)

    # --- Fall back to fetching from HuggingFace ---------------------------
    import urllib.request

    url = f"https://huggingface.co/{model_name}/raw/main/config.json"
    try:
        req = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio"})
        with urllib.request.urlopen(req, timeout = 10) as resp:
            cfg = json.loads(resp.read().decode())
        _config_json_cache[model_name] = cfg
        return cfg
    except Exception as exc:
        logger.debug(
            "Could not fetch config.json for '%s': %s", model_name, exc
        )
        _config_json_cache[model_name] = None
        return None


def _check_config_needs_550(model_name: str) -> bool:
    """Check ``config.json`` for architectures or model_type that require
    transformers 5.5.0.  Uses the shared ``_get_config_json`` cache."""
    cfg = _get_config_json(model_name)
    if cfg is None:
        return False
    archs = cfg.get("architectures", [])
    if any(a in _TRANSFORMERS_550_ARCHITECTURES for a in archs):
        return True
    return cfg.get("model_type") in _TRANSFORMERS_550_MODEL_TYPES


def get_transformers_tier(model_name: str) -> str:
    """Return the transformers tier required for *model_name*.

    Returns ``"550"`` for models needing transformers 5.5.0 (e.g. Gemma 4),
    ``"530"`` for models needing transformers 5.3.0 (e.g. Ministral-3, Qwen3 MoE),
    or ``"default"`` for everything else (4.57.x).

    Fast path: substring checks (no I/O) for both tiers run first.
    Slow path: single config.json fetch (cached) checks model_type for
    both tiers, then tokenizer_config.json as final fallback.
    """
    lowered = model_name.lower()

    # --- Fast substring checks (no I/O) -----------------------------------
    if any(sub in lowered for sub in TRANSFORMERS_550_MODEL_SUBSTRINGS):
        return "550"
    if any(sub in lowered for sub in TRANSFORMERS_5_MODEL_SUBSTRINGS):
        return "530"

    # --- config.json model_type / architecture check (single fetch) -------
    cfg = _get_config_json(model_name)
    if cfg is not None:
        model_type = cfg.get("model_type", "")
        archs = cfg.get("architectures", [])

        # Check 5.5.0 first
        if model_type in _TRANSFORMERS_550_MODEL_TYPES:
            return "550"
        if any(a in _TRANSFORMERS_550_ARCHITECTURES for a in archs):
            return "550"

        # Check 5.3.0
        if model_type in _TRANSFORMERS_530_MODEL_TYPES:
            return "530"

    # --- Final fallback: tokenizer_config.json for 5.3.0 ------------------
    if _check_tokenizer_config_needs_v5(model_name):
        return "530"

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
    # NOTE: bitsandbytes is intentionally EXCLUDED — it registers torch custom
    # operators at import time via torch.library.define(). Those registrations
    # live in torch's global operator registry which survives module purge.
    # Re-importing bitsandbytes after purge → duplicate registration → crash.
    # Our own modules that import from transformers at module level
    # (e.g. model_config.py: `from transformers import AutoConfig`)
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
        # Check directory exists
        if not any(
            (Path(venv_dir) / d).is_dir()
            for d in (pkg_name_norm, pkg_name_norm.replace("_", "-"))
        ):
            return False
        # For unpinned packages, existence is enough
        if pkg_version is None:
            continue
        # Check version via .dist-info metadata
        dist_info_found = False
        for di in Path(venv_dir).glob(f"{pkg_name_norm}-*.dist-info"):
            metadata = di / "METADATA"
            if not metadata.is_file():
                continue
            for line in metadata.read_text(errors = "replace").splitlines():
                if line.startswith("Version:"):
                    installed_ver = line.split(":", 1)[1].strip()
                    if installed_ver != pkg_version:
                        logger.info(
                            "%s has %s==%s but need %s",
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
    """Backwards-compat: check the 5.5.0 venv."""
    return _venv_dir_is_valid(_VENV_T5_550_DIR, _VENV_T5_550_PACKAGES)


def _install_to_dir(pkg: str, target_dir: str) -> bool:
    """Install a single package into *target_dir*, preferring uv then pip."""
    # Try uv first (faster) if already on PATH -- do NOT install uv at runtime
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

    # Fallback to pip
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

    logger.warning(
        "%s not found or incomplete at %s -- installing at runtime", label, venv_dir
    )
    shutil.rmtree(venv_dir, ignore_errors = True)
    os.makedirs(venv_dir, exist_ok = True)
    for pkg in packages:
        if not _install_to_dir(pkg, venv_dir):
            return False
    logger.info("Installed %s to %s", label, venv_dir)
    return True


def _ensure_venv_t5_530_exists() -> bool:
    """Ensure .venv_t5_530/ exists with transformers 5.3.0."""
    return _ensure_venv_dir(
        _VENV_T5_530_DIR, _VENV_T5_530_PACKAGES, "transformers 5.3.0"
    )


def _ensure_venv_t5_550_exists() -> bool:
    """Ensure .venv_t5_550/ exists with transformers 5.5.0."""
    return _ensure_venv_dir(
        _VENV_T5_550_DIR, _VENV_T5_550_PACKAGES, "transformers 5.5.0"
    )


def _ensure_venv_t5_exists() -> bool:
    """Backwards-compat: ensure the 5.5.0 venv exists."""
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
    for d in (_VENV_T5_530_DIR, _VENV_T5_550_DIR):
        while d in sys.path:
            sys.path.remove(d)
    logger.info("Removed venv_t5 dirs from sys.path")

    count = _purge_modules()
    logger.info("Purged %d cached modules", count)

    import transformers

    logger.info("Reverted to transformers %s", transformers.__version__)


def ensure_transformers_version(model_name: str) -> None:
    """Ensure the correct ``transformers`` version is active for *model_name*.

    Uses sys.path with .venv_t5_530/ or .venv_t5_550/ (pre-installed by setup.sh):
      • Need 5.5.0 → prepend .venv_t5_550/ to sys.path, purge modules.
      • Need 5.3.0 → prepend .venv_t5_530/ to sys.path, purge modules.
      • Need 4.x  → remove all .venv_t5_*/ from sys.path, purge modules.

    For LoRA adapters with custom names, the base model is resolved from
    ``adapter_config.json`` before checking.

    NOTE: Training and inference use subprocess isolation instead of this
    function. This is only used by the export path (routes/export.py).
    """
    # Resolve LoRA adapters to their base model for accurate detection
    resolved = _resolve_base_model(model_name)
    tier = get_transformers_tier(resolved)

    if tier == "550":
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
        # Different 5.x → need to switch (e.g. 5.3.0 loaded but need 5.5.0)
        in_memory_major = int(in_memory.split(".")[0])
        if in_memory_major == target_major and venv_dir is None:
            # Both are default (4.x) — close enough
            logger.info(
                "transformers %s already loaded — correct for '%s'",
                in_memory,
                model_name,
            )
            return

    # --- Switch version -----------------------------------------------------
    if venv_dir is not None:
        # First remove any other 5.x venv from sys.path
        _deactivate_5x()
        if not ensure_fn():
            raise RuntimeError(
                f"Cannot activate transformers {target_version}: "
                f"venv missing at {venv_dir}"
            )
        logger.info("Activating transformers %s…", target_version)
        _activate_venv(venv_dir, f"transformers {target_version}")
    else:
        logger.info(
            "Reverting to default transformers %s…", TRANSFORMERS_DEFAULT_VERSION
        )
        _deactivate_5x()

    final = _get_in_memory_version()
    logger.info("✓ transformers version is now %s", final)
