# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Automatic transformers version switching.

Some newer model architectures (Ministral-3, GLM-4.7-Flash, Qwen3-30B-A3B MoE,
tiny_qwen3_moe) require transformers>=5.3.0, while everything else needs the
default 4.57.x that ships with Unsloth.

When loading a LoRA adapter with a custom name, we resolve the base model from
``adapter_config.json`` and check *that* against the model list.

Strategy:
  Training and inference run in subprocesses that activate the correct version
  via sys.path (prepending .venv_t5/ for 5.x models). See:
    - core/training/worker.py
    - core/inference/worker.py

  For export (still in-process), ensure_transformers_version() does a lightweight
  sys.path swap using the same .venv_t5/ directory pre-installed by setup.sh.
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

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Lowercase substrings — if ANY appears anywhere in the lowered model name,
# we need transformers 5.x.
TRANSFORMERS_5_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "ministral-3-",  # Ministral-3-{3,8,14}B-{Instruct,Reasoning,Base}-2512
    "glm-4.7-flash",  # GLM-4.7-Flash
    "qwen3-30b-a3b",  # Qwen3-30B-A3B-Instruct-2507 and variants
    "qwen3.5",  # Qwen3.5 family (35B-A3B, etc.)
    "qwen3-next",  # Qwen3-Next and variants
    "tiny_qwen3_moe",  # imdatta0/tiny_qwen3_moe_2.8B_0.7B
)

# Tokenizer classes that only exist in transformers>=5.x
_TRANSFORMERS_5_TOKENIZER_CLASSES: set[str] = {
    "TokenizersBackend",
}

# Cache for dynamic tokenizer_config.json lookups to avoid repeated fetches
_tokenizer_class_cache: dict[str, bool] = {}

# Versions
TRANSFORMERS_5_VERSION = "5.3.0"
TRANSFORMERS_DEFAULT_VERSION = "4.57.6"

# Pre-installed directory for transformers 5.x — created by setup.sh / setup.ps1
_VENV_T5_DIR = str(Path.home() / ".unsloth" / "studio" / ".venv_t5")


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


def needs_transformers_5(model_name: str) -> bool:
    """Return True if *model_name* belongs to an architecture that requires
    ``transformers>=5.3.0``.

    First checks the hardcoded substring list for known models, then
    dynamically fetches ``tokenizer_config.json`` from HuggingFace to check
    if the tokenizer_class (e.g. ``TokenizersBackend``) requires v5.
    """
    lowered = model_name.lower()
    if any(sub in lowered for sub in TRANSFORMERS_5_MODEL_SUBSTRINGS):
        return True
    return _check_tokenizer_config_needs_v5(model_name)


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


_VENV_T5_PACKAGES = (
    f"transformers=={TRANSFORMERS_5_VERSION}",
    "huggingface_hub==1.7.1",
    "hf_xet==1.4.2",
    "tiktoken",
)


def _venv_t5_is_valid() -> bool:
    """Return True if .venv_t5/ has all required packages at the correct versions."""
    if not os.path.isdir(_VENV_T5_DIR) or not os.listdir(_VENV_T5_DIR):
        return False
    # Check that the key package directories exist AND match the required version
    for pkg_spec in _VENV_T5_PACKAGES:
        parts = pkg_spec.split("==")
        pkg_name = parts[0]
        pkg_version = parts[1] if len(parts) > 1 else None
        pkg_name_norm = pkg_name.replace("-", "_")
        # Check directory exists
        if not any(
            (Path(_VENV_T5_DIR) / d).is_dir()
            for d in (pkg_name_norm, pkg_name_norm.replace("_", "-"))
        ):
            return False
        # For unpinned packages, existence is enough
        if pkg_version is None:
            continue
        # Check version via .dist-info metadata
        dist_info_found = False
        for di in Path(_VENV_T5_DIR).glob(f"{pkg_name_norm}-*.dist-info"):
            metadata = di / "METADATA"
            if not metadata.is_file():
                continue
            for line in metadata.read_text(errors = "replace").splitlines():
                if line.startswith("Version:"):
                    installed_ver = line.split(":", 1)[1].strip()
                    if installed_ver != pkg_version:
                        logger.info(
                            ".venv_t5 has %s==%s but need %s",
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


def _install_to_venv_t5(pkg: str) -> bool:
    """Install a single package into .venv_t5/, preferring uv then pip."""
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
                _VENV_T5_DIR,
                "--no-deps",
                "--upgrade",
                pkg,
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
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
            _VENV_T5_DIR,
            "--no-deps",
            "--upgrade",
            pkg,
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
    )
    if result.returncode != 0:
        logger.error("install failed:\n%s", result.stdout)
        return False
    return True


def _ensure_venv_t5_exists() -> bool:
    """Ensure .venv_t5/ exists with all required packages. Install if missing."""
    if _venv_t5_is_valid():
        return True

    logger.warning(
        ".venv_t5 not found or incomplete at %s -- installing at runtime", _VENV_T5_DIR
    )
    shutil.rmtree(_VENV_T5_DIR, ignore_errors = True)
    os.makedirs(_VENV_T5_DIR, exist_ok = True)
    for pkg in _VENV_T5_PACKAGES:
        if not _install_to_venv_t5(pkg):
            return False
    logger.info("Installed transformers 5.x to %s", _VENV_T5_DIR)
    return True


def _activate_5x() -> None:
    """Prepend .venv_t5/ to sys.path, purge stale modules, reimport."""
    if not _ensure_venv_t5_exists():
        raise RuntimeError(
            f"Cannot activate transformers 5.x: .venv_t5 missing at {_VENV_T5_DIR}"
        )

    if _VENV_T5_DIR not in sys.path:
        sys.path.insert(0, _VENV_T5_DIR)
        logger.info("Prepended %s to sys.path", _VENV_T5_DIR)

    count = _purge_modules()
    logger.info("Purged %d cached modules", count)

    import transformers

    logger.info("Loaded transformers %s", transformers.__version__)


def _deactivate_5x() -> None:
    """Remove .venv_t5/ from sys.path, purge stale modules, reimport."""
    while _VENV_T5_DIR in sys.path:
        sys.path.remove(_VENV_T5_DIR)
    logger.info("Removed %s from sys.path", _VENV_T5_DIR)

    count = _purge_modules()
    logger.info("Purged %d cached modules", count)

    import transformers

    logger.info("Reverted to transformers %s", transformers.__version__)


def ensure_transformers_version(model_name: str) -> None:
    """Ensure the correct ``transformers`` version is active for *model_name*.

    Uses sys.path with .venv_t5/ (pre-installed by setup.sh):
      • Need 5.x → prepend .venv_t5/ to sys.path, purge modules.
      • Need 4.x → remove .venv_t5/ from sys.path, purge modules.

    For LoRA adapters with custom names, the base model is resolved from
    ``adapter_config.json`` before checking.

    NOTE: Training and inference use subprocess isolation instead of this
    function. This is only used by the export path (routes/export.py).
    """
    # Resolve LoRA adapters to their base model for accurate detection
    resolved = _resolve_base_model(model_name)
    want_5 = needs_transformers_5(resolved)
    target_version = TRANSFORMERS_5_VERSION if want_5 else TRANSFORMERS_DEFAULT_VERSION
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
        in_memory_major = int(in_memory.split(".")[0])
        if in_memory_major == target_major:
            logger.info(
                "transformers %s already loaded — correct for '%s'",
                in_memory,
                model_name,
            )
            return

    # --- Switch version -----------------------------------------------------
    if want_5:
        logger.info("Activating transformers %s via .venv_t5…", TRANSFORMERS_5_VERSION)
        _activate_5x()
    else:
        logger.info(
            "Reverting to default transformers %s…", TRANSFORMERS_DEFAULT_VERSION
        )
        _deactivate_5x()

    final = _get_in_memory_version()
    logger.info("✓ transformers version is now %s", final)
