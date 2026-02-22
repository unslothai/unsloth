"""
Automatic transformers version switching.

Some newer model architectures (Ministral-3, GLM-4.7-Flash, Qwen3-30B-A3B MoE,
tiny_qwen3_moe) require transformers>=5.1.0, while everything else needs the
default 4.57.x that ships with Unsloth.

When loading a LoRA adapter with a custom name, we resolve the base model from
``adapter_config.json`` and check *that* against the model list.

Strategy (sys.path overlay):
  • The default transformers (4.57.x) always lives in site-packages.
  • When 5.x is needed, we ``pip install --target <dir> --no-deps`` to a
    separate directory and **prepend** it to ``sys.path``.
  • To revert, we simply **remove** that directory from ``sys.path``.
  • After either change we purge cached modules so the next import picks
    up the correct version.
"""

import importlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure our logger is visible even if root logger isn't configured for INFO.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(
        logging.Formatter("[%(name)s|%(levelname)s]%(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Lowercase substrings — if ANY appears anywhere in the lowered model name,
# we need transformers 5.x.
TRANSFORMERS_5_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "ministral-3-",       # Ministral-3-{3,8,14}B-{Instruct,Reasoning,Base}-2512
    "glm-4.7-flash",      # GLM-4.7-Flash
    "qwen3-30b-a3b",      # Qwen3-30B-A3B-Instruct-2507 and variants
    "tiny_qwen3_moe",     # imdatta0/tiny_qwen3_moe_2.8B_0.7B
)

# Versions
TRANSFORMERS_5_VERSION = "5.1.0"
TRANSFORMERS_DEFAULT_VERSION = "4.57.1"

# Persistent directory for the transformers 5.x overlay — lives next to .venv/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # studio/backend/utils/ → project root
_OVERLAY_DIR = str(_PROJECT_ROOT / ".venv_overlay")


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
                    model_name, base,
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
                    model_name, base,
                )
                return base
        except Exception as exc:
            logger.debug(
                "get_base_model_from_lora failed for '%s': %s",
                model_name, exc,
            )

    return model_name


def needs_transformers_5(model_name: str) -> bool:
    """Return True if *model_name* belongs to an architecture that requires
    ``transformers>=5.1.0``."""
    lowered = model_name.lower()
    return any(sub in lowered for sub in TRANSFORMERS_5_MODEL_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Version switching
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
    "bitsandbytes",
)


def _purge_modules() -> int:
    """Remove all cached modules for transformers and its dependents.

    Returns the number of modules purged.
    """
    importlib.invalidate_caches()
    to_remove = [
        k for k in list(sys.modules.keys())
        if any(k == p or k.startswith(p + ".") for p in _PURGE_PREFIXES)
    ]
    for key in to_remove:
        del sys.modules[key]
    return len(to_remove)


# Packages to install into the overlay (each with --no-deps).
_OVERLAY_PACKAGES = (
    f"transformers=={TRANSFORMERS_5_VERSION}",
    "huggingface_hub>=1.3.0,<2.0",
)


def _install_overlay() -> None:
    """Install transformers 5.x and its critical deps into the overlay
    directory and prepend it to ``sys.path``."""
    # Check if overlay needs (re)creation.
    # If the overlay exists but is missing huggingface_hub, it's stale.
    needs_install = (
        not os.path.isdir(_OVERLAY_DIR)
        or not os.listdir(_OVERLAY_DIR)
        or not os.path.isdir(os.path.join(_OVERLAY_DIR, "huggingface_hub"))
    )

    if needs_install:
        # Clean up stale overlay if present
        if os.path.isdir(_OVERLAY_DIR):
            shutil.rmtree(_OVERLAY_DIR)
        os.makedirs(_OVERLAY_DIR, exist_ok=True)

        for pkg in _OVERLAY_PACKAGES:
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--target", _OVERLAY_DIR,
                "--no-deps",
                pkg,
            ]
            logger.info("Installing %s to overlay: %s", pkg, " ".join(cmd))
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0:
                logger.error("pip install failed:\n%s", result.stdout)
                raise RuntimeError(
                    f"Failed to install {pkg} to {_OVERLAY_DIR}.\n"
                    f"pip output:\n{result.stdout}"
                )
        logger.info("Overlay install succeeded")
    else:
        logger.info("Overlay directory already exists at %s", _OVERLAY_DIR)

    # Prepend to sys.path (if not already there)
    if _OVERLAY_DIR not in sys.path:
        sys.path.insert(0, _OVERLAY_DIR)
        logger.info("Prepended %s to sys.path", _OVERLAY_DIR)

    # Purge old modules and force fresh import
    count = _purge_modules()
    logger.info("Purged %d cached modules", count)

    import transformers
    logger.info("Loaded transformers %s from overlay", transformers.__version__)


def _remove_overlay() -> None:
    """Remove the overlay directory from ``sys.path`` so the default
    site-packages version (4.57.x) takes effect again."""
    # Remove from sys.path
    changed = False
    while _OVERLAY_DIR in sys.path:
        sys.path.remove(_OVERLAY_DIR)
        changed = True

    if changed:
        logger.info("Removed %s from sys.path", _OVERLAY_DIR)

    # Purge old modules and force fresh import
    count = _purge_modules()
    logger.info("Purged %d cached modules", count)

    import transformers
    logger.info("Reverted to transformers %s from site-packages",
                transformers.__version__)


def ensure_transformers_version(model_name: str) -> None:
    """Ensure the correct ``transformers`` version is active for *model_name*.

    Uses sys.path overlay:
      • Need 5.x → install to separate dir, prepend sys.path, purge modules.
      • Need 4.x → remove overlay from sys.path, purge modules.

    For LoRA adapters with custom names, the base model is resolved from
    ``adapter_config.json`` before checking.

    Call this at the top of every model-loading code path (training ``/start``,
    inference ``/load``, export ``/load-checkpoint``).
    """
    # Resolve LoRA adapters to their base model for accurate detection
    resolved = _resolve_base_model(model_name)
    want_5 = needs_transformers_5(resolved)
    target_version = TRANSFORMERS_5_VERSION if want_5 else TRANSFORMERS_DEFAULT_VERSION
    target_major = int(target_version.split(".")[0])

    # Check what's actually loaded in memory
    in_memory = _get_in_memory_version()
    overlay_active = _OVERLAY_DIR in sys.path

    logger.info(
        "Version check for '%s' (resolved: '%s'): need=%s, "
        "in_memory=%s, overlay_active=%s",
        model_name, resolved, target_version, in_memory, overlay_active,
    )

    # --- Already correct? ---------------------------------------------------
    if in_memory is not None:
        in_memory_major = int(in_memory.split(".")[0])
        if in_memory_major == target_major:
            logger.info(
                "transformers %s already loaded — correct for '%s'",
                in_memory, model_name,
            )
            return

    # --- Switch version -----------------------------------------------------
    if want_5:
        logger.info("Activating transformers %s overlay…", TRANSFORMERS_5_VERSION)
        _install_overlay()
    else:
        logger.info("Reverting to default transformers %s…", TRANSFORMERS_DEFAULT_VERSION)
        _remove_overlay()

    final = _get_in_memory_version()
    logger.info("✓ transformers version is now %s", final)
