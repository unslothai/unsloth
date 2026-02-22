"""
Automatic transformers version switching.

Some newer model architectures (Ministral-3, GLM-4.7-Flash, Qwen3-30B-A3B MoE,
tiny_qwen3_moe) require transformers>=5.1.0, while everything else needs the
default 4.57.x that ships with Unsloth.

When loading a LoRA adapter with a custom name, we resolve the base model from
``adapter_config.json`` and check *that* against the model list.

This module detects the model being loaded and ensures the correct transformers
version is installed before proceeding.  The pip install is done *without*
``--force-reinstall`` to avoid rebuilding unrelated dependencies.
"""

import importlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

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


def _resolve_base_model(model_name: str) -> str:
    """If *model_name* points to a LoRA adapter, return its base model.

    Checks for ``adapter_config.json`` locally first (covers the common case of
    local output directories with custom names).  For HF repo IDs that look
    like they might be LoRA adapters, falls back to
    ``get_base_model_from_lora``.

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
                    "Resolved LoRA adapter '%s' → base model '%s'", model_name, base,
                )
                return base
        except Exception as exc:
            logger.debug("Could not read %s: %s", adapter_cfg_path, exc)

    # --- Only try the heavier fallback for paths that look like local dirs ---
    # (Avoids triggering noisy warnings for plain HF model IDs like
    # "unsloth/GLM-4.7-Flash" which are obviously not LoRA adapters.)
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
                "get_base_model_from_lora failed for '%s': %s", model_name, exc,
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
    """Return the transformers version currently loaded in this process,
    or None if transformers hasn't been imported yet."""
    tf = sys.modules.get("transformers")
    if tf is not None:
        return getattr(tf, "__version__", None)
    return None


def _get_on_disk_version() -> str | None:
    """Return the transformers version installed on disk (pip metadata)."""
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None


def _pip_install(package_spec: str) -> None:
    """Run ``pip install <package_spec>`` using the running interpreter's pip.

    Mirrors the approach used in ``unsloth_zoo.llama_cpp.check_pip`` —
    ``sys.executable -m pip`` is always the safest choice.
    """
    cmd = [sys.executable, "-m", "pip", "install", package_spec]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        logger.error("pip install failed:\n%s", result.stdout)
        raise RuntimeError(
            f"Failed to install {package_spec}. "
            f"pip output:\n{result.stdout}"
        )
    logger.info("pip install succeeded for %s", package_spec)


def _reload_transformers() -> None:
    """Purge transformers AND all packages that cache transformers internals
    from ``sys.modules``, then force a fresh re-import.

    Simply clearing ``transformers.*`` is not enough because libraries like
    ``unsloth``, ``peft``, ``trl``, and ``accelerate`` bind transformers
    classes/functions into their own module-level state.  We must evict them
    all so the next ``import`` picks up the freshly pip-installed version.
    """
    importlib.invalidate_caches()

    # All top-level prefixes that hold references to transformers internals.
    _PREFIXES = (
        "transformers",
        "unsloth",
        "unsloth_zoo",
        "peft",
        "trl",
        "accelerate",
        "auto_gptq",
        "bitsandbytes",
    )

    to_remove = [
        k for k in list(sys.modules.keys())
        if any(k == p or k.startswith(p + ".") for p in _PREFIXES)
    ]
    for key in to_remove:
        del sys.modules[key]

    logger.info("Purged %d cached modules (%s)", len(to_remove),
                ", ".join(_PREFIXES))

    # Force a fresh import so the new version is loaded into the process.
    import transformers  # noqa: F811
    logger.info("Re-imported transformers — version is now %s",
                transformers.__version__)


def ensure_transformers_version(model_name: str) -> None:
    """Ensure the correct ``transformers`` version is installed for *model_name*.

    Checks BOTH the on-disk version (pip metadata) and the in-memory version
    (``transformers.__version__``) because they can diverge after a previous
    pip install in the same process.

    * If the model needs 5.x and the loaded version is already 5.x → no-op.
    * If the model needs 5.x but 4.x is loaded → pip install 5.1.0 + reload.
    * If the model does NOT need 5.x but 5.x is loaded → downgrade + reload.
    * Otherwise → no-op.

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

    # --- Check what's actually loaded in memory first -----------------------
    in_memory = _get_in_memory_version()
    on_disk = _get_on_disk_version()

    logger.info(
        "Version check for '%s' (resolved: '%s'): need=%s, "
        "in_memory=%s, on_disk=%s",
        model_name, resolved, target_version, in_memory, on_disk,
    )

    if in_memory is not None:
        in_memory_major = int(in_memory.split(".")[0])
        if in_memory_major == target_major:
            logger.info(
                "transformers %s in memory — already correct for '%s'",
                in_memory, model_name,
            )
            return
        # Wrong major in memory — need to switch
        logger.info(
            "transformers %s loaded in memory but need %s — switching…",
            in_memory, target_version,
        )
    elif on_disk is not None:
        on_disk_major = int(on_disk.split(".")[0])
        if on_disk_major == target_major:
            logger.info(
                "transformers %s on disk (not yet imported) — correct for '%s'",
                on_disk, model_name,
            )
            return
        logger.info(
            "transformers %s on disk but need %s — switching…",
            on_disk, target_version,
        )
    else:
        logger.warning("transformers is not installed — skipping version check")
        return

    # --- pip install the target version if needed ---------------------------
    if on_disk is None or int(on_disk.split(".")[0]) != target_major:
        _pip_install(f"transformers=={target_version}")

    # --- Purge and reload ---------------------------------------------------
    _reload_transformers()

    final = _get_in_memory_version()
    logger.info("Transformers version is now %s (in memory)", final)
