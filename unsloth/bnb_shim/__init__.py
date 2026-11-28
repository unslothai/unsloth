# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""4-bit QLoRA without bitsandbytes: registers Unsloth's stand-in package (./bitsandbytes) under the
name `bitsandbytes` when the real one is not installed, or is installed but cannot load its native
library on a CUDA / ROCm GPU. A working bitsandbytes is never shadowed.

The stand-in is found by a meta path finder that also serves its distribution metadata, so
transformers', accelerate's and PEFT's availability and version checks pass and pre-quantized
bnb-4bit checkpoints load through their usual code paths. NF4 runs on Unsloth's Triton kernels.

UNSLOTH_BNB_SHIM: unset / "auto" as above, "0" never, "1" always (even over a working install).
Stdlib only at import: unsloth/__init__.py installs this before transformers is imported.
"""

import importlib.abc
import importlib.machinery
import importlib.metadata
import os
import sys

__all__ = ["SHIM_VERSION", "install", "is_active", "is_shim", "refuse_int8", "vllm_blocked"]

SHIM_VERSION = "0.49.2"
_ROOT = os.path.dirname(os.path.abspath(__file__))
_NAME = "bitsandbytes"
_STATE = {"active": False, "reason": None}


def _mode():
    value = os.environ.get("UNSLOTH_BNB_SHIM", "auto").strip().lower()
    if value in ("0", "false", "no", "off", "never"):
        return "off"
    if value in ("1", "true", "yes", "on", "force", "always"):
        return "force"
    return "auto"


class _ShimDistribution(importlib.metadata.Distribution):
    _TEXT = {
        "METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {_NAME}\n"
            f"Version: {SHIM_VERSION}\n"
            "Summary: Unsloth NF4 stand-in for bitsandbytes (Triton kernels, no native library)\n"
        ),
        "top_level.txt": f"{_NAME}\n",
        "INSTALLER": "unsloth\n",
    }

    def read_text(self, filename):
        return self._TEXT.get(filename)

    def locate_file(self, path):
        return os.path.join(_ROOT, str(path))


class _ShimFinder(importlib.abc.MetaPathFinder):
    _unsloth_bnb_shim = True

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname != _NAME and not fullname.startswith(_NAME + "."):
            return None
        search = [_ROOT] if fullname == _NAME else path
        if not search:
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, search)

    def find_distributions(self, context = importlib.metadata.DistributionFinder.Context()):
        name = getattr(context, "name", None)
        if name is None or name.lower().replace("-", "_").replace(".", "_") == _NAME:
            yield _ShimDistribution()


def is_shim(module = None):
    """True if `module` (default: the imported `bitsandbytes`) is the stand-in."""
    if module is None:
        module = sys.modules.get(_NAME)
        if module is None:
            return _STATE["active"]
    return bool(getattr(module, "__unsloth_shim__", False))


def is_active():
    return _STATE["active"]


def _purge():
    for name in [m for m in sys.modules if m == _NAME or m.startswith(_NAME + ".")]:
        sys.modules.pop(name, None)


def _has_gpu():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _real_unusable():
    """Installed bitsandbytes that cannot run on this machine's CUDA / ROCm GPU: the import raises,
    or its native library did not load (bitsandbytes >= 0.46 then imports with throwing stubs)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        device_type = "hip" if torch.version.hip else "cuda"
    except Exception:
        return False
    try:
        # Pin BNB_ROCM_VERSION first (Windows ROCm wheels whose DLL suffix differs from
        # torch.version.hip), as _gpu_init does, or a working install would probe as broken.
        from unsloth.import_fixes import maybe_set_windows_rocm_bnb_version
        maybe_set_windows_rocm_bnb_version()
    except Exception:
        pass
    try:
        import bitsandbytes
    except Exception:
        return True
    try:
        from unsloth.bnb_availability import native_kernels_ready
        return not native_kernels_ready(bitsandbytes, device_type)
    except Exception:
        return False


def _refresh_availability_caches():
    """transformers / PEFT / accelerate imported earlier may have cached 'not installed'."""
    import_utils = sys.modules.get("transformers.utils.import_utils")
    if import_utils is not None:
        try:
            if getattr(import_utils, "_bitsandbytes_available", None) is False:
                import_utils._bitsandbytes_available = True
            for name in ("is_bitsandbytes_available", "is_bitsandbytes_multi_backend_available"):
                getattr(getattr(import_utils, name, None), "cache_clear", lambda: None)()
        except Exception:
            pass
    integration = sys.modules.get("transformers.integrations.bitsandbytes")
    if integration is not None and getattr(integration, "bnb", None) is None:
        # Imported while bitsandbytes was missing, so its module-level `import bitsandbytes as bnb`
        # never ran; bind the stand-in its functions read.
        try:
            import bitsandbytes
            integration.bnb = bitsandbytes
        except Exception:
            pass
    for module_name, names in (
        ("peft.import_utils", ("is_bnb_available", "is_bnb_4bit_available")),
        (
            "accelerate.utils.imports",
            ("is_bnb_available", "is_4bit_bnb_available", "is_8bit_bnb_available"),
        ),
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name in names:
            try:
                getattr(getattr(module, name, None), "cache_clear", lambda: None)()
            except Exception:
                pass


def install():
    """Register the stand-in if wanted; returns True when `import bitsandbytes` gives the stand-in."""
    if _STATE["active"]:
        return True
    mode = _mode()
    if mode == "off":
        return False
    loaded = sys.modules.get(_NAME)
    if loaded is not None and is_shim(loaded):
        # Already the stand-in (registered by another copy of this module): keep its classes.
        _STATE["active"] = True
        return True
    if loaded is not None and mode == "auto":
        # Imported before Unsloth: whatever it is, the process already uses it.
        return False
    real = loaded is not None or _real_installed()
    if mode == "auto" and not _has_gpu():
        # Its kernels need a CUDA / ROCm GPU; elsewhere nothing would use 4-bit anyway.
        return False
    if mode == "auto" and real:
        if not _real_unusable():
            return False
        reason = "its native library cannot load on this GPU"
    elif mode == "force":
        reason = "UNSLOTH_BNB_SHIM=1"
    else:
        reason = "it is not installed"
    _purge()
    finder = _ShimFinder()
    if real or mode == "force":
        sys.meta_path.insert(0, finder)
    else:
        sys.meta_path.append(finder)
    _STATE["active"] = True
    _STATE["reason"] = reason
    _refresh_availability_caches()
    if os.environ.get("UNSLOTH_BNB_SHIM_QUIET", "0") != "1":
        print(
            f"Unsloth: bitsandbytes is unavailable ({reason}) - 4bit QLoRA uses Unsloth's own NF4 "
            "kernels instead (UNSLOTH_BNB_SHIM=0 turns this off). 8-bit optimizers fall back to "
            "torch, LLM.int8 is unsupported."
        )
    return True


def _real_installed():
    import importlib.util
    try:
        return importlib.util.find_spec(_NAME) is not None
    except (ImportError, ValueError):
        return False


INT8_MESSAGE = (
    "Unsloth: 8-bit LLM.int8 (load_in_8bit=True) needs the bitsandbytes package, which is not "
    "installed or could not load its native library. Use load_in_4bit=True (Unsloth runs NF4 "
    "without bitsandbytes) or 16-bit, or install bitsandbytes."
)


def refuse_int8(load_in_8bit):
    """Loaders call this before weights load: LLM.int8 needs the native library."""
    if load_in_8bit and is_shim():
        raise RuntimeError(INT8_MESSAGE)


def vllm_blocked(
    model_name, load_in_4bit, load_in_8bit, load_in_16bit, full_finetuning, quantization_config
):
    """True when fast_inference would hand vLLM a bitsandbytes model while the stand-in is active:
    vLLM's bitsandbytes loader needs the native library. 16-bit loads keep vLLM."""
    if not is_shim():
        return False
    if load_in_8bit:
        return True
    config = quantization_config
    if isinstance(config, dict):
        wants = (
            config.get("load_in_4bit")
            or config.get("load_in_8bit")
            or "bitsandbytes" in str(config.get("quant_method", ""))
        )
    elif config is not None:
        wants = getattr(config, "load_in_4bit", False) or getattr(config, "load_in_8bit", False)
    else:
        wants = False
    if wants or "bnb" in str(model_name).lower():
        return True
    return bool(load_in_4bit) and not load_in_16bit and not full_finetuning
