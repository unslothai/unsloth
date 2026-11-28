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

__all__ = ["SHIM_VERSION", "install", "is_active", "is_shim", "loader_guard"]

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
    return bool(getattr(module, "__unsloth_shim__", False))


def is_active():
    return _STATE["active"]


def _purge():
    for name in [m for m in sys.modules if m == _NAME or m.startswith(_NAME + ".")]:
        sys.modules.pop(name, None)


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
    if loaded is not None and not is_shim(loaded) and mode == "auto":
        # Imported before Unsloth: whatever it is, the process already uses it.
        return False
    real = loaded is not None or _real_installed()
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
    if real:
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


def loader_guard(load_in_4bit, load_in_8bit, fast_inference):
    """Called by the loaders before weights load. With the stand-in active: refuse load_in_8bit, and
    turn off fast_inference for 4-bit (vLLM's bitsandbytes loader needs the native library).
    Returns the fast_inference to use."""
    if not is_shim():
        return fast_inference
    if load_in_8bit:
        raise RuntimeError(INT8_MESSAGE)
    if fast_inference and load_in_4bit:
        print(
            "Unsloth: fast_inference (vLLM) loads 4-bit weights with the bitsandbytes native library, "
            "which is unavailable - continuing with fast_inference = False."
        )
        return False
    return fast_inference
