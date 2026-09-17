# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's copy of ``unsloth/import_fixes.py::fix_torchao_safe_int_mm_repr_probe``, not an import
of it: the diffusion process must not execute ``unsloth/__init__``. The two copies are mutually inert.

torchao's int8 GEMM asks whether it is being traced by formatting its input's repr, which on a real
CUDA tensor calls ``.item()``: a device sync per eager int8 linear, and an outright
``cudaErrorStreamCaptureUnsupported`` inside ``torch.cuda.graph``. Installed from a meta path finder
at import time, not at quantise time, because the int8 prequant path only ``torch.load``s torchao
subclasses and never calls ``quantize_``.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import inspect
import logging
import os
import sys

logger = logging.getLogger(__name__)


_TORCHAO_INTMM_MODULES = (
    "torchao.kernel.intmm",  # every release up to and including 0.18.0
    "torchao.quantization.quantize_.workflows.int8.kernels",  # main after pytorch/ao#4718
)
_TORCHAO_INTMM_MODULE = _TORCHAO_INTMM_MODULES[0]  # kept for callers that named the old home
_TORCHAO_INTMM_SENTINEL = "__unsloth_torchao_intmm_patch__"
_TORCHAO_INT_MM_ENV = "UNSLOTH_TORCHAO_INT_MM_FIX"

# The copy below hard-codes torchao's cuBLAS guards and fp32 fallback: ALL must be in the installed source.
_TORCHAO_SAFE_INT_MM_MARKERS = (
    "input.__repr__()",
    "dynamo_is_compiling()",
    "out_dtype(torch.ops.aten.mm.default",
    "j_is_nonzero_multiple_of_8",
    "k_is_nonzero_multiple_of_8",
    "mat2.is_contiguous()",
)


def _is_fake_tensor(x):
    """Covers exactly the set the substring repr probe did: ``is_fake`` unwraps FunctionalTensor too."""
    try:
        from torch._subclasses.fake_tensor import is_fake
        return bool(is_fake(x))
    except Exception:
        pass
    try:
        from torch._subclasses.fake_tensor import FakeTensor
        return isinstance(x, FakeTensor)
    except Exception:
        return False


def _make_safe_int_mm(mod, original):
    """A FULL copy of torchao 0.17.0's body, not a wrapper: its very first statement IS the probe."""
    import torch

    out_dtype = mod.out_dtype
    dynamo_is_compiling = mod.dynamo_is_compiling

    @functools.wraps(original)
    def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        if dynamo_is_compiling() or _is_fake_tensor(input):
            if input.device.type == "cpu":
                # Matmul in int32 is slow on CPU and not supported well by Inductor cpp backend
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, input.float(), mat2.float()
                )
            return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)

        assert (
            mat2.device == input.device
        ), f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
        device_cpu = "cpu" in [mat2.device.type, input.device.type]
        j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
        k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
        bad_dimensions_for_cublas = not (j_is_nonzero_multiple_of_8 and k_is_nonzero_multiple_of_8)

        if device_cpu or bad_dimensions_for_cublas:
            return torch.matmul(input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)).to(
                input.device.type
            )

        if not mat2.is_contiguous():  # silently gives incorrect result without this
            mat2 = mat2.contiguous()
        if (not input.is_contiguous()) and (
            input.shape[0] % 8 != 0
        ):  # gives cryptic error without this
            input = input.contiguous()
        try:
            return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)
        except Exception:
            # H100 float8: "addmm_cuda" not implemented for 'Float8_e4m3fn'
            return torch.matmul(input.to(torch.float32), mat2.to(torch.float32)).to(torch.int32)

    safe_int_mm.__unsloth_patched__ = True
    safe_int_mm.__unsloth_original__ = original
    return safe_int_mm


def _patch_torchao_intmm_module(mod):
    """Rebind ``safe_int_mm`` on either torchao home; True when this call installed it."""
    original = getattr(mod, "safe_int_mm", None)
    if original is None or not callable(original):
        return False
    if getattr(original, "__unsloth_patched__", False):
        return False
    # Off the module, never imported here, so this fails closed rather than borrowing our own operators.
    if not hasattr(mod, "out_dtype") or not hasattr(mod, "dynamo_is_compiling"):
        return False
    try:
        source = inspect.getsource(original)
    except Exception:
        return False
    missing = [marker for marker in _TORCHAO_SAFE_INT_MM_MARKERS if marker not in source]
    if missing:
        if "input.__repr__()" not in missing:
            logger.warning(
                "Unsloth: torchao's safe_int_mm still probes input.__repr__() but its body "
                "changed (%s missing), so the capture-safe replacement was not installed. "
                "Eager int8 keeps syncing the device on every linear.",
                ", ".join(missing),
            )
        return False
    patched = _make_safe_int_mm(mod, original)
    try:
        mod.safe_int_mm = patched
    except Exception:
        return False
    # `torchao.kernel` and `torchao.quantization` re-export the function OBJECT, so they need a sweep.
    for name, other in tuple(sys.modules.items()):
        if other is None or not (name == "torchao" or name.startswith("torchao.")):
            continue
        if getattr(other, "safe_int_mm", None) is original:
            try:
                setattr(other, "safe_int_mm", patched)
            except Exception:
                pass
    return True


class _TorchaoIntmmLoader(importlib.abc.Loader):
    """The real loader, plus the patch once the module body finishes. Failures here leave torchao unpatched."""

    __slots__ = ("_loader",)

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            _patch_torchao_intmm_module(module)
        except Exception:
            pass

    def __getattr__(self, attribute):
        return getattr(self._loader, attribute)


class _TorchaoIntmmPatchFinder(importlib.abc.MetaPathFinder):
    """Inserted at the FRONT of sys.meta_path: the module really exists, so PathFinder would answer first."""

    __slots__ = (_TORCHAO_INTMM_SENTINEL, "_finding")

    def __init__(self):
        setattr(self, _TORCHAO_INTMM_SENTINEL, True)
        self._finding = False  # find_spec below walks sys.meta_path again

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname not in _TORCHAO_INTMM_MODULES or self._finding:
            return None
        self._finding = True
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            return None
        finally:
            self._finding = False
        if spec is None or spec.loader is None:
            return None
        if not hasattr(spec.loader, "exec_module"):
            return None  # a loader from before PEP 451; leave the import entirely alone
        try:
            spec.loader = _TorchaoIntmmLoader(spec.loader)
        except Exception:
            return None
        return spec


def install_torchao_int_mm_patch():
    """Install the capture-safe ``safe_int_mm`` if torchao is present; idempotent, so several modules may each
    ask for it. ``UNSLOTH_TORCHAO_INT_MM_FIX=0`` keeps upstream's behaviour. True when patched or the finder was
    installed, False when there is nothing to do, None when torchao is absent or the fix is off."""
    if os.environ.get(_TORCHAO_INT_MM_ENV, "1").strip() == "0":
        return None
    try:
        if importlib.util.find_spec("torchao") is None:
            return None
    except Exception:
        return None
    patched_now = False
    for name in _TORCHAO_INTMM_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        try:
            patched_now = _patch_torchao_intmm_module(module) or patched_now
        except Exception:
            pass
    if all(name in sys.modules for name in _TORCHAO_INTMM_MODULES):
        return patched_now  # nothing left for a finder to catch
    for finder in sys.meta_path:
        if getattr(finder, _TORCHAO_INTMM_SENTINEL, False):
            return patched_now
    sys.meta_path.insert(0, _TorchaoIntmmPatchFinder())
    return True
