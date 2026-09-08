# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's copy of the torchao ``safe_int_mm`` fix.

The same patch lives in ``unsloth/import_fixes.py``
(``fix_torchao_safe_int_mm_repr_probe``), and this file is a deliberate second
copy rather than an import of it: the diffusion process does NOT import
unsloth, and ``import unsloth.import_fixes`` would execute ``unsloth/__init__``
and ``_gpu_init``, which the diffusion paths avoid on purpose. The two copies
are mutually inert -- whichever runs first wins, because the second one sees
either the ``__unsloth_patched__`` attribute on the replacement or the
sentinel on the finder already in ``sys.meta_path`` -- so a process that has
both (the LM side and a diffusion backend in one interpreter) patches once.

What it fixes: torchao's int8 GEMM asks whether it is being traced by
formatting the repr of its input, which on a real CUDA tensor calls ``.item()``
-- a device sync on every eager int8 linear, and an outright
``cudaErrorStreamCaptureUnsupported`` inside ``torch.cuda.graph``. Studio hits
this on the int8 transformer-quant path AND on the int8 prequant path (which
only ``torch.load``s torchao subclasses and never calls ``quantize_``), so the
patch installs from a meta path finder at import time, not at quantise time.

stdlib-only at import: torch is touched only inside the finder/loader, i.e.
only once something has already imported torchao, so importing this module
costs a few string compares in the spawned probe child.
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


# torchao's int8 GEMM asks "am I being traced?" with `"FakeTensor" in input.__repr__()`
# (torchao/kernel/intmm.py::safe_int_mm, same body from 0.10.0 through v0.18.0; on main after
# pytorch/ao#4718 the function lives in torchao/quantization/quantize_/workflows/int8/kernels.py
# with the probe intact). On a real CUDA tensor __repr__ formats the element values, so it calls
# .item(): a full device sync on EVERY eager int8 linear, and cudaErrorStreamCaptureUnsupported
# the moment that lands inside a torch.cuda.graph capture. Replacing the string probe with the
# question it was approximating (is this tensor fake?) was measured bit-identical on the GEMM
# itself, on a compiled Linear and on a whole compiled DiT, and lets int8 capture.
_TORCHAO_INTMM_MODULES = (
    "torchao.kernel.intmm",  # every release up to and including 0.18.0
    "torchao.quantization.quantize_.workflows.int8.kernels",  # main after the int8 workflow move
)
_TORCHAO_INTMM_MODULE = _TORCHAO_INTMM_MODULES[0]  # kept for callers that named the old home
_TORCHAO_INTMM_SENTINEL = "__unsloth_torchao_intmm_patch__"
_TORCHAO_INT_MM_ENV = "UNSLOTH_TORCHAO_INT_MM_FIX"

# The landmarks of the body that was verified bit-identical. ALL of them must be in the
# installed source before the copy below is allowed to stand in for it: the copy hard-codes
# torchao's cuBLAS dimension guards, its contiguity fixes and its fp32 fallback, so a body that
# no longer matches is a body this fix must not impersonate.
_TORCHAO_SAFE_INT_MM_MARKERS = (
    "input.__repr__()",
    "dynamo_is_compiling()",
    "out_dtype(torch.ops.aten.mm.default",
    "j_is_nonzero_multiple_of_8",
    "k_is_nonzero_multiple_of_8",
    "mat2.is_contiguous()",
)


def _is_fake_tensor(x):
    """Exactly what the "FakeTensor" substring probe was trying to answer.

    ``torch._subclasses.fake_tensor.is_fake`` unwraps FunctionalTensor and the
    wrapper subclasses whose repr also said "FakeTensor", so it covers the same
    set the string test did. ``isinstance`` is the fallback for a torch that
    does not export the helper; a torch without either answers False, which
    only costs the compile branch that dynamo_is_compiling() already catches.
    """
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
    """torchao 0.17.0's ``safe_int_mm`` body with only the probe replaced.

    A FULL copy on purpose. The original's very first statement IS the probe,
    so a wrapper that tested for fake tensors and then delegated the eager
    branch to the original would sync exactly as before (this shortcut was
    written, measured and thrown away). Everything else here is torchao's:
    the compile branch, the cuBLAS dimension guards, the contiguity fixes and
    the fp32 fallback for float8 on H100.
    """
    import torch

    # torchao picks these per torch version, so take ITS bindings instead of importing our own:
    # the copy has to run the operators the module it replaces would have run.
    out_dtype = mod.out_dtype
    dynamo_is_compiling = mod.dynamo_is_compiling

    @functools.wraps(original)
    def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        # torch.compile path
        if dynamo_is_compiling() or _is_fake_tensor(input):
            if input.device.type == "cpu":
                # Matmul in int32 is slow on CPU and not supported well by Inductor cpp backend
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, input.float(), mat2.float()
                )
            return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)

        # error checking for cublas path
        assert (
            mat2.device == input.device
        ), f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
        device_cpu = "cpu" in [mat2.device.type, input.device.type]
        # with input.shape = [i,j] and mat2.shape = [j,k]
        j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
        k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
        bad_dimensions_for_cublas = not (j_is_nonzero_multiple_of_8 and k_is_nonzero_multiple_of_8)

        if device_cpu or bad_dimensions_for_cublas:
            # fallback path
            return torch.matmul(input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)).to(
                input.device.type
            )

        # cublas paths
        if not mat2.is_contiguous():  # silently gives incorrect result without this
            mat2 = mat2.contiguous()
        if (not input.is_contiguous()) and (
            input.shape[0] % 8 != 0
        ):  # gives cryptic error without this
            input = (
                input.contiguous()
            )  # (it seems the transpose makes cublas check the above j constraint on i)
        try:
            return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)
        except Exception:
            # fallback path, would run on H100 for float8 dtypes
            # Exception on H100 float8 dtype : "addmm_cuda" not implemented for 'Float8_e4m3fn'
            return torch.matmul(input.to(torch.float32), mat2.to(torch.float32)).to(torch.int32)

    safe_int_mm.__unsloth_patched__ = True
    safe_int_mm.__unsloth_original__ = original
    return safe_int_mm


def _patch_torchao_intmm_module(mod):
    """Rebind ``safe_int_mm`` on an imported module that defines it (either torchao home).

    Returns True when this call installed the replacement, False when there is
    nothing to do: already patched, or a body this fix declines to recognise.
    """
    original = getattr(mod, "safe_int_mm", None)
    if original is None or not callable(original):
        return False
    if getattr(original, "__unsloth_patched__", False):
        return False  # the other copy of this fix got there first
    # Resolved off the module, never imported here, so the gate fails closed on a torchao that
    # stops carrying them rather than on a copy that silently uses different operators.
    if not hasattr(mod, "out_dtype") or not hasattr(mod, "dynamo_is_compiling"):
        return False
    try:
        source = inspect.getsource(original)
    except Exception:
        return False
    missing = [marker for marker in _TORCHAO_SAFE_INT_MM_MARKERS if marker not in source]
    if missing:
        if "input.__repr__()" not in missing:
            # The sync is still there but the body around it moved, so the verified copy is no
            # longer a copy of what torchao runs. Say so once and leave torchao alone.
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
    # int_matmul and int_scaled_matmul live in this module and resolve the name through module
    # globals, so they follow the rebind for free. `torchao.kernel` and `torchao.quantization`
    # re-export the function OBJECT, so those bindings need the sweep.
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
    """The real loader, plus the patch the moment the module body finishes.

    Every failure here degrades to "torchao is unpatched", never to a broken
    import: the wrapped loader runs first and its result is returned untouched.
    """

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
        # is_package / get_source / get_resource_reader and friends still belong to the real one.
        return getattr(self._loader, attribute)


class _TorchaoIntmmPatchFinder(importlib.abc.MetaPathFinder):
    """Patches the module that defines ``safe_int_mm`` the instant something imports it.

    Inserted at the FRONT of sys.meta_path, unlike the appended alias finders
    next to it: this module really exists, so PathFinder would answer first and
    a finder at the back would never be consulted. It answers only for the two
    dotted names torchao has kept the function under, and hands back the real
    spec with the loader wrapped, so the import is byte for byte the one that
    would have happened. The re-entrancy
    flag is for the find_spec below, which walks sys.meta_path again.
    """

    __slots__ = (_TORCHAO_INTMM_SENTINEL, "_finding")

    def __init__(self):
        setattr(self, _TORCHAO_INTMM_SENTINEL, True)
        self._finding = False

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
    """Install the capture-safe ``safe_int_mm`` if torchao is present.

    Idempotent and safe to call from several modules: a second call finds the
    replacement (or the finder) already in place and no-ops. Set
    ``UNSLOTH_TORCHAO_INT_MM_FIX=0`` to keep upstream's behaviour.

    Returns True when patched or when the finder was installed, False when
    there is nothing to do, None when torchao is absent or the fix is disabled.
    """
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
            # A stubbed or half-built torchao is not worth an import error in the caller.
            pass
    if all(name in sys.modules for name in _TORCHAO_INTMM_MODULES):
        return patched_now  # nothing left for a finder to catch
    for finder in sys.meta_path:
        if getattr(finder, _TORCHAO_INTMM_SENTINEL, False):
            return patched_now
    sys.meta_path.insert(0, _TorchaoIntmmPatchFinder())
    return True
