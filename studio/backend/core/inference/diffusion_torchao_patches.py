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
import importlib
import importlib.abc
import importlib.util
import inspect
import logging
import os
import re
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
    # Every diffusion, video and diffusion-training entry point calls this, so it also carries the
    # peft LoRA guard, which is independent of the int_mm switch.
    try:
        install_peft_torchao_dispatch_guard()
    except Exception:
        pass
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


# Studio's copy of ``unsloth/import_fixes.py::fix_peft_torchao_missing_tensor_subclass`` (#11168), for the
# same reason as the int_mm copy above: the diffusion process never runs ``unsloth/__init__``, so the
# original never reaches ``pipe.load_lora_weights``. peft <= 0.18 imports LinearActivationQuantizedTensor
# inside ``dispatch_torchao`` for every LoRA target, torchao 0.18 deleted it, and Studio installs torchao
# 0.18 on torch >= 2.12, so every diffusion LoRA load raised. Both copies mark their wrapper
# ``__unsloth_patched__``, so whichever runs second leaves the first in place.
# The spellings a torchao removal produces (class, its old module, the whole ``torchao.dtypes`` package on main).
# Only these, so a BROKEN torchao still raises.
_PEFT_TORCHAO_MISSING_TENSOR_SUBCLASS = re.compile(
    r"linear_?activation_?quantized_?tensor|affine_?quantized_?tensor"
    r"|no module named '?torchao\.dtypes'?(?![.\w])",
    re.IGNORECASE | re.DOTALL,
)


# The two imports `dispatch_torchao` performs, in upstream's order.
_PEFT_TORCHAO_TENSOR_SUBCLASSES = (
    ("torchao.dtypes", "AffineQuantizedTensor"),
    ("torchao.quantization", "LinearActivationQuantizedTensor"),
)


def _peft_torchao_tensor_subclasses():
    """``(classes, missing)`` for the two subclasses `dispatch_torchao` checks: a tuple usable as
    the second argument of ``isinstance``, plus the names that are gone. Only a failure naming one
    of those two counts as gone; anything else is a broken install and is re-raised."""
    classes = []
    missing = []
    for module_name, class_name in _PEFT_TORCHAO_TENSOR_SUBCLASSES:
        try:
            classes.append(getattr(importlib.import_module(module_name), class_name))
        except (ImportError, AttributeError) as exc:
            if _PEFT_TORCHAO_MISSING_TENSOR_SUBCLASS.search(str(exc)) is None:
                raise
            missing.append(class_name)
    return tuple(classes), tuple(missing)


def _guard_peft_torchao_dispatcher(original):
    """Wrap one `dispatch_torchao` so a removed tensor subclass costs only that class.

    The wrapper calls through, so a torchao shipping both classes behaves exactly as upstream.
    Only on a failure naming one of the two does it redo upstream's work against the classes that
    remain, so an AffineQuantizedTensor weight still gets a TorchaoLoraLinear.
    """
    warned = [False]
    # The defining module's own globals, so the degraded path uses the very objects upstream would
    # have, including an is_torchao_available already patched by the sibling fix.
    namespace = getattr(original, "__globals__", None)
    if not isinstance(namespace, dict):
        namespace = {}

    def _upstream(name, module_name):
        value = namespace.get(name, None)
        if value is not None:
            return value
        return getattr(importlib.import_module(module_name), name, None)

    def _redo_dispatch(classes, args, kwargs):
        """Upstream's body, with `classes` standing in for the two-class isinstance tuple.

        Arguments are read by POSITION because peft 0.19 renamed the third parameter from
        `lora_config` to `config`, and forwarded as `target, adapter_name, **kwargs`, which is
        exactly how peft <= 0.18 builds TorchaoLoraLinear.
        """
        try:
            signature = inspect.signature(original)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
        except (TypeError, ValueError):
            return None
        positional = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        layer_kwargs = {}
        for name, parameter in signature.parameters.items():
            if parameter.kind is parameter.VAR_KEYWORD:
                layer_kwargs = dict(arguments.get(name, None) or {})
        if len(positional) < 2:
            return None
        target = arguments.get(positional[0], None)
        adapter_name = arguments.get(positional[1], None)
        if target is None or adapter_name is None:
            return None

        base_tuner_layer = _upstream("BaseTunerLayer", "peft.tuners.tuners_utils")
        if base_tuner_layer is not None and isinstance(target, base_tuner_layer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target
        if not hasattr(target_base_layer, "weight"):
            return None
        is_torchao_available = _upstream("is_torchao_available", "peft.import_utils")
        if is_torchao_available is not None and not is_torchao_available():
            return None
        if not classes or not isinstance(target_base_layer.weight, classes):
            return None  # upstream's answer for an unrecognised weight: the loop moves on
        torchao_lora_linear = _upstream("TorchaoLoraLinear", "peft.tuners.lora.torchao")
        if torchao_lora_linear is None:
            return None
        return torchao_lora_linear(target, adapter_name, **layer_kwargs)

    @functools.wraps(original)
    def dispatch_torchao(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ImportError as exc:
            message = str(exc)
            if _PEFT_TORCHAO_MISSING_TENSOR_SUBCLASS.search(message) is None:
                raise
            classes, missing = _peft_torchao_tensor_subclasses()
            if not missing:
                # The message named one of the two classes but both are there, so this came from
                # somewhere else in the dispatcher and is a real failure. Redoing the dispatch
                # would swallow it and re-run whatever construction already happened.
                raise
        if not warned[0]:
            warned[0] = True
            kept = ", ".join(cls.__name__ for cls in classes) or "none of them"
            logger.warning(
                f"Unsloth Studio: This torchao no longer ships "
                f"{' and '.join(missing) or 'a tensor subclass'}, which peft's "
                f"torchao LoRA dispatcher imports to run one isinstance check "
                f"({message}), so that dispatcher raised for every LoRA layer. "
                f"Studio now runs the same check against the subclasses this "
                f"torchao does ship ({kept}), so weights of those types still get "
                f"peft's torchao LoRA layer and everything else gets an ordinary "
                f"one. Only weights of the removed subclass cannot be matched; "
                f"install `torchao<0.18` if you need those."
            )
        return _redo_dispatch(classes, args, kwargs)

    dispatch_torchao.__unsloth_patched__ = True
    return dispatch_torchao


_PEFT_LORA_DISPATCH_MODULE = "peft.tuners.lora.model"
_PEFT_TORCHAO_GUARD_SENTINEL = "__unsloth_studio_peft_torchao_guard__"


def _patch_peft_torchao_dispatchers():
    """Wrap every loaded copy of ``dispatch_torchao``: ``peft.tuners.lora.model`` binds it by value,
    so patching the defining module alone leaves the dispatch list raising. True when anything changed."""
    wrapped = []
    patched = False
    for mod_name, mod in tuple(sys.modules.items()):
        if not mod_name.startswith("peft") or mod is None:
            continue
        original = getattr(mod, "dispatch_torchao", None)
        if original is None or not callable(original):
            continue
        if getattr(original, "__unsloth_patched__", False):
            continue
        replacement = None
        for seen, wrapper in wrapped:
            if seen is original:
                replacement = wrapper
                break
        if replacement is None:
            try:
                replacement = _guard_peft_torchao_dispatcher(original)
            except Exception:
                continue
            wrapped.append((original, replacement))
        try:
            setattr(mod, "dispatch_torchao", replacement)
            patched = True
        except Exception:
            pass
    return patched


class _PeftLoraDispatchLoader(importlib.abc.Loader):
    """The real loader, plus the guard once ``peft.tuners.lora.model`` (and so ``.torchao``) has run."""

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
            _patch_peft_torchao_dispatchers()
        except Exception:
            pass

    def __getattr__(self, attribute):
        return getattr(self._loader, attribute)


class _PeftTorchaoGuardFinder(importlib.abc.MetaPathFinder):
    """Front of sys.meta_path, like the int_mm finder, so importing peft stays lazy."""

    __slots__ = (_PEFT_TORCHAO_GUARD_SENTINEL, "_finding")

    def __init__(self):
        setattr(self, _PEFT_TORCHAO_GUARD_SENTINEL, True)
        self._finding = False

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname != _PEFT_LORA_DISPATCH_MODULE or self._finding:
            return None
        self._finding = True
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            return None
        finally:
            self._finding = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        try:
            spec.loader = _PeftLoraDispatchLoader(spec.loader)
        except Exception:
            return None
        return spec


def install_peft_torchao_dispatch_guard():
    """Guard peft's ``dispatch_torchao`` now if peft's LoRA module is loaded, else when it is. Idempotent.
    True when patched or the finder was installed, False when there is nothing to do, None without peft."""
    try:
        if importlib.util.find_spec("peft") is None:
            return None
    except Exception:
        return None
    if _PEFT_LORA_DISPATCH_MODULE in sys.modules:
        try:
            return _patch_peft_torchao_dispatchers()
        except Exception:
            return False
    for finder in sys.meta_path:
        if getattr(finder, _PEFT_TORCHAO_GUARD_SENTINEL, False):
            return False
    sys.meta_path.insert(0, _PeftTorchaoGuardFinder())
    return True
