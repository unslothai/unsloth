# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read the hosted INT8 ``.pt`` pre-quant checkpoints on a torchao that no longer ships their classes.

Every published INT8 pickle except Qwen-Image-2.1's names torchao's v1 int8 stack:
``LinearActivationQuantizedTensor`` over an ``AffineQuantizedTensor`` with a ``PlainAQTTensorImpl``
and ``PlainLayout``, activations quantized by ``_int8_symm_per_token_reduced_range_quant``.
torchao 0.18 deleted all five, so the constructor allowlist has nothing to point those names at,
the load is refused, and Studio falls back to the dense bf16 transformer (66 GB for MiniMax-H3).

The data inside is plain: an int8 ``[N, K]`` matrix and one scale per output row. So the pickle is
still read with ``weights_only = True``, but with Unsloth-owned stand-ins registered under the five
deleted names. The stand-ins only hold attributes (their ``__torch_dispatch__`` raises), nothing in
the file is ever called, and each weight is checked to be exactly symmetric per-row int8 before it
is rebuilt as the installed torchao's ``Int8Tensor`` with per-row dynamic activation quant, i.e.
what ``Int8DynamicActivationInt8WeightConfig`` itself produces on that release. Weights are
unchanged bit for bit; activations are quantized the way that torchao does it now.

Only engaged when the classes are really gone. A torchao that still ships them keeps loading the
pickle as before.
"""

from __future__ import annotations

import enum
import threading
from typing import Any, Optional

_LAQT = "torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor"
_AQT = "torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor"
_IMPL = "torchao.dtypes.uintx.plain_layout.PlainAQTTensorImpl"
_LAYOUT = "torchao.dtypes.utils.PlainLayout"
_ACT_QUANT = "torchao.quantization.quant_api._int8_symm_per_token_reduced_range_quant"
_ZERO_POINT_DOMAIN = "torchao.quantization.quant_primitives.ZeroPointDomain"

# The classes whose absence this module exists for. All or nothing: a torchao shipping some of
# them would hand back a mix of real and stand-in objects, which nothing here is written against.
LEGACY_INT8_CLASS_NAMES = (_LAQT, _AQT, _IMPL, _LAYOUT)


def _resolve(name: str) -> Any:
    import importlib
    module, _, attr = name.rpartition(".")
    try:
        return getattr(importlib.import_module(module), attr)
    except Exception:  # noqa: BLE001 - absent module or attribute both mean "not shipped"
        return None


def legacy_int8_classes_missing() -> bool:
    """True when torchao ships NONE of the v1 int8 classes (0.18 and later)."""
    return all(_resolve(n) is None for n in LEGACY_INT8_CLASS_NAMES)


def names_legacy_int8_class(exc: BaseException) -> bool:
    """Whether a refused ``weights_only`` load was refused for one of the deleted v1 int8 names."""
    text = str(exc)
    return any(name in text for name in (*LEGACY_INT8_CLASS_NAMES, _ACT_QUANT))


def _int8_tensor_api() -> Optional[tuple]:
    """``(Int8Tensor, QuantizeTensorToInt8Kwargs, PerRow)`` or None."""
    try:
        from core._torchao_stub import is_stubbed
        if is_stubbed("torchao"):
            return None
    except Exception:  # noqa: BLE001 - no stub module means nothing is stubbed
        pass
    try:
        from torchao.quantization import PerRow
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
            Int8Tensor,
            QuantizeTensorToInt8Kwargs,
        )
    except Exception:  # noqa: BLE001
        return None
    return Int8Tensor, QuantizeTensorToInt8Kwargs, PerRow


_SUPPORTED: Optional[bool] = None


def legacy_int8_decode_supported() -> bool:
    """Whether this install needs AND can use the rebuild below. Memoised."""
    global _SUPPORTED
    if _SUPPORTED is not None:
        return _SUPPORTED
    ok = False
    try:
        import torch

        parts = str(torch.__version__).split("+")[0].split(".")
        torch_ok = (int(parts[0]), int(parts[1])) >= (2, 6)
        api = _int8_tensor_api()
        if torch_ok and api is not None and legacy_int8_classes_missing():
            # Build one tiny weight the way the rebuild does, so a release whose constructor moved
            # answers no here rather than after a plan has dropped the dense shards.
            _to_int8_tensor(
                torch.zeros(2, 4, dtype = torch.int8),
                torch.ones(2, dtype = torch.bfloat16),
                torch.bfloat16,
                api,
            )
            ok = True
    except Exception:  # noqa: BLE001
        ok = False
    _SUPPORTED = ok
    return ok


_STANDINS: Optional[tuple] = None
_STANDINS_LOCK = threading.Lock()


def _standins() -> dict:
    """The stand-in objects, keyed by the pickled name they replace. Built lazily (importing this
    module needs no torch) and per torch module, since tests swap ``sys.modules["torch"]``."""
    global _STANDINS
    import torch

    cached = _STANDINS
    if cached is not None and cached[0] is torch:
        return cached[1]
    with _STANDINS_LOCK:
        if _STANDINS is not None and _STANDINS[0] is torch:
            return _STANDINS[1]

        class _Inert(torch.Tensor):
            # _rebuild_wrapper_subclass needs a dispatch hook to exist; any op reaching one is a bug.
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(
                cls,
                func,
                types,
                args = (),
                kwargs = None,
            ):
                raise RuntimeError(f"legacy int8 stand-in cannot run {func}")

        class LegacyLinearActivationQuantized(_Inert):
            pass

        class LegacyAffineQuantized(_Inert):
            pass

        class LegacyPlainImpl(_Inert):
            pass

        class LegacyPlainLayout:
            pass

        def legacy_int8_act_quant(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("legacy int8 activation quant stand-in is never called")

        class LegacyZeroPointDomain(enum.Enum):
            INT = 1
            FLOAT = 2
            NONE = 3

        standins = {
            _LAQT: LegacyLinearActivationQuantized,
            _AQT: LegacyAffineQuantized,
            _IMPL: LegacyPlainImpl,
            _LAYOUT: LegacyPlainLayout,
            _ACT_QUANT: legacy_int8_act_quant,
            _ZERO_POINT_DOMAIN: LegacyZeroPointDomain,
        }
        _STANDINS = (torch, standins)
        return standins


def _to_int8_tensor(qdata: Any, scale: Any, dtype: Any, api: tuple) -> Any:
    Int8Tensor, QuantizeTensorToInt8Kwargs, PerRow = api
    n, k = qdata.shape
    # Scale kept in its stored dtype: the dequantized weight is then exactly what the v1 tensor held.
    return Int8Tensor(
        qdata,
        scale.reshape(n, 1),
        [1, k],
        dtype,
        act_quant_kwargs = QuantizeTensorToInt8Kwargs(granularity = PerRow()),
    )


def _rebuild_weight(name: str, w: Any, standins: dict, api: tuple) -> Any:
    """One stand-in ``LinearActivationQuantizedTensor`` -> ``Int8Tensor``, or raise on anything else."""
    import torch

    def bad(why: str) -> ValueError:
        return ValueError(f"legacy int8 weight {name!r}: {why}")

    state = getattr(w, "__dict__", {})
    if state.get("input_quant_func") is not standins[_ACT_QUANT]:
        raise bad("activation quant is not the per-token int8 one")
    if state.get("quant_kwargs"):
        raise bad("unexpected activation quant kwargs")
    aqt = state.get("original_weight_tensor")
    if not isinstance(aqt, standins[_AQT]):
        raise bad("inner weight is not an AffineQuantizedTensor")
    impl = getattr(aqt, "tensor_impl", None)
    if not isinstance(impl, standins[_IMPL]):
        raise bad("inner storage is not the plain layout")
    if not isinstance(getattr(impl, "_layout", None), standins[_LAYOUT]):
        raise bad("inner layout is not PlainLayout")
    qdata = getattr(impl, "int_data", None)
    scale = getattr(impl, "scale", None)
    if type(qdata) is not torch.Tensor or qdata.dtype != torch.int8 or qdata.dim() != 2:
        raise bad("int data is not a 2-D int8 tensor")
    n, k = qdata.shape
    if tuple(w.shape) != (n, k) or tuple(aqt.shape) != (n, k):
        raise bad(f"shape mismatch {tuple(w.shape)} vs data {(n, k)}")
    if tuple(getattr(aqt, "block_size", ()) or ()) != (1, k):
        raise bad(f"block size {getattr(aqt, 'block_size', None)} is not per-row")
    if type(scale) is not torch.Tensor or not scale.is_floating_point() or scale.numel() != n:
        raise bad("scale is not one float per output row")
    zero_point = getattr(impl, "zero_point", None)
    if zero_point is not None:
        if type(zero_point) is not torch.Tensor or zero_point.device.type == "meta":
            raise bad("zero point present")
        if bool(torch.any(zero_point != 0)):
            raise bad("asymmetric (non-zero zero point)")
    domain = getattr(aqt, "zero_point_domain", None)
    if getattr(domain, "name", None) not in ("NONE", "INT"):
        raise bad(f"zero point domain {domain!r}")
    return _to_int8_tensor(qdata, scale, w.dtype, api)


_LOAD_LOCK = threading.Lock()


def load_legacy_int8_pickle(path: str, **kwargs: Any) -> Any:
    """``torch.load(path, weights_only = True)`` with the stand-ins, legacy weights rebuilt.

    The stand-ins are registered only for the duration of the load and only under names this
    torchao does not ship, so no other load in the process can resolve them. The lock keeps two
    overlapping reads from removing each other's registration."""
    import torch

    api = _int8_tensor_api()
    if api is None:
        raise RuntimeError("this torchao has no Int8Tensor to rebuild legacy int8 weights into")
    standins = _standins()
    pairs = [(obj, name) for name, obj in standins.items() if _resolve(name) is None]
    with _LOAD_LOCK:
        with torch.serialization.safe_globals(pairs):
            ckpt = torch.load(path, weights_only = True, **kwargs)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if not isinstance(state_dict, dict):
        return ckpt
    legacy = standins[_LAQT]
    stray = (standins[_AQT], standins[_IMPL])
    for key in list(state_dict.keys()):
        value = state_dict[key]
        if isinstance(value, legacy):
            state_dict[key] = _rebuild_weight(key, value, standins, api)
        elif isinstance(value, stray):
            raise ValueError(f"legacy int8 weight {key!r} is not wrapped for activation quant")
    return ckpt
