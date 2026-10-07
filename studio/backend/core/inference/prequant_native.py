# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rebuild a safetensors pre-quant checkpoint's int8 / fp8 weights without torchao's deserializer.

The file layout is torchao's own (``flatten_tensor_state_dict``): ``<fqn>._weight_qdata`` and
``<fqn>._weight_scale`` plus one JSON entry per weight. Reading it back through torchao's
``unflatten_tensor_state_dict`` ties every load to that release's constructor kwargs, and that has
already broken once (0.18 wrote ``reduce_range``, which 0.17 refuses). The two schemes Unsloth ships
are simple enough to describe without torchao:

    int8  w[n, k] = qdata[n, k] (int8) * scale[n]          symmetric, per output row
          activations: dynamic symmetric per-token int8
    fp8   w[n, k] = qdata[n, k] (float8_e4m3fn) * scale[n]  per output row (or one scale per tensor)
          activations: dynamic fp8 with the recorded granularity

so this module parses the JSON itself, validates the tensors against that description, and builds
whatever class the INSTALLED torchao uses for the scheme. In particular an int8 artifact converted from
a torchao v1 checkpoint (``unsloth_quant_layout.int8_source == "torchao_v1"``) is rebuilt as the v1
``LinearActivationQuantizedTensor`` where torchao still ships it (<= 0.17), which makes it
bit-identical, class for class, to reading the original ``.pt`` there; 0.18 and later get the
``Int8Tensor`` the legacy-pickle decoder produces. Anything this module does not recognise
returns None and the caller uses torchao's own reader, so nvfp4 / mxfp8 artifacts are unaffected.

Kill switch: ``UNSLOTH_PREQUANT_NATIVE_REBUILD=0`` always uses torchao's reader.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Optional

NATIVE_REBUILD_ENV = "UNSLOTH_PREQUANT_NATIVE_REBUILD"

# Namespaced unsloth_* so torchao's is_metadata_torchao and released Studio builds ignore them.
QUANT_LAYOUT_KEY = "unsloth_quant_layout"
SOURCE_KEY = "unsloth_source"
QUANT_LAYOUT_VERSION = 1
INT8_SOURCE_V1 = "torchao_v1"

# Must match what prequant_legacy_int8._rebuild_weight validates; anything else is not rebuilt as v1.
INT8_V1_FACTS = {
    "act_quant": "_int8_symm_per_token_reduced_range_quant",
    "quant_kwargs": {},
    "zero_point": None,
    "zero_point_domain": "NONE",
    "quant_min": None,
    "quant_max": None,
    "scale_shape": "out",
    "layout": "PlainLayout",
}

# Same rule as prequant_safetensors._INERT_FIELD_VALUES.
_INERT = (False, None, 0)

_DTYPES = ("bfloat16", "float16", "float32", "float8_e4m3fn", "float8_e5m2", "int8")


class _Unsupported(Exception):
    """This entry needs torchao's own reader; not an error."""


def native_rebuild_enabled() -> bool:
    raw = (os.environ.get(NATIVE_REBUILD_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _torchao_api() -> Optional[dict]:
    """The torchao names the rebuild needs, or None (absent / stubbed torchao)."""
    try:
        from core._torchao_stub import is_stubbed
        if is_stubbed("torchao"):
            return None
    except Exception:  # noqa: BLE001 - no stub module means nothing is stubbed
        pass
    try:
        from torchao.float8.inference import Float8MMConfig
        from torchao.quantization import Float8Tensor, PerRow, PerTensor
        from torchao.quantization.quant_primitives import MappingType
        from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            QuantizeTensorToFloat8Kwargs,
        )
    except Exception:  # noqa: BLE001
        return None
    api = {
        "Float8Tensor": Float8Tensor,
        "Float8MMConfig": Float8MMConfig,
        "QuantizeTensorToFloat8Kwargs": QuantizeTensorToFloat8Kwargs,
        "KernelPreference": KernelPreference,
        "MappingType": MappingType,
        "PerRow": PerRow,
        "PerTensor": PerTensor,
    }
    try:
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
            Int8Tensor,
            QuantizeTensorToInt8Kwargs,
        )
        api["Int8Tensor"] = Int8Tensor
        api["QuantizeTensorToInt8Kwargs"] = QuantizeTensorToInt8Kwargs
    except Exception:  # noqa: BLE001 - no Int8Tensor: int8 entries go to torchao's reader
        pass
    return api


def _v1_int8_api() -> Optional[dict]:
    """torchao's v1 int8 classes when this release still ships ALL of them (<= 0.17), else None."""
    try:
        from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
        from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl
        from torchao.dtypes.utils import PlainLayout
        from torchao.quantization.linear_activation_quantized_tensor import (
            LinearActivationQuantizedTensor,
        )
        from torchao.quantization.quant_api import _int8_symm_per_token_reduced_range_quant
        from torchao.quantization.quant_primitives import ZeroPointDomain
    except Exception:  # noqa: BLE001
        return None
    return {
        "AffineQuantizedTensor": AffineQuantizedTensor,
        "PlainAQTTensorImpl": PlainAQTTensorImpl,
        "PlainLayout": PlainLayout,
        "LinearActivationQuantizedTensor": LinearActivationQuantizedTensor,
        "act_quant": _int8_symm_per_token_reduced_range_quant,
        "ZeroPointDomain": ZeroPointDomain,
    }


def _construct(cls: Any, data: dict, what: str) -> Any:
    """``cls(**data)`` keeping only fields ``cls`` has; an unknown field must be inert to be dropped."""
    if dataclasses.is_dataclass(cls):
        known = {f.name for f in dataclasses.fields(cls)}
    elif hasattr(cls, "_fields"):
        known = set(cls._fields)
    else:
        raise _Unsupported(f"{what}: {cls!r} is neither a dataclass nor a NamedTuple")
    extra = {k: v for k, v in data.items() if k not in known}
    live = {k: v for k, v in extra.items() if v not in _INERT}
    if live:
        name = sorted(live)[0]
        raise ValueError(
            f"{what} records {name}={live[name]!r}, which this torchao cannot construct; "
            "upgrade torchao to read this checkpoint"
        )
    return cls(**{k: v for k, v in data.items() if k in known})


def _decode(value: Any, api: dict, what: str) -> Any:
    """torchao's JSON encoding of one attribute, decoded against a FIXED table of types."""
    if isinstance(value, list):
        return [_decode(v, api, what) for v in value]
    if not isinstance(value, dict) or "_type" not in value:
        return value
    kind, data = value.get("_type"), value.get("_data")
    if kind == "torch.dtype":
        import torch
        if data not in _DTYPES:
            raise _Unsupported(f"{what}: dtype {data!r}")
        return getattr(torch, data)
    if kind in ("MappingType", "KernelPreference"):
        enum_cls = api[kind]
        if not isinstance(data, str) or not hasattr(enum_cls, data):
            raise _Unsupported(f"{what}: {kind} {data!r}")
        return getattr(enum_cls, data)
    if kind in ("PerRow", "PerTensor") and data is not None and not isinstance(data, dict):
        raise _Unsupported(f"{what}: {kind} {data!r}")
    if kind in ("PerRow", "PerTensor") and set(data or {}) - (
        {"dim"} if kind == "PerRow" else set()
    ):
        raise _Unsupported(f"{what}: {kind} fields {sorted(set(data) - {'dim'})}")
    if kind == "PerRow":
        dim = (data or {}).get("dim", -1)
        try:
            return api["PerRow"](dim = dim)
        except TypeError:
            if dim != -1:
                raise _Unsupported(f"{what}: PerRow(dim={dim}) on a torchao without dim") from None
            return api["PerRow"]()
    if kind == "PerTensor":
        return api["PerTensor"]()
    if kind in ("Float8MMConfig", "QuantizeTensorToFloat8Kwargs", "QuantizeTensorToInt8Kwargs"):
        if kind not in api or not isinstance(data, dict):
            raise _Unsupported(f"{what}: {kind}")
        fields = {k: _decode(v, api, what) for k, v in data.items()}
        return _construct(api[kind], fields, f"{what} {kind}")
    raise _Unsupported(f"{what}: attribute type {kind!r}")


def _flat_key(name: str, data_name: str) -> str:
    module_fqn, weight_name = name.rsplit(".", 1)
    return f"{module_fqn}._{weight_name}_{data_name}"


def _granularity_name(g: Any) -> str:
    return type(g).__name__


def _rebuild_int8(
    name: str, entry: dict, tensors: dict, used: set, api: dict, v1: Optional[dict]
) -> Any:
    import torch

    data = entry.get("_data") or {}
    data_names = list(entry.get("_tensor_data_names") or [])
    if set(data_names) - {"qdata", "scale", "zero_point"} or not {"qdata", "scale"} <= set(
        data_names
    ):
        raise _Unsupported(f"{name}: int8 tensor data {data_names}")
    keys = {d: _flat_key(name, d) for d in data_names}
    missing = [k for k in keys.values() if k not in tensors]
    if missing:
        raise ValueError(
            f"int8 weight {name!r}: missing {missing[0]!r}; the checkpoint is incomplete"
        )
    qdata, scale = tensors[keys["qdata"]], tensors[keys["scale"]]
    zero_point = tensors.get(keys["zero_point"]) if "zero_point" in keys else None
    # Only Unsloth's layout (2-D, per-row scale); other int8 weights go to torchao's reader.
    if qdata.dtype != torch.int8 or qdata.dim() != 2:
        raise _Unsupported(f"{name}: int8 qdata {qdata.dtype} {tuple(qdata.shape)}")
    n, k = qdata.shape
    if not scale.is_floating_point() or scale.numel() != n:
        raise _Unsupported(f"{name}: int8 scale {tuple(scale.shape)}")
    if list(data.get("block_size") or []) != [1, k]:
        raise _Unsupported(f"{name}: int8 block size {data.get('block_size')}")
    known = {"block_size", "dtype", "act_quant_kwargs", "reduce_range"}
    if any(v not in _INERT for k2, v in data.items() if k2 not in known):
        raise _Unsupported(f"{name}: int8 attributes {sorted(set(data) - known)}")
    if zero_point is not None and bool(torch.any(zero_point != 0)):
        raise _Unsupported(f"{name}: asymmetric int8")
    dtype = _decode(data.get("dtype"), api, name)
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise _Unsupported(f"{name}: int8 output dtype {dtype}")
    act = data.get("act_quant_kwargs")
    act_fields = ((act or {}).get("_data") or {}) if act is not None else None
    reduce_range = bool(data.get("reduce_range") or False)
    if act_fields is not None:
        gran = act_fields.get("granularity") or {}
        mapping = act_fields.get("mapping_type") or {"_type": "MappingType", "_data": "SYMMETRIC"}
        if gran.get("_type") != "PerRow" or (gran.get("_data") or {}).get("dim", -1) != -1:
            raise _Unsupported(f"{name}: int8 activation granularity {gran}")
        if mapping.get("_data") != "SYMMETRIC":
            raise _Unsupported(f"{name}: int8 activation mapping {mapping}")
        reduce_range = reduce_range or bool(act_fields.get("reduce_range") or False)
    for d in keys.values():
        used.add(d)

    if v1 is not None and act_fields is not None and not reduce_range:
        impl = v1["PlainAQTTensorImpl"](qdata, scale.reshape(n), None, v1["PlainLayout"]())
        aqt = v1["AffineQuantizedTensor"](
            impl,
            (1, k),
            torch.Size((n, k)),
            quant_min = None,
            quant_max = None,
            zero_point_domain = v1["ZeroPointDomain"].NONE,
            dtype = dtype,
        )
        return v1["LinearActivationQuantizedTensor"](aqt, v1["act_quant"], {})

    if "Int8Tensor" not in api:
        raise _Unsupported(f"{name}: this torchao has no Int8Tensor")
    import inspect

    params = inspect.signature(api["Int8Tensor"].__new__).parameters
    kwargs: dict = {}
    if act_fields is not None:
        kwargs["act_quant_kwargs"] = _decode(act, api, name)
    if "reduce_range" in params:
        kwargs["reduce_range"] = bool(data.get("reduce_range") or False)
    elif data.get("reduce_range"):
        raise ValueError(
            f"int8 weight {name!r} records reduce_range=True, which this torchao cannot construct"
        )
    if zero_point is not None and "zero_point" in params:
        kwargs["zero_point"] = zero_point
    return api["Int8Tensor"](qdata, scale, [1, k], dtype, **kwargs)


def _rebuild_fp8(name: str, entry: dict, tensors: dict, used: set, api: dict) -> Any:
    import torch

    data = entry.get("_data") or {}
    data_names = list(entry.get("_tensor_data_names") or [])
    if set(data_names) != {"qdata", "scale"}:
        raise _Unsupported(f"{name}: fp8 tensor data {data_names}")
    keys = {d: _flat_key(name, d) for d in data_names}
    missing = [k for k in keys.values() if k not in tensors]
    if missing:
        raise ValueError(
            f"fp8 weight {name!r}: missing {missing[0]!r}; the checkpoint is incomplete"
        )
    qdata, scale = tensors[keys["qdata"]], tensors[keys["scale"]]
    if qdata.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) or qdata.dim() != 2:
        raise _Unsupported(f"{name}: fp8 qdata {qdata.dtype} {tuple(qdata.shape)}")
    n, k = qdata.shape
    block = list(data.get("block_size") or [])
    if not scale.is_floating_point() or not (
        (block == [1, k] and scale.numel() == n) or (block == [n, k] and scale.numel() == 1)
    ):
        raise _Unsupported(f"{name}: fp8 block size {block} with scale {tuple(scale.shape)}")
    known = {"block_size", "mm_config", "act_quant_kwargs", "kernel_preference", "dtype"}
    extra = {k2: v for k2, v in data.items() if k2 not in known}
    if any(v not in _INERT for v in extra.values()):
        raise _Unsupported(f"{name}: fp8 attributes {sorted(extra)}")
    dtype = _decode(data.get("dtype"), api, name)
    mm_config = _decode(data.get("mm_config"), api, name)
    act = _decode(data.get("act_quant_kwargs"), api, name)
    if act is not None and _granularity_name(getattr(act, "granularity", None)) not in (
        "PerRow",
        "PerTensor",
    ):
        raise _Unsupported(f"{name}: fp8 activation granularity")
    pref = _decode(data.get("kernel_preference"), api, name)
    for d in keys.values():
        used.add(d)
    kwargs = {"block_size": block, "mm_config": mm_config, "act_quant_kwargs": act, "dtype": dtype}
    if "kernel_preference" in data:
        kwargs["kernel_preference"] = pref
    return api["Float8Tensor"](qdata, scale, **kwargs)


def read_quant_layout(raw: dict) -> dict:
    try:
        layout = json.loads(raw.get(QUANT_LAYOUT_KEY) or "{}")
    except Exception:  # noqa: BLE001 - a corrupt optional key is ignored, not fatal
        return {}
    return layout if isinstance(layout, dict) else {}


def int8_rebuilds_as_v1(raw: dict) -> bool:
    """Whether an int8 artifact converted from torchao v1 is rebuilt as v1 on THIS install."""
    layout = read_quant_layout(raw)
    if layout.get("int8_source") != INT8_SOURCE_V1:
        return False
    if (layout.get("int8_v1") or {}) != INT8_V1_FACTS:
        return False
    return _v1_int8_api() is not None


def native_unflatten(
    tensors: dict,
    raw: dict,
    *,
    path: str = "",
) -> Optional[dict]:
    """``tensors`` + ``raw`` header -> state dict, or None when torchao's reader has to do it.

    ``tensors`` is not modified when this returns None. Raises ValueError for a checkpoint that is
    recognised but incomplete (a listed tensor or a weight part missing, a tensor the header does not
    list, a live field this torchao lacks): torchao's reader would fail the same file, later and less
    clearly. A weight of a layout this module does not model returns None instead.
    """
    api = _torchao_api()
    if api is None:
        return None
    try:
        names = json.loads(raw.get("tensor_names") or "null")
    except Exception:  # noqa: BLE001
        return None
    if (
        not isinstance(names, list)
        or not all(isinstance(n, str) and n for n in names)
        or len(set(names)) != len(names)
    ):
        return None
    v1 = _v1_int8_api() if int8_rebuilds_as_v1(raw) else None
    out: dict = {}
    used: set = set()
    try:
        for name in names:
            try:
                entry = json.loads(raw.get(name) or "null")
            except Exception:  # noqa: BLE001
                raise _Unsupported(f"{name}: unreadable entry") from None
            if not isinstance(entry, dict):
                raise _Unsupported(f"{name}: no entry")
            kind = entry.get("_type")
            if kind == "Tensor":
                if name not in tensors:
                    raise ValueError(f"{path} lists {name!r} but has no such tensor")
                out[name] = tensors[name]
                used.add(name)
            elif "." not in name:
                raise _Unsupported(f"{name}: root-level subclass")
            elif kind == "Int8Tensor":
                out[name] = _rebuild_int8(name, entry, tensors, used, api, v1)
            elif kind == "Float8Tensor":
                out[name] = _rebuild_fp8(name, entry, tensors, used, api)
            else:
                raise _Unsupported(f"{name}: {kind}")
    except _Unsupported:
        return None
    leftover = sorted(set(tensors) - used)
    if leftover:
        raise ValueError(
            f"{path} has {len(leftover)} tensor(s) its header does not account for "
            f"(e.g. {leftover[0]!r}); the checkpoint is incomplete or was edited"
        )
    return out


def quant_layout_header(state_dict: Any, *, int8_source: Optional[str] = None) -> dict:
    """The ``unsloth_quant_layout`` description a converter writes for ``state_dict``."""
    schemes = set()
    for value in state_dict.values():
        name = type(value).__name__
        if name in ("Int8Tensor", "LinearActivationQuantizedTensor"):
            schemes.add("int8")
        elif name == "Float8Tensor":
            schemes.add("fp8")
    layout: dict = {"version": QUANT_LAYOUT_VERSION, "weights": {}}
    if "int8" in schemes:
        layout["weights"]["int8"] = {
            "tensors": "<fqn>._weight_qdata int8 [out, in], <fqn>._weight_scale float [out, 1]",
            "dequant": "weight = qdata * scale (symmetric, per output row, no zero point)",
            "activation": "dynamic symmetric per-token int8",
        }
        if int8_source:
            layout["int8_source"] = int8_source
            if int8_source == INT8_SOURCE_V1:
                layout["int8_v1"] = dict(INT8_V1_FACTS)
    if "fp8" in schemes:
        layout["weights"]["fp8"] = {
            "tensors": "<fqn>._weight_qdata float8_e4m3fn [out, in], <fqn>._weight_scale float32 [out, 1]",
            "dequant": "weight = qdata * scale (per output row)",
            "activation": "dynamic float8 with the recorded granularity",
        }
    return layout
