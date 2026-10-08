# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load a ComfyUI-format quantized single-file DiT, or refuse it by name.

ComfyUI keeps quantized codes under each layer's ``.weight`` with ``.weight_scale`` (some formats also
``.input_scale``, ...) beside it, the format declared per layer (``<layer>.comfy_quant`` JSON tensor), in
the header's ``_quantization_metadata`` table, or by the legacy ``scaled_fp8`` marker + ``.scale_weight``.
Stock ``from_single_file`` casts the codes as weights and drops the scales: the load succeeds and renders noise.

From the header alone: int8_tensorwise (optionally ConvRot) maps into Studio's int8 runtime with codes and
per-row scales unchanged; float8_e4m3fn into Studio's per-row fp8 runtime (scalar scale repeated per row);
nvfp4 into Studio's FlashInfer NVFP4 Linear and mxfp8 into a ``torch._scaled_mm`` block-scaled Linear
(``diffusion_comfy_block``); where none runs, or for layers Studio keeps in bf16 and float8_e5m2, layers
dequantize. Any other format is refused by name, never mis-loaded.

Row mapping: the family's converter runs once on float64 row tags ``(source key, row)``; the tags that
come out say which source rows each diffusers weight holds. A converter that splices columns, changes
the column count or transforms values leaves undecodable tags and is refused. (Tags are constant along a
row: a pure input-column reorder would go unseen; no diffusers converter does one.)
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

from .diffusion_comfy_block import BLOCK_SIZES, MXFP8, NVFP4, tiled_scale_numel

COMFY_QUANT_SUFFIX = ".comfy_quant"
QUANT_METADATA_KEY = "_quantization_metadata"
LEGACY_SCALED_FP8_KEY = "scaled_fp8"
COMFY_INT8_ENV = "UNSLOTH_DIFFUSION_COMFY_INT8"
COMFY_FP8_ENV = "UNSLOTH_DIFFUSION_COMFY_FP8"
# Header metadata our own converter writes (base model, scheme); ComfyUI ignores it.
CONVERSION_RECORD_KEY = "unsloth_comfy_conversion"

INT8_TENSORWISE = "int8_tensorwise"
FP8_E4M3 = "float8_e4m3fn"
FP8_FORMATS = {"float8_e4m3fn": "F8_E4M3", "float8_e5m2": "F8_E5M2"}

# Every companion tensor a ComfyUI quantized layer can carry. A suffix outside what a supported format
# consumes means a format this loader does not implement, so it is refused rather than dropped.
_SCALE_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".pre_quant_scale",
    ".weight_s_rel",
    ".weight_s_channel",
    ".weight_codebook",
    ".scale_weight",
    ".scale_input",
)
_CONSUMED = {
    INT8_TENSORWISE: (".weight_scale",),
    "float8_e4m3fn": (".weight_scale", ".input_scale"),
    "float8_e5m2": (".weight_scale", ".input_scale"),
    NVFP4: (".weight_scale", ".weight_scale_2", ".input_scale", ".pre_quant_scale"),
    MXFP8: (".weight_scale", ".input_scale"),
}
_LEGACY_RENAME = {".scale_weight": ".weight_scale", ".scale_input": ".input_scale"}

# Row tag: key index * 2**24 + row, exact in float64 for any real checkpoint.
_TAG = float(1 << 24)
# Tag width of the fast pass: enough columns to see a splice (first / middle / last differ), none of the real width.
_NARROW_TAG_COLUMNS = 4
_MAX_HEADER_BYTES = 256 * 1024 * 1024


@dataclass(frozen = True)
class ComfyQuantLayer:
    """One quantized layer of a ComfyUI checkpoint, named as in the file (no ``.weight``)."""

    name: str
    format: str
    convrot: bool = False
    group: int = 0


@dataclass
class ComfyQuantScan:
    """What the header declares; ``problems`` lists everything this module will not load."""

    layers: dict = field(default_factory = dict)
    legacy_scaled_fp8: bool = False
    problems: list = field(default_factory = list)

    def counts(self) -> dict:
        out: dict = {}
        for layer in self.layers.values():
            key = layer.format + ("+convrot" if layer.convrot else "")
            out[key] = out.get(key, 0) + 1
        return out


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{os.path.basename(path)} is not a safetensors file")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > _MAX_HEADER_BYTES:
            raise ValueError(f"{os.path.basename(path)} has an implausible safetensors header")
        return json.loads(handle.read(size)), 8 + size


# A comfy_quant declaration is a few hundred bytes of JSON; a file can claim any span, so never read a large one.
_MAX_COMFY_QUANT_BYTES = 1 << 20


def _read_json_tensor(path: str, entry: dict, base: int) -> Any:
    start, end = entry["data_offsets"]
    if int(start) < 0 or not 0 <= int(end) - int(start) <= _MAX_COMFY_QUANT_BYTES:
        raise ValueError(f"comfy_quant declaration spans {int(end) - int(start)} bytes")
    with open(path, "rb") as handle:
        handle.seek(base + int(start))
        return json.loads(handle.read(int(end) - int(start)).decode("utf-8"))


def _layer_conf(raw: Any) -> tuple[Optional[str], bool, int]:
    """``(format, convrot, group)`` with ComfyUI's own defaults (``params`` nesting, group 256)."""
    if not isinstance(raw, dict):
        return None, False, 0
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    fmt = raw.get("format")
    convrot = bool(raw.get("convrot", params.get("convrot", False)))
    group = raw.get("convrot_groupsize", params.get("convrot_groupsize", 256)) if convrot else 0
    return (fmt if isinstance(fmt, str) else None), convrot, group


def scan_comfy_quant(path: Optional[str]) -> Optional[ComfyQuantScan]:
    """The ComfyUI quantization a safetensors single file declares, or None for a plain checkpoint.

    Torch-free: reads the header and the few hundred bytes of ``.comfy_quant`` JSON. Never raises for
    a file it cannot read as safetensors (the regular loader reports that); returns a scan whose
    ``problems`` are non-empty for one it must refuse."""
    if not path or not str(path).lower().endswith(".safetensors") or not os.path.isfile(path):
        return None
    try:
        header, base = _read_header(str(path))
    except Exception:  # noqa: BLE001 -- not ours to diagnose; from_single_file will
        return None
    metadata = header.pop("__metadata__", None) or {}
    keys = set(header)
    declared: dict = {}
    scan = ComfyQuantScan()
    for key, entry in header.items():
        if not key.endswith(COMFY_QUANT_SUFFIX):
            continue
        name = key[: -len(COMFY_QUANT_SUFFIX)]
        try:
            declared[name] = _read_json_tensor(str(path), entry, base)
        except Exception:  # noqa: BLE001
            scan.problems.append(f"{name}: unreadable comfy_quant declaration")
    raw_meta = metadata.get(QUANT_METADATA_KEY) if isinstance(metadata, dict) else None
    if raw_meta:
        # ComfyUI lets the header's table win over per-layer tensors.
        try:
            table = json.loads(raw_meta).get("layers") or {}
            declared.update({str(k): v for k, v in table.items()})
        except Exception:  # noqa: BLE001
            scan.problems.append(f"unreadable {QUANT_METADATA_KEY} header")
    if LEGACY_SCALED_FP8_KEY in keys and not raw_meta:
        scan.legacy_scaled_fp8 = True
        marker = str(header[LEGACY_SCALED_FP8_KEY].get("dtype", ""))
        fmt = "float8_e5m2" if marker == "F8_E5M2" else "float8_e4m3fn"
        for key in keys:
            if key.endswith(".scale_weight"):
                declared.setdefault(key[: -len(".scale_weight")], {"format": fmt})

    scaled = {
        key[: -len(suffix)] for key in keys for suffix in _SCALE_SUFFIXES if key.endswith(suffix)
    }
    for name in sorted(scaled - set(declared)):
        # Undeclared scales: only an fp8 weight with a plain weight_scale has one meaning.
        weight = header.get(name + ".weight") or {}
        dtype = weight.get("dtype")
        fmt = next((f for f, d in FP8_FORMATS.items() if d == dtype), None)
        extra = [
            s
            for s in _SCALE_SUFFIXES
            if name + s in keys and s not in (".weight_scale", ".input_scale")
        ]
        if fmt is not None and name + ".weight_scale" in keys and not extra:
            declared[name] = {"format": fmt}
        else:
            scan.problems.append(
                f"{name}: scale tensors with no quantization format declared (weight {dtype})"
            )
    if not declared and not scan.problems and not scan.legacy_scaled_fp8:
        return None

    unsupported: dict = {}
    for name, raw in sorted(declared.items()):
        fmt, convrot, group = _layer_conf(raw)
        weight = header.get(name + ".weight")
        why = None
        if fmt is None:
            why = "no format"
        elif fmt not in _CONSUMED:
            unsupported[fmt] = unsupported.get(fmt, 0) + 1
            continue
        elif weight is None:
            why = f"{fmt} declared but there is no {name}.weight"
        elif len(weight.get("shape") or ()) != 2:
            why = f"{fmt} weight of shape {weight.get('shape')} (only 2-D linears are supported)"
        elif fmt in BLOCK_SIZES:
            why = (
                f"ConvRot on {fmt}" if convrot else _block_layer_problem(name, fmt, weight, header)
            )
        elif fmt == INT8_TENSORWISE and weight.get("dtype") != "I8":
            why = f"int8_tensorwise weight stored as {weight.get('dtype')}"
        elif fmt in FP8_FORMATS and weight.get("dtype") not in ("U8", FP8_FORMATS[fmt]):
            why = f"{fmt} weight stored as {weight.get('dtype')}"
        elif convrot and fmt != INT8_TENSORWISE:
            why = f"ConvRot on {fmt}"
        elif convrot and not _is_power_of_four(group):
            why = f"ConvRot group {group!r} is not a power of 4"
        elif convrot and int(weight["shape"][1]) % int(group):
            why = f"in_features {weight['shape'][1]} not divisible by ConvRot group {group}"
        else:
            scale = header.get(name + ".weight_scale") or header.get(name + ".scale_weight")
            rows = int(weight["shape"][0])
            n = 1
            for d in (scale or {}).get("shape") or ():
                n *= int(d)
            if scale is None:
                why = f"{fmt} declared but {name}.weight_scale is missing"
            elif n not in (1, rows):
                why = f"weight_scale of shape {scale.get('shape')} for {rows} rows (block scales)"
            else:
                leftover = [
                    s
                    for s in _SCALE_SUFFIXES
                    if name + s in keys and _LEGACY_RENAME.get(s, s) not in _CONSUMED[fmt]
                ]
                if leftover:
                    why = (
                        f"{fmt} layer carries {', '.join(leftover)}, which this format does not use"
                    )
        if why:
            scan.problems.append(f"{name}: {why}")
            continue
        scan.layers[name] = ComfyQuantLayer(name, fmt, convrot, int(group) if convrot else 0)
    for fmt, count in sorted(unsupported.items()):
        scan.problems.append(f"{count} layer(s) in ComfyUI format {fmt!r}")
    return scan


_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}


def comfy_resident_mib(
    path: Optional[str],
    scan: Optional[ComfyQuantScan] = None,
    *,
    keep_int8: bool,
    keep_fp8: bool,
    keep_nvfp4: bool = False,
    keep_mxfp8: bool = False,
    compute_bytes: int = 2,
    keep_key: Any = None,
    exclude_tokens: Any = (),
    min_features: int = 0,
    fp8_divisible: int = 0,
    block_divisible: Optional[dict] = None,
    key_map: Any = None,
) -> Optional[int]:
    """What the loader leaves resident for a ComfyUI-quantized file, priced from its header: a quantized
    weight a runtime keeps costs its stored bytes, one that is dequantized costs ``numel * compute_bytes``
    (2x an int8 / fp8 file), every other floating tensor is cast to the compute dtype. The file name says
    nothing reliable here (an fp8 file Studio runs natively is not upcast; an int8 one with no runtime is).
    ``keep_key(key)`` limits the count to the keys the loader reads (a file bundling other components);
    an int8 layer whose name holds one of ``exclude_tokens`` is priced dequantized, as Studio's int8 filter
    leaves it, and so is a layer the runtime filter skips (in / out features under ``min_features``, or fp8 features
    not multiples of ``fp8_divisible``). ``key_map`` names layers as the loader's filter sees them (an original-layout
    file). None when the header cannot be read. Torch-free."""
    try:
        scan = scan if scan is not None else scan_comfy_quant(path)
        if scan is None:
            return None
        header, _base = _read_header(str(path))
    except Exception:  # noqa: BLE001 -- unknown size: the caller keeps its own estimate
        return None
    header.pop("__metadata__", None)

    def _fits(shape: Any, divisible: int) -> bool:
        if len(shape or ()) != 2:
            return False
        out_f, in_f = (int(d) for d in shape)
        if min(out_f, in_f) < min_features:
            return False
        return not divisible or not (out_f % divisible or in_f % divisible)

    def _excluded(key: str, shape: Any) -> bool:
        names = [key]
        if key_map is not None:
            try:
                names = [k for k, _rows in key_map(key, tuple(shape or ()))] or names
            except ValueError:
                pass
        return any(t in n for n in names for t in exclude_tokens)

    def _logical(fmt: str, shape: Any) -> Any:
        if fmt == NVFP4 and len(shape or ()) == 2:
            return (int(shape[0]), int(shape[1]) * 2)
        return shape

    kept = {}
    for name, layer in scan.layers.items():
        shape = (header.get(name + ".weight") or {}).get("shape")
        kept[name + ".weight"] = (
            layer.format == INT8_TENSORWISE
            and keep_int8
            and not _excluded(name + ".weight", shape)
            and _fits(shape, 0)
        ) or (
            (layer.format == FP8_E4M3 and keep_fp8 and _fits(shape, fp8_divisible))
            or (
                {NVFP4: keep_nvfp4, MXFP8: keep_mxfp8}.get(layer.format, False)
                # input smoothing has no runtime Linear: the loader dequantizes such a layer
                and name + ".pre_quant_scale" not in header
                and _fits(
                    _logical(layer.format, shape), (block_divisible or {}).get(layer.format, 0)
                )
            )
        )
    packed = {name + ".weight" for name, layer in scan.layers.items() if layer.format == NVFP4}
    total = 0
    for key, entry in header.items():
        if keep_key is not None and not keep_key(key):
            continue
        shape = entry.get("shape") or ()
        numel = 1
        for d in shape:
            numel *= int(d)
        dtype = str(entry.get("dtype", ""))
        stored = numel * _DTYPE_BYTES.get(dtype, 2)
        if key in kept:
            total += stored if kept[key] else numel * compute_bytes * (2 if key in packed else 1)
        elif key.endswith(COMFY_QUANT_SUFFIX) or any(key.endswith(s) for s in _SCALE_SUFFIXES):
            total += stored
        elif dtype.startswith(("F", "BF")):
            total += numel * compute_bytes
        else:
            total += stored
    return -(-total // (1024 * 1024)) if total > 0 else None


def _numel(entry: Optional[dict]) -> int:
    n = 1
    for d in (entry or {}).get("shape") or ():
        n *= int(d)
    return n


def _block_layer_problem(name: str, fmt: str, weight: dict, header: dict) -> Optional[str]:
    """Why an ``nvfp4`` / ``mxfp8`` layer cannot be decoded, from its header entries, or None."""
    rows, stored = (int(d) for d in weight["shape"])
    if fmt == NVFP4:
        if weight.get("dtype") != "U8":
            return f"nvfp4 weight stored as {weight.get('dtype')}"
        cols = stored * 2
    else:
        if weight.get("dtype") not in ("F8_E4M3", "U8"):
            return f"mxfp8 weight stored as {weight.get('dtype')}"
        cols = stored
    block = BLOCK_SIZES[fmt]
    if cols % block:
        return f"{fmt} weight with {cols} columns (not a multiple of {block})"
    scale = header.get(name + ".weight_scale")
    want = ("F8_E4M3", "U8") if fmt == NVFP4 else ("F8_E8M0", "U8")
    if scale is None:
        return f"{fmt} declared but {name}.weight_scale is missing"
    if scale.get("dtype") not in want:
        return f"{fmt} block scales stored as {scale.get('dtype')}"
    if _numel(scale) != tiled_scale_numel(rows, cols // block):
        return f"{fmt} block scales of shape {scale.get('shape')} for a {rows}x{cols} weight (not the 128x4 tiled layout)"
    if fmt == NVFP4:
        second = header.get(name + ".weight_scale_2")
        if second is None or _numel(second) != 1:
            return f"nvfp4 needs a scalar {name}.weight_scale_2"
        smooth = header.get(name + ".pre_quant_scale")
        if smooth is not None and _numel(smooth) != cols:
            return f"pre_quant_scale of shape {smooth.get('shape')} for {cols} input features"
    entry = header.get(name + ".input_scale")
    if entry is not None and _numel(entry) != 1:
        return f"input_scale of shape {entry.get('shape')} (only a per-tensor scale is supported)"
    leftover = [
        s
        for s in _SCALE_SUFFIXES
        if name + s in header and _LEGACY_RENAME.get(s, s) not in _CONSUMED[fmt]
    ]
    if leftover:
        return f"{fmt} layer carries {', '.join(leftover)}, which this format does not use"
    return None


def _is_power_of_four(size: Any) -> bool:
    from .diffusion_convrot import is_power_of_four
    return is_power_of_four(size)


def comfy_quant_error(scan: Optional[ComfyQuantScan], filename: str = "") -> Optional[str]:
    """A user-facing refusal for ``scan``, or None when every declared layer is loadable."""
    if scan is None or not scan.problems:
        return None
    shown = "; ".join(scan.problems[:4])
    more = f" (and {len(scan.problems) - 4} more)" if len(scan.problems) > 4 else ""
    return (
        f"{filename or 'This checkpoint'} is a ComfyUI-quantized checkpoint Studio cannot run "
        f"faithfully: {shown}{more}. Supported ComfyUI formats are int8_tensorwise (with or "
        "without ConvRot), float8_e4m3fn, float8_e5m2, nvfp4 and mxfp8; use the bf16 file, a GGUF, "
        "or Studio's own int8 / fp8 transformer quantization instead."
    )


def refuse_comfy_quant(path: Optional[str]) -> Optional[ComfyQuantScan]:
    """``scan_comfy_quant`` that raises ``ValueError`` for a checkpoint it would have to refuse."""
    scan = scan_comfy_quant(path)
    problem = comfy_quant_error(scan, os.path.basename(str(path or "")))
    if problem:
        raise ValueError(problem)
    return scan


def _dequant(codes: Any, scale: Any, group: int, dtype: Any) -> Any:
    """``codes * scale`` in float32, the ConvRot rotation undone, cast to ``dtype``."""
    import torch

    rows = codes.shape[0]
    weight = codes.to(torch.float32) * scale.to(torch.float32).reshape(-1, 1).expand(rows, 1)
    if group:
        from .diffusion_convrot import build_convrot_hadamard

        # The file holds W @ blockdiag(H).T; H is symmetric and orthogonal, so W = that @ H.
        h = build_convrot_hadamard(group, device = weight.device, dtype = torch.float32)
        cols = weight.shape[1]
        weight = (weight.reshape(rows, cols // group, group) @ h).reshape(rows, cols)
    return weight.to(dtype)


@dataclass
class _BlockWeight:
    """Row segments ``(codes, plain scales, extras)`` from one or more layers."""

    fmt: str
    parts: list


def _dequant_parts(fmt: str, parts: list, dtype: Any) -> Any:
    import torch

    from .diffusion_comfy_block import dequant_block

    dense = [
        dequant_block(
            fmt,
            codes,
            scale,
            tensor_scale = extra.get("tensor_scale"),
            pre_quant_scale = extra.get("pre_quant_scale"),
            dtype = dtype,
        )
        for codes, scale, extra in parts
    ]
    return dense[0] if len(dense) == 1 else torch.cat(dense)


def _block_runtime_args(value: "_BlockWeight") -> Optional[tuple]:
    """``(codes, scales, tensor_scale, input_scale)`` when one runtime Linear holds ``value`` exactly, else None.
    Segments share an input: the largest static input_scale (none clips); any missing one = dynamic (None)."""
    import torch

    extras = [extra for _c, _s, extra in value.parts]
    codes = (
        torch.cat([c for c, _s, _e in value.parts]) if len(value.parts) > 1 else value.parts[0][0]
    )
    scale = (
        torch.cat([s for _c, s, _e in value.parts]) if len(value.parts) > 1 else value.parts[0][1]
    )
    if value.fmt == MXFP8:
        return codes, scale, None, None
    tensor_scales = {e.get("tensor_scale") for e in extras}
    input_scales = [e.get("input_scale") for e in extras]
    if len(tensor_scales) != 1 or None in tensor_scales:
        return None
    if any(e.get("pre_quant_scale") is not None for e in extras):
        return None
    return codes, scale, tensor_scales.pop(), None if None in input_scales else max(input_scales)


def _int8_tensor(name: str, codes: Any, scale: Any, dtype: Any) -> Optional[Any]:
    """The torchao int8 weight a hosted INT8 checkpoint rebuilds into, built by the same decoder."""
    import torch

    from .prequant_native import native_unflatten

    rows, cols = codes.shape
    dtype_name = str(dtype).replace("torch.", "")
    entry = {
        "_type": "Int8Tensor",
        "_data": {
            "act_quant_kwargs": {
                "_type": "QuantizeTensorToInt8Kwargs",
                "_data": {
                    "granularity": {"_type": "PerRow", "_data": {"dim": -1}},
                    "mapping_type": {"_type": "MappingType", "_data": "SYMMETRIC"},
                    "reduce_range": False,
                },
            },
            "reduce_range": False,
            "block_size": [1, cols],
            "dtype": {"_type": "torch.dtype", "_data": dtype_name},
        },
        "_tensor_data_names": ["zero_point", "qdata", "scale"],
    }
    stem = name.rsplit(".", 1)[0]
    tensors = {
        f"{stem}._weight_qdata": codes.contiguous(),
        f"{stem}._weight_scale": scale.to(torch.float32).reshape(rows, 1).contiguous(),
        f"{stem}._weight_zero_point": torch.zeros((rows, 1), dtype = torch.int8),
    }
    raw = {"tensor_names": json.dumps([name]), name: json.dumps(entry)}
    rebuilt = native_unflatten(tensors, raw)
    return None if rebuilt is None else rebuilt[name]


def _fp8_tensor(
    name: str,
    codes: Any,
    scale: Any,
    dtype: Any,
    fast_accum: Optional[bool] = None,
) -> Optional[Any]:
    """The torchao fp8 weight a hosted FP8 checkpoint rebuilds into (per-row ``Float8Tensor``, dynamic
    per-row fp8 activations with the same 1e-12 floor, the plain-torch kernel), built by the same decoder.
    A per-tensor scale is the per-row layout with every row's scale equal, so it is repeated, not changed."""
    import torch

    from .diffusion_transformer_quant import _resolve_fast_accum
    from .prequant_native import native_unflatten

    rows, cols = codes.shape
    dtype_name = str(dtype).replace("torch.", "")
    kernel = {"_type": "KernelPreference", "_data": "TORCH"}
    entry = {
        "_type": "Float8Tensor",
        "_data": {
            "block_size": [1, cols],
            "mm_config": {
                "_type": "Float8MMConfig",
                "_data": {
                    "emulate": False,
                    "use_fast_accum": bool(_resolve_fast_accum(fast_accum)),
                    "pad_inner_dim": False,
                },
            },
            "act_quant_kwargs": {
                "_type": "QuantizeTensorToFloat8Kwargs",
                "_data": {
                    "float8_dtype": {"_type": "torch.dtype", "_data": FP8_E4M3},
                    "granularity": {"_type": "PerRow", "_data": {"dim": -1}},
                    "mm_config": None,
                    "hp_value_lb": 1e-12,
                    "hp_value_ub": None,
                    "kernel_preference": kernel,
                },
            },
            "kernel_preference": kernel,
            "dtype": {"_type": "torch.dtype", "_data": dtype_name},
        },
        "_tensor_data_names": ["qdata", "scale"],
    }
    stem = name.rsplit(".", 1)[0]
    tensors = {
        f"{stem}._weight_qdata": codes.contiguous(),
        f"{stem}._weight_scale": scale.to(torch.float32)
        .reshape(-1, 1)
        .expand(rows, 1)
        .contiguous(),
    }
    raw = {"tensor_names": json.dumps([name]), name: json.dumps(entry)}
    rebuilt = native_unflatten(tensors, raw)
    return None if rebuilt is None else rebuilt[name]


def _mapping(transformer_cls: Any) -> tuple[Any, Any]:
    from diffusers.loaders import single_file_model as sfm

    name = sfm._get_single_file_loadable_mapping_class(transformer_cls)
    entry = sfm.SINGLE_FILE_LOADABLE_CLASSES.get(name or "")
    if not entry:
        # The single-file branch registers Studio's converters for classes diffusers lacks (Qwen-Image-2.1) just
        # before its own call; the hosted prequant route reaches here without passing through it.
        try:
            from .diffusion import _register_unregistered_single_file_classes
            _register_unregistered_single_file_classes()
        except Exception:  # noqa: BLE001 -- nothing more to register: refused below
            pass
        name = sfm._get_single_file_loadable_mapping_class(transformer_cls)
        entry = sfm.SINGLE_FILE_LOADABLE_CLASSES.get(name or "")
    if not entry:
        raise ValueError(f"{transformer_cls.__name__} has no single-file converter")
    return entry["checkpoint_mapping_fn"], sfm


def _decode_rows(
    name: str,
    tagged: Any,
    sources: list,
    width: Optional[int] = None,
) -> list:
    """``[(source index, first row, n rows)]`` for a converted int8 weight, or raise. ``width``: the tag width
    the weights were given (None: their real column count)."""
    import torch

    if tagged.dtype != torch.float64 or tagged.dim() != 2 or tagged.numel() == 0:
        raise ValueError(f"{name}: the converter changed an int8 weight's dtype or rank")
    col = tagged[:, 0]
    for j in (tagged.shape[1] // 2, tagged.shape[1] - 1):
        if not torch.equal(col, tagged[:, j]):
            raise ValueError(f"{name}: the converter mixed columns of an int8 weight")
    # Pure Python on one list: per-weight tensor ops each wake the intra-op pool and cost more than the decode.
    segments: list = []
    for value in col.tolist():
        if value < 0 or value != int(value):
            raise ValueError(f"{name}: the converter transformed int8 weight values")
        i, r = divmod(int(value), int(_TAG))
        if i >= len(sources) or r >= sources[i][1].shape[0]:
            raise ValueError(f"{name}: an int8 row does not decode to a source row")
        if segments and segments[-1][0] == i and segments[-1][1] + segments[-1][2] == r:
            segments[-1][2] += 1
        else:
            segments.append([i, r, 1])
    if any(
        tagged.shape[1] != (sources[i][1].shape[1] if width is None else width)
        for i, _r, _n in segments
    ):
        raise ValueError(f"{name}: the converter changed an int8 weight's column count")
    return segments


def original_layout(transformer_cls: Any, path: str) -> Optional[dict]:
    """The key map / prepare / dtype hooks for a class with no diffusers single-file converter, or None."""
    name = getattr(transformer_cls, "__name__", "")
    if name == "HunyuanVideo15Transformer3DModel":
        from .video_hv15_comfy import comfy_layout
    elif name == "MiniMaxH3Transformer3DModel":
        from .video_minimax_h3_comfy import comfy_layout
    else:
        return None
    return comfy_layout(path)


def _apply_key_map(state: dict, kept: list, key_map: Any) -> dict:
    """``state`` (file keys) renamed and row-split by ``key_map``; kept layers become ``(codes, scale, group, format)``.

    Row selections are views where they are one contiguous run, so no layer is copied except a reordered one."""
    import torch

    sources = {
        layer.name + ".weight": (codes, scale, layer.group, layer.format)
        for layer, codes, scale, *_extra in kept
    }

    def take(value: Any, rows: Any) -> Any:
        if rows is None:
            return value
        if isinstance(value, tuple):
            codes, scale, group, fmt = value
            scale = scale.reshape(-1, 1).expand(codes.shape[0], 1)
            return (take(codes, rows), take(scale, rows), group, fmt)
        if value.dtype not in (torch.int8, torch.uint8) and value.element_size() == 1:
            # float8: concatenation goes through uint8 (not implemented for float8 on every torch)
            return take(value.view(torch.uint8), rows).view(value.dtype)
        parts = [value[int(r) : int(r) + int(n)] for r, n in rows]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    converted: dict = {}
    for key in list(state):
        value = state.pop(key)
        if value is None:
            value = sources.pop(key)
        data = value[0] if isinstance(value, tuple) else value
        rows_total = data.shape[0] if data.dim() else 0
        for new_key, rows in key_map(key, tuple(data.shape)):
            if rows is not None and any(r < 0 or n <= 0 or r + n > rows_total for r, n in rows):
                raise ValueError(f"{key}: row selection {rows} outside its {rows_total} rows")
            converted[new_key] = take(value, rows)
    return converted


def comfy_torchao_quantized(model: Any) -> bool:
    """Whether a ComfyUI load left torchao-quantized weights in ``model`` (eager is several times slower)."""
    info = getattr(model, "_unsloth_comfy_quant", None) or {}
    return "torchao" in (info.get("backend"), info.get("fp8_backend"))


def comfy_int8_backend(
    target: Any,
    family: Optional[str],
    base_repo: Optional[str] = None,
    *,
    offload: bool = False,
) -> Optional[str]:
    """Which of Studio's int8 runtimes takes the int8 layers, by the rule its own int8 quant follows:
    ``"torchao"`` (``Int8Tensor``) on a resident plan, ``"native"`` (the torchao-free twin, plain
    buffers the offload hooks can move) under offload or on a host without the torchao path, None
    (dequantize to bf16) where neither runs. ``UNSLOTH_DIFFUSION_COMFY_INT8=0`` always dequantizes."""
    if (os.environ.get(COMFY_INT8_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return None
    try:
        from .diffusion_transformer_quant import (
            TQ_INT8,
            native_quant_scheme,
            select_transformer_quant_scheme,
        )
        if native_quant_scheme(target, TQ_INT8, family = family, offload = offload) == TQ_INT8:
            return "native"
        if not offload and (
            select_transformer_quant_scheme(target, TQ_INT8, family = family, base_repo = base_repo)
            == TQ_INT8
        ):
            return "torchao"
    except Exception:  # noqa: BLE001 -- an unanswerable probe takes the exact dense path
        pass
    return None


def comfy_fp8_backend(
    target: Any,
    family: Optional[str],
    base_repo: Optional[str] = None,
    *,
    offload: bool = False,
) -> Optional[str]:
    """Which of Studio's fp8 runtimes takes the ``float8_e4m3fn`` layers, by the rule its own fp8 quant
    follows: ``"torchao"`` (the per-row ``Float8Tensor`` and ``_scaled_mm``) on a resident plan whose GPU
    passes Studio's fp8 probe, ``"native"`` (the torchao-free weight-only twin) where Studio's own fp8
    quant runs natively (ROCm, the stubbed torchao), None (dequantize to bf16) where neither runs: an
    older GPU, CPU, MPS, or an offloaded plan on the torchao path (the model offload hooks cannot move a
    ``Float8Tensor``: "Attempted to set the storage of a tensor on device cuda:0 to a storage on ... cpu").
    ``UNSLOTH_DIFFUSION_COMFY_FP8=0`` always dequantizes."""
    if (os.environ.get(COMFY_FP8_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return None
    try:
        import torch

        from .diffusion_transformer_quant import (
            TQ_FP8,
            native_quant_scheme,
            select_transformer_quant_scheme,
        )

        if native_quant_scheme(target, TQ_FP8, family = family, offload = offload) == TQ_FP8:
            return "native"
        if (
            not offload
            and getattr(target, "dtype", None) is torch.bfloat16
            and select_transformer_quant_scheme(target, TQ_FP8, family = family, base_repo = base_repo)
            == TQ_FP8
        ):
            return "torchao"
    except Exception:  # noqa: BLE001 -- an unanswerable probe takes the exact dense path
        pass
    return None


def comfy_conversion_record(path: Optional[str]) -> dict:
    """The ``unsloth_comfy_conversion`` header record our converter writes (base model, scheme, family),
    or {} for any other file. Torch-free, never raises."""
    try:
        header, _base = _read_header(str(path))
        raw = (header.get("__metadata__") or {}).get(CONVERSION_RECORD_KEY)
        record = json.loads(raw) if raw else {}
        return record if isinstance(record, dict) else {}
    except Exception:  # noqa: BLE001 -- no record is not an error
        return {}


# The ComfyUI layer format each Studio prequant scheme is published in.
PREQUANT_SCHEME_FORMATS = {"int8": (INT8_TENSORWISE,), "fp8": (FP8_E4M3,)}


def comfy_prequant_scheme(scan: Optional[ComfyQuantScan]) -> Optional[str]:
    """The Studio prequant scheme a ComfyUI file's quantized layers all share, else None."""
    if scan is None or scan.problems or not scan.layers:
        return None
    formats = {layer.format for layer in scan.layers.values()}
    for scheme, allowed in PREQUANT_SCHEME_FORMATS.items():
        if formats <= set(allowed):
            return scheme
    return None


def load_comfy_prequant(
    transformer_cls: Any,
    path: str,
    *,
    scheme: str,
    base: str,
    family: Optional[str],
    dtype: Any,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    fast_accum: Optional[bool] = None,
    min_features: Optional[int] = None,
    config_subfolder: str = "transformer",
    logger: Any = None,
) -> Any:
    """A hosted (or local) ComfyUI-format prequant as the transformer Studio's own ``scheme`` checkpoint
    rebuilds into, on the CPU: int8 layers as ``Int8Tensor`` under ConvRot-rotating Linears, fp8 layers as
    per-row ``Float8Tensor``. Placement, small-M padding and the rest stay with the caller, exactly as for
    Studio's own checkpoints. Raises ``ValueError`` for a file it must refuse: a format Studio cannot run,
    layers of another scheme, or a recorded base model that is not ``base``."""
    name = os.path.basename(str(path))
    scan = scan_comfy_quant(path)
    if scan is None:
        raise ValueError(f"{name} is not a ComfyUI-quantized checkpoint")
    problem = comfy_quant_error(scan, name)
    if problem:
        raise ValueError(problem)
    found = comfy_prequant_scheme(scan)
    if found != scheme:
        raise ValueError(
            f"{name} holds {', '.join(f'{k} x{v}' for k, v in sorted(scan.counts().items()))}, "
            f"not a {scheme} checkpoint"
        )
    if not family:
        raise ValueError(f"{name}: a ComfyUI checkpoint needs the model family to pick its layers")
    record = comfy_conversion_record(path)
    recorded_base = record.get("base_model_id")
    if recorded_base and base:
        from .diffusion_prequant import _same_base_model
        if not _same_base_model(str(recorded_base), str(base)):
            raise ValueError(f"{name} was converted from {recorded_base}, not {base}")
    kwargs = {
        "config": base,
        "subfolder": config_subfolder,
        "torch_dtype": dtype,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if hf_token:
        kwargs["token"] = hf_token
    import torch

    if scheme == "fp8" and dtype is not torch.bfloat16:
        raise ValueError(f"{name}: Studio's fp8 runtime needs a bfloat16 pipeline, not {dtype}")
    model = load_comfy_quant_transformer(
        transformer_cls,
        path,
        scan,
        kwargs,
        int8_backend = "torchao" if scheme == "int8" else None,
        fp8_backend = "torchao" if scheme == "fp8" else None,
        family = family,
        fast_accum = fast_accum,
        min_features = min_features,
        finalize = False,
        logger = logger,
    )
    kept = (getattr(model, "_unsloth_comfy_quant", None) or {}).get(scheme, 0)
    if not kept:
        # never a dense model under a prequant label: the caller falls back to its own quantize path
        raise ValueError(f"{name}: no layer could be rebuilt as a {scheme} weight on this install")
    return model


def load_comfy_quant_transformer(
    transformer_cls: Any,
    path: str,
    scan: ComfyQuantScan,
    sf_kwargs: dict,
    *,
    int8_backend: Optional[str],
    fp8_backend: Optional[str] = None,
    nvfp4_backend: Optional[str] = None,
    mxfp8_backend: Optional[str] = None,
    family: Optional[str] = None,
    target: Any = None,
    fast_accum: Optional[bool] = None,
    min_features: Optional[int] = None,
    finalize: bool = True,
    logger: Any = None,
    keep_key: Any = None,
    pre_convert: Any = None,
    key_map: Any = None,
    prepare_model: Any = None,
    keep_dtype: Any = None,
) -> Any:
    """Build ``transformer_cls`` from the ComfyUI-quantized ``path``.

    ``sf_kwargs`` are the ``from_single_file`` kwargs the caller would have used (``config``,
    ``subfolder``, ``torch_dtype``, ``token``, ``cache_dir``, ``local_files_only``). With an
    ``int8_backend`` (``comfy_int8_backend``) the int8 layers Studio's own int8 filter selects keep
    their codes and scales, as torchao ``Int8Tensor`` weights under ConvRot-rotating Linears
    (``"torchao"``) or as native int8 twins (``"native"``); with an ``fp8_backend``
    (``comfy_fp8_backend``) the ``float8_e4m3fn`` layers Studio's own fp8 filter selects do the same,
    as per-row ``Float8Tensor`` weights or native fp8 twins; with an ``nvfp4_backend`` / ``mxfp8_backend``
    (``diffusion_comfy_block.comfy_block_backend``) the nvfp4 / mxfp8 layers the matching Studio filter selects
    keep their codes on Studio's FlashInfer NVFP4 Linear / the ``torch._scaled_mm`` MXFP8 Linear. Everything
    else is dequantized to the compute dtype. ``finalize`` applies the small-M padding here (the single-file path); the hosted
    prequant loader passes False and pads after placement, as for its own checkpoints.

    ``keep_key(key)`` limits the read to the DiT's keys of a file that bundles other components;
    ``pre_convert(state)`` renames keys (never values) before the family converter runs. A family with no
    diffusers single-file converter passes ``key_map(key, shape)`` (or registers it in ``original_layout``):
    ``[(diffusers key, rows)]`` per file key, ``rows`` None (the whole tensor) or ``[(first row, n rows), ...]``;
    ``prepare_model(model)`` reshapes the freshly built model before the weights load, and ``keep_dtype(key)``
    names a dtype a file tensor keeps instead of the compute dtype. Raises ``ValueError`` for a checkpoint it
    must refuse."""
    import torch
    from safetensors.torch import load_file

    from .diffusion_comfy_block import (
        build_runtime_linear,
        decode_layer,
        dequant_block,
        logical_cols,
    )
    from .diffusion_transformer_quant import (
        DEFAULT_MIN_LINEAR_FEATURES,
        TQ_FP8,
        TQ_INT8,
        TQ_MXFP8,
        TQ_NVFP4,
        divisible_for_scheme,
        exclude_tokens_for_scheme,
        make_filter_fn,
        native_int8_act,
    )

    problem = comfy_quant_error(scan, os.path.basename(path))
    if problem:
        raise ValueError(problem)
    if key_map is None:
        layout = original_layout(transformer_cls, path)
        if layout:
            key_map = layout.get("key_map")
            prepare_model = prepare_model or layout.get("prepare_model")
            keep_dtype = keep_dtype or layout.get("keep_dtype")
    kwargs = dict(sf_kwargs)
    dtype = kwargs.pop("torch_dtype", None) or kwargs.pop("dtype", None) or torch.bfloat16
    if dtype is not torch.bfloat16:
        # Studio's fp8 runtime asserts bf16 weights; an fp16 / fp32 pipeline dequantizes them instead.
        fp8_backend = None
        mxfp8_backend = None
    if keep_key is None:
        state = load_file(str(path))
    else:
        from safetensors import safe_open

        with safe_open(str(path), framework = "pt", device = "cpu") as handle:
            state = {k: handle.get_tensor(k) for k in handle.keys() if keep_key(k)}
        stray = [n for n in scan.layers if n + ".weight" not in state]
        if stray:
            raise ValueError(
                f"{os.path.basename(path)}: ComfyUI-quantized layers outside the denoiser "
                f"({stray[0]}) are not supported"
            )
    state.pop(LEGACY_SCALED_FP8_KEY, None)
    for key in [k for k in state if k.endswith(COMFY_QUANT_SUFFIX)]:
        del state[key]
    for key in list(state):
        for old, new in _LEGACY_RENAME.items():
            if key.endswith(old):
                state[key[: -len(old)] + new] = state.pop(key)

    keep_fp32 = getattr(transformer_cls, "_keep_in_fp32_modules", None) or []
    if isinstance(keep_fp32, str):
        keep_fp32 = [keep_fp32]
    # fp16: _keep_in_fp32_modules names diffusers keys (Wan time_embedder, scale_shift_table), so every layer goes
    # through the converter first and dtypes are decided on the converted names.
    fp16_keep = dtype == torch.float16 and bool(keep_fp32)

    def _dtype_for(key: str) -> Any:
        return torch.float32 if fp16_keep and any(m in key.split(".") for m in keep_fp32) else dtype

    backends = {
        INT8_TENSORWISE: int8_backend,
        FP8_E4M3: fp8_backend,
        NVFP4: nvfp4_backend,
        MXFP8: mxfp8_backend,
    }
    sources: list = []  # (layer, codes, scale, block extras or None) kept for a runtime
    dequantized = 0
    for layer in scan.layers.values():
        codes = state.pop(layer.name + ".weight")
        scale = state.pop(layer.name + ".weight_scale")
        # int8 / fp8 / mxfp8 activations are quantized per call; only nvfp4 uses its static input_scale.
        input_scale = state.pop(layer.name + ".input_scale", None)
        extra = None
        if layer.format in BLOCK_SIZES:
            tensor_scale = state.pop(layer.name + ".weight_scale_2", None)
            extra = {
                "tensor_scale": None
                if tensor_scale is None
                else float(tensor_scale.float().reshape(-1)[0]),
                "input_scale": None
                if input_scale is None
                else float(input_scale.float().reshape(-1)[0]),
                "pre_quant_scale": state.pop(layer.name + ".pre_quant_scale", None),
            }
            codes, scale = decode_layer(layer.format, codes, scale)
        elif layer.format in FP8_FORMATS and codes.dtype == torch.uint8:
            codes = codes.view(getattr(torch, layer.format))
        # A key-mapped (original-layout) family moves rows whole, which the tiled block scales do not survive.
        if (backends.get(layer.format) or fp16_keep) and not (
            extra is not None and key_map is not None
        ):
            sources.append((layer, codes, scale, extra))
            state[layer.name + ".weight"] = None  # placeholder, tagged below
        elif extra is not None:
            state[layer.name + ".weight"] = _dequant_parts(
                layer.format, [(codes, scale, extra)], dtype
            )
            dequantized += 1
        else:
            state[layer.name + ".weight"] = _dequant(codes, scale, layer.group, dtype)
            dequantized += 1

    for key, value in list(state.items()):
        if value is None or not value.is_floating_point():
            continue
        wanted_dtype = keep_dtype(key) if keep_dtype is not None else None
        if wanted_dtype is not None:
            state[key] = value.to(wanted_dtype)
            continue
        if fp16_keep or value.dtype == dtype:
            continue
        state[key] = value.to(dtype)
    from accelerate import init_empty_weights

    mapping_fn, sfm = _mapping(transformer_cls) if key_map is None else (None, None)
    config_repo = kwargs.pop("config", None)
    subfolder = kwargs.pop("subfolder", None)
    token = kwargs.pop("token", None)
    cache_dir = kwargs.pop("cache_dir", None)
    local_files_only = bool(kwargs.pop("local_files_only", False))
    if not isinstance(config_repo, str):
        raise ValueError("a ComfyUI-quantized single file needs the base repo's transformer config")
    config = transformer_cls.load_config(
        config_repo,
        subfolder = subfolder,
        token = token,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
    )
    expected, optional = transformer_cls._get_signature_keys(transformer_cls)
    config.update({k: v for k, v in kwargs.items() if k in expected or k in optional})
    with init_empty_weights():
        model = transformer_cls.from_config(config)
    if prepare_model is not None:
        prepare_model(model)
    wanted = model.state_dict()

    def _convert_tagged(width: Optional[int]) -> dict:
        # float64 row tags, ``width`` columns wide (None: real width); converted tags name each output's source rows.
        for index, (layer, codes, *_rest) in enumerate(sources):
            rows = torch.arange(codes.shape[0], dtype = torch.float64) + index * _TAG
            cols = codes.shape[1] if width is None else width
            state[layer.name + ".weight"] = rows.view(-1, 1).expand(codes.shape[0], cols)
        checkpoint = pre_convert(dict(state)) if pre_convert is not None else state
        if sfm._should_convert_state_dict_to_diffusers(wanted, checkpoint):
            out = mapping_fn(
                config = config,
                checkpoint = dict(checkpoint),
                **sfm._get_mapping_function_kwargs(mapping_fn, **kwargs),
            )
        else:
            out = dict(checkpoint)
        for name in [k for k, v in out.items() if torch.is_tensor(v) and v.dtype == torch.float64]:
            segments = _decode_rows(name, out[name], sources, width = width)
            kinds = {(sources[i][0].format, sources[i][0].group) for i, _r, _n in segments}
            if len(kinds) != 1:
                raise ValueError(f"{name}: rows from layers of different formats or ConvRot groups")
            fmt, group = kinds.pop()
            if fmt in BLOCK_SIZES:
                blocks = [
                    (sources[i][1][r : r + n], sources[i][2][r : r + n], sources[i][3])
                    for i, r, n in segments
                ]
                logical = (sum(n for _i, _r, n in segments), logical_cols(fmt, blocks[0][0]))
                if name in wanted and tuple(wanted[name].shape) != logical:
                    raise ValueError(
                        f"{name}: rebuilt as {logical}, the model expects {tuple(wanted[name].shape)}"
                    )
                out[name] = _BlockWeight(fmt, blocks)
                continue
            parts = [sources[i][1][r : r + n] for i, r, n in segments]
            # one segment (a split or renamed layer) is a row slice of a contiguous tensor: no copy
            codes = parts[0] if len(parts) == 1 else torch.cat(parts)
            scale = torch.cat(
                [
                    sources[i][2].reshape(-1, 1).expand(sources[i][1].shape[0], 1)[r : r + n]
                    for i, r, n in segments
                ]
            )
            if name in wanted and tuple(wanted[name].shape) != tuple(codes.shape):
                raise ValueError(
                    f"{name}: rebuilt as {tuple(codes.shape)}, the model expects {tuple(wanted[name].shape)}"
                )
            out[name] = (codes, scale, group, fmt)
        return out

    if key_map is not None:
        # Rows move whole (no float64 tags: a 21 GB file would need ~5x that in tags)
        converted = _apply_key_map(state, sources, key_map)
        for name, value in converted.items():
            if isinstance(value, tuple) and name in wanted:
                if tuple(wanted[name].shape) != tuple(value[0].shape):
                    raise ValueError(
                        f"{name}: rebuilt as {tuple(value[0].shape)}, the model expects {tuple(wanted[name].shape)}"
                    )
    else:
        try:
            # Narrow tags first (full width is slow on a large DiT); anything unproven reruns at full width.
            converted = _convert_tagged(_NARROW_TAG_COLUMNS if sources else None)
        except Exception:  # noqa: BLE001 -- the full-width pass raises the real refusal
            converted = _convert_tagged(None)
    del state
    if fp16_keep:
        for key, value in list(converted.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                converted[key] = value.to(_dtype_for(key))

    min_features = DEFAULT_MIN_LINEAR_FEATURES if min_features is None else int(min_features)
    schemes = {INT8_TENSORWISE: TQ_INT8, FP8_E4M3: TQ_FP8, NVFP4: TQ_NVFP4, MXFP8: TQ_MXFP8}
    filters = {
        fmt: make_filter_fn(
            min_features,
            exclude_name_tokens = exclude_tokens_for_scheme(scheme, family),
            require_divisible = divisible_for_scheme(scheme),
        )
        for fmt, scheme in schemes.items()
    }
    modules = dict(model.named_modules())
    rotations: dict = {}
    native: dict = {}
    block_native: dict = {}  # fqn -> (fmt, codes, scale, tensor_scale, input_scale, bias)
    built = {TQ_INT8: 0, TQ_FP8: 0, TQ_NVFP4: 0, TQ_MXFP8: 0}
    for name, value in list(converted.items()):
        if isinstance(value, _BlockWeight):
            fmt = value.fmt
            fqn = name[: -len(".weight")] if name.endswith(".weight") else name
            module = modules.get(fqn)
            runtime = _block_runtime_args(value)
            if (
                backends.get(fmt) is not None
                and runtime is not None
                and isinstance(module, torch.nn.Linear)
                and name.endswith(".weight")
                and filters[fmt](module, fqn)
                and not any(m in fqn.split(".") for m in keep_fp32)
            ):
                codes, scale, tensor_scale, input_scale = runtime
                bias = converted.get(fqn + ".bias")
                if torch.is_tensor(bias):
                    bias = bias.to(dtype)
                block_native[fqn] = (fmt, codes, scale, tensor_scale, input_scale, bias)
                del converted[name]
                built[schemes[fmt]] += 1
            else:
                converted[name] = _dequant_parts(fmt, value.parts, _dtype_for(name))
                dequantized += 1
            continue
        if not isinstance(value, tuple):
            continue
        codes, scale, group, fmt = value
        scheme = schemes.get(fmt)
        backend = backends.get(fmt)
        fqn = name[: -len(".weight")] if name.endswith(".weight") else name
        module = modules.get(fqn)
        weight = None
        keep = (
            backend is not None
            and module is not None
            and name.endswith(".weight")
            and filters[fmt](module, fqn)
            # Studio's fp8 quant leaves the Linears a DiT keeps in fp32 (require_bf16) at full precision.
            and not (scheme == TQ_FP8 and any(m in fqn.split(".") for m in keep_fp32))
        )
        if keep:
            if backend == "native":
                native[fqn] = (codes, scale, group, scheme)
                del converted[name]
                built[scheme] += 1
                continue
            if scheme == TQ_INT8:
                weight = _int8_tensor(name, codes, scale, dtype)
            else:
                weight = _fp8_tensor(name, codes, scale, dtype, fast_accum)
        if weight is None:
            converted[name] = _dequant(codes, scale, group, _dtype_for(name))
            dequantized += 1
            continue
        converted[name] = weight
        built[scheme] += 1
        if group:
            rotations.setdefault(group, []).append(fqn)
    if len(rotations) > 1:
        raise ValueError(f"ConvRot groups {sorted(rotations)} in one checkpoint")

    def _install_native(model: Any) -> None:
        from .diffusion_native_quant import native_linear_class

        cls = native_linear_class()
        act_int8 = native_int8_act(target) if target is not None else False
        for fqn, (codes, scale, group, scheme) in native.items():
            parent_name, _, leaf = fqn.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            # the Linear supplies only shapes, compute dtype and bias (assigned from the checkpoint below)
            linear = getattr(parent, leaf).to(dtype)
            layer = cls(
                linear,
                scheme,
                act_int8 = act_int8 and scheme == TQ_INT8,
                rot_group = group,
                codes = codes,
                scale = scale,
            )
            setattr(parent, leaf, layer)
            converted[fqn + ".weight_q"] = layer.weight_q
            converted[fqn + ".weight_scale"] = layer.weight_scale
        for fqn, (fmt, codes, scale, tensor_scale, input_scale, bias) in block_native.items():
            parent_name, _, leaf = fqn.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            layer = build_runtime_linear(
                fmt,
                getattr(parent, leaf),
                codes,
                scale,
                tensor_scale = tensor_scale,
                input_scale = input_scale,
                bias = bias,
            )
            setattr(parent, leaf, layer)
            for buffer_name, buffer in layer.named_buffers(recurse = False):
                converted[f"{fqn}.{buffer_name}"] = buffer

    _install_native(model)
    missing, unexpected = model.load_state_dict(converted, strict = False, assign = True)
    ignore = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
    if ignore:
        import re
        unexpected = [k for k in unexpected if not any(re.search(p, k) for p in ignore)]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} is missing {len(missing)} weight(s) of "
            f"{transformer_cls.__name__} after conversion (e.g. {missing[0]!r})"
        )
    if any(p.device.type == "meta" for p in model.parameters()) or any(
        b.device.type == "meta" for b in model.buffers()
    ):
        # Same rebuild the hosted prequant loader does for buffers made in __init__.
        model = transformer_cls.from_config(config)
        if prepare_model is not None:
            prepare_model(model)
        _install_native(model)
        model.load_state_dict(converted, strict = False, assign = True)
    loaded = set(converted)
    del converted
    # __init__ buffers (Wan's float64 rope) follow the compute dtype; checkpoint buffers keep theirs (H3 curve table)
    for module_name, module in model.named_modules():
        if module_name in block_native:
            continue  # fp32 scales / alpha the block kernels need; codes are not floating
        for buffer_name, buffer in list(module._buffers.items()):
            if (f"{module_name}.{buffer_name}" if module_name else buffer_name) in loaded:
                continue
            if buffer is not None and buffer.is_floating_point():
                want = _dtype_for(f"{module_name}.{buffer_name}")
                if buffer.dtype != want:
                    module._buffers[buffer_name] = buffer.to(want)
    if unexpected and logger is not None:
        logger.warning(
            "diffusion.comfy_quant: %d checkpoint key(s) unused by %s (e.g. %s)",
            len(unexpected),
            transformer_cls.__name__,
            unexpected[0],
        )

    rotated: tuple = ()
    if rotations:
        from .diffusion_convrot import apply_activation_rotation, rotation_metadata

        group, fqns = next(iter(rotations.items()))
        rotated = apply_activation_rotation(model, rotation_metadata(group, fqns), logger = logger)
        device = getattr(target, "torch_device", None) or getattr(target, "device", None)
        if device is not None:
            try:
                from .diffusion_convrot import warm_rotation_cache
                warm_rotation_cache(model, device, dtype)
            except Exception:  # noqa: BLE001 -- only saves one recompile
                pass
    native_schemes = {v[3] for v in native.values()} | {
        schemes[v[0]] for v in block_native.values()
    }
    on_torchao = {s for s, count in built.items() if count and s not in native_schemes}
    if finalize:
        from .diffusion_transformer_quant import apply_small_m_padding, apply_zero_row_guard
        for scheme in sorted(on_torchao):
            apply_small_m_padding(model, scheme, family, logger = logger)
            apply_zero_row_guard(model, scheme, family, logger = logger)
    if any(built.values()):
        try:
            # the scheme the runtime treats this transformer as: the one most of its layers run
            model._unsloth_runtime_quant = max(built, key = lambda s: built[s])
        except Exception:  # noqa: BLE001 -- marker is best-effort
            pass
    if built[TQ_NVFP4]:
        try:
            model._unsloth_nvfp4_backend = nvfp4_backend
        except Exception:  # noqa: BLE001 -- marker is best-effort
            pass
        # per-model step-protect controller, as for Studio's own NVFP4 checkpoints
        from .diffusion_nvfp4_protect import attach_own_controller
        attach_own_controller(model)
    model.eval()
    convrot = len(rotated) + sum(1 for v in native.values() if v[2])
    if logger is not None:
        logger.info(
            "diffusion.comfy_quant: %s loaded from a ComfyUI checkpoint (%s): %d int8 layers kept as "
            "int8 (%s, %d ConvRot), %d fp8 layers kept as fp8 (%s), %d nvfp4 layers kept as nvfp4 (%s), "
            "%d mxfp8 layers kept as mxfp8 (%s), %d layers dequantized to %s",
            transformer_cls.__name__,
            ", ".join(f"{k} x{v}" for k, v in sorted(scan.counts().items())),
            built[TQ_INT8],
            f"{int8_backend} runtime" if built[TQ_INT8] else "no int8 runtime",
            convrot,
            built[TQ_FP8],
            f"{fp8_backend} runtime" if built[TQ_FP8] else "no fp8 runtime",
            built[TQ_NVFP4],
            f"{nvfp4_backend} runtime" if built[TQ_NVFP4] else "no nvfp4 runtime",
            built[TQ_MXFP8],
            f"{mxfp8_backend} runtime" if built[TQ_MXFP8] else "no mxfp8 runtime",
            dequantized,
            str(dtype).replace("torch.", ""),
        )
    try:
        info = {
            "backend": int8_backend if built[TQ_INT8] else None,
            "int8": built[TQ_INT8],
            "convrot": convrot,
            "fp8_backend": fp8_backend if built[TQ_FP8] else None,
            "fp8": built[TQ_FP8],
            "dequantized": dequantized,
        }
        for fmt, backend in ((NVFP4, nvfp4_backend), (MXFP8, mxfp8_backend)):
            if scan.counts().get(fmt):
                info[f"{fmt}_backend"] = backend if built[schemes[fmt]] else None
                info[fmt] = built[schemes[fmt]]
        model._unsloth_comfy_quant = info
    except Exception:  # noqa: BLE001 -- diagnostic marker only
        pass
    return model
