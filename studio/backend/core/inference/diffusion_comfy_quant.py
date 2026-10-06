# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load a ComfyUI-format quantized single-file DiT, or refuse it by name.

ComfyUI's quantized checkpoints keep the quantized codes under the layer's own ``.weight`` key and put
the dequant parameters beside it: ``.weight_scale`` (and for some formats ``.input_scale``,
``.weight_scale_2``, ...). The format of each layer is declared in one of three ways:

    per layer     ``<layer>.comfy_quant``: a uint8 tensor holding JSON, ``{"format": "int8_tensorwise",
                  "convrot": true, "convrot_groupsize": 256}``
    whole file    the safetensors header's ``_quantization_metadata``: ``{"layers": {<layer>: {...}}}``
    legacy        a ``scaled_fp8`` marker tensor plus ``<layer>.scale_weight`` / ``.scale_input``

A stock ``from_single_file`` knows none of this: it casts the int8 / fp8 codes to the compute dtype
as if they were weights and reports the scales as unused keys, so the load succeeds and every
quantized layer is off by its scale (pure noise for int8). This module reads the declaration from the
header, before a weight byte is loaded, and either

- maps ``int8_tensorwise`` layers (optionally ConvRot-rotated) into Studio's own int8 runtime with the
  codes and per-row scales unchanged: the same ``Int8Tensor`` and ``ConvRotLinear`` a hosted
  INT8 / INT8-ConvRot checkpoint rebuilds into,
- or dequantizes them (``codes * scale``, then the ConvRot rotation undone) where that runtime is not
  available or the layer is one Studio keeps in bf16, and dequantizes fp8 layers
  (``float8_e4m3fn`` / ``float8_e5m2``, scalar or per-row scale) on load,
- and refuses anything else (nvfp4, mxfp8, the 4/6-bit int8 packings, an unknown or missing format,
  scales with no declaration) with an error naming the format, never a silent mis-load.

The row bookkeeping is generic: the family's own single-file converter runs once, with every int8
weight replaced by a float64 tensor whose rows are tagged ``(source key, row)``. Whatever splits,
concatenations and renames the converter applies, the tags that come out say which source rows each
diffusers weight is made of, and the codes and scales are gathered by row from the file. A converter
that splices columns of different layers, changes the column count, casts or transforms values
leaves tags that do not decode, and the load is refused. (The tags are constant along a row, so a
pure reordering of input columns would go unseen; no diffusers converter reorders a Linear's inputs.)

Reading the declaration is torch-free.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

COMFY_QUANT_SUFFIX = ".comfy_quant"
QUANT_METADATA_KEY = "_quantization_metadata"
LEGACY_SCALED_FP8_KEY = "scaled_fp8"
COMFY_INT8_ENV = "UNSLOTH_DIFFUSION_COMFY_INT8"
COMFY_FP8_ENV = "UNSLOTH_DIFFUSION_COMFY_FP8"

INT8_TENSORWISE = "int8_tensorwise"
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
}
_LEGACY_RENAME = {".scale_weight": ".weight_scale", ".scale_input": ".input_scale"}

# Row tag: key index * 2**24 + row, exact in float64 for any real checkpoint.
_TAG = float(1 << 24)
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


def _read_json_tensor(path: str, entry: dict, base: int) -> Any:
    start, end = entry["data_offsets"]
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
        "without ConvRot), float8_e4m3fn and float8_e5m2; use the bf16 file, a GGUF, or Studio's "
        "own int8 / fp8 transformer quantization instead."
    )


def refuse_comfy_quant(path: Optional[str]) -> Optional[ComfyQuantScan]:
    """``scan_comfy_quant`` that raises ``ValueError`` for a checkpoint it would have to refuse."""
    scan = scan_comfy_quant(path)
    problem = comfy_quant_error(scan, os.path.basename(str(path or "")))
    if problem:
        raise ValueError(problem)
    return scan


_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def comfy_quant_resident_mib(
    path: str,
    scan: ComfyQuantScan,
    *,
    kept_formats: Any = (),
    exclude_tokens: Any = (),
    keep_key: Any = None,
    compute_bytes: int = 2,
) -> Optional[int]:
    """Resident MiB of the model ``load_comfy_quant_transformer`` builds from ``path``, from the header alone.

    A layer whose format is in ``kept_formats`` (an int8 one only if its name holds none of ``exclude_tokens``) stays 1 byte per
    weight plus a float32 scale per row; every other quantized layer is dequantized to ``compute_bytes`` per weight,
    which the file size under-states by 2x. Plain tensors count as stored (float32 ones narrow to
    ``compute_bytes``). ``keep_key`` limits the count to the keys the loader reads. None when unreadable."""
    try:
        header, _base = _read_header(str(path))
    except Exception:  # noqa: BLE001 -- the caller keeps its file-size estimate
        return None
    header.pop("__metadata__", None)
    quant = {name + ".weight": layer for name, layer in scan.layers.items()}
    total = 0
    for key, entry in header.items():
        if keep_key is not None and not keep_key(key):
            continue
        if key.endswith(COMFY_QUANT_SUFFIX) or key == LEGACY_SCALED_FP8_KEY:
            continue
        if any(key.endswith(s) for s in _SCALE_SUFFIXES):
            continue
        shape = entry.get("shape") or []
        n = 1
        for d in shape:
            n *= int(d)
        layer = quant.get(key)
        if layer is not None:
            kept = layer.format in kept_formats and not (
                layer.format == INT8_TENSORWISE and any(t in key for t in exclude_tokens)
            )
            total += n + 4 * int(shape[0]) if kept else n * compute_bytes
            continue
        size = _DTYPE_BYTES.get(str(entry.get("dtype")), compute_bytes)
        total += n * (min(size, compute_bytes) if size >= 4 else size)
    return int(total / (1024 * 1024))


# ------------------------------------------------------------------------------------------- loading
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


def _fp8_tensor(codes: Any, scale: Any, dtype: Any, templates: dict, device: Any = None) -> Any:
    """The torchao fp8 weight Studio's own fp8 quant makes, holding ``codes`` and per-row ``scale`` unchanged.

    The tensor type and its settings (per-row activation quant, mm config, kernel preference) come from quantizing a
    small Linear of the same width with Studio's fp8 config, so they match the runtime path on every torchao
    version; only the codes and scales are swapped in. A per-tensor scale is the same scale on every row."""
    import torch

    rows, cols = codes.shape
    template = templates.get(cols)
    if template is None:
        from torchao.quantization import quantize_

        from .diffusion_transformer_quant import TQ_FP8, _make_quant_config

        # 16 rows: torchao skips an fp8 Linear whose features are not 16-aligned (scaled_mm)
        probe = torch.nn.Linear(cols, 16, bias = False, dtype = dtype, device = device or "cpu")
        quantize_(probe, _make_quant_config(TQ_FP8), filter_fn = lambda *_a: True)
        template = templates[cols] = probe.weight.detach()
    names, attrs = template.__tensor_flatten__()
    if list(names) != ["qdata", "scale"] or template.qdata.dtype != codes.dtype:
        raise ValueError(f"unexpected torchao fp8 layout {type(template).__name__} {names}")
    data = {
        "qdata": codes.contiguous(),
        "scale": scale.to(torch.float32).reshape(-1, 1).expand(rows, 1).contiguous(),
    }
    return type(template).__tensor_unflatten__(data, attrs, None, None)


def _mapping(transformer_cls: Any) -> tuple[Any, Any]:
    from diffusers.loaders import single_file_model as sfm

    name = sfm._get_single_file_loadable_mapping_class(transformer_cls)
    entry = sfm.SINGLE_FILE_LOADABLE_CLASSES.get(name or "")
    if not entry:
        raise ValueError(f"{transformer_cls.__name__} has no single-file converter")
    return entry["checkpoint_mapping_fn"], sfm


def _decode_rows(name: str, tagged: Any, sources: list) -> list:
    """``[(source index, first row, n rows)]`` for a converted int8 weight, or raise."""
    import torch

    if tagged.dtype != torch.float64 or tagged.dim() != 2 or tagged.numel() == 0:
        raise ValueError(f"{name}: the converter changed an int8 weight's dtype or rank")
    col = tagged[:, 0]
    for j in (tagged.shape[1] // 2, tagged.shape[1] - 1):
        if not torch.equal(col, tagged[:, j]):
            raise ValueError(f"{name}: the converter mixed columns of an int8 weight")
    if not torch.equal(col, col.floor()) or bool((col < 0).any()):
        raise ValueError(f"{name}: the converter transformed int8 weight values")
    ids = (col / _TAG).floor().long()
    rows = (col - ids.double() * _TAG).long()
    segments: list = []
    for i, r in zip(ids.tolist(), rows.tolist()):
        if i >= len(sources) or r >= sources[i][1].shape[0]:
            raise ValueError(f"{name}: an int8 row does not decode to a source row")
        if segments and segments[-1][0] == i and segments[-1][1] + segments[-1][2] == r:
            segments[-1][2] += 1
        else:
            segments.append([i, r, 1])
    if any(tagged.shape[1] != sources[i][1].shape[1] for i, _r, _n in segments):
        raise ValueError(f"{name}: the converter changed an int8 weight's column count")
    return segments


# Denoiser classes diffusers has no single-file converter for, mapped to the module describing their original
# (ComfyUI) layout through ``comfy_layout(path) -> {"key_map", "prepare_model", "keep_dtype"}``. Imported on demand, so
# every caller of ``load_comfy_quant_transformer`` (single file or a hosted ComfyUI-format twin) gets them.
_ORIGINAL_LAYOUTS = {
    "HunyuanVideo15Transformer3DModel": "video_hv15_comfy",
    "MiniMaxH3Transformer3DModel": "video_minimax_h3_comfy",
}


def original_layout(transformer_cls: Any, path: str) -> Optional[dict]:
    """The key map / prepare / dtype hooks for a class with no diffusers converter, or None."""
    module = _ORIGINAL_LAYOUTS.get(getattr(transformer_cls, "__name__", ""))
    if module is None:
        return None
    import importlib

    return importlib.import_module(f"{__package__}.{module}").comfy_layout(path)


def _apply_key_map(state: dict, kept: list, key_map: Any) -> dict:
    """``state`` (file keys) renamed and row-split by ``key_map``; kept layers become ``(codes, scale, group)``.

    Row selections are views where they are one contiguous run, so no layer is copied except a reordered one."""
    import torch

    sources = {layer.name + ".weight": (codes, scale, layer.group) for layer, codes, scale in kept}

    def take(value: Any, rows: Any) -> Any:
        if rows is None:
            return value
        if isinstance(value, tuple):
            codes, scale, group = value
            scale = scale.reshape(-1, 1).expand(codes.shape[0], 1)
            return (take(codes, rows), take(scale, rows), group)
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
    """``"torchao"`` when Studio's own fp8 quant would run this family resident on ``target`` (the fp8 codes then
    keep their storage and run Studio's fp8 GEMM), else None (dequantize to the compute dtype). Studio keeps fp8
    on a resident DiT only, so an offloaded plan dequantizes. ``UNSLOTH_DIFFUSION_COMFY_FP8=0`` always
    dequantizes."""
    if offload or (os.environ.get(COMFY_FP8_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return None
    try:
        from .diffusion_transformer_quant import (
            TQ_FP8,
            native_quant_scheme,
            select_transformer_quant_scheme,
        )
        if native_quant_scheme(target, TQ_FP8, family = family) is not None:
            return None  # the torchao-free fp8 twin builds its own codes; dequantize instead
        if (
            select_transformer_quant_scheme(target, TQ_FP8, family = family, base_repo = base_repo)
            == TQ_FP8
        ):
            return "torchao"
    except Exception:  # noqa: BLE001 -- an unanswerable probe takes the exact dense path
        pass
    return None


def load_comfy_quant_transformer(
    transformer_cls: Any,
    path: str,
    scan: ComfyQuantScan,
    sf_kwargs: dict,
    *,
    int8_backend: Optional[str],
    family: Optional[str] = None,
    target: Any = None,
    logger: Any = None,
    fp8_backend: Optional[str] = None,
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
    (``"torchao"``) or as native int8 twins (``"native"``); everything else is dequantized to the
    compute dtype. With ``fp8_backend="torchao"`` (``comfy_fp8_backend``) float8_e4m3fn layers likewise keep
    their codes as torchao fp8 weights with per-row scales. ``keep_key(key)`` limits the read to the DiT's keys
    of a file that bundles other components; ``pre_convert(state)`` renames keys (never values) before the
    family converter runs. A family with no diffusers single-file converter passes ``key_map(key, shape)``:
    ``[(diffusers key, rows)]`` per file key, ``rows`` None (the whole tensor) or ``[(first row, n rows), ...]``;
    ``prepare_model(model)`` reshapes the freshly built model before the weights load, and ``keep_dtype(key)``
    names a dtype a file tensor keeps instead of the compute dtype. Raises ``ValueError`` for a checkpoint it
    must refuse."""
    import torch
    from safetensors.torch import load_file

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

    int8_sources: list = []  # (layer, codes, scale)
    dequantized = 0
    for layer in scan.layers.values():
        codes = state.pop(layer.name + ".weight")
        scale = state.pop(layer.name + ".weight_scale")
        state.pop(layer.name + ".input_scale", None)
        if layer.format in FP8_FORMATS and codes.dtype == torch.uint8:
            codes = codes.view(getattr(torch, layer.format))
        if (layer.format == INT8_TENSORWISE and int8_backend) or (
            layer.format == "float8_e4m3fn" and fp8_backend and not layer.group
        ):
            int8_sources.append((layer, codes, scale))
            state[layer.name + ".weight"] = None  # placeholder, tagged below
        else:
            state[layer.name + ".weight"] = _dequant(codes, scale, layer.group, dtype)
            dequantized += 1

    keep_fp32 = getattr(transformer_cls, "_keep_in_fp32_modules", None) or []
    if isinstance(keep_fp32, str):
        keep_fp32 = [keep_fp32]
    for key, value in list(state.items()):
        if value is None or not value.is_floating_point():
            continue
        wanted_dtype = keep_dtype(key) if keep_dtype is not None else None
        if wanted_dtype is not None:
            state[key] = value.to(wanted_dtype)
            continue
        if value.dtype == dtype:
            continue
        if dtype == torch.float16 and any(m in key.split(".") for m in keep_fp32):
            continue
        state[key] = value.to(dtype)
    if key_map is None:
        for index, (layer, codes, _scale) in enumerate(int8_sources):
            rows = torch.arange(codes.shape[0], dtype = torch.float64) + index * _TAG
            state[layer.name + ".weight"] = rows.view(-1, 1).expand(codes.shape)

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
    if pre_convert is not None:
        state = pre_convert(state)
    if key_map is not None:
        converted = _apply_key_map(state, int8_sources, key_map)
    elif sfm._should_convert_state_dict_to_diffusers(wanted, state):
        converted = mapping_fn(
            config = config,
            checkpoint = dict(state),
            **sfm._get_mapping_function_kwargs(mapping_fn, **kwargs),
        )
    else:
        converted = dict(state)
    del state

    rotations: dict = {}
    native: dict = {}
    built = 0
    for name in [
        k for k, v in converted.items() if torch.is_tensor(v) and v.dtype == torch.float64
    ]:
        segments = _decode_rows(name, converted[name], int8_sources)
        sources = {int8_sources[i][0].group for i, _r, _n in segments}
        if len(sources) != 1:
            raise ValueError(f"{name}: rows from layers with different ConvRot groups")
        if len({int8_sources[i][0].format for i, _r, _n in segments}) != 1:
            raise ValueError(f"{name}: rows from layers of different quantization formats")
        group = sources.pop()
        code_dtype = int8_sources[segments[0][0]][1].dtype
        # Through uint8: concatenation is not implemented for float8 on every torch this supports.
        codes = torch.cat(
            [int8_sources[i][1][r : r + n].view(torch.uint8) for i, r, n in segments]
        ).view(code_dtype)
        scale = torch.cat(
            [
                int8_sources[i][2].reshape(-1, 1).expand(int8_sources[i][1].shape[0], 1)[r : r + n]
                for i, r, n in segments
            ]
        )
        converted[name] = (codes, scale, group)

    from .diffusion_transformer_quant import (
        DEFAULT_MIN_LINEAR_FEATURES,
        TQ_FP8,
        TQ_INT8,
        divisible_for_scheme,
        exclude_tokens_for_scheme,
        make_filter_fn,
        native_int8_act,
    )

    filter_fn = make_filter_fn(
        DEFAULT_MIN_LINEAR_FEATURES,
        exclude_name_tokens = exclude_tokens_for_scheme(TQ_INT8, family),
    )
    # The filter Studio's own fp8 quant applies (scaled_mm needs 16-aligned features).
    fp8_filter_fn = make_filter_fn(
        DEFAULT_MIN_LINEAR_FEATURES,
        exclude_name_tokens = exclude_tokens_for_scheme(TQ_FP8, family),
        require_divisible = divisible_for_scheme(TQ_FP8),
    )
    fp8_templates: dict = {}
    fp8_built = 0
    modules = dict(model.named_modules())
    for name, value in list(converted.items()):
        if not isinstance(value, tuple):
            continue
        codes, scale, group = value
        fqn = name[: -len(".weight")] if name.endswith(".weight") else name
        module = modules.get(fqn)
        weight = None
        if codes.dtype != torch.int8:
            if module is not None and name.endswith(".weight") and fp8_filter_fn(module, fqn):
                device = getattr(target, "torch_device", None) if target is not None else None
                weight = _fp8_tensor(codes, scale, dtype, fp8_templates, device = device)
            if weight is None:
                converted[name] = _dequant(codes, scale, group, dtype)
                dequantized += 1
            else:
                converted[name] = weight
                fp8_built += 1
            continue
        if module is not None and name.endswith(".weight") and filter_fn(module, fqn):
            if int8_backend == "native":
                native[fqn] = value
                del converted[name]
                built += 1
                continue
            weight = _int8_tensor(name, codes, scale, dtype)
        if weight is None:
            converted[name] = _dequant(codes, scale, group, dtype)
            dequantized += 1
            continue
        converted[name] = weight
        built += 1
        if group:
            rotations.setdefault(group, []).append(fqn)
    if len(rotations) > 1:
        raise ValueError(f"ConvRot groups {sorted(rotations)} in one checkpoint")

    def _install_native(model: Any) -> None:
        from .diffusion_native_quant import native_linear_class

        cls = native_linear_class()
        act_int8 = native_int8_act(target) if target is not None else False
        for fqn, (codes, scale, group) in native.items():
            parent_name, _, leaf = fqn.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            # the Linear supplies only shapes, compute dtype and bias (assigned from the checkpoint below)
            linear = getattr(parent, leaf).to(dtype)
            layer = cls(
                linear, TQ_INT8, act_int8 = act_int8, rot_group = group, codes = codes, scale = scale
            )
            setattr(parent, leaf, layer)
            converted[fqn + ".weight_q"] = layer.weight_q
            converted[fqn + ".weight_scale"] = layer.weight_scale

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
    del converted
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
    if built:
        if not native:
            from .diffusion_transformer_quant import apply_small_m_padding
            apply_small_m_padding(model, TQ_INT8, family, logger = logger)
        try:
            model._unsloth_runtime_quant = TQ_INT8
        except Exception:  # noqa: BLE001 -- marker is best-effort
            pass
    if fp8_built:
        from .diffusion_transformer_quant import apply_small_m_padding
        apply_small_m_padding(model, TQ_FP8, family, logger = logger)
        if not built:
            try:
                model._unsloth_runtime_quant = TQ_FP8
            except Exception:  # noqa: BLE001 -- marker is best-effort
                pass
    model.eval()
    if logger is not None:
        logger.info(
            "diffusion.comfy_quant: %s loaded from a ComfyUI checkpoint (%s): %d int8 layers kept as "
            "int8 (%s, %d ConvRot), %d layers dequantized to %s",
            transformer_cls.__name__,
            ", ".join(f"{k} x{v}" for k, v in sorted(scan.counts().items())),
            built,
            f"{int8_backend} runtime" if built else "no int8 runtime",
            len(rotated) + sum(1 for v in native.values() if v[2]),
            dequantized,
            str(dtype).replace("torch.", ""),
        )
        if fp8_built:
            logger.info(
                "diffusion.comfy_quant: %d fp8 layers kept as fp8 (%s runtime)", fp8_built, fp8_backend
            )
    try:
        model._unsloth_comfy_quant = {
            "backend": int8_backend if built else None,
            "int8": built,
            "convrot": len(rotated) + sum(1 for v in native.values() if v[2]),
            "dequantized": dequantized,
        }
        if fp8_built:
            model._unsloth_comfy_quant.update({"fp8": fp8_built, "fp8_backend": fp8_backend})
    except Exception:  # noqa: BLE001 -- diagnostic marker only
        pass
    return model
