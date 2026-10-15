# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read and write a pre-quantized checkpoint as ``.safetensors`` instead of a ``torch.save`` pickle.

torchao's weight subclasses (``Float8Tensor``, ``Int8Tensor``, ``NVFP4Tensor``, ...) are wrapper
tensors: several plain tensors plus reconstruction metadata behind one logical weight. safetensors
stores flat tensors only, which is why every pre-quant artifact Unsloth has published so far is a
pickle. torchao >= 0.16 closes that gap with ``flatten_tensor_state_dict`` /
``unflatten_tensor_state_dict``, the same pair ``transformers`` and ``diffusers`` use for their own
torchao checkpoints, so the subclass is split into ``<fqn>._weight_qdata`` / ``._weight_scale`` and a
JSON description of how to put it back.

Why bother, given the pickle is already read under ``weights_only`` plus a constructor allowlist:

- That allowlist is a real gate, not a formality. ``restricted_prequant_load_supported`` returns
  False when torch cannot express it or when torchao does not ship a scheme's constructors, and the
  callers in ``video.py`` then refuse the checkpoint outright and fall back to dense. A safetensors
  artifact has no constructors to allowlist, so that whole failure mode does not apply to it.
- A pickle has to be parsed before anything about it is known. A safetensors header is a JSON blob at
  the front of the file, so the scheme / base model / filter checks run off a few KB, before a single
  weight byte is read, and a mismatched artifact costs a header read rather than a full parse.
- mmap, lazy per-tensor reads, the Hub's file viewer and cross-framework reads all come for free.

**On-disk layout.** Tensors are exactly what ``flatten_tensor_state_dict`` produces. The header is
torchao's own metadata (``tensor_names`` plus one JSON entry per weight, which is what makes the file
readable by anything that understands torchao checkpoints) with two namespaced keys added:

    "unsloth_format"   -> the same format tag the pickle carries in ckpt["format"]
    "unsloth_metadata" -> JSON of the same dict the pickle carries in ckpt["metadata"]

``is_metadata_torchao`` walks ``tensor_names`` only, so the extra keys do not disturb it.

**Writing needs a newer torchao than reading does.** Only the new-style subclasses can be flattened,
and through torchao 0.17 the int8 config still produces the legacy
``LinearActivationQuantizedTensor`` over an ``AffineQuantizedTensor``; 0.18 produces ``Int8Tensor``,
which flattens. Both versions RECONSTRUCT an ``Int8Tensor`` file happily, so this binds the machine
that builds an artifact, never the one that loads it. fp8 has been flattenable since 0.16.

**No new format tag.** The tags (``..._v1``, the rotated ``..._v2``) describe what the WEIGHTS mean,
and that is unchanged by the container; a rotated safetensors artifact is still ``_v2`` and still
refused by a build that cannot rotate. Old builds are kept away from these files by the FILENAME,
not the tag: they resolve ``<Model>-<SCHEME>.pt`` and know nothing of a ``.safetensors`` sibling, so
a repo can host both and each build reads the one it understands.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, Optional

SAFETENSORS_SUFFIX = ".safetensors"

UNSLOTH_FORMAT_KEY = "unsloth_format"
UNSLOTH_METADATA_KEY = "unsloth_metadata"
# Root-level plain tensors: torchao would rsplit their keys on ".".
UNSLOTH_ROOT_PREFIX = "unsloth_root::"
UNSLOTH_ROOT_KEYS_KEY = "unsloth_root_tensors"
_UNSLOTH_HEADER_KEYS = (
    UNSLOTH_FORMAT_KEY,
    UNSLOTH_METADATA_KEY,
    UNSLOTH_ROOT_KEYS_KEY,
    "unsloth_quant_layout",
    "unsloth_source",
)

# First torchao release shipping flatten/unflatten at this path.
MIN_TORCHAO_VERSION = (0, 16)


def is_safetensors_checkpoint(name: Optional[str]) -> bool:
    """Whether ``name`` (a filename or a path) names a safetensors pre-quant artifact."""
    return bool(name) and str(name).lower().endswith(SAFETENSORS_SUFFIX)


def _torchao_helpers() -> Optional[tuple]:
    """The is_stubbed check must come first: a Windows ROCm stub fakes every torchao import."""
    try:
        from core._torchao_stub import is_stubbed
        if is_stubbed("torchao"):
            return None
    except Exception:  # noqa: BLE001 - no stub module to ask means nothing is stubbed
        pass
    try:
        from torchao.prototype.safetensors.safetensors_support import (
            flatten_tensor_state_dict,
            unflatten_tensor_state_dict,
        )
    except Exception:  # noqa: BLE001 - torchao absent, too old, or the module moved
        return None
    # 0.14 has this module but no Int8Tensor: unflatten fails (KeyError '_data') after planning.
    if _version_tuple(_torchao_version()) < MIN_TORCHAO_VERSION:
        return None
    return flatten_tensor_state_dict, unflatten_tensor_state_dict


def _version_tuple(version: Optional[str]) -> tuple:
    """Unparseable reads as new enough (the feature import decides)."""
    try:
        parts = str(version).split("+")[0].split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:  # noqa: BLE001
        return (999, 0)


def _torchao_version() -> Optional[str]:
    try:
        import torchao
        return getattr(torchao, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def safetensors_prequant_supported() -> bool:
    """Whether this install can read (and write) safetensors pre-quant checkpoints."""
    if _torchao_helpers() is None:
        return False
    try:
        import safetensors  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def plain_safetensors_supported() -> bool:
    """The layerwise-cast encoder format stores plain tensors, without torchao subclasses."""
    try:
        from safetensors import safe_open  # noqa: F401
    except ImportError:
        return False
    return True


def load_plain_prequant_safetensors(
    path: str,
    *,
    device: str = "cpu",
    skip_names: Iterable[str] = (),
) -> dict:
    """Read a plain-tensor prequant checkpoint without torchao; refuses subclass metadata and unlisted tensors.

    ``skip_names`` are validated but never read."""
    from safetensors import safe_open

    with safe_open(path, framework = "pt", device = device) as handle:
        raw = handle.metadata() or {}
        fmt = raw.get(UNSLOTH_FORMAT_KEY)
        if not fmt:
            raise ValueError(f"{path} is not an Unsloth pre-quant checkpoint")
        metadata = json.loads(raw.get(UNSLOTH_METADATA_KEY) or "{}")
        names = json.loads(raw.get("tensor_names") or "null")
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} has invalid pre-quant metadata")
        if (
            not isinstance(names, list)
            or not all(isinstance(name, str) and name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(f"{path} has invalid tensor_names")
        keys = set(handle.keys())
        roots = {key for key in keys if key.startswith(UNSLOTH_ROOT_PREFIX)}
        root_names = {key[len(UNSLOTH_ROOT_PREFIX) :] for key in roots}
        if (
            any(not name or "." in name for name in root_names)
            or root_names.intersection(names)
            or set(names).intersection(roots)
            or set(names) != keys - roots
        ):
            raise ValueError(f"{path} has tensors its header does not account for")
        for name in names:
            if json.loads(raw.get(name) or "null") != {"_type": "Tensor"}:
                raise ValueError(f"{path} requires a tensor-subclass reader for {name!r}")
        skip = frozenset(skip_names)
        state_dict = {name: handle.get_tensor(name) for name in names if name not in skip}
        state_dict.update(
            (key[len(UNSLOTH_ROOT_PREFIX) :], handle.get_tensor(key)) for key in roots
        )
    return {"format": str(fmt), "state_dict": state_dict, "metadata": metadata}


def _first(value: Any) -> Any:
    """Never unpack blindly: unwraps torchao's (dict, extra) tuple, and passes a bare dict through."""
    return value[0] if isinstance(value, tuple) else value


def _root_level_keys(state_dict: Any) -> list:
    """Keys with no ``.`` in them, in the order the state dict gives them."""
    try:
        keys = list(state_dict.keys())
    except Exception:  # noqa: BLE001 - not a mapping is the caller's problem, not this check's
        return []
    return [k for k in keys if "." not in str(k)]


def unsupported_state_dict_keys(state_dict: Any) -> list:
    """Root-level quantized keys, which cannot round-trip; plain root tensors use UNSLOTH_ROOT_PREFIX."""
    roots = _root_level_keys(state_dict)
    if not roots:
        return []
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch means no way to tell the two apart; refuse as before
        return roots
    return [k for k in roots if type(state_dict[k]) is not torch.Tensor]


def save_prequant_safetensors(path: str, *, fmt: str, state_dict: Any, metadata: Any) -> None:
    """Raises instead of writing a pickle under a .safetensors name when torchao cannot serialize."""
    helpers = _torchao_helpers()
    if helpers is None:
        raise RuntimeError(
            "this install cannot write a safetensors pre-quant checkpoint: torchao >= "
            f"{'.'.join(str(p) for p in MIN_TORCHAO_VERSION)} is required for "
            "torchao.prototype.safetensors.safetensors_support"
        )
    from safetensors.torch import save_file

    undotted = unsupported_state_dict_keys(state_dict)
    if undotted:
        raise ValueError(
            "these state dict keys have no '.' and are not plain tensors, so torchao's unflatten "
            f"cannot address them and this checkpoint would never load: {undotted[:8]}"
            + (f" (+{len(undotted) - 8} more)" if len(undotted) > 8 else "")
            + ". Write this build as a .pt checkpoint instead."
        )

    roots = _root_level_keys(state_dict)
    quantizable = (
        {k: v for k, v in state_dict.items() if k not in set(roots)} if roots else state_dict
    )

    flatten, _ = helpers
    try:
        flat, torchao_metadata = flatten(quantizable)
    except ValueError as exc:
        # torchao <= 0.17 int8 yields legacy tensors with no flatten support: a build-machine limit.
        if "Unsupported tensor type" not in str(exc):
            raise
        raise ValueError(
            f"{exc}. This torchao ({_torchao_version() or 'unknown'}) still quantises to a legacy "
            "tensor subclass that cannot be written to safetensors; torchao >= 0.18 produces the "
            "flattenable subclasses for every scheme Unsloth ships. Upgrade torchao to build this "
            "artifact, or write it as a .pt checkpoint."
        ) from exc

    # torchao metadata is already JSON-encoded; encoding again breaks is_metadata_torchao.
    header = dict(torchao_metadata or {})
    header[UNSLOTH_FORMAT_KEY] = str(fmt)
    header[UNSLOTH_METADATA_KEY] = json.dumps(metadata or {}, default = str)
    if roots:
        flat = dict(flat)
        for key in roots:
            # safetensors refuses views.
            flat[f"{UNSLOTH_ROOT_PREFIX}{key}"] = state_dict[key].contiguous()
        header[UNSLOTH_ROOT_KEYS_KEY] = json.dumps([str(k) for k in roots])
    save_file(flat, path, metadata = header)


def read_prequant_header(path: str) -> Optional[dict]:
    """Reads only the header, so it is cheap on multi-GB files; None means not an Unsloth checkpoint."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework = "pt") as handle:
            raw = handle.metadata() or {}
    except Exception:  # noqa: BLE001 - unreadable or not safetensors at all
        return None
    fmt = raw.get(UNSLOTH_FORMAT_KEY)
    if not fmt:
        return None
    try:
        metadata = json.loads(raw.get(UNSLOTH_METADATA_KEY) or "{}")
    except Exception:  # noqa: BLE001 - a corrupt metadata blob is not a usable checkpoint
        return None
    if not isinstance(metadata, dict):
        return None
    return {"format": str(fmt), "metadata": metadata}


# torchao's exact phrasing for a field its constructor will not take.
_UNEXPECTED_KWARG = re.compile(r"unexpected keyword argument '([^']+)'")

# Inert values only; any other value is a real setting and must not be dropped.
_INERT_FIELD_VALUES = (False, None, 0)


def _drop_field(value: Any, name: str, removed: list) -> Any:
    """``value`` with every nested occurrence of key ``name`` removed, recording what was dropped."""
    if isinstance(value, dict):
        out = {}
        for key, inner in value.items():
            if key == name:
                removed.append(inner)
                continue
            out[key] = _drop_field(inner, name, removed)
        return out
    if isinstance(value, list):
        return [_drop_field(v, name, removed) for v in value]
    return value


def _header_without_inert_tensor_field(
    header: dict, tensors: dict, name: str, *, path: str
) -> Optional[dict]:
    import torch

    pruned = dict(header)
    victims = []
    for key, value in header.items():
        if key in _UNSLOTH_HEADER_KEYS:
            continue
        try:
            parsed = json.loads(value)
        except Exception:  # noqa: BLE001 - not every header entry is JSON
            continue
        names = parsed.get("_tensor_data_names") if isinstance(parsed, dict) else None
        if not isinstance(names, list) or name not in names or "." not in key:
            continue
        module_fqn, weight_name = key.rsplit(".", 1)
        flat_key = f"{module_fqn}._{weight_name}_{name}"
        tensor = tensors.get(flat_key)
        if tensor is None:
            raise ValueError(
                f"{path} lists {name!r} for {key} but has no {flat_key!r} tensor; the checkpoint "
                "is incomplete or was edited"
            )
        if bool(torch.any(tensor != 0)):
            raise ValueError(
                f"{path} records a non-zero {name!r} for {key}, which this torchao "
                f"({_torchao_version() or 'unknown'}) cannot construct. Upgrade torchao to read "
                "this checkpoint."
            )
        parsed["_tensor_data_names"] = [n for n in names if n != name]
        pruned[key] = json.dumps(parsed)
        victims.append(flat_key)
    if not victims:
        return None
    for flat_key in victims:
        tensors.pop(flat_key, None)
    return pruned


def _header_without_unconstructible_fields(
    unflatten: Any,
    tensors: Any,
    raw: dict,
    *,
    path: str,
    attempts: int = 8,
) -> dict:
    """Drops a field the constructor rejects only if its value is the default; otherwise refuses."""
    header = dict(raw)
    dropped: list = []
    for _ in range(attempts):
        try:
            unflatten(tensors, header)
            break
        except Exception as exc:  # noqa: BLE001 - only the one shape below is acted on
            match = _UNEXPECTED_KWARG.search(str(exc))
            if match is None:
                break
            name = match.group(1)
            removed: list = []
            pruned = {}
            for key, value in header.items():
                if key in _UNSLOTH_HEADER_KEYS:
                    pruned[key] = value
                    continue
                try:
                    parsed = json.loads(value)
                except Exception:  # noqa: BLE001 - not every header entry is JSON
                    pruned[key] = value
                    continue
                pruned[key] = json.dumps(_drop_field(parsed, name, removed))
            if not removed:
                pruned = _header_without_inert_tensor_field(header, tensors, name, path = path)
                if pruned is None:
                    break
                dropped.append(name)
                header = pruned
                continue
            live = [v for v in removed if v not in _INERT_FIELD_VALUES]
            if live:
                raise ValueError(
                    f"{path} records {name!r}={live[0]!r}, which this torchao "
                    f"({_torchao_version() or 'unknown'}) cannot construct. Upgrade torchao to read "
                    "this checkpoint; dropping the field would load the weights under settings the "
                    "file did not ask for."
                ) from exc
            dropped.append(name)
            header = pruned
    if dropped:
        print(
            f"note: {path} carries {', '.join(sorted(set(dropped)))} from a newer torchao; "
            f"the field is inert here and was ignored",
            flush = True,
        )
    return header


_ST_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "C64": "complex64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "BOOL": "bool",
}

# safetensors refuses headers past 100 MB; a pre-quant header is a few MB at most.
_MAX_HEADER_BYTES = 100_000_000


def _mapped_tensors(path: str) -> tuple:
    """Maps the file instead of copying it to host RAM; a bad header raises so the caller re-reads."""
    import struct

    import torch

    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) != 8:
            raise ValueError(f"{path} is not a safetensors file")
        (length,) = struct.unpack("<Q", head)
        if length <= 0 or length > _MAX_HEADER_BYTES or 8 + length > size:
            raise ValueError(f"{path} has a corrupt safetensors header")
        header = json.loads(fh.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"{path} has a corrupt safetensors header")
    metadata = header.pop("__metadata__", None) or {}
    base = 8 + length
    entries = []
    for name, info in header.items():
        try:
            dtype = getattr(torch, _ST_DTYPES[info["dtype"]])
            start, end = (int(v) for v in info["data_offsets"])
            shape = list(info["shape"])
        except Exception:  # noqa: BLE001 - unknown dtype (on this torch) or a malformed entry
            raise ValueError(f"{path}: unreadable tensor entry {name!r}") from None
        if not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape):
            raise ValueError(f"{path}: tensor {name!r} has an invalid shape")
        count = 1
        for d in shape:
            count *= d
        itemsize = torch.empty((), dtype = dtype).element_size()
        if count * itemsize != end - start:
            raise ValueError(f"{path}: tensor {name!r} size does not match its shape")
        entries.append((start, end, name, dtype, shape, itemsize))
    entries.sort(key = lambda e: (e[0], e[1]))
    cursor = 0
    for start, end, name, *_ in entries:
        if start != cursor:
            raise ValueError(f"{path}: tensor {name!r} overlaps another or leaves a gap")
        cursor = end
    if cursor != size - base:
        raise ValueError(f"{path}: the data section does not match its header")
    storage = (
        torch.UntypedStorage.from_file(path, shared = False, nbytes = size) if size > base else None
    )
    tensors = {}
    for start, end, name, dtype, shape, itemsize in entries:
        offset = base + start
        if start == end or storage is None:
            tensors[name] = torch.empty(shape, dtype = dtype)
        elif offset % itemsize == 0:
            tensors[name] = torch.empty(0, dtype = dtype).set_(storage, offset // itemsize, shape)
        else:
            # safetensors packs tensors back to back, so one can be misaligned for its dtype: copy it.
            raw = torch.empty(0, dtype = torch.uint8).set_(storage, offset, (end - start,))
            tensors[name] = raw.clone().view(dtype).reshape(shape)
    return dict(metadata), tensors


def load_prequant_safetensors(
    path: str,
    *,
    device: str = "cpu",
    mmap: bool = False,
) -> dict:
    """Same dict as the pickle path so downstream validation is shared and the formats cannot drift."""
    helpers = _torchao_helpers()
    if helpers is None:
        raise RuntimeError(
            "this install cannot read a safetensors pre-quant checkpoint: torchao >= "
            f"{'.'.join(str(p) for p in MIN_TORCHAO_VERSION)} is required for "
            "torchao.prototype.safetensors.safetensors_support"
        )
    _, unflatten = helpers
    if mmap and str(device) == "cpu":
        raw, tensors = _mapped_tensors(path)
    else:
        from safetensors import safe_open
        with safe_open(path, framework = "pt", device = device) as handle:
            raw = dict(handle.metadata() or {})
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    fmt = raw.get(UNSLOTH_FORMAT_KEY)
    if not fmt:
        raise ValueError(
            f"{path} is a safetensors file but not an Unsloth pre-quant checkpoint "
            f"(no {UNSLOTH_FORMAT_KEY!r} in its header)"
        )
    metadata = json.loads(raw.get(UNSLOTH_METADATA_KEY) or "{}")

    # Lifted out BEFORE unflatten: torchao would rsplit these on ".".
    roots = {
        key[len(UNSLOTH_ROOT_PREFIX) :]: tensors.pop(key)
        for key in [k for k in tensors if k.startswith(UNSLOTH_ROOT_PREFIX)]
    }

    from .prequant_native import native_rebuild_enabled, native_unflatten

    state_dict = native_unflatten(tensors, raw, path = path) if native_rebuild_enabled() else None
    reader = "native" if state_dict is not None else "torchao"
    if state_dict is None:
        # A newer torchao's field an older constructor lacks (how a published int8 file broke): dropped if inert.
        raw = _header_without_unconstructible_fields(unflatten, tensors, raw, path = path)

        rebuilt = unflatten(tensors, raw)
        state_dict = _first(rebuilt)
        leftover = rebuilt[1] if isinstance(rebuilt, tuple) and len(rebuilt) > 1 else None
        if leftover:
            raise ValueError(
                f"{path} has {len(leftover)} tensor(s) its header does not account for "
                f"(e.g. {sorted(leftover)[0]!r}); the checkpoint is incomplete or was edited"
            )
    if roots:
        state_dict.update(roots)
    return {"format": str(fmt), "state_dict": state_dict, "metadata": metadata, "reader": reader}


def scheme_is_flattenable(quant_config: Any, *, features: int = 512) -> Optional[bool]:
    """Probes a tiny Linear, since importable helpers are not enough: torchao 0.17 cannot flatten int8."""
    helpers = _torchao_helpers()
    if helpers is None:
        return False
    try:
        import torch
        from torchao.quantization import quantize_

        probe = torch.nn.Linear(features, features, bias = False)
        quantize_(probe, quant_config)
    except Exception:  # noqa: BLE001 - an unprobeable config is not evidence of anything
        return None
    flatten, _ = helpers
    try:
        flatten(probe.state_dict())
    except ValueError as exc:
        if "Unsupported tensor type" in str(exc):
            return False
        return None
    except Exception:  # noqa: BLE001
        return None
    return True
