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
import re
from typing import Any, Optional

SAFETENSORS_SUFFIX = ".safetensors"

# Header keys carrying what the pickle kept in the top-level dict. Namespaced so they cannot collide
# with torchao's, which are ``tensor_names`` plus one key per tensor FQN.
UNSLOTH_FORMAT_KEY = "unsloth_format"
UNSLOTH_METADATA_KEY = "unsloth_metadata"
# Root-level PLAIN tensors, carried beside torchao's flat set rather than through it. Written under
# this prefix and listed under this header key, so the reader restores their bare names and torchao
# never sees a key it would try to rsplit on ".".
UNSLOTH_ROOT_PREFIX = "unsloth_root::"
UNSLOTH_ROOT_KEYS_KEY = "unsloth_root_tensors"

# The torchao release that first shipped the flatten/unflatten pair under this import path. Below it
# the helpers are absent and a safetensors artifact simply cannot be read, so the loader says so and
# falls back rather than guessing at the layout.
MIN_TORCHAO_VERSION = (0, 16)


def is_safetensors_checkpoint(name: Optional[str]) -> bool:
    """Whether ``name`` (a filename or a path) names a safetensors pre-quant artifact."""
    return bool(name) and str(name).lower().endswith(SAFETENSORS_SUFFIX)


def _torchao_helpers() -> Optional[tuple]:
    """``(flatten, unflatten)`` from torchao, or None when this install cannot do it.

    Imported lazily and by feature rather than by version string: the module moving out of
    ``prototype`` is a rename we should follow silently, and a version parse is not evidence the
    symbols exist. Never raises -- an install without them has no safetensors support, which is a
    fallback, not an error.

    The stub comes FIRST because importing by feature is not enough against it. On Windows ROCm
    ``install_torchao_windows_rocm_stub`` installs a meta-path finder that answers every
    ``torchao.*`` import with fabricated callables, so the import below succeeds and hands back two
    names that return None. Planning would then read safetensors as supported, drop the dense
    shards, and the load would unflatten nothing. The pickle probe already asks ``is_stubbed`` for
    the same reason (``diffusion_prequant._register_prequant_safe_globals``).
    """
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


def _first(value: Any) -> Any:
    """Both torchao helpers return a 2-tuple whose first element is the dict we want.

    Unpacking blindly is the bug this exists to prevent: ``flatten_tensor_state_dict`` returns
    ``(tensors, metadata)`` and ``unflatten_tensor_state_dict`` returns ``(state_dict, _)``, and a
    tuple handed on to ``load_state_dict`` fails with a type error that says nothing about torchao.
    A release that returns the bare dict instead is handled by the same line.
    """
    return value[0] if isinstance(value, tuple) else value


def _root_level_keys(state_dict: Any) -> list:
    """Keys with no ``.`` in them, in the order the state dict gives them."""
    try:
        keys = list(state_dict.keys())
    except Exception:  # noqa: BLE001 - not a mapping is the caller's problem, not this check's
        return []
    return [k for k in keys if "." not in str(k)]


def unsupported_state_dict_keys(state_dict: Any) -> list:
    """Root-level keys this container cannot round-trip: the QUANTIZED ones.

    ``unflatten_tensor_state_dict`` does ``tensor_name.rsplit(".", 1)`` to split a key into module
    fqn and weight name, so a root-level entry (``x_pad_token`` rather than ``embed.weight``) cannot
    go through torchao at all. Plain tensors do not need to: they are written beside the flat set
    under ``UNSLOTH_ROOT_PREFIX`` and restored on read, which is what z-image needs, since
    ``ZImageTransformer2DModel`` holds ``x_pad_token`` and ``cap_pad_token`` at the root and every
    safetensors build of it was refused after the download and the GPU quantization had finished.

    A root-level TENSOR SUBCLASS is still refused: reconstructing one is exactly the job of the
    flatten pair that cannot address it, so there is nothing to carry it in.
    """
    roots = _root_level_keys(state_dict)
    if not roots:
        return []
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch means no way to tell the two apart; refuse as before
        return roots
    return [k for k in roots if type(state_dict[k]) is not torch.Tensor]


def save_prequant_safetensors(path: str, *, fmt: str, state_dict: Any, metadata: Any) -> None:
    """Write ``state_dict`` (quantized, tensor subclasses and plain tensors alike) to ``path``.

    ``fmt`` and ``metadata`` land in the header beside torchao's own description of every tensor.
    Raises when the install cannot serialize this way: the caller is an offline builder that was
    ASKED for safetensors, and silently writing a pickle under a ``.safetensors`` name would be the
    worst of both.
    """
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

    # Root-level plain tensors go beside the flat set, not through it. Split BEFORE flatten so
    # torchao only ever sees dotted keys, and the pair stays exactly the one it supports.
    roots = _root_level_keys(state_dict)
    quantizable = (
        {k: v for k, v in state_dict.items() if k not in set(roots)} if roots else state_dict
    )

    flatten, _ = helpers
    try:
        flat, torchao_metadata = flatten(quantizable)
    except ValueError as exc:
        # Only the NEW torchao tensor subclasses can be flattened. The one that bites in practice is
        # int8: through torchao 0.17 ``Int8DynamicActivationInt8WeightConfig`` still produces the
        # legacy ``LinearActivationQuantizedTensor`` over an ``AffineQuantizedTensor``, which has no
        # flatten support, while 0.18 produces ``Int8Tensor``, which does. Reading is NOT affected
        # (0.17 reconstructs an ``Int8Tensor`` file fine), so this is a constraint on the machine
        # that BUILDS an artifact, and it deserves to say so rather than surface as a bare
        # "Unsupported tensor type" from inside torchao.
        if "Unsupported tensor type" not in str(exc):
            raise
        raise ValueError(
            f"{exc}. This torchao ({_torchao_version() or 'unknown'}) still quantises to a legacy "
            "tensor subclass that cannot be written to safetensors; torchao >= 0.18 produces the "
            "flattenable subclasses for every scheme Unsloth ships. Upgrade torchao to build this "
            "artifact, or write it as a .pt checkpoint."
        ) from exc

    # torchao hands back metadata ALREADY JSON-encoded, i.e. a str -> str dict ready for safetensors.
    # Encoding it a second time round-trips perfectly against a symmetric reader and is still wrong:
    # ``is_metadata_torchao`` then rejects the header and every transformers / diffusers loader
    # refuses the file. The break only shows up on someone else's machine, so pass it through.
    header = dict(torchao_metadata or {})
    header[UNSLOTH_FORMAT_KEY] = str(fmt)
    header[UNSLOTH_METADATA_KEY] = json.dumps(metadata or {}, default = str)
    if roots:
        flat = dict(flat)
        for key in roots:
            # ``contiguous`` because safetensors refuses a view, and a root buffer sliced out of a
            # larger allocation is exactly the shape that arrives as one.
            flat[f"{UNSLOTH_ROOT_PREFIX}{key}"] = state_dict[key].contiguous()
        header[UNSLOTH_ROOT_KEYS_KEY] = json.dumps([str(k) for k in roots])
    save_file(flat, path, metadata = header)


def read_prequant_header(path: str) -> Optional[dict]:
    """``{"format": ..., "metadata": {...}}`` read from the header alone, or None.

    No tensor is touched, so this is the cheap way to answer "what scheme is this artifact, and is
    it even ours" on a multi-GB file. None means "not an Unsloth safetensors pre-quant checkpoint",
    which every caller treats as unknown rather than as an error.
    """
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


# torchao reports a field its constructor will not take through this exact phrasing, wrapped in its
# own "Failed to create instance of <Class>" message.
_UNEXPECTED_KWARG = re.compile(r"unexpected keyword argument '([^']+)'")

# Values that mean "this field is not doing anything", so an older torchao that has never heard of
# the field behaves identically without it. Anything else is a real setting and must not be dropped.
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
        if key in (UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY, UNSLOTH_ROOT_KEYS_KEY):
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
    """``raw``, minus fields THIS torchao cannot construct, when dropping them changes nothing.

    torchao's tensor subclasses are reconstructed from their serialized dataclass kwargs, so a
    checkpoint written by a newer release carries fields an older constructor rejects outright:
    0.18 added ``reduce_range`` to ``QuantizeTensorToInt8Kwargs``, and a 0.17 install answers the
    published Qwen-Image-2.1 int8 artifact with ``Failed to create instance of
    QuantizeTensorToInt8Kwargs: ... unexpected keyword argument 'reduce_range'``. The loader then
    reports no usable checkpoint and the dense bf16 denoiser is downloaded and quantized at runtime,
    which is the whole saving gone, for a field the file records as ``false``.

    Driven by the error rather than by a version table: the failure names the field, so only the
    field that actually blocks this install is touched, and a release that adds a different one is
    handled with no code change. A dropped field carrying a NON-default value is refused instead,
    since silently loading weights under settings the file did not ask for is worse than falling
    back. A dry run against ``unflatten`` is the only way to learn the name, and it is cheap: the
    constructor raises on the first tensor.
    """
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
                if key in (UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY, UNSLOTH_ROOT_KEYS_KEY):
                    pruned[key] = value
                    continue
                try:
                    parsed = json.loads(value)
                except Exception:  # noqa: BLE001 - not every header entry is JSON
                    pruned[key] = value
                    continue
                pruned[key] = json.dumps(_drop_field(parsed, name, removed))
            if not removed:
                # Tensor field, e.g. 0.18's all-zero ``zero_point`` that 0.16 cannot take.
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
        # Worth saying out loud: the artifact was built by a newer torchao than this one.
        print(
            f"note: {path} carries {', '.join(sorted(set(dropped)))} from a newer torchao; "
            f"the field is inert here and was ignored",
            flush = True,
        )
    return header


def load_prequant_safetensors(path: str, *, device: str = "cpu") -> dict:
    """Read ``path`` into the SAME dict shape the pickle path returns.

    Returning ``{"format", "state_dict", "metadata"}`` rather than a new type is deliberate: every
    validation and load step downstream (scheme / base / min_features / fast_accum checks, the fp8
    activation-floor probe that reads the reconstructed tensors, the rotation biconditional, the
    kernel-preference pin) then runs unchanged on both containers, so the two formats cannot drift
    into having different acceptance rules.
    """
    helpers = _torchao_helpers()
    if helpers is None:
        raise RuntimeError(
            "this install cannot read a safetensors pre-quant checkpoint: torchao >= "
            f"{'.'.join(str(p) for p in MIN_TORCHAO_VERSION)} is required for "
            "torchao.prototype.safetensors.safetensors_support"
        )
    from safetensors import safe_open

    _, unflatten = helpers
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

    # Lifted out BEFORE unflatten: torchao would rsplit these on "." and fail, and they are plain
    # tensors that never needed it. Keyed off the prefix rather than the header list, so a file
    # written by a build that recorded one and not the other still reads; the list is the order.
    roots = {
        key[len(UNSLOTH_ROOT_PREFIX) :]: tensors.pop(key)
        for key in [k for k in tensors if k.startswith(UNSLOTH_ROOT_PREFIX)]
    }

    # A newer torchao can record a field an older one's constructor does not take, which is how a
    # published int8 checkpoint stopped loading. Dropped here when it is inert; see the helper.
    raw = _header_without_unconstructible_fields(unflatten, tensors, raw, path = path)

    # torchao reads its OWN keys out of the same header; ours are namespaced and simply ignored. The second element
    # is what it could NOT account for: a subclass missing one of its parts (a truncated or hand-edited file) is
    # skipped there rather than raised, which would reach load_state_dict as a bare missing-key error saying nothing
    # about the artifact. Name it here instead.
    rebuilt = unflatten(tensors, raw)
    state_dict = _first(rebuilt)
    leftover = rebuilt[1] if isinstance(rebuilt, tuple) and len(rebuilt) > 1 else None
    if leftover:
        raise ValueError(
            f"{path} has {len(leftover)} tensor(s) its header does not account for "
            f"(e.g. {sorted(leftover)[0]!r}); the checkpoint is incomplete or was edited"
        )
    if roots:
        # Back under their bare names, so the caller's load_state_dict sees the shape the model
        # declares. A dotted key can never collide with one of these.
        state_dict.update(roots)
    return {"format": str(fmt), "state_dict": state_dict, "metadata": metadata}


def scheme_is_flattenable(quant_config: Any, *, features: int = 512) -> Optional[bool]:
    """Whether THIS torchao can flatten what ``quant_config`` quantizes a weight to.

    ``safetensors_prequant_supported`` answers a different question: the helpers import, the package
    is there. That is necessary and not sufficient, and int8 is the case where the gap bites. Through
    torchao 0.17 ``Int8DynamicActivationInt8WeightConfig`` still produces a
    ``LinearActivationQuantizedTensor`` over an ``AffineQuantizedTensor``, which ``flatten`` refuses;
    0.18 produces ``Int8Tensor``, which it accepts. Without this the builder passes its preflight,
    downloads the model, spends the hours of GPU quantization, and only then discovers it cannot
    write the file it was asked for.

    Probed on one tiny CPU Linear rather than read off a version string, for the same reason
    ``_torchao_helpers`` imports by feature: the constraint is what the installed release actually
    produces. ``None`` means the probe itself could not run (no torch, a config this torchao will not
    apply to a bare Linear), which callers must treat as "proceed": refusing a build because the
    probe was unavailable would be worse than the late failure it exists to prevent.
    """
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
