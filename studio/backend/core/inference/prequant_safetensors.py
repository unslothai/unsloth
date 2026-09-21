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
from typing import Any, Optional

SAFETENSORS_SUFFIX = ".safetensors"

# Header keys carrying what the pickle kept in the top-level dict. Namespaced so they cannot collide
# with torchao's, which are ``tensor_names`` plus one key per tensor FQN.
UNSLOTH_FORMAT_KEY = "unsloth_format"
UNSLOTH_METADATA_KEY = "unsloth_metadata"

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
    """
    try:
        from torchao.prototype.safetensors.safetensors_support import (
            flatten_tensor_state_dict,
            unflatten_tensor_state_dict,
        )
    except Exception:  # noqa: BLE001 - torchao absent, too old, or the module moved
        return None
    return flatten_tensor_state_dict, unflatten_tensor_state_dict


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


def unsupported_state_dict_keys(state_dict: Any) -> list:
    """Keys torchao's unflatten cannot round-trip: the ones with no ``.`` in them.

    ``unflatten_tensor_state_dict`` does ``tensor_name.rsplit(".", 1)`` to split a key into module
    fqn and weight name, so a ROOT-level parameter or buffer (``pos_embed`` rather than
    ``embed.weight``) raises ``ValueError: not enough values to unpack`` when the file is read back.
    Saving one would produce an artifact that writes cleanly and can never be loaded, so the builder
    refuses up front and says which keys are the problem. Every DiT Unsloth ships today is dotted
    throughout; this exists so a future one fails at build time rather than in a user's loader.
    """
    try:
        keys = list(state_dict.keys())
    except Exception:  # noqa: BLE001 - not a mapping is the caller's problem, not this check's
        return []
    return [k for k in keys if "." not in str(k)]


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
            "these state dict keys have no '.' and cannot be read back by torchao's unflatten, "
            f"so a safetensors checkpoint carrying them would never load: {undotted[:8]}"
            + (f" (+{len(undotted) - 8} more)" if len(undotted) > 8 else "")
            + ". Write this build as a .pt checkpoint instead."
        )

    flatten, _ = helpers
    try:
        flat, torchao_metadata = flatten(state_dict)
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
    return {"format": str(fmt), "state_dict": state_dict, "metadata": metadata}
