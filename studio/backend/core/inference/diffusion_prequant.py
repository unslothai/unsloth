# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load a *pre-quantized* transformer instead of quantising a dense one on the GPU.

The runtime transformer_quant path loads the dense bf16 transformer and ``quantize_``s it in place,
materialising the full bf16 weights on the GPU first (~2x the GGUF peak, plus the full bf16
download). When a transformer was already quantised and saved
(``scripts/build_prequant_checkpoint.py``), this loads those weights directly: build the skeleton on
``meta`` (``init_empty_weights`` + ``from_config``), ``load_state_dict(assign=True)`` the quantized
state dict (subclass tensors assigned, not copied, so dense bf16 never touches the GPU), then move
to device. Measured (B200, Z-Image fp8): GPU load peak 12.9 -> 6.3 GB, download 12 -> 6.28 GB,
output bit-identical (LPIPS 0.0). The checkpoint carries the same scheme + ``min_features`` as the
runtime path, so the result matches quantising on the fly.

Two containers, one dict. Historically the artifact could only be a ``torch.save`` pickle, because
torchao's weight subclasses are wrapper tensors and safetensors stores flat ones; that pickle is read
under ``weights_only`` plus the constructor ALLOWLIST below, never as a free one, since it is a
mutable remote file reached by loads that never asked for a scheme (auto resolves an unset precision
to a hosted checkpoint) and "first-party repo" cannot stand in for that restriction. torchao >= 0.16
can flatten those subclasses, so an artifact may now also be ``.safetensors`` (see
``prequant_safetensors``), which needs no allowlist at all and answers the validation questions from
its header. ``_load_prequant_checkpoint`` returns the same ``{"format", "state_dict", "metadata"}``
for both, so every check in this module applies to them identically and the two cannot drift apart.

Best-effort and lazily imported: a missing / mismatched / unreadable checkpoint returns None and the
caller falls back to dense-quantise (then GGUF). Inert with nothing configured.
"""

from __future__ import annotations

import contextvars
import functools
import re as _re
import threading as _threading
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Optional

from .diffusion_nvfp4_flag import nvfp4_blocked

# torch.save dict layout tag; bump on an on-disk change so old/foreign artifacts are rejected
PREQUANT_FORMAT = "unsloth_prequant_transformer_state_dict_v1"

# v2 = rotated weights; old builds must refuse it (they would render wrong pixels).
# v2 MUST declare a rotation and v1 must NOT.
PREQUANT_FORMAT_ROTATED = "unsloth_prequant_transformer_state_dict_v2"

# Own tag, else an older build reads it as whole-model nvfp4. v3 MUST declare a policy, v1/v2 must NOT.
PREQUANT_FORMAT_POLICY = "unsloth_prequant_transformer_state_dict_v3"

PREQUANT_FORMATS = (PREQUANT_FORMAT, PREQUANT_FORMAT_ROTATED, PREQUANT_FORMAT_POLICY)

DEFAULT_PREQUANT_COMPONENT = "transformer"


def prequant_format_for(metadata: Any) -> str:
    """The format tag to stamp for ``metadata``; v2 and v3 share one tag slot, so declaring both is refused."""
    from .diffusion_convrot import declares_rotation
    from .diffusion_nvfp4_policy import declares_policy

    rotated = declares_rotation(metadata)
    policy = declares_policy(metadata)
    if rotated and policy:
        raise ValueError(
            "a pre-quant checkpoint cannot declare both an activation rotation and a per-layer "
            "nvfp4 policy: the format tag can only warn older builds about one of them"
        )
    if policy:
        return PREQUANT_FORMAT_POLICY
    return PREQUANT_FORMAT_ROTATED if rotated else PREQUANT_FORMAT


# Request-supplied paths load only inside an operator allowlist (weights_only either way).
ALLOW_LOCAL_PREQUANT_PATH_ENV = "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH"

# Extra pickle constructors allowed under weights_only=True. Both the pickle name and the
# real __module__ are listed; names a torchao lacks are skipped. New schemes add theirs.
_PREQUANT_SAFE_GLOBALS: tuple[tuple[str, str], ...] = (
    ("torchao.dtypes.affine_quantized_tensor", "AffineQuantizedTensor"),
    ("torchao.dtypes.uintx.plain_layout", "PlainAQTTensorImpl"),
    ("torchao.dtypes.utils", "PlainLayout"),
    ("torchao.quantization.linear_activation_quantized_tensor", "LinearActivationQuantizedTensor"),
    ("torchao.quantization.quant_api", "_int8_symm_per_token_reduced_range_quant"),
    ("torchao.quantization.quant_primitives", "ZeroPointDomain"),
    ("torchao.quantization.quant_primitives", "MappingType"),
    ("torchao.quantization", "Float8Tensor"),
    ("torchao.quantization.quantize_.workflows.float8.float8_tensor", "Float8Tensor"),
    (
        "torchao.quantization.quantize_.workflows.float8.float8_tensor",
        "QuantizeTensorToFloat8Kwargs",
    ),
    ("torchao.quantization.quantize_.common.kernel_preference", "KernelPreference"),
    ("torchao.quantization.granularity", "PerRow"),
    ("torchao.quantization.granularity", "PerTensor"),
    ("torchao.float8.inference", "Float8MMConfig"),
    # mxfp8 / nvfp4: local bakes only; torchao registers them only on prototype import.
    ("torchao.prototype.mx_formats.mx_tensor", "MXTensor"),
    ("torchao.prototype.mx_formats.mx_tensor", "QuantizeTensorToMXKwargs"),
    ("torchao.prototype.mx_formats.config", "ScaleCalculationMode"),
    ("torchao.prototype.mx_formats.nvfp4_tensor", "NVFP4Tensor"),
    ("torchao.prototype.mx_formats.nvfp4_tensor", "QuantizeTensorToNVFP4Kwargs"),
    # torch.save stamps TorchVersion; without it every torchao checkpoint is refused.
    ("torch.torch_version", "TorchVersion"),
)


def _prequant_safe_globals() -> list:
    """``(object, pickled name)`` pairs to register; names this torchao lacks are skipped."""
    import importlib

    pairs = []
    for module, name in _PREQUANT_SAFE_GLOBALS:
        try:
            obj = getattr(importlib.import_module(module), name)
        except Exception:  # noqa: BLE001 -- a name this release does not ship is not allowed
            continue
        pairs.append((obj, f"{module}.{name}"))
    return pairs


_SAFE_GLOBALS_LOCK = _threading.Lock()
_SAFE_GLOBALS_REGISTERED: Optional[bool] = None
_RESOLVED_SAFE_GLOBALS: set = set()

_FP8_REQUIRED_GLOBALS: frozenset = frozenset(
    {
        "torchao.quantization.Float8Tensor",
        "torchao.quantization.quantize_.workflows.float8.float8_tensor."
        "QuantizeTensorToFloat8Kwargs",
        "torchao.quantization.quantize_.common.kernel_preference.KernelPreference",
        "torchao.quantization.granularity.PerRow",
        "torchao.float8.inference.Float8MMConfig",
        "torch.torch_version.TorchVersion",
    }
)

_SCHEME_REQUIRED_GLOBALS: dict = {
    "int8": frozenset(
        {
            "torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor",
            "torchao.dtypes.uintx.plain_layout.PlainAQTTensorImpl",
            "torchao.dtypes.utils.PlainLayout",
            "torchao.quantization.linear_activation_quantized_tensor."
            "LinearActivationQuantizedTensor",
            "torchao.quantization.quant_api._int8_symm_per_token_reduced_range_quant",
            "torchao.quantization.quant_primitives.ZeroPointDomain",
            "torch.torch_version.TorchVersion",
        }
    ),
    "fp8": _FP8_REQUIRED_GLOBALS,
    "mxfp8": frozenset(
        {
            "torchao.prototype.mx_formats.mx_tensor.MXTensor",
            "torchao.prototype.mx_formats.mx_tensor.QuantizeTensorToMXKwargs",
            "torchao.prototype.mx_formats.config.ScaleCalculationMode",
            "torchao.quantization.quantize_.common.kernel_preference.KernelPreference",
        }
    ),
    "nvfp4": frozenset(
        {
            "torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor",
            "torchao.prototype.mx_formats.nvfp4_tensor.QuantizeTensorToNVFP4Kwargs",
        }
    )
    | _FP8_REQUIRED_GLOBALS,
}


def _tuple_safe_globals_supported() -> bool:
    """Whether this torch's ``add_safe_globals`` understands ``(object, name)`` pairs (2.6+). Asked
    by VERSION rather than by trying it: 2.4/2.5 accept the pairs silently and only fail later,
    in ``_get_user_allowed_globals``, which reads ``f.__module__`` off every entry of a
    PROCESS-WIDE list, so a tuple left there breaks every other weights_only load in Unsloth.
    Nothing is registered unless the answer here is yes."""
    try:
        import torch
        parts = str(torch.__version__).split("+")[0].split(".")
        return (int(parts[0]), int(parts[1])) >= (2, 6)
    except Exception:  # noqa: BLE001 -- an unreadable version is not a supported one
        return False


def _register_prequant_safe_globals() -> bool:
    """Register the allowlist ONCE, process-wide and permanently. True when the load can run.

    Not the ``safe_globals`` context manager, deliberately: it adds on entry and REMOVES on exit
    against a process-wide table, so two overlapping reads (a download-plan probe beside a load;
    both arrive on the route's thread pool) let whichever finishes first strip the allowlist out
    from under the other's ``torch.load``, failing a good checkpoint and dropping it to dense.
    Adding once and never removing has no such window.

    The widening this costs is small and bounded: other ``weights_only`` loads in the process also
    accept these torch/torchao tensor constructors, which build tensors and nothing else. A pickle
    naming ANY global is still refused.

    Registration takes ``(object, name)`` pairs so a re-exported class is registered under the name
    the pickle records, and that form is version-checked BEFORE anything is registered (see
    ``_tuple_safe_globals_supported``). Below 2.6 nothing is registered and
    ``restricted_prequant_load_supported`` tells planning to stop offering pre-quant sources at all.
    Answered once and memoised, including the failure.
    """
    global _SAFE_GLOBALS_REGISTERED

    if _SAFE_GLOBALS_REGISTERED is not None:
        return _SAFE_GLOBALS_REGISTERED
    with _SAFE_GLOBALS_LOCK:
        if _SAFE_GLOBALS_REGISTERED is not None:
            return _SAFE_GLOBALS_REGISTERED
        ok = False
        try:
            from core._torchao_stub import is_stubbed

            import torch

            add = getattr(torch.serialization, "add_safe_globals", None)
            # A stubbed torchao (Windows ROCm) fabricates every class, so never register it.
            if add is not None and not is_stubbed("torchao") and _tuple_safe_globals_supported():
                pairs = _prequant_safe_globals()
                resolved = {name for _obj, name in pairs}
                if "torch.torch_version.TorchVersion" in resolved and any(
                    name.startswith("torchao.") for name in resolved
                ):
                    add(pairs)
                    _RESOLVED_SAFE_GLOBALS.update(resolved)
                    try:
                        torch._weights_only_unpickler._get_user_allowed_globals()
                    except AttributeError:  # noqa: BLE001 -- private; absence is not a failure
                        pass
                    ok = True
        except Exception:  # noqa: BLE001 -- no allowlist means no restricted load, never a raise
            ok = False
        _SAFE_GLOBALS_REGISTERED = ok
        return ok


def restricted_prequant_load_supported(
    scheme: Optional[str] = None, filename: Optional[str] = None
) -> bool:
    """Whether this install can read a pre-quant checkpoint, for ``scheme`` when one is named.

    Without the allowlist there is no safe way to open a pre-quant pickle and the loader refuses.
    Planning has to ask the same question BEFORE it sizes the load: a plan that counts on a 6 GB
    artifact, drops the dense shards and evicts the resident pipeline has nothing left when the
    refusal arrives. ``usable_prequant_source`` therefore answers None here, hosted and local alike,
    which is the same answer the loader will give.

    PER SCHEME, because the schemes do not share constructors and torchao does not retire them
    together: ``AffineQuantizedTensor`` and its layout carry every int8 checkpoint and are already
    deprecated upstream (pytorch/ao#2752), so a release that drops them while keeping
    ``Float8Tensor`` leaves fp8 loadable and int8 not. An unknown or unnamed scheme gets the floor
    answer the registration itself already checked.

    ``filename`` is the artifact the question is actually about, when the caller knows it. A
    safetensors checkpoint names no constructors, so there is nothing to allowlist and none of the
    above applies: it needs torchao's flatten/unflatten helpers and nothing else. Answering the
    pickle's question for a safetensors artifact would refuse a perfectly loadable file on exactly
    the installs the new container exists to unblock (a torch too old for ``add_safe_globals``, a
    torchao missing a scheme's constructors, the stubbed torchao on Windows ROCm).
    """
    from .prequant_safetensors import is_safetensors_checkpoint, safetensors_prequant_supported

    if is_safetensors_checkpoint(filename):
        return safetensors_prequant_supported()
    if not _register_prequant_safe_globals():
        return False
    key = (scheme or "").strip().lower()
    required = _SCHEME_REQUIRED_GLOBALS.get(key)
    if required is None or required <= _RESOLVED_SAFE_GLOBALS:
        return True
    if key == "int8":
        # torchao 0.18 deleted the v1 int8 classes every hosted INT8 pickle names; see prequant_legacy_int8.
        from .prequant_legacy_int8 import legacy_int8_decode_supported
        return legacy_int8_decode_supported()
    return False


def _torch_load_prequant(path: str, **kwargs: Any) -> Any:
    """``torch.load`` a pre-quant checkpoint under the allowlist above. ``weights_only = True`` is
    the whole point: a pickle that may name any global is remote code execution the moment the
    artifact is not the one that was published. Everything the format legitimately needs is
    allowlisted, so the restriction costs nothing and a mutated artifact raises
    ``UnpicklingError`` into the caller's dense fallback instead of running. A torch that cannot
    express the allowlist is refused outright, never reopened unrestricted."""
    import torch

    if not _register_prequant_safe_globals():
        raise RuntimeError(
            "this torch cannot register the pre-quant constructor allowlist (needs "
            "torch.serialization.add_safe_globals with (object, name) support, i.e. >= 2.6), so "
            "a pre-quant checkpoint cannot be deserialized without allowing arbitrary pickle "
            "globals"
        )
    from .prequant_legacy_int8 import (
        legacy_int8_decode_supported,
        load_legacy_int8_pickle,
        names_legacy_int8_class,
        rebuild_stray_standins,
    )

    try:
        ckpt = torch.load(path, weights_only = True, **kwargs)
    except Exception as exc:  # noqa: BLE001 - only the deleted-int8-classes refusal is retried
        if names_legacy_int8_class(exc) and legacy_int8_decode_supported():
            return load_legacy_int8_pickle(path, **kwargs)
        raise
    return rebuild_stray_standins(ckpt)


def _load_prequant_checkpoint(path: str, **kwargs: Any) -> Any:
    """Read a pre-quant checkpoint of EITHER container, as the same ``{"format", "state_dict",
    "metadata"}`` dict.

    Dispatched on the file extension rather than on content sniffing: the extension is what the
    resolver asked the Hub for, so a repo hosting both containers cannot serve one and be parsed as
    the other. ``kwargs`` are the pickle path's (``map_location``, ``mmap``); safetensors reads to
    CPU, which is where the pickle path maps too, and honours ``mmap`` the same way."""
    from .prequant_safetensors import is_safetensors_checkpoint, load_prequant_safetensors

    if is_safetensors_checkpoint(path):
        return load_prequant_safetensors(path, mmap = bool(kwargs.get("mmap")))
    return _torch_load_prequant(path, **kwargs)


_PREQUANT_MMAP_ENV = "UNSLOTH_DIFFUSION_PREQUANT_MMAP"


def prequant_mmap_enabled(destination: Any) -> bool:
    """Map only for a non-CPU destination: a host-placed module would keep the file open (Windows then cannot delete it)."""
    import os

    raw = (os.environ.get(_PREQUANT_MMAP_ENV) or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    dest = str(destination or "").strip().lower()
    return bool(dest) and not dest.startswith("cpu") and dest != "meta"


def _read_prequant_for(
    path: str,
    destination: Any,
    logger: Any = None,
) -> Any:
    """``_load_prequant_checkpoint`` mapped when enabled; anything the mapping cannot open is re-read in full."""
    if prequant_mmap_enabled(destination):
        try:
            return _load_prequant_checkpoint(path, map_location = "cpu", mmap = True)
        except Exception as exc:  # noqa: BLE001 - retried unmapped; the real error resurfaces there
            if logger is not None:
                logger.info(
                    "diffusion.prequant: mapped read failed (%s: %s); reading the checkpoint into memory",
                    type(exc).__name__,
                    str(exc).splitlines()[0][:200] if str(exc) else "",
                )
    return _load_prequant_checkpoint(path, map_location = "cpu")


_PREQUANT_TOGGLE_TOKENS = {"1", "true", "yes", "on", "0", "false", "no", "off"}


def _allowed_prequant_roots() -> list:
    """Operator-allowlisted directories whose pre-quant checkpoints may be unpickled.
    ``UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH`` = one or more dirs (``os.pathsep``-separated). A bare
    truthy/falsey toggle is ignored: it must name a directory, so no "allow all" mode."""
    import os

    raw = (os.environ.get(ALLOW_LOCAL_PREQUANT_PATH_ENV) or "").strip()
    if not raw:
        return []
    roots = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part or part.lower() in _PREQUANT_TOGGLE_TOKENS:
            continue
        try:
            roots.append(os.path.realpath(os.path.expanduser(part)))
        except Exception:  # noqa: BLE001 - a bad entry is simply not allowlisted
            continue
    return roots


def _local_prequant_path_allowed(path: str) -> bool:
    """True only when ``path`` resolves inside an allowlisted directory. ``realpath`` first so a
    symlink cannot point an allowlisted name at a file outside the allowed roots."""
    import os

    roots = _allowed_prequant_roots()
    if not roots:
        return False
    try:
        real = os.path.realpath(os.path.expanduser(path))
    except Exception:  # noqa: BLE001
        return False
    return any(real == r or real.startswith(r + os.sep) for r in roots)


def local_prequant_path_ready(path: str) -> bool:
    """True only when a local pre-quant path would actually load: inside an allowlisted root AND the
    file is present. The auto-policy planner checks this before budgeting the small prequant
    plan, so it never skips the dense shards for a path the loader will refuse (which would evict
    the resident pipeline then rebuild dense under an undersized plan -> OOM)."""
    import os

    if not _local_prequant_path_allowed(path):
        return False
    return os.path.isfile(os.path.expanduser(path))


# Operator mirror checked before the Hub: <root>/<owner>/<repo>/<filename>, os.pathsep roots.
PREQUANT_MIRROR_ENV = "UNSLOTH_DIFFUSION_PREQUANT_MIRROR"


def prequant_mirror_path(
    repo_id: Optional[str],
    name: Optional[str],
    readable: Any = None,
) -> Optional[str]:
    """``<mirror>/<repo_id>/<name>`` when a configured mirror holds that file, else None. Never raises.

    A converted safetensors copy of a requested pickle wins over the pickle itself (same weights, no
    constructor allowlist), but only when ``readable(<sibling name>)`` says this install can open it:
    otherwise a host without torchao's flatten helpers would be handed a file it then refuses, after
    planning had already counted it as cached. ``readable`` defaults to "yes"."""
    import os

    raw = (os.environ.get(PREQUANT_MIRROR_ENV) or "").strip()
    if not raw or not repo_id or not name:
        return None
    parts = [*str(repo_id).split("/"), *str(name).split("/")]
    if any(p in ("", ".", "..") or "\\" in p for p in parts):
        return None
    wanted = [parts]
    for suffix in (".pt", ".pth"):
        if parts[-1].endswith(suffix):
            sibling = parts[-1][: -len(suffix)] + ".safetensors"
            try:
                ok = readable is None or bool(readable(sibling))
            except Exception:  # noqa: BLE001 - an unanswerable question is a no
                ok = False
            if ok:
                wanted.insert(0, [*parts[:-1], sibling])
            break
    for root in raw.split(os.pathsep):
        root = root.strip()
        if not root:
            continue
        for rel in wanted:
            try:
                base = os.path.realpath(os.path.expanduser(root))
                candidate = os.path.realpath(os.path.join(base, *rel))
            except Exception:  # noqa: BLE001 - a bad root is simply not a mirror
                break
            if candidate.startswith(base.rstrip(os.sep) + os.sep) and os.path.isfile(candidate):
                return candidate
    return None


def _first_mirrored(repo_id: Optional[str], names: Sequence[str], readable: Any) -> Optional[str]:
    """The mirror's copy of the first of ``names`` it holds, checked for ALL names before any Hub call."""
    for name in names:
        hit = prequant_mirror_path(repo_id, name, readable)
        if hit is not None:
            return hit
    return None


PREFER_SAFETENSORS_ENV = "UNSLOTH_PREQUANT_PREFER_SAFETENSORS"

_PICKLE_SUFFIXES = (".pt", ".pth")
_logged_twin_choices: set = set()

# ``<stem>-ComfyUI.safetensors``: the twin of a hosted artifact both ComfyUI and Studio load; older builds never ask for it.
COMFY_PREQUANT_TAG = "-ComfyUI"
COMFY_PREQUANT_SUFFIX = COMFY_PREQUANT_TAG + ".safetensors"
# 0 drops the ComfyUI-format names from the chain (back to Studio's own containers only).
COMFY_PREQUANT_ENV = "UNSLOTH_DIFFUSION_PREQUANT_COMFY"
_ARTIFACT_SUFFIXES = (".safetensors",) + _PICKLE_SUFFIXES


def is_comfy_prequant_filename(name: Optional[str]) -> bool:
    """Whether ``name`` is the ComfyUI-format spelling of a hosted artifact."""
    return bool(name) and str(name).endswith(COMFY_PREQUANT_SUFFIX)


def _artifact_stem(name: Optional[str]) -> Optional[str]:
    """``<stem>`` of a hosted ``<stem>.safetensors`` / ``.pt`` / ``.pth`` / ``-ComfyUI.safetensors`` artifact;
    None for the legacy ``transformer_<scheme>.pt`` names, which have no ComfyUI twin."""
    if not name or "/" in str(name):
        return None
    name = str(name)
    if name.startswith("transformer_"):
        return None
    if name.endswith(COMFY_PREQUANT_SUFFIX):
        return name[: -len(COMFY_PREQUANT_SUFFIX)] or None
    for suffix in _ARTIFACT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] or None
    return None


def comfy_prequant_filename(name: Optional[str]) -> Optional[str]:
    """The ComfyUI-format twin of the hosted artifact ``name`` (``<stem>-ComfyUI.safetensors``), else None."""
    stem = _artifact_stem(name)
    if stem is None or is_comfy_prequant_filename(name):
        return None
    return stem + COMFY_PREQUANT_SUFFIX


def comfy_prequant_enabled() -> bool:
    import os
    return (os.environ.get(COMFY_PREQUANT_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


# Twins that hold Studio's weights bit for bit (int8) lead the chain. ComfyUI fp8 has one scale per tensor, a coarser
# second rounding of Studio's per-row fp8, so an fp8 twin trails Studio's own fp8 artifact.
COMFY_TWIN_LEADS = frozenset({"int8"})


def with_comfy_twins(names: Sequence[str], *, lead: bool = True) -> list:
    """``names`` with each artifact's ComfyUI-format twin added. ``lead``: right ahead of the artifact's first name,
    so the one file ComfyUI also loads is the default download and the containers older builds request stay behind
    it as the fallback when the repo does not host the twin; otherwise right behind the artifact's last name.
    Order-preserving, no duplicates."""
    names = [n for n in names if n]
    out: list = []
    for index, name in enumerate(names):
        twin = comfy_prequant_filename(name)
        stem = _artifact_stem(name)
        if twin and twin not in names and twin not in out:
            if lead:
                out.append(twin)
            elif not any(_artifact_stem(n) == stem for n in names[index + 1 :]):
                out.extend(n for n in (name, twin) if n not in out)
                continue
        if name not in out:
            out.append(name)
    return out


def _hub_name_cached(
    repo_id: Optional[str], name: Optional[str], root: Optional[str]
) -> Optional[str]:
    """``repo_id/name``'s path in ONE Hub cache root (None = huggingface_hub's own), else None. Never raises."""
    if not repo_id or not name:
        return None
    try:
        import os

        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 - no cache API to ask: treat as not cached
        return None
    try:
        hit = try_to_load_from_cache(repo_id, name, cache_dir = root)
    except Exception:  # noqa: BLE001 - a malformed cache entry is not a hit
        return None
    # a str is the cached path; a miss is None and a known-absent file is a sentinel object
    return hit if isinstance(hit, str) and os.path.isfile(hit) else None


def _twin_cache_roots(cache_dir: Optional[str]) -> tuple:
    """Every cache root a download could reuse: the caller's, the live setting, huggingface_hub's own."""
    roots: list = []
    live = None
    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        live = active_hf_hub_cache()
    except Exception:  # noqa: BLE001 - outside the Studio backend there is no live setting
        live = None
    for root in (cache_dir, live, None):
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def explain_container_choice(
    repo_id: Optional[str],
    chosen: Optional[str],
    candidates: Sequence[str],
    order: Sequence[str],
    readable: Any = None,
    logger: Any = None,
) -> Optional[str]:
    """Log once per file why a ``.pt`` was resolved although the chain names its safetensors twin
    (cached twin, twin not hosted yet, or safetensors unreadable here). Returns the reason. Never raises."""
    try:
        if not repo_id or not chosen or not chosen.lower().endswith(_PICKLE_SUFFIXES):
            return None
        stem = chosen[: chosen.rfind(".")]
        twin = stem + ".safetensors"
        if twin not in candidates:
            return None
        try:
            twin_ok = readable is None or bool(readable(twin))
        except Exception:  # noqa: BLE001
            twin_ok = False
        if not twin_ok:
            reason = "this install cannot read the .safetensors container"
        elif twin in order and chosen in order and order.index(chosen) < order.index(twin):
            return "cached"
        else:
            reason = "the repo does not host the .safetensors twin yet"
        key = (repo_id, chosen, reason)
        if key not in _logged_twin_choices:
            _logged_twin_choices.add(key)
            log = logger
            if log is None:
                import logging
                log = logging.getLogger(__name__)
            log.info(
                "diffusion.prequant_pickle: %s: using %s instead of %s because %s",
                repo_id,
                chosen,
                twin,
                reason,
            )
        return reason
    except Exception:  # noqa: BLE001 - a log line, never a failure
        return None


def prefer_cached_pickle_twins(
    repo_id: Optional[str],
    names: Sequence[str],
    *,
    readable: Any = None,
    cache_dir: Optional[str] = None,
    logger: Any = None,
    log: bool = True,
    roots: Optional[Sequence[Optional[str]]] = None,
) -> list:
    """``names`` with a cached, readable ``<stem>.pt``/``.pth`` moved ahead of its uncached
    ``<stem>.safetensors`` twin, so an existing user does not re-download the same weights.

    Same stem only: a different artifact never jumps the queue. The safetensors name stays right
    behind, so a twin since removed from the Hub 404s onto it. ``PREFER_SAFETENSORS_ENV`` disables.
    Pure cache lookups, never raises."""
    import os

    out = [n for n in names if n]
    if not repo_id or len(out) < 2:
        return out
    try:
        roots = tuple(roots) if roots is not None else _twin_cache_roots(cache_dir)

        def _cached(name: str) -> bool:
            return any(_hub_name_cached(repo_id, name, root) is not None for root in roots)

        def _ok(name: str) -> bool:
            try:
                return readable is None or bool(readable(name))
            except Exception:  # noqa: BLE001 - an unanswerable question is a no
                return False

        # A cached container of an artifact beats an uncached one of the SAME artifact: never download a second copy.
        for comfy in [n for n in out if is_comfy_prequant_filename(n)]:
            stem = _artifact_stem(comfy)
            twins = [n for n in out if n != comfy and _artifact_stem(n) == stem]
            if not twins:
                continue
            hit = next((t for t in twins if _ok(t) and _cached(t)), None)
            if _cached(comfy):
                if hit is None and _ok(comfy) and out.index(comfy) > out.index(twins[0]):
                    out.remove(comfy)
                    out.insert(out.index(twins[0]), comfy)
                continue
            if hit is None:
                continue
            out.remove(comfy)
            out.insert(out.index(twins[-1]) + 1, comfy)
            key = (repo_id, hit, comfy)
            if log and key not in _logged_twin_choices:
                _logged_twin_choices.add(key)
                sink = logger
                if sink is None:
                    import logging
                    sink = logging.getLogger(__name__)
                sink.info(
                    "diffusion.prequant_cached_twin: %s: using the cached %s; its ComfyUI-format twin %s is "
                    "not downloaded",
                    repo_id,
                    hit,
                    comfy,
                )
        if (os.environ.get(PREFER_SAFETENSORS_ENV) or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return out

        for st in [
            n
            for n in out
            if n.lower().endswith(".safetensors") and not is_comfy_prequant_filename(n)
        ]:
            stem = st[: -len(".safetensors")]
            twin = next(
                (
                    stem + suffix
                    for suffix in _PICKLE_SUFFIXES
                    if stem + suffix in out and _ok(stem + suffix) and _cached(stem + suffix)
                ),
                None,
            )
            if twin is None or _cached(st):
                continue
            out.remove(twin)
            out.insert(out.index(st), twin)
            key = (repo_id, twin)
            if log and key not in _logged_twin_choices:
                _logged_twin_choices.add(key)
                sink = logger
                if sink is None:
                    import logging
                    sink = logging.getLogger(__name__)
                sink.info(
                    "diffusion.prequant_cached_pickle: %s: using the cached %s; the %s twin is not "
                    "downloaded (set %s=1 to fetch it instead)",
                    repo_id,
                    twin,
                    st,
                    PREFER_SAFETENSORS_ENV,
                )
    except Exception:  # noqa: BLE001 - a preference, never a new failure
        return [n for n in names if n]
    return out


_LOCAL_FILES_ONLY = contextvars.ContextVar("unsloth_prequant_local_files_only", default = False)


def scoped_local_files_only(fn: Any) -> Any:
    """Run ``fn`` with its ``local_files_only`` kwarg visible to every cache probe it reaches, so a
    load that may not download plans like an offline one. Nested calls keep an outer True."""

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _LOCAL_FILES_ONLY.set(
            bool(kwargs.get("local_files_only")) or _LOCAL_FILES_ONLY.get()
        )
        try:
            return fn(*args, **kwargs)
        finally:
            _LOCAL_FILES_ONLY.reset(token)

    return _wrapper


def hub_offline() -> bool:
    """huggingface_hub's offline switch. Never raises."""
    try:
        from huggingface_hub import constants
        return bool(constants.HF_HUB_OFFLINE)
    except Exception:  # noqa: BLE001 - no hub library: nothing can be downloaded either
        return True


def hub_name_known_absent(
    repo_id: Optional[str],
    name: Optional[str],
    cache_dir: Optional[str] = None,
) -> bool:
    """True when a Hub cache root holds a ``.no_exist`` marker (a recorded 404) for ``name``. Never raises."""
    if not repo_id or not name:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 - no cache API: nothing is known
        return False
    for root in _twin_cache_roots(cache_dir):
        try:
            hit = try_to_load_from_cache(repo_id, name, cache_dir = root)
        except Exception:  # noqa: BLE001 - a malformed cache entry says nothing
            continue
        if hit is not None and not isinstance(hit, str):
            return True
    return False


def first_cached_as_resolved(
    repo_id: Optional[str],
    names: Sequence[str],
    *,
    is_cached: Any,
    online: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """The first cached name in ``names`` (resolver order) the load opens without downloading anything
    first, else None. Offline that is the first cached name. Online an uncached name ahead of it blocks
    (the resolver would fetch it) unless the cache recorded its 404; otherwise a cached INT8 ``.pt``
    reads as free while the load fetches an uncached INT8-ConvRot. A twin still ahead of the hit was
    not reordered (kill switch, unreadable pickle), so the load fetches it too.
    ``online=None`` reads huggingface_hub's offline switch and the load's ``local_files_only``. Never raises."""
    try:
        if online is None:
            online = not hub_offline() and not _LOCAL_FILES_ONLY.get()
        ahead: list = []
        for name in names:
            if not name:
                continue
            if not is_cached(name):
                ahead.append(name)
                continue
            if online:
                for other in ahead:
                    if not hub_name_known_absent(repo_id, other, cache_dir):
                        return None
            return name
    except Exception:  # noqa: BLE001 - a planning aid: unanswerable reads as not cached
        return None
    return None


@dataclass(frozen = True)
class PrequantSource:
    """Where a pre-quantized checkpoint lives. ``kind`` is "path" (a local file) or "repo" (Hub repo
    id in ``location`` + ``filename``; ``fallback_filenames`` are tried IN ORDER when the primary
    name is absent, covering the ``.pt`` container and repos still on the legacy
    transformer_<scheme>.pt)."""

    kind: str
    location: str
    filename: Optional[str] = None
    fallback_filenames: tuple[str, ...] = ()
    # Family-declared names are real; derived ones are guesses that may 404.
    declared_filenames: tuple[str, ...] = ()

    @property
    def fallback_filename(self) -> Optional[str]:
        """The first fallback. Kept because callers outside this module read it by name, and
        because it is what "the one other name to try" meant before the chain existed."""
        return self.fallback_filenames[0] if self.fallback_filenames else None

    @property
    def candidate_filenames(self) -> tuple[str, ...]:
        """Every name this source may resolve, best first. The single place that ordering lives, so
        the downloader, the cache probe and the planner cannot disagree about which file wins."""
        return tuple(n for n in (self.filename, *self.fallback_filenames) if n)


def candidate_filenames_of(source: Any) -> tuple[str, ...]:
    """``source``'s names, best first, for anything SHAPED like a source.

    Planning passes lightweight stand-ins that carry ``filename`` / ``fallback_filename`` and
    nothing else, so reading the property directly turns one of those into an AttributeError that
    is swallowed into "no prequant plan" and a silent dense fallback. Falling back to the two older
    attributes keeps every such caller working while the real dataclass answers the full chain."""
    names = getattr(source, "candidate_filenames", None)
    if names is None:
        names = (
            getattr(source, "filename", None),
            getattr(source, "fallback_filename", None),
        )
    return tuple(n for n in names if n)


def prequant_filename(scheme: str) -> str:
    """The legacy checkpoint filename for ``scheme`` inside a Hub repo."""
    return f"transformer_{scheme}.pt"


def prequant_repo_filename(
    repo_id: str,
    scheme: str,
    suffix: str = ".pt",
    *,
    component: Optional[str] = None,
) -> str:
    """Filename for ``scheme`` in ``repo_id``; a non-default ``component`` inserts -<component>-.

    ``suffix`` picks the container. It defaults to ``.pt`` so every existing caller keeps naming the
    artifact it already names; ``derived_prequant_filenames`` is what puts the safetensors spelling
    of the same name ahead of it."""
    model = repo_id.rsplit("/", 1)[-1]
    for drop in ("-fp8", "-int8", "-nvfp4", "-mxfp8", "-quantized"):
        if model.lower().endswith(drop):
            model = model[: -len(drop)]
            break
    part = (component or "").strip()
    if part and part != DEFAULT_PREQUANT_COMPONENT:
        return f"{model}-{part}-{scheme.upper()}{suffix}"
    return f"{model}-{scheme.upper()}{suffix}"


def derived_prequant_filenames(
    repo_id: str,
    scheme: str,
    *,
    component: Optional[str] = None,
) -> tuple[str, ...]:
    """The names to try for ``(repo_id, scheme)``, best first, safetensors AHEAD of the pickle.

    Preferring safetensors is a policy decision rather than a detail: it needs no constructor
    allowlist (so it loads on installs where the pickle is refused outright), it validates from its
    header before a weight is read, and it cannot carry pickle opcodes at all. Deriving the
    preference here rather than per family means a repo that gains a ``.safetensors`` sibling is
    picked up with no code change, and a repo that never does keeps resolving exactly what it
    resolves today, because the ``.pt`` names stay in the chain behind it.
    """
    names = (
        prequant_repo_filename(repo_id, scheme, ".safetensors", component = component),
        prequant_repo_filename(repo_id, scheme, ".pt", component = component),
    )
    part = (component or "").strip()
    if part and part != DEFAULT_PREQUANT_COMPONENT:
        return names
    return names + (prequant_filename(scheme),)


def resolve_prequant_source(
    fam: Any,
    scheme: str,
    *,
    path_override: Optional[str] = None,
    base_repo: Optional[str] = None,
    task: Optional[str] = None,
) -> Optional[PrequantSource]:
    """Resolve where the checkpoint for ``(fam, scheme)`` comes from.

    Priority: an explicit local ``path_override``; then the family's hosted repo for ``scheme``
    (variant-specific when ``base_repo`` names a base with its own baked checkpoint); then None,
    meaning no pre-quant and the caller quantises dense. Pure: no IO, no torch.

    ``task`` names the workflow / denoiser PARTITION the load is bringing up, for the families that
    host more than one under a single repo and scheme (MiniMax-H3's keyframe and reference
    denoisers). It only ever selects a more specific filename: unset, or set to a task the family
    declares nothing for, resolves exactly what it resolved before.

    Both names are repo-ROOT names. Every hosted prequant repo, image and video alike, keeps its
    checkpoints at the root, so there is no directory to prepend; a repo that nested them would 404
    on the primary AND on the fallback and the load would silently fall back to dense.
    """
    if nvfp4_blocked(scheme):
        return None
    override = (path_override or "").strip()
    if override:
        return PrequantSource(kind = "path", location = override, filename = None)
    preferred = None
    agnostic = None
    try:
        from .diffusion_families import family_prequant_filename, family_prequant_repo

        repo_id = family_prequant_repo(fam, scheme, base_repo = base_repo)
        preferred = family_prequant_filename(fam, scheme, task = task)
        agnostic = family_prequant_filename(fam, scheme) if task else preferred
    except Exception:  # noqa: BLE001 - a bad family object must not break the load
        repo_id = None
    if repo_id:
        derived = derived_prequant_filenames(repo_id, scheme)
        # A family may name a second artifact; it becomes primary. Task-specific names get NO
        # fallback: another partition's denoiser would pass every check with the wrong weights.
        task_specific = preferred is not None and preferred != agnostic
        if task_specific:
            return PrequantSource(
                kind = "repo",
                location = repo_id,
                filename = preferred,
                declared_filenames = (preferred,),
            )
        # Family-declared name first when there is one, then the derived chain, which puts the
        # safetensors spelling ahead of the pickle. Order-preserving dedup so a family that declares
        # exactly what the chain would derive does not make the downloader ask twice for it.
        # Declared names are the default repo's files; a variant repo gets its own spelling of each.
        variant_repo = _is_variant_prequant_repo(fam, repo_id)
        if variant_repo and preferred:
            declared = (prequant_repo_filename(repo_id, scheme, ".safetensors"),)
        else:
            declared = (preferred,) if preferred else ()
        # rotated artifact (hosted) first, only in its own repo; the plain chain stays behind it (offline, older caches)
        from .diffusion_transformer_quant import (
            convrot_prequant_filename,
            convrot_prequant_repo,
            convrot_prequant_variant_repo,
            int8_convrot_enabled,
        )

        fam_name = getattr(fam, "name", None)
        rotated = convrot_prequant_filename(scheme, fam_name)
        rotated_repo = convrot_prequant_repo(scheme, fam_name)
        if (
            variant_repo
            and convrot_prequant_variant_repo(scheme, fam_name, repo_id)
            and int8_convrot_enabled(fam_name)
        ):
            # A variant repo listed as hosting its own rotated build, named like its derived chain.
            declared = (prequant_repo_filename(repo_id, scheme, "-ConvRot.safetensors"),) + declared
        elif (
            rotated
            and rotated_repo
            and str(repo_id).strip().lower() == rotated_repo.lower()
            and int8_convrot_enabled(fam_name)
        ):
            declared = (rotated,) + declared
        names: list[str] = []
        for name in declared + derived:
            if name and name not in names:
                names.append(name)
        if comfy_prequant_enabled() and _comfy_prequant_family(fam):
            names = with_comfy_twins(names, lead = scheme in COMFY_TWIN_LEADS)
        return PrequantSource(
            kind = "repo",
            location = repo_id,
            filename = names[0],
            fallback_filenames = tuple(names[1:]),
            declared_filenames = declared,
        )
    return None


def _is_variant_prequant_repo(fam: Any, repo_id: Optional[str]) -> bool:
    """Whether ``repo_id`` is one of ``fam``'s per-base variant repos and not one of its default repos."""
    key = str(repo_id or "").strip().lower()
    if not key:
        return False
    try:
        defaults = {str(r).strip().lower() for _s, r in getattr(fam, "prequant_repos", ()) or ()}
        variants = {
            str(e[2]).strip().lower()
            for e in getattr(fam, "prequant_variant_repos", ()) or ()
            if len(e) == 3
        }
    except Exception:  # noqa: BLE001 - a malformed table keeps the declared names, today's behaviour
        return False
    return key in variants and key not in defaults


def _comfy_prequant_family(fam: Any) -> bool:
    """Whether ``fam``'s hosted artifacts may resolve to a ComfyUI-format twin: the image families, whose
    denoiser loads through ``load_prequantized_transformer`` as one whole module."""
    try:
        from .diffusion_families import DiffusionFamily
        return isinstance(fam, DiffusionFamily)
    except Exception:  # noqa: BLE001 - no registry: keep the chain as it was
        return False


_LOCAL_PREQUANT_SCHEME: dict[tuple[str, int, int], Optional[str]] = {}


def local_prequant_scheme(path: str) -> Optional[str]:
    """The scheme a local pre-quant checkpoint records, or None when it cannot be read.

    ``resolve_prequant_source`` hands back a ``path`` source for ANY override, whatever scheme was
    asked for: the file is never inspected. That is fine when the caller named the scheme, but under
    ``auto`` the ladder picks one and an override baked for a different scheme then reads as an
    available pre-quant. Planning skips staging the dense transformer, the loader reaches the same
    ``metadata.scheme`` check that runs at load time, refuses the file, and with no dense fallback
    the pick silently drops to GGUF.

    Cheap despite the file size: ``mmap`` plus ``map_location = "meta"`` maps the storages instead
    of reading them, so only the pickle structure is parsed (~1s on a 34 GB checkpoint). Cached on
    (path, mtime, size) because the auto ladder asks once per candidate scheme. Read under the same
    allowlisted ``weights_only`` load the loader uses, so probing a file that turns out not to be a
    checkpoint cannot execute anything either.

    A safetensors artifact answers from its HEADER, so the probe reads a few KB of JSON and no
    tensor at all, and needs neither the allowlist nor a torchao that can rebuild the subclasses.
    """
    import os

    try:
        real = os.path.expanduser(path)
        st = os.stat(real)
        # Nanoseconds: a same-size swap within one second would reuse the old scheme.
        key = (real, st.st_mtime_ns, int(st.st_size))
    except Exception:  # noqa: BLE001 -- unreadable is "unknown", handled by the caller
        return None
    if key in _LOCAL_PREQUANT_SCHEME:
        return _LOCAL_PREQUANT_SCHEME[key]
    from .prequant_safetensors import is_safetensors_checkpoint, read_prequant_header

    scheme: Optional[str] = None
    try:
        if is_safetensors_checkpoint(real):
            obj = read_prequant_header(real)
        else:
            obj = _torch_load_prequant(real, map_location = "meta", mmap = True)
        if isinstance(obj, dict) and obj.get("format") in PREQUANT_FORMATS:
            recorded = (obj.get("metadata") or {}).get("scheme")
            scheme = str(recorded) if recorded else None
    except Exception:  # noqa: BLE001 -- a checkpoint we cannot parse is "unknown", never a match
        scheme = None
    if scheme is None and is_safetensors_checkpoint(real):
        try:
            from .diffusion_comfy_quant import comfy_prequant_scheme, scan_comfy_quant
            scheme = comfy_prequant_scheme(scan_comfy_quant(real))
        except Exception:  # noqa: BLE001 -- unreadable is "unknown"
            scheme = None
    _LOCAL_PREQUANT_SCHEME[key] = scheme
    return scheme


def hosted_nvfp4_repo_ids() -> frozenset:
    """Lowercased repo ids any image/video family registers for nvfp4, read off the family tables."""
    from dataclasses import fields, is_dataclass

    from .diffusion_nvfp4_flag import is_nvfp4

    families: list = []
    for module in ("diffusion_families", "video_families"):
        try:
            mod = __import__(f"{__package__}.{module}", fromlist = ["_FAMILIES"])
            families.extend(getattr(mod, "_FAMILIES", ()) or ())
        except Exception:  # noqa: BLE001 - a table that fails to import names no repo
            continue
    repos = set()
    for fam in families:
        if not is_dataclass(fam):
            continue
        for field in fields(fam):
            value = getattr(fam, field.name, None)
            if not isinstance(value, tuple):
                continue
            for row in value:
                if not isinstance(row, tuple) or not row or not isinstance(row[-1], str):
                    continue
                if "/" in row[-1] and any(isinstance(x, str) and is_nvfp4(x) for x in row[:-1]):
                    repos.add(row[-1].strip().lower())
    return frozenset(repos)


_PREQUANT_PROBE_SUFFIXES = (".safetensors", ".pt", ".pth")
_PREQUANT_PROBE_LIMIT = 16


def _cached_snapshot_dirs(repo_id: str) -> list:
    """The local snapshot folders of Hub repo ``repo_id``, newest first. No network."""
    import os

    parts = repo_id.strip().strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        return []
    roots = []
    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        roots.append(active_hf_hub_cache())
    except Exception:  # noqa: BLE001
        pass
    try:
        from huggingface_hub import constants
        roots.append(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        pass
    found = []
    for root in dict.fromkeys(r for r in roots if r):
        snapshots = os.path.join(root, "models--" + "--".join(parts), "snapshots")
        try:
            entries = [os.path.join(snapshots, name) for name in os.listdir(snapshots)]
        except OSError:
            continue
        entries = [e for e in entries if os.path.isdir(e)]
        entries.sort(key = lambda e: os.path.getmtime(e), reverse = True)
        found.extend(entries)
    return found


def _artifacts_declare_nvfp4(folder: str) -> bool:
    """Whether a pre-quant artifact at the root of ``folder`` records ``scheme = nvfp4``."""
    import os

    from .diffusion_nvfp4_flag import is_nvfp4

    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return False
    if "model_index.json" in names:
        return False
    probed = 0
    for name in names:
        if not name.lower().endswith(_PREQUANT_PROBE_SUFFIXES):
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if is_nvfp4(local_prequant_scheme(path)):
            return True
        probed += 1
        if probed >= _PREQUANT_PROBE_LIMIT:
            break
    return False


def declares_nvfp4_checkpoint(model_path: Optional[str]) -> bool:
    """Whether ``model_path`` is an NVFP4 prequant checkpoint: recorded metadata first, else registry or name. Never raises."""
    import os

    raw = str(model_path or "").strip()
    if not raw:
        return False
    try:
        local = os.path.expanduser(raw)
        if os.path.isfile(local):
            if not local.lower().endswith(_PREQUANT_PROBE_SUFFIXES):
                return False
            from .diffusion_nvfp4_flag import is_nvfp4
            return is_nvfp4(local_prequant_scheme(local))
        if os.path.isdir(local):
            if _artifacts_declare_nvfp4(local):
                return True
        else:
            key = raw.strip("/").lower()
            if key in hosted_nvfp4_repo_ids():
                return True
            if any(_artifacts_declare_nvfp4(snap) for snap in _cached_snapshot_dirs(raw)):
                return True
    except Exception:  # noqa: BLE001 - a probe that fails is "not known to be NVFP4"
        pass
    return raw.rstrip("/\\").lower().endswith("-nvfp4")


FINGERPRINT_ALGO = "md5-packed-v1"

# Payload attrs per torchao class NAME, hash order; unlisted reads "not covered", never "equal".
_FINGERPRINT_PAYLOAD: dict = {
    "NVFP4Tensor": ("qdata", "scale", "per_tensor_scale"),
    "Float8Tensor": ("qdata", "scale"),
    "Int8Tensor": (
        "qdata",
        "scale",
        "zero_point",
        "act_quant_scale",
        "act_quant_zero_point",
        "act_pre_scale",
    ),
    "MXTensor": ("qdata", "scale"),
    "LinearActivationQuantizedTensor": ("original_weight_tensor",),
    "AffineQuantizedTensor": ("tensor_impl",),
    "PlainAQTTensorImpl": ("int_data", "scale", "zero_point"),
}


def _packed_bytes(tensor: Any, torch: Any) -> bytes:
    """``tensor``'s raw bytes via ``view(torch.uint8)``, exact for dtypes numpy cannot represent."""
    t = tensor.detach().contiguous()
    if t.dim() == 0:
        t = t.reshape(1)
    if t.dtype is not torch.uint8:
        t = t.view(torch.uint8)
    return t.cpu().numpy().tobytes()


def _hash_packed_payload(tensor: Any, digest: Any, torch: Any) -> bool:
    """Feed one weight's packed payload (with attribute names) into ``digest``; False when uncovered."""
    names = _FINGERPRINT_PAYLOAD.get(type(tensor).__name__)
    if names is None:
        return False
    hashed = False
    for name in names:
        value = getattr(tensor, name, None)
        if value is None:
            continue
        digest.update(name.encode("utf-8"))
        if type(value).__name__ in _FINGERPRINT_PAYLOAD:
            if not _hash_packed_payload(value, digest, torch):
                return False
            hashed = True
            continue
        digest.update(_packed_bytes(value, torch))
        hashed = True
    return hashed


def packed_weight_fingerprint(state_dict: Any, *, select: Any = None) -> dict:
    """md5 of every quantized weight's packed payload by fqn (``select`` narrows it); unknown ones go under ``skipped``."""
    import hashlib

    import torch

    modules: dict = {}
    skipped: list = []
    items = state_dict.items() if hasattr(state_dict, "items") else ()
    for key, tensor in items:
        if key != "weight" and not str(key).endswith(".weight"):
            continue
        if select is not None and not select(str(key)):
            continue
        digest = hashlib.md5()
        try:
            covered = _hash_packed_payload(tensor, digest, torch)
        except Exception:  # noqa: BLE001 -- an unreadable payload is uncovered, never a raise
            covered = False
        if covered:
            modules[key] = digest.hexdigest()
        else:
            skipped.append(key)
    return {
        "algo": FINGERPRINT_ALGO,
        "count": len(modules),
        "modules": modules,
        "skipped": skipped,
    }


FINGERPRINT_MODE_ENV = "UNSLOTH_PREQUANT_FINGERPRINT"
FINGERPRINT_MODES = ("full", "sample", "off")
FINGERPRINT_SAMPLE_RATE = 8


def _fingerprint_mode() -> str:
    """``full`` (default) / ``sample`` / ``off``. An unrecognised value reads as the default."""
    import os

    mode = (os.environ.get(FINGERPRINT_MODE_ENV) or "").strip().lower()
    return mode if mode in FINGERPRINT_MODES else "full"


def _fingerprint_sampled(fqn: str) -> bool:
    """Stable 1-in-8 by md5 of the name (``hash()`` is randomised per process)."""
    import hashlib
    return hashlib.md5(fqn.encode("utf-8")).digest()[0] % FINGERPRINT_SAMPLE_RATE == 0


def _verified_marker(path: Any, expected: dict) -> Optional[tuple]:
    """(marker file, identity) for a checkpoint whose full check passed before: same real file, size, mtime and
    recorded fingerprint. None when the file cannot be identified (then every load checks in full)."""
    import hashlib
    import json
    import os

    try:
        real = os.path.realpath(str(path))
        st = os.stat(real)
        from .diffusion_compile_cache import cache_root

        recorded = hashlib.md5(json.dumps(expected, sort_keys = True).encode("utf-8")).hexdigest()
        identity = {
            "path": real,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "fingerprint": recorded,
        }
        name = hashlib.md5(real.encode("utf-8")).hexdigest() + ".json"
        return cache_root() / "prequant_verified" / name, identity
    except Exception:  # noqa: BLE001 -- unidentifiable: check in full
        return None


def _already_verified(marker: Optional[tuple]) -> bool:
    import json
    if marker is None:
        return False
    try:
        return json.loads(marker[0].read_text(encoding = "utf-8")) == marker[1]
    except Exception:  # noqa: BLE001 -- absent or unreadable marker: check in full
        return False


def _remember_verified(marker: Optional[tuple]) -> None:
    import json
    import os

    if marker is None:
        return
    try:
        marker[0].parent.mkdir(parents = True, exist_ok = True)
        tmp = marker[0].with_suffix(".tmp")
        tmp.write_text(json.dumps(marker[1], sort_keys = True), encoding = "utf-8")
        os.replace(tmp, marker[0])
    except Exception:  # noqa: BLE001 -- the next load just checks in full again
        pass


def _verify_packed_fingerprint(
    state_dict: Any,
    metadata: Any,
    *,
    logger: Any = None,
    path: Any = None,
) -> bool:
    """Check the artifact's packed fingerprint: a mismatch drops to dense, a missing or uncomputable block passes.

    A full pass is remembered per file (real path, size, mtime, recorded fingerprint) in the Studio cache, so later
    loads of the same unchanged file skip the md5 of every weight; any change to the file checks in full again."""
    block = (metadata or {}).get("fingerprint")
    expected = (block or {}).get("modules") if isinstance(block, dict) else None
    if not expected:
        return True
    mode = _fingerprint_mode()
    marker = _verified_marker(path, expected) if path is not None and mode == "full" else None
    if _already_verified(marker):
        if logger is not None:
            logger.info(
                "diffusion.prequant: fingerprint verified on an earlier load of this unchanged file (%d weights)",
                len(expected),
            )
        return True
    if mode == "off":
        if logger is not None:
            logger.debug(
                "diffusion.prequant: fingerprint check disabled (%s=off)", FINGERPRINT_MODE_ENV
            )
        return True
    try:
        actual = (
            packed_weight_fingerprint(
                state_dict,
                select = _fingerprint_sampled if mode == "sample" else None,
            ).get("modules")
            or {}
        )
    except Exception as exc:  # noqa: BLE001 -- an uncomputable fingerprint checks nothing
        _warn(logger, "fingerprint", exc)
        return True
    if not actual:
        _warn(
            logger,
            "fingerprint",
            RuntimeError(
                f"this build recognised none of the {len(expected)} quantized weights the "
                "checkpoint fingerprinted (a torchao payload rename?); loading it unverified"
            ),
        )
        return True
    checked = [k for k in expected if mode != "sample" or _fingerprint_sampled(k)]
    differing = sorted(k for k in checked if actual.get(k) != expected[k])
    counted = mode != "sample" and len(actual) != len(expected)
    if not differing and not counted:
        if logger is not None:
            logger.info(
                "diffusion.prequant: fingerprint verified (%d of %d quantized weights, %s)",
                len(checked),
                len(expected),
                mode,
            )
        _remember_verified(marker)
        return True
    if logger is not None:
        logger.error(
            "diffusion.prequant: fingerprint MISMATCH (%d of %d checked weights differ, %d "
            "weights present vs %d recorded): %s. The checkpoint does not hold the bytes it was "
            "built with; falling back to the dense path",
            len(differing),
            len(checked),
            len(actual),
            len(expected),
            ", ".join(differing[:5]) or "counts only",
        )
    return False


def usable_prequant_source(
    fam: Any,
    scheme: str,
    *,
    path_override: Optional[str] = None,
    base_repo: Optional[str] = None,
) -> Optional[PrequantSource]:
    """``resolve_prequant_source``, but a local path counts only when the loader would accept it:
    inside the allowlist AND present on disk AND baked for THIS scheme. Otherwise resolves to None
    so memory planning falls back to dense-fit checks up front, instead of the loader refusing the
    path only after the resident pipeline was evicted and dense bf16 materialises under a plan that
    never budgeted for it (evict-then-OOM). Hosted-repo sources are unaffected.

    The scheme check matters most under ``auto``, which picks a scheme the user never named: an int8
    override must not read as an available fp8 pre-quant just because the file exists. A checkpoint
    whose scheme cannot be read is treated as not usable, matching every other unknown here, since
    the loader would reject it too.

    An install that cannot restrict the load has no usable source AT ALL, hosted included: the
    loader refuses every checkpoint there, and a plan that had already dropped the dense shards for
    one would find that out after the eviction. That question is asked of the RESOLVED names rather
    than of the scheme alone, because it has different answers for the two containers: a repo whose
    primary artifact is safetensors is usable on an install that could not open a pickle at all, and
    resolving first is what lets the source say so. Any one loadable name is enough, since the
    resolver tries them in order and the first that exists wins.
    """
    src = resolve_prequant_source(fam, scheme, path_override = path_override, base_repo = base_repo)
    if src is None:
        return None
    candidates = (
        [src.location]
        if getattr(src, "kind", None) == "path"
        else list(candidate_filenames_of(src))
    ) or [None]
    readable = [n for n in candidates if restricted_prequant_load_supported(scheme, n)]
    if not readable:
        return None
    # A derived safetensors name is a guess: require it declared or cached before planning on it.
    from .prequant_safetensors import is_safetensors_checkpoint

    if src.kind == "repo" and all(is_safetensors_checkpoint(n) for n in readable):
        declared = set(getattr(src, "declared_filenames", ()) or ())
        if (
            not any(n in declared for n in readable)
            and cached_checkpoint_path(src, names = readable, online = False) is None
        ):
            return None
    if src.kind == "path":
        if not local_prequant_path_ready(src.location):
            return None
        if local_prequant_scheme(src.location) != scheme:
            return None
        if not _comfy_prequant_family(fam):
            # only the image denoiser loader reads a ComfyUI-format file; anywhere else it would be refused at load
            try:
                from .diffusion_comfy_quant import scan_comfy_quant

                from os.path import expanduser
                if scan_comfy_quant(expanduser(src.location)) is not None:
                    return None
            except Exception:  # noqa: BLE001 -- unreadable: the scheme check above already decided
                pass
    return src


def cached_checkpoint_path(
    source: Any,
    *,
    cache_dir: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    online: Optional[bool] = None,
) -> Optional[str]:
    """The path of a hosted (``kind == "repo"``) checkpoint ALREADY in the local Hub cache. A pure
    lookup (a refs read plus a stat, no network), so memory planning can ask on every pick.

    Every candidate name counts, IN PREFERENCE ORDER, not just the primary. Primary-only was right
    while the primary was the only name a repo could realistically host; it is wrong the moment the
    chain leads with a safetensors name that most repos do not have yet, because then every existing
    ``.pt`` repo reads as "this would have to download several GB" and loses to the GGUF even though
    its checkpoint is sitting in the cache. Walking the chain in order keeps the anti-staleness
    property that motivated primary-only: the better name still wins whenever it is present.

    A name this install cannot OPEN is never a hit, in either direction. The obvious direction is a
    cached ``.pt`` on a host whose torch or torchao cannot restrict that load; the inverse is a
    cached ``.safetensors`` on a host without torchao's flatten helpers, which reads the pickle
    perfectly well. Both end the same way if this answers yes: the planner commits to the prequant,
    drops the dense shards, and ``_resolve_checkpoint_path`` then filters out the very file the
    cache hit was about and finds the other name uncached. Asked per NAME rather than per scheme,
    because the container is what decides it.

    Online the answer is the file the load opens (``first_cached_as_resolved``); ``online=False`` asks
    whether ANY readable name is cached.

    ``names`` narrows the chain further, to a caller's own subset.

    Both cache roots are searched: Unsloth pins the LIVE cache setting while an unpinned
    ``hf_hub_download`` falls back to huggingface_hub's import-time constant. Never raises."""
    roots = (cache_dir, None) if cache_dir else (None,)
    wanted = set(names) if names is not None else None

    def _readable(name: str) -> bool:
        try:
            return bool(restricted_prequant_load_supported(None, name))
        except Exception:  # noqa: BLE001 - a pure lookup that never raises, as documented above
            return True

    candidates = [
        n
        for n in candidate_filenames_of(source)
        if (wanted is None or n in wanted) and _readable(n)
    ]
    location = getattr(source, "location", None)
    if getattr(source, "kind", None) == "repo":
        mirrored = _first_mirrored(location, candidates, _readable)
        if mirrored is not None:
            return mirrored
    hits: dict = {}

    def _hit(name: str) -> Optional[str]:
        if name not in hits:
            hits[name] = next(
                (
                    h
                    for h in (_cached_in_root(source, root, name) for root in roots)
                    if h is not None
                ),
                None,
            )
        return hits[name]

    ordered = prefer_cached_pickle_twins(
        location, candidates, readable = _readable, cache_dir = cache_dir, log = False
    )
    name = first_cached_as_resolved(
        location,
        ordered,
        is_cached = lambda n: _hit(n) is not None,
        online = online,
        cache_dir = cache_dir,
    )
    return _hit(name) if name else None


def _cached_in_root(
    source: Any,
    root: Optional[str],
    name: Optional[str] = None,
) -> Optional[str]:
    """One checkpoint name's path inside ONE cache root, or None. Defaults to the primary name; the
    resolver passes ``fallback_filename`` once the primary turns out to be absent. Never raises."""
    if source is None or getattr(source, "kind", None) != "repo":
        return None
    name = name or getattr(source, "filename", None)
    if not name:
        return None
    try:
        import os

        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 - no cache API to ask: treat as not cached
        return None
    try:
        hit = try_to_load_from_cache(source.location, name, cache_dir = root)
    except Exception:  # noqa: BLE001 - a malformed cache entry is not a hit
        return None
    return hit if isinstance(hit, str) and os.path.isfile(hit) else None


def prequant_checkpoint_cached(
    source: Any,
    *,
    cache_dir: Optional[str] = None,
    online: Optional[bool] = None,
) -> bool:
    """True when ``source`` resolves from the cache, i.e. enabling prequant costs no download."""
    return cached_checkpoint_path(source, cache_dir = cache_dir, online = online) is not None


def _pin_kernel_preference(state_dict: Any, logger: Any = None) -> int:
    """Force every loaded fp8 weight onto the plain-torch kernel, matching the local path.

    ``_fp8_config`` pins ``KernelPreference.TORCH`` when it BUILDS a config, because AUTO silently
    switches to the MSLK kernel wherever an mslk package is importable (sm90+). A hosted checkpoint
    escapes that pin entirely: the preference is serialized on each Float8Tensor, and every
    published one carries AUTO. Restoring it re-arms the exact kernel the pin exists to avoid, and
    ``mslk.f8f8bf16_rowwise`` has no fake impl, so the first COMPILED generate dies with "Operator
    does not support running with fake tensors" -- an HTTP 500 on the default speed mode, reachable
    the moment the pre-quant repos are readable.

    Safe to rewrite in place: the preference selects a matmul kernel, it is not weight data, so the
    tensors stay bit-identical and the checkpoint's own sha256 still describes them. The plain-torch
    path is also the faster one compiled (an opaque extern call blocks inductor quantize fusion), so
    this costs nothing.
    """
    try:
        from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    except Exception:  # noqa: BLE001 -- enum moved or absent: leave the checkpoint as saved
        return 0
    pinned = 0
    for t in state_dict.values():
        if getattr(t, "kernel_preference", None) not in (None, KernelPreference.TORCH):
            try:
                t.kernel_preference = KernelPreference.TORCH
                pinned += 1
            except Exception:  # noqa: BLE001 -- frozen subclass: nothing else to try
                pass
    if pinned and logger is not None:
        logger.info("diffusion.prequant: pinned %d weights to the plain-torch fp8 kernel", pinned)
    return pinned


def load_prequantized_transformer(
    transformer_cls: Any,
    base: str,
    source: PrequantSource,
    *,
    device: str,
    dtype: Any,
    hf_token: Optional[str] = None,
    scheme: str,
    min_features: Optional[int] = None,
    fast_accum: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    prepare_model: Optional[Any] = None,
    config_subfolder: str = "transformer",
    component: Optional[str] = None,
    local_files_only: bool = False,
    logger: Any = None,
    placement_device: Optional[str] = None,
    family: Optional[str] = None,
) -> Optional[Any]:
    """Load the pre-quantized transformer described by ``source`` onto ``device``.

    ``family`` (the Studio family name) is what a ComfyUI-format artifact needs to pick the layers Studio's
    own ``scheme`` quantizes; Studio's own checkpoints record it themselves.

    ``placement_device`` (default ``device``) is where the module is materialised; ``device`` selects kernels.

    ``cache_dir`` is the live Hub cache root, as every other loader call pins it: unset, a fetch
    lands under huggingface_hub's import-time constant, so a mid-session cache change re-downloads
    into a root Unsloth no longer reads.

    ``component`` must match the checkpoint's (MoE experts share every other field); ``prepare_model``
    runs between ``from_config`` and ``load_state_dict``, the one window to reshape the skeleton. A
    declared activation rotation is installed unconditionally: a miss renders wrong pixels silently.

    Returns the placed transformer, or None on any problem (missing / mismatched / unreadable
    checkpoint, unsupported meta-init, or a rotation this build cannot apply exactly) so the caller
    falls back to dense-quantise. Best-effort: never raises for an unavailable artifact.
    """
    _LAST_FAILURE.text = None
    try:
        if source.kind == "path" and not _local_prequant_path_allowed(source.location):
            _warn(
                logger,
                f"{scheme}:path",
                RuntimeError(
                    "request-supplied local pre-quant path refused (loading arbitrary weights "
                    f"into the served model); set {ALLOW_LOCAL_PREQUANT_PATH_ENV} to an "
                    "allowlisted directory containing trusted checkpoints to permit it",
                ),
            )
            return None

        path = _resolve_checkpoint_path(
            source,
            hf_token,
            cache_dir,
            local_files_only = local_files_only,
            scheme = scheme,
            logger = logger,
        )
        if path is None:
            return None

        # A ComfyUI-format file rebuilds into the same torchao weights as Studio's own checkpoint of this scheme.
        from .diffusion_comfy_quant import load_comfy_prequant, scan_comfy_quant

        comfy_scan = scan_comfy_quant(path)
        comfy_rotated = False
        if comfy_scan is not None:
            if prepare_model is not None or (component and component != DEFAULT_PREQUANT_COMPONENT):
                raise ValueError(
                    "a ComfyUI-format checkpoint is only read as a whole denoiser, not as the "
                    f"{component or 'reshaped'} component"
                )
            transformer = load_comfy_prequant(
                transformer_cls,
                path,
                scheme = scheme,
                base = base,
                family = family,
                dtype = dtype,
                hf_token = hf_token,
                cache_dir = cache_dir,
                local_files_only = local_files_only,
                fast_accum = fast_accum,
                min_features = min_features,
                config_subfolder = config_subfolder,
                logger = logger,
            )
            from .diffusion_convrot import declares_rotation, is_rotated_linear, warm_rotation_cache

            comfy_rotated = any(is_rotated_linear(m) for m in transformer.modules())
            metadata = {
                "family": family,
                "torch_dtype": str(dtype).replace("torch.", ""),
                "comfy_format": True,
            }
        else:
            # A safetensors artifact, or a torch.save pickle deserialized under the constructor ALLOWLIST above and never
            # as a free-running one. First-party hosting is no reason to execute whatever bytes arrive: the artifact is
            # mutable, fetched over the network, and reached by loads that never asked for one (auto resolves an unset
            # precision to a hosted checkpoint), so a mutated file must fail to load rather than run. Both containers
            # hand back the same dict, so every check below applies to them equally.
            ckpt = _read_prequant_for(path, placement_device or device, logger)
            if not _validate_checkpoint(
                ckpt,
                scheme,
                base,
                logger,
                min_features = min_features,
                fast_accum = fast_accum,
                component = component,
            ):
                return None
            state_dict = ckpt["state_dict"]
            # The only check reading what the artifact HOLDS: corruption after build passes the rest.
            if not _verify_packed_fingerprint(
                state_dict, ckpt.get("metadata") or {}, logger = logger, path = path
            ):
                return None
            _repair_legacy_checkpoint(ckpt, scheme, logger)
            _pin_kernel_preference(state_dict, logger)

            # Read from the root that actually supplied the checkpoint: after a mid-session cache change the pinned root
            # may be gone or read-only, and load_config's raise is swallowed below into a None return, silently dropping a
            # prequant whose checkpoint is cached and already loaded.
            config = _load_transformer_config(
                transformer_cls,
                base,
                hf_token,
                cache_dir,
                path,
                config_subfolder,
                local_files_only = local_files_only,
            )
            from accelerate import init_empty_weights

            metadata = ckpt.get("metadata") or {}
            with init_empty_weights():
                transformer = transformer_cls.from_config(config)
            if prepare_model is not None:
                prepare_model(transformer, metadata)
            transformer.load_state_dict(state_dict, strict = True, assign = True)
            if _has_meta_tensors(transformer):
                # Non-persistent buffers (built in __init__, absent from the state dict) stay on meta. Rebuild on CPU so
                # they hold real values, then re-assign the quantized weights; dense bf16 never reaches the GPU.
                transformer = transformer_cls.from_config(config)
                # The retry REPLACES the module, so the hook has to run again: skipping it here would load the same state
                # dict into a differently shaped model, and this branch is the one families with non-persistent buffers
                # always take -- the mismatch would be the norm, not the corner case, and strict=True would surface it as
                # a bare key error.
                if prepare_model is not None:
                    prepare_model(transformer, metadata)
                transformer.load_state_dict(state_dict, strict = True, assign = True)

            # The ONLINE half of an activation rotation, applied here rather than in a family's ``prepare_model`` hook so
            # that no route can load a rotated checkpoint without it: the offline half is already baked into the weights
            # that were just assigned, and a rotated weight met by an unrotated activation renders plausible garbage with
            # nothing to catch. A no-op for every artifact that declares no rotation, and a RAISE (caught below into the
            # dense fallback) for one this build cannot honour exactly. After load_state_dict because the meta retry above
            # rebuilds the module; before apply_small_m_padding because padding reparents the Linears and the recorded
            # fqns name the unwrapped tree.
            from .diffusion_convrot import (
                apply_activation_rotation,
                declares_rotation,
                warm_rotation_cache,
            )

            apply_activation_rotation(transformer, metadata, logger = logger)

            if scheme == "nvfp4":
                from .diffusion_nvfp4_linear import convert_nvfp4_backend
                from .diffusion_nvfp4_ops import select_nvfp4_backend
                convert_nvfp4_backend(
                    transformer, metadata, select_nvfp4_backend(device), logger = logger
                )
            # assign=True shares the tensors: a live reference doubles the peak on unified memory.
            del state_dict
            del ckpt

        from .diffusion_fast_load import fast_upload

        with fast_upload([transformer], placement_device or device, logger = logger):
            transformer = transformer.to(placement_device or device)
        if declares_rotation(metadata) or comfy_rotated:
            try:
                import torch

                on = next(iter(transformer.parameters()), None)
                dtype = getattr(
                    torch, str(metadata.get("torch_dtype") or "bfloat16"), torch.bfloat16
                )
                warm_rotation_cache(
                    transformer,
                    on.device if on is not None and placement_device is None else device,
                    dtype,
                )
            except Exception:  # noqa: BLE001
                pass
        # Small-M padding, else quantised small-M linears raise in _int_mm. After load and .to().
        from .diffusion_transformer_quant import apply_small_m_padding, apply_zero_row_guard

        apply_small_m_padding(transformer, scheme, metadata.get("family"), logger = logger)
        apply_zero_row_guard(transformer, scheme, metadata.get("family"), logger = logger)
        # from_config starts in train mode; match from_pretrained's eval().
        try:
            transformer.eval()
        except Exception:  # noqa: BLE001 - eval() is best-effort
            pass
        if scheme == "nvfp4":
            try:
                from .diffusion_nvfp4_linear import nvfp4_prewarm
                nvfp4_prewarm(transformer, (1,), logger = logger)
            except Exception as exc:  # noqa: BLE001 - an untuned layer still runs
                _warn(logger, f"{scheme}:prewarm", exc)
        try:
            transformer._unsloth_runtime_quant = scheme
        except Exception:  # noqa: BLE001 - marker is best-effort
            pass
        try:
            transformer._unsloth_prequant_path = path
        except Exception:  # noqa: BLE001 - marker is best-effort
            pass
        if logger is not None:
            logger.info(
                "diffusion.prequant: loaded %s checkpoint (%s) onto %s",
                scheme,
                source.kind,
                placement_device or device,
            )
        return transformer
    except Exception as exc:  # noqa: BLE001 - fall back to the dense-quantise path
        _warn(logger, f"{scheme}:{source.kind}", exc)
        return None


def _entry_not_found_errors() -> tuple:
    """``(EntryNotFoundError, LocalEntryNotFoundError)`` for both huggingface_hub majors. On 1.x the
    base splits into a remote 404 and ``LocalEntryNotFoundError`` (no copy in this root, no
    network); on BOTH majors local subclasses the base, so catch it first where they differ.
    Private markers on an unexpected layout are raised by nothing, keeping today's behaviour."""
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except Exception:  # noqa: BLE001 - older/newer hub layouts

        class EntryNotFoundError(Exception):  # type: ignore[no-redef]
            pass

    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except Exception:  # noqa: BLE001

        class LocalEntryNotFoundError(EntryNotFoundError):  # type: ignore[no-redef]
            pass

    return EntryNotFoundError, LocalEntryNotFoundError


def _download_checkpoint_name(
    source: PrequantSource,
    name: str,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    *,
    propagate_missing: bool,
    local_files_only: bool = False,
) -> str:
    """Download ONE checkpoint filename, reusing a copy that sits under the other cache root. Pinned
    to ``cache_dir``, hf_hub_download would not look there and would re-fetch multiple GB, so
    re-run it THROUGH that root rather than return the raw path: the blob is reused after one
    HEAD, a republished checkpoint is picked up rather than pinned stale, and offline still
    resolves off the cached pointer. ``propagate_missing`` says another filename is still to be
    tried, so a remote 404 for THIS one must reach the caller's fallback branch; swallowing it
    would return the stale other-root copy of a name the repo no longer publishes. A local cache
    miss is not that verdict, and with no name left to try neither is a 404: both keep the copy
    already found."""
    from huggingface_hub import hf_hub_download

    EntryNotFoundError, LocalEntryNotFoundError = _entry_not_found_errors()

    if cache_dir is not None and _cached_in_root(source, cache_dir, name) is None:
        elsewhere = _cached_in_root(source, None, name)
        if elsewhere is not None:
            try:
                return hf_hub_download(
                    repo_id = source.location,
                    filename = name,
                    token = hf_token,
                    cache_dir = None,
                    local_files_only = local_files_only,
                )
            except LocalEntryNotFoundError:
                return elsewhere
            except EntryNotFoundError:
                if not propagate_missing:
                    return elsewhere
                raise
            except Exception:  # noqa: BLE001 - revalidation is a bonus, never a new failure
                return elsewhere
    return hf_hub_download(
        repo_id = source.location,
        filename = name,
        token = hf_token,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
    )


def _resolve_checkpoint_path(
    source: PrequantSource,
    hf_token: Optional[str],
    cache_dir: Optional[str] = None,
    *,
    local_files_only: bool = False,
    scheme: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    """The local file path for ``source``, downloading from the Hub if needed; None if absent.
    ``local_files_only`` is the caller's promise that this load may not fetch anything, so a
    cache miss answers None and the build falls back rather than pulling several GB nobody asked
    for.

    ``scheme`` drops the names this install could not deserialize anyway, which is the SAME filter
    the download plan applies. It has to be the same one: with only the plan filtering, a repo
    hosting both containers would have the plan stage the readable one while this fetched the
    other, downloading a second artifact to fail on it and then falling back to dense weights the
    plan had already left out. Unset keeps the whole chain, for the callers that have no scheme to
    offer."""
    if source.kind == "path":
        import os

        # Expand ~ (the allowlist gate already did), else os.path.isfile sees a literal "~".
        expanded = os.path.expanduser(source.location)
        return expanded if os.path.isfile(expanded) else None
    if source.kind == "repo":
        EntryNotFoundError, LocalEntryNotFoundError = _entry_not_found_errors()
        names = list(candidate_filenames_of(source))
        all_names = list(names)
        if scheme is not None:
            readable = [n for n in names if restricted_prequant_load_supported(scheme, n)]
            names = readable or names
        if not names:
            return None
        # Mirror first for EVERY name: else a Hub copy of an earlier name is downloaded and the mirror never used.
        mirrored = _first_mirrored(
            source.location, names, lambda n: restricted_prequant_load_supported(scheme, n)
        )
        if mirrored is not None:
            return mirrored
        names = prefer_cached_pickle_twins(
            source.location,
            names,
            readable = lambda n: restricted_prequant_load_supported(scheme, n),
            cache_dir = cache_dir,
            logger = logger,
            roots = tuple(dict.fromkeys((cache_dir, None))),
        )
        for index, name in enumerate(names):
            last = index == len(names) - 1
            try:
                path = _download_checkpoint_name(
                    source,
                    name,
                    hf_token,
                    cache_dir,
                    # Only the last name may swallow its 404, so the next candidate is tried.
                    propagate_missing = not last,
                    local_files_only = local_files_only,
                )
                explain_container_choice(
                    source.location,
                    name,
                    all_names,
                    names,
                    readable = lambda n: restricted_prequant_load_supported(scheme, n),
                    logger = logger,
                )
                return path
            except LocalEntryNotFoundError:
                # Before its base class: online, this means the Hub is unreachable, not that the name
                # is absent. Offline, treat it as a 404.
                if not local_files_only or last:
                    cached = (
                        None
                        if last
                        else cached_checkpoint_path(
                            source, cache_dir = cache_dir, names = names[index + 1 :], online = False
                        )
                    )
                    if cached is None:
                        raise
                    return cached
            except EntryNotFoundError:
                if last:
                    raise
    return None


def _config_cache_roots(checkpoint_path: str, cache_dir: Optional[str]) -> tuple:
    """Cache roots to read the transformer config from, the checkpoint's OWN root first.
    ``_resolve_checkpoint_path`` may answer from huggingface_hub's import-time root even when
    Unsloth pins its live one, so pinning the config to the live root alone misses in exactly the
    cache-moved/offline case the checkpoint lookup just accepted, and load_config's raise is
    swallowed into a None return. The other root is still tried second."""
    if cache_dir is None:
        return (None,)
    import os

    try:
        # normcase: Windows path case differences would reverse the order below.
        root = os.path.normcase(os.path.realpath(cache_dir))
        real = os.path.normcase(os.path.realpath(checkpoint_path))
        under_live = real == root or real.startswith(root + os.sep)
    except Exception:  # noqa: BLE001 - an unresolvable path keeps today's order
        under_live = True
    return (cache_dir, None) if under_live else (None, cache_dir)


def _load_transformer_config(
    transformer_cls: Any,
    base: str,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    checkpoint_path: str,
    subfolder: str = "transformer",
    *,
    local_files_only: bool = False,
) -> Any:
    """``transformer_cls.load_config`` against the checkpoint's cache root, then the other one. The
    config is a few KB, but it is still a Hub fetch, and a load that promised to reach nothing
    has to keep that promise for the small files too."""
    last: Optional[BaseException] = None
    for root in _config_cache_roots(checkpoint_path, cache_dir):
        try:
            return transformer_cls.load_config(
                base,
                subfolder = subfolder,
                token = hf_token,
                cache_dir = root,
                local_files_only = local_files_only,
            )
        except Exception as exc:  # noqa: BLE001 - try the other root before giving up
            last = exc
    raise last  # type: ignore[misc]


# By NAME: re-exported under several paths, and this check must not import torchao.
_FLOAT8_TENSOR_CLASS = "Float8Tensor"


def _fp8_activation_floor_present(
    state_dict: Any,
    logger: Any,
    *,
    warn: bool = True,
) -> bool:
    """True unless the first Float8Tensor has no activation lower bound (by class: NVFP4Tensor lacks one too)."""
    from .diffusion_transformer_quant import TQ_FP8

    try:
        items = state_dict.items() if hasattr(state_dict, "items") else ()
        for name, tensor in items:
            if type(tensor).__name__ != _FLOAT8_TENSOR_CLASS:
                continue
            kwargs = getattr(tensor, "act_quant_kwargs", None)
            if kwargs is None:
                continue
            if getattr(kwargs, "hp_value_lb", None):
                return True
            if not warn:
                return False
            _warn(
                logger,
                TQ_FP8,
                ValueError(
                    f"fp8 checkpoint has no activation scale floor on {name!r} "
                    "(built before activation_value_lb); a zero activation row renders black. "
                    "Rebuild it"
                ),
            )
            return False
    except Exception:  # noqa: BLE001 -- an unreadable state dict is the other checks' problem
        return True
    return True


def _fp8_kwargs_missing_floor(tensor: Any) -> Optional[Any]:
    if type(tensor).__name__ != _FLOAT8_TENSOR_CLASS:
        return None
    kwargs = getattr(tensor, "act_quant_kwargs", None)
    if kwargs is None or getattr(kwargs, "hp_value_lb", None):
        return None
    return kwargs


def _fp8_activation_floor_restorable(state_dict: Any) -> bool:
    """Whether every unfloored Float8Tensor differs from the runtime config in the floor alone.

    torchao's fp8 weight quantiser never reads ``hp_value_lb``, so the weight bytes of an artifact built before
    ``activation_value_lb`` equal what ``_make_quant_config`` builds today."""
    try:
        items = state_dict.items() if hasattr(state_dict, "items") else ()
        for _name, tensor in items:
            kwargs = _fp8_kwargs_missing_floor(tensor)
            if kwargs is None:
                continue
            if (
                not hasattr(kwargs, "hp_value_lb")
                or getattr(kwargs, "hp_value_ub", None) is not None
            ):
                return False
    except Exception:  # noqa: BLE001 -- cannot prove it: keep refusing
        return False
    return True


def _restore_fp8_activation_floor(state_dict: Any, logger: Any = None) -> int:
    """Write the runtime activation floor into every unfloored Float8Tensor; returns how many.
    Fresh kwargs per tensor: a pickle may share one instance between tensors."""
    import copy
    import dataclasses

    from .diffusion_transformer_quant import FP8_ACTIVATION_VALUE_LB

    restored = 0
    for _name, tensor in list(state_dict.items()):
        kwargs = _fp8_kwargs_missing_floor(tensor)
        if kwargs is None:
            continue
        if dataclasses.is_dataclass(kwargs) and not isinstance(kwargs, type):
            fixed = dataclasses.replace(kwargs, hp_value_lb = FP8_ACTIVATION_VALUE_LB)
        else:
            fixed = copy.copy(kwargs)
            fixed.hp_value_lb = FP8_ACTIVATION_VALUE_LB
        tensor.act_quant_kwargs = fixed
        restored += 1
    if restored and logger is not None:
        logger.info(
            "diffusion.prequant: restored the fp8 activation scale floor (%g) on %d weights built "
            "before activation_value_lb",
            FP8_ACTIVATION_VALUE_LB,
            restored,
        )
    return restored


def _repair_legacy_checkpoint(
    ckpt: Any,
    scheme: str,
    logger: Any = None,
) -> None:
    from .diffusion_nvfp4_policy import declares_policy
    from .diffusion_transformer_quant import TQ_FP8
    if scheme == TQ_FP8 or declares_policy(ckpt.get("metadata") or {}):
        _restore_fp8_activation_floor(ckpt["state_dict"], logger)


def _validate_activation_rotation(ckpt_format: Any, meta: Any, scheme: str, logger: Any) -> bool:
    """Reject a checkpoint whose activation rotation this build cannot honour EXACTLY.

    Three ways an artifact and a loader can disagree about the rotation, and all three end in the
    same place -- weights in a rotated basis multiplied by unrotated activations, which is finite,
    raises nothing, and renders quietly wrong -- so all three are refused here rather than
    discovered later. First, the artifact declares a rotation and is tagged v1: only v2 makes an
    Unsloth too old for this code refuse it, so a v1 tag on rotated weights is a hazard to every
    OTHER build, and the builder that produced it is not one to trust about anything else in the
    file. Second, the artifact is tagged v2 and declares none: nothing here would rotate, and the
    tag says something was meant to. Third, the rotation is declared but its contract does not parse
    (an unknown kind, a group that is not a power of 4, an absent or malformed fqn list).

    Refusing costs a dense fallback: slower and bigger, never wrong.
    """
    from .diffusion_convrot import declares_rotation, rotation_metadata_error

    rotated = declares_rotation(meta)
    tagged = ckpt_format == PREQUANT_FORMAT_ROTATED
    if rotated != tagged:
        _warn(
            logger,
            scheme,
            ValueError(
                f"checkpoint format {ckpt_format!r} and its activation rotation disagree "
                f"(declares a rotation: {rotated}); a rotated checkpoint must be tagged "
                f"{PREQUANT_FORMAT_ROTATED!r} so older builds refuse it instead of running it "
                "unrotated"
            ),
        )
        return False
    problem = rotation_metadata_error(meta)
    if problem:
        _warn(logger, scheme, ValueError(problem))
        return False
    return True


def _validate_policy(ckpt_format: Any, meta: Any, scheme: str, logger: Any) -> bool:
    """Reject a checkpoint whose per-layer nvfp4 policy this build cannot reproduce EXACTLY."""
    from .diffusion_nvfp4_policy import (
        NVFP4_POLICY_KEY,
        declares_policy,
        policy_expected_counts,
        policy_metadata_error,
        resolve_policy,
    )
    from .diffusion_transformer_quant import TQ_NVFP4

    declared = declares_policy(meta)
    tagged = ckpt_format == PREQUANT_FORMAT_POLICY
    if declared != tagged:
        _warn(
            logger,
            scheme,
            ValueError(
                f"checkpoint format {ckpt_format!r} and its per-layer nvfp4 policy disagree "
                f"(declares a policy: {declared}); a policy checkpoint must be tagged "
                f"{PREQUANT_FORMAT_POLICY!r} so older builds refuse it instead of loading it as a "
                "whole-model artifact"
            ),
        )
        return False
    if not declared:
        return True
    problem = policy_metadata_error(meta)
    if problem:
        _warn(logger, scheme, ValueError(problem))
        return False
    block = meta.get(NVFP4_POLICY_KEY)
    if scheme != TQ_NVFP4:
        _warn(
            logger,
            scheme,
            ValueError(
                f"checkpoint declares the nvfp4 policy {block.get('policy_id')!r} but its scheme "
                f"is {scheme!r}"
            ),
        )
        return False
    policy = resolve_policy(meta.get("family"), meta.get("base_model_id"))
    if policy is None:
        _warn(
            logger,
            scheme,
            ValueError(
                f"checkpoint declares the nvfp4 policy {block.get('policy_id')!r} but this build "
                f"resolves none for family {meta.get('family')!r} on base "
                f"{meta.get('base_model_id')!r}"
            ),
        )
        return False
    declared_id = (block.get("policy_id"), int(block.get("policy_version")))
    expected_id = (policy.policy_id, int(policy.version))
    if declared_id != expected_id:
        _warn(
            logger,
            scheme,
            ValueError(
                f"checkpoint nvfp4 policy {declared_id!r} != the one this build resolves for "
                f"{meta.get('base_model_id')!r}, {expected_id!r}"
            ),
        )
        return False
    declared_counts = {str(key): int(value) for key, value in (block.get("counts") or {}).items()}
    expected_counts = policy_expected_counts(policy)
    if declared_counts != expected_counts:
        _warn(
            logger,
            scheme,
            ValueError(
                f"checkpoint nvfp4 policy {policy.policy_id!r} records {declared_counts!r}, this "
                f"build's table expects {expected_counts!r}"
            ),
        )
        return False
    return True


def hosted_fast_accum_conflict(scheme: str, fast_accum: Optional[bool]) -> bool:
    """Whether a FORCED fp8 accumulate rules out every HOSTED checkpoint for ``scheme``.

    ``scripts/build_prequant_checkpoint.py`` bakes the auto choice (``_resolve_fast_accum(None)``)
    into the fp8 artifacts, and ``_validate_checkpoint`` refuses a baked value differing from a
    forced one. A planner that seeds without asking drops the released shards for a checkpoint the
    load must reject. Only fp8 bakes the field, so every other scheme is False. No IO, no torch."""
    from .diffusion_transformer_quant import TQ_FP8, _resolve_fast_accum

    if fast_accum is None or scheme != TQ_FP8:
        return False
    return bool(fast_accum) != _resolve_fast_accum(None)


def _validate_checkpoint(
    ckpt: Any,
    scheme: str,
    base: str,
    logger: Any,
    min_features: Optional[int] = None,
    fast_accum: Optional[bool] = None,
    component: Optional[str] = None,
) -> bool:
    """Reject a wrong format / scheme / base / filter / denoiser; absent ``min_features`` / ``fast_accum`` pass."""
    if not isinstance(ckpt, dict) or ckpt.get("format") not in PREQUANT_FORMATS:
        _warn(logger, scheme, ValueError("unrecognised pre-quant checkpoint format"))
        return False
    if "state_dict" not in ckpt:
        _warn(logger, scheme, ValueError("pre-quant checkpoint has no state_dict"))
        return False
    meta = ckpt.get("metadata") or {}
    if not _validate_activation_rotation(ckpt.get("format"), meta, scheme, logger):
        return False
    if not _validate_policy(ckpt.get("format"), meta, scheme, logger):
        return False
    if meta.get("scheme") != scheme:
        _warn(logger, scheme, ValueError(f"checkpoint scheme {meta.get('scheme')!r} != {scheme!r}"))
        return False
    # fp8 requires per-row granularity; old checkpoints without it are re-quantised.
    from .diffusion_nvfp4_policy import declares_policy
    from .diffusion_transformer_quant import FP8_GRANULARITY, TQ_FP8

    holds_fp8 = scheme == TQ_FP8 or declares_policy(meta)
    if holds_fp8 and meta.get("fp8_granularity") != FP8_GRANULARITY:
        _warn(
            logger,
            scheme,
            ValueError(
                f"fp8 checkpoint granularity {meta.get('fp8_granularity')!r} != "
                f"{FP8_GRANULARITY!r} (stale per-tensor artifact); rebuild it"
            ),
        )
        return False
    # fp8 also requires the activation scale floor (zero rows give NaN), checked on the
    # tensors, fail-closed. A missing floor alone is repaired by _repair_legacy_checkpoint.
    if holds_fp8 and not _fp8_activation_floor_present(ckpt.get("state_dict"), logger, warn = False):
        if not _fp8_activation_floor_restorable(ckpt.get("state_dict")):
            _fp8_activation_floor_present(ckpt.get("state_dict"), logger)
            return False
        if logger is not None:
            logger.info(
                "diffusion.prequant: fp8 checkpoint predates activation_value_lb; the runtime "
                "floor will be written into its weights"
            )
    ckpt_base = meta.get("base_model_id")
    if base:
        # Wrong-base keys can load strict=True; the builder always records base_model_id.
        if not ckpt_base:
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint metadata missing base_model_id; refusing for base {base!r}"
                ),
            )
            return False
        if not _same_base_model(ckpt_base, base):
            _warn(logger, scheme, ValueError(f"checkpoint base {ckpt_base!r} != {base!r}"))
            return False
    if min_features is not None:
        ckpt_min = meta.get("min_features")
        if ckpt_min is not None and int(ckpt_min) != int(min_features):
            _warn(
                logger,
                scheme,
                ValueError(f"checkpoint min_features {ckpt_min!r} != runtime {min_features!r}"),
            )
            return False
    # The int8 exclusion set is scheme-derived, so a token-list change would leave old checkpoints with a stale baked
    # set that passes scheme+min_features then crashes at the first denoise. Reject a checkpoint that quantised a
    # layer the runtime excludes; absent is accepted, and so is a superset (extra bf16 Linears load as stored; the
    # hosted Wan2.2 fp8 files keep ``condition_embedder`` bf16).
    ckpt_excludes = meta.get("exclude_name_tokens")
    if ckpt_excludes is not None:
        from .diffusion_transformer_quant import exclude_tokens_for_scheme
        expected = tuple(exclude_tokens_for_scheme(scheme, meta.get("family")))
        if not isinstance(ckpt_excludes, (list, tuple)) or not set(expected) <= set(ckpt_excludes):
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint exclude_name_tokens {ckpt_excludes!r} do not cover the runtime {expected!r}"
                ),
            )
            return False
    ckpt_require_bf16 = meta.get("require_bf16")
    if ckpt_require_bf16 is not None:
        from .diffusion_transformer_quant import _REQUIRE_BF16_SCHEMES
        expected_require_bf16 = scheme in _REQUIRE_BF16_SCHEMES
        if bool(ckpt_require_bf16) != expected_require_bf16:
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint require_bf16 {bool(ckpt_require_bf16)!r} != {expected_require_bf16!r}"
                ),
            )
            return False
    ckpt_divisible = meta.get("require_divisible")
    if ckpt_divisible is not None:
        from .diffusion_transformer_quant import divisible_for_scheme
        expected_divisible = divisible_for_scheme(scheme)
        if int(ckpt_divisible) != expected_divisible:
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint require_divisible {ckpt_divisible!r} != {expected_divisible!r}"
                ),
            )
            return False
    elif logger is not None:
        logger.debug(
            "diffusion.prequant: checkpoint records no require_divisible (built before the "
            "field); accepting it for %s",
            scheme,
        )
    if component:
        ckpt_component = meta.get("component")
        if ckpt_component is not None and str(ckpt_component) != str(component):
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint component {ckpt_component!r} != {component!r}; this is another "
                    "denoiser of the same family and would load clean and render wrong"
                ),
            )
            return False
        if ckpt_component is None and logger is not None:
            logger.debug(
                "diffusion.prequant: checkpoint records no component; accepting it for %r",
                component,
            )
    if fast_accum is not None:
        ckpt_fa = meta.get("fast_accum")
        if ckpt_fa is not None and bool(ckpt_fa) != bool(fast_accum):
            _warn(
                logger,
                scheme,
                ValueError(f"checkpoint fast_accum {ckpt_fa!r} != requested {bool(fast_accum)!r}"),
            )
            return False
    return True


def _same_base_model(a: str, b: str) -> bool:
    """Tolerant base-model id compare: exact, or same final path/repo segment (e.g.
    ``/models/Z-Image-Turbo`` vs ``Tongyi-MAI/Z-Image-Turbo``). Both sides normalise through
    ``canonical_base`` first, so a mirror id in a baked ``base_model_id`` check cannot refuse the
    checkpoint and send the load down the multi-GB dense download. Today's mirrors keep the repo
    name, so the tail compare would cover them, but this must not depend on that."""
    from .diffusion_families import canonical_base

    a, b = canonical_base(a), canonical_base(b)

    def _tail(x: str) -> str:
        return x.replace("\\", "/").rstrip("/").split("/")[-1].lower()

    return a == b or _tail(a) == _tail(b)


def _unhook_from_manager(
    manager: Any,
    module: Any,
    *,
    logger: Any = None,
    what: str,
) -> bool:
    """Remove ``module`` from a ComponentsManager's offload rotation without moving it."""
    hooks = list(getattr(manager, "model_hooks", None) or ())
    target = next((hook for hook in hooks if getattr(hook, "model", None) is module), None)
    if target is None:
        return False
    try:
        target.remove()
        # Unlist it too, else another pre_forward can evict it to CPU with no hook to bring it back.
        for hook in hooks:
            others = getattr(getattr(hook, "hook", None), "other_hooks", None)
            if others:
                hook.hook.other_hooks = [item for item in others if item is not target]
        manager.model_hooks = [hook for hook in hooks if hook is not target]
        return True
    except Exception as exc:  # noqa: BLE001 -- the caller still places the module
        _warn(logger, what, exc)
        return False


def pin_prequantized_module(
    manager: Any,
    module: Any,
    device: Any,
    *,
    logger: Any = None,
    label: str = "pre-quantized denoiser",
) -> bool:
    """Keep a module resident on ``device``, out of a ComponentsManager's rotation.

    ``ComponentsManager.enable_auto_cpu_offload`` parks every component on the CPU and moves each
    one onto the accelerator inside its own ``pre_forward``, i.e. from within the block that is
    already executing. A torchao-quantized module does not survive that move: the device change
    reaches ``return_and_correct_aliasing``, which tries to alias a CPU storage to an accelerator
    tensor and raises ``Attempted to set the storage of a tensor on device "cuda:0" to a storage on
    different device "cpu"``, and MiniMax-H3's denoise loop dies on its first step. Moving the same
    module at load time, outside any executing block, works -- so the fix is to place it once here
    and take it out of the rotation rather than to move it per forward.

    That is also what a pre-quantized denoiser is for: the hosted H3 checkpoint is ~20 GB against
    66.3 GB dense, so keeping it resident is the saving being spent. The other components keep their
    hooks, and the strategy sizes its decisions from live free memory, so the encoder and the VAEs
    still offload around it.

    For a torchao module that placement is REQUIRED, for the reason above. A caller may also pin a
    plain dense module, where it is an optimisation instead: a module that moves per forward cannot
    be regionally compiled either, since the onload hooks wrap the forward the graph would replace.
    That caller owns the fit check (this function sizes nothing) and passes its own ``label``.

    Returns True when the module was pinned. Best-effort on the hook surgery: if the manager does
    not look the way this expects, the module is still placed on ``device`` and False is returned,
    which is the behaviour before pinning existed.
    """
    pinned = _unhook_from_manager(manager, module, logger = logger, what = "pin:hook")
    module.to(device)
    if logger is not None:
        logger.info(
            "diffusion.prequant: %s pinned on %s (offload rotation: %s)",
            label,
            device,
            "removed" if pinned else "unchanged",
        )
    return pinned


def tensor_payload_bytes(tensor: Any) -> int:
    """Payload bytes; torchao subclasses report their logical bf16 size, so sum the inner tensors."""
    flatten = getattr(tensor, "__tensor_flatten__", None)
    if callable(flatten) and type(tensor).__name__ not in ("Tensor", "Parameter"):
        try:
            return sum(tensor_payload_bytes(getattr(tensor, name)) for name in flatten()[0])
        except Exception:  # noqa: BLE001 -- fall through to the logical size
            pass
    return int(tensor.numel()) * int(tensor.element_size())


def torchao_group_offload_supported() -> bool:
    """Whether group offloading swaps torchao internals (diffusers >= 0.40); older ones half-move via ``param.data``."""
    try:
        from diffusers.hooks import apply_group_offloading  # noqa: F401
        from diffusers.hooks import group_offloading as go
    except Exception:  # noqa: BLE001 -- no group offloading at all
        return False
    return callable(getattr(go, "_is_torchao_tensor", None)) and callable(
        getattr(go, "_swap_torchao_tensor", None)
    )


def _weights_pinnable(module: Any) -> bool:
    """Whether every weight class answers ``is_pinned`` (torchao <= 0.17's v1 int8 class raises)."""
    from itertools import chain

    seen: set = set()
    try:
        for tensor in chain(module.parameters(), module.buffers()):
            cls = type(tensor)
            if cls in seen:
                continue
            seen.add(cls)
            tensor.is_pinned()
    except Exception:  # noqa: BLE001 -- unimplemented for this subclass
        return False
    return True


def _evict_rotation_hook(manager: Any, device: Any) -> Any:
    """Pre-hook offloading rotating components on ``device``: the manager only evicts inside another
    managed pre_forward, so the conditioner would otherwise stay beside the whole denoise loop."""
    import torch

    execution = torch.device(device)

    @torch.compiler.disable
    def _pre_forward(_module: Any, _args: Any) -> None:
        moved = False
        for hook in list(getattr(manager, "model_hooks", None) or ()):
            model = getattr(hook, "model", None)
            try:
                where = next(model.parameters()).device
            except Exception:  # noqa: BLE001 -- nothing to measure, nothing to move
                continue
            if where.type != execution.type:
                continue
            if execution.index is not None and where.index not in (None, execution.index):
                continue
            hook.offload()
            moved = True
        if moved and execution.type == "cuda":
            torch.cuda.empty_cache()

    return _pre_forward


def _move_groups_outside_inference_mode(module: Any) -> int:
    """Run group offload moves outside inference_mode (torchao v1 int8 raises
    ``Cannot set version_counter for inference tensor``). Returns the number of groups wrapped."""
    import torch

    seen: set[int] = set()
    for submodule in module.modules():
        hooks = getattr(getattr(submodule, "_diffusers_hook", None), "hooks", None) or {}
        for hook in list(hooks.values()):
            group = getattr(hook, "group", None)
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            for name in ("onload_", "offload_"):
                move = getattr(group, name, None)
                if not callable(move):
                    continue

                def _outside(_move: Any = move) -> Any:
                    with torch.inference_mode(False), torch.no_grad():
                        return _move()

                setattr(group, name, torch.compiler.disable(_outside))
    return len(seen)


def stream_prequantized_module(
    manager: Any,
    module: Any,
    device: Any,
    *,
    logger: Any = None,
    label: str = "pre-quantized denoiser",
) -> Optional[str]:
    """Stream a torchao module block by block via group offloading, outside the ComponentsManager rotation.

    Returns ``"stream"`` (fully pinned, async copies), ``"stream_lazy"`` (pinned one group at a time), ``"sync"``
    (unpinnable weights) or None (nothing changed).
    Raises once the module is unhooked: the caller only streams what does not fit pinned, so a resident
    fallback would OOM or be refused on every render."""
    if not torchao_group_offload_supported():
        return None
    import inspect

    import torch
    from diffusers.hooks import apply_group_offloading

    from .diffusion_memory import (
        DEFAULT_GROUP_BLOCKS,
        _remove_group_offload_hooks,
        _streamed_pin_plan,
        install_group_offload_buffer_restore,
        install_group_offload_hooks_eager,
    )

    onload = torch.device(device)
    if not _unhook_from_manager(manager, module, logger = logger, what = "stream:hook"):
        return None
    try:
        # assign=True leaves weights trainable; group offload then breaks on Int8Tensor's aten.view.
        module.requires_grad_(False)
        install_group_offload_buffer_restore()
        install_group_offload_hooks_eager()
        use_stream = onload.type == "cuda" and _weights_pinnable(module)
        if onload.type == "cuda" and not use_stream:
            from .prequant_legacy_int8 import convert_legacy_int8_weights
            converted = convert_legacy_int8_weights(module)
            if converted:
                use_stream = _weights_pinnable(module)
                if logger is not None:
                    logger.info(
                        "diffusion.prequant: rebuilt %d v1 int8 weights as Int8Tensor for streaming",
                        converted,
                    )
        kwargs: dict[str, Any] = {
            "onload_device": onload,
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": DEFAULT_GROUP_BLOCKS,
            "use_stream": use_stream,
        }
        params = inspect.signature(apply_group_offloading).parameters
        if use_stream:
            if "non_blocking" in params:
                kwargs["non_blocking"] = True
            if "record_stream" in params:
                kwargs["record_stream"] = True
            if "low_cpu_mem_usage" in params:
                from itertools import chain
                payload_mib = sum(
                    tensor_payload_bytes(t) for t in chain(module.parameters(), module.buffers())
                ) // (1024 * 1024)
                kwargs["low_cpu_mem_usage"] = not _streamed_pin_plan(payload_mib, 0, logger)[0]
        # Pin through one slab arena: per-tensor pin_memory() rounds each weight to a power of two.
        from .diffusion_pinned_arena import pinned_arena_for_group_offload

        with pinned_arena_for_group_offload(
            enabled = None if use_stream and not kwargs.get("low_cpu_mem_usage") else False
        ) as arena:
            apply_group_offloading(module, **kwargs)
        if arena is not None and arena.payload_bytes:
            module._unsloth_pin_arena_bytes = (arena.payload_bytes, arena.reserved_bytes)
            if logger is not None:
                logger.info(
                    "diffusion.prequant: %s pinned once in %.2f GB of slabs (%.2f GB of weights)",
                    label,
                    arena.reserved_bytes / 1e9,
                    arena.payload_bytes / 1e9,
                )
        _move_groups_outside_inference_mode(module)
        module.register_forward_pre_hook(_evict_rotation_hook(manager, onload))
    except Exception as exc:
        _remove_group_offload_hooks(module)
        raise RuntimeError(f"group offloading could not be set up for the {label}: {exc}") from exc
    mode = (
        ("stream_lazy" if kwargs.get("low_cpu_mem_usage") else "stream") if use_stream else "sync"
    )
    if logger is not None:
        logger.info(
            "diffusion.prequant: %s streamed block by block on %s (%s copies)",
            label,
            device,
            "overlapped" if use_stream else "synchronous",
        )
    return mode


def _has_meta_tensors(module: Any) -> bool:
    """True if any parameter or buffer is still on the meta device after loading."""
    from itertools import chain
    try:
        return any(
            getattr(t, "is_meta", False) for t in chain(module.parameters(), module.buffers())
        )
    except Exception:  # noqa: BLE001
        return False


_LAST_FAILURE = _threading.local()
# Absolute paths reduced to the last component; not after a word, ':' or '/' (URLs, repo ids).
_ABS_PATH = _re.compile(
    r"(?<![\w:/.])(?:[A-Za-z]:[\\/]|/)(?:[^\s'\"<>|:;,()\[\]\\/]+[\\/])+(?=[^\s\\/])"
)


def _warn(logger: Any, what: str, exc: Exception) -> None:
    _LAST_FAILURE.text = _ABS_PATH.sub("", f"{type(exc).__name__}: {exc}")[:300]
    if logger is not None:
        logger.warning("diffusion.prequant: %s failed: %s", what, exc)


def last_prequant_failure() -> Optional[str]:
    text = getattr(_LAST_FAILURE, "text", None)
    _LAST_FAILURE.text = None
    return text


def _unreadable_why(scheme: str) -> str:
    try:
        import torch
        import torchao
        ao = getattr(torchao, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "torch or torchao is unavailable"
    if not _tuple_safe_globals_supported():
        return f"torch {torch.__version__} is older than 2.6"
    try:
        from core._torchao_stub import is_stubbed
        if is_stubbed("torchao"):
            return "torchao is not available on this platform"
    except Exception:  # noqa: BLE001
        pass
    missing = _SCHEME_REQUIRED_GLOBALS.get(scheme, frozenset()) - _RESOLVED_SAFE_GLOBALS
    if missing:
        return (
            f"torchao {ao} no longer ships {len(missing)} class(es) the checkpoint was saved with"
        )
    return f"torch {torch.__version__} / torchao {ao} cannot deserialize it"


def prequant_unreadable_reason(
    fam: Any,
    scheme: Optional[str],
    *,
    base_repo: Optional[str] = None,
    task: Optional[str] = None,
) -> Optional[str]:
    """Status line when ``fam`` hosts a ``scheme`` checkpoint this install cannot open; never raises."""
    if not scheme:
        return None
    try:
        src = resolve_prequant_source(fam, scheme, base_repo = base_repo, task = task)
        if src is None or getattr(src, "kind", None) != "repo":
            return None
        from .prequant_safetensors import is_safetensors_checkpoint

        readable = [
            n for n in candidate_filenames_of(src) if restricted_prequant_load_supported(scheme, n)
        ]
        if readable and all(is_safetensors_checkpoint(n) for n in readable):
            declared = set(getattr(src, "declared_filenames", ()) or ())
            if (
                not declared.intersection(readable)
                and cached_checkpoint_path(src, names = readable, online = False) is None
            ):
                readable = []
        if readable:
            return None
        return (
            f"the hosted {scheme} checkpoint ({src.location}) cannot be read here: "
            f"{_unreadable_why(scheme)}"
        )
    except Exception:  # noqa: BLE001 - a status nicety must never break a load
        return None
