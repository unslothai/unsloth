# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows converting loads and integrated NVIDIA loads use pread instead of mmap; UNSLOTH_SAFETENSORS_PREAD=0|1."""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import threading
from typing import Any, Iterable, Iterator, Optional

PREAD_ENV = "UNSLOTH_SAFETENSORS_PREAD"

_lock = threading.Lock()
_depth = 0
_original: Any = None
_diffusers_original: dict[Any, Any] = {}
_DIFFUSERS_LOADERS = ("diffusers.models.model_loading_utils", "diffusers.models.modeling_utils")
# per load thread, so concurrent media loads never share rules: (target dtype or None = fp32-only video rule, kept dirs,
# every shard, diffusers shards too)
_local = threading.local()


def _active_rules() -> tuple[tuple[Optional[str], frozenset[str], bool, bool], ...]:
    return getattr(_local, "rules", ())


def pread_wanted(integrated_cuda: bool = False) -> bool:
    """Env wins; else Windows or an integrated NVIDIA target (DGX Spark 19.6 s mmap vs 0.84 s pread)."""
    value = (os.environ.get(PREAD_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no"):
        return False
    if value in ("1", "on", "true", "yes"):
        return True
    return sys.platform == "win32" or bool(integrated_cuda)


def _supports_backend(safe_open: Any) -> bool:
    # safetensors < 0.8 has no backend keyword (TypeError); wrappers drop __text_signature__, so read the innermost.
    fn = safe_open
    for _ in range(8):
        inner = getattr(fn, "__wrapped__", None)
        if inner is None:
            break
        fn = inner
    return "backend" in (getattr(fn, "__text_signature__", None) or "")


def pread_available() -> bool:
    """Whether a scope opened now routes shards through ``pread`` (Windows, safetensors >= 0.8, not disabled)."""
    if not pread_wanted():
        return False
    try:
        from safetensors import safe_open
        return _supports_backend(safe_open)
    except Exception:  # noqa: BLE001
        return False


def _backend_for(filename: Any, diffusers: bool = False) -> Optional[str]:
    """ "pread", "keep" (a component the small-host route loads at its stored dtype) or None (caller's default).
    Diffusers shards use ``pread`` only under an integrated-CUDA rule."""
    from pathlib import Path

    from .diffusion_small_host import _safetensors_bytes_by_dtype

    path = Path(filename)
    rules = _active_rules()
    if any(path.parent.name in keep for _, keep, _, _ in rules):
        return "keep"
    if diffusers:
        return "pread" if any(every and wide for _, _, every, wide in rules) else None
    stored = {k: v for k, v in _safetensors_bytes_by_dtype(path).items() if k != "kept"}
    if not stored:
        return None
    dominant = max(stored.items(), key = lambda kv: kv[1])[0]
    for to, _, every, _ in rules:
        if every:
            return "pread"
        if to is None and stored.get("float32", 0) * 2 > sum(stored.values()):
            return "pread"
        if to is not None and dominant in ("bfloat16", "float16", "float32") and dominant != to:
            return "pread"
    return None


def _core_pread_below(safe_open: Any) -> bool:
    # unsloth's Windows pread patch (transformers < 5.17) may sit underneath
    fn = safe_open
    for _ in range(8):
        if fn is None:
            return False
        if getattr(fn, "_unsloth_windows_pread", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def _with_pread(safe_open: Any) -> Any:
    @functools.wraps(safe_open)
    def pread_safe_open(filename: Any, *args: Any, **kwargs: Any) -> Any:
        # transformers 5.13 - 5.16 pass backend="mmap" explicitly; 5.17 already picks pread on Windows (#48341).
        if not _active_rules():
            return safe_open(filename, *args, **kwargs)
        choice = _backend_for(filename)
        if choice == "pread" and kwargs.get("backend") in (None, "mmap"):
            kwargs["backend"] = "pread"
        elif choice == "keep" and kwargs.get("backend") == "pread":
            # transformers >= 5.17 asks for pread on Windows itself; the small-host route needs this shard paged
            kwargs["backend"] = "mmap"
        if choice != "pread" and _core_pread_below(safe_open):
            # this scope decides inside a media load, over unsloth's patch too
            kwargs["_unsloth_keep_mmap"] = True
        return safe_open(filename, *args, **kwargs)

    return pread_safe_open


def _with_pread_state_dict(load_state_dict: Any) -> Any:
    import inspect

    try:
        signature: Any = inspect.signature(load_state_dict)
    except (TypeError, ValueError):
        signature = None

    def _plain_shard(checkpoint_file: Any, args: tuple, kwargs: dict) -> Optional[Any]:
        # only a plain .safetensors shard, mmap allowed, no DDUF (0.42 drops dduf_entries, so bind by name)
        if signature is None or not isinstance(checkpoint_file, (str, os.PathLike)):
            return None
        try:
            bound = signature.bind(checkpoint_file, *args, **kwargs)
        except TypeError:
            return None
        bound.apply_defaults()
        given = dict(bound.arguments)
        given.pop(next(iter(signature.parameters)), None)
        location = given.pop("map_location", "cpu")
        if (
            given.pop("disable_mmap", False)
            or given.pop("dduf_entries", None)
            or any(given.values())
        ):
            return None
        if not isinstance(location, str) or location == "meta":
            return None
        return location if str(checkpoint_file).endswith(".safetensors") else None

    @functools.wraps(load_state_dict)
    def pread_load_state_dict(checkpoint_file: Any, *args: Any, **kwargs: Any) -> Any:
        location = _plain_shard(checkpoint_file, args, kwargs) if _active_rules() else None
        if location is not None and _backend_for(checkpoint_file, diffusers = True) == "pread":
            try:
                from safetensors import safe_open
                with safe_open(
                    str(checkpoint_file), framework = "pt", device = location, backend = "pread"
                ) as handle:
                    get_all = getattr(handle, "get_tensors", None)
                    if callable(get_all):
                        return get_all()
                    return {name: handle.get_tensor(name) for name in handle.keys()}
            except Exception:  # noqa: BLE001 - diffusers' own read raises its own error on a bad shard
                pass
        return load_state_dict(checkpoint_file, *args, **kwargs)

    return pread_load_state_dict


def _install_diffusers() -> None:
    # under _lock, first scope only: diffusers already imported (a media load), safetensors with the backend keyword
    try:
        from safetensors import safe_open
        if not _supports_backend(safe_open):
            return
    except Exception:  # noqa: BLE001
        return
    for name in _DIFFUSERS_LOADERS:
        module = sys.modules.get(name)
        current = getattr(module, "load_state_dict", None)
        if module is None or not callable(current):
            continue
        _diffusers_original[module] = current
        module.load_state_dict = _with_pread_state_dict(current)


def _restore_diffusers() -> None:
    for module, original in _diffusers_original.items():
        if getattr(getattr(module, "load_state_dict", None), "__wrapped__", None) is original:
            module.load_state_dict = original
    _diffusers_original.clear()


@contextlib.contextmanager
def pread_safetensors(
    convert_to: Any = None,
    keep: Iterable[str] = (),
    every_shard: bool = False,
    integrated_cuda: bool = False,
) -> Iterator[bool]:
    """Route ``transformers.modeling_utils.safe_open`` of fp32 shards, and with ``convert_to`` of every shard stored in
    another dtype outside the ``keep`` component dirs, through ``pread``; yields whether it did. ``every_shard`` reads
    same-dtype shards through ``pread`` too: on an integrated GPU the device copy is host commit, and a mapped shard
    stays charged until its last tensor has moved. ``integrated_cuda`` implies ``every_shard`` on any OS and covers
    diffusers shards too. Nests; a concurrent load in another thread only gets the same
    bytes through another backend."""
    global _depth, _original
    installed = False
    previous_rules = None
    if pread_wanted(integrated_cuda):
        try:
            import transformers.modeling_utils as modeling_utils
            with _lock:
                current = getattr(modeling_utils, "safe_open", None)
                if _depth:
                    installed = True
                elif current is not None and _supports_backend(current):
                    _original = current
                    modeling_utils.safe_open = _with_pread(current)
                    _install_diffusers()
                    installed = True
                if installed:
                    _depth += 1
        except Exception:  # noqa: BLE001 - the mmap backend still loads wherever commit allows
            installed = False
        if installed:
            to = None if convert_to is None else str(convert_to).replace("torch.", "")
            every = bool(every_shard or integrated_cuda)
            previous_rules = _active_rules()
            _local.rules = previous_rules + ((to, frozenset(keep), every, bool(integrated_cuda)),)
    try:
        yield installed
    finally:
        if installed:
            _local.rules = previous_rules
            with _lock:
                _depth -= 1
                if not _depth:
                    import transformers.modeling_utils as modeling_utils

                    if getattr(modeling_utils.safe_open, "__wrapped__", None) is _original:
                        modeling_utils.safe_open = _original
                    _original = None
                    _restore_diffusers()
