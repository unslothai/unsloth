# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows converting loads and integrated NVIDIA loads use pread instead of mmap. A ROCm integrated GPU (unified memory)
uses pread only when the load would pressure host memory: pread costs it ~6% load time and saves ~40% peak RSS.
UNSLOTH_SAFETENSORS_PREAD=auto (default) | pread (=1) | mmap (=0)."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import struct
import sys
import threading
from typing import Any, Iterable, Iterator, Optional

PREAD_ENV = "UNSLOTH_SAFETENSORS_PREAD"
FADVISE_ENV = "UNSLOTH_SAFETENSORS_FADVISE"
GiB = 1 << 30
# ROCm iGPU: pread when MemAvailable - checkpoint bytes < max(16 GiB, 25% RAM) (gfx1151 mmap peak RSS 30.3 vs 17.5 GiB)
PRESSURE_FLOOR_BYTES = 16 * GiB
PRESSURE_FRACTION = 0.25

_lock = threading.Lock()
_depth = 0
_original: Any = None
_diffusers_original: dict[Any, Any] = {}
_DIFFUSERS_LOADERS = ("diffusers.models.model_loading_utils", "diffusers.models.modeling_utils")
# per load thread, so concurrent media loads never share rules: (target dtype or None = fp32-only video rule, kept dirs,
# every shard, diffusers shards too[, memory-pressure gate of a ROCm integrated GPU])
_local = threading.local()
# diagnostics only
fadvise_calls = 0
last_reader_decision: Optional[dict[str, Any]] = None


def _active_rules() -> tuple[tuple[Any, ...], ...]:
    return getattr(_local, "rules", ())


def reader_mode() -> str:
    """``UNSLOTH_SAFETENSORS_PREAD``: "pread" (1/on/true/yes/pread), "mmap" (0/off/false/no/mmap), else "auto"."""
    value = (os.environ.get(PREAD_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no", "mmap"):
        return "mmap"
    if value in ("1", "on", "true", "yes", "pread"):
        return "pread"
    return "auto"


def pread_wanted(integrated_cuda: bool = False) -> bool:
    """Env wins; else Windows or an integrated NVIDIA target (DGX Spark 19.6 s mmap vs 0.84 s pread)."""
    mode = reader_mode()
    if mode != "auto":
        return mode == "pread"
    return sys.platform == "win32" or bool(integrated_cuda)


def host_memory() -> Optional[tuple[int, int]]:
    """(available, total) bytes: /proc/meminfo MemAvailable / MemTotal on Linux, psutil elsewhere. None if unknown."""
    try:
        if sys.platform.startswith("linux"):
            fields = {}
            with open("/proc/meminfo", encoding = "ascii") as fh:
                for line in fh:
                    name, _, rest = line.partition(":")
                    if name in ("MemAvailable", "MemTotal"):
                        fields[name] = int(rest.split()[0]) * 1024
            if "MemAvailable" in fields and "MemTotal" in fields:
                return fields["MemAvailable"], fields["MemTotal"]
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)
    except Exception:  # noqa: BLE001 - unknown: the caller keeps mmap
        return None


def checkpoint_bytes(shard: Any) -> int:
    """Bytes of every ``.safetensors`` file in the checkpoint holding ``shard``: the snapshot root (the dir with
    ``model_index.json`` above a component dir) or the shard's own dir. Variants count too (errs towards pread)."""
    from pathlib import Path

    path = Path(shard)
    root = path.parent
    patterns = ["*.safetensors"]
    if (root.parent / "model_index.json").exists():
        root = root.parent
        patterns.append(
            "*/*.safetensors"
        )  # component dirs, one level: never a walk of an arbitrary tree
    total = 0
    try:
        for f in (f for pattern in patterns for f in root.glob(pattern)):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def pressure_decision(
    load_bytes: int, memory: Optional[tuple[int, int]] = None
) -> tuple[bool, str]:
    """(use pread, reason) for a ROCm integrated GPU under ``auto``: pread when MemAvailable minus the bytes this load
    maps is under max(16 GiB, 25% of RAM)."""
    memory = host_memory() if memory is None else memory
    if memory is None:
        return False, "host memory unreadable"
    available, total = memory
    headroom = max(PRESSURE_FLOOR_BYTES, int(total * PRESSURE_FRACTION))
    left = available - int(load_bytes)
    numbers = f"MemAvailable {available / GiB:.1f} GiB - load {load_bytes / GiB:.1f} GiB = {left / GiB:.1f} GiB"
    if left < headroom:
        return True, f"{numbers} < headroom {headroom / GiB:.1f} GiB"
    return False, f"{numbers} >= headroom {headroom / GiB:.1f} GiB"


class _PressureGate:
    """A ROCm integrated GPU scope's reader, decided once at its first shard (that load's checkpoint size)."""

    def __init__(self) -> None:
        self.pread: Optional[bool] = None

    def decide(self, shard: Any) -> bool:
        global last_reader_decision
        if self.pread is not None:
            return self.pread
        mode = reader_mode()
        if mode == "pread":
            use, reason = True, f"{PREAD_ENV}=pread"
        else:
            load_bytes = checkpoint_bytes(shard)
            use, reason = pressure_decision(load_bytes)
        self.pread = use
        reader = "pread" if use else "mmap"
        if not use and sys.platform == "win32":
            reason += "; Windows converting loads still read through pread"
        last_reader_decision = {"reader": reader, "reason": reason, "mode": mode}
        try:
            from loggers import get_logger
            get_logger(__name__).info(
                "safetensors reader: %s on a ROCm integrated GPU (%s)", reader, reason
            )
        except Exception:  # noqa: BLE001 - logging never fails a load
            pass
        return use


def _rule_reads_all(rule: tuple, path: Any) -> tuple[bool, bool]:
    """(every shard, diffusers shards too) for one rule; a pressure gate grants both only when it picks pread."""
    _, _, every, wide = rule[:4]
    gate = rule[4] if len(rule) > 4 else None
    if gate is not None and not (every and wide) and gate.decide(path):
        return True, True
    return bool(every), bool(wide)


def _drop_wanted(path: Any) -> bool:
    """Drop read tensors' pages: only under a pressure gate that picked pread (unless UNSLOTH_SAFETENSORS_FADVISE=0)."""
    value = (os.environ.get(FADVISE_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no") or not hasattr(os, "posix_fadvise"):
        return False
    return any(len(r) > 4 and r[4] is not None and r[4].decide(path) for r in _active_rules())


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
    if any(path.parent.name in rule[1] for rule in rules):
        return "keep"
    if diffusers:
        return "pread" if any(all(_rule_reads_all(rule, path)) for rule in rules) else None
    stored = {k: v for k, v in _safetensors_bytes_by_dtype(path).items() if k != "kept"}
    if not stored:
        return None
    dominant = max(stored.items(), key = lambda kv: kv[1])[0]
    for rule in rules:
        to = rule[0]
        every, _ = _rule_reads_all(rule, path)
        if every:
            return "pread"
        if len(rule) > 4 and rule[4] is not None and sys.platform != "win32":
            # a ROCm integrated GPU's gate picked mmap: final for this rule (Windows keeps its commit rule below)
            continue
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
                if _drop_wanted(checkpoint_file):
                    return _read_dropping_pages(safe_open, str(checkpoint_file), location)
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


def _tensor_spans(filename: Any) -> dict[str, tuple[int, int]]:
    """Absolute (offset, length) of every tensor's bytes, from the safetensors header."""
    from .diffusion_component_sizes import MAX_HEADER_BYTES

    with open(filename, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        # The length is file-supplied: bound it before reading, so a corrupt shard falls back to safe_open's own check.
        if size > MAX_HEADER_BYTES or 8 + size > os.fstat(fh.fileno()).st_size:
            raise ValueError(f"{filename}: safetensors header length {size} is out of range")
        header = json.loads(fh.read(size))
    base = 8 + size
    spans = {}
    for key, entry in header.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        begin, end = entry["data_offsets"]
        spans[key] = (base + int(begin), int(end) - int(begin))
    return spans


def _read_dropping_pages(safe_open: Any, filename: str, location: str) -> dict[str, Any]:
    """Each tensor through ``pread``, then POSIX_FADV_DONTNEED on its clean file pages: a cold load does not leave the
    checkpoint cached in the pool the GPU shares (gfx1151: 28.5 -> 15.4 GiB cached). An unreadable header keeps them."""
    global fadvise_calls
    fd = None
    try:
        spans = _tensor_spans(filename)
        fd = os.open(filename, os.O_RDONLY)
    except Exception:  # noqa: BLE001
        spans = {}
    out = {}
    try:
        with safe_open(filename, framework = "pt", device = location, backend = "pread") as handle:
            for name in handle.keys():
                out[name] = handle.get_tensor(name)
                span = spans.get(name)
                if fd is not None and span is not None and span[1] > 0:
                    try:
                        os.posix_fadvise(fd, span[0], span[1], os.POSIX_FADV_DONTNEED)
                        fadvise_calls += 1
                    except OSError:
                        pass
    finally:
        if fd is not None:
            os.close(fd)
    return out


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
    integrated_rocm: bool = False,
) -> Iterator[bool]:
    """Route ``transformers.modeling_utils.safe_open`` of fp32 shards, and with ``convert_to`` of every shard stored in
    another dtype outside the ``keep`` component dirs, through ``pread``; yields whether it did. ``every_shard`` reads
    same-dtype shards through ``pread`` too: on an integrated GPU the device copy is host commit, and a mapped shard
    stays charged until its last tensor has moved. ``integrated_cuda`` implies ``every_shard`` on any OS and covers
    diffusers shards too. ``integrated_rocm`` does the same only when the load would pressure host memory (decided at
    its first shard; ``UNSLOTH_SAFETENSORS_PREAD=pread`` forces it), else the shards stay mapped. Nests; a concurrent
    load in another thread only gets the same bytes through another backend."""
    global _depth, _original
    installed = False
    previous_rules = None
    # a ROCm integrated GPU installs the scope under auto too; its gate picks the reader at the first shard
    if pread_wanted(integrated_cuda) or (integrated_rocm and reader_mode() == "auto"):
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
            rule: tuple = (to, frozenset(keep), every, bool(integrated_cuda))
            if integrated_rocm and not integrated_cuda:
                rule += (_PressureGate(),)
            _local.rules = previous_rules + (rule,)
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
