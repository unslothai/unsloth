# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local float16 / bfloat16 cache of components loaded from fp32 shards (Wan, Z-Image, SDXL, Lumina, ...), written
from the live tensors after publish.

Later loads read the half-size file at the same dtype (a no-op cast, bit-identical); a sampled source comparison
refuses a module changed in place. Shares ``diffusion_prequant_cache``'s root, locks, manifest and LRU.
Kill switch: ``UNSLOTH_DIFFUSION_CAST_CACHE=0`` (``UNSLOTH_DIFFUSION_PREQUANT_CACHE=0`` disables it too);
``UNSLOTH_DIFFUSION_CAST_CACHE=fp16`` / ``=bf16`` keeps one target dtype only; ``UNSLOTH_DIFFUSION_CAST_CACHE_IMAGE=0``
turns the image pipeline path off.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from . import diffusion_prequant_cache as _cache

CAST_CACHE_ENV = "UNSLOTH_DIFFUSION_CAST_CACHE"
CAST_CACHE_IMAGE_ENV = "UNSLOTH_DIFFUSION_CAST_CACHE_IMAGE"
_OFF = ("0", "off", "false", "no", "none")
_HALF = {"float16": "float16", "fp16": "float16", "bfloat16": "bfloat16", "bf16": "bfloat16"}
# 2: layout-1 entries written under block streaming may hold tensors read before their upload landed.
CAST_LAYOUT_VERSION = 2
CAST_META_KEY = "unsloth_cast_component"
_LIBRARIES = ("diffusers", "transformers")
_STEMS = {"diffusers": "diffusion_pytorch_model", "transformers": "model"}
_WIDE_TAGS = ("F32", "F64")
_FLOAT_TAGS = ("F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "F8_E8M0")
# Per-shard source sample cap; a bigger read would grow the page cache.
_SAMPLE_MAX_BYTES = 64 << 20
_DROP_EVERY_BYTES = 256 << 20
# Per-component wait for group-offloaded tensors to return to host before reading the device copy.
_ONLOAD_WAIT_SECONDS = 10.0


def enabled() -> bool:
    raw = (os.environ.get(CAST_CACHE_ENV) or "").strip().lower()
    return raw not in _OFF and _cache.enabled()


def image_enabled() -> bool:
    raw = (os.environ.get(CAST_CACHE_IMAGE_ENV) or "").strip().lower()
    return raw not in _OFF and enabled()


def _target_dtypes() -> tuple:
    raw = (os.environ.get(CAST_CACHE_ENV) or "").strip().lower()
    return (_HALF[raw],) if raw in _HALF else ("float16", "bfloat16")


def _is_half(dtype: Any) -> bool:
    return str(dtype).replace("torch.", "") in _target_dtypes()


def _component_dtype(torch_dtype: Any, component: str) -> Any:
    if isinstance(torch_dtype, dict):
        return torch_dtype.get(component, torch_dtype.get("default"))
    return torch_dtype


def _minor(module_name: str) -> Optional[str]:
    try:
        import importlib.metadata as md
        return _cache._minor(md.version(module_name))
    except Exception:  # noqa: BLE001
        return None


@dataclass
class CastCandidate:
    component: str
    library: str
    class_name: str
    cls: Any
    dtype: Any
    source: str
    shards: list
    config_path: str
    key: str
    parts: dict
    names: list = field(default_factory = list)
    nbytes: int = 0


def _read_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    try:
        with open(path, "r", encoding = "utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _shards_of(
    source: str, component: str, stem: str, cache_dir: Optional[str], revision: Optional[str]
) -> Optional[list]:
    """The shard files ``from_pretrained(source/component)`` reads, or None when unsure (both layouts, or any missing)."""
    index = _cache._local_file(
        source, f"{component}/{stem}.safetensors.index.json", cache_dir, revision
    )
    single = _cache._local_file(source, f"{component}/{stem}.safetensors", cache_dir, revision)
    if index is not None and single is not None:
        return None
    if single is not None:
        return [str(single)]
    weight_map = (_read_json(index) or {}).get("weight_map") or {}
    names = sorted(set(str(v) for v in weight_map.values()))
    if not names:
        return None
    paths = [_cache._local_file(source, f"{component}/{n}", cache_dir, revision) for n in names]
    return [str(p) for p in paths] if all(paths) else None


def _shard_header(path: str) -> Optional[dict]:
    raw = _cache.read_header_bytes(path)
    if raw is None:
        return None
    try:
        header = json.loads(raw[8:].decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    header.pop("__metadata__", None)
    return header


def _cast_bytes(shards: list) -> Optional[int]:
    """16-bit bytes if every floating tensor is fp32 / fp64, else None. An estimate: fp32-kept modules count at 16 bits."""
    seen = False
    total = 0
    for shard in shards:
        header = _shard_header(shard)
        if header is None:
            return None
        for entry in header.values():
            if not isinstance(entry, dict):
                continue
            tag = entry.get("dtype")
            try:
                begin, end = (int(v) for v in entry.get("data_offsets") or (0, 0))
            except (TypeError, ValueError):
                return None
            if tag in _FLOAT_TAGS:
                if tag not in _WIDE_TAGS:
                    return None
                seen = True
                total += (end - begin) // (4 if tag == "F32" else 8) * 2
            else:
                total += end - begin
    return total if seen else None


def _all_wide(shards: list) -> bool:
    return _cast_bytes(shards) is not None


def _fit_budget(cands: list, logger: Any = None) -> list:
    """A set over the LRU budget would evict its own entries every load. Greedy largest-first, so it is deterministic."""
    budget = _cache.max_cache_bytes()
    if budget is None:
        return cands
    keep, used = set(), 0
    for cand in sorted(cands, key = lambda c: (-c.nbytes, c.component)):
        if used + cand.nbytes > budget:
            _log(
                logger,
                "info",
                f"{cand.component}: not cached ({cand.nbytes / 2**30:.1f} GiB would not fit the "
                f"{budget / 2**30:.0f} GiB cache budget beside this load's other components)",
            )
            continue
        used += cand.nbytes
        keep.add(cand.component)
    return [c for c in cands if c.component in keep]


def _load_class(library: str, class_name: str) -> Any:
    try:
        if library == "diffusers":
            import diffusers as module
        else:
            import transformers as module
        return getattr(module, class_name, None)
    except Exception:  # noqa: BLE001
        return None


def candidates(
    source: Any,
    *,
    torch_dtype: Any,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    skip: Iterable[str] = (),
    base_repo: Optional[str] = None,
    logger: Any = None,
) -> list:
    """Components at ``source`` that load fp32 shards at float16 / bfloat16, with cache keys. Local reads only; never
    raises."""
    if not enabled() or not source:
        return []
    out: list = []
    try:
        import torch

        source = str(source)
        index = _read_json(_cache._local_file(source, "model_index.json", cache_dir, revision))
        if index is None:
            return []
        skip = set(skip)
        for component, spec in sorted(index.items()):
            if component.startswith("_") or component in skip:
                continue
            if not isinstance(spec, (list, tuple)) or len(spec) != 2:
                continue
            library, class_name = spec
            if library not in _LIBRARIES or not isinstance(class_name, str):
                continue
            dtype = _component_dtype(torch_dtype, component)
            if not _is_half(dtype):
                continue
            cls = _load_class(library, class_name)
            if cls is None or not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
                continue
            shards = _shards_of(source, component, _STEMS[library], cache_dir, revision)
            nbytes = _cast_bytes(shards) if shards else None
            if nbytes is None:
                continue
            config_path = _cache._local_file(
                source, f"{component}/config.json", cache_dir, revision
            )
            if config_path is None:
                continue
            fp = _cache.source_fingerprint(shards)
            if fp is None:
                continue
            parts = {
                "kind": "cast",
                "layout": CAST_LAYOUT_VERSION,
                "policy": _cache.POLICY_VERSION,
                "component": component,
                "library": library,
                "class": class_name,
                "dtype": str(dtype).replace("torch.", ""),
                "keep_in_fp32": sorted(
                    str(m) for m in (getattr(cls, "_keep_in_fp32_modules", None) or ())
                ),
                "source": fp,
                "config": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                "base": _cache.base_identity(base_repo or source),
                "torch": _cache._minor(torch.__version__),
                library: _minor(library),
            }
            if class_name == "AutoencoderKL":
                # Retires entries written from an fp16-decode-rescaled VAE.
                parts["vae_fp16_decode"] = 1
            out.append(
                CastCandidate(
                    component = component,
                    library = library,
                    class_name = class_name,
                    cls = cls,
                    dtype = dtype,
                    source = source,
                    shards = shards,
                    config_path = str(config_path),
                    key = _cache.cache_key(parts),
                    parts = parts,
                    nbytes = nbytes,
                )
            )
    except Exception as exc:  # noqa: BLE001
        _log(logger, "warning", f"candidates skipped ({type(exc).__name__}: {str(exc)[:200]})")
        return []
    return _fit_budget(out, logger)


def cached_components(cands: list) -> list:
    try:
        root = _cache.cache_root()
        return [c.component for c in cands or () if (root / c.key / _cache.MANIFEST_NAME).is_file()]
    except Exception:  # noqa: BLE001
        return []


def inject(
    pipe_kwargs: dict,
    cands: list,
    *,
    logger: Any = None,
    cancel_check: Optional[Any] = None,
    label: str = "diffusion",
) -> list:
    """Returns the misses. Never raises, except a cancel from ``cancel_check``."""
    try:
        cands = [c for c in cands or () if c.component not in pipe_kwargs]
        if not cands:
            return []
        hits = read_all(cands, logger = logger, cancel_check = cancel_check)
        pipe_kwargs.update(hits)
        if hits and logger is not None:
            logger.info(
                "%s.cast_cache: read %s from the local cast cache", label, ", ".join(sorted(hits))
            )
        return [c for c in cands if c.component not in hits]
    except Exception as exc:  # noqa: BLE001 - a cache problem is a miss
        if cancel_check is not None:
            cancel_check()
        _log(logger, "warning", f"inject skipped ({type(exc).__name__}: {str(exc)[:200]})")
        return []


def _log(logger: Any, level: str, msg: str) -> None:
    if logger is not None:
        try:
            getattr(logger, level)("diffusion.cast_cache: %s", msg)
        except Exception:  # noqa: BLE001
            pass


def _header_metadata(path: Any) -> Optional[dict]:
    raw = _cache.read_header_bytes(path)
    if raw is None:
        return None
    try:
        meta = json.loads(raw[8:].decode("utf-8")).get("__metadata__") or {}
        return json.loads(meta[CAST_META_KEY])
    except Exception:  # noqa: BLE001
        return None


def _weights_link(kdir: Path, library: str) -> None:
    """``from_pretrained`` reads ``<stem>.safetensors``; the entry's artifact is ``model.safetensors``."""
    name = f"{_STEMS[library]}.safetensors"
    if name == _cache.FILE_NAME:
        return
    link = kdir / name
    if link.exists():
        return
    try:
        os.symlink(_cache.FILE_NAME, link)
    except OSError:
        os.link(kdir / _cache.FILE_NAME, link)  # no symlink rights (Windows): a hard link


def _load_kwargs(cand: CastCandidate) -> dict:
    kwargs: dict = {"low_cpu_mem_usage": True, "use_safetensors": True, "output_loading_info": True}
    use_dtype_kw = False
    if cand.library == "transformers":
        try:
            import transformers
            from packaging import version
            use_dtype_kw = version.parse(transformers.__version__) >= version.parse("4.56.0")
        except Exception:  # noqa: BLE001
            use_dtype_kw = False
    kwargs["dtype" if use_dtype_kw else "torch_dtype"] = cand.dtype
    return kwargs


def read(
    cand: CastCandidate,
    *,
    logger: Any = None,
    cancel_check: Optional[Any] = None,
) -> Any:
    """The cached cast ``cand.component``, or None (a miss: the dense load runs). Never raises."""
    if not enabled():
        return None
    try:
        root = _cache.cache_root()
        if not (root / cand.key / _cache.MANIFEST_NAME).is_file():
            return None
        if _cache._hub_source_stale(cand.shards, logger):
            return None
        with _cache._cache_lock():
            artifact = _cache._entry(cand.key)
            if artifact is None:
                return None
            with _cache._lock:
                if os.path.realpath(str(artifact)) in _cache._REJECTED:
                    return None
            _cache._touch(artifact.parent)
        meta = _header_metadata(artifact)
        if not isinstance(meta, dict) or meta.get("key") != cand.key:
            _cache.note_rejected(artifact, logger)
            return None
        if not _cache.verify_entry(artifact, logger):
            _cache.note_rejected(artifact, logger)
            return None
        kdir = artifact.parent
        if not (kdir / "config.json").is_file():
            _cache.note_rejected(artifact, logger)
            return None
        _weights_link(kdir, cand.library)
        t0 = time.perf_counter()
        model, info = cand.cls.from_pretrained(str(kdir), **_load_kwargs(cand))
        bad = {
            k: list(v)
            for k, v in (info or {}).items()
            if k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") and v
        }
        if bad:
            _log(
                logger,
                "warning",
                f"{cand.component}: entry did not load cleanly ({str(bad)[:200]}); not used",
            )
            _cache.note_rejected(artifact, logger)
            return None
        if any(t.device.type == "meta" for t in (*model.parameters(), *model.buffers())):
            _cache.note_rejected(artifact, logger)
            return None
        model.eval()
        _log(
            logger,
            "info",
            f"{cand.component}: read the cached {meta.get('dtype')} cast ({artifact.stat().st_size / 2**30:.2f} GiB) "
            f"in {time.perf_counter() - t0:.1f} s instead of the {meta.get('source_gib', '?')} GiB source shards",
        )
        _cache.note_hit(artifact, logger)
        return model
    except Exception as exc:  # noqa: BLE001 - an unusable entry is a miss
        _log(
            logger,
            "warning",
            f"{cand.component}: read failed: {type(exc).__name__}: {str(exc)[:200]}",
        )
        return None


def read_all(
    cands: list,
    *,
    logger: Any = None,
    cancel_check: Optional[Any] = None,
) -> dict:
    """``{component: module}`` of every cached candidate. Never raises."""
    out: dict = {}
    for cand in cands or ():
        if cancel_check is not None:
            cancel_check()
        module = read(cand, logger = logger, cancel_check = cancel_check)
        if module is not None:
            out[cand.component] = module
    return out


def _tied_keys(module: Any) -> set:
    keys: set = set()
    for attr in ("_tied_weights_keys", "all_tied_weights_keys"):
        value = getattr(module, attr, None)
        if isinstance(value, dict):
            keys.update(str(k) for k in value)
        elif isinstance(value, (list, tuple, set)):
            keys.update(str(k) for k in value)
    return keys


def _resolve_ref(module: Any, name: str) -> Any:
    """The Parameter / buffer itself (offload groups key host copies by it)."""
    parent_name, _, leaf = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    tensor = (
        parent._parameters.get(leaf) if leaf in parent._parameters else parent._buffers.get(leaf)
    )
    if tensor is None:
        raise KeyError(name)
    return tensor


def _resolve(module: Any, name: str) -> Any:
    tensor = _resolve_ref(module, name)
    return tensor.data if hasattr(tensor, "data") else tensor


def _offload_groups(module: Any) -> list:
    out: list = []
    seen: set = set()
    for m in module.modules():
        hooks = getattr(getattr(m, "_diffusers_hook", None), "hooks", None)
        if not isinstance(hooks, dict):
            continue
        for hook in list(hooks.values()):
            group = getattr(hook, "group", None)
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            if isinstance(getattr(group, "cpu_param_dict", None), dict):
                out.append(group)
    return out


def _state_names(module: Any) -> Optional[list]:
    """``[(name, dtype, shape)]`` to write: the state dict minus tied aliases ``from_pretrained`` restores."""
    import torch

    tied = _tied_keys(module)
    seen: dict = {}
    out = []
    for name, tensor in module.state_dict(keep_vars = True).items():
        if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.device.type == "meta":
            return None
        ident = (tensor.data_ptr(), tensor.dtype, tuple(tensor.shape), tensor.device)
        if tensor.numel() and ident in seen and name in tied:
            continue
        seen.setdefault(ident, name)
        out.append((name, tensor.dtype, tuple(tensor.shape)))
    return out


def capture(
    pipe: Any,
    cands: list,
    *,
    logger: Any = None,
) -> list:
    """Right after ``from_pretrained``: record what each missed candidate's module holds (names only, no tensors)."""
    out = []
    for cand in cands or ():
        try:
            module = getattr(pipe, cand.component, None)
            if module is None or type(module).__name__ != cand.class_name:
                continue
            names = _state_names(module)
            if not names:
                continue
            cand.names = names
            out.append(cand)
        except Exception as exc:  # noqa: BLE001
            _log(logger, "info", f"{cand.component}: not cached ({type(exc).__name__}: {exc})")
    return out


def _hooked(module: Any) -> bool:
    return any(
        getattr(m, "_diffusers_hook", None) is not None or getattr(m, "_hf_hook", None) is not None
        for m in module.modules()
    )


class _Reader:
    """Host copies of one component's live tensors, read while the pipeline may be rendering.

    Streamed tensors come from their offload group's host copy: device ``.data`` is re-pointed before the prefetch
    lands, so reading it races the copy stream (NaN entries). Other device tensors are read on a side stream after
    a per-read device synchronise, never behind the render on the default stream."""

    def __init__(self, module: Any, check: Any) -> None:
        self.module = module
        self.check = check
        self.groups = _offload_groups(module)
        self.wait = _ONLOAD_WAIT_SECONDS if _hooked(module) else 0.0
        self.stream = None

    def _host_copy(self, ref: Any) -> Any:
        for group in self.groups:
            try:
                src = group.cpu_param_dict.get(ref)
            except Exception:  # noqa: BLE001
                src = None
            if (
                src is not None
                and getattr(src, "device", None) is not None
                and src.device.type == "cpu"
            ):
                return src
        return None

    def tensor(self, name: str) -> Any:
        while True:
            ref = _resolve_ref(self.module, name)
            host = self._host_copy(ref)
            if host is not None:
                return host.data if hasattr(host, "data") else host
            tensor = ref.data if hasattr(ref, "data") else ref
            if tensor.device.type == "cpu":
                return tensor
            if self.wait <= 0:
                break
            self.check()
            time.sleep(0.25)
            self.wait -= 0.25
        import torch

        if tensor.device.type != "cuda":
            return tensor.to("cpu")
        torch.cuda.synchronize(tensor.device)
        if self.stream is None:
            self.stream = torch.cuda.Stream(device = tensor.device)
        with torch.cuda.stream(self.stream):
            host = tensor.to("cpu", non_blocking = False)
        self.stream.synchronize()
        return host


def _sample_keys(header: dict, names: dict) -> list:
    """The largest cast tensor up to ``_SAMPLE_MAX_BYTES`` and the smallest tensor kept wide, of one shard."""
    import torch

    cast, wide = [], []
    for key, entry in header.items():
        if key not in names or entry.get("dtype") not in _WIDE_TAGS:
            continue
        begin, end = (int(v) for v in entry["data_offsets"])
        if end - begin > _SAMPLE_MAX_BYTES:
            continue
        (wide if names[key] in (torch.float32, torch.float64) else cast).append((end - begin, key))
    return ([max(cast)[1]] if cast else []) + ([min(wide)[1]] if wide else [])


def _matches_source(cand: CastCandidate, reader: _Reader) -> Optional[str]:
    """None when sampled live tensors equal the source cast bit for bit, else why not (a module changed in place)."""
    import torch
    from safetensors import safe_open

    names = {n: d for n, d, _s in cand.names}
    sampled = 0
    for shard in cand.shards:
        picks = _sample_keys(_shard_header(shard) or {}, names)
        if not picks:
            continue
        with safe_open(shard, framework = "pt") as fh:
            for key in picks:
                reader.check()
                live = reader.tensor(key)
                src = fh.get_tensor(key)
                if tuple(src.shape) != tuple(live.shape):
                    return f"{key} changed shape"
                if not torch.equal(
                    src.to(live.dtype).view(-1).view(torch.uint8),
                    live.reshape(-1).view(torch.uint8),
                ):
                    return f"{key} differs from its source cast"
                del src, live
                sampled += 1
    return None if sampled else "no tensor to compare against the source"


def _drop_pages(fd: int, upto: int) -> None:
    try:
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, upto, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def _write(path: str, cand: CastCandidate, metadata: dict, hasher: Any, reader: "_Reader") -> int:
    """safetensors of ``cand.names`` read live one tensor at a time (``save_file``'s order, so mmap stays aligned)."""
    import torch

    order = sorted(
        cand.names,
        key = lambda e: (-torch.empty((), dtype = e[1]).element_size(), e[0]),
    )
    # transformers 4.x refuses a safetensors file whose metadata does not name its framework.
    header: dict = {
        "__metadata__": {"format": "pt", **{str(k): str(v) for k, v in metadata.items()}}
    }
    offset = 0
    for name, dtype, shape in order:
        tag = _cache._DTYPE_TAGS.get(str(dtype).replace("torch.", ""))
        if tag is None:
            raise ValueError(f"{name}: dtype {dtype} has no safetensors tag")
        count = 1
        for d in shape:
            count *= int(d)
        size = count * torch.empty((), dtype = dtype).element_size()
        header[name] = {"dtype": tag, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators = (",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    written = 0
    dropped = 0
    with open(path, "wb") as fh:
        for chunk in (struct.pack("<Q", len(blob)), blob):
            fh.write(chunk)
            hasher.update(chunk)
            written += len(chunk)
        for name, dtype, shape in order:
            reader.check()
            tensor = reader.tensor(name)
            if tensor.dtype != dtype or tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"{name} changed after the load")
            if tensor.numel() == 0:
                continue
            data = memoryview(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy())
            fh.write(data)
            hasher.update(data)
            written += len(data)
            del data, tensor
            if written - dropped >= _DROP_EVERY_BYTES:
                fh.flush()
                _drop_pages(fh.fileno(), written)
                dropped = written
        fh.flush()
        os.fsync(fh.fileno())
        _drop_pages(fh.fileno(), written)
    return 8 + len(blob) + offset


def save(
    cand: CastCandidate,
    module: Any,
    *,
    logger: Any = None,
    check: Optional[Any] = None,
) -> Optional[str]:
    """Persist ``module`` (the cast ``cand.component``); returns the artifact path or None (logged). Never raises."""
    check = check or (lambda: None)
    t0 = time.perf_counter()
    root = _cache.cache_root()
    try:
        names = _state_names(module)
        live = {n: (d, s) for n, d, s in names or ()}
        # Tensors added after the load (fuse_qkv's to_qkv) are left out: the next load re-adds them from the same
        # weights. A renamed / removed / re-typed one (LoRA wrap, quantize, layerwise fp8 storage) refuses the write.
        if names is None or any(n not in live for n, _d, _s in cand.names):
            _log(
                logger, "info", f"{cand.component}: not cached (its tensors changed after the load)"
            )
            return None
        if any(live[n] != (d, s) for n, d, s in cand.names):
            _log(
                logger,
                "info",
                f"{cand.component}: not cached (a tensor changed dtype or shape after the load)",
            )
            return None
        # Weights were rescaled in place; a cached copy would be rescaled again on the next load.
        if getattr(module, "_unsloth_fp16_decode", False):
            _log(
                logger, "info", f"{cand.component}: not cached (rescaled in place for fp16 decode)"
            )
            return None
        names = list(cand.names)
        reader = _Reader(module, check)
        why = _matches_source(cand, reader)
        if why is not None:
            _log(logger, "info", f"{cand.component}: not cached ({why})")
            return None
        import torch

        nbytes = 0
        for _n, dtype, shape in names:
            count = 1
            for d in shape:
                count *= int(d)
            nbytes += count * torch.empty((), dtype = dtype).element_size()
        budget = _cache.max_cache_bytes()
        if budget is not None and nbytes > budget:
            _log(logger, "info", f"{cand.component}: not cached (larger than the cache budget)")
            return None
        with _cache._writer_lock(root, cand.key) as locked:
            if not locked:
                return None
            existing = _cache._entry(cand.key)
            if existing is not None:
                with _cache._lock:
                    rejected = os.path.realpath(str(existing)) in _cache._REJECTED
                if not rejected:
                    return None
                _cache._reclaim(root / cand.key)
            root.mkdir(parents = True, exist_ok = True)
            need = nbytes + max(_cache._DISK_MARGIN_BYTES, nbytes // 20)
            free = shutil.disk_usage(str(root)).free
            if free < need:
                _cache.evict(root = root, extra_free_bytes = need - free, keep = cand.key, logger = logger)
                free = shutil.disk_usage(str(root)).free
            if free < need:
                _log(
                    logger,
                    "info",
                    f"{cand.component}: not cached (disk short: {free / 2**30:.1f} GiB free, "
                    f"need {need / 2**30:.1f} GiB)",
                )
                return None
            source_bytes = sum(os.path.getsize(os.path.realpath(s)) for s in cand.shards)
            meta = {
                "key": cand.key,
                "component": cand.component,
                "class": cand.class_name,
                "dtype": cand.parts["dtype"],
                "source_gib": round(source_bytes / 2**30, 2),
                "torch_version": str(torch.__version__),
            }
            kdir = root / cand.key
            kdir.mkdir(parents = True, exist_ok = True)
            shutil.copyfile(cand.config_path, kdir / "config.json")
            fd, tmp = tempfile.mkstemp(
                dir = str(kdir), prefix = f".{_cache.FILE_NAME}.", suffix = _cache._TEMP_SUFFIX
            )
            os.close(fd)
            hasher = hashlib.sha256()
            try:
                size = _write(
                    tmp,
                    cand,
                    {CAST_META_KEY: json.dumps(meta, sort_keys = True)},
                    hasher,
                    reader,
                )
                os.replace(tmp, kdir / _cache.FILE_NAME)
                tmp = None
            finally:
                if tmp is not None:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            sha = hasher.hexdigest()
            _cache._record_verified(kdir, sha)
            manifest = {
                "key": cand.key,
                "bytes": size,
                "sha256": sha,
                "created": time.time(),
                "parts": cand.parts,
            }
            _cache._atomic_bytes(
                kdir / _cache.MANIFEST_NAME,
                json.dumps(manifest, indent = 1, default = str).encode("utf-8"),
            )
            _cache._touch(kdir)
        _log(
            logger,
            "info",
            f"{cand.component}: saved the {cand.parts['dtype']} cast ({cand.key}, {size / 2**30:.2f} GiB from "
            f"{source_bytes / 2**30:.2f} GiB of source shards, in {time.perf_counter() - t0:.1f} s)",
        )
        _cache.evict(logger = logger, keep = cand.key)
        return str(kdir / _cache.FILE_NAME)
    except Exception as exc:  # noqa: BLE001 - a cache write never fails a load
        _log(
            logger,
            "warning",
            f"{cand.component}: save failed: {type(exc).__name__}: {str(exc)[:200]}",
        )
        return None


class CastWrites:
    """Background writes of one load's cast components; registered with the prequant cache's background set."""

    def __init__(
        self,
        pipe: Any,
        cands: list,
        logger: Any = None,
    ) -> None:
        import weakref

        self._pipe = weakref.ref(pipe)
        self._cands = list(cands)
        self.key = ",".join(c.key for c in self._cands)
        self._logger = logger
        self._cancel = threading.Event()
        self._finished = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.results: dict = {}

    def _check(self) -> None:
        if self._cancel.is_set():
            raise RuntimeError("cast cache write cancelled")

    def start(self) -> bool:
        with _cache._lock:
            if self._thread is not None or self._cancel.is_set():
                return False
            self._thread = threading.Thread(target = self._run, name = "cast-cache-write", daemon = True)
            _cache._BACKGROUND.add(self)
        try:
            self._thread.start()
        except Exception:  # noqa: BLE001
            with _cache._lock:
                self._thread = None
            self._close()
            return False
        return True

    def _run(self) -> None:
        try:
            for cand in self._cands:
                self._check()
                pipe = self._pipe()
                module = getattr(pipe, cand.component, None) if pipe is not None else None
                pipe = None
                # torch.compile(module) wraps it; the weights (and their names) are the original's.
                module = getattr(module, "_orig_mod", module)
                if module is None:
                    continue
                self.results[cand.component] = save(
                    cand, module, logger = self._logger, check = self._check
                )
                module = None
        except Exception as exc:  # noqa: BLE001
            _log(self._logger, "info", f"background write stopped ({str(exc)[:200]})")
        finally:
            self._close()

    def _close(self) -> None:
        self._cands = []
        with _cache._lock:
            _cache._BACKGROUND.discard(self)
        self._finished.set()

    def discard(self) -> None:
        with _cache._lock:
            started = self._thread is not None
            self._cancel.set()
        if not started:
            self._close()

    def cancel(self) -> None:
        self.discard()

    def done(self) -> bool:
        return self._finished.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._finished.wait(timeout)


def defer_writes(
    pipe: Any,
    cands: list,
    *,
    logger: Any = None,
) -> Optional[CastWrites]:
    """Capture the missed candidates of a just-loaded ``pipe``; ``start()`` the result after publish. Never raises."""
    try:
        if not enabled() or not cands:
            return None
        captured = capture(pipe, cands, logger = logger)
        return CastWrites(pipe, captured, logger) if captured else None
    except Exception as exc:  # noqa: BLE001
        _log(logger, "warning", f"write skipped ({type(exc).__name__}: {exc})")
        return None


def finish(writes: Any, timeout: float = 60.0) -> None:
    """Teardown: let a running write finish for ``timeout`` s, then cancel it. Never raises."""
    if writes is None:
        return
    try:
        if not writes.wait(timeout):
            writes.cancel()
            writes.wait(30.0)
    except Exception:  # noqa: BLE001
        pass
