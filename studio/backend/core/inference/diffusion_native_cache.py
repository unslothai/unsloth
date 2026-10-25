# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local cache of native (torchao-free) runtime-quantized denoisers, one entry per component.

Shares ``diffusion_prequant_cache``'s root, locks, manifest and eviction; the key adds the native build spec so an
entry is only served to the identical native build.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from typing import Any, Optional

from . import diffusion_prequant_cache as _cache

NATIVE_LAYOUT_VERSION = 1
NATIVE_META_KEY = "unsloth_native_layers"


def native_spec(
    scheme: str,
    family: Optional[str],
    *,
    act_int8: bool,
    scale_dtype: Any = None,
) -> dict:
    """Everything beside the dense weights that ``quantize_transformer``'s native path bakes into the bytes."""
    from .diffusion_native_quant import NATIVE_INT8, int8_rotation_group
    from .diffusion_transformer_quant import DEFAULT_MIN_LINEAR_FEATURES, exclude_tokens_for_scheme

    act = bool(act_int8) and scheme == NATIVE_INT8
    return {
        "layout": NATIVE_LAYOUT_VERSION,
        "scheme": scheme,
        "act_int8": act,
        "rot_group": int(int8_rotation_group()) if act else 0,
        "scale_dtype": str(scale_dtype) if scale_dtype is not None and not act else None,
        "min_features": DEFAULT_MIN_LINEAR_FEATURES,
        "exclude_name_tokens": sorted(set(exclude_tokens_for_scheme(scheme, family)) | {"lora_"}),
        "require_bf16": True,
    }


def _key_for(
    fam: Any,
    spec: dict,
    base: Optional[str],
    component: str,
    *,
    base_id: Optional[str] = None,
    shards: Optional[list] = None,
) -> Optional[tuple[str, dict]]:
    if spec.get("scheme") not in _cache.CACHEABLE_SCHEMES:
        return None
    if shards is None:
        shards = _cache.source_shards(base, component)
    if not shards:
        return None
    fp = _cache.source_fingerprint(shards)
    if fp is None:
        return None
    try:
        import torch
        torch_minor = _cache._minor(torch.__version__)
    except Exception:  # noqa: BLE001
        torch_minor = None
    parts = {
        "policy": _cache.POLICY_VERSION,
        "family": str(getattr(fam, "name", fam if isinstance(fam, str) else None) or ""),
        "transformer_class": str(getattr(fam, "transformer_class", None) or ""),
        "scheme": spec["scheme"],
        "component": component,
        "source": fp,
        "base": _cache.base_identity(base_id if base_id is not None else base),
        "torch": torch_minor,
        "native": dict(spec),
    }
    return _cache.cache_key(parts), parts


def _header_metadata(path: Any) -> Optional[dict]:
    raw = _cache.read_header_bytes(path)
    if raw is None:
        return None
    try:
        meta = json.loads(raw[8:].decode("utf-8")).get("__metadata__") or {}
        return json.loads(meta[NATIVE_META_KEY])
    except Exception:  # noqa: BLE001 - not a native entry
        return None


def lookup(
    fam: Any,
    spec: dict,
    *,
    base_repo: Optional[str],
    source_base: Optional[str] = None,
    component: str = "transformer",
    logger: Any = None,
) -> Optional[str]:
    """``source_base`` is where the dense shards are found (default ``base_repo``), as for ``save``.

    Path of a cached native build of ``component`` for this exact dense source and ``spec``, or None. Local reads
    (plus the Hub staleness probe a torchao hit also makes); never raises."""
    if not _cache.enabled():
        return None
    try:
        recorded_base = str(base_repo or getattr(fam, "base_repo", "") or "")
        source = source_base or base_repo or getattr(fam, "base_repo", None)
        key = shards = None
        seen: set = set()
        for candidate in (
            _cache.source_shards(source, component),
            *_cache._shard_sets(source, component),
        ):
            if not candidate or tuple(candidate) in seen:
                continue
            seen.add(tuple(candidate))
            keyed = _key_for(fam, spec, source, component, base_id = recorded_base, shards = candidate)
            if (
                keyed is not None
                and (_cache.cache_root() / keyed[0] / _cache.MANIFEST_NAME).is_file()
            ):
                key, shards = keyed[0], candidate
                break
        if key is None:
            return None
        if _cache._hub_source_stale(shards, logger):
            return None
        with _cache._cache_lock():
            artifact = _cache._entry(key)
            if artifact is None:
                return None
            with _cache._lock:
                rejected = os.path.realpath(str(artifact)) in _cache._REJECTED
            if rejected:
                return None
            _cache._touch(artifact.parent)
        meta = _header_metadata(artifact)
        if not isinstance(meta, dict) or meta.get("key") != key or meta.get("spec") != spec:
            _cache.note_rejected(artifact, logger)
            return None
        return str(artifact)
    except Exception as exc:  # noqa: BLE001
        _cache._log(logger, "warning", f"native lookup failed: {type(exc).__name__}: {exc}")
        return None


def _layer_facts(transformer: Any, spec: dict) -> Optional[dict]:
    """``{fqn: facts}`` of every native Linear, or None when one disagrees with ``spec`` (a build the key does not
    describe)."""
    from .diffusion_native_quant import is_native_linear

    layers: dict = {}
    for name, mod in transformer.named_modules():
        if not is_native_linear(mod):
            continue
        if mod.scheme != spec["scheme"] or bool(mod.act_int8) != bool(spec["act_int8"]):
            return None
        rot = int(getattr(mod, "rot_group", 0) or 0)
        if rot not in (0, int(spec["rot_group"])):
            return None
        layers[name] = {
            "scheme": mod.scheme,
            "act_int8": bool(mod.act_int8),
            "rot_group": rot,
            "in_features": int(mod.in_features),
            "out_features": int(mod.out_features),
            "compute_dtype": str(mod.compute_dtype).replace("torch.", ""),
            "fp8_format": str(getattr(mod, "fp8_format", "float8_e4m3fn")),
            "bias": getattr(mod, "bias", None) is not None,
        }
    return layers or None


def save(
    transformer: Any,
    fam: Any,
    spec: dict,
    *,
    base_repo: Optional[str],
    source_base: Optional[str] = None,
    component: str = "transformer",
    logger: Any = None,
    cancel_check: Optional[Any] = None,
) -> Optional[str]:
    """Persist a just-built native denoiser; returns the artifact path or None (logged). Never raises."""
    if not _cache.enabled():
        return None
    t0 = time.perf_counter()
    root = _cache.cache_root()
    try:
        import torch

        info = dict(getattr(transformer, "_unsloth_runtime_quant_info", None) or {})
        if not info.get("native") or info.get("scheme") != spec.get("scheme"):
            _cache._log(
                logger,
                "info",
                f"native {component}: skipped (not a native {spec.get('scheme')} build)",
            )
            return None
        if info.get("cache_path"):
            return None
        recorded_base = str(base_repo or getattr(fam, "base_repo", "") or "")
        keyed = _key_for(fam, spec, source_base or base_repo, component, base_id = recorded_base)
        if keyed is None:
            _cache._log(
                logger,
                "info",
                f"native {component}: skipped (dense source shards not found locally)",
            )
            return None
        key, parts = keyed
        layers = _layer_facts(transformer, spec)
        if layers is None:
            _cache._log(logger, "info", f"native {component}: skipped (layers differ from the key)")
            return None
        with _cache._writer_lock(root, key) as locked:
            if not locked:
                _cache._log(logger, "info", f"native {component}: skipped (no cross-process lock)")
                return None
            existing = _cache._entry(key)
            replacing = None
            if existing is not None:
                real = os.path.realpath(str(existing))
                with _cache._lock:
                    replacing = real if real in _cache._REJECTED else None
                if replacing is None:
                    _cache._log(
                        logger, "info", f"native {component}: skipped (already cached as {key})"
                    )
                    return None
                # an entry load() rejected (same size, bad bytes) would otherwise block every rebuild for good
                _cache._log(
                    logger, "info", f"native {component}: replacing {key} (it failed to load)"
                )
            state = transformer.state_dict()
            if any(".lora_" in k or k.startswith("lora_") for k in state):
                _cache._log(logger, "info", f"native {component}: skipped (baked LoRA adapters)")
                return None
            for name, tensor in state.items():
                if (
                    type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                    or tensor.device.type == "meta"
                ):
                    _cache._log(
                        logger,
                        "info",
                        f"native {component}: skipped ({name} is not a plain tensor)",
                    )
                    return None
            nbytes = sum(_cache._tensor_bytes(t) for t in state.values())
            budget = _cache.max_cache_bytes()
            if budget is not None and nbytes > budget:
                _cache._log(
                    logger, "info", f"native {component}: skipped (larger than the cache budget)"
                )
                return None
            root.mkdir(parents = True, exist_ok = True)
            need = nbytes + max(_cache._DISK_MARGIN_BYTES, nbytes // 20)
            free = shutil.disk_usage(str(root)).free
            if free < need:
                _cache.evict(root = root, extra_free_bytes = need - free, keep = key, logger = logger)
                free = shutil.disk_usage(str(root)).free
            if free < need:
                _cache._log(
                    logger,
                    "info",
                    f"native {component}: skipped (disk short: {free / 2**30:.1f} GiB free, "
                    f"need {need / 2**30:.1f} GiB)",
                )
                return None
            meta = {
                "key": key,
                "spec": spec,
                "family": parts["family"],
                "component": component,
                "base_model_id": recorded_base,
                "transformer_class": parts["transformer_class"] or type(transformer).__name__,
                "torch_version": str(torch.__version__),
                "layers": layers,
            }
            header = {NATIVE_META_KEY: json.dumps(meta, sort_keys = True)}
            kdir = root / key
            if replacing is not None:
                _cache._reclaim(kdir)
                with _cache._lock:
                    _cache._REJECTED.discard(replacing)
            kdir.mkdir(parents = True, exist_ok = True)
            fd, tmp = tempfile.mkstemp(
                dir = str(kdir), prefix = f".{_cache.FILE_NAME}.", suffix = _cache._TEMP_SUFFIX
            )
            os.close(fd)
            try:
                hasher = hashlib.sha256()
                size = _cache.write_streaming_safetensors(
                    tmp, state, header, cancel_check = cancel_check, hasher = hasher
                )
                del state
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
                "key": key,
                "bytes": size,
                "sha256": sha,
                "created": time.time(),
                "parts": parts,
            }
            _cache._atomic_bytes(
                kdir / _cache.MANIFEST_NAME,
                json.dumps(manifest, indent = 1, default = str).encode("utf-8"),
            )
            _cache._touch(kdir)
        _cache._log(
            logger,
            "info",
            f"native {component}: saved ({key}, {size / 2**30:.2f} GiB in {time.perf_counter() - t0:.1f} s)",
        )
        _cache.evict(logger = logger, keep = key)
        return str(kdir / _cache.FILE_NAME)
    except Exception as exc:  # noqa: BLE001 - a cache write never fails a load (the caller re-checks a cancel)
        _cache._log(
            logger,
            "warning",
            f"native {component}: save failed: {type(exc).__name__}: {str(exc)[:200]}",
        )
        return None


def load(
    path: str,
    transformer_cls: Any,
    config_source: str,
    *,
    component: str = "transformer",
    spec: dict,
    config_kwargs: Optional[dict] = None,
    logger: Any = None,
    cancel_check: Optional[Any] = None,
) -> Any:
    """Rebuild the cached native ``component`` (config from ``config_source``), or None. Never raises; a cancel
    from ``cancel_check`` is a miss that leaves the entry usable (the caller re-checks it)."""
    cancelled = [False]

    def _check() -> None:
        if cancel_check is None:
            return
        try:
            cancel_check()
        except BaseException:
            cancelled[0] = True
            raise

    try:
        import torch
        from safetensors.torch import load_file

        from .diffusion_native_quant import native_linear_class

        if not _cache.verify_entry(path, logger, cancel_check = _check):
            if not cancelled[0]:
                _cache.note_rejected(path, logger)
            return None
        meta = _header_metadata(path)
        if (
            not isinstance(meta, dict)
            or meta.get("spec") != spec
            or meta.get("component") != component
        ):
            _cache.note_rejected(path, logger)
            return None
        layers = meta.get("layers") or {}
        config = transformer_cls.load_config(
            config_source, subfolder = component, **(config_kwargs or {})
        )
        try:
            from accelerate import init_empty_weights
        except Exception:  # noqa: BLE001
            return None
        # Buffers stay real: non-persistent ones (rope tables) are built by __init__ as a dense load builds them.
        with init_empty_weights(include_buffers = False):
            model = transformer_cls.from_config(config)
        state = load_file(str(path))
        cls = native_linear_class()
        for fqn, facts in layers.items():
            parent_name, _, leaf = fqn.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            stand_in = torch.nn.Linear(
                int(facts["in_features"]),
                int(facts["out_features"]),
                bias = bool(facts["bias"]),
                device = "meta",
                dtype = getattr(torch, str(facts["compute_dtype"])),
            )
            codes = state[f"{fqn}.weight_q"]
            if facts["scheme"] != "int8":
                codes = codes.view(getattr(torch, str(facts.get("fp8_format") or "float8_e4m3fn")))
            scale = state[f"{fqn}.weight_scale"].view(torch.float32)
            setattr(
                parent,
                leaf,
                cls(
                    stand_in,
                    facts["scheme"],
                    act_int8 = bool(facts["act_int8"]),
                    rot_group = int(facts["rot_group"] or 0),
                    codes = codes,
                    scale = scale,
                ),
            )
        model.load_state_dict(state, strict = True, assign = True)
        del state
        if any(t.device.type == "meta" for t in (*model.parameters(), *model.buffers())):
            _cache._log(logger, "warning", f"native {component}: entry left meta tensors; not used")
            _cache.note_rejected(path, logger)
            return None
        model.eval()
        try:
            model._unsloth_runtime_quant = spec["scheme"]
            model._unsloth_runtime_quant_info = {
                "scheme": spec["scheme"],
                "native": True,
                "cache_path": str(path),
            }
        except Exception:  # noqa: BLE001
            pass
        _cache.note_hit(path, logger)
        return model
    except Exception as exc:  # noqa: BLE001 - an unusable entry is a miss
        if cancelled[0]:
            return None
        _cache._log(
            logger,
            "warning",
            f"native {component}: read failed: {type(exc).__name__}: {str(exc)[:200]}",
        )
        try:
            _cache.note_rejected(path, logger)
        except Exception:  # noqa: BLE001
            pass
        return None


STREAM_BUILD_ENV = "UNSLOTH_DIFFUSION_NATIVE_STREAM_BUILD"
# Dtypes a bf16 load never casts; any other floating dtype falls back to the dense load.
_STREAM_DTYPES = {
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def stream_build_enabled() -> bool:
    value = (os.environ.get(STREAM_BUILD_ENV) or "").strip().lower()
    return value not in ("0", "off", "false", "no")


def _dense_shards(source: Any, component: str, cache_dir: Optional[str]) -> Optional[list[str]]:
    """The shard files ``from_pretrained(source, subfolder = component)`` reads (index first, as diffusers), or None."""
    stem = f"{component}/diffusion_pytorch_model"
    index = _cache._local_file(str(source), f"{stem}.safetensors.index.json", cache_dir)
    if index is not None:
        with open(index, "r", encoding = "utf-8") as fh:
            weight_map = (json.load(fh) or {}).get("weight_map") or {}
        names = sorted(set(str(v) for v in weight_map.values()))
        paths = [_cache._local_file(str(source), f"{component}/{n}", cache_dir) for n in names]
        return [str(p) for p in paths] if names and all(paths) else None
    single = _cache._local_file(str(source), f"{stem}.safetensors", cache_dir)
    return [str(single)] if single else None


def _shard_entries(path: str) -> Optional[tuple[int, dict]]:
    """(data start, ``{key: header entry}``) of a shard holding only dtypes in ``_STREAM_DTYPES``, else None."""
    raw = _cache.read_header_bytes(path)
    if raw is None:
        return None
    header = json.loads(raw[8:].decode("utf-8"))
    header.pop("__metadata__", None)
    if any(entry.get("dtype") not in _STREAM_DTYPES for entry in header.values()):
        return None
    return len(raw), header


def _map_tensor(fd: int, start: int, entry: dict) -> Any:
    """One tensor over its own private mapping, unmapped when freed (a whole-shard mapping stays resident while any
    tensor survives)."""
    import mmap

    import torch

    begin, end = (int(v) for v in entry["data_offsets"])
    dtype = getattr(torch, _STREAM_DTYPES[entry["dtype"]])
    shape = [int(d) for d in entry["shape"]]
    count = 1
    for d in shape:
        count *= d
    if end - begin != count * torch.empty((), dtype = dtype).element_size():
        raise ValueError("shard entry size does not match its shape")
    if end == begin:
        return torch.empty(shape, dtype = dtype)
    offset = start + begin
    aligned = offset - offset % mmap.ALLOCATIONGRANULARITY
    # Copy-on-write: writable, nothing reaches the file.
    window = mmap.mmap(
        fd, offset + (end - begin) - aligned, offset = aligned, access = mmap.ACCESS_COPY
    )
    flat = torch.frombuffer(window, dtype = torch.uint8, count = end - begin, offset = offset - aligned)
    return flat.view(dtype).reshape(shape)


def build_from_shards(
    transformer_cls: Any,
    config_source: str,
    spec: dict,
    *,
    family: Optional[str],
    dtype: Any,
    component: str = "transformer",
    config_kwargs: Optional[dict] = None,
    logger: Any = None,
    cancel_check: Optional[Any] = None,
) -> Any:
    """Build the native ``component`` per tensor, bit-identical to the in-place build, or None for the dense load
    (also on cancel, which the caller re-checks). Never raises."""
    cancelled = [False]

    def _check() -> None:
        if cancel_check is None:
            return
        try:
            cancel_check()
        except BaseException:
            cancelled[0] = True
            raise

    t0 = time.perf_counter()
    try:
        import torch

        from .diffusion_native_quant import NATIVE_INT8, native_linear_class
        from .diffusion_transformer_quant import exclude_tokens_for_scheme, make_filter_fn

        def skip(why: str) -> None:
            _cache._log(logger, "info", f"native {component}: per-tensor build skipped ({why})")

        if not stream_build_enabled() or transformer_cls is None:
            return None
        if dtype is not torch.bfloat16:
            return skip(f"loads at {str(dtype).replace('torch.', '')}, not bfloat16")
        if getattr(transformer_cls, "_keep_in_fp32_modules", None):
            return skip("the class keeps fp32 modules")
        scheme = spec["scheme"]
        cache_dir = (config_kwargs or {}).get("cache_dir")
        shards = _dense_shards(config_source, component, cache_dir)
        if not shards:
            return skip("dense shards not found locally")
        if _cache._hub_source_stale(shards, logger):
            return None
        entries = []
        for shard in shards:
            found = _shard_entries(shard)
            if found is None:
                return skip("a shard holds a dtype other than bf16")
            entries.append((shard, *found))
        stored = {key: entry for _shard, _start, header in entries for key, entry in header.items()}
        _check()
        config = transformer_cls.load_config(
            config_source, subfolder = component, **(config_kwargs or {})
        )
        try:
            from accelerate import init_empty_weights
        except Exception:  # noqa: BLE001
            return skip("accelerate is missing")
        # Buffers stay real: non-persistent ones (rope tables) are built by __init__ as a dense load builds them.
        with init_empty_weights(include_buffers = False):
            model = transformer_cls.from_config(config)
        # The in-place build filters the loaded module, whose weights carry the stored dtype.
        for name, mod in model.named_modules():
            entry = stored.get(f"{name}.weight") if name else None
            if (
                isinstance(mod, torch.nn.Linear)
                and entry is not None
                and mod.weight.device.type == "meta"
            ):
                mod.weight = torch.nn.Parameter(
                    torch.empty(
                        mod.weight.shape,
                        dtype = getattr(torch, _STREAM_DTYPES[entry["dtype"]]),
                        device = "meta",
                    ),
                    requires_grad = mod.weight.requires_grad,
                )
        filter_fn = make_filter_fn(
            int(spec["min_features"]),
            exclude_name_tokens = exclude_tokens_for_scheme(scheme, family) + ("lora_",),
            require_bf16 = True,
        )
        quant = {name for name, mod in model.named_modules() if name and filter_fn(mod, name)}
        act_int8 = scheme == NATIVE_INT8 and bool(spec["act_int8"])
        rot_group = int(spec["rot_group"]) if act_int8 else 0
        cls = native_linear_class()
        state: dict = {}
        built = 0
        for shard, start, header in entries:
            fd = os.open(shard, os.O_RDONLY)
            try:
                for key, entry in sorted(
                    header.items(), key = lambda kv: int(kv[1]["data_offsets"][0])
                ):
                    _check()
                    tensor = _map_tensor(fd, start, entry)
                    fqn, _, leaf = key.rpartition(".")
                    if leaf != "weight" or fqn not in quant:
                        state[key] = tensor
                        continue
                    parent_name, _, attr = fqn.rpartition(".")
                    parent = model.get_submodule(parent_name) if parent_name else model
                    dense = getattr(parent, attr)
                    stand_in = torch.nn.Linear(
                        int(dense.in_features),
                        int(dense.out_features),
                        bias = dense.bias is not None,
                        device = "meta",
                        dtype = tensor.dtype,
                    )
                    stand_in.weight = torch.nn.Parameter(tensor, requires_grad = False)
                    del tensor
                    layer = cls(stand_in, scheme, act_int8 = act_int8, rot_group = rot_group)
                    del stand_in
                    setattr(parent, attr, layer)
                    state[f"{fqn}.weight_q"] = layer.weight_q
                    state[f"{fqn}.weight_scale"] = layer.weight_scale
                    built += 1
            finally:
                # The mappings outlive the descriptor; each closes with its tensor.
                os.close(fd)
        if built != len(quant):
            return skip(f"{len(quant) - built} quantizable weights missing from the shards")
        model.load_state_dict(state, strict = True, assign = True)
        del state
        if any(t.device.type == "meta" for t in (*model.parameters(), *model.buffers())):
            return skip("tensors left on meta")
        model.eval()
        try:
            model._unsloth_runtime_quant = scheme
            model._unsloth_runtime_quant_info = {"scheme": scheme, "native": True}
        except Exception:  # noqa: BLE001
            pass
        _cache._log(
            logger,
            "info",
            f"native {component}: {scheme} "
            + (
                ("W8A8 torch._int_mm" + (f" ConvRot g{rot_group}" if rot_group else ""))
                if act_int8
                else "weight-only"
            )
            + f" built per tensor from {len(shards)} dense shard(s) on {built} linears "
            f"in {time.perf_counter() - t0:.1f} s",
        )
        return model
    except Exception as exc:  # noqa: BLE001 - the caller loads the dense module whole instead
        if cancelled[0]:
            return None
        _cache._log(
            logger,
            "warning",
            f"native {component}: per-tensor build failed: {type(exc).__name__}: {str(exc)[:200]}",
        )
        return None


def build_spec(transformer: Any, family: Optional[str]) -> Optional[dict]:
    """``native_spec`` of an in-memory native build, read from its layers, or None (no native layers, or a mix)."""
    try:
        from .diffusion_native_quant import is_native_linear

        schemes, acts = set(), set()
        for mod in transformer.modules():
            if is_native_linear(mod):
                schemes.add(mod.scheme)
                acts.add(bool(mod.act_int8))
        if len(schemes) != 1 or len(acts) != 1:
            return None
        return native_spec(next(iter(schemes)), family, act_int8 = next(iter(acts)))
    except Exception:  # noqa: BLE001
        return None


class NativePendingWrite:
    """Background ``save``, awaited by ``finish_background_writes``; writes live host tensors (no snapshot copy)."""

    def __init__(
        self,
        run: Any,
        key: str,
        logger: Any = None,
    ) -> None:
        import threading

        self._run_fn = run
        self.key = key
        self._logger = logger
        self._cancel = threading.Event()
        self._finished = threading.Event()
        self._thread: Optional[Any] = None
        self.result: Optional[str] = None
        with _cache._lock:
            _cache._BACKGROUND.add(self)

    def _check(self) -> None:
        if self._cancel.is_set():
            raise RuntimeError("native cache write cancelled")

    def start(self) -> bool:
        import threading

        with _cache._lock:
            if self._thread is not None or self._run_fn is None or self._cancel.is_set():
                return False
            self._thread = threading.Thread(
                target = self._run, name = f"native-cache-write-{self.key[:8]}", daemon = True
            )
        try:
            self._thread.start()
        except Exception:  # noqa: BLE001 - no writer: the next load re-quantizes
            with _cache._lock:
                self._thread = None
            self._close()
            return False
        return True

    def _run(self) -> None:
        try:
            if self._run_fn is not None:
                self.result = self._run_fn(self._check)
        except Exception as exc:  # noqa: BLE001 - a background cache write never surfaces
            _cache._log(self._logger, "warning", f"native: background save failed: {exc}")
        finally:
            self._close()

    def _close(self) -> None:
        self._run_fn = None
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


def save_deferred(
    transformer: Any,
    fam: Any,
    *,
    base_repo: Optional[str],
    source_base: Optional[str] = None,
    component: str = "transformer",
    logger: Any = None,
    cancel_check: Optional[Any] = None,
) -> Any:
    """A ``NativePendingWrite`` (start after publish), the sync ``save`` result, or None. Never raises."""
    try:
        if not _cache.enabled():
            return None
        spec = build_spec(transformer, getattr(fam, "name", None))
        if spec is None:
            _cache._log(logger, "info", f"native {component}: skipped (no single native recipe)")
            return None
        recorded_base = str(base_repo or getattr(fam, "base_repo", "") or "")
        keyed = _key_for(fam, spec, source_base or base_repo, component, base_id = recorded_base)
        if keyed is None:
            _cache._log(
                logger,
                "info",
                f"native {component}: skipped (dense source shards not found locally)",
            )
            return None

        def run(check: Optional[Any] = None) -> Optional[str]:
            def both() -> None:
                if check is not None:
                    check()
                if cancel_check is not None and check is None:
                    cancel_check()

            return save(
                transformer,
                fam,
                spec,
                base_repo = base_repo,
                source_base = source_base,
                component = component,
                logger = logger,
                cancel_check = both,
            )

        if not _cache.async_enabled():
            return run()
        return NativePendingWrite(run, keyed[0], logger)
    except Exception as exc:  # noqa: BLE001
        _cache._log(
            logger, "warning", f"native {component}: save skipped ({type(exc).__name__}: {exc})"
        )
        return None


def read_for_load(
    fam: Any,
    spec: Optional[dict],
    transformer_cls: Any,
    *,
    base_repo: Optional[str],
    source_base: Optional[str],
    component: str = "transformer",
    config_kwargs: Optional[dict] = None,
    logger: Any = None,
    cancel_check: Optional[Any] = None,
) -> Any:
    """``lookup`` + ``load`` of the cached native ``component``, or None (a miss). Never raises."""
    if spec is None or transformer_cls is None:
        return None
    path = lookup(
        fam, spec, base_repo = base_repo, source_base = source_base, component = component, logger = logger
    )
    if path is None:
        return None
    return load(
        path,
        transformer_cls,
        str(source_base or base_repo),
        component = component,
        spec = spec,
        config_kwargs = config_kwargs,
        logger = logger,
        cancel_check = cancel_check,
    )
