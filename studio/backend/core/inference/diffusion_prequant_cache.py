# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local pre-quantized checkpoint cache for denoisers that have no hosted pre-quant.

A family with a hosted ``unsloth/<model>-FP8`` repo loads its quantized denoiser straight from an mmap'd
checkpoint: the dense bf16 transformer is never read. A family without one (FLUX.1-Kontext, Qwen-Image-Edit,
Wan 2.1, HunyuanVideo-1.5, ...) reads the dense weights and quantizes them on every load. This module keeps the
result of that runtime quantization on disk, in exactly the container and metadata the hosted artifacts use, so
the next load takes the hosted path (``resolve_prequant_source`` -> ``load_prequantized_transformer``) from a
local file instead.

KEY. sha256 over everything that changes the quantized bytes: every source shard's safetensors HEADER (name,
8-byte length and JSON: dtype, shape and byte range of every tensor) plus its content identity (the Hub cache blob
name, which IS the file's sha256, or size + mtime for a local dir; two uploads of one architecture share a header),
so the multi-GB payload is never read; the denoiser component, the scheme, ``POLICY_VERSION``,
the transformer class, the Linear filter (min_features, exclude tokens, require_bf16, divisibility), the ConvRot
spec, fp8 fast-accum, and the torch / torchao minor versions. A miss is normal and falls back to quantizing.

TRUST. The cache root is written only by Studio and holds safetensors (no pickle), so a file under the exact root
counts as allowlisted without ``UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH``; ``realpath`` first, so a link inside the root
that points out of it is refused.

ON DISK, per key dir: ``model.safetensors`` then ``manifest.json``. Both go through a temp file in the same dir
plus ``os.replace``; the manifest is written last and is the commit point, so an interrupted save leaves either no
entry or a complete one. A lookup needs the manifest, a matching file size and the recorded key.

HOST RAM. The writer streams: the header is built from the flattened tensors' dtype / shape / byte size, then each
tensor is copied device -> host and written on its own, so the save holds one tensor at a time on the host rather
than the whole state dict.

ENV:
  UNSLOTH_DIFFUSION_PREQUANT_CACHE         0 disables lookups and saves (default on)
  UNSLOTH_DIFFUSION_PREQUANT_CACHE_DIR     cache root (default <studio cache>/diffusion_prequant_cache)
  UNSLOTH_DIFFUSION_PREQUANT_CACHE_MAX_GB  LRU budget in GB (default 40; 0 disables eviction)
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
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

ENV_MODE = "UNSLOTH_DIFFUSION_PREQUANT_CACHE"
ENV_DIR = "UNSLOTH_DIFFUSION_PREQUANT_CACHE_DIR"
ENV_MAX_GB = "UNSLOTH_DIFFUSION_PREQUANT_CACHE_MAX_GB"
DEFAULT_MAX_GB = 40.0

# Bump when anything about how Studio quantizes at load changes in a way the key below does not already see.
POLICY_VERSION = 1

FILE_NAME = "model.safetensors"
MANIFEST_NAME = "manifest.json"
LAST_USED_NAME = "last_used"
_TEMP_SUFFIX = ".tmp"
_TOMBSTONE_SUFFIX = ".evicting"
# A key used this recently is never evicted: another process may be loading from it.
_EVICT_GRACE_SECONDS = 600.0
# Free space kept after a save: max(this, 5% of the artifact).
_DISK_MARGIN_BYTES = 2 << 30
_MAX_HEADER_BYTES = 100 << 20

_LEGACY_ROOT = Path.home() / ".cache" / "unsloth" / "diffusion_prequant_cache"

# torch dtype name -> safetensors dtype tag (the reverse of prequant_safetensors._ST_DTYPES).
_DTYPE_TAGS = {
    "bfloat16": "BF16",
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
    "complex64": "C64",
    "float8_e4m3fn": "F8_E4M3",
    "float8_e5m2": "F8_E5M2",
    "float8_e8m0fnu": "F8_E8M0",
    "int8": "I8",
    "uint8": "U8",
    "int16": "I16",
    "uint16": "U16",
    "int32": "I32",
    "uint32": "U32",
    "int64": "I64",
    "uint64": "U64",
    "bool": "BOOL",
}

_lock = threading.Lock()
_HEADER_MEMO: dict[tuple, str] = {}


# ------------------------------------------------------------------------------------------------- settings
def enabled() -> bool:
    raw = (os.environ.get(ENV_MODE) or "").strip().lower()
    return raw not in ("0", "off", "false", "no", "none")


def cache_root() -> Path:
    raw = (os.environ.get(ENV_DIR) or "").strip()
    if raw:
        return Path(os.path.expanduser(raw))
    try:
        from utils.paths.storage_roots import cache_root as studio_cache_root
    except ImportError:
        return _LEGACY_ROOT
    return studio_cache_root() / "diffusion_prequant_cache"


def max_cache_bytes() -> Optional[int]:
    raw = (os.environ.get(ENV_MAX_GB) or "").strip()
    try:
        gb = float(raw) if raw else DEFAULT_MAX_GB
    except ValueError:
        gb = DEFAULT_MAX_GB
    if gb != gb or gb in (float("inf"), float("-inf")):
        gb = DEFAULT_MAX_GB
    if not gb > 0:
        return None
    return int(gb * (1 << 30))


def path_in_cache(path: Any) -> bool:
    """True when ``path`` really resolves to an artifact inside the cache root (symlinks resolved on both sides)."""
    try:
        real = os.path.realpath(os.path.expanduser(str(path)))
        root = os.path.realpath(str(cache_root()))
    except Exception:  # noqa: BLE001
        return False
    if not real.startswith(root + os.sep):
        return False
    rel = os.path.relpath(real, root).split(os.sep)
    return len(rel) == 2 and rel[1] == FILE_NAME


# ------------------------------------------------------------------------------------------------- key
def _minor(version: Any) -> str:
    parts = str(version or "").split("+")[0].split(".")
    return ".".join(parts[:2])


def _toolchain() -> dict:
    out: dict[str, Any] = {}
    try:
        import torch
        out["torch"] = _minor(torch.__version__)
    except Exception:  # noqa: BLE001
        out["torch"] = None
    try:
        import importlib.metadata as md
        out["torchao"] = _minor(md.version("torchao"))
    except Exception:  # noqa: BLE001
        out["torchao"] = None
    return out


def read_header_bytes(path: Any) -> Optional[bytes]:
    """The 8-byte length plus the JSON header of a safetensors file, or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) != 8:
                return None
            (length,) = struct.unpack("<Q", head)
            if length <= 0 or length > _MAX_HEADER_BYTES:
                return None
            body = fh.read(length)
            if len(body) != length:
                return None
            return head + body
    except OSError:
        return None


def _content_identity(real: str, st: Any) -> str:
    """What says the PAYLOAD is unchanged, beside the header (two uploads of one architecture share a header).

    A Hub cache snapshot links to ``blobs/<sha256 of the file>``, a free content hash. Anything else (a local model
    dir) falls back to size + mtime, so an edited file re-quantizes rather than serving the old build."""
    name = os.path.basename(real)
    if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
        return "blob:" + name
    return f"stat:{st.st_size}:{st.st_mtime_ns}"


def source_fingerprint(shards: Iterable[Any]) -> Optional[str]:
    """sha256 over (basename, header, content identity) of every shard, order-independent. None when any shard is
    unreadable.

    Memoised on (realpath, mtime_ns, size): planners resolve sources several times per load."""
    digest = hashlib.sha256()
    entries = []
    for shard in shards:
        try:
            real = os.path.realpath(str(shard))
            st = os.stat(real)
        except OSError:
            return None
        memo = (real, st.st_mtime_ns, st.st_size)
        with _lock:
            cached = _HEADER_MEMO.get(memo)
        if cached is None:
            header = read_header_bytes(real)
            if header is None:
                return None
            cached = hashlib.sha256(
                header + b"\0" + _content_identity(real, st).encode("utf-8")
            ).hexdigest()
            with _lock:
                _HEADER_MEMO[memo] = cached
        entries.append((os.path.basename(str(shard)), cached))
    if not entries:
        return None
    for name, sha in sorted(entries):
        digest.update(name.encode("utf-8") + b"\0" + sha.encode("ascii") + b"\n")
    return digest.hexdigest()


def _hub_cache_dir() -> Optional[str]:
    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        return active_hf_hub_cache()
    except Exception:  # noqa: BLE001
        return None


def _hub_cache_roots() -> list:
    """The live Hub cache root, plus huggingface_hub's import-time root when it differs (``cache_dir = None``)."""
    roots = [_hub_cache_dir()]
    try:
        from huggingface_hub import constants
        other = constants.HF_HUB_CACHE
        if other and (roots[0] is None or os.path.realpath(other) != os.path.realpath(roots[0])):
            roots.append(other)
    except Exception:  # noqa: BLE001
        pass
    return roots


def _local_file(repo_or_dir: str, filename: str, cache_dir: Optional[str]) -> Optional[str]:
    """``filename`` of a local directory or a Hub repo already in the cache; never a network call."""
    if os.path.isdir(os.path.expanduser(repo_or_dir)):
        path = os.path.join(os.path.expanduser(repo_or_dir), filename)
        return path if os.path.isfile(path) else None
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(repo_or_dir, filename, cache_dir = cache_dir)
    except Exception:  # noqa: BLE001 - not a repo id, or no cache
        return None
    return hit if isinstance(hit, str) and os.path.isfile(hit) else None


def source_shards(
    base: Optional[str],
    component: str = "transformer",
    *,
    cache_dir: Optional[str] = None,
) -> Optional[list[str]]:
    """Local paths of the dense safetensors shards of ``base``'s ``component``, or None when not all are local."""
    if not base:
        return None
    if cache_dir is not None:
        return _source_shards_in(base, component, cache_dir)
    # The live root first, then huggingface_hub's import-time root when a cache-folder change moved the live one: the
    # loaders fetch with reuse_other_cache_root, so the dense shards may sit in either. A key needs only the shard
    # headers and blob names, which are the same bytes wherever they are stored.
    roots = [r for r in _hub_cache_roots() if r] or [None]
    for root in roots:
        found = _source_shards_in(base, component, root)
        if found:
            return found
    return None


def _source_shards_in(base: str, component: str, cache_dir: Optional[str]) -> Optional[list[str]]:
    candidates = [str(base)]
    try:
        from .diffusion_families import canonical_base, mirror_repo
        for alt in (mirror_repo(base), canonical_base(base)):
            if alt and alt not in candidates:
                candidates.append(alt)
    except Exception:  # noqa: BLE001
        pass
    stem = f"{component}/diffusion_pytorch_model"
    for repo in candidates:
        index = _local_file(repo, f"{stem}.safetensors.index.json", cache_dir)
        if index is not None:
            try:
                with open(index, "r", encoding = "utf-8") as fh:
                    weight_map = (json.load(fh) or {}).get("weight_map") or {}
                names = sorted(set(str(v) for v in weight_map.values()))
            except Exception:  # noqa: BLE001
                continue
            paths = [_local_file(repo, f"{component}/{n}", cache_dir) for n in names]
            if names and all(paths):
                return [str(p) for p in paths]
            continue
        single = _local_file(repo, f"{stem}.safetensors", cache_dir)
        if single is not None:
            return [single]
    return None


def key_parts(
    *,
    family: Optional[str],
    transformer_class: Optional[str],
    scheme: str,
    component: str,
    source_fp: str,
) -> dict:
    """Everything besides the source weights that changes the quantized bytes this cache would hold."""
    from .diffusion_transformer_quant import (
        DEFAULT_MIN_LINEAR_FEATURES,
        TQ_FP8,
        _REQUIRE_BF16_SCHEMES,
        _resolve_fast_accum,
        convrot_spec_for_scheme,
        divisible_for_scheme,
        exclude_tokens_for_scheme,
        int8_convrot_enabled,
    )

    group, suffixes = convrot_spec_for_scheme(scheme, family)
    return {
        "policy": POLICY_VERSION,
        "family": str(family or ""),
        "transformer_class": str(transformer_class or ""),
        "scheme": scheme,
        "component": component,
        "source": source_fp,
        "min_features": DEFAULT_MIN_LINEAR_FEATURES,
        "exclude_name_tokens": sorted(exclude_tokens_for_scheme(scheme, family)),
        "require_bf16": scheme in _REQUIRE_BF16_SCHEMES,
        "require_divisible": divisible_for_scheme(scheme),
        "convrot": [int(group or 0), sorted(suffixes or ())]
        if group and int8_convrot_enabled(family)
        else None,
        "fast_accum": _resolve_fast_accum(None) if scheme == TQ_FP8 else None,
        **_toolchain(),
    }


def cache_key(parts: dict) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys = True, default = str).encode("utf-8")
    ).hexdigest()[:32]


# Only the torchao schemes whose runtime and hosted builds are the same artifact. nvfp4 runs a per-family policy and
# a backend conversion at load; mxfp8 has no hosted twin to mirror.
CACHEABLE_SCHEMES = ("int8", "fp8")


def _key_for(
    fam: Any, scheme: str, base: Optional[str], component: str
) -> Optional[tuple[str, dict]]:
    if scheme not in CACHEABLE_SCHEMES:
        return None
    shards = source_shards(base, component)
    if not shards:
        return None
    fp = source_fingerprint(shards)
    if fp is None:
        return None
    parts = key_parts(
        family = getattr(fam, "name", fam if isinstance(fam, str) else None),
        transformer_class = getattr(fam, "transformer_class", None),
        scheme = scheme,
        component = component,
        source_fp = fp,
    )
    return cache_key(parts), parts


# ------------------------------------------------------------------------------------------------- lookup
def _entry(key: str) -> Optional[Path]:
    """The artifact of ``key`` when its manifest commits to it, else None."""
    kdir = cache_root() / key
    manifest = kdir / MANIFEST_NAME
    artifact = kdir / FILE_NAME
    try:
        data = json.loads(manifest.read_text(encoding = "utf-8"))
        if not isinstance(data, dict) or data.get("key") != key:
            return None
        if artifact.stat().st_size != int(data.get("bytes", -1)):
            return None
    except Exception:  # noqa: BLE001 - absent / torn / unreadable: a miss
        return None
    return artifact


def lookup(
    fam: Any,
    scheme: str,
    *,
    base_repo: Optional[str],
    component: str = "transformer",
    logger: Any = None,
) -> Optional[str]:
    """Path of a cached artifact for this exact source + scheme, or None. Local reads only, never raises."""
    if not enabled():
        return None
    try:
        keyed = _key_for(fam, scheme, base_repo or getattr(fam, "base_repo", None), component)
        if keyed is None:
            return None
        key, _parts = keyed
        # Check + touch under the lock eviction holds while it re-checks an entry: once this returns, the entry was
        # used within the eviction grace window, so no process removes it before the loader opens it.
        with _cache_lock():
            artifact = _entry(key)
            if artifact is None:
                return None
            _touch(artifact.parent)
        return str(artifact)
    except Exception as exc:  # noqa: BLE001
        _log(logger, "warning", f"skipped (lookup failed: {type(exc).__name__}: {exc})")
        return None


def note_hit(path: Any, logger: Any = None) -> None:
    """Called once the loader really used ``path``."""
    _log(logger, "info", f"hit ({Path(str(path)).parent.name})")


_LOCK_NAME = ".lock"


# Nesting depth of _cache_lock in this thread's process (introspection for tests only).
_LOCK_DEPTH = [0]


@contextmanager
def _cache_lock(root: Optional[Path] = None):
    """Cross-process lock over the cache root, held for short check-and-act sections (lookup's check + touch,
    eviction's re-check + rename). POSIX flock; where that is unavailable the sections still run, unlocked."""
    fh = None
    try:
        root = root if root is not None else cache_root()
        root.mkdir(parents = True, exist_ok = True)
        fh = open(root / _LOCK_NAME, "a+b")
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
    except OSError:
        fh = None
    _LOCK_DEPTH[0] += 1
    try:
        yield
    finally:
        _LOCK_DEPTH[0] -= 1
        if fh is not None:
            try:
                fh.close()  # closing the descriptor drops the flock
            except OSError:
                pass


def _touch(kdir: Path) -> None:
    try:
        p = kdir / LAST_USED_NAME
        p.touch(exist_ok = True)
        os.utime(p, None)
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------- writer
def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def write_streaming_safetensors(
    path: Any,
    tensors: dict,
    metadata: Optional[dict] = None,
) -> int:
    """Write ``tensors`` (any device) to ``path`` as safetensors, holding one tensor at a time on the host.

    Same byte layout ``safetensors.torch.save_file`` produces: entries ordered by dtype size (largest first) then
    name, so every tensor starts on its own alignment and the mmap reader never has to copy one. Returns the file
    size. ``path`` is written in place; the caller renames it."""
    import torch

    order = sorted(tensors.items(), key = lambda kv: (-kv[1].element_size(), kv[0]))
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    for name, tensor in order:
        tag = _DTYPE_TAGS.get(str(tensor.dtype).replace("torch.", ""))
        if tag is None:
            raise ValueError(f"{name}: dtype {tensor.dtype} has no safetensors tag")
        size = _tensor_bytes(tensor)
        header[name] = {
            "dtype": tag,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    blob = json.dumps(header, separators = (",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _name, tensor in order:
            if tensor.numel() == 0:
                continue
            flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            host = flat.cpu() if flat.device.type != "cpu" else flat
            fh.write(memoryview(host.numpy()))
            del host, flat
        fh.flush()
        os.fsync(fh.fileno())
    return 8 + len(blob) + offset


def _atomic_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir = str(path.parent), prefix = f".{path.name}.", suffix = _TEMP_SUFFIX)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _flatten(state_dict: dict) -> tuple[dict, dict]:
    """``(flat tensors, header)`` in the exact layout ``save_prequant_safetensors`` writes, without host copies."""
    from .prequant_safetensors import (
        UNSLOTH_ROOT_KEYS_KEY,
        UNSLOTH_ROOT_PREFIX,
        _root_level_keys,
        _torchao_helpers,
        unsupported_state_dict_keys,
    )

    helpers = _torchao_helpers()
    if helpers is None:
        raise RuntimeError("torchao safetensors helpers unavailable")
    if unsupported_state_dict_keys(state_dict):
        raise ValueError("root-level quantized tensors cannot be written as safetensors")
    roots = _root_level_keys(state_dict)
    quantizable = (
        {k: v for k, v in state_dict.items() if k not in set(roots)} if roots else state_dict
    )
    flatten, _ = helpers
    flat, torchao_metadata = flatten(quantizable)
    header = dict(torchao_metadata or {})
    flat = dict(flat)
    if roots:
        for key in roots:
            flat[f"{UNSLOTH_ROOT_PREFIX}{key}"] = state_dict[key]
        header[UNSLOTH_ROOT_KEYS_KEY] = json.dumps([str(k) for k in roots])
    return flat, header


def _rotation(transformer: Any, info: dict) -> dict:
    fqns = tuple(info.get("convrot_fqns") or ())
    group = info.get("convrot_group")
    if not fqns or not group:
        return {}
    from .diffusion_convrot import rotation_metadata

    return rotation_metadata(int(group), fqns)


def save_runtime_quant(
    transformer: Any,
    fam: Any,
    *,
    base_repo: Optional[str],
    source_base: Optional[str] = None,
    component: str = "transformer",
    logger: Any = None,
) -> Optional[str]:
    """Persist a transformer ``quantize_transformer`` just quantized, so the next load skips the dense read.

    ``base_repo`` is the upstream id recorded as ``base_model_id`` (what the loader checks); ``source_base`` is the
    repo / dir the dense weights were really read from (a mirror), default ``base_repo``. Returns the artifact path,
    or None (with a ``skipped`` log line naming why). Never raises."""
    if not enabled():
        return None
    t0 = time.perf_counter()
    try:
        info = dict(getattr(transformer, "_unsloth_runtime_quant_info", None) or {})
        scheme = info.get("scheme")
        if not info or scheme not in CACHEABLE_SCHEMES:
            _log(logger, "info", f"skipped (scheme {scheme or 'unknown'} is not cached)")
            return None
        if info.get("native"):
            _log(logger, "info", "skipped (native weight-only layers have no safetensors layout)")
            return None
        if info.get("lora"):
            _log(logger, "info", "skipped (baked LoRA adapters)")
            return None
        import torch

        if info.get("dtype") not in (None, "None", str(torch.bfloat16), "bfloat16"):
            _log(logger, "info", f"skipped (model dtype {info.get('dtype')} is not bfloat16)")
            return None
        keyed = _key_for(fam, scheme, source_base or base_repo, component)
        if keyed is None:
            _log(logger, "info", "skipped (dense source shards not found locally)")
            return None
        key, parts = keyed
        # The runtime filter must be the one the key describes, or the artifact would not match a fresh quantize.
        for name in ("min_features", "require_bf16", "require_divisible"):
            if name in info and info[name] != parts[name]:
                _log(logger, "info", f"skipped (runtime {name} {info[name]!r} != {parts[name]!r})")
                return None
        if (
            "exclude_name_tokens" in info
            and sorted(t for t in info["exclude_name_tokens"] if t != "lora_")
            != parts["exclude_name_tokens"]
        ):
            _log(logger, "info", "skipped (runtime exclusion list differs)")
            return None
        if parts["fast_accum"] is not None:
            # The cache bakes the auto accumulate mode, exactly like the hosted fp8 builds, and lookup cannot see a
            # forced one (planners treat a forced non-default value as ruling out every pre-quant: see
            # diffusion_prequant.hosted_fast_accum_conflict). A build quantized under a forced different mode would be
            # labelled as the auto one, so it is not cached.
            from .diffusion_transformer_quant import _resolve_fast_accum
            if _resolve_fast_accum(info.get("fast_accum")) != parts["fast_accum"]:
                _log(
                    logger,
                    "info",
                    "skipped (fp8 quantized with a forced non-default accumulate mode)",
                )
                return None
        if bool(info.get("convrot_fqns")) != bool(parts["convrot"]):
            _log(logger, "info", "skipped (ConvRot state differs from the key)")
            return None
        if _entry(key) is not None:
            _log(logger, "info", f"skipped (already cached as {key})")
            return None

        state_dict = transformer.state_dict()
        if any(".lora_" in k or k.startswith("lora_") for k in state_dict):
            _log(logger, "info", "skipped (baked LoRA adapters)")
            return None
        try:
            flat, header = _flatten(state_dict)
        except Exception as exc:  # noqa: BLE001 - legacy int8 subclass on torchao < 0.18, root subclass, ...
            _log(
                logger, "info", f"skipped (not flattenable: {type(exc).__name__}: {str(exc)[:160]})"
            )
            return None
        nbytes = sum(_tensor_bytes(t) for t in flat.values())
        root = cache_root()
        root.mkdir(parents = True, exist_ok = True)
        free = shutil.disk_usage(str(root)).free
        need = nbytes + max(_DISK_MARGIN_BYTES, nbytes // 20)
        if free < need:
            # Stale entries may be what fills the volume: rotate them out before deciding there is no room.
            evict(root = root, extra_free_bytes = need - free, keep = key, logger = logger)
            free = shutil.disk_usage(str(root)).free
        if free < need:
            _log(
                logger,
                "info",
                f"skipped (disk short: {free / 2**30:.1f} GiB free, need {need / 2**30:.1f} GiB)",
            )
            return None
        budget = max_cache_bytes()
        if budget is not None and nbytes > budget:
            _log(logger, "info", "skipped (artifact larger than the cache budget)")
            return None

        from .diffusion_prequant import prequant_format_for
        from .prequant_safetensors import UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY

        metadata = {
            "base_model_id": str(base_repo or getattr(fam, "base_repo", "") or ""),
            "family": parts["family"],
            "scheme": scheme,
            "min_features": parts["min_features"],
            "exclude_name_tokens": list(parts["exclude_name_tokens"]),
            "require_bf16": parts["require_bf16"],
            "require_divisible": parts["require_divisible"],
            "fast_accum": parts["fast_accum"],
            "torch_dtype": "bfloat16",
            "quant_backend": "torchao",
            "transformer_class": parts["transformer_class"] or type(transformer).__name__,
            "component": component,
            "torch_version": str(torch.__version__),
            "torchao_version": parts.get("torchao"),
            "local_prequant_cache": {"key": key, "policy": POLICY_VERSION},
        }
        if scheme == "fp8":
            from .diffusion_transformer_quant import FP8_GRANULARITY
            metadata["fp8_granularity"] = FP8_GRANULARITY
        metadata.update(_rotation(transformer, info))
        header[UNSLOTH_FORMAT_KEY] = prequant_format_for(metadata)
        header[UNSLOTH_METADATA_KEY] = json.dumps(metadata, default = str)

        kdir = root / key
        kdir.mkdir(parents = True, exist_ok = True)
        fd, tmp = tempfile.mkstemp(dir = str(kdir), prefix = f".{FILE_NAME}.", suffix = _TEMP_SUFFIX)
        os.close(fd)
        try:
            size = write_streaming_safetensors(tmp, flat, header)
            del flat, state_dict
            os.replace(tmp, kdir / FILE_NAME)
            tmp = None
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        manifest = {"key": key, "bytes": size, "created": time.time(), "parts": parts}
        _atomic_bytes(
            kdir / MANIFEST_NAME, json.dumps(manifest, indent = 1, default = str).encode("utf-8")
        )
        _touch(kdir)
        _log(
            logger,
            "info",
            f"saved ({key}, {size / 2**30:.2f} GiB in {time.perf_counter() - t0:.1f} s)",
        )
        evict(logger = logger, keep = key)
        return str(kdir / FILE_NAME)
    except Exception as exc:  # noqa: BLE001 - a cache write must never fail a load
        _log(logger, "warning", f"skipped (save failed: {type(exc).__name__}: {str(exc)[:200]})")
        return None


# ------------------------------------------------------------------------------------------------- eviction
def _dir_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path, followlinks = False):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def _last_used(path: Path) -> float:
    for name in (LAST_USED_NAME, MANIFEST_NAME):
        try:
            return (path / name).stat().st_mtime
        except OSError:
            continue
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _is_key_dir(path: Path) -> bool:
    return len(path.name) == 32 and all(c in "0123456789abcdef" for c in path.name)


def evict(
    *,
    root: Optional[Path] = None,
    max_bytes: Optional[int] = None,
    keep: Optional[str] = None,
    extra_free_bytes: int = 0,
    logger: Any = None,
) -> list[str]:
    """Delete least recently used keys until the root fits its budget, and ``extra_free_bytes`` below what it holds
    now (room a save needs on a full volume). A disabled budget (0) never evicts. Never raises."""
    removed: list[str] = []
    try:
        root = root if root is not None else cache_root()
        budget = max_bytes if max_bytes is not None else max_cache_bytes()
        if budget is None or not root.is_dir():
            return removed
        entries = []
        for child in root.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name.endswith(_TOMBSTONE_SUFFIX):
                shutil.rmtree(child, ignore_errors = True)
            elif _is_key_dir(child):
                entries.append((_last_used(child), child, _dir_bytes(child)))
        total = sum(e[2] for e in entries)
        if extra_free_bytes > 0:
            budget = min(budget, max(0, total - int(extra_free_bytes)))
        cutoff = time.time() - _EVICT_GRACE_SECONDS
        for used, path, size in sorted(entries, key = lambda e: e[0]):
            if total <= budget:
                break
            if path.name == keep or used > cutoff:
                continue
            tomb = path.with_name(
                f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}{_TOMBSTONE_SUFFIX}"
            )
            # The scan's timestamp may be stale: another process can have resolved this entry since. Re-check under
            # the lock lookup takes for its check + touch, so a resolved entry is never renamed away under a loader.
            with _cache_lock(root):
                if _last_used(path) > cutoff:
                    continue
                try:
                    os.rename(path, tomb)
                except OSError:
                    continue
            shutil.rmtree(tomb, ignore_errors = True)
            total -= size
            removed.append(path.name)
        if removed:
            _log(logger, "info", f"evicted {len(removed)} key(s); {total / 2**30:.2f} GiB in use")
    except Exception as exc:  # noqa: BLE001
        _log(logger, "warning", f"eviction failed: {exc}")
    return removed


def _log(logger: Any, level: str, msg: str) -> None:
    if logger is not None:
        getattr(logger, level)("diffusion.prequant_cache: %s", msg)
