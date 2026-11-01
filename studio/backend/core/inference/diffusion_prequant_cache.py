# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime-quantized denoiser cache, hosted artifact format; only Studio writes it (safetensors, no
pickle), so files under its exact realpath are allowlisted; manifest is commit point"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import shutil
import sys
import struct
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

from . import hub_reachability as _reach

ENV_MODE = "UNSLOTH_DIFFUSION_PREQUANT_CACHE"
ENV_DIR = "UNSLOTH_DIFFUSION_PREQUANT_CACHE_DIR"
ENV_MAX_GB = "UNSLOTH_DIFFUSION_PREQUANT_CACHE_MAX_GB"
ENV_ASYNC = "UNSLOTH_DIFFUSION_PREQUANT_CACHE_ASYNC"
DEFAULT_MAX_GB = 40.0

# Bump when anything about how Studio quantizes at load changes in a way the key below does not already see.
POLICY_VERSION = 1

FILE_NAME = "model.safetensors"
MANIFEST_NAME = "manifest.json"
LAST_USED_NAME = "last_used"
# sha256 of the artifact, recorded with the stat it was last checked against (save, or the first load after a change).
VERIFIED_NAME = "verified"
_HASH_CHUNK = 16 << 20
_TEMP_SUFFIX = ".tmp"
_TOMBSTONE_SUFFIX = ".evicting"
# Keys used within the grace window are spared (another process may be loading them) unless still over budget,
# then only the last _EVICT_ACTIVE_SECONDS (lookup to open), so fresh saves cannot grow past the budget.
_EVICT_GRACE_SECONDS = 600.0
_EVICT_ACTIVE_SECONDS = 60.0
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
# Realpaths of entries a load in this process resolved but could not use: the next save of that key replaces them.
_REJECTED: set[str] = set()


# ------------------------------------------------------------------------------------------------- settings
def enabled() -> bool:
    raw = (os.environ.get(ENV_MODE) or "").strip().lower()
    return raw not in ("0", "off", "false", "no", "none")


def async_enabled() -> bool:
    raw = (os.environ.get(ENV_ASYNC) or "").strip().lower()
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
    """Payload identity: the Hub blob name (its sha256), else stat incl. inode + ctime, which a size/mtime-preserving
    replacement still changes."""
    name = os.path.basename(real)
    if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
        return "blob:" + name
    return f"stat:{st.st_size}:{st.st_mtime_ns}:{st.st_ino}:{st.st_ctime_ns}"


def source_fingerprint(shards: Iterable[Any]) -> Optional[str]:
    """Order-independent sha256 over (basename, header, content identity) of every shard; None if any is unreadable.
    Memoised: planners resolve sources several times per load."""
    digest = hashlib.sha256()
    entries = []
    for shard in shards:
        try:
            real = os.path.realpath(str(shard))
            st = os.stat(real)
        except OSError:
            return None
        memo = (real, st.st_mtime_ns, st.st_size, st.st_ino, st.st_ctime_ns)
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


def _local_file(
    repo_or_dir: str,
    filename: str,
    cache_dir: Optional[str],
    revision: Optional[str] = None,
) -> Optional[str]:
    """``filename`` of a local directory or a Hub repo already in the cache; never a network call."""
    if os.path.isdir(os.path.expanduser(repo_or_dir)):
        path = os.path.join(os.path.expanduser(repo_or_dir), filename)
        return path if os.path.isfile(path) else None
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(repo_or_dir, filename, cache_dir = cache_dir, revision = revision)
    except Exception:  # noqa: BLE001 - not a repo id, or no cache
        return None
    return hit if isinstance(hit, str) and os.path.isfile(hit) else None


def _unreferenced_revisions(repo: str, cache_dir: Optional[str]) -> list:
    """Cached snapshot commits of ``repo``, newest first, only when there is no ``refs/main`` (a commit-pinned
    download writes no ref, so ``try_to_load_from_cache`` cannot see it). Never raises."""
    if os.path.isdir(os.path.expanduser(repo)):
        return []
    try:
        if cache_dir is None:
            from huggingface_hub import constants
            cache_dir = constants.HF_HUB_CACHE
        repo_dir = Path(cache_dir) / ("models--" + str(repo).replace("/", "--"))
        if (repo_dir / "refs" / "main").is_file():
            return []
        snaps = [p for p in (repo_dir / "snapshots").iterdir() if p.is_dir()]
        snaps.sort(key = lambda p: p.stat().st_mtime, reverse = True)
        return [p.name for p in snaps]
    except Exception:  # noqa: BLE001 - no cache for this repo
        return []


def _shard_sets(
    base: Optional[str],
    component: str,
    cache_dir: Optional[str] = None,
):
    """Every complete local shard set of ``base``'s ``component``, in preference order (see ``source_shards``)."""
    if not base:
        return
    if cache_dir is not None:
        yield from _shard_sets_in(base, component, cache_dir)
        return
    # Loaders fetch with reuse_other_cache_root, so the shards may sit in the live or the import-time root.
    roots = [r for r in _hub_cache_roots() if r] or [None]
    for root in roots:
        yield from _shard_sets_in(base, component, root)


def source_shards(
    base: Optional[str],
    component: str = "transformer",
    *,
    cache_dir: Optional[str] = None,
) -> Optional[list[str]]:
    """Local paths of the dense safetensors shards of ``base``'s ``component``, or None when not all are local."""
    for found in _shard_sets(base, component, cache_dir):
        return found
    return None


def _shard_sets_in(base: str, component: str, cache_dir: Optional[str]):
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
        # One revision at a time, so a set never mixes snapshots.
        for revision in [None, *_unreferenced_revisions(repo, cache_dir)]:
            index = _local_file(repo, f"{stem}.safetensors.index.json", cache_dir, revision)
            if index is not None:
                try:
                    with open(index, "r", encoding = "utf-8") as fh:
                        weight_map = (json.load(fh) or {}).get("weight_map") or {}
                    names = sorted(set(str(v) for v in weight_map.values()))
                except Exception:  # noqa: BLE001
                    continue
                paths = [_local_file(repo, f"{component}/{n}", cache_dir, revision) for n in names]
                if names and all(paths):
                    yield [str(p) for p in paths]
                continue
            single = _local_file(repo, f"{stem}.safetensors", cache_dir, revision)
            if single is not None:
                yield [single]


def base_identity(base: Optional[str]) -> str:
    """The base as ``_same_base_model`` compares it; keyed so byte-identical bases that fail that check never share."""
    raw = str(base or "").strip()
    if not raw:
        return ""
    try:
        from .diffusion_families import canonical_base
        raw = str(canonical_base(raw) or raw)
    except Exception:  # noqa: BLE001
        pass
    return raw.replace("\\", "/").rstrip("/").split("/")[-1].lower()


def key_parts(
    *,
    family: Optional[str],
    transformer_class: Optional[str],
    scheme: str,
    component: str,
    source_fp: str,
    base: Optional[str] = None,
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
        streamed_weight_only_layout,
    )

    group, suffixes = convrot_spec_for_scheme(scheme, family)
    # Only when set (streamed qwen-image int8), so every other build keeps its existing key.
    extra = streamed_weight_only_layout(scheme, family)
    return {
        "policy": POLICY_VERSION,
        "family": str(family or ""),
        "transformer_class": str(transformer_class or ""),
        "scheme": scheme,
        "component": component,
        "source": source_fp,
        "base": base_identity(base),
        "min_features": DEFAULT_MIN_LINEAR_FEATURES,
        "exclude_name_tokens": sorted(exclude_tokens_for_scheme(scheme, family)),
        "require_bf16": scheme in _REQUIRE_BF16_SCHEMES,
        "require_divisible": divisible_for_scheme(scheme),
        "convrot": [int(group or 0), sorted(suffixes or ())]
        if group and int8_convrot_enabled(family)
        else None,
        "fast_accum": _resolve_fast_accum(None) if scheme == TQ_FP8 else None,
        **extra,
        **_toolchain(),
    }


def cache_key(parts: dict) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys = True, default = str).encode("utf-8")
    ).hexdigest()[:32]


# Only schemes whose runtime and hosted builds are identical (nvfp4 converts per family at load; mxfp8 has no twin).
CACHEABLE_SCHEMES = ("int8", "fp8")


def _key_for(
    fam: Any,
    scheme: str,
    base: Optional[str],
    component: str,
    base_id: Optional[str] = None,
    shards: Optional[list] = None,
) -> Optional[tuple[str, dict]]:
    """``base`` is where the dense shards are found (or ``shards`` themselves); ``base_id`` the base model the artifact
    is recorded and validated against (default ``base``)."""
    if scheme not in CACHEABLE_SCHEMES:
        return None
    if shards is None:
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
        base = base_id if base_id is not None else base,
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
    source_base: Optional[str] = None,
) -> Optional[str]:
    """Path of a cached artifact for this exact source + scheme, or None. Local reads only, never raises."""
    if not enabled() or scheme not in CACHEABLE_SCHEMES:
        return None
    try:
        source = base_repo or getattr(fam, "base_repo", None)
        key = shards = None
        # Every shard set: a repo without refs/main may hold several commit-pinned snapshots.
        seen: set = set()
        # Serve only a build of the exact shards already loaded; other snapshots may differ under the same keys.
        candidates = (
            (source_shards(source_base, component),)
            if source_base
            else (source_shards(source, component), *_shard_sets(source, component))
        )
        for candidate in candidates:
            if not candidate or tuple(candidate) in seen:
                continue
            seen.add(tuple(candidate))
            keyed = _key_for(fam, scheme, source, component, shards = candidate)
            # Ask the Hub only when there is an entry to serve (a manifest), never on a plain miss.
            if keyed is not None and (cache_root() / keyed[0] / MANIFEST_NAME).is_file():
                key, shards = keyed[0], candidate
                break
        if key is None:
            return None
        if _hub_source_stale(shards, logger):
            return None
        # Check + touch under eviction's lock, so no process removes the entry before the loader opens it.
        with _cache_lock():
            artifact = _entry(key)
            if artifact is None:
                return None
            with _lock:
                rejected = os.path.realpath(str(artifact)) in _REJECTED
            if rejected:
                # Refused earlier in this process: a miss until replaced, so retries don't pick it again.
                return None
            _touch(artifact.parent)
        # The planner's header probe drops such an entry before the loader's rejection hook runs: record it
        # here, or the dense build would see it as already cached and never replace it.
        from .diffusion_prequant import local_prequant_scheme

        if local_prequant_scheme(str(artifact)) != scheme:
            note_rejected(artifact, logger)
            return None
        return str(artifact)
    except Exception as exc:  # noqa: BLE001
        _log(logger, "warning", f"skipped (lookup failed: {type(exc).__name__}: {exc})")
        return None


def _stat_stamp(path: Path) -> Optional[list]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns]


def _record_verified(kdir: Path, sha: str) -> None:
    stamp = _stat_stamp(kdir / FILE_NAME)
    if stamp is None:
        return
    try:
        _atomic_bytes(
            kdir / VERIFIED_NAME, json.dumps({"sha256": sha, "stat": stamp}).encode("utf-8")
        )
    except OSError:
        pass


def verify_entry(
    path: Any,
    logger: Any = None,
    cancel_check: Optional[Any] = None,
) -> bool:
    """Whether the artifact still hashes to its manifest sha256 (skipped when the stat matches the last check); no
    digest fails. Never raises."""
    try:
        artifact = Path(os.path.realpath(str(path)))
        kdir = artifact.parent
        manifest = json.loads((kdir / MANIFEST_NAME).read_text(encoding = "utf-8"))
        expected = manifest.get("sha256") if isinstance(manifest, dict) else None
        if not isinstance(expected, str) or len(expected) != 64:
            _log(logger, "warning", f"entry {kdir.name} records no payload digest")
            return False
        stamp = _stat_stamp(artifact)
        try:
            seen = json.loads((kdir / VERIFIED_NAME).read_text(encoding = "utf-8"))
            if seen.get("sha256") == expected and seen.get("stat") == stamp and stamp is not None:
                return True
        except Exception:  # noqa: BLE001 - no / torn marker: hash it
            pass
        digest = hashlib.sha256()
        with open(artifact, "rb") as fh:
            while True:
                if cancel_check is not None:
                    cancel_check()
                chunk = fh.read(_HASH_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
        if digest.hexdigest() != expected:
            _log(
                logger,
                "warning",
                f"entry {kdir.name} failed its payload digest (corrupt or modified)",
            )
            return False
        _record_verified(kdir, expected)
        return True
    except Exception as exc:  # noqa: BLE001
        _log(logger, "warning", f"entry check failed: {type(exc).__name__}: {exc}")
        return False


def note_rejected(path: Any, logger: Any = None) -> None:
    """Mark an unusable entry so the dense fallback's save replaces it. Never raises."""
    try:
        if not path_in_cache(path):
            return
        with _lock:
            _REJECTED.add(os.path.realpath(str(path)))
        _log(
            logger,
            "warning",
            f"entry {Path(str(path)).parent.name} failed to load; the next save replaces it",
        )
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------- hub revision
# A remote answer is reused this long: planners resolve the source several times per load.
_REMOTE_TTL_SECONDS = 300.0
_REMOTE_TIMEOUT_SECONDS = 5.0
_REMOTE_MEMO: dict[tuple, tuple] = {}
_GONE = "gone"
# Shared hub_reachability verdict: after one failure, lookups answer "unknown" (local snapshot stands) for the TTL.
_UNREACHABLE = _reach._UNREACHABLE
_endpoint = _reach.endpoint


def _hub_coords(path: Any) -> Optional[tuple[str, str, str, str]]:
    """``(repo_id, filename, blob, commit)`` of a Hub snapshot path, else None. ``blob`` is the plain filename in a
    symlink-less cache (Windows without Developer Mode)."""
    parts = Path(str(path)).parts
    try:
        i = len(parts) - 1 - parts[::-1].index("snapshots")
    except ValueError:
        return None
    if i < 1 or i + 2 >= len(parts) or not parts[i - 1].startswith("models--"):
        return None
    repo = parts[i - 1][len("models--") :].replace("--", "/")
    return (
        repo,
        "/".join(parts[i + 2 :]),
        _repo_blob_name(str(path), os.path.join(*parts[:i], "blobs")),
        parts[i + 1],
    )


def _repo_blob_name(path: str, blobs_dir: str) -> str:
    """Name of the ``<repo>/blobs`` entry a snapshot file links to, i.e. its etag. huggingface_hub >= 1.32 links that
    entry on to a shared store (``<cache>/blobs/<xx>/<xet hash>``), so only the first hop into ``blobs_dir`` counts."""
    blobs = os.path.normcase(os.path.realpath(blobs_dir))
    current = path
    for _ in range(8):
        if os.path.normcase(os.path.realpath(os.path.dirname(current))) == blobs:
            return os.path.basename(current)
        try:
            target = os.readlink(current)
        except (OSError, ValueError):
            break  # not a link: a copied file (no symlinks) or the chain's end
        current = os.path.join(os.path.dirname(current), target)
    return os.path.basename(os.path.realpath(path))


def _is_content_hash(name: str) -> bool:
    return len(name) in (40, 64) and all(c in "0123456789abcdef" for c in name)


def _remote_etag(
    repo: str,
    filename: str,
    revision: str = "main",
) -> Optional[str]:
    """The Hub etag of ``filename`` at ``revision``, ``_GONE`` if absent, None if unanswerable (treated as offline)."""
    memo = (repo, filename, revision)
    with _lock:
        hit = _REMOTE_MEMO.get(memo)
    if hit is not None and time.monotonic() - hit[0] < _REMOTE_TTL_SECONDS:
        return hit[1]
    endpoint = _endpoint()
    if _reach.hub_unreachable(endpoint):
        return None  # the Hub did not answer moments ago: unanswerable, without another timeout
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.utils import EntryNotFoundError
    except Exception:  # noqa: BLE001
        return None
    try:
        from .diffusion_prequant import _LOAD_HF_TOKEN
        token = _LOAD_HF_TOKEN.get()  # a gated / private base answers only to the load's own token
    except Exception:  # noqa: BLE001
        token = None
    try:
        etag = get_hf_file_metadata(
            hf_hub_url(repo, filename, revision = revision),
            token = token or None,
            timeout = _REMOTE_TIMEOUT_SECONDS,
        ).etag
        value = str(etag) if etag else None  # already normalized (no quotes, no W/ prefix)
    except EntryNotFoundError:
        value = _GONE
    except Exception as exc:  # noqa: BLE001 - offline, gated without a token, rate limited: unanswerable
        _reach.note_hub_failure(exc, endpoint)
        return None
    _reach.note_hub_success(endpoint)
    with _lock:
        _UNREACHABLE.pop(endpoint, None)
        _REMOTE_MEMO[memo] = (time.monotonic(), value)
    return value


def _hub_source_stale(shards: Optional[list], logger: Any = None) -> bool:
    """Whether Hub ``main`` moved past these shards' snapshot: a cache hit skips the ``from_pretrained`` request that
    would advance the ref. Offline loads keep the local snapshot."""
    if not shards:
        return False
    try:
        from .diffusion_prequant import _LOCAL_FILES_ONLY, hub_offline
        if _LOCAL_FILES_ONLY.get() or hub_offline():
            return False
    except Exception:  # noqa: BLE001
        return False
    for shard in shards:
        coords = _hub_coords(shard)
        if coords is None:
            continue
        repo, filename, blob, commit = coords
        etag = _remote_etag(repo, filename)
        if etag is not None and not _is_content_hash(blob):
            # No symlink content hash: ask what the snapshot's own commit serves.
            blob = _remote_etag(repo, filename, revision = commit)
            if blob is None or blob == _GONE:
                continue  # unanswerable: the local snapshot stands, as offline
        if etag is not None and etag != blob:
            _log(logger, "info", f"miss ({repo}:{filename} changed on the Hub since it was cached)")
            return True
    return False


def note_hit(path: Any, logger: Any = None) -> None:
    """Called once the loader really used ``path``."""
    _log(logger, "info", f"hit ({Path(str(path)).parent.name})")


_LOCK_NAME = ".lock"


# Nesting depth of _cache_lock (tests only).
_LOCK_DEPTH = [0]


# msvcrt.LK_LOCK retries for about 10 s, then raises; the sections it guards take milliseconds.
_WINDOWS_LOCK_ATTEMPTS = 6


def _lock_file(fh: Any) -> Optional[str]:
    """Exclusive flock, else msvcrt lock; returns the kind taken or None."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return "flock"
    try:
        import msvcrt
    except ImportError:
        return None
    for attempt in range(_WINDOWS_LOCK_ATTEMPTS):
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            return "msvcrt"
        except OSError:
            if attempt == _WINDOWS_LOCK_ATTEMPTS - 1:
                raise
    return None


def _unlock_file(fh: Any, kind: Optional[str]) -> None:
    if kind == "msvcrt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    # flock is dropped by closing the descriptor


@contextmanager
def _cache_lock(root: Optional[Path] = None):
    """Cross-process lock over the root for short check-and-act sections; runs unlocked if no lock exists."""
    fh = None
    kind = None
    try:
        root = root if root is not None else cache_root()
        root.mkdir(parents = True, exist_ok = True)
        fh = open(root / _LOCK_NAME, "a+b")
        try:
            kind = _lock_file(fh)
        except OSError:
            kind = None
    except OSError:
        fh = None
    _LOCK_DEPTH[0] += 1
    try:
        yield
    finally:
        _LOCK_DEPTH[0] -= 1
        if fh is not None:
            try:
                _unlock_file(fh, kind)
            except OSError:
                pass
            try:
                fh.close()
            except OSError:
                pass


def _writer_lock_path(root: Path, key: str) -> Path:
    # Beside the key dir, not in it: eviction renames the dir, and the lock must outlive that rename.
    return root / f".{key}.writing"


def _try_lock_file(fh: Any) -> Optional[bool]:
    """Non-blocking lock: True taken, False held elsewhere, None unsupported."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    try:
        import msvcrt
    except ImportError:
        return None
    try:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except OSError:
        return False


@contextmanager
def _writer_lock(root: Path, key: str):
    """Per-key cross-process build lease from the "already cached?" check through publication; yields False when no
    lock can be taken (caller skips rather than publishing unlocked)."""
    fh = None
    kind = None
    try:
        root.mkdir(parents = True, exist_ok = True)
        fh = open(_writer_lock_path(root, key), "a+b")
        kind = _lock_file(fh)
    except OSError:
        kind = None
    try:
        yield kind is not None
    finally:
        if fh is not None:
            try:
                _unlock_file(fh, kind)
            except OSError:
                pass
            try:
                fh.close()
            except OSError:
                pass


def _writer_active(root: Path, key: str) -> bool:
    """Whether a build of ``key`` is in progress in some process (its lease is held). Never raises."""
    path = _writer_lock_path(root, key)
    if not path.exists():
        return False
    try:
        with open(path, "a+b") as fh:
            taken = _try_lock_file(fh)
            if taken:
                _unlock_file(fh, "msvcrt" if sys.platform == "win32" else "flock")
            return taken is False
    except OSError:
        return False


def _reclaim(kdir: Path) -> None:
    """Remove a key dir: renamed to a tombstone under the cache lock, then deleted. Never raises."""
    tomb = kdir.with_name(f"{kdir.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}{_TOMBSTONE_SUFFIX}")
    try:
        with _cache_lock(kdir.parent):
            os.rename(kdir, tomb)
    except OSError:
        return
    shutil.rmtree(tomb, ignore_errors = True)


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
    cancel_check: Optional[Any] = None,
    hasher: Optional[Any] = None,
    release: bool = False,
) -> int:
    """Write ``tensors`` as safetensors one tensor at a time on the host; returns the file size.
    Ordered like ``save_file`` (dtype size desc, then name) so every tensor stays aligned for mmap. ``release`` pops
    each tensor from ``tensors`` once written, so a caller's last reference frees it during the write."""
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
    names = [name for name, _tensor in order]
    order = tensor = None
    blob = json.dumps(header, separators = (",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)

    def put(data: Any) -> None:
        fh.write(data)
        if hasher is not None:
            hasher.update(data)

    with open(path, "wb") as fh:
        put(struct.pack("<Q", len(blob)))
        put(blob)
        for name in names:
            if cancel_check is not None:
                cancel_check()
            tensor = tensors.pop(name) if release else tensors[name]
            if tensor.numel() == 0:
                continue
            if isinstance(tensor, _FileSlice):
                tensor.stream(put, cancel_check)
                del tensor
                continue
            if isinstance(tensor, _FlatRef):
                tensor = tensor.tensor()
            flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            host = flat.cpu() if flat.device.type != "cpu" else flat
            put(memoryview(host.numpy()))
            del host, flat, tensor
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


class _FlatRef:
    """A flat entry read from the live module at write time: torchao's flatten clones every quantized tensor on
    device, a second full VRAM copy when the card is fullest."""

    __slots__ = ("_get", "dtype", "shape", "_numel", "_esize")

    def __init__(self, get: Any, tensor: Any) -> None:
        self._get = get
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self._numel = int(tensor.numel())
        self._esize = int(tensor.element_size())

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._esize

    def tensor(self) -> Any:
        return self._get()


class _FileSlice:
    """A view of an mmapped shard held as (fd, offset, size), not a reference that would pin the whole mapping.
    Read back with ``pread`` from the same inode; a changed size / mtime fails the write."""

    __slots__ = ("fd", "offset", "nbytes", "dtype", "shape", "_numel", "_esize", "stamp")

    def __init__(self, fd: int, offset: int, tensor: Any, stamp: tuple) -> None:
        self.fd = fd
        self.offset = int(offset)
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self._numel = int(tensor.numel())
        self._esize = int(tensor.element_size())
        self.nbytes = self._numel * self._esize
        self.stamp = stamp

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._esize

    def stream(
        self,
        put: Any,
        cancel_check: Optional[Any] = None,
    ) -> None:
        st = os.fstat(self.fd)
        if (st.st_size, st.st_mtime_ns) != self.stamp:
            raise OSError("a mapped source shard changed during the cache write")
        pos, end = self.offset, self.offset + self.nbytes
        while pos < end:
            if cancel_check is not None:
                cancel_check()
            data = os.pread(self.fd, min(_HASH_CHUNK, end - pos), pos)
            if not data:
                raise OSError("short read from a mapped source shard")
            put(data)
            pos += len(data)


def _file_regions() -> list:
    """``[(start, end, file_offset, dev, inode, path)]`` sorted by start; empty without /proc/self/maps."""
    out = []
    try:
        with open("/proc/self/maps", "r", encoding = "utf-8", errors = "replace") as fh:
            for line in fh:
                parts = line.split(None, 5)
                if len(parts) < 6:
                    continue
                path = parts[5].strip()
                if not path.startswith("/") or path.endswith("(deleted)") or parts[4] == "0":
                    continue
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                major, minor = (int(x, 16) for x in parts[3].split(":"))
                out.append((lo, hi, int(parts[2], 16), (major, minor), int(parts[4]), path))
    except (OSError, ValueError):
        return []
    out.sort()
    return out


def _region_of(regions: list, ptr: int, nbytes: int) -> Optional[tuple]:
    import bisect

    i = bisect.bisect_right(regions, (ptr, float("inf"))) - 1
    if i < 0:
        return None
    region = regions[i]
    return region if region[0] <= ptr and ptr + nbytes <= region[1] else None


def _flatten(state_dict: dict) -> tuple[dict, dict]:
    """``(flat tensors, header)`` in ``save_prequant_safetensors``'s layout; peak extra VRAM is one param's clone."""
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
    roots = set(_root_level_keys(state_dict))
    flatten, _ = helpers
    header: dict = {}
    flat: dict = {}
    names: list = []
    for key, value in state_dict.items():
        if key in roots:
            continue
        part, meta = flatten({key: value})
        names.append(key)
        header[key] = meta[key]
        if "." in key:
            prefix, _, last = key.rpartition(".")
            stem = f"{prefix}._{last}_"
        else:
            stem = None
        for name, clone in part.items():
            if name == key:  # a plain tensor is passed through, not cloned
                flat[name] = value
                continue
            field = name[len(stem) :] if stem is not None and name.startswith(stem) else None
            inner = getattr(value, field, None) if field else None
            if (
                inner is not None
                and tuple(inner.shape) == tuple(clone.shape)
                and inner.dtype == clone.dtype
            ):
                flat[name] = _FlatRef(lambda v = value, f = field: getattr(v, f), clone)
            else:
                flat[name] = (
                    clone  # unrecognised naming: keep torchao's copy (correct, not bounded)
                )
        del part
    header["tensor_names"] = json.dumps(names)
    if roots:
        for key in _root_level_keys(state_dict):
            flat[f"{UNSLOTH_ROOT_PREFIX}{key}"] = state_dict[key]
        header[UNSLOTH_ROOT_KEYS_KEY] = json.dumps([str(k) for k in _root_level_keys(state_dict)])
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
    cancel_check: Optional[Any] = None,
    defer: bool = False,
    host_snapshot: bool = False,
) -> Any:
    """Persist a just-quantized transformer; returns the artifact path or None (logged). Never raises.
    ``base_repo`` is recorded as ``base_model_id``; ``source_base`` is where the dense weights were read (a mirror).
    ``defer`` runs every check now and returns a ``PendingWrite`` owning host tensors only; falls back to the
    synchronous write when the build is off-host, RAM is short, another write runs, or ASYNC=0.
    ``host_snapshot``: copy a device build to host now and write after the load; if it cannot defer, skip it."""
    if not enabled():
        return None
    t0 = time.perf_counter()
    cancelled = [False]
    user_cancel_check = cancel_check

    def cancel_check() -> None:
        if user_cancel_check is None:
            return
        try:
            user_cancel_check()
        except BaseException:
            cancelled[0] = True
            raise

    stack = contextlib.ExitStack()
    handed_off = False
    flat = header = state_dict = job = None
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
        recorded_base = str(base_repo or getattr(fam, "base_repo", "") or "")
        keyed = _key_for(fam, scheme, source_base or base_repo, component, base_id = recorded_base)
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
            # Entries bake the auto accumulate like hosted fp8; a forced-mode build would be mislabelled as auto.
            from .diffusion_transformer_quant import _resolve_fast_accum
            if _resolve_fast_accum(info.get("fast_accum")) != parts["fast_accum"]:
                _log(
                    logger,
                    "info",
                    "skipped (fp8 quantized with a forced non-default accumulate mode)",
                )
                return None
        wo_keys = ("weight_only_name_tokens", "weight_only_mode", "weight_only_outliers")
        wanted_wo = {k: parts[k] for k in wo_keys if k in parts}
        if {k: info[k] for k in wo_keys if k in info} != wanted_wo:
            _log(logger, "info", "skipped (weight-only modulation state differs from the key)")
            return None
        if wanted_wo and wanted_wo["weight_only_mode"] != "wo":
            _log(logger, "info", "skipped (diagnostic W8A8 modulation mode)")
            return None
        if bool(info.get("convrot_fqns")) != bool(parts["convrot"]):
            _log(logger, "info", "skipped (ConvRot state differs from the key)")
            return None
        if not stack.enter_context(_writer_lock(cache_root(), key)):
            _log(logger, "info", "skipped (no cross-process lock available for the build)")
            return None
        existing = _entry(key)
        replacing = None
        if existing is not None:
            real = os.path.realpath(str(existing))
            with _lock:
                replacing = real if real in _REJECTED else None
            if replacing is None:
                _log(logger, "info", f"skipped (already cached as {key})")
                return None
            _log(logger, "info", f"replacing {key} (it failed to load)")

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
        # Before eviction: an over-budget artifact must not cost existing entries.
        budget = max_cache_bytes()
        if budget is not None and nbytes > budget:
            _log(logger, "info", "skipped (artifact larger than the cache budget)")
            return None
        root = cache_root()
        root.mkdir(parents = True, exist_ok = True)
        if replacing is not None:
            # Reclaim the rejected artifact first: eviction keeps ``key``, so it would hold a second copy's disk.
            _reclaim(root / key)
            with _lock:
                _REJECTED.discard(replacing)
        free = shutil.disk_usage(str(root)).free
        need = nbytes + max(_DISK_MARGIN_BYTES, nbytes // 20)
        if free < need:
            evict(root = root, extra_free_bytes = need - free, keep = key, logger = logger)
            free = shutil.disk_usage(str(root)).free
        if free < need:
            _log(
                logger,
                "info",
                f"skipped (disk short: {free / 2**30:.1f} GiB free, need {need / 2**30:.1f} GiB)",
            )
            return None

        from .diffusion_prequant import prequant_format_for
        from .prequant_safetensors import UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY

        metadata = {
            "base_model_id": recorded_base,
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
        if wanted_wo:
            metadata["weight_only_name_tokens"] = list(wanted_wo["weight_only_name_tokens"])
            metadata["weight_only_outliers"] = int(wanted_wo["weight_only_outliers"])
        metadata.update(_rotation(transformer, info))
        header[UNSLOTH_FORMAT_KEY] = prequant_format_for(metadata)
        header[UNSLOTH_METADATA_KEY] = json.dumps(metadata, default = str)

        job = _WriteJob(key, root / key, flat, header, parts, nbytes, t0, logger)
        flat = header = None
        if defer:
            why = _defer_blocker(job, host_snapshot = host_snapshot, cancel_check = cancel_check)
            if why is None:
                state_dict = None
                pending = PendingWrite(job, stack.pop_all())
                handed_off = True
                _log(
                    logger,
                    "info",
                    f"writing {key} ({nbytes / 2**30:.2f} GiB) on a background thread once the load is published "
                    f"(holds {job.held_bytes / 2**30:.2f} GiB, {job.file_bytes / 2**30:.2f} GiB re-read from the "
                    f"source shards, pins {job.pinned_map_bytes / 2**30:.2f} GiB of mappings)",
                )
                return pending
            if host_snapshot and async_enabled():
                _log(
                    logger,
                    "info",
                    f"skipped ({why}; a resident build is never written inside the load)",
                )
                return None
            _log(logger, "info", f"writing {key} before the load returns ({why})")
        state_dict = None
        return job.run(cancel_check)
    except Exception as exc:  # noqa: BLE001 - a cache write must never fail a load
        if cancelled[0]:
            _log(logger, "info", "skipped (load cancelled during the save)")
            return None
        _log(logger, "warning", f"skipped (save failed: {type(exc).__name__}: {str(exc)[:200]})")
        return None
    finally:
        stack.close()
        flat = header = state_dict = job = None
        if not handed_off:
            _release_device_cache()


class _WriteJob:
    """One prepared artifact; ``run`` raises on any failure with no entry published."""

    def __init__(
        self,
        key: str,
        kdir: Path,
        flat: dict,
        header: dict,
        parts: dict,
        nbytes: int,
        t0: float,
        logger: Any,
    ) -> None:
        self.key = key
        self.kdir = kdir
        self.flat = flat
        self.header = header
        self.parts = parts
        self.nbytes = nbytes
        self.t0 = t0
        self.logger = logger
        self.release = False
        self.fds: dict = {}
        self.held_bytes = nbytes
        self.pinned_map_bytes = 0
        self.file_bytes = 0

    def tensors(self) -> Iterable[Any]:
        return self.flat.values() if self.flat is not None else ()

    def resolve(self) -> None:
        """Materialize ``_FlatRef``s and turn mmap views into ``_FileSlice``s so the snapshot pins no mapping."""
        regions = _file_regions()
        out: dict = {}
        held = file_bytes = 0
        pinned: dict = {}
        for name, t in self.flat.items():
            if isinstance(t, _FlatRef):
                t = t.tensor()
            size = _tensor_bytes(t)
            region = None
            if regions and size and getattr(t.device, "type", None) == "cpu":
                region = _region_of(regions, int(t.data_ptr()), size)
            if region is not None and t.is_contiguous():
                piece = self._slice(t, region)
                if piece is not None:
                    out[name] = piece
                    file_bytes += size
                    continue
            if region is not None:
                pinned[region[0]] = region[1] - region[0]
            out[name] = t
            held += size
        self.flat = out
        self.held_bytes = held
        self.pinned_map_bytes = sum(pinned.values())
        self.file_bytes = file_bytes
        self.release = True

    def _slice(self, t: Any, region: tuple) -> Optional[_FileSlice]:
        lo, _hi, file_offset, dev, inode, path = region
        entry = self.fds.get(path)
        if entry is None:
            try:
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            except OSError:
                return None
            st = os.fstat(fd)
            if (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino) != (*dev, inode):
                os.close(fd)
                return None
            entry = self.fds[path] = (fd, (st.st_size, st.st_mtime_ns))
        fd, stamp = entry
        return _FileSlice(fd, file_offset + int(t.data_ptr()) - lo, t, stamp)

    def snapshot_to_host(self, cancel_check: Optional[Any] = None) -> None:
        """Copy to host so the writer never touches the device (a render may be capturing a CUDA graph)."""
        out: dict = {}
        for name, t in self.flat.items():
            if cancel_check is not None:
                cancel_check()
            if isinstance(t, _FlatRef):
                t = t.tensor()
            if getattr(getattr(t, "device", None), "type", "cpu") != "cpu":
                t = t.detach().to("cpu")
            out[name] = t
        self.flat = out

    def close(self) -> None:
        fds, self.fds = self.fds, {}
        for fd, _stamp in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass

    def run(self, cancel_check: Optional[Any] = None) -> str:
        kdir = self.kdir
        kdir.mkdir(parents = True, exist_ok = True)
        fd, tmp = tempfile.mkstemp(dir = str(kdir), prefix = f".{FILE_NAME}.", suffix = _TEMP_SUFFIX)
        os.close(fd)
        try:
            # Hashed during the write (no second pass); the loader verifies it.
            hasher = hashlib.sha256()
            extra = {"release": True} if self.release else {}
            size = write_streaming_safetensors(
                tmp, self.flat, self.header, cancel_check = cancel_check, hasher = hasher, **extra
            )
            self.flat = None
            os.replace(tmp, kdir / FILE_NAME)
            tmp = None
        finally:
            self.close()
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        sha = hasher.hexdigest()
        _record_verified(kdir, sha)
        manifest = {
            "key": self.key,
            "bytes": size,
            "sha256": sha,
            "created": time.time(),
            "parts": self.parts,
        }
        _atomic_bytes(
            kdir / MANIFEST_NAME, json.dumps(manifest, indent = 1, default = str).encode("utf-8")
        )
        _touch(kdir)
        _log(
            self.logger,
            "info",
            f"saved ({self.key}, {size / 2**30:.2f} GiB in {time.perf_counter() - self.t0:.1f} s)",
        )
        evict(logger = self.logger, keep = self.key)
        return str(kdir / FILE_NAME)


# Host RAM a deferred write must leave free beyond the build it keeps alive (the rest of the load pins encoders).
_ASYNC_RESERVE_BYTES = 4 << 30
_ASYNC_RESERVE_FRACTION = 0.10
_BACKGROUND: set = set()


def _cgroup_memory_bytes() -> tuple[Optional[int], Optional[int]]:
    """Tightest cgroup v2 ``(limit, available)``; available adds back reclaimable file pages."""
    try:
        with open("/proc/self/cgroup", "r", encoding = "utf-8") as fh:
            rel = next((ln.split("::", 1)[1].strip() for ln in fh if ln.startswith("0::")), None)
    except OSError:
        return None, None
    if rel is None:
        return None, None
    best: tuple[Optional[int], Optional[int]] = (None, None)
    node = Path("/sys/fs/cgroup") / rel.lstrip("/")
    root = Path("/sys/fs/cgroup")
    while True:
        try:
            raw = (node / "memory.max").read_text(encoding = "utf-8").strip()
            if raw != "max":
                limit = int(raw)
                used = int((node / "memory.current").read_text(encoding = "utf-8").strip())
                stat = dict(
                    ln.split()
                    for ln in (node / "memory.stat").read_text(encoding = "utf-8").splitlines()
                    if ln.strip()
                )
                avail = max(0, limit - used + int(stat.get("file", 0)))
                if best[1] is None or avail < best[1]:
                    best = (limit, avail)
        except (OSError, ValueError):
            pass
        if node == root or node.parent == node:
            break
        node = node.parent
    return best


def _host_memory_bytes() -> tuple[Optional[int], Optional[int]]:
    try:
        from .diffusion_memory import _system_memory_mib
        total, avail = _system_memory_mib()
    except Exception:  # noqa: BLE001
        return None, None
    if total is None or avail is None:
        return None, None
    total, avail = int(total) << 20, int(avail) << 20
    limit, cg_avail = _cgroup_memory_bytes()
    if limit is not None:
        total, avail = min(total, limit), min(avail, cg_avail)
    return total, avail


def _defer_blocker(
    job: _WriteJob,
    host_snapshot: bool = False,
    cancel_check: Optional[Any] = None,
) -> Optional[str]:
    """Why ``job`` must be written now, or None: it must be wholly on the host and what it keeps alive must fit."""
    if not async_enabled():
        return f"{ENV_ASYNC}=0"
    with _lock:
        if any(not p.done() for p in _BACKGROUND):
            return "another background cache write is running"
    off_host = 0
    for t in job.tensors():
        if isinstance(t, _FlatRef):
            t = t.tensor()
        device = getattr(t, "device", None)
        if getattr(device, "type", None) != "cpu":
            if not host_snapshot:
                return f"the build is on {device}, not the host"
            off_host += _tensor_bytes(t)
    total, avail = _host_memory_bytes()
    if total is None:
        return "host memory unknown"
    if off_host:
        reserve = max(_ASYNC_RESERVE_BYTES, int(total * _ASYNC_RESERVE_FRACTION))
        if avail < off_host + reserve:
            return (
                f"host RAM short for a host copy of the build: {avail / 2**30:.1f} GiB available, the copy is "
                f"{off_host / 2**30:.1f} GiB and needs {reserve / 2**30:.1f} GiB spare"
            )
        job.snapshot_to_host(cancel_check)
        job.resolve()
        return None
    if not job.release:
        job.resolve()
    keeps = job.held_bytes + job.pinned_map_bytes
    reserve = max(_ASYNC_RESERVE_BYTES, int(total * _ASYNC_RESERVE_FRACTION))
    if avail < keeps + reserve:
        return (
            f"host RAM short for a background copy: {avail / 2**30:.1f} GiB available, the snapshot keeps "
            f"{keeps / 2**30:.1f} GiB alive and needs {reserve / 2**30:.1f} GiB spare"
        )
    return None


class _WriteCancelled(Exception):
    pass


class PendingWrite:
    """Threaded prepared write holding only the lease and host tensors; the manifest is the commit point."""

    def __init__(self, job: _WriteJob, stack: contextlib.ExitStack) -> None:
        self._job: Optional[_WriteJob] = job
        self._stack: Optional[contextlib.ExitStack] = stack
        self.key = job.key
        self.nbytes = job.nbytes
        self._logger = job.logger
        self._cancel = threading.Event()
        self._finished = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.result: Optional[str] = None
        with _lock:
            _BACKGROUND.add(self)

    def _check(self) -> None:
        if self._cancel.is_set():
            raise _WriteCancelled()

    def start(self) -> bool:
        with _lock:
            if self._thread is not None or self._job is None or self._cancel.is_set():
                return False
            self._thread = threading.Thread(
                target = self._run, name = f"prequant-cache-write-{self.key[:8]}", daemon = True
            )
        try:
            self._thread.start()
        except Exception as exc:  # noqa: BLE001 - no writer thread: drop the snapshot and lease, the next load re-quantises
            with _lock:
                self._thread = None
            _log(
                self._logger,
                "warning",
                f"skipped (background writer did not start: {type(exc).__name__})",
            )
            self._close()
            return False
        return True

    def _run(self) -> None:
        job = self._job
        try:
            if job is not None:
                job.t0 = time.perf_counter()
                self.result = job.run(self._check)
        except _WriteCancelled:
            _log(
                self._logger,
                "info",
                f"skipped ({self.key} cancelled before its background write finished)",
            )
        except Exception as exc:  # noqa: BLE001 - a background cache write never surfaces
            _log(
                self._logger,
                "warning",
                f"skipped (background save failed: {type(exc).__name__}: {str(exc)[:200]})",
            )
        finally:
            self._close()

    def _close(self) -> None:
        job, self._job = self._job, None
        if job is not None:
            job.close()
        job = None
        stack, self._stack = self._stack, None
        if stack is not None:
            try:
                stack.close()
            except Exception:  # noqa: BLE001
                pass
        with _lock:
            _BACKGROUND.discard(self)
        self._finished.set()

    def discard(self) -> None:
        with _lock:
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


def cancel_background_writes(timeout: float = 30.0) -> bool:
    with _lock:
        pending = list(_BACKGROUND)
    for p in pending:
        p.cancel()
    deadline = time.monotonic() + max(0.0, timeout)
    return all(p.wait(max(0.0, deadline - time.monotonic())) for p in pending)


def wait_background_writes(timeout: Optional[float] = None) -> bool:
    with _lock:
        pending = [p for p in _BACKGROUND if p._thread is not None]
    deadline = None if timeout is None else time.monotonic() + timeout
    for p in pending:
        left = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not p.wait(left):
            return False
    return True


FINISH_TIMEOUT_SECONDS = 600.0


def finish_background_writes(timeout: float = FINISH_TIMEOUT_SECONDS, logger: Any = None) -> bool:
    """Bounded wait, then cancel the rest, so the next load never shares host RAM with a held build."""
    with _lock:
        running = [p for p in _BACKGROUND if p._thread is not None and not p.done()]
    if running:
        _log(
            logger,
            "info",
            f"waiting for the background write of {', '.join(p.key for p in running)} before releasing the model",
        )
    finished = wait_background_writes(timeout)
    cancel_background_writes(timeout = 30.0)
    return finished


def _cancel_at_exit() -> None:
    finish_background_writes()


atexit.register(_cancel_at_exit)


def _release_device_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


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
    """LRU-evict until under budget and ``extra_free_bytes`` freed; a disabled budget never evicts. Never raises."""
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
        now = time.time()
        passes = [now - _EVICT_GRACE_SECONDS, now - _EVICT_ACTIVE_SECONDS]
        for cutoff, (used, path, size) in (
            (c, e) for c in passes for e in sorted(entries, key = lambda e: e[0])
        ):
            if total <= budget:
                break
            if path.name == keep or path.name in removed or used > cutoff:
                continue
            if _writer_active(root, path.name):
                continue  # a build is streaming into it: not idle, whatever its timestamps say
            tomb = path.with_name(
                f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}{_TOMBSTONE_SUFFIX}"
            )
            # Re-check under lookup's lock: another process may have resolved this entry since the scan.
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
