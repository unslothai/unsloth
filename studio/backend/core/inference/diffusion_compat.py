# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Metadata-only compatibility preflight for a diffusion pick.

A FLUX.2 GGUF only carries the transformer; its size (``inner_dim``) has to agree with the companion
diffusers base repo the loader assembles around it. ``assert_flux2_gguf_matches_base`` already
catches a mismatch, but it opens the downloaded checkpoint, so it fires from inside
``load_pipeline`` -- after the prefetch pulled ~19 GB of base shards and after the resident pipeline
was torn down to make room.

This module answers the same question from metadata alone: one HTTP range request for the first few
hundred KiB of the GGUF, where its tensor table lives. That is cheap enough to run at SELECTION time
(``/images/download-plan``) and again on the pre-eviction path, so the refusal lands before a byte
moves and before anything is unloaded.

Fail-open throughout, deliberately: an unreadable or truncated header, a base repo outside the size
table, an offline host, a server that ignores Range all yield "no opinion", and the load proceeds
with the loader's own guard as the backstop. A false positive here would refuse a pick that works,
which is strictly worse than the download this saves. (A known ungated MIRROR of a base is not an
exception: it is byte-identical to what it copies, ``canonical_base`` maps it back, and it is
checked like its upstream.)
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from hub.utils.hf_tokens import (
    ANONYMOUS_CACHE_IDENTITY,
    qualify_cache_identity,
    HfTokenArg,
    is_anonymous,
    normalize_token,
)
from core.inference.diffusion_families import (
    flux2_base_inner_dim,
    flux2_mismatch_reason,
    gguf_flux2_inner_dim,
    gguf_flux2_inner_dim_from_header,
    resolve_local_gguf_child,
)

# FLUX.2 tensor table sits in the first ~15 KiB; this is only a buffer ceiling.
_GGUF_HEADER_BYTES = 256 * 1024
# One short read. A pick is blocked on this in the UI, and a slow Hub must not stall the picker;
# a timeout is just another fail-open.
_HEADER_TIMEOUT_SECONDS = 15
# How long to wait for an interrupted read to notice before leaving it to the GC.
_ABANDON_GRACE_SECONDS = 0.5

# Memoises misses too; token fingerprint and file identity keep a stale None from sticking.
_INNER_DIM_CACHE: dict[tuple[str, str, str, Optional[tuple]], Optional[int]] = {}
_INNER_DIM_CACHE_MAX = 256
_CACHE_LOCK = threading.Lock()


def _token_fingerprint(token: HfTokenArg) -> str:
    """Non-reversible tag for a token, never the token itself; anonymous and ambient tokens differ."""
    if is_anonymous(token):
        return ANONYMOUS_CACHE_IDENTITY
    if not token:
        return ""
    return qualify_cache_identity(
        token, hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:16]
    )


def _file_identity(path: Optional[str]) -> Optional[tuple]:
    """Path, size and mtime_ns, so a replaced file is a new checkpoint; an unreadable stat never matches."""
    if path is None:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return (path, object())
    return (path, stat.st_size, stat.st_mtime_ns)


def _local_gguf_path(repo_id: str, gguf_filename: str) -> Optional[str]:
    """On-disk GGUF path for a local or cached pick; a file-valued repo_id is used as-is."""
    try:
        local_root = Path(repo_id).expanduser()
        if local_root.is_file():
            return str(local_root)
        if local_root.exists():
            return str(resolve_local_gguf_child(local_root, gguf_filename))
    # OSError/RuntimeError: invalid path characters, or an unresolvable '~' -> a remote id.
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        from utils.hf_cache_settings import active_hf_hub_cache

        # Not via diffusion.hub_cache_dir: that module imports this one.
        for root in (active_hf_hub_cache(), None):
            hit = try_to_load_from_cache(repo_id, gguf_filename, cache_dir = root)
            if isinstance(hit, str) and Path(hit).is_file():
                return hit
    except Exception:  # noqa: BLE001 — a cache we cannot read is not a verdict
        pass
    return None


def _snapshot_revision(path: Optional[str]) -> Optional[str]:
    """Commit read from a cached file's snapshots/sha parent; None outside the HF cache."""
    if not path:
        return None
    parts = Path(path).parts
    try:
        idx = len(parts) - 1 - parts[::-1].index("snapshots")
    except ValueError:
        return None
    return parts[idx + 1] if idx + 1 < len(parts) - 1 else None


def _hub_revision(repo_id: str, gguf_filename: str, hf_token: Optional[str]) -> Optional[str]:
    """The commit the Hub currently serves this file at, or None when it cannot be asked.

    One HEAD, no body: the caller only needs to know whether the local copy is still the current
    one, and an offline or erroring host must leave today's verdict alone."""
    from core.inference import hub_reachability

    if hub_reachability.hub_unreachable():
        return None  # the Hub failed to connect moments ago: unanswerable, without another timeout
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        meta = get_hf_file_metadata(
            hf_hub_url(repo_id, gguf_filename),
            token = hf_token,
            timeout = _HEADER_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 — a revision we cannot read is not a verdict
        hub_reachability.note_hub_failure(exc)
        return None
    hub_reachability.note_hub_success()
    return getattr(meta, "commit_hash", None) or None


def _read_local_header(path: str) -> bytes:
    """The first ``_GGUF_HEADER_BYTES`` of a file on disk, or b"" when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            return handle.read(_GGUF_HEADER_BYTES)
    # ValueError: open() rejects an embedded NUL rather than raising OSError.
    except (OSError, ValueError):
        return b""


def _ranged_stream(session: Any, url: str, headers: dict) -> Any:
    """Ranged GET on either httpx or requests; branch on callable stream, since the two APIs differ."""
    if callable(getattr(session, "stream", None)):
        # The Hub answers resolve URLs with a 302 to the CDN; httpx does not follow by default.
        return session.stream(
            "GET",
            url,
            headers = headers,
            timeout = _HEADER_TIMEOUT_SECONDS,
            follow_redirects = True,
        )
    return session.get(
        url,
        headers = headers,
        timeout = _HEADER_TIMEOUT_SECONDS,
        stream = True,
    )


def _iter_body(response: Any, chunk_size: int):
    """The response body in chunks, from httpx's reader or requests'."""
    reader = getattr(response, "iter_bytes", None) or response.iter_content
    return reader(chunk_size)


def _interrupt_read(response: Any) -> None:
    """Half-closes the socket to wake a blocked read; best effort, as the caller abandons its worker."""
    if response is None:
        return
    try:
        response.raw.shutdown()
    except Exception:  # noqa: BLE001 — a deadline that cannot fire must not become a new failure
        try:
            response.close()
        except Exception:  # noqa: BLE001
            pass


def _read_gguf_header(
    repo_id: str,
    gguf_filename: str,
    hf_token: Optional[str],
    *,
    revision: Optional[str] = None,
    max_bytes: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> bytes:
    """Wall-clock bound over the whole read, since a per-byte timeout would let a trickle outlast it."""
    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session
    except Exception:  # noqa: BLE001 — an unexpected hub layout leaves today's behaviour
        return b""
    max_bytes = _GGUF_HEADER_BYTES if max_bytes is None else max_bytes
    timeout_seconds = _HEADER_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    buffer = bytearray()
    # Published by the worker as soon as it has something interruptible; read by this thread on
    # timeout. A one-element list rather than a nonlocal, so the worker's assignment is visible.
    holder: list[Any] = [None]

    def _fetch() -> None:
        try:
            headers = dict(build_hf_headers(token = hf_token))
            headers["Range"] = f"bytes=0-{max_bytes - 1}"
            with _ranged_stream(
                get_session(), hf_hub_url(repo_id, gguf_filename, revision = revision), headers
            ) as response:
                holder[0] = response
                # A server ignoring Range answers 200 with the whole multi-GB checkpoint.
                if response.status_code != 206:
                    return
                deadline = time.monotonic() + timeout_seconds
                for chunk in _iter_body(response, 65536):
                    # extend, not +=: augmented assignment would rebind a local.
                    buffer.extend(chunk)
                    if len(buffer) >= max_bytes or time.monotonic() > deadline:
                        break
        # Keep partial bytes: the parser is truncation-safe.
        except Exception:  # noqa: BLE001 — offline, deadline fired, or the peer went away
            pass

    # iter_content blocks until a full chunk arrives, so the worker cannot see its deadline.
    watchdog = threading.Timer(timeout_seconds, lambda: _interrupt_read(holder[0]))
    watchdog.daemon = True
    watchdog.start()
    worker = threading.Thread(target = _fetch, name = "gguf-header-read", daemon = True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        _interrupt_read(holder[0])
        worker.join(_ABANDON_GRACE_SECONDS)
    watchdog.cancel()
    # Same prefix parse as remote; the full loader memory-maps every tensor, so use it only when the prefix says nothing, e.g. a header past the cap
    return bytes(buffer[:max_bytes])


def flux2_inner_dim_for_pick(
    repo_id: str,
    gguf_filename: Optional[str],
    hf_token: Optional[str] = None,
    *,
    allow_network: bool = True,
) -> Optional[int]:
    """inner_dim from disk or a header range read, memoised; allow_network=False never waits on the Hub."""
    if not repo_id or not gguf_filename or not gguf_filename.lower().endswith(".gguf"):
        return None
    token = normalize_token(hf_token)
    # File identity is part of the memo key.
    local = _local_gguf_path(repo_id, gguf_filename)
    key = (repo_id, gguf_filename, _token_fingerprint(token), _file_identity(local))
    # Memo before the offline bail, so begin_load reuses a plan-time answer.
    with _CACHE_LOCK:
        if key in _INNER_DIM_CACHE:
            return _INNER_DIM_CACHE[key]
    if local is None and not allow_network:
        return None
    if local is not None:
        # Same prefix parse as the remote path, so both read the file the same way: the loader's
        # backstop memory-maps the whole multi-GB checkpoint and builds a view over every tensor,
        # which is a lot of work for a table in the first 15 KiB. Fall back to it only if the
        # prefix said nothing, so a header past the cap is still answered.
        inner_dim = gguf_flux2_inner_dim_from_header(_read_local_header(local))
        if inner_dim is None:
            inner_dim = gguf_flux2_inner_dim(local)
    else:
        inner_dim = gguf_flux2_inner_dim_from_header(
            _shared_gguf_header(repo_id, gguf_filename, token, local)
        )
    with _CACHE_LOCK:
        # Plain FIFO-ish eviction: this only bounds a session's worth of picks, and a re-probe
        # after an eviction costs one range request.
        if len(_INNER_DIM_CACHE) >= _INNER_DIM_CACHE_MAX:
            _INNER_DIM_CACHE.clear()
        _INNER_DIM_CACHE[key] = inner_dim
    return inner_dim


def _revalidated_inner_dim(
    repo_id: str, gguf_filename: str, hf_token: Optional[str], got: int
) -> Optional[int]:
    """Re-checks a stale cached copy on the Hub, only on a would-be refusal; unknown revisions keep got."""
    cached = _snapshot_revision(_local_gguf_path(repo_id, gguf_filename))
    if cached is None:
        return got
    token = normalize_token(hf_token)
    live = _hub_revision(repo_id, gguf_filename, token)
    if live is None or live == cached:
        return got
    return gguf_flux2_inner_dim_from_header(_read_gguf_header(repo_id, gguf_filename, token))


def flux2_pick_mismatch(
    fam: Any,
    repo_id: str,
    gguf_filename: Optional[str],
    base_repo: Optional[str],
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """Why this GGUF cannot load on this base, or None; pass the resolved upstream base_repo."""
    if not gguf_filename or not str(getattr(fam, "name", "")).startswith("flux.2"):
        return None
    want = flux2_base_inner_dim(base_repo)
    # Cheapest order: a base outside the size table (a local path, a repo we do not ship) leaves
    # nothing to compare against, so it must not cost a round trip either.
    if want is None:
        return None
    got = flux2_inner_dim_for_pick(repo_id, gguf_filename, hf_token)
    if got is not None and got != want:
        got = _revalidated_inner_dim(repo_id, gguf_filename, hf_token, got)
    return flux2_mismatch_reason(
        Path(str(gguf_filename)).name,
        str(base_repo),
        got,
        want,
    )


# Shared with the chat gate and listing classifier so they cannot drift.
from utils.gguf_archs import (  # noqa: E402 -- beside the cache it keys
    SPEECH_GGUF_ARCHS as _SPEECH_GGUF_ARCHS,
    is_speech_gguf_architecture,
)

_SPEECH_ARCH_CACHE: dict[
    tuple[str, str, str, Optional[tuple]], tuple[Optional[str], Optional[float]]
] = {}
_SPEECH_ARCH_CACHE_MAX = 256
# Only a true On Device checkpoint is permanent; Hub-backed verdicts age out.
_SPEECH_REMOTE_TTL_SECONDS = 60.0

# Shared header read for the inner-dim and speech probes; revalidation bypasses it.
_HEADER_PREFIX_CACHE: dict[tuple[str, str, str, Optional[tuple]], tuple[bytes, float]] = {}
_HEADER_PREFIX_CACHE_MAX = 32


def _shared_gguf_header(
    repo_id: str, gguf_filename: str, token: Optional[str], local: Optional[str]
) -> bytes:
    """``_read_gguf_header``, read once for the probes that run back to back on one pick."""
    key = (repo_id, gguf_filename, _token_fingerprint(token), _file_identity(local))
    now = time.monotonic()
    with _CACHE_LOCK:
        memo = _HEADER_PREFIX_CACHE.get(key)
        if memo is not None:
            prefix, expires_at = memo
            if now < expires_at:
                return prefix
            del _HEADER_PREFIX_CACHE[key]
    prefix = _read_gguf_header(repo_id, gguf_filename, token)
    # An empty prefix is a failed read, and the two probes disagreeing about that is not worth a
    # sticky miss: each still memoises its own "no verdict" on its own terms.
    if not prefix:
        return prefix
    with _CACHE_LOCK:
        if len(_HEADER_PREFIX_CACHE) >= _HEADER_PREFIX_CACHE_MAX:
            _HEADER_PREFIX_CACHE.clear()
        _HEADER_PREFIX_CACHE[key] = (prefix, now + _SPEECH_REMOTE_TTL_SECONDS)
    return prefix


def _arch_from_prefix(prefix: bytes, gguf_filename: str) -> Optional[str]:
    """Magic, version and the two counts: anything shorter is not a GGUF at all."""
    if len(prefix) < 24:
        return None
    try:
        import tempfile

        from utils.models.gguf_metadata import read_gguf_architecture
        with tempfile.TemporaryDirectory(prefix = "unsloth-speech-probe-") as probe_dir:
            # Named after the real file: a GGUF without architecture is judged by name.
            probe_path = os.path.join(probe_dir, os.path.basename(gguf_filename))
            with open(probe_path, "wb") as handle:
                handle.write(prefix)
            return (read_gguf_architecture(probe_path) or "").strip().lower() or None
    except Exception:  # noqa: BLE001 -- a probe that failed is not a verdict
        return None


def _revalidated_speech_arch(
    repo_id: str,
    gguf_filename: str,
    token: Optional[str],
    local: Optional[str],
    arch: Optional[str],
    allow_network: bool = True,
) -> Optional[str]:
    """Re-checks arch on the Hub both ways; a stale allow would admit speech bytes. Offline keeps arch."""
    cached = _snapshot_revision(local)
    if cached is None:
        return arch
    if not allow_network:
        # A cache-only caller cannot wear the HEAD; the caller declines to memoise this answer,
        # so the next one that CAN reach the Hub still revalidates it.
        return arch
    live = _hub_revision(repo_id, gguf_filename, token)
    if live is None or live == cached:
        return arch
    refreshed = _arch_from_prefix(_read_gguf_header(repo_id, gguf_filename, token), gguf_filename)
    # Keep the old verdict on a failed re-read; failing open is only for unknown picks.
    return refreshed if refreshed is not None else arch


def _speech_probe_architecture(
    repo_id: str,
    gguf_filename: str,
    hf_token: Optional[str],
    allow_network: bool = True,
) -> Optional[str]:
    """Memo keyed on token fingerprint and file identity, so a failed probe is not reused across tokens."""
    token = normalize_token(hf_token)
    # Resolved BEFORE the memo is consulted, because the file's identity is part of the key.
    local = _local_gguf_path(repo_id, gguf_filename)
    key = (repo_id, gguf_filename, _token_fingerprint(token), _file_identity(local))
    with _CACHE_LOCK:
        memo = _SPEECH_ARCH_CACHE.get(key)
        if memo is not None:
            arch, expires_at = memo
            if expires_at is None or time.monotonic() < expires_at:
                return arch
            del _SPEECH_ARCH_CACHE[key]
    if local is None and not allow_network:
        # Memo or local header only, as the size pairing does. Nothing is memoised, so the next
        # caller that CAN wait still gets a real answer instead of this one's silence.
        return None
    prefix = (
        _read_local_header(local)
        if local
        else _shared_gguf_header(repo_id, gguf_filename, token, local)
    )
    arch = _arch_from_prefix(prefix, gguf_filename)
    # Permanent only for a true on-device file (no revision to fall behind); a cached Hub snapshot must age out or it asks the Hub once per file per process
    arch = _revalidated_speech_arch(repo_id, gguf_filename, token, local, arch, allow_network)
    # Skipped revision check is half an answer: do not memoise.
    if not allow_network and _snapshot_revision(local) is not None:
        return arch
    with _CACHE_LOCK:
        if len(_SPEECH_ARCH_CACHE) >= _SPEECH_ARCH_CACHE_MAX:
            _SPEECH_ARCH_CACHE.clear()
        # auto saves by default (without a saved bundle no user gets a warm restart); the SAVE env var overrides: "0" load-only, "1" on
        permanent = local is not None and _snapshot_revision(local) is None
        _SPEECH_ARCH_CACHE[key] = (
            arch,
            None if permanent else time.monotonic() + _SPEECH_REMOTE_TTL_SECONDS,
        )
    return arch


def speech_pick_refusal(
    repo_id: str,
    gguf_filename: Optional[str],
    hf_token: Optional[str] = None,
    allow_network: bool = True,
) -> Optional[str]:
    """Why this diffusion pick cannot load, when it names a speech or text-encoder GGUF, else None.

    A media pick names its file, and ``detect_family_for_pick`` resolves the family from the FOLDER
    rather than that name, so a csm quant sitting beside a FLUX denoiser answers flux.1: the pick
    pulls the checkpoint and tears the resident pipeline down before the loader finds out.

    Metadata only, like the FLUX.2 pairing above: a cached copy answers with no request, else one
    range request. Fails open on everything -- no filename, an unreadable header, an offline host,
    a server that ignores Range -- because refusing a pick that works is worse than the download
    this saves.
    """
    # A ".gguf" name only, as the size pairing does: a single_file pick names a .safetensors,
    # which has no GGUF header, and a range request to learn that on every such load is waste.
    if not repo_id or not gguf_filename or not gguf_filename.lower().endswith(".gguf"):
        return None
    arch = _speech_probe_architecture(repo_id, gguf_filename, hf_token, allow_network)
    if is_speech_gguf_architecture(arch):
        # The Mimi vocoder puts a whole sentence in general.architecture.
        named = f"{arch} " if arch in _SPEECH_GGUF_ARCHS else ""
        return (
            f"'{os.path.basename(gguf_filename)}' is a {named}speech checkpoint, which no image "
            "or video backend can decode. Pick one of this folder's media GGUFs instead."
        )
    # A text-encoder GGUF (a FLUX.2-klein Qwen3 encoder repo tagged text-to-image, #9875) matches a family by name, then
    # loads as the denoiser with every tensor unmapped and dies in .to() with "Cannot copy out of meta tensor".
    from core.inference.diffusion_content import _TE_GGUF_ARCHS

    if arch in _TE_GGUF_ARCHS:
        return (
            f"'{os.path.basename(gguf_filename)}' is a text encoder ({arch} GGUF), not a diffusion "
            "model (DiT/UNet), so it cannot be loaded as the image or video model. Pick the "
            "model's diffusion GGUF instead."
        )
    return None


def assert_pick_is_not_speech(
    repo_id: str,
    gguf_filename: Optional[str],
    hf_token: Optional[str] = None,
    allow_network: bool = True,
) -> None:
    """Raises ValueError, not RuntimeError, so the route returns 400 rather than a bare 500."""
    reason = speech_pick_refusal(repo_id, gguf_filename, hf_token, allow_network)
    if reason is not None:
        raise ValueError(reason)


def assert_flux2_pick_compatible(
    fam: Any,
    repo_id: str,
    gguf_filename: Optional[str],
    base_repo: Optional[str],
    hf_token: Optional[str] = None,
) -> None:
    """Raises ValueError, not RuntimeError, so the route returns 400 rather than a bare 500."""
    reason = flux2_pick_mismatch(fam, repo_id, gguf_filename, base_repo, hf_token)
    if reason is not None:
        raise ValueError(reason)


def _flux3_pipeline_index(
    repo_id: str, token: Optional[str], allow_network: bool
) -> Optional[dict]:
    """A FLUX.3 pipeline's ``modular_model_index.json``: the local folder's, the cached copy, or (when allowed) a
    fetch of that one small file. None on anything unreadable."""
    import json

    from .diffusion_flux3 import FLUX3_INDEX

    try:
        root = Path(repo_id).expanduser()
        if root.is_dir():
            return json.loads((root / FLUX3_INDEX).read_text(encoding = "utf-8-sig"))
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        # The gated vendor repo's manifest is read from its byte-identical unsloth mirror (pure table lookup), so a
        # load that fetches everything from the mirror does not also probe, and 401 on, the vendor repo.
        from core.inference.diffusion_families import prefer_ungated_mirror

        source = prefer_ungated_mirror(repo_id, token, files = (FLUX3_INDEX,))
        local = _local_gguf_path(repo_id, FLUX3_INDEX)
        if local is None and source != repo_id:
            local = _local_gguf_path(source, FLUX3_INDEX)
        if local is None and allow_network:
            from huggingface_hub import hf_hub_download
            from utils.hf_cache_settings import active_hf_hub_cache
            local = hf_hub_download(
                source, FLUX3_INDEX, token = token, cache_dir = active_hf_hub_cache()
            )
        if local is None:
            return None
        payload = json.loads(Path(local).read_text(encoding = "utf-8-sig"))
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 -- offline, gated or odd: the loader's own check still runs
        return None


def _flux3_pick_tensor_names(
    repo_id: str, filename: str, token: Optional[str], allow_network: bool
) -> tuple[Optional[list[str]], bool]:
    """Tensor names of a FLUX.3 single file / GGUF pick without reading a weight, and whether it declares ComfyUI
    quantization: the file on disk, else a GGUF header range read or the safetensors header from the Hub. None
    names on anything unreadable."""
    from .diffusion_families import (
        _checkpoint_comfy_declared,
        _checkpoint_tensor_names,
        flux3_comfy_declared,
        gguf_tensor_names_from_header,
    )

    local = _local_gguf_path(repo_id, filename)
    if local is not None:
        return _checkpoint_tensor_names(local), _checkpoint_comfy_declared(local)
    if not allow_network:
        return None, False
    try:
        if filename.lower().endswith(".gguf"):
            return gguf_tensor_names_from_header(
                _shared_gguf_header(repo_id, filename, token, None)
            ), False
        from huggingface_hub import HfApi

        meta = HfApi().parse_safetensors_file_metadata(repo_id, filename, token = token)
        names = list(meta.tensors)
        return names, flux3_comfy_declared(names, getattr(meta, "metadata", None))
    except Exception:  # noqa: BLE001 -- offline, gated or odd: the loader's own check still runs
        return None, False


def flux3_pick_mismatch(
    fam: Any,
    repo_id: str,
    gguf_filename: Optional[str],
    hf_token: Optional[str] = None,
    allow_network: bool = True,
) -> Optional[str]:
    """Why a FLUX.3 pick cannot load under ``fam``, or None when nothing is known to be wrong: a single file /
    GGUF of the other variant (or a BFL quantized package) from its tensor names, a full pipeline of the other
    variant from its manifest. Headers and the manifest only; None on anything unreadable."""
    name = str(getattr(fam, "name", ""))
    if not name.startswith("flux.3"):
        return None
    if not gguf_filename:
        from .diffusion_flux3 import flux3_family_blocks_mismatch, flux3_index_family
        declared = flux3_index_family(
            _flux3_pipeline_index(repo_id, normalize_token(hf_token), allow_network)
        )
        return flux3_family_blocks_mismatch(name, declared, repo_id)
    from .diffusion_families import flux3_variant_mismatch_reason

    names, comfy_declared = _flux3_pick_tensor_names(
        repo_id, gguf_filename, normalize_token(hf_token), allow_network
    )
    return flux3_variant_mismatch_reason(
        name, Path(str(gguf_filename)).name, names, comfy_declared = comfy_declared
    )


def assert_flux3_pick_compatible(
    fam: Any,
    repo_id: str,
    gguf_filename: Optional[str],
    hf_token: Optional[str] = None,
    allow_network: bool = True,
) -> None:
    """Refuse a FLUX.3 pick of the other variant before anything is downloaded or unloaded
    (``flux3_pick_mismatch``). The loader's own checks run after the resident pipeline is gone."""
    reason = flux3_pick_mismatch(fam, repo_id, gguf_filename, hf_token, allow_network)
    if reason is not None:
        raise ValueError(reason)


def _reset_inner_dim_cache() -> None:
    """Drop the memoised header probes. Tests only."""
    with _CACHE_LOCK:
        _INNER_DIM_CACHE.clear()
        _SPEECH_ARCH_CACHE.clear()
        _HEADER_PREFIX_CACHE.clear()
