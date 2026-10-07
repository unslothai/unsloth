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
_HEADER_TIMEOUT_SECONDS = 15
_ABANDON_GRACE_SECONDS = 0.5

# Memoises misses too; token fingerprint and file identity keep a stale None from sticking.
_INNER_DIM_CACHE: dict[tuple[str, str, str, Optional[tuple]], Optional[int]] = {}
_INNER_DIM_CACHE_MAX = 256
_CACHE_LOCK = threading.Lock()


def _token_fingerprint(token: HfTokenArg) -> str:
    """A stable, non-reversible tag for a token, or "" for none. Never the token itself: this
    lands in a process-global dict that a traceback or a heap dump would render.

    A caller forced anonymous tags apart from one that may still use the ambient token,
    so neither reads back the other's verdict."""
    if is_anonymous(token):
        return ANONYMOUS_CACHE_IDENTITY
    if not token:
        return ""
    return qualify_cache_identity(
        token, hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:16]
    )


def _file_identity(path: Optional[str]) -> Optional[tuple]:
    """(path, size, mtime_ns) for a local checkpoint, or None when the pick is remote.

    A file replaced under the same name is a different checkpoint, and stat is the cheapest thing
    that says so. An unreadable stat returns a unique object rather than a constant, so a file we
    cannot identify is never memoised as equal to anything else."""
    if path is None:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return (path, object())
    return (path, stat.st_size, stat.st_mtime_ns)


def _local_gguf_path(repo_id: str, gguf_filename: str) -> Optional[str]:
    """The on-disk checkpoint for this pick, or None when it has to come off the Hub.

    Covers a local On Device directory, a pick that NAMES the checkpoint outright, and a Hub file
    already in either cache root: reading a file we hold beats a range request, and it is the same
    file ``_resolve_gguf_path`` will open.

    The file case is resolved the way the loader resolves it:
    ``VideoBackend._resolve_checkpoint_path`` answers a file-valued ``repo_id`` with that file,
    ignoring ``gguf_filename``, and ``validate_load_request`` admits exactly that pick, so
    ``/video/load`` really can be handed one. Appending the filename under a file instead raises
    ``FileNotFoundError``, an ``OSError``, swallowed below as "remote id" -- and failing open on
    the pick the loader is about to open directly is the one hole this exists to close."""
    try:
        local_root = Path(repo_id).expanduser()
        if local_root.is_file():
            return str(local_root)
        if local_root.exists():
            return str(resolve_local_gguf_child(local_root, gguf_filename))
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
    """The commit a cached Hub file was downloaded at, read off its ``snapshots/<sha>/`` parent.

    None for anything that is not an HF cache entry -- an On Device checkpoint is the file the
    loader opens, so there is no revision to be behind."""
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
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        meta = get_hf_file_metadata(
            hf_hub_url(repo_id, gguf_filename),
            token = hf_token,
            timeout = _HEADER_TIMEOUT_SECONDS,
        )
    except Exception:  # noqa: BLE001 — a revision we cannot read is not a verdict
        return None
    return getattr(meta, "commit_hash", None) or None


def _read_local_header(path: str) -> bytes:
    """The first ``_GGUF_HEADER_BYTES`` of a file on disk, or b"" when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            return handle.read(_GGUF_HEADER_BYTES)
    # open() raises ValueError on an embedded NUL.
    except (OSError, ValueError):
        return b""


def _ranged_stream(session: Any, url: str, headers: dict) -> Any:
    """A context manager over a ranged GET, on either HTTP client huggingface_hub ships.

    ``huggingface_hub`` 1.0 replaced requests with httpx, and ``get_session`` returns whichever
    the installed version builds. The two streaming APIs do not overlap: httpx has no
    ``stream = True`` keyword (it streams via ``Client.stream``), so asking for one on 1.x raises
    ``TypeError`` inside the worker's blanket except and every remote probe silently reads nothing
    -- a preflight that refuses nothing. studio.txt floors 1.23 on python >= 3.10 and pins 0.36
    below it, so BOTH are shipped and both have to work.

    ``Client.stream`` is a method; ``requests.Session.stream`` is a plain bool attribute, so the
    branch tests for a callable rather than for the name."""
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
    """Make a read parked on ``response`` return, so the whole-body deadline can be enforced.

    ``urllib3.HTTPResponse.shutdown`` half-closes the socket, which is the only thing that wakes a
    thread blocked inside ``iter_content``: ``Response.close`` drops the file object while the
    socket stays readable, so the read sits there regardless. Best effort -- and on a urllib3
    older than 2.3, which is where ``shutdown`` first appears, there is nothing here that can wake
    it. An httpx response has no ``raw`` at all, so it takes the ``close`` branch. The caller does
    not depend on this working; it reads on a worker it can abandon.

    ``None`` means the worker has not got a response yet -- it is still inside connect or the
    header wait -- so there is nothing to half-close and abandoning it is the whole bound."""
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
    """A bounded prefix of a Hub-hosted GGUF, or b"" when it cannot be read.

    One wall-clock bound over the WHOLE operation: requests' own timeout is per-byte, so a
    trickled response would block this fail-open path past any deadline armed after ``get()``.
    Request and drain therefore run on a worker this call can abandon."""
    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session
    except Exception:  # noqa: BLE001 — an unexpected hub layout leaves today's behaviour
        return b""
    max_bytes = _GGUF_HEADER_BYTES if max_bytes is None else max_bytes
    timeout_seconds = _HEADER_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    buffer = bytearray()
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
    return bytes(buffer[:max_bytes])


def flux2_inner_dim_for_pick(
    repo_id: str,
    gguf_filename: Optional[str],
    hf_token: Optional[str] = None,
    *,
    allow_network: bool = True,
) -> Optional[int]:
    """``inner_dim`` of the GGUF this pick names, WITHOUT downloading it, or None.

    Reads the file when it is already on disk, otherwise range-reads its header off the Hub.
    Memoised per (repo, filename) so the plan, the pre-eviction preflight and the native asset
    resolver share one probe.

    ``allow_network = False`` answers from the memo or from disk and gives up rather than making
    the range request, for a caller that must not block: the range read is bounded but the bound
    is seconds, and a request thread that only wants a hint should not wear them. Nothing is
    memoised in that case, so the next caller that CAN wait still gets a real answer."""
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
        inner_dim = gguf_flux2_inner_dim_from_header(_read_local_header(local))
        if inner_dim is None:
            inner_dim = gguf_flux2_inner_dim(local)
    else:
        inner_dim = gguf_flux2_inner_dim_from_header(
            _shared_gguf_header(repo_id, gguf_filename, token, local)
        )
    with _CACHE_LOCK:
        if len(_INNER_DIM_CACHE) >= _INNER_DIM_CACHE_MAX:
            _INNER_DIM_CACHE.clear()
        _INNER_DIM_CACHE[key] = inner_dim
    return inner_dim


def _revalidated_inner_dim(
    repo_id: str, gguf_filename: str, hf_token: Optional[str], got: int
) -> Optional[int]:
    """``got`` again, re-read off the Hub when it came from a cached copy the Hub has moved past.

    ``try_to_load_from_cache`` resolves the LOCAL ``refs/main``, so a checkpoint republished at the
    same filename would otherwise refuse a pick that the loader's own ``hf_hub_download`` refreshes
    and loads. Runs only on a would-be refusal; an unknown revision keeps ``got``, and a live
    header we cannot read is no opinion."""
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
    """Why this GGUF cannot load against this base, or None when nothing is known to be wrong.

    ``base_repo`` must be the RESOLVED upstream id (``_resolve_base_repo``), the same one the
    loader's own guard is handed, so all the checks on this pairing agree."""
    if not gguf_filename or not str(getattr(fam, "name", "")).startswith("flux.2"):
        return None
    want = flux2_base_inner_dim(base_repo)
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
    if not prefix:
        return prefix
    with _CACHE_LOCK:
        if len(_HEADER_PREFIX_CACHE) >= _HEADER_PREFIX_CACHE_MAX:
            _HEADER_PREFIX_CACHE.clear()
        _HEADER_PREFIX_CACHE[key] = (prefix, now + _SPEECH_REMOTE_TTL_SECONDS)
    return prefix


def _arch_from_prefix(prefix: bytes, gguf_filename: str) -> Optional[str]:
    """``general.architecture`` out of a header prefix, or None when it says nothing."""
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
    """*arch* again, re-read off the Hub when the cached copy it came from is behind.

    ``try_to_load_from_cache`` resolves the LOCAL ``refs/main``, so a republished checkpoint is
    judged off bytes ``hf_hub_download`` is about to replace. BOTH directions, unlike the size
    pairing (refusals only): a stale allow hands csm bytes to a media loader after the download
    and the teardown, the very outcome this preflight exists to prevent. An unknown revision or
    an unreadable live header keeps *arch*, so an offline host never flips a verdict, and no
    CACHED copy means no revision to be behind -- an uncached remote pick and an On Device file
    both skip the HEAD. Memoised by the caller: one HEAD per cached copy per token per session."""
    cached = _snapshot_revision(local)
    if cached is None:
        return arch
    if not allow_network:
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
    """``general.architecture`` of a pick, from a cached copy or one range request.

    Keyed like the inner-dim memo beside it, for the same two reasons: the token fingerprint,
    because a probe that failed on an expired credential caches "no verdict" and the retry with a
    working one would read that back and let the speech file through to the download; the file
    identity, because a checkpoint replaced under the same name is a different checkpoint."""
    token = normalize_token(hf_token)
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
        return None
    prefix = (
        _read_local_header(local)
        if local
        else _shared_gguf_header(repo_id, gguf_filename, token, local)
    )
    arch = _arch_from_prefix(prefix, gguf_filename)
    arch = _revalidated_speech_arch(repo_id, gguf_filename, token, local, arch, allow_network)
    # Skipped revision check is half an answer: do not memoise.
    if not allow_network and _snapshot_revision(local) is not None:
        return arch
    with _CACHE_LOCK:
        if len(_SPEECH_ARCH_CACHE) >= _SPEECH_ARCH_CACHE_MAX:
            _SPEECH_ARCH_CACHE.clear()
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
    """Why this diffusion pick cannot load, when it names a speech GGUF, else None.

    A media pick names its file, and ``detect_family_for_pick`` resolves the family from the FOLDER
    rather than that name, so a csm quant sitting beside a FLUX denoiser answers flux.1: the pick
    pulls the checkpoint and tears the resident pipeline down before the loader finds out.

    Metadata only, like the FLUX.2 pairing above: a cached copy answers with no request, else one
    range request. Fails open on everything -- no filename, an unreadable header, an offline host,
    a server that ignores Range -- because refusing a pick that works is worse than the download
    this saves.
    """
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
    return None


def assert_pick_is_not_speech(
    repo_id: str,
    gguf_filename: Optional[str],
    hf_token: Optional[str] = None,
    allow_network: bool = True,
) -> None:
    """Refuse a speech GGUF pick before anything is downloaded or unloaded.

    ``ValueError`` like the FLUX.2 assert: /images/load maps it to 400 and the download-plan
    catches it, whereas a RuntimeError escapes the plan as a bare 500."""
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
    """Refuse an incompatible FLUX.2 pick before anything is downloaded or unloaded.

    ``ValueError``, like every other unloadable-pick refusal: /images/load maps it to 400 and
    ``/images/download-plan`` catches it, whereas a RuntimeError escapes the plan as a bare 500."""
    reason = flux2_pick_mismatch(fam, repo_id, gguf_filename, base_repo, hf_token)
    if reason is not None:
        raise ValueError(reason)


def _reset_inner_dim_cache() -> None:
    """Drop the memoised header probes. Tests only."""
    with _CACHE_LOCK:
        _INNER_DIM_CACHE.clear()
        _SPEECH_ARCH_CACHE.clear()
        _HEADER_PREFIX_CACHE.clear()
