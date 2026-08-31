# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Metadata-only compatibility preflight for a diffusion pick.

A FLUX.2 GGUF only carries the transformer; its size (``inner_dim``) has to agree with the
companion diffusers base repo the loader assembles around it. ``assert_flux2_gguf_matches_base``
already catches a mismatch, but it opens the downloaded checkpoint, so it fires from inside
``load_pipeline`` -- after the prefetch pulled ~19 GB of base shards and after the resident
pipeline was torn down to make room. The user paid for both to be told the pick was never valid.

This module answers the same question from metadata alone: one HTTP range request for the first
few hundred KiB of the GGUF, which is where its tensor table lives. That is cheap enough to run
at SELECTION time (``/images/download-plan``) and again on the pre-eviction path, so the refusal
lands before a byte moves and before anything is unloaded.

Fail-open throughout, deliberately: an unreadable or truncated header, a base repo outside the
size table, an offline host, a server that ignores Range -- all yield "no opinion", and the load
proceeds exactly as it does today with the loader's own guard as the backstop. A false positive
here would refuse a pick that works, which is strictly worse than the download this saves.

(A known ungated MIRROR of a base is not an exception to that: it is byte-identical to what it
copies, ``canonical_base`` maps it back, and it is checked like its upstream.)
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from core.inference.diffusion_families import (
    flux2_base_inner_dim,
    flux2_mismatch_reason,
    gguf_flux2_inner_dim,
    gguf_flux2_inner_dim_from_header,
    resolve_local_gguf_child,
)

# The FLUX.2 tensor table sits in the first ~15 KiB (149 tensors for klein-4B, 201 for 9B). This
# is the ceiling on what the range request may buffer, not an expectation: a prefix that stops
# mid-table makes the parse raise, which reads as "no opinion".
_GGUF_HEADER_BYTES = 256 * 1024
# One short read. A pick is blocked on this in the UI, and a slow Hub must not stall the picker;
# a timeout is just another fail-open.
_HEADER_TIMEOUT_SECONDS = 15
# How long to wait for an interrupted read to notice before leaving it to the GC.
_ABANDON_GRACE_SECONDS = 0.5

# (repo_id, gguf_filename, token fingerprint, local file identity) -> inner_dim or None. Bounded
# and process-local. It memoises the MISS too -- the three checks on one pick would otherwise
# re-probe an unreachable Hub three times, and a sticky None is the degradation this module
# promises anyway. Which is exactly why the last two key parts exist: a sticky None must not
# outlive its cause.
#
#   * the TOKEN, fingerprinted rather than stored. Keying on mere presence made every non-empty
#     token one key, so a first probe with an expired credential poisoned the valid one that
#     replaced it for the rest of the process.
#   * the local file's IDENTITY (path, size, mtime). A checkpoint swapped in place keeps its path,
#     so keying on the name alone answers the new file with the old file's dim -- refusing a valid
#     9B pairing, or handing sd.cpp the 4B text encoders. It also makes the file ARRIVING a new
#     key, so a miss taken before a download finished re-probes off disk for free.
_INNER_DIM_CACHE: dict[tuple[str, str, str, Optional[tuple]], Optional[int]] = {}
_INNER_DIM_CACHE_MAX = 256
_CACHE_LOCK = threading.Lock()


def _token_fingerprint(token: Optional[str]) -> str:
    """A stable, non-reversible tag for a token, or "" for none. Never the token itself: this
    lands in a process-global dict that a traceback or a heap dump would render."""
    if not token:
        return ""
    return hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:16]


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
    # OSError/RuntimeError: invalid path characters, or an unresolvable '~' -> a remote id.
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        from utils.hf_cache_settings import active_hf_hub_cache

        # The live root first, then huggingface_hub's import-time constant, the same pair the
        # loader resolves a staged file through. Read directly rather than through
        # ``diffusion.hub_cache_dir``: that module imports this one.
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
    # ValueError: open() rejects an embedded NUL rather than raising OSError.
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
        # httpx does not follow redirects by default and the Hub answers a resolve URL with a
        # 302 to the CDN, so an unfollowed hop would read as "not 206" and fail open.
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


def _read_gguf_header(repo_id: str, gguf_filename: str, hf_token: Optional[str]) -> bytes:
    """The first ``_GGUF_HEADER_BYTES`` of a Hub-hosted GGUF, or b"" when they cannot be read.

    One wall-clock bound over the WHOLE operation, request included. requests' timeout is an
    inactivity timeout: a peer (or an intermediary) that trickles response HEADERS resets it on
    every byte, so a deadline armed only once ``get()`` has returned leaves the caller blocked
    before the bounded body reader is ever reached -- and this runs on the /images/load route
    thread and on the download-plan path, both of which promised to fail open in seconds.

    So the request AND the drain run on a worker this call can walk away from. On timeout the
    response, if one exists by then, is half-closed to wake the worker; either way the caller
    returns with whatever arrived."""
    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session
    except Exception:  # noqa: BLE001 — an unexpected hub layout leaves today's behaviour
        return b""
    buffer = bytearray()
    # Published by the worker as soon as it has something interruptible; read by this thread on
    # timeout. A one-element list rather than a nonlocal, so the worker's assignment is visible.
    holder: list[Any] = [None]

    def _fetch() -> None:
        try:
            headers = dict(build_hf_headers(token = hf_token))
            headers["Range"] = f"bytes=0-{_GGUF_HEADER_BYTES - 1}"
            with _ranged_stream(
                get_session(), hf_hub_url(repo_id, gguf_filename), headers
            ) as response:
                holder[0] = response
                # 206 or nothing. A server (or a proxy) that ignored the Range header answers 200
                # with the WHOLE checkpoint, and streaming that into memory is the multi-GB
                # download this preflight exists to prevent.
                if response.status_code != 206:
                    return
                deadline = time.monotonic() + _HEADER_TIMEOUT_SECONDS
                for chunk in _iter_body(response, 65536):
                    # extend, not `+=`: augmented assignment to a closed-over name would rebind
                    # it as a local of _fetch and lose every byte.
                    buffer.extend(chunk)
                    if len(buffer) >= _GGUF_HEADER_BYTES or time.monotonic() > deadline:
                        break
        # Keep what arrived rather than discarding it: the deadline firing on a merely SLOW link
        # still leaves the tensor table (the first ~15 KiB) in hand, and the parser is
        # truncation-safe -- swept over every prefix length of five header layouts, no cut ever
        # produces a wrong dim, so a short prefix is answered or ignored. TRUNCATION only: a
        # header with flipped bytes can still parse to a wrong dim (~0.6% under a 1-4 byte flip),
        # which the loader's own full-file backstop shares. TLS makes that unlikely on this path.
        except Exception:  # noqa: BLE001 — offline, deadline fired, or the peer went away
            pass

    # The watchdog exists as well as the join because iter_content blocks inside urllib3 until a
    # whole 64 KiB chunk has arrived and every dribbled byte resets the socket timeout, so the
    # worker cannot notice its own deadline. Half-closing the socket is what makes that read
    # return -- on urllib3 >= 2.3, where HTTPResponse.shutdown exists. requirements/studio.txt
    # floors it, but an install predating that floor keeps whatever it resolved, so the bound
    # here cannot depend on the version underneath us: the worker is abandonable either way.
    watchdog = threading.Timer(_HEADER_TIMEOUT_SECONDS, lambda: _interrupt_read(holder[0]))
    watchdog.daemon = True
    watchdog.start()
    worker = threading.Thread(target = _fetch, name = "gguf-header-read", daemon = True)
    worker.start()
    worker.join(_HEADER_TIMEOUT_SECONDS)
    if worker.is_alive():
        _interrupt_read(holder[0])
        worker.join(_ABANDON_GRACE_SECONDS)
    watchdog.cancel()
    # bytes() snapshots under the GIL, so an abandoned worker still appending cannot tear the
    # copy; it can only lose a chunk that arrived too late to matter.
    return bytes(buffer[:_GGUF_HEADER_BYTES])


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
    # A ".gguf" name only: a single_file load names a .safetensors, which has no GGUF header, and
    # spending a range request to learn that on every such load is pure waste.
    if not repo_id or not gguf_filename or not gguf_filename.lower().endswith(".gguf"):
        return None
    token = (hf_token or "").strip() or None
    # Resolved BEFORE the memo is consulted, because the file's identity is part of the key. Two
    # stats, against a probe that is otherwise an HTTP round trip.
    local = _local_gguf_path(repo_id, gguf_filename)
    key = (repo_id, gguf_filename, _token_fingerprint(token), _file_identity(local))
    # The memo FIRST, before the offline bail below. A plan-time probe has usually already
    # answered for this exact pick, and returning None here anyway made the caller that cannot
    # wait (begin_load, allow_network = False) fall back to the filename heuristic -- publishing
    # the 4B encoder repos for a renamed 9B checkpoint, so the delete-cached guard did not cover
    # its real companion repo until the worker re-probed.
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
    """``got`` again, re-read off the Hub when it came from a cached copy the Hub has moved past.

    ``try_to_load_from_cache`` resolves the LOCAL ``refs/main``, so a checkpoint republished at the
    same filename would otherwise refuse a pick that the loader's own ``hf_hub_download`` refreshes
    and loads. Runs only on a would-be refusal; an unknown revision keeps ``got``, and a live
    header we cannot read is no opinion."""
    cached = _snapshot_revision(_local_gguf_path(repo_id, gguf_filename))
    if cached is None:
        return got
    token = (hf_token or "").strip() or None
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


# GGUF ``general.architecture`` values nothing in Unsloth can decode. Beside the FLUX.2 check
# because both ask whether the pick is loadable, off the same prefix. The set itself lives in a
# leaf module, shared with the chat gate and the listing classifier so they cannot drift.
from utils.gguf_archs import (  # noqa: E402 -- beside the cache it keys
    SPEECH_GGUF_ARCHS as _SPEECH_GGUF_ARCHS,
    is_speech_gguf_architecture,
)

_SPEECH_ARCH_CACHE: dict[
    tuple[str, str, str, Optional[tuple]], tuple[Optional[str], Optional[float]]
] = {}
_SPEECH_ARCH_CACHE_MAX = 256
# Every remote-backed verdict ages out. An UNCACHED one keys on a local identity of None, so a
# republish under the same filename changes nothing about the key. A SNAPSHOT-backed one keys on
# the file's identity, which a republish does change -- but only once the new bytes are down, and
# the entry memoises a revision check that ran only the first time, so holding it forever means
# never asking the Hub again for the life of the process. Only a true On Device checkpoint is
# permanent: it is the file the loader opens, so there is no revision to be behind. Matches the
# variant listing's own freshness window for moved revisions.
_SPEECH_REMOTE_TTL_SECONDS = 60.0

# (repo_id, gguf_filename, token fingerprint, local file identity) -> the header prefix.
#
# The inner-dim probe and the speech probe read the SAME first _GGUF_HEADER_BYTES of the SAME
# file, and a flux.2 pick that is not a size mismatch runs both: two range requests, each with its
# own _HEADER_TIMEOUT_SECONDS, so a picker the user waits on could wear twice its documented
# bound. They share the read now, keyed and aged exactly like the speech memo beside it, so this
# adds no staleness the module did not already accept.
#
# Deliberately NOT consulted by the revalidation paths: their whole job is to re-read a file the
# Hub has republished, and answering those from a memo would defeat them.
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
    """``general.architecture`` out of a header prefix, or None when it says nothing."""
    # Magic, version and the two counts: anything shorter is not a GGUF at all.
    if len(prefix) < 24:
        return None
    try:
        import tempfile

        from utils.models.gguf_metadata import read_gguf_architecture
        with tempfile.TemporaryDirectory(prefix = "unsloth-speech-probe-") as probe_dir:
            # Named after the real file, like the chat-side probe: a GGUF declaring no
            # architecture is judged by its name, which a temp name would lose.
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
        # A cache-only caller cannot wear the HEAD; the caller declines to memoise this answer,
        # so the next one that CAN reach the Hub still revalidates it.
        return arch
    live = _hub_revision(repo_id, gguf_filename, token)
    if live is None or live == cached:
        return arch
    refreshed = _arch_from_prefix(_read_gguf_header(repo_id, gguf_filename, token), gguf_filename)
    # A re-read that said nothing -- failed range request, or an unparseable new header -- keeps
    # the verdict we had rather than replacing it with silence. Failing open on an UNKNOWN pick is
    # the contract; throwing away a known one let a csm file through on a dropped connection.
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
    token = (hf_token or "").strip() or None
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
    # Inside the memo, so a republished checkpoint is caught in either direction and the HEAD is
    # spent once per cached copy rather than on every pick.
    arch = _revalidated_speech_arch(repo_id, gguf_filename, token, local, arch, allow_network)
    # A cached copy whose revision check was skipped is only HALF an answer, so it must not be
    # memoised: the network-allowed caller behind it would read this back and never revalidate.
    if not allow_network and _snapshot_revision(local) is not None:
        return arch
    with _CACHE_LOCK:
        if len(_SPEECH_ARCH_CACHE) >= _SPEECH_ARCH_CACHE_MAX:
            _SPEECH_ARCH_CACHE.clear()
        # Permanent only for a true On Device file, which has no revision to be behind. A cached
        # Hub snapshot ages out like an uncached pick: its entry memoises a revision check, and
        # holding that forever would ask the Hub exactly once per file per process.
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
    # A ".gguf" name only, as the size pairing does: a single_file pick names a .safetensors,
    # which has no GGUF header, and a range request to learn that on every such load is waste.
    if not repo_id or not gguf_filename or not gguf_filename.lower().endswith(".gguf"):
        return None
    arch = _speech_probe_architecture(repo_id, gguf_filename, hf_token, allow_network)
    if is_speech_gguf_architecture(arch):
        # Named only when the header carried an identifier: the Mimi vocoder puts a whole
        # sentence in general.architecture, and quoting that back reads as gibberish.
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
