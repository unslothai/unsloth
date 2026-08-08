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

# (repo_id, gguf_filename, has_token) -> inner_dim or None. Bounded and process-local. It memoises
# the MISS too -- the three checks on one pick would otherwise re-probe an unreachable Hub three
# times, and a sticky None is precisely the degradation this module promises anyway -- which is why
# the token's PRESENCE is part of the key: a gated GGUF probed anonymously misses, and pasting a
# token afterwards has to earn a fresh probe rather than inherit that None for the session.
_INNER_DIM_CACHE: dict[tuple[str, str, bool], Optional[int]] = {}
_INNER_DIM_CACHE_MAX = 256
_CACHE_LOCK = threading.Lock()


def _local_gguf_path(repo_id: str, gguf_filename: str) -> Optional[str]:
    """The on-disk checkpoint for this pick, or None when it has to come off the Hub.

    Covers a local On Device directory and a Hub file already in either cache root: reading a file
    we hold beats a range request, and it is the same file ``_resolve_gguf_path`` will open."""
    try:
        local_root = Path(repo_id).expanduser()
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


def _read_local_header(path: str) -> bytes:
    """The first ``_GGUF_HEADER_BYTES`` of a file on disk, or b"" when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            return handle.read(_GGUF_HEADER_BYTES)
    # ValueError: open() rejects an embedded NUL rather than raising OSError.
    except (OSError, ValueError):
        return b""


def _read_gguf_header(repo_id: str, gguf_filename: str, hf_token: Optional[str]) -> bytes:
    """The first ``_GGUF_HEADER_BYTES`` of a Hub-hosted GGUF, or b"" when they cannot be read."""
    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session
    except Exception:  # noqa: BLE001 — an unexpected hub layout leaves today's behaviour
        return b""
    try:
        headers = dict(build_hf_headers(token = hf_token))
        headers["Range"] = f"bytes=0-{_GGUF_HEADER_BYTES - 1}"
        with get_session().get(
            hf_hub_url(repo_id, gguf_filename),
            headers = headers,
            timeout = _HEADER_TIMEOUT_SECONDS,
            stream = True,
        ) as response:
            # 206 or nothing. A server (or a proxy) that ignored the Range header answers 200 with
            # the WHOLE checkpoint, and streaming that into memory is the multi-GB download this
            # preflight exists to prevent.
            if response.status_code != 206:
                return b""
            # requests' timeout is per socket read, so a server trickling a byte at a time holds
            # the picker open forever inside the byte cap. One deadline for the whole body.
            deadline = time.monotonic() + _HEADER_TIMEOUT_SECONDS
            buffer = bytearray()
            for chunk in response.iter_content(chunk_size = 65536):
                buffer += chunk
                if len(buffer) >= _GGUF_HEADER_BYTES or time.monotonic() > deadline:
                    break
            return bytes(buffer[:_GGUF_HEADER_BYTES])
    except Exception:  # noqa: BLE001 — offline/transient must not block a load
        return b""


def flux2_inner_dim_for_pick(
    repo_id: str,
    gguf_filename: Optional[str],
    hf_token: Optional[str] = None,
) -> Optional[int]:
    """``inner_dim`` of the GGUF this pick names, WITHOUT downloading it, or None.

    Reads the file when it is already on disk, otherwise range-reads its header off the Hub.
    Memoised per (repo, filename) so the plan, the pre-eviction preflight and the native asset
    resolver share one probe."""
    # A ".gguf" name only: a single_file load names a .safetensors, which has no GGUF header, and
    # spending a range request to learn that on every such load is pure waste.
    if not repo_id or not gguf_filename or not gguf_filename.lower().endswith(".gguf"):
        return None
    token = (hf_token or "").strip() or None
    key = (repo_id, gguf_filename, token is not None)
    with _CACHE_LOCK:
        if key in _INNER_DIM_CACHE:
            return _INNER_DIM_CACHE[key]
    local = _local_gguf_path(repo_id, gguf_filename)
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
            _read_gguf_header(repo_id, gguf_filename, token)
        )
    with _CACHE_LOCK:
        # Plain FIFO-ish eviction: this only bounds a session's worth of picks, and a re-probe
        # after an eviction costs one range request.
        if len(_INNER_DIM_CACHE) >= _INNER_DIM_CACHE_MAX:
            _INNER_DIM_CACHE.clear()
        _INNER_DIM_CACHE[key] = inner_dim
    return inner_dim


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
    return flux2_mismatch_reason(
        Path(str(gguf_filename)).name,
        str(base_repo),
        flux2_inner_dim_for_pick(repo_id, gguf_filename, hf_token),
        want,
    )


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
