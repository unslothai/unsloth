# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in: fetch a GGUF a /v1 request names but this server doesn't have.

Auto-switch only loads models already on disk. With
``openai_api_auto_download_model`` on, a miss that looks like a real Hub repo is
downloaded in the background instead of erroring, and the request is told to
retry rather than being held open: a quant is routinely tens of GB, far longer
than any client (or the Cloudflare edge on ``--secure``) will wait, and the
inference lifecycle gate must not be held meanwhile. The resident model keeps
serving throughout, and the retry that lands after the download is served by the
new model through the ordinary auto-switch path.

Admission is deliberately narrow, since a request only needs an API key:
- ``namespace/name`` only. A bare name like ``gpt-4`` falls through to the
  resident model exactly as before, so drop-in clients are unaffected.
- GGUF repos only, decided from the remote file list, not the repo name. GGUF
  runs under llama.cpp, which never imports repo Python.
- Anything declaring ``auto_map`` is refused, so ``trust_remote_code`` can only
  ever be granted deliberately in the UI, never by an API call.
- One download at a time, so a key holder cannot fan out fetches.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Hub metadata is one small request; keep it short so a slow Hub can't stall the
# request path for long.
_MODEL_INFO_TIMEOUT_S = 8.0
# Headroom left free after the download, so filling the disk can't wedge the box.
_DISK_RESERVE_BYTES = 5 * 1024**3
_WATCH_POLL_S = 2.0
# A stalled watcher must not pin the single-flight slot forever.
_MAX_WATCH_S = 24 * 60 * 60
_RETRY_AFTER_S = 30
_MAX_LISTED_VARIANTS = 8


@dataclass(frozen = True)
class AutoDownloadRefusal:
    """Why this request cannot be served yet. The route turns it into an
    HTTPException with the surface's own error envelope."""

    status: int
    code: str
    message: str
    retry_after: Optional[int] = None


@dataclass
class _Active:
    repo_id: str
    # None while the Hub probe is still deciding which quant to fetch.
    variant: Optional[str] = None
    expected_bytes: int = 0
    monitor_id: Optional[str] = None
    started_at: float = 0.0


_lock = threading.Lock()
_active: Optional[_Active] = None


def _public_label(repo_id: str, variant: Optional[str]) -> str:
    return f"{repo_id}:{variant}" if variant else repo_id


def split_model_ref(requested: str) -> tuple[str, Optional[str]]:
    """``org/repo:QUANT`` -> ``("org/repo", "QUANT")``; no suffix -> variant None.

    Splits on the last colon only when the suffix carries no slash, so a Windows
    drive letter or a path never reads as a quant.
    """
    text = (requested or "").strip()
    base, sep, suffix = text.rpartition(":")
    if not sep or not base or "/" in suffix or not suffix:
        return text, None
    return base.strip(), suffix.strip()


def is_downloadable_ref(requested: str) -> bool:
    """Whether *requested* is shaped like a Hub repo we may fetch.

    Requires an explicit namespace. That keeps ``gpt-4`` and other foreign ids
    falling through untouched, and avoids the bare-name ``unsloth/`` prefixing in
    ModelConfig.from_identifier turning an unrelated label into a real repo.
    """
    from hub.utils.paths import is_valid_repo_id

    repo_id, variant = split_model_ref(requested)
    if "/" not in repo_id or not is_valid_repo_id(repo_id):
        return False
    if variant is not None:
        from hub.utils.paths import is_valid_gguf_variant
        return is_valid_gguf_variant(variant)
    return True


def _gguf_variants(siblings) -> dict[str, int]:
    """Quant label -> total bytes, from a model_info sibling list.

    Mirrors list_gguf_variants: companions (mmproj/MTP) and big-endian builds
    are not selectable quants, and sharded quants sum across their shards.
    """
    from utils.models.model_config import (
        _extract_quant_label,
        _is_big_endian_gguf_path,
        _is_mmproj,
        _is_mtp_drafter,
    )

    sizes: dict[str, int] = {}
    for sibling in siblings or []:
        name = getattr(sibling, "rfilename", "") or ""
        if not name.lower().endswith(".gguf"):
            continue
        quant = _extract_quant_label(name)
        if _is_mmproj(name) or _is_mtp_drafter(name) or _is_big_endian_gguf_path(name, quant):
            continue
        sizes[quant] = sizes.get(quant, 0) + int(getattr(sibling, "size", 0) or 0)
    return sizes


def _enough_disk(need_bytes: int) -> tuple[bool, int]:
    """(fits, free_bytes). Fail-open on an unreadable cache root: the download
    worker runs its own preflight, this only adds the reserve margin."""
    try:
        from hub.utils.hf_cache_state import hf_cache_root

        root = hf_cache_root(create = True)
        if root is None:
            return True, 0
        free = shutil.disk_usage(root).free
    except Exception:
        return True, 0
    return free >= need_bytes + _DISK_RESERVE_BYTES, free


def _gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.1f} GB"


async def _job_state(repo_id: str, variant: Optional[str]) -> tuple[str, Optional[str]]:
    from hub.services.models import downloads
    try:
        status = await downloads.get_download_status_response(repo_id, variant or "")
        return status.state, status.error
    except Exception as exc:
        logger.debug("auto-download: status probe failed for %r: %s", repo_id, exc)
        return "idle", None


async def _progress_percent(
    repo_id: str, variant: Optional[str], expected_bytes: int, hf_token: Optional[str]
) -> Optional[float]:
    """0-100, or None. The hub service reports a 0-1 fraction, so scale it."""
    from hub.services.models import downloads
    try:
        payload = await downloads.get_gguf_download_progress_response(
            repo_id, variant or "", expected_bytes, hf_token
        )
        fraction = payload.get("progress")
        if not isinstance(fraction, (int, float)):
            return None
        return min(100.0, max(0.0, float(fraction) * 100.0))
    except Exception:
        return None


def _release(repo_id: str) -> None:
    global _active
    with _lock:
        if _active is not None and _active.repo_id == repo_id:
            _active = None


async def _watch(active: _Active, hf_token: Optional[str]) -> None:
    """Poll a dispatched job so the monitor row resolves and the resolver cache
    is dropped the moment the weights land."""
    from core.inference import api_monitor as monitor_module
    from core.inference.local_model_resolver import invalidate_index

    api_monitor = monitor_module.api_monitor
    deadline = time.monotonic() + _MAX_WATCH_S
    try:
        while time.monotonic() < deadline:
            await asyncio.sleep(_WATCH_POLL_S)
            state, error = await _job_state(active.repo_id, active.variant)
            if state == "running":
                api_monitor.set_progress(
                    active.monitor_id,
                    await _progress_percent(
                        active.repo_id, active.variant, active.expected_bytes, hf_token
                    ),
                )
                continue
            if state == "complete":
                # Drop the 5s resolver cache so the next retry resolves the new
                # model instead of missing it again.
                await asyncio.to_thread(invalidate_index)
                api_monitor.finish(active.monitor_id, status = "completed")
            elif state == "idle":
                # The job vanished without a terminal state (worker killed).
                api_monitor.fail_open(active.monitor_id, "Download did not complete")
            else:
                api_monitor.fail_open(active.monitor_id, error or f"Download {state}")
            return
        api_monitor.fail_open(active.monitor_id, "Download timed out")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("auto-download: watcher failed for %r: %s", active.repo_id, exc)
        api_monitor.fail_open(active.monitor_id, "Download tracking failed")
    finally:
        _release(active.repo_id)


def _downloading_refusal(label: str, percent: Optional[float]) -> AutoDownloadRefusal:
    progress = f" ({percent:.0f}% done)" if percent is not None else ""
    return AutoDownloadRefusal(
        status = 503,
        code = "model_downloading",
        message = (f"Downloading '{label}'{progress}. Retry shortly. Track it in Unsloth Studio."),
        retry_after = _RETRY_AFTER_S,
    )


async def maybe_auto_download(
    requested_model: str, *, hf_token: Optional[str] = None
) -> Optional[AutoDownloadRefusal]:
    """Start (or report on) a background fetch of *requested_model*.

    Returns None when the request should carry on unchanged, or a refusal the
    caller must raise. Only called after the local resolver has already missed.
    """
    global _active

    repo_id, wanted_variant = split_model_ref(requested_model)
    if not is_downloadable_ref(requested_model):
        return None

    # Adopt or reject against the single in-flight slot before touching the
    # network, so retries during a long download stay free.
    with _lock:
        current = _active
        if current is not None and current.repo_id == repo_id:
            adopted = current
        elif current is not None:
            return AutoDownloadRefusal(
                status = 503,
                code = "model_download_busy",
                message = (
                    f"Already downloading '{_public_label(current.repo_id, current.variant)}'. "
                    f"Retry '{requested_model}' once it finishes."
                ),
                retry_after = _RETRY_AFTER_S,
            )
        else:
            adopted = None
            _active = _Active(repo_id = repo_id, started_at = time.time())

    if adopted is not None:
        state, error = await _job_state(adopted.repo_id, adopted.variant)
        if state in ("running", "cancelling"):
            return _downloading_refusal(
                _public_label(adopted.repo_id, adopted.variant),
                await _progress_percent(
                    adopted.repo_id, adopted.variant, adopted.expected_bytes, hf_token
                ),
            )
        if state == "error":
            # Surface once, then free the slot so a retry can start over.
            _release(adopted.repo_id)
            return AutoDownloadRefusal(
                status = 502,
                code = "model_download_failed",
                message = f"Downloading '{requested_model}' failed: {error or 'unknown error'}",
            )
        if adopted.variant is None:
            # Another request is still probing this same repo.
            return _downloading_refusal(adopted.repo_id, None)
        # complete/idle: the watcher is about to release; ask for one more retry.
        return _downloading_refusal(_public_label(adopted.repo_id, adopted.variant), 100.0)

    try:
        return await _admit_and_start(repo_id, wanted_variant, requested_model, hf_token)
    except Exception:
        _release(repo_id)
        raise


async def _admit_and_start(
    repo_id: str, wanted_variant: Optional[str], requested_model: str, hf_token: Optional[str]
) -> Optional[AutoDownloadRefusal]:
    from hub.utils.hf_errors import hf_error_status

    def _probe():
        from huggingface_hub import HfApi
        return HfApi(token = hf_token).model_info(
            repo_id, files_metadata = True, timeout = _MODEL_INFO_TIMEOUT_S
        )

    try:
        info = await asyncio.to_thread(_probe)
    except Exception as exc:
        _release(repo_id)
        status = hf_error_status(exc)
        if status == 403:
            return AutoDownloadRefusal(
                status = 403,
                code = "model_access_denied",
                message = (
                    f"'{repo_id}' is gated on Hugging Face. Accept its licence and add an "
                    "access token in Unsloth Studio, then retry."
                ),
            )
        if status == 404:
            # A private repo reads as absent without a token; don't confirm either way.
            return AutoDownloadRefusal(
                status = 404,
                code = "model_not_found",
                message = f"'{repo_id}' was not found on Hugging Face, or is not accessible.",
            )
        logger.warning("auto-download: Hub lookup failed for %r: %s", repo_id, exc)
        return AutoDownloadRefusal(
            status = 503,
            code = "model_lookup_failed",
            message = f"Could not reach Hugging Face to look up '{repo_id}'. Retry shortly.",
            retry_after = _RETRY_AFTER_S,
        )

    variants = _gguf_variants(getattr(info, "siblings", None))
    if not variants:
        _release(repo_id)
        return AutoDownloadRefusal(
            status = 400,
            code = "model_not_supported",
            message = (
                f"'{repo_id}' has no GGUF weights. Automatic download serves GGUF only; "
                "load other formats from Unsloth Studio."
            ),
        )

    # trust_remote_code gate. _config_has_auto_map is tri-state: refuse on True
    # and on None (unreadable), rather than _requires_trust_remote_code_for_model,
    # which swallows errors into False. Fine as a UI hint, wrong as an admission.
    from utils.security.consent import _config_has_auto_map

    has_auto_map = await asyncio.to_thread(_config_has_auto_map, repo_id, hf_token)
    if has_auto_map is not False:
        _release(repo_id)
        unknown = has_auto_map is None
        return AutoDownloadRefusal(
            status = 403,
            code = "remote_code_consent_required",
            message = (
                f"'{repo_id}' "
                + (
                    "could not be checked for custom code"
                    if unknown
                    else "ships custom code that runs on load"
                )
                + ". Load it once in Unsloth Studio to review and approve it, then retry."
            ),
        )

    variant = _match_variant(wanted_variant, variants)
    if variant is None:
        _release(repo_id)
        listed = sorted(variants)
        shown = ", ".join(listed[:_MAX_LISTED_VARIANTS])
        extra = len(listed) - _MAX_LISTED_VARIANTS
        return AutoDownloadRefusal(
            status = 404,
            code = "model_not_found",
            message = (
                f"'{repo_id}' has no quant '{wanted_variant}'. Available quants: "
                f"{shown}{f' and {extra} more' if extra > 0 else ''}."
            ),
        )

    expected_bytes = variants[variant]
    fits, free = _enough_disk(expected_bytes)
    if not fits:
        _release(repo_id)
        return AutoDownloadRefusal(
            status = 507,
            code = "insufficient_disk_space",
            message = (
                f"'{_public_label(repo_id, variant)}' needs {_gb(expected_bytes)} plus "
                f"{_gb(_DISK_RESERVE_BYTES)} headroom, but only {_gb(free)} is free."
            ),
        )

    return await _dispatch(repo_id, variant, expected_bytes, requested_model, hf_token)


def _match_variant(wanted: Optional[str], variants: dict[str, int]) -> Optional[str]:
    """Resolve the requested quant against what the repo actually has.

    An explicit quant matches case-insensitively and must exist: never quietly
    substitute another, unlike the loader's low-disk fallback. A bare repo id
    uses the same preference order as a manual load.
    """
    if wanted:
        lowered = {name.lower(): name for name in variants}
        return lowered.get(wanted.strip().lower())
    from utils.models.model_config import _pick_best_gguf

    # _pick_best_gguf ranks filenames by quant substring, so give it synthesized
    # "<label>.gguf" names and take the label back off the winner.
    best = _pick_best_gguf([f"{name}.gguf" for name in variants])
    return best[: -len(".gguf")] if best else None


async def _dispatch(
    repo_id: str, variant: str, expected_bytes: int, requested_model: str, hf_token: Optional[str]
) -> AutoDownloadRefusal:
    global _active

    from core.inference.api_monitor import api_monitor
    from hub.schemas.downloads import DownloadModelRequest
    from hub.services.models import downloads

    label = _public_label(repo_id, variant)
    try:
        await downloads.download_model_response(
            DownloadModelRequest(repo_id = repo_id, gguf_variant = variant), hf_token
        )
    except Exception as exc:
        _release(repo_id)
        status = getattr(exc, "status_code", None)
        if status == 409:
            # A manual load or hub download already owns this repo.
            return AutoDownloadRefusal(
                status = 503,
                code = "model_download_busy",
                message = f"'{repo_id}' is already being downloaded or loaded. Retry shortly.",
                retry_after = _RETRY_AFTER_S,
            )
        logger.warning("auto-download: could not start %r: %s", label, exc)
        return AutoDownloadRefusal(
            status = 502,
            code = "model_download_failed",
            message = f"Could not start downloading '{requested_model}'.",
        )

    monitor_id = api_monitor.record_lifecycle(
        event = "download", model = label, reason = "api", running = True
    )
    with _lock:
        if _active is not None and _active.repo_id == repo_id:
            _active.variant = variant
            _active.expected_bytes = expected_bytes
            _active.monitor_id = monitor_id
            tracked = _active
        else:  # released underneath us; still track the job we just started
            tracked = _Active(repo_id, variant, expected_bytes, monitor_id, time.time())
            _active = tracked

    asyncio.create_task(_watch(tracked, hf_token))
    logger.info("auto-download: started %s (%s)", label, _gb(expected_bytes))
    return AutoDownloadRefusal(
        status = 503,
        code = "model_downloading",
        message = (
            f"Downloading '{label}' ({_gb(expected_bytes)}). Retry shortly. "
            "Track it in Unsloth Studio."
        ),
        retry_after = _RETRY_AFTER_S,
    )


def reset_for_tests() -> None:
    global _active
    with _lock:
        _active = None
