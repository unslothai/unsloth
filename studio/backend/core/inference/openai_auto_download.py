# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in: fetch a GGUF a /v1 request names but this server doesn't have.

Auto-switch only loads models already on disk. With
``openai_api_auto_download_model`` on, a miss that looks like a real Hub repo is
fetched in the background and the request is told to retry rather than held
open: a quant is routinely tens of GB, far longer than any client (or the
Cloudflare edge on ``--secure``) will wait, and the inference lifecycle gate must
not be held meanwhile. The resident model keeps serving, and the retry that lands
after the download goes through the ordinary auto-switch path.

Admission is deliberately narrow, since a request only needs an API key:
- ``namespace/name`` only, and only when the Hub confirms GGUF weights. A
  namespace is not evidence of intent (LiteLLM and OpenRouter address every
  provider that way), so ``gpt-4`` and ``anthropic/claude-3.5-sonnet`` alike
  fall through to the resident model as before.
- GGUF only, decided from the remote file list, not the repo name: GGUF runs
  under llama.cpp, which never imports repo Python.
- ``auto_map`` is refused, so ``trust_remote_code`` is only ever granted
  deliberately in the UI, never by an API call.
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

# Keep the Hub probe short so a slow Hub can't stall the request path.
_MODEL_INFO_TIMEOUT_S = 8.0
# auth_check and hf_hub_download take no timeout of their own and run while the
# provisional slot is held, so an unresponsive Hub would pin the single flight. The
# code probe fetches up to three configs, so it gets more room than the auth call.
_CODE_PROBE_TIMEOUT_S = 20.0
# Headroom left free after the download, so filling the disk can't wedge the box.
_DISK_RESERVE_BYTES = 5 * 1024**3
_WATCH_POLL_S = 2.0
# A stalled watcher must not pin the single-flight slot forever.
_MAX_WATCH_S = 24 * 60 * 60
# Past the watch window the row is resolved, so poll only to see whether the
# worker is still alive and still owns the slot.
_TIMED_OUT_POLL_S = 60.0
_RETRY_AFTER_S = 30
# Long enough for a client honouring Retry-After to come back and be told, short
# enough that one that never returns cannot hold the slot.
_FAILED_HOLD_S = 3 * _RETRY_AFTER_S
_MAX_LISTED_VARIANTS = 8


@dataclass(frozen = True)
class AutoDownloadRefusal:
    """Why this request cannot be served yet; the route raises it in the
    surface's own error envelope."""

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
    # Set when the worker failed. Held until a retry surfaces it: Retry-After is far
    # longer than the watcher poll, so the client would restart the same failing download.
    error: Optional[str] = None
    failed_at: float = 0.0


_lock = threading.Lock()
_active: Optional[_Active] = None

# Repos the Hub says are not servable, so a "vendor/model" miss doesn't re-probe every request.
_NOT_SERVABLE_TTL_S = 10 * 60
_NOT_SERVABLE_MAX = 256
_cache_lock = threading.Lock()
_not_servable: dict[str, float] = {}


def _public_label(repo_id: str, variant: Optional[str]) -> str:
    return f"{repo_id}:{variant}" if variant else repo_id


def split_model_ref(requested: str) -> tuple[str, Optional[str]]:
    """``org/repo:QUANT`` -> ``("org/repo", "QUANT")``; no suffix -> variant None.

    Splits on the last colon. A suffix bearing either separator is only a variant
    when a real Hub repo precedes it: "build/llama-13b" is a subdirectory GGUF key
    the catalog advertises, while "C:/models/x.gguf" -- and the native Windows
    "C:\\models\\x.gguf" -- leaves a drive letter that is no repo id.
    """
    text = (requested or "").strip()
    base, sep, suffix = text.rpartition(":")
    if not sep or not base or not suffix:
        return text, None
    stripped = base.strip()
    if "/" in suffix.replace("\\", "/"):
        from hub.utils.paths import is_valid_repo_id
        if "/" not in stripped or not is_valid_repo_id(stripped):
            return text, None
    return stripped, suffix.strip()


def is_downloadable_ref(requested: str) -> bool:
    """Whether *requested* is shaped like a Hub repo we may fetch.

    Requires an explicit namespace: keeps ``gpt-4`` and other foreign ids falling
    through, and stops ModelConfig.from_identifier's bare-name ``unsloth/``
    prefixing from turning an unrelated label into a real repo.
    """
    from hub.utils.paths import is_valid_repo_id

    repo_id, variant = split_model_ref(requested)
    if "/" not in repo_id or not is_valid_repo_id(repo_id):
        return False
    if variant is not None:
        from hub.utils.paths import is_valid_gguf_variant
        return is_valid_gguf_variant(variant)
    return True


def looks_like_gguf_hub_repo_id(repo_id: str) -> bool:
    """Whether *repo_id* names an Unsloth catalog entry, not a LiteLLM/OpenRouter label.

    Namespaced ids without a GGUF suffix are foreign routing labels (``openai/gpt-4o``).
    ``-GGUF`` and the ``unsloth/`` namespace mark ids clients pick from this server's
    model list (GGUF and Transformers-backed entries alike); a mistyped one must 404
    instead of being answered by another loaded model.
    """
    text = (repo_id or "").strip()
    if "/" not in text:
        return False
    from hub.utils.paths import is_valid_repo_id

    if not is_valid_repo_id(text):
        return False
    name = text.rsplit("/", 1)[-1]
    if name.lower().endswith("-gguf"):
        return True
    return text.lower().startswith("unsloth/")


def looks_like_quant(variant: Optional[str]) -> bool:
    """Whether a ``:suffix`` names a GGUF quant rather than a foreign tag.

    Neither a namespace nor a colon proves a request was meant for this server
    (``vendor/model`` is LiteLLM/OpenRouter, ``name:latest`` is Ollama). A real
    quant label does.
    """
    import re

    from hub.utils.gguf import is_h3_denoiser_variant_key
    from utils.models.model_config import _GGUF_KNOWN_QUANT_RE

    if not variant:
        return False
    # _extract_quant_label can append a bpw modifier (IQ4_XS-3.53bpw); still a quant.
    label = re.sub(r"-[0-9]+(?:\.[0-9]+)?bpw$", "", variant.strip(), flags = re.IGNORECASE)
    # A qualified key is one of OUR advertised rows: a path (``distilled/model-Q6_K``) or an H3
    # root stem (``minimax_h3_ref2va_pruned-Q6_K``). Explicit, so it must MISS when absent;
    # falling through served the caller a different checkpoint under the requested id.
    normalized = label.replace("\\", "/")
    if "/" in normalized or is_h3_denoiser_variant_key(normalized):
        return True
    return _GGUF_KNOWN_QUANT_RE.fullmatch(label) is not None


def _hub_token(hf_token: Optional[str]):
    """The caller's token, or an explicit False. None makes huggingface_hub fall
    back to a cached login (here the server owner's); only False is anonymous."""
    return hf_token or False


def _servable_key(repo_id: str, hf_token: Optional[str]) -> str:
    """Cache key, per credential.

    The Hub 404s a private repo the caller cannot see, so a tokenless verdict says
    nothing about a caller who has one. Digested, so no token is held here.
    """
    import hashlib

    seen_as = hashlib.sha256(hf_token.encode()).hexdigest()[:16] if hf_token else "anon"
    return f"{repo_id.lower()}\n{seen_as}"


def _mark_not_servable(repo_id: str, hf_token: Optional[str]) -> None:
    with _cache_lock:
        if len(_not_servable) >= _NOT_SERVABLE_MAX:
            _not_servable.clear()
        _not_servable[_servable_key(repo_id, hf_token)] = time.monotonic() + _NOT_SERVABLE_TTL_S


def _is_not_servable(repo_id: str, hf_token: Optional[str]) -> bool:
    key = _servable_key(repo_id, hf_token)
    with _cache_lock:
        expires = _not_servable.get(key)
        if expires is None:
            return False
        if expires <= time.monotonic():
            del _not_servable[key]
            return False
        return True


def _gated_refusal(repo_id: str) -> AutoDownloadRefusal:
    return AutoDownloadRefusal(
        status = 403,
        code = "model_access_denied",
        message = (
            f"'{repo_id}' is gated on Hugging Face. Accept its licence, then retry with "
            "your own token in the X-Unsloth-HF-Token header: automatic download never "
            "uses this server's Hugging Face identity."
        ),
    )


async def _bounded_probe(fn, *args, timeout: float, default):
    """Run a blocking Hub probe off the loop, bounding only the wait.

    The thread is left to finish (a blocking socket read cannot be cancelled); the
    caller takes *default*, chosen per call site so a timeout errs the safe way.
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout)
    except (TimeoutError, asyncio.TimeoutError):
        logger.debug("hub probe %s timed out after %ss", getattr(fn, "__name__", fn), timeout)
        return default


def _auth_denied(repo_id: str, hf_token: Optional[str]) -> bool:
    """Whether this token lacks file access to a gated repo. False when the
    check is inconclusive: the download's own auth is the real gate."""
    from hub.utils.hf_errors import hf_error_status

    try:
        from huggingface_hub import auth_check
        auth_check(repo_id, token = _hub_token(hf_token))
    except Exception as exc:
        return hf_error_status(exc) in (401, 403)
    return False


def _gguf_variants(siblings, repo_id: str = "") -> dict[str, int]:
    """Quant label -> bytes the download will actually fetch.

    Mirrors list_gguf_variants for the selectable labels: companions (mmproj/MTP),
    calibration imatrices and big-endian builds are not quants, and sharded quants sum across shards.
    Bytes come from the download plan, which folds companions back into every
    quant, so the disk reserve is measured against what the worker fetches.
    """
    from hub.utils.gguf import extract_quant_label as canonical_quant_label
    from hub.utils.gguf import _is_selectable_repo_gguf, gguf_variant_key
    from hub.utils.gguf_plan import build_gguf_variant_plans
    from utils.models.model_config import (
        _extract_quant_label,
        _is_big_endian_gguf_path,
        _is_imatrix_path,
        _is_mmproj,
        _is_mtp_drafter,
    )

    from hub.utils.gguf import drop_shadowed_appledouble_siblings

    # Plans and advertised variants are derived separately, so both are filtered here.
    siblings = [
        sibling
        for sibling in drop_shadowed_appledouble_siblings(list(siblings or []))
        if _is_selectable_repo_gguf(repo_id, getattr(sibling, "rfilename", "") or "")
    ]
    plans = build_gguf_variant_plans(siblings)
    sizes: dict[str, int] = {}
    for sibling in siblings:
        name = getattr(sibling, "rfilename", "") or ""
        if not name.lower().endswith(".gguf"):
            continue
        label = _extract_quant_label(name)
        # The identity the PLAN is keyed on. A repo holding several checkpoints at one quant
        # advertises a qualified key per checkpoint, and keying this map on the bare label left
        # every one of those rows a hard miss here: a 404 instead of the download.
        quant = gguf_variant_key(name)
        if not looks_like_quant(quant):
            # With no recognized quant token the extractors part ways: this one takes
            # the last hyphenated segment ("7b" of llama-7b) while the plan and worker
            # key the whole stem, so advertising ours dispatches an unresolvable variant.
            quant = canonical_quant_label(name) or quant
        # The endian test reads a quant TOKEN, so it gets the label, not the path-qualified key.
        if (
            _is_mmproj(name)
            or _is_mtp_drafter(name)
            or _is_imatrix_path(name)
            or _is_big_endian_gguf_path(name, label)
        ):
            continue
        plan = plans.get(quant.lower())
        if plan is not None:
            sizes[quant] = plan.download_size_bytes
        else:
            sizes[quant] = sizes.get(quant, 0) + int(getattr(sibling, "size", 0) or 0)
    return sizes


def _remaining_bytes(repo_id: str, plan, expected_bytes: int) -> int:
    """Bytes still to fetch: a resumed quant or a companion shared with another
    quant is already on disk, and charging for it can 507 a download that fits."""
    try:
        from hub.utils.download_registry import existing_blob_bytes

        hashes = frozenset(
            file.sha256 for file in getattr(plan, "expected_files", ()) or () if file.sha256
        )
        if not hashes:
            return expected_bytes
        return max(0, expected_bytes - existing_blob_bytes("model", repo_id, hashes))
    except Exception:
        return expected_bytes


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
        # "unknown", not "idle": idle ends the watch, and a failed probe proves nothing.
        logger.debug("auto-download: status probe failed for %r: %s", repo_id, exc)
        return "unknown", None


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


def _release(active: Optional[_Active]) -> None:
    """Free the single-flight slot, but only while *active* still owns it.

    Keying on ``repo_id`` alone let a stale operation clear a newer one: variant A
    errors, an adopting request frees the slot, a retry starts B, then A's watcher
    matches the repo and clears B, admitting a second download alongside it.
    """
    global _active
    if active is None:
        return
    with _lock:
        if _active is active:
            _active = None


async def _watch(active: _Active, hf_token: Optional[str]) -> None:
    """Poll a dispatched job so the monitor row resolves and the resolver cache
    is dropped the moment the weights land."""
    from core.inference import api_monitor as monitor_module

    api_monitor = monitor_module.api_monitor
    deadline = time.monotonic() + _MAX_WATCH_S
    timed_out = False
    try:
        while True:
            await asyncio.sleep(_TIMED_OUT_POLL_S if timed_out else _WATCH_POLL_S)
            state, error = await _job_state(active.repo_id, active.variant)
            if state in ("running", "cancelling", "unknown"):
                if timed_out:
                    # A running worker still owns the slot: releasing on the clock alone
                    # would admit a second multi-GB download beside it. "unknown" cannot
                    # confirm it is alive, so release then, or a broken probe wedges us.
                    if state == "unknown":
                        return
                    continue
                if time.monotonic() >= deadline:
                    api_monitor.fail_open(active.monitor_id, "Download timed out")
                    timed_out = True
                    continue
                # Only "running" has progress; the others are still in flight, so keep the slot.
                if state == "running":
                    api_monitor.set_progress(
                        active.monitor_id,
                        await _progress_percent(
                            active.repo_id, active.variant, active.expected_bytes, hf_token
                        ),
                    )
                continue
            if state == "cancelled":
                api_monitor.finish(active.monitor_id, status = "cancelled")
                return
            if state == "complete":
                # No invalidate here: finalize_worker_exit already dropped the cache and
                # warmed it; a second would mark that fresh scan stale and push a
                # synchronous rescan onto the client's retry.
                api_monitor.finish(active.monitor_id, status = "completed")
            elif state == "idle":
                # The job vanished without a terminal state (worker killed).
                api_monitor.fail_open(active.monitor_id, "Download did not complete")
            else:
                api_monitor.fail_open(active.monitor_id, error or f"Download {state}")
                # Keep the slot so the next retry is told it failed instead of
                # silently restarting the same download.
                active.error = error or f"Download {state}"
                active.failed_at = time.monotonic()
                return
            return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("auto-download: watcher failed for %r: %s", active.repo_id, exc)
        api_monitor.fail_open(active.monitor_id, "Download tracking failed")
    finally:
        if not active.failed_at:
            _release(active)


def _downloading_refusal(label: str, percent: Optional[float]) -> AutoDownloadRefusal:
    progress = f" ({percent:.0f}% done)" if percent is not None else ""
    return AutoDownloadRefusal(
        status = 503,
        code = "model_downloading",
        message = (f"Downloading '{label}'{progress}. Retry shortly. Track it in Unsloth Studio."),
        retry_after = _RETRY_AFTER_S,
    )


async def _is_downloadable_model(repo_id: str, hf_token: Optional[str]) -> bool:
    """Whether the Hub has this repo with GGUF weights we could fetch.

    Only asked while another download holds the slot, to tell a second download
    apart from an ordinary foreign label. Any failure answers False: refusing
    would strand normal traffic for the length of the download.
    """
    if _is_not_servable(repo_id, hf_token):
        return False

    def _probe():
        from huggingface_hub import HfApi
        return HfApi(token = _hub_token(hf_token)).model_info(repo_id, timeout = _MODEL_INFO_TIMEOUT_S)

    try:
        info = await asyncio.to_thread(_probe)
    except Exception:
        return False
    # The same filter admission uses, not a bare .gguf test: mmproj, MTP drafters and
    # big-endian builds are companions, not quants. Answering otherwise would hold an
    # ordinary foreign label at model_download_busy for an unrelated download.
    servable = bool(_gguf_variants(getattr(info, "siblings", None), repo_id))
    if not servable:
        _mark_not_servable(repo_id, hf_token)
    return servable


async def maybe_auto_download(
    requested_model: str,
    *,
    hf_token: Optional[str] = None,
    require_vision: bool = False,
    subject: Optional[str] = None,
    via_api_key: bool = False,
) -> Optional[AutoDownloadRefusal]:
    """Start (or report on) a background fetch of *requested_model*.

    Returns None when the request should carry on unchanged, or a refusal the
    caller must raise. Only called after the local resolver has already missed.

    ``require_vision`` refuses a target with no mmproj companion rather than spend
    gigabytes on weights that cannot answer the request; the local capability guard
    only ever sees an already-downloaded model.

    ``subject`` and ``via_api_key`` describe the caller for the monitor row this
    opens: the same /v1 endpoints serve Unsloth's own chat on a session JWT, so the
    download is not API-key traffic unless the request that asked for it was.
    """
    global _active

    repo_id, wanted_variant = split_model_ref(requested_model)
    if not is_downloadable_ref(requested_model):
        return None
    if _is_not_servable(repo_id, hf_token) and not looks_like_quant(wanted_variant):
        return None

    # Settle the single-flight slot before the network, so retries during a download stay cheap.
    busy: Optional[_Active] = None
    with _lock:
        current = _active
        if current is not None and current.failed_at:
            # A held failure only owns the slot until someone is told about it.
            if current.repo_id != repo_id and time.monotonic() - current.failed_at > _FAILED_HOLD_S:
                _active = current = None
        if current is not None and current.repo_id == repo_id:
            adopted = current
        elif current is not None:
            adopted = None
            busy = current
        else:
            adopted = None
            provisional = _Active(repo_id = repo_id, started_at = time.time())
            _active = provisional

    if busy is not None:
        # Refusing before the probe blocks ordinary drop-in traffic: a namespaced label
        # that is no downloadable GGUF repo (LiteLLM/OpenRouter style) would be told to
        # wait out a multi-hour download. Only a downloadable label is a 2nd download.
        if not await _is_downloadable_model(repo_id, hf_token):
            return None
        return AutoDownloadRefusal(
            status = 503,
            code = "model_download_busy",
            message = (
                f"Already downloading '{_public_label(busy.repo_id, busy.variant)}'. "
                f"Retry '{requested_model}' once it finishes."
            ),
            retry_after = _RETRY_AFTER_S,
        )

    if adopted is not None:
        if adopted.variant is None:
            # Still probing: no job yet, and a stale whole-repo error would free the probe's slot.
            return _downloading_refusal(adopted.repo_id, None)
        state, error = await _job_state(adopted.repo_id, adopted.variant)
        if state in ("running", "cancelling", "unknown"):
            return _downloading_refusal(
                _public_label(adopted.repo_id, adopted.variant),
                await _progress_percent(
                    adopted.repo_id, adopted.variant, adopted.expected_bytes, hf_token
                ),
            )
        if state == "error" or adopted.error:
            error = error or adopted.error
            # Surface once, then free the slot so a retry can start over.
            _release(adopted)
            return AutoDownloadRefusal(
                status = 502,
                code = "model_download_failed",
                message = f"Downloading '{requested_model}' failed: {error or 'unknown error'}",
            )
        # complete/idle/cancelled: the watcher is about to free the slot, so retry once more.
        return _downloading_refusal(
            _public_label(adopted.repo_id, adopted.variant),
            100.0 if state == "complete" else None,
        )

    try:
        return await _admit_and_start(
            repo_id,
            wanted_variant,
            requested_model,
            hf_token,
            provisional,
            require_vision,
            subject = subject,
            via_api_key = via_api_key,
        )
    except BaseException:
        # Not `except Exception`: a cancel mid-probe would otherwise wedge the provisional slot.
        _release(provisional)
        raise


async def _admit_and_start(
    repo_id: str,
    wanted_variant: Optional[str],
    requested_model: str,
    hf_token: Optional[str],
    active: _Active,
    require_vision: bool = False,
    *,
    subject: Optional[str] = None,
    via_api_key: bool = False,
) -> Optional[AutoDownloadRefusal]:
    from hub.utils.hf_errors import hf_error_status

    def _probe():
        from huggingface_hub import HfApi
        return HfApi(token = _hub_token(hf_token)).model_info(
            repo_id, files_metadata = True, timeout = _MODEL_INFO_TIMEOUT_S
        )

    try:
        info = await asyncio.to_thread(_probe)
    except Exception as exc:
        _release(active)
        status = hf_error_status(exc)
        if status == 401:
            return AutoDownloadRefusal(
                status = 401,
                code = "model_access_denied",
                message = (
                    f"Hugging Face rejected the token sent for '{repo_id}'. Replace the "
                    "X-Unsloth-HF-Token header with a valid token; retrying will not help."
                ),
            )
        if status == 403:
            return _gated_refusal(repo_id)
        if status == 404:
            _mark_not_servable(repo_id, hf_token)
            # Unknown to the Hub reads as a foreign label; only an explicit quant makes it ours.
            if not looks_like_quant(wanted_variant):
                return None
            # A private repo reads as absent without a token; don't confirm either way.
            return AutoDownloadRefusal(
                status = 404,
                code = "model_not_found",
                message = (
                    f"'{repo_id}' was not found on Hugging Face, or is not accessible. "
                    "If it is private, send a token in the X-Unsloth-HF-Token header."
                ),
            )
        logger.warning("auto-download: Hub lookup failed for %r: %s", repo_id, exc)
        return AutoDownloadRefusal(
            status = 503,
            code = "model_lookup_failed",
            message = f"Could not reach Hugging Face to look up '{repo_id}'. Retry shortly.",
            retry_after = _RETRY_AFTER_S,
        )

    # Inconclusive on timeout: the download's own auth is the real gate.
    if getattr(info, "gated", False) and await _bounded_probe(
        _auth_denied, repo_id, hf_token, timeout = _MODEL_INFO_TIMEOUT_S, default = False
    ):
        # Metadata for a gated repo is not file access; unchecked, the config read below lies.
        _release(active)
        return _gated_refusal(repo_id)

    variants = _gguf_variants(getattr(info, "siblings", None), repo_id)
    if not variants:
        _release(active)
        _mark_not_servable(repo_id, hf_token)
        if not looks_like_quant(wanted_variant):
            return None
        return AutoDownloadRefusal(
            status = 400,
            code = "model_not_supported",
            message = (
                f"'{repo_id}' has no GGUF weights. Automatic download serves GGUF only; "
                "load other formats from Unsloth Studio."
            ),
        )

    # trust_remote_code gate: _config_has_auto_map is tri-state, so refuse on True and on None.
    from utils.security.consent import _config_has_auto_map

    # _hub_token, not the raw token: None lets huggingface_hub fall back to a cached
    # server login, so a caller-named repo would be probed with this server's identity.
    # Defaults to None on timeout, which refuses: unchecked is not cleared.
    has_auto_map = await _bounded_probe(
        _config_has_auto_map,
        repo_id,
        _hub_token(hf_token),
        timeout = _CODE_PROBE_TIMEOUT_S,
        default = None,
    )
    if has_auto_map is not False:
        _release(active)
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
        _release(active)
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
    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    plan = plan_for_variant(
        build_gguf_variant_plans(list(getattr(info, "siblings", None) or [])), variant
    )
    if require_vision and not (plan and plan.mmproj_filenames):
        _release(active)
        return AutoDownloadRefusal(
            status = 400,
            code = "invalid_value",
            message = (
                f"'{_public_label(repo_id, variant)}' ships no mmproj companion, so it "
                "cannot answer the image or audio input in this request. It was not "
                "downloaded."
            ),
        )

    need_bytes = _remaining_bytes(repo_id, plan, expected_bytes)
    fits, free = _enough_disk(need_bytes)
    if not fits:
        _release(active)
        return AutoDownloadRefusal(
            status = 507,
            code = "insufficient_disk_space",
            message = (
                f"'{_public_label(repo_id, variant)}' needs {_gb(need_bytes)} plus "
                f"{_gb(_DISK_RESERVE_BYTES)} headroom, but only {_gb(free)} is free."
            ),
        )

    return await _dispatch(
        repo_id,
        variant,
        expected_bytes,
        requested_model,
        hf_token,
        active,
        subject = subject,
        via_api_key = via_api_key,
    )


def preferred_quant(labels) -> Optional[str]:
    """The quant a plain load would pick from *labels*, or None.

    The one ranking for "which quant did they mean": local resolution, remote
    admission and /v1/models must agree, or a bare id means a different quant
    depending on which of them answered it.
    """
    from utils.models.model_config import _pick_best_gguf

    # _pick_best_gguf ranks filenames and matches upper-case tokens, so feed "<LABEL>.gguf".
    synthetic: dict[str, str] = {}
    for name in labels:
        synthetic.setdefault(f"{name.upper()}.gguf", name)
    best = _pick_best_gguf(list(synthetic))
    return synthetic.get(best) if best else None


def _bare_quant_alias(wanted: str, lowered: dict[str, str]) -> Optional[str]:
    """The one qualified variant whose quant token is *wanted*, or None when it names 0 or 2+.

    A key is a pure function of the path, so a repo that files every quant under one shared
    container qualifies all of them even though the directory disambiguates nothing, and the bare
    spelling every stored id uses then matches no key at all.
    """
    from hub.utils.gguf import bare_quant_alias

    target = (wanted or "").strip().lower()
    if not target:
        return None
    # PATH-qualified keys only, not is_qualified_gguf_variant_key: an H3 root stem's bare quant
    # names both partitions, so it must miss rather than serve one of them.
    matches = [
        name
        for key, name in lowered.items()
        if "/" in key and bare_quant_alias(key).lower() == target
    ]
    return matches[0] if len(matches) == 1 else None


def _match_variant(wanted: Optional[str], variants: dict[str, int]) -> Optional[str]:
    """Resolve the requested quant against what the repo actually has.

    An explicit quant matches case-insensitively and must exist: never quietly
    substitute another, unlike the loader's low-disk fallback. A bare repo id, or a
    tag that names no quant (":latest", ":8b"), uses the same preference order as a
    manual load, matching what the local resolver does with the same tag.
    """
    if wanted:
        # Exact first, whatever shape: a repo of generically named GGUFs has real
        # variants like "llama-13b" that are valid worker keys but not quant-shaped,
        # and defaulting past one would fetch a model nobody asked for.
        lowered = {name.lower(): name for name in variants}
        exact = lowered.get(wanted.strip().lower())
        if exact is None:
            # Same bare-quant fallback the plan lookup makes, and it has to be made HERE too:
            # admission rejects against this map first, so a repo that files every quant under
            # one shared container answered a legacy org/repo:Q4_K_M with a 404 and the worker's
            # fallback was never reached. Unambiguous only, for the same reason.
            exact = _bare_quant_alias(wanted, lowered)
        if exact is not None or looks_like_quant(wanted):
            # A quant-shaped suffix that matches nothing is a miss, never a swap.
            return exact
    # A BARE org/repo means the ROOT checkpoint, so a qualified sibling must not be ranked
    # against it: preferred_quant is order-sensitive, and once the map carried a key per
    # checkpoint a repo with distilled/model-Q6_K beside model-Q6_K could serve the sibling for
    # a bare id -- the same id that resolves to the root locally. Same filter
    # local_model_resolver._local_gguf_entry applies, so both resolvers answer one id one way.
    # A repo with nothing at the root falls back to the whole set rather than refusing.
    unqualified = {name: size for name, size in variants.items() if "/" not in name}
    return preferred_quant(unqualified or variants)


async def _dispatch(
    repo_id: str,
    variant: str,
    expected_bytes: int,
    requested_model: str,
    hf_token: Optional[str],
    active: _Active,
    *,
    subject: Optional[str] = None,
    via_api_key: bool = False,
) -> AutoDownloadRefusal:
    global _active

    from core.inference.api_monitor import api_monitor
    from hub.schemas.downloads import DownloadModelRequest
    from hub.services.models import downloads

    label = _public_label(repo_id, variant)
    busy = AutoDownloadRefusal(
        status = 503,
        code = "model_download_busy",
        message = f"'{repo_id}' is already being downloaded or loaded. Retry shortly.",
        retry_after = _RETRY_AFTER_S,
    )
    try:
        dispatched = await downloads.download_model_response(
            DownloadModelRequest(repo_id = repo_id, gguf_variant = variant),
            hf_token,
            allow_ambient_token = False,
        )
    except Exception as exc:
        _release(active)
        status = getattr(exc, "status_code", None)
        if status == 409:
            # A manual load or hub download already owns this repo.
            return busy
        logger.warning("auto-download: could not start %r: %s", label, exc)
        return AutoDownloadRefusal(
            status = 502,
            code = "model_download_failed",
            message = f"Could not start downloading '{requested_model}'.",
        )

    # accepted=False means no worker launched, so report the conflict instead of taking the slot.
    if isinstance(dispatched, dict) and not dispatched.get("accepted", True):
        _release(active)
        logger.info("auto-download: dispatch refused for %s (%s)", label, dispatched.get("state"))
        return busy

    monitor_id = api_monitor.record_lifecycle(
        # Reason "api" since only /v1 reaches auto-download, but that is not API-key
        # traffic: Unsloth's chat calls /v1 on a JWT, and marking its download would pop
        # the overlay mid-chat. So attribution comes from the request, plus its caller,
        # since the row is shared.
        event = "download",
        model = label,
        reason = "api",
        running = True,
        via_api_key = via_api_key,
        subject = subject,
    )
    with _lock:
        if _active is active:
            active.variant = variant
            active.expected_bytes = expected_bytes
            active.monitor_id = monitor_id
            tracked = active
        else:
            # Released underneath us: track the job we started, but never stomp a newer owner.
            tracked = _Active(repo_id, variant, expected_bytes, monitor_id, time.time())
            if _active is None:
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
    with _cache_lock:
        _not_servable.clear()
