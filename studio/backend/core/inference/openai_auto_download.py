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
from utils.account_context import current_account_id

logger = get_logger(__name__)

# Keep the Hub probe short so a slow Hub can't stall the request path.
_MODEL_INFO_TIMEOUT_S = 8.0
# Hub calls take no timeout and run holding the single-flight slot, so an unresponsive Hub would pin it; the code probe fetches up to three configs
_CODE_PROBE_TIMEOUT_S = 20.0
# Headroom left free after the download, so filling the disk can't wedge the box.
_DISK_RESERVE_BYTES = 5 * 1024**3
_WATCH_POLL_S = 2.0
# A stalled watcher must not pin the single-flight slot forever.
_MAX_WATCH_S = 24 * 60 * 60
# past the watch window the row is resolved, so poll only to see whether the worker is alive and still owns the slot
_TIMED_OUT_POLL_S = 60.0
_RETRY_AFTER_S = 30
# Long enough for a Retry-After client to return; short enough that no stuck one holds the slot.
_FAILED_HOLD_S = 3 * _RETRY_AFTER_S
_MAX_LISTED_VARIANTS = 8
# Probe the selected weight: speech GGUFs need not publish tokenizer sidecars.
_REMOTE_GGUF_SPEECH_PROBE_BYTES = 32 * 1024**2
_REMOTE_GGUF_SPEECH_PROBE_TIMEOUT_S = _CODE_PROBE_TIMEOUT_S - 2.0


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
    # Held until a retry surfaces it, else the client restarts the same failing download.
    error: Optional[str] = None
    failed_at: float = 0.0
    # Who asked: another account's busy answer names no repo or quant.
    account_id: Optional[str] = None


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
    """Splits on the last colon, but only a real Hub repo before it makes the suffix a variant."""
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
    """Needs an explicit namespace, so bare ids like gpt-4 never gain an unsloth/ prefix."""
    from hub.utils.paths import is_valid_repo_id

    repo_id, variant = split_model_ref(requested)
    if "/" not in repo_id or not is_valid_repo_id(repo_id):
        return False
    if variant is not None:
        from hub.utils.paths import is_valid_gguf_variant
        return is_valid_gguf_variant(variant)
    return True


def looks_like_gguf_hub_repo_id(repo_id: str) -> bool:
    """Only -GGUF or unsloth/ ids are catalog ids; a mistyped one must 404, not reach another model."""
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
    """A namespace or colon proves nothing about the caller's target; only a real quant label does."""
    import re

    from hub.utils.gguf import is_h3_denoiser_variant_key
    from utils.models.model_config import _GGUF_KNOWN_QUANT_RE

    if not variant:
        return False
    # _extract_quant_label can append a bpw modifier (IQ4_XS-3.53bpw); still a quant.
    label = re.sub(r"-[0-9]+(?:\.[0-9]+)?bpw$", "", variant.strip(), flags = re.IGNORECASE)
    # A qualified key must MISS when absent; falling through serves a different checkpoint.
    normalized = label.replace("\\", "/")
    if "/" in normalized or is_h3_denoiser_variant_key(normalized):
        return True
    return _GGUF_KNOWN_QUANT_RE.fullmatch(label) is not None


def _hub_token(hf_token: Optional[str]):
    """The caller's token, or an explicit False. None makes huggingface_hub fall
    back to a cached login (here the server owner's); only False is anonymous."""
    return hf_token or False


def _endpoint() -> str:
    from huggingface_hub import constants
    return constants.ENDPOINT.rstrip("/")


def _servable_key(repo_id: str, hf_token: Optional[str], endpoint: str) -> str:
    """Keyed per credential, since a tokenless 404 says nothing about a caller that has one; digested, so no token is held"""
    import hashlib

    seen_as = hashlib.sha256(hf_token.encode()).hexdigest()[:16] if hf_token else "anon"
    return f"{repo_id.lower()}\n{seen_as}\n{endpoint}"


def _mark_not_servable(repo_id: str, hf_token: Optional[str], endpoint: str) -> None:
    with _cache_lock:
        if len(_not_servable) >= _NOT_SERVABLE_MAX:
            _not_servable.clear()
        _not_servable[_servable_key(repo_id, hf_token, endpoint)] = (
            time.monotonic() + _NOT_SERVABLE_TTL_S
        )


def _is_not_servable(
    repo_id: str,
    hf_token: Optional[str],
    endpoint: Optional[str] = None,
) -> bool:
    key = _servable_key(repo_id, hf_token, endpoint or _endpoint())
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
    """Bounds only the wait: the thread keeps running, and the caller takes a default that fails safe."""
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


def _probe_remote_gguf_audio_type(
    repo_id: str, gguf_filename: str, hf_token: Optional[str], revision: Optional[str]
) -> tuple[Optional[str], bool]:
    try:
        from core.inference.diffusion_compat import _read_gguf_header
        from utils.models.gguf_metadata import classify_gguf_tts_audio_prefix

        prefix = _read_gguf_header(
            repo_id,
            gguf_filename,
            _hub_token(hf_token),
            revision = revision,
            max_bytes = _REMOTE_GGUF_SPEECH_PROBE_BYTES,
            timeout_seconds = _REMOTE_GGUF_SPEECH_PROBE_TIMEOUT_S,
        )
        return classify_gguf_tts_audio_prefix(prefix) if prefix else (None, False)
    except Exception as exc:
        logger.debug("remote GGUF speech probe failed for %s/%s: %s", repo_id, gguf_filename, exc)
        return None, False


def _gguf_variants(siblings, repo_id: str = "") -> dict[str, int]:
    """Sizes only selectable quants, with companions folded in, so the disk reserve matches the fetch."""
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
        # Keyed on the qualified key so multi-checkpoint repos resolve per checkpoint.
        quant = gguf_variant_key(name)
        if not looks_like_quant(quant):
            # Without a quant token the extractors differ; key the whole stem like the plan and worker.
            quant = canonical_quant_label(name) or quant
        # the endian test reads a quant TOKEN, so it gets the label, not the path-qualified key
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
    return f"{num_bytes / 1e9:.1f} GB"


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
    """Frees the slot only if active still owns it, since a stale watcher could clear a newer download."""
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
                    # A running worker owns the slot; "unknown" releases so a broken probe cannot wedge us.
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
                # No invalidate: finalize_worker_exit already refreshed the cache.
                api_monitor.finish(active.monitor_id, status = "completed")
            elif state == "idle":
                # The job vanished without a terminal state (worker killed).
                api_monitor.fail_open(active.monitor_id, "Download did not complete")
            else:
                api_monitor.fail_open(active.monitor_id, error or f"Download {state}")
                # keep the slot so the next retry is told it failed instead of silently restarting the same download
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
    """Any failure answers False, since refusing would strand ordinary traffic; one Hub for the whole lookup, so its answer is filed under that Hub"""
    endpoint = _endpoint()
    if _is_not_servable(repo_id, hf_token, endpoint):
        return False

    def _probe():
        from huggingface_hub import HfApi
        return HfApi(endpoint = endpoint, token = _hub_token(hf_token)).model_info(
            repo_id, timeout = _MODEL_INFO_TIMEOUT_S
        )

    try:
        info = await asyncio.to_thread(_probe)
    except Exception:
        return False
    # Same filter as admission, not a bare .gguf test: otherwise an ordinary foreign label is held at model_download_busy for an unrelated download
    servable = bool(_gguf_variants(getattr(info, "siblings", None), repo_id))
    if not servable:
        _mark_not_servable(repo_id, hf_token, endpoint)
    return servable


async def maybe_auto_download(
    requested_model: str,
    *,
    hf_token: Optional[str] = None,
    require_vision: bool = False,
    require_speech: bool = False,
    subject: Optional[str] = None,
    via_api_key: bool = False,
) -> Optional[AutoDownloadRefusal]:
    """Refuses incapable targets before any download; via_api_key marks API-key traffic, since /v1 also serves the Unsloth chat on a JWT"""
    global _active

    repo_id, wanted_variant = split_model_ref(requested_model)
    if not is_downloadable_ref(requested_model):
        return None
    if _is_not_servable(repo_id, hf_token) and not looks_like_quant(wanted_variant):
        return None

    # settle the single-flight slot before the network, so retries during a download stay cheap
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
            provisional = _Active(
                repo_id = repo_id, started_at = time.time(), account_id = current_account_id()
            )
            _active = provisional

    if busy is not None:
        # Probe first: non-GGUF namespaced labels (LiteLLM/OpenRouter style) must not be refused.
        if not await _is_downloadable_model(repo_id, hf_token):
            return None
        if busy.account_id == current_account_id():
            what = f"Already downloading '{_public_label(busy.repo_id, busy.variant)}'."
        else:
            what = "Another download is in progress."
        return AutoDownloadRefusal(
            status = 503,
            code = "model_download_busy",
            message = f"{what} Retry '{requested_model}' once it finishes.",
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
            require_speech,
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
    require_speech: bool = False,
    *,
    subject: Optional[str] = None,
    via_api_key: bool = False,
) -> Optional[AutoDownloadRefusal]:
    from hub.utils.hf_errors import hf_error_status, modelscope_missing

    endpoint = _endpoint()

    def _probe():
        from huggingface_hub import HfApi
        return HfApi(endpoint = endpoint, token = _hub_token(hf_token)).model_info(
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
            _mark_not_servable(repo_id, hf_token, endpoint)
            if not looks_like_quant(wanted_variant):
                return None
            return AutoDownloadRefusal(
                status = 404,
                code = "model_not_found",
                message = modelscope_missing(exc)
                or (
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
        _release(active)
        return _gated_refusal(repo_id)

    variants = _gguf_variants(getattr(info, "siblings", None), repo_id)
    if not variants:
        _release(active)
        _mark_not_servable(repo_id, hf_token, endpoint)
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

    # _hub_token, not the raw token: None falls back to the server cached login; a timeout defaults to None, which refuses, since unchecked is not cleared
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

    variant = _match_variant(wanted_variant, variants, repo_id)
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

    if require_speech:
        from functools import partial
        from utils.audio_tokens import GGUF_TTS_AUDIO_TYPES
        from utils.models.model_config import detect_audio_type_checked

        audio_type, definitive = await _bounded_probe(
            partial(
                detect_audio_type_checked,
                repo_id,
                hf_token = _hub_token(hf_token),
                revision = getattr(info, "sha", None),
            ),
            timeout = _CODE_PROBE_TIMEOUT_S,
            default = (None, False),
        )
        if definitive and (audio_type is None or audio_type in GGUF_TTS_AUDIO_TYPES):
            sidecar_audio_type = audio_type
            main_files = sorted(getattr(plan, "main_filenames", ()) or ())
            probed_audio_type, probed_definitive = await _bounded_probe(
                partial(
                    _probe_remote_gguf_audio_type,
                    repo_id,
                    main_files[0] if main_files else "",
                    hf_token,
                    getattr(info, "sha", None),
                ),
                timeout = _CODE_PROBE_TIMEOUT_S,
                default = (None, False),
            )
            if probed_definitive:
                audio_type, definitive = probed_audio_type, True
            elif sidecar_audio_type in GGUF_TTS_AUDIO_TYPES:
                audio_type, definitive = sidecar_audio_type, True
            else:
                audio_type, definitive = None, False
        if not definitive:
            _release(active)
            return AutoDownloadRefusal(
                status = 503,
                code = "model_lookup_failed",
                message = (
                    f"Could not verify that '{_public_label(repo_id, variant)}' supports "
                    "text-to-speech. It was not downloaded; retry shortly."
                ),
                retry_after = _RETRY_AFTER_S,
            )
        if audio_type not in GGUF_TTS_AUDIO_TYPES:
            _release(active)
            return AutoDownloadRefusal(
                status = 400,
                code = "invalid_value",
                message = (
                    f"'{_public_label(repo_id, variant)}' is not a supported "
                    "text-to-speech GGUF model. It was not downloaded."
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
    """The one ranking of which quant a bare id means, shared so every surface agrees."""
    from utils.models.model_config import _pick_best_gguf

    # _pick_best_gguf ranks filenames, so feed "<LABEL>.gguf"
    synthetic: dict[str, str] = {}
    for name in labels:
        synthetic.setdefault(f"{name}.gguf", name)
    best = _pick_best_gguf(list(synthetic))
    return synthetic.get(best) if best else None


def _bare_quant_alias(wanted: str, lowered: dict[str, str]) -> Optional[str]:
    """Returns the one qualified variant with that quant; a shared container matches several, so None."""
    from hub.utils.gguf import bare_quant_alias

    target = (wanted or "").strip().lower()
    if not target:
        return None
    # Path-qualified keys only, not is_qualified_gguf_variant_key: an H3 root stem with a bare quant is ambiguous and must miss
    matches = [
        name
        for key, name in lowered.items()
        if "/" in key and bare_quant_alias(key).lower() == target
    ]
    return matches[0] if len(matches) == 1 else None


def _match_variant(
    wanted: Optional[str],
    variants: dict[str, int],
    repo_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve the requested quant against what the repo actually has.

    An explicit quant matches case-insensitively and must exist: never quietly
    substitute another, unlike the loader's low-disk fallback. A bare repo id, or a
    tag that names no quant (":latest", ":8b"), uses the same preference order as a
    manual load, matching what the local resolver does with the same tag.
    """
    if wanted:
        # Exact match first: generic names like llama-13b are valid keys but not quant-shaped, and defaulting past one fetches the wrong model
        lowered = {name.lower(): name for name in variants}
        exact = lowered.get(wanted.strip().lower())
        if exact is None:
            # Same bare-quant fallback as the plan lookup, made here too since admission rejects against this map first; unambiguous only
            exact = _bare_quant_alias(wanted, lowered)
        if exact is not None or looks_like_quant(wanted):
            # A quant-shaped suffix that matches nothing is a miss, never a swap.
            return exact
    # A bare id means default_gguf_scope, the same scope every resolver applies (preferred_quant is order-sensitive).
    from hub.utils.gguf import default_gguf_scope

    return preferred_quant(default_gguf_scope(variants, repo_id))


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
            return busy
        logger.warning("auto-download: could not start %r: %s", label, exc)
        return AutoDownloadRefusal(
            status = 502,
            code = "model_download_failed",
            message = f"Could not start downloading '{requested_model}'.",
        )

    # accepted=False means no worker launched, so report the conflict instead of taking the slot
    if isinstance(dispatched, dict) and not dispatched.get("accepted", True):
        _release(active)
        logger.info("auto-download: dispatch refused for %s (%s)", label, dispatched.get("state"))
        return busy

    monitor_id = api_monitor.record_lifecycle(
        # Not API-key traffic: chat calls /v1 on a JWT, so attribution comes from the request.
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
            tracked = _Active(
                repo_id,
                variant,
                expected_bytes,
                monitor_id,
                time.time(),
                account_id = active.account_id,
            )
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
