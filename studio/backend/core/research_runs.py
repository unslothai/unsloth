# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small in-process supervisor for durable local Deep Research."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator, Callable

import httpx

from auth import storage as auth_storage
from core.inference.llama_admission import llama_admission_config_from_env
from core.inference.message_content import message_text_with_pastes
from core.inference.stream_errors import stream_error_from_chunk
from core.inference.tool_loop_controller import is_tool_error, strip_result_for_model
from core.inference.tools import EMPTY_SEARCH_RESULTS, RAG_SOURCES_SENTINEL, execute_tool
from core.inference.web_access_policy import check_url_access, website_policy_prompt
from core.research.parsing import (
    _MAX_PREVIEW_LABELS,
    _next_unused_seed_action,
    _normalize_research_state,
    _normalize_synthesis_audit,
    _parse_and_validate_action,
    _parse_and_validate_plan,
    _parse_json_object,
    _recover_report_from_reasoning,
    _report_after_boundary,
    _streamed_titles,
)
from core.research.citations import (
    _allowed_document_citations,
    _citation_title,
    _document_source_citation,
    _validate_report_document_sources,
    _validate_report_sources,
)
from core.research.redaction import _sanitize_public_query, _shield_untrusted
from core.research.prompts import (
    _AGENT_SYSTEM_PROMPT,
    _REPORT_BOUNDARY_MARKER,
    _REPORT_SYSTEM_PROMPT,
    _SYNTHESIS_AUDIT_SYSTEM_PROMPT,
    _planner_system_prompt,
    _system_prompt_with_instructions,
)
from loggers import get_logger
from storage import research_runs_db as db
from storage.studio_db import (
    get_chat_message,
    is_sqlite_busy_error,
    list_chat_messages,
    upsert_chat_message,
)

logger = get_logger(__name__)
_URL_BLOCK = re.compile(
    r"Title:\s*(?P<title>[^\n]*)\nURL:\s*(?P<url>https?://[^\s]+)\nSnippet:\s*(?P<snippet>.*?)(?=\n\n---|\Z)",
    re.DOTALL,
)
_WALL_CLOCK_TIMEOUT_CANCEL_MESSAGE = "research-wall-clock-timeout"
_MAX_ERROR_CHARS = 500
_MAX_CONTEXT_CHARS = 12_000
_MAX_CONTEXT_MESSAGE_CHARS = 4_000
_MAX_SYNTHESIS_EVIDENCE_CHARS = 32_000
# The synthesis prompt must fit the loaded context or it is silently truncated and the report
# degenerates into an echo of the evidence tail. The context box accepts anything from 128 up, so
# the budget adapts; unknown context keeps the full cap.
_MIN_SYNTHESIS_EVIDENCE_CHARS = 1_500
# Each section keeps a floor: overflow on a tiny context is recoverable, an empty prompt is not.
_MIN_QUESTION_CHARS = 800
_SYNTHESIS_EVIDENCE_CHARS_PER_TOKEN = 3.0
_SYNTHESIS_CONTEXT_RESERVE_TOKENS = 4_096
# Below this loaded context the prompt scaffolding alone fills the window, so grounding is skipped.
_AUTO_SCRAPE_MIN_CONTEXT_TOKENS = 8_192
# OFF by default (UNSLOTH_RESEARCH_AUTO_SCRAPE=1): benchmarking showed no reliable accuracy gain
# over snippets while adding latency, and it is safe only with the context gate in _research.
_AUTO_SCRAPE_TOP_K = 3
_AUTO_SCRAPE_TOTAL_CHARS = 6_000
_WEB_RAG_TOP_N = 6
_WEB_RAG_MIN_SCORE = 0.30
# routes.inference's 400 when nothing is loaded is transient, not a bad request.
_MODEL_WAIT_POLL_SECONDS = 2.0
# A model that keeps disappearing would re-send forever, so cap how many times one call may wait.
_MAX_MODEL_WAITS = 3
_NO_MODEL_LOADED_DETAIL = "No model loaded"
# routes.inference reports the same unloaded state this way when auto-switch finds no local match.
_MODEL_NOT_FOUND_CODE = "model_not_found"
# routes.inference 503s with this while an auto-switch to the run's model is still loading.
_MODEL_SWITCH_FAILED_CODE = "model_switch_failed"
# Used when the 503 carries no usable Retry-After, and as the step between switch retries.
_MODEL_SWITCH_RETRY_SECONDS = 5.0
# Long enough for a load already in flight; past that the refusal is the honest answer.
_NAMED_MODEL_WAIT_SECONDS = 60.0
# Transport keepalives prevent HTTP read timeouts without proving that a model is progressing.
_MODEL_FIRST_OUTPUT_TIMEOUT_SECONDS = 120.0
_MODEL_OUTPUT_IDLE_TIMEOUT_SECONDS = 120.0
# Cancellation is cooperative, so bound the unwind; a stuck iterator holds a timed-out call open for
# the rest of the wall clock.
_STREAM_CLEANUP_TIMEOUT_SECONDS = 5.0
# The SSE comment routes/inference.py sends while queued, not while the backend is silent.
_ADMISSION_WAIT_COMMENT = ": admission-wait"
_ADMISSION_DONE_COMMENT = ": admission-done"
# Queue notices arrive on the configured heartbeat, so allow for a few missed ones.
_ADMISSION_HEARTBEAT_MISSES = 3
# Also the ceiling for any budget, so the poll loop stays bounded however long generation itself runs.
_DEFAULT_MODEL_TIMEOUT_SECONDS = 900.0
_MAX_MODEL_WAIT_BUDGET_SECONDS = 3600.0
# Past the hourly windows providers reset on, short of parking a run and its lease on one mistaken header.
_MAX_RATE_LIMIT_WAIT_SECONDS = 3600.0
# Retry waits measure against this: a key that expires mid-backoff fails auth without reaching the provider.
_MODEL_CALL_KEY_LIFETIME_SECONDS = 2 * 60 * 60
# Headroom so the named stall guards expire before HTTPX's own read timeout does.
_STREAM_READ_TIMEOUT_MARGIN_SECONDS = 30.0


def _model_wait_budget(run: dict) -> float:
    """Share of the request budget one model wait may spend, clamped so an unlimited or
    oversized budget still leaves the poll loop bounded."""
    timeout = float(run["config"]["budgets"]["modelTimeoutSeconds"])
    capped = min(timeout or _DEFAULT_MODEL_TIMEOUT_SECONDS, _MAX_MODEL_WAIT_BUDGET_SECONDS)
    return capped / (_MAX_MODEL_WAITS + 1)


def _select_synthesis_report(content: str, reasoning: str) -> str:
    content_report = _report_after_boundary(content, _REPORT_BOUNDARY_MARKER)
    if content_report:
        return content_report
    reasoning_report = _report_after_boundary(reasoning, _REPORT_BOUNDARY_MARKER)
    if content_report == "":
        return reasoning_report or ""
    if content.strip():
        return content.strip()
    return reasoning_report or ""


def _synthesis_needs_recovery(report: str, finish_reason: str | None) -> bool:
    return finish_reason == "length" or not report


def _auto_scrape_default() -> int:
    """Server default for ``budgets["maxAutoScrape"]``: 0 (off) unless
    ``UNSLOTH_RESEARCH_AUTO_SCRAPE`` enables it (``1``/``true`` -> ``_AUTO_SCRAPE_TOP_K``, or an
    explicit count clamped to ``[0, _AUTO_SCRAPE_TOP_K]``)."""
    raw = os.environ.get("UNSLOTH_RESEARCH_AUTO_SCRAPE", "").strip().lower()
    if not raw:
        return 0
    if raw in ("0", "false", "no", "off"):
        return 0
    if raw in ("1", "true", "yes", "on"):
        return _AUTO_SCRAPE_TOP_K
    try:
        return max(0, min(int(raw), _AUTO_SCRAPE_TOP_K))
    except ValueError:
        return 0


# Nav menus, language sidebars and percent-encoded link lists are not evidence and derail retrieval.
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_LIST_PREFIX = re.compile(r"^(?:[\*\-\+•]|\d+[.)])\s")
_BLANK_RUN = re.compile(r"\n{3,}")
# Bare tracking URLs arrive as one unbroken token (prose never has an 80-char word) and a small
# model will latch onto and echo it.
_LONG_TOKEN = re.compile(r"\S{80,}")


def _clean_scraped_text(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if len(_PERCENT_ESCAPE.findall(stripped)) >= 4:
            continue
        if _LONG_TOKEN.search(stripped):
            continue
        prose = _MD_LINK.sub(r"\1", stripped).strip()
        if "](" in stripped and (
            _LIST_PREFIX.match(stripped) or len(prose) <= max(30, len(stripped) // 3)
        ):
            continue
        kept.append(line)
    return _BLANK_RUN.sub("\n\n", "\n".join(kept)).strip()


class RunCancelled(Exception):
    pass


class LeaseLost(Exception):
    pass


class ModelOutputIdleTimeout(httpx.ReadTimeout):
    # Default message: the stream reader raises the class the deadline names.
    def __init__(self, message: str = "Local model stopped producing output"):
        super().__init__(message)


class ModelFirstOutputTimeout(httpx.ReadTimeout):
    def __init__(self, message: str = "Local model never produced output"):
        super().__init__(message)


class ModelWallClockTimeout(httpx.ReadTimeout):
    pass


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, ModelFirstOutputTimeout):
        return "Local model never started producing output"
    if isinstance(exc, ModelOutputIdleTimeout):
        return "Local model stopped producing output before completion"
    if isinstance(exc, ModelWallClockTimeout):
        return "Local model request exhausted its total time budget"
    if isinstance(exc, httpx.TimeoutException):
        return "Local model request timed out"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"Local model request failed with HTTP {exc.response.status_code}"
    # str() must stay the server's own text so routes/inference.py's token-count regex still matches;
    # reading it here dropped the Model settings hint from an oversize refusal.
    friendly = getattr(exc, "friendly", None)
    text = friendly if isinstance(friendly, str) and friendly else str(exc)
    text = text.replace("\n", " ").strip()
    return (text or exc.__class__.__name__)[:_MAX_ERROR_CHARS]


def _extract_text(message: dict) -> str:
    return message_text_with_pastes(message).strip()


def _research_question_context(
    thread_id: str,
    user_message_id: str,
    override: str = "",
) -> tuple[str, str]:
    """The question to research plus the conversation that led to it.

    ``override`` is the question the model handed off, which folds in what the conversation
    established and is what the user actually wants researched. The raw message stands in for
    runs created without one.
    """
    messages = list_chat_messages(thread_id)
    by_id = {str(message["id"]): message for message in messages}
    user = by_id.get(user_message_id)
    question = override.strip() or _extract_text(user or {})
    if not user:
        return question, "[]"

    ancestors: list[dict] = []
    seen = {user_message_id}
    parent_id = user.get("parentId")
    while isinstance(parent_id, str) and parent_id and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            break
        ancestors.append(parent)
        parent_id = parent.get("parentId")
    ancestors.reverse()

    remaining = _MAX_CONTEXT_CHARS
    turns: list[dict[str, str]] = []
    for message in reversed(ancestors):
        text = _extract_text(message).strip()
        role = str(message.get("role") or "").strip()
        if not text or role not in {"user", "assistant"}:
            continue
        text = text[:_MAX_CONTEXT_MESSAGE_CHARS]
        if len(text) > remaining:
            text = text[:remaining]
        if not text:
            break
        turns.append({"role": role, "content": text})
        remaining -= len(text)
        if remaining <= 0:
            break
    turns.reverse()
    return question, json.dumps(turns, ensure_ascii = False)


def _positive_int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _peek_inference_backend() -> Any:
    """The orchestrator if one already exists, else None. Never constructs one.

    A resumed durable run probes on uvicorn's loop, and constructing reaches
    get_default_models() -> get_device(), so a cold probe would block the loop on the torch
    import just to answer "nothing is loaded". A patched core.inference getter still wins:
    that is the seam these probes are injected through.
    """
    from core.inference import get_inference_backend

    # Native / transformers: the orchestrator the API layer reads (not the subprocess singleton).
    try:
        from core.inference.orchestrator import get_inference_backend as _real
        from core.inference.orchestrator import peek_inference_backend
    except Exception:
        return get_inference_backend()
    return (
        get_inference_backend() if get_inference_backend is not _real else peek_inference_backend()
    )


def _loaded_context_length() -> int | None:
    """Best-effort read of the active model's context window in tokens, or None if unknown.

    Mirrors routes.inference._monitor_context_length (llama.cpp backend, else the inference
    orchestrator) so grounding sizes evidence to the same context the API layer serves. The ML
    backends live in a worker subprocess, so the core.inference.inference singleton is unpopulated
    here and importing it pulls in the ML stack; read the orchestrator the routes use instead."""
    try:
        from routes.inference import get_llama_cpp_backend
        llama = get_llama_cpp_backend()
        if getattr(llama, "is_loaded", False):
            ctx = _positive_int_or_none(getattr(llama, "context_length", None))
            if ctx is not None:
                return ctx
    except Exception:
        logger.debug("research.context_probe_llama_failed", exc_info = True)
    try:
        backend = _peek_inference_backend()
        name = getattr(backend, "active_model_name", None)
        models = getattr(backend, "models", {}) or {}
        info = models.get(name) if (name and isinstance(models, dict)) else None
        for candidate in (
            (info or {}).get("context_length"),
            getattr(backend, "context_length", None),
            getattr(backend, "max_seq_length", None),
        ):
            ctx = _positive_int_or_none(candidate)
            if ctx is not None:
                return ctx
    except Exception:
        logger.debug("research.context_probe_failed", exc_info = True)
    return None


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    """Conservative prompt token estimate for max_tokens clamping.

    Uses the same chars-per-token heuristic as synthesis evidence budgeting so
    Deep Research sizes output against the same context window it already probes.
    """
    chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        chars += len(text)
    return max(1, int(chars / _SYNTHESIS_EVIDENCE_CHARS_PER_TOKEN) + len(messages) * 4)


def _clamp_max_tokens_for_context(
    requested: int,
    messages: list[dict],
    *,
    context_length: int | None = None,
) -> int:
    ctx = context_length if context_length is not None else _loaded_context_length()
    if not ctx:
        return requested
    available = max(1, ctx - _estimate_prompt_tokens(messages))
    return max(1, min(requested, available))


def _resolve_max_tokens(
    max_tokens: int | None, inference: dict[str, Any], messages: list[dict]
) -> int:
    requested = int(max_tokens or inference.get("maxTokens") or 4096)
    ceiling = 16384 if max_tokens is not None else 8192
    capped = min(requested, ceiling)
    return _clamp_max_tokens_for_context(capped, messages)


def _normalize_completion_usage(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(key)
        if isinstance(value, (int, float)):
            usage[key] = int(value)
    return usage or None


def _completion_hit_context_wall(
    usage: dict[str, int] | None,
    *,
    requested_max_tokens: int,
    context_length: int | None = None,
) -> bool:
    if not usage:
        return False
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    ctx = context_length if context_length is not None else _loaded_context_length()
    if ctx is not None and total_tokens >= ctx:
        return True
    return completion_tokens < requested_max_tokens


def _synthesis_length_limit_error(
    usage: dict[str, int] | None, *, requested_max_tokens: int
) -> str:
    if _completion_hit_context_wall(usage, requested_max_tokens = requested_max_tokens):
        return (
            "Local model report hit the loaded context window before completion. "
            "Increase Context Length in chat settings or reduce the research evidence size."
        )
    return "Local model report reached its output limit before completion"


async def _model_unloaded(response: httpx.Response) -> str | None:
    """Which "not servable right now" refusal this is, or None for any other failure.

    All three are transient for a durable run -- the model can be loaded again -- unlike any
    other 4xx. ``"empty"`` is routes.inference's 400 for a backend with nothing loaded.
    ``"named"`` is its 404 model_not_found, which the same condition produces when auto-switch
    is on and the name resolves to nothing local: a model mid-load or mid-update looks exactly
    like a model that will never resolve, so the caller waits on it far more briefly.
    ``"switching"`` is its 503 model_switch_failed, raised while a swap to the run's model is
    still loading; the generic 5xx backoff gave up in three seconds, well inside a real load.
    """
    if response.status_code not in (400, 404, 503):
        return None
    try:
        body = await response.aread()
    except Exception:
        return None
    text = body.decode("utf-8", "replace")
    if response.status_code == 400:
        return "empty" if _NO_MODEL_LOADED_DETAIL in text else None
    if response.status_code == 503:
        return "switching" if _MODEL_SWITCH_FAILED_CODE in text else None
    return "named" if _MODEL_NOT_FOUND_CODE in text else None


def _retry_after_delay(raw: object) -> float | None:
    """A Retry-After value as a delay in seconds, or None when it names none or has passed.

    RFC 9110 defines the field as ``HTTP-date / delay-seconds``, and providers behind a CDN do
    send dates. Reading only the number backs off a second inside a cooldown with minutes left."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        delay = float(text)
    except ValueError:
        try:
            at = parsedate_to_datetime(text)
        except (IndexError, TypeError, ValueError):
            return None
        if at is None:
            return None
        if at.tzinfo is None:
            # RFC 9110 dates are GMT; a form that omits the zone is not a local time.
            at = at.replace(tzinfo = timezone.utc)
        delay = (at - datetime.now(timezone.utc)).total_seconds()
    return delay if delay > 0 else None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The response's Retry-After delay in seconds, or None when it is absent or already past."""
    return _retry_after_delay(response.headers.get("Retry-After"))


async def _peek_stream_head(lines: AsyncIterator[str]) -> str | None:
    """The stream's first line, or None when it ends without one.

    One line only: a queue notice belongs to the loop that refreshes the admission bound, and the
    refusal below is always a stream's first line."""
    async for line in lines:
        return line
    return None


def _stream_rate_limit_delay(head: str | None) -> float | None:
    """The delay a proxied provider rate-limit refusal asks for: None when the stream does not
    open with one, 0.0 when it names no delay.

    core.inference.external_provider turns an upstream non-200 into a 200 stream carrying one
    OpenAI-shaped error line, so a 429 survives only there: as ``code`` (``type`` for the ChatGPT
    connection) plus the forwarded Retry-After."""
    if head is None or not head.startswith("data:"):
        return None
    try:
        chunk = json.loads(head[5:].strip())
    except (TypeError, ValueError):
        return None
    error = chunk.get("error") if isinstance(chunk, dict) else None
    if not isinstance(error, dict):
        return None
    if str(error.get("code")) != "429" and error.get("type") != "rate_limit_error":
        return None
    metadata = error.get("metadata")
    if isinstance(metadata, dict) and metadata.get("terminal"):
        # Quota exhausted rather than throttled: no wait clears it, so surface it now.
        return None
    requested = error.get("retry_after")
    if requested is None and isinstance(metadata, dict):
        requested = metadata.get("retry_after")
    return _retry_after_delay(requested) or 0.0


async def _with_head(head: str | None, rest: AsyncIterator[str]) -> AsyncIterator[str]:
    """``rest`` with the line already read off it put back in front.

    Explicitly against None, not truthiness: a blank SSE separator line is a line."""
    if head is not None:
        yield head
    async for line in rest:
        yield line


def _rate_limit_wait(requested: float, remaining: float, headroom: float) -> float:
    """How much of a provider's requested retry delay this call can afford.

    The delay is the provider's, not a share of the model-load budget, so it is bounded by what is
    left of the call minus the room the re-send needs; coming back early only spends an attempt on
    the same refusal. The standing ceiling covers a run with no wall clock at all."""
    # Never reserve all of what is left: a call whose wall clock equals its first-output budget would
    # collapse every wait to zero.
    headroom = min(headroom, remaining / 2)
    return max(0.0, min(requested, _MAX_RATE_LIMIT_WAIT_SECONDS, remaining - headroom))


def _local_model_ready() -> bool:
    """Whether the local chat-completions path has a model to serve, using the same two checks
    routes.inference.openai_chat_completions makes before it 400s. Fails open when neither
    backend can be probed, so a probe failure can only run a request, never withhold one."""
    probed = False
    try:
        from routes.inference import get_llama_cpp_backend
        if getattr(get_llama_cpp_backend(), "is_loaded", False):
            return True
        probed = True
    except Exception:
        logger.debug("research.model_probe_llama_failed", exc_info = True)
    try:
        # No orchestrator yet is a real answer (nothing is loaded), not a failed probe.
        if getattr(_peek_inference_backend(), "active_model_name", None):
            return True
        probed = True
    except Exception:
        logger.debug("research.model_probe_failed", exc_info = True)
    return not probed


def _fit_source_catalog(catalog: str, max_chars: int) -> str:
    """Trim whole catalog entries from the tail so every surviving URL stays citable.

    Slicing mid-entry would hand the model a truncated URL, which the validator then strips.
    """
    if max_chars <= 0 or len(catalog) <= max_chars:
        return catalog if max_chars > 0 else ""
    kept: list[str] = []
    used = 0
    for entry in catalog.split("\n\n") if "\n\n" in catalog else catalog.splitlines(True):
        used += len(entry)
        if used > max_chars:
            break
        kept.append(entry)
    return ("".join(kept) if not kept or kept[0].endswith("\n") else "\n\n".join(kept)).rstrip()


def _fit_decision_inputs(
    question: str, plan: dict, system_chars: int, total_budget: int | None
) -> tuple[str, str]:
    """Fit the decision question and plan while keeping the plan valid JSON."""
    full_plan = json.dumps(plan, ensure_ascii = False)
    if total_budget is None:
        minimum_question_chars = min(len(question), _MIN_QUESTION_CHARS)
        research_reserve = 0
        plan_budget = len(full_plan)
    else:
        input_budget = max(0, total_budget - system_chars)
        if input_budget < len("{}"):
            raise ValueError("Loaded model context is too small for a research decision")
        minimum_question_chars = min(
            len(question),
            _MIN_QUESTION_CHARS,
            max(0, input_budget - len("{}")),
        )
        research_reserve = min(
            _MIN_SYNTHESIS_EVIDENCE_CHARS,
            max(0, input_budget - minimum_question_chars - len("{}")),
        )
        plan_budget = max(0, input_budget - minimum_question_chars - research_reserve)
    if len(full_plan) <= plan_budget:
        fitted_plan = full_plan
    else:
        fitted_plan = "{}"
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        for count in range(len(steps) + 1):
            candidate = json.dumps(
                {"title": plan.get("title") or "Research plan", "steps": steps[:count]},
                ensure_ascii = False,
            )
            if len(candidate) > plan_budget:
                break
            fitted_plan = candidate
    question_budget = _trimmable_budget(
        total_budget,
        system_chars + len(fitted_plan) + research_reserve,
        _MAX_SYNTHESIS_EVIDENCE_CHARS,
    )
    return question[:question_budget], fitted_plan


@asynccontextmanager
async def _wall_clock_timeout(seconds: float | None) -> AsyncIterator[None]:
    """Use asyncio.timeout when available, with the same behavior on Python 3.9/3.10."""
    if seconds is None:
        yield
        return
    timeout = getattr(asyncio, "timeout", None)
    if timeout is not None:
        async with timeout(seconds):
            yield
        return

    task = asyncio.current_task()
    if task is None:
        yield
        return
    expired = False

    def cancel() -> None:
        nonlocal expired
        expired = True
        task.cancel(_WALL_CLOCK_TIMEOUT_CANCEL_MESSAGE)

    handle = asyncio.get_running_loop().call_later(seconds, cancel)
    try:
        yield
    except asyncio.CancelledError as exc:
        if expired and exc.args == (_WALL_CLOCK_TIMEOUT_CANCEL_MESSAGE,):
            raise asyncio.TimeoutError from exc
        raise
    finally:
        handle.cancel()


def _prompt_char_budget(reserve_tokens: int) -> int | None:
    """Chars the whole prompt may occupy on the loaded context, or None when it is unknown.

    The output reserve is capped at half the window: a flat reserve at or above the context
    (4096 on the 4096-token GGUF floor) would leave a budget of 0 and empty the prompt, and a
    truncated completion is far better than one that never saw the question.
    """
    ctx = _loaded_context_length()
    if not ctx:
        return None
    reserve = min(reserve_tokens, max(1, ctx // 2))
    return int(max(0, ctx - reserve) * _SYNTHESIS_EVIDENCE_CHARS_PER_TOKEN)


def _trimmable_budget(total: int | None, fixed_chars: int, hard_cap: int) -> int:
    """Chars left for a trimmable section once the rest of the prompt is counted.

    Budgeting one section against the context while the others are unbounded does not stop an
    overflow: at a 2048-token context the untrimmable scaffolding alone is several times the
    window. Returns 0 rather than a floor, since a short report beats a failed run.
    """
    if total is None:
        return hard_cap
    return max(0, min(hard_cap, total - fixed_chars))


def _synthesis_evidence_budget(fixed_chars: int = 0) -> int:
    """Char budget for synthesis evidence (full cap when the context is unknown)."""
    return _trimmable_budget(
        _prompt_char_budget(_SYNTHESIS_CONTEXT_RESERVE_TOKENS),
        fixed_chars,
        _MAX_SYNTHESIS_EVIDENCE_CHARS,
    )


def _bounded_synthesis_evidence(
    notes: list[str], max_chars: int = _MAX_SYNTHESIS_EVIDENCE_CHARS
) -> str:
    if not notes:
        return "(none)"
    if max_chars <= 0:
        return ""
    # Split evenly across notes: a per-note floor would let the earliest notes consume the whole budget
    # and drop later steps entirely.
    separator = "\n\n"
    available = max(0, max_chars - len(separator) * (len(notes) - 1))
    base, remainder = divmod(available, len(notes))
    suffix = "\n[Evidence truncated]"
    bounded = []
    for index, note in enumerate(notes):
        limit = base + (1 if index < remainder else 0)
        if len(note) <= limit:
            bounded.append(note)
        elif limit <= len(suffix):
            bounded.append(note[:limit])
        else:
            bounded.append(note[: limit - len(suffix)].rstrip() + suffix)
    return separator.join(bounded)[:max_chars]


def _fit_synthesis_context(
    notes: list[str],
    prioritized_payloads: list[dict[str, Any]],
    fixed_chars: int = 0,
) -> tuple[str, list[str]]:
    """Share the adaptive synthesis budget between evidence and JSON prompt blocks.

    Payloads are considered in priority order. A payload that would consume the minimum evidence
    allocation is replaced with an empty object. This keeps every emitted block valid JSON while
    preventing model-derived state or an audit near its output cap from overflowing a small model
    context.
    """
    total_budget = _synthesis_evidence_budget(fixed_chars)
    placeholder = "{}"
    minimum_evidence = min(_MIN_SYNTHESIS_EVIDENCE_CHARS, total_budget)
    remaining_payload_budget = max(
        0,
        total_budget - minimum_evidence - len(placeholder) * len(prioritized_payloads),
    )
    serialized_payloads = []
    for payload in prioritized_payloads:
        candidate = json.dumps(payload, ensure_ascii = False) if payload else placeholder
        extra_chars = max(0, len(candidate) - len(placeholder))
        if extra_chars <= remaining_payload_budget:
            serialized_payloads.append(candidate)
            remaining_payload_budget -= extra_chars
        else:
            serialized_payloads.append(placeholder)
    evidence_budget = max(0, total_budget - sum(map(len, serialized_payloads)))
    return _bounded_synthesis_evidence(notes, evidence_budget), serialized_payloads


def _merge_scraped_evidence(raw_result: str, scraped_section: str) -> str:
    """Combine the raw search snippets with grounded page-body chunks (additive).

    Replacing ``raw_result`` with ``scraped_section`` regressed below snippet-only accuracy:
    when the retrieved chunk was a distractor the answer-bearing snippet was lost. Keep the
    snippets first and append the grounded excerpts. If either side is empty the other is
    returned unchanged.
    """
    raw = (raw_result or "").strip()
    scraped = (scraped_section or "").strip()
    if not scraped:
        return raw_result
    if not raw:
        return scraped_section
    return f"{raw}\n\nAdditional detail retrieved from the pages above:\n{scraped}"


def _split_rag_result(result: str) -> tuple[str, list[dict[str, Any]]]:
    if RAG_SOURCES_SENTINEL not in result:
        return result, []
    text, raw_sources = result.split(RAG_SOURCES_SENTINEL, 1)
    try:
        candidates = json.loads(raw_sources)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text.rstrip(), []
    if not isinstance(candidates, list):
        return text.rstrip(), []
    sources = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        sources.append(
            {
                "kind": "knowledge_base",
                "chunkId": candidate.get("chunkId"),
                "documentId": candidate.get("documentId"),
                "filename": str(candidate.get("filename") or "Document")[:500],
                "page": candidate.get("page"),
                "score": candidate.get("score"),
                "snippet": str(candidate.get("text") or "")[:2000],
            }
        )
    return text.rstrip(), sources


def _research_step_failed(web_result: str, rag_sources: list[dict]) -> bool:
    """A step that gathered no evidence failed, whether the tool errored or simply matched nothing.

    Reporting an empty search as completed hid the one outcome the user needs to see: the report
    was written without the evidence that step was supposed to supply.
    """
    if rag_sources:
        return False
    return is_tool_error(web_result) or web_result.strip() in EMPTY_SEARCH_RESULTS


def _run_moved_on(fresh: dict | None, attempt: int) -> bool:
    """Whether the run this worker was running has since been re-pointed at a newer question.

    A thread reuses its one run row for its lifetime, so between committing a terminal status
    and writing the terminal reply the user can stop the run and ask something else: the row
    is reset, its assistant binding moves, and the reply below -- resolved by run id -- would
    stamp "Research cancelled." and researchStatus cancelled onto the NEW question's
    placeholder, where it stays until that question reaches its own terminal write.

    retryCount is the attempt epoch, which rebind_cancelled advances for exactly this reason.
    """
    if not fresh:
        return True
    return int(fresh.get("retryCount") or 0) != attempt


def _update_assistant(
    run: dict,
    text: str,
    status: str,
    sources: list[dict] | None = None,
    reasoning: str = "",
    completion_worker_id: str | None = None,
) -> None:
    message_id = db.discover_and_bind_assistant_message(run["id"])
    if not message_id:
        if status not in db.TERMINAL_STATUSES:
            return
        message_id, _created = db.create_and_bind_terminal_fallback(
            run["id"],
            text = text,
            status = status,
            sources = sources,
            completion_worker_id = completion_worker_id,
            expected_attempt = int(run.get("retryCount") or 0),
        )
        if not message_id:
            return
    existing = get_chat_message(run["threadId"], message_id) or {}
    content = existing.get("content") if isinstance(existing.get("content"), list) else []
    # Only replace this worker's text/source parts; retain artifacts, reasoning, and other extensions.
    replaced_types = {"text", "source"}
    if reasoning:
        replaced_types.add("reasoning")
    retained = [
        part
        for part in content
        if not isinstance(part, dict)
        or part.get("type") not in replaced_types
        or part.get("researchRunId") not in (None, run["id"])
    ]
    if reasoning:
        retained.append({"type": "reasoning", "text": reasoning, "researchRunId": run["id"]})
    retained.append({"type": "text", "text": text, "researchRunId": run["id"]})
    for source in sources or []:
        retained.append(
            {
                "type": "source",
                "sourceType": "url",
                "id": source["url"],
                "url": source["url"],
                "title": source.get("title") or source["url"],
                "metadata": {"description": source.get("snippet") or ""},
                "researchRunId": run["id"],
            }
        )
    metadata = dict(existing.get("metadata") or {})
    metadata.update(
        {
            "researchRunId": run["id"],
            "researchStatus": status,
            "researchPlanRevision": run.get("planRevision", 0),
            "serverManaged": True,
        }
    )
    upsert_chat_message(
        {
            "id": message_id,
            "threadId": run["threadId"],
            "parentId": existing.get("parentId") or run["userMessageId"],
            "role": "assistant",
            "content": retained,
            "attachments": existing.get("attachments"),
            "metadata": metadata,
            "createdAt": existing.get("createdAt") or db.now_ms(),
        },
        allow_research_update = True,
        expected_research_run_id = run["id"],
        expected_research_attempt = int(run.get("retryCount") or 0),
    )


class ResearchSupervisor:
    def __init__(
        self,
        app: Any,
        poll_seconds: float = 0.5,
    ) -> None:
        self.app = app
        self.poll_seconds = poll_seconds
        self.worker_id = uuid.uuid4().hex
        self._stopping = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._lost_leases: set[str] = set()

    def start(self) -> None:
        db.recover_expired()
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name = "research-supervisor")

    async def stop(self) -> None:
        self._stopping.set()
        try:
            if self._task is not None:
                for cancel_event in self._cancel_events.values():
                    cancel_event.set()
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    # Polling is intentionally sufficient for one local process; requests never own tasks.
                    pass
        finally:
            await asyncio.to_thread(db.release_worker_leases, self.worker_id)

    def wake(self) -> None:
        pass

    def cancel(self, run_id: str) -> None:
        self._cancel_events.setdefault(run_id, threading.Event()).set()

    def _cancel_event(self, run_id: str) -> threading.Event:
        return self._cancel_events.setdefault(run_id, threading.Event())

    async def _check_active(self, run_id: str) -> None:
        if run_id in self._lost_leases:
            raise LeaseLost()
        cancelled, owns_lease = await asyncio.gather(
            asyncio.to_thread(db.is_cancel_requested, run_id),
            asyncio.to_thread(db.owns_lease, run_id, self.worker_id),
        )
        if cancelled:
            self.cancel(run_id)
            raise RunCancelled()
        if not owns_lease:
            raise LeaseLost()
        if self._cancel_event(run_id).is_set():
            raise RunCancelled()

    async def _auto_scrape_sources(
        self,
        run: dict,
        question: str,
        step_sources: list[dict],
        fetched_urls: set[str],
        *,
        limit: int,
        tool_timeout: int,
        website_policy: dict | None,
    ) -> tuple[str, list[str]]:
        """Concurrently read up to ``limit`` of this step's accepted source URLs and return the
        chunks most relevant to the question as ``<chunk>`` evidence, plus the URLs read.

        URLs are already access checked and deduplicated by the caller, so no new sources are
        created. Failures, timeouts, unreadable pages, and low-relevance chunks are dropped;
        the caller enforces cancellation."""
        cap = max(0, min(limit, _AUTO_SCRAPE_TOP_K))
        if cap <= 0:
            return "", []
        targets = []
        for source in step_sources:
            url = str(source.get("url") or "")
            if url and url not in fetched_urls:
                targets.append(source)
            if len(targets) >= cap:
                break
        if not targets:
            return "", []
        cancel_event = self._cancel_event(run["id"])
        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    execute_tool,
                    "web_search",
                    {"url": source["url"]},
                    cancel_event = cancel_event,
                    timeout = tool_timeout,
                    website_policy = website_policy,
                )
                for source in targets
            ),
            return_exceptions = True,
        )
        pages = []
        fetched = []
        for source, result in zip(targets, results):
            if isinstance(result, BaseException) or not isinstance(result, str):
                continue
            body = strip_result_for_model(result, "web_search")
            if is_tool_error(body):
                continue
            body = _clean_scraped_text(body)
            if not body:
                continue
            fetched.append(source["url"])
            pages.append(
                {
                    "text": body,
                    "title": source.get("title") or source["url"],
                    "url": source["url"],
                }
            )
        if not pages:
            return "", []
        # Runs off the event loop, since embedding and the sqlite/vec index work are CPU/GPU bound.
        from core.rag import web_rank

        section, _sources = await asyncio.to_thread(
            web_rank.retrieve_web_chunks,
            pages,
            question,
            top_n = _WEB_RAG_TOP_N,
            min_score = _WEB_RAG_MIN_SCORE,
            char_budget = _AUTO_SCRAPE_TOTAL_CHARS,
        )
        if not section:
            return "", []
        return (
            "Relevant passages retrieved from the top results (already read):\n\n" + section,
            fetched,
        )

    async def _check_worker_write(self, run_id: str, written: bool) -> None:
        if written:
            return
        await self._check_active(run_id)
        raise LeaseLost()

    async def _finish_after_lease_loss(self, run_id: str) -> str | None:
        while True:
            try:
                return await asyncio.to_thread(
                    db.finish,
                    run_id,
                    self.worker_id,
                    "failed",
                    "Worker lease expired",
                    None,
                    True,
                )
            except sqlite3.OperationalError:
                logger.warning(
                    "research.lease_loss_finish_retry run_id=%s",
                    run_id,
                    exc_info = True,
                )
                await asyncio.sleep(1)

    def note_server_port(self, server: Any) -> None:
        if isinstance(getattr(self.app.state, "server_port", None), int):
            return
        if (
            isinstance(server, tuple)
            and len(server) >= 2
            and isinstance(server[1], int)
            and server[1] > 0
        ):
            self.app.state.research_request_port = server[1]

    def note_request_port(self, request: Any) -> None:
        self.note_server_port(getattr(request, "scope", {}).get("server"))

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                if self._server_port() is None:
                    await asyncio.sleep(self.poll_seconds)
                    continue
                run = await asyncio.to_thread(db.claim_next, self.worker_id)
                if run is None:
                    await asyncio.sleep(self.poll_seconds)
                    continue
                await self._process(run)
            except asyncio.CancelledError:
                raise
            except sqlite3.OperationalError as exc:
                # Losing the writer lock is normal for polling, not a fault; neither branch may re-raise, since that
                # escapes the while loop and stops the supervisor for the life of the process.
                if is_sqlite_busy_error(exc):
                    logger.warning("research.supervisor_db_busy: %s", exc)
                else:
                    logger.exception("research.supervisor_iteration_failed")
                await asyncio.sleep(1)
            except Exception:
                logger.exception("research.supervisor_iteration_failed")
                await asyncio.sleep(1)

    def _server_port(self) -> int | None:
        port = getattr(self.app.state, "server_port", None)
        if not isinstance(port, int) or port <= 0:
            port = getattr(self.app.state, "research_request_port", None)
        if not isinstance(port, int) or port <= 0:
            return None
        return port

    def _endpoint(self) -> str:
        port = self._server_port()
        if port is None:
            raise RuntimeError("Research is waiting for the Unsloth server port")
        return f"http://127.0.0.1:{port}/v1/chat/completions"

    async def _wait_for_local_model(
        self,
        run: dict,
        max_seconds: float | None = None,
    ) -> bool:
        """Wait, up to the run's model timeout, for a model to be loaded again; True if one was.

        A durable run resumes after an Unsloth restart and is approved long after it was created,
        so the model it was started with can be gone. Waiting keeps the run alive instead of
        ending it on a non-retryable 400 that discards every step and source it gathered.

        ``max_seconds`` bounds the wait for refusals that name the model rather than report an
        empty backend. A load already in flight finishes inside it; anything else (an ejected
        model, a llama.cpp update, a name that no longer resolves) needs a user action that no
        wait can outlast, so surfacing the refusal beats burning the whole budget first."""
        loop = asyncio.get_running_loop()
        # Share the model budget across the allowed waits: spending it all on one lets the enclosing wall
        # clock fire first and bury the real refusal.
        budget = _model_wait_budget(run)
        if max_seconds is not None:
            budget = min(budget, max_seconds)
        deadline = loop.time() + budget
        logger.info("research.waiting_for_local_model run_id=%s", run["id"])
        while loop.time() < deadline:
            await self._check_active(run["id"])
            await asyncio.sleep(_MODEL_WAIT_POLL_SECONDS)
            if _local_model_ready():
                return True
        return False

    async def _wait_for_model_switch(self, run: dict, response: httpx.Response, waits: int) -> None:
        """Wait out an in-flight model switch before re-sending.

        A model is loaded, so ``_local_model_ready`` cannot tell this apart from success: only
        the next send can. Honour the server's Retry-After and lengthen the gap each time, since
        the swap it is waiting on loads a whole model.
        """
        run_id = run["id"]
        step = _retry_after_seconds(response) or _MODEL_SWITCH_RETRY_SECONDS
        # Same budget share as _wait_for_local_model: one wait must leave room for the others and for the
        # refusal, or the enclosing wall clock fires first and reports a timeout instead.
        remaining = min(step * waits, _NAMED_MODEL_WAIT_SECONDS, _model_wait_budget(run))
        logger.info("research.waiting_for_model_switch run_id=%s seconds=%.0f", run_id, remaining)
        while remaining > 0:
            await self._check_active(run_id)
            await asyncio.sleep(min(_MODEL_WAIT_POLL_SECONDS, remaining))
            remaining -= _MODEL_WAIT_POLL_SECONDS
        await self._check_active(run_id)

    async def _wait_out_rate_limit(
        self, run: dict, requested: float, deadline: float, headroom: float
    ) -> None:
        """Wait out a provider's retry delay, whatever carried it, against the same budget."""
        remaining = deadline - asyncio.get_running_loop().time()
        await self._wait_out_retry_after(
            run["id"], _rate_limit_wait(requested, remaining, headroom)
        )

    async def _wait_out_retry_after(self, run_id: str, delay: float) -> None:
        """Wait out a provider-set retry delay in the same poll-sized slices as the model waits
        above, so a cancel or a lost lease ends the run during the wait rather than after it."""
        remaining = delay
        while remaining > 0:
            await self._check_active(run_id)
            await asyncio.sleep(min(_MODEL_WAIT_POLL_SECONDS, remaining))
            remaining -= _MODEL_WAIT_POLL_SECONDS

    @staticmethod
    def _absorb_late_task(run_id: str, what: str, task: asyncio.Task) -> None:
        """Retrieve the outcome of a task that outlived the cleanup bound."""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning(
                "research.%s_late_cleanup_failed run_id=%s", what, run_id, exc_info = error
            )

    def _absorb_when_done(self, run_id: str, task: asyncio.Task, what: str) -> None:
        """Arrange for a task still running past cleanup to have its outcome retrieved."""
        if task.done():
            self._absorb_late_task(run_id, what, task)
            return
        task.add_done_callback(lambda finished: self._absorb_late_task(run_id, what, finished))

    async def _discard_task(self, run_id: str, task: asyncio.Task, what: str) -> None:
        """Cancel a pending task and absorb its outcome, without waiting forever.

        Awaiting it keeps a late error from surfacing as an unretrieved task exception;
        bounding the wait keeps an iterator that declines cancellation from pinning the
        caller here, and swallowing only its own outcome keeps the real error intact.
        """
        task.cancel()
        try:
            await asyncio.wait({task}, timeout = _STREAM_CLEANUP_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            # Must keep propagating, but the child outlives this frame, so hand it over first.
            # Bound expired but the task lives on: absorb its outcome when it cooperates.
            self._absorb_when_done(run_id, task, what)
            raise
        if not task.done():
            logger.warning("research.%s_cleanup_timed_out run_id=%s", what, run_id)
            self._absorb_when_done(run_id, task, what)
            return
        try:
            task.result()
        except (asyncio.CancelledError, StopAsyncIteration):
            pass
        except Exception:
            logger.warning("research.%s_cleanup_failed run_id=%s", what, run_id, exc_info = True)

    async def _iter_stream_lines(
        self,
        run_id: str,
        response: httpx.Response,
        semantic_deadline: Callable[[], tuple[float, type[BaseException]] | None] | None = None,
    ) -> AsyncIterator[str]:
        iterator = response.aiter_lines().__aiter__()

        def wait_timeout() -> float:
            if semantic_deadline is None:
                return 0.2
            deadline = semantic_deadline()
            if deadline is None:
                return 0.2
            at, expired = deadline
            remaining = at - asyncio.get_running_loop().time()
            if remaining > 0:
                return min(0.2, remaining)
            # Named by the caller, so a first-output deadline is never reported as a stall.
            raise expired()

        while True:
            if self._cancel_event(run_id).is_set():
                await self._check_active(run_id)
            timeout = wait_timeout()
            line_task = asyncio.create_task(anext(iterator))
            discarded = False
            try:
                while not line_task.done():
                    await asyncio.wait({line_task}, timeout = timeout)
                    if self._cancel_event(run_id).is_set():
                        # Set first: the finally must not spend the bound on it again.
                        discarded = True
                        await self._discard_task(run_id, line_task, "stream_iterator")
                        await self._check_active(run_id)
                    # A line that arrived during the wait is earned; recomputing the deadline first would let an expiry
                    # in the same turn discard it.
                    if line_task.done():
                        break
                    timeout = wait_timeout()
                try:
                    line = line_task.result()
                except StopAsyncIteration:
                    return
            finally:
                if not discarded and not line_task.done():
                    await self._discard_task(run_id, line_task, "stream_iterator")
            yield line

    async def _stream_completion(
        self,
        run: dict,
        messages: list[dict],
        *,
        json_mode: bool = False,
        report_progress: bool = True,
        phase: str = "unknown",
        step_position: int | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        preview_labels: bool = False,
    ) -> tuple[str, str, str | None, dict[str, int] | None]:
        call_id = uuid.uuid4().hex
        expires = (
            datetime.now(timezone.utc) + timedelta(seconds = _MODEL_CALL_KEY_LIFETIME_SECONDS)
        ).isoformat()
        key_minted = asyncio.get_running_loop().time()
        token, key = await asyncio.to_thread(
            auth_storage.create_api_key,
            username = run["ownerSubject"],
            # The name is load-bearing: the external-provider route scopes its saved-credential exception to
            # exactly this workflow.
            name = auth_storage.DEEP_RESEARCH_WORKFLOW_KEY_NAME,
            expires_at = expires,
            internal = True,
        )
        config = run["config"]
        inference = config.get("inferenceRequest") or {}
        payload: dict[str, Any] = {
            "model": inference.get("model") or config.get("model") or "",
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Keep every model hop in this durable run on one isolated Codex prompt-cache session rather than
            # sharing the transport fallback.
            "thread_id": f"research:{run['id']}",
            # Both opt-outs are needed: --enable-tools overrides a per-request enable_tools, and an omitted
            # enabled_tools resolves to every built-in, python and terminal included.
            "tool_choice": "none",
            "enabled_tools": [],
            "temperature": inference.get("temperature", 0.2),
        }

        # The route's _sanitize_config already refused anything but an enabled saved connection of a studio-
        # tools-capable provider type.
        if inference.get("providerType"):
            payload.update(
                {
                    "provider_id": inference["providerId"],
                    "provider_type": inference["providerType"],
                    "external_model": inference["externalModel"],
                }
            )
        if inference.get("topP") is not None:
            payload["top_p"] = inference["topP"]
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        elif inference.get("enableThinking") is not None:
            payload["enable_thinking"] = inference["enableThinking"]
        if enable_thinking is False:
            payload["reasoning_effort"] = "none"
        elif inference.get("reasoningEffort") is not None:
            payload["reasoning_effort"] = inference["reasoningEffort"]
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        report = ""
        reasoning = ""
        pending_report = ""
        pending_reasoning = ""
        pending_reasoning_offset = 0
        last_progress_flush = asyncio.get_running_loop().time()
        finish_reason: str | None = None
        usage: dict[str, int] | None = None
        semantic_output_at: float | None = None
        first_output_deadline: float | None = None
        emitted_labels = 0

        async def flush_progress() -> None:
            nonlocal pending_report, pending_reasoning, pending_reasoning_offset
            nonlocal last_progress_flush
            if pending_reasoning:
                try:
                    seq = await asyncio.to_thread(
                        db.append_worker_event,
                        run["id"],
                        self.worker_id,
                        "reasoning.updated",
                        {
                            "reasoningDelta": pending_reasoning,
                            "reasoningOffset": pending_reasoning_offset,
                            "phase": phase,
                            "callId": call_id,
                            **(
                                {"stepPosition": step_position} if step_position is not None else {}
                            ),
                        },
                    )
                    if seq is None:
                        await self._check_active(run["id"])
                        raise LeaseLost()
                    pending_reasoning = ""
                except (LeaseLost, RunCancelled):
                    raise
                except Exception:
                    logger.warning(
                        "research.reasoning_flush_failed run_id=%s",
                        run["id"],
                        exc_info = True,
                    )
                    last_progress_flush = asyncio.get_running_loop().time()
                    return
            if report_progress and pending_report:
                try:
                    written = await asyncio.to_thread(
                        db.set_report_progress,
                        run["id"],
                        report,
                        pending_report,
                        self.worker_id,
                    )
                    if not written:
                        await self._check_active(run["id"])
                        raise LeaseLost()
                    pending_report = ""
                except (LeaseLost, RunCancelled):
                    raise
                except Exception:
                    logger.warning(
                        "research.report_flush_failed run_id=%s",
                        run["id"],
                        exc_info = True,
                    )
            last_progress_flush = asyncio.get_running_loop().time()

        try:
            await self._note_phase(run["id"], "phase.started", phase, call_id, step_position)
            model_timeout = float(config["budgets"]["modelTimeoutSeconds"])
            # Configurable, capped by a finite run wall clock; legacy runs use the default.
            first_output_budget = float(
                config["budgets"].get(
                    "firstOutputTimeoutSeconds", _MODEL_FIRST_OUTPUT_TIMEOUT_SECONDS
                )
            )
            if model_timeout > 0:
                first_output_budget = min(first_output_budget, model_timeout)
            # Unlimited only drops the total wall clock; this bound also caps the silence between queue notices,
            # so it has to clear the heartbeat they are paced by.
            admission_gap_budget = max(
                first_output_budget,
                _MODEL_OUTPUT_IDLE_TIMEOUT_SECONDS,
                llama_admission_config_from_env().keepalive_interval_s
                * _ADMISSION_HEARTBEAT_MISSES,
            )
            timeout = (
                httpx.Timeout(model_timeout)
                if model_timeout
                else httpx.Timeout(
                    first_output_budget,
                    # Strictly looser than the guards above, so a stall is reported by name rather than as a message-
                    # less HTTPX ReadTimeout.
                    read = admission_gap_budget + _STREAM_READ_TIMEOUT_MARGIN_SECONDS,
                )
            )
            loop = asyncio.get_running_loop()

            def semantic_deadline() -> tuple[float, type[BaseException]] | None:
                if semantic_output_at is None:
                    if first_output_deadline is None:
                        return None
                    return first_output_deadline, ModelFirstOutputTimeout
                return (
                    semantic_output_at + _MODEL_OUTPUT_IDLE_TIMEOUT_SECONDS,
                    ModelOutputIdleTimeout,
                )

            call_started = loop.time()
            # No backoff below may outlast this call's wall clock or the key the re-send authenticates with.
            retry_deadline = key_minted + _MODEL_CALL_KEY_LIFETIME_SECONDS
            if model_timeout:
                retry_deadline = min(retry_deadline, call_started + model_timeout)
            async with (
                _wall_clock_timeout(model_timeout or None),
                httpx.AsyncClient(timeout = timeout, trust_env = False) as client,
            ):
                response: httpx.Response | None = None
                send_task: asyncio.Task | None = None
                # A retry builds a fresh task, so the guard starts over with it.
                send_discarded = False
                model_waits = 0
                attempt = 0
                try:
                    while True:
                        payload["max_tokens"] = _resolve_max_tokens(
                            max_tokens,
                            inference,
                            messages,
                        )
                        request = client.build_request(
                            "POST",
                            self._endpoint(),
                            json = payload,
                            headers = {"Authorization": f"Bearer {token}"},
                        )
                        try:
                            send_task = asyncio.create_task(client.send(request, stream = True))
                            send_discarded = False
                            while not send_task.done():
                                await asyncio.wait({send_task}, timeout = 0.2)
                                if self._cancel_event(run["id"]).is_set():
                                    # Set first: a send outlasting the bound is not waited on twice.
                                    send_discarded = True
                                    await self._discard_task(run["id"], send_task, "send")
                                    await self._check_active(run["id"])
                            response = await send_task
                            response.raise_for_status()
                            first_output_deadline = loop.time() + first_output_budget
                        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                            # Only reachable before a body byte is touched, so a re-send cannot duplicate report text.
                            unloaded = (
                                await _model_unloaded(exc.response)
                                if isinstance(exc, httpx.HTTPStatusError)
                                else None
                            )
                            rate_limited = (
                                isinstance(exc, httpx.HTTPStatusError)
                                and exc.response.status_code == 429
                            )
                            retryable = (
                                not isinstance(exc, httpx.HTTPStatusError)
                                or exc.response.status_code >= 500
                                or rate_limited
                            )
                            if unloaded:
                                model_waits += 1
                                if model_waits > _MAX_MODEL_WAITS:
                                    raise
                            elif not retryable or attempt == 2:
                                raise
                            if response is not None:
                                # Manual stream mode owns the connection; release it to re-send.
                                await response.aclose()
                                response = None
                            if unloaded == "switching":
                                await self._wait_for_model_switch(run, exc.response, model_waits)
                            elif unloaded:
                                # Nothing loaded (restart, eject): wait for a model to come back, without
                                # spending a transport attempt.
                                if not await self._wait_for_local_model(
                                    run,
                                    _NAMED_MODEL_WAIT_SECONDS if unloaded == "named" else None,
                                ):
                                    raise
                            else:
                                delay = 2**attempt
                                if rate_limited:
                                    # This runs to minutes, so re-read the run while it waits.
                                    await self._wait_out_rate_limit(
                                        run,
                                        _retry_after_seconds(exc.response) or delay,
                                        retry_deadline,
                                        first_output_budget,
                                    )
                                else:
                                    await asyncio.sleep(delay)
                                attempt += 1
                                # re-check the lease and cancellation before re-sending.
                                await self._check_active(run["id"])
                            continue
                        # A proxied provider 429 arrives as a 200 whose first line is the refusal, so the
                        # status cannot see
                        # it; no body byte is used yet.
                        stream = self._iter_stream_lines(run["id"], response, semantic_deadline)
                        head = await _peek_stream_head(stream)
                        throttled = _stream_rate_limit_delay(head)
                        if throttled is None or attempt == 2:
                            # Out of attempts: let the stream raise the provider's own error.
                            break
                        await stream.aclose()
                        await response.aclose()
                        response = None
                        await self._wait_out_rate_limit(
                            run, throttled or 2**attempt, retry_deadline, first_output_budget
                        )
                        attempt += 1
                        await self._check_active(run["id"])
                    async for line in _with_head(head, stream):
                        if self._cancel_event(run["id"]).is_set():
                            await self._check_active(run["id"])
                        if not line.startswith("data:"):
                            # Queueing has no timeout by design, so suspend for it and start the budget when the slot is
                            # granted.
                            if line.startswith(_ADMISSION_WAIT_COMMENT):
                                # Unlimited has no wall clock behind this, so bound the gap between queue
                                # notices; each notice
                                # refreshes it.
                                first_output_deadline = (
                                    None if model_timeout else loop.time() + admission_gap_budget
                                )
                            elif line.startswith(_ADMISSION_DONE_COMMENT):
                                first_output_deadline = loop.time() + first_output_budget
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        if not data:
                            continue
                        # Arming research in the composer is the approval, so the plan is queued as it is stored
                        # rather than parked for a second confirmation.
                        # revoked before the phase event, so a cancel there cannot leak a live key.
                        try:
                            chunk = json.loads(data)
                            _stream_error = stream_error_from_chunk(chunk)
                            if _stream_error is not None:
                                # The server's own text names the cause and both token counts; flattening it to
                                # a fixed string left
                                # the user nothing to act on.
                                raise _stream_error
                            normalized_usage = _normalize_completion_usage(
                                chunk.get("usage") if isinstance(chunk, dict) else None
                            )
                            if normalized_usage is not None:
                                usage = normalized_usage
                            choice = chunk.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            if isinstance(choice.get("finish_reason"), str):
                                finish_reason = choice["finish_reason"]
                            text = delta.get("content")
                        except (AttributeError, IndexError, json.JSONDecodeError, TypeError):
                            continue
                        thought = delta.get("reasoning_content")
                        if isinstance(thought, str) and thought:
                            semantic_output_at = loop.time()
                            if not pending_reasoning:
                                pending_reasoning_offset = len(reasoning)
                            reasoning += thought
                            pending_reasoning += thought
                        if isinstance(text, str) and text:
                            semantic_output_at = loop.time()
                            report += text
                            pending_report += text
                            # only a closing quote completes a title; per-token rescans cost ~170ms.
                            if preview_labels and '"' in text:
                                emitted_labels = await self._emit_preview_labels(
                                    run["id"], phase, call_id, report, emitted_labels
                                )
                        pending_chars = len(pending_reasoning) + len(pending_report)
                        if (
                            pending_chars >= 512
                            or pending_chars > 0
                            and asyncio.get_running_loop().time() - last_progress_flush >= 0.25
                        ):
                            await flush_progress()
                    if semantic_output_at is None:
                        raise ModelFirstOutputTimeout("Local model never produced output")
                finally:
                    if send_task is not None and not send_discarded and not send_task.done():
                        await self._discard_task(run["id"], send_task, "send")
                    if (
                        response is None
                        and send_task is not None
                        and send_task.done()
                        and not send_task.cancelled()
                    ):
                        try:
                            response = send_task.result()
                        except Exception:
                            pass
                    if response is not None:
                        try:
                            await response.aclose()
                        except Exception:
                            # Closing a broken stream is best-effort and must not replace the generation result
                            # or the error
                            # that caused teardown.
                            logger.warning(
                                "research.stream_cleanup_failed run_id=%s",
                                run["id"],
                                exc_info = True,
                            )
            await flush_progress()
            return report, reasoning, finish_reason, usage
        except (ModelFirstOutputTimeout, ModelOutputIdleTimeout, ModelWallClockTimeout):
            raise
        except httpx.ReadTimeout as exc:
            # Transport backstop: HTTPX raises this with no message, so name the stall instead.
            if semantic_output_at is None:
                raise ModelFirstOutputTimeout("Local model never produced output") from exc
            raise ModelOutputIdleTimeout("Local model stopped producing output") from exc
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise ModelWallClockTimeout(
                "Local model request exceeded its wall-clock timeout"
            ) from exc
        finally:
            # revoked before the phase event, so a cancel there cannot leak a live key.
            try:
                await asyncio.to_thread(auth_storage.revoke_internal_api_key, int(key["id"]))
            except Exception:
                logger.warning(
                    "research.api_key_cleanup_failed run_id=%s",
                    run["id"],
                    exc_info = True,
                )
            await self._note_phase(run["id"], "phase.ended", phase, call_id, step_position)

    async def _emit_preview_labels(
        self, run_id: str, phase: str, call_id: str, streamed: str, already_emitted: int
    ) -> int:
        """Publish each plan step title as the planner finishes writing it, and return the
        running total. Turns a multi-minute silent JSON generation into visible progress."""
        titles = _streamed_titles(streamed)
        emitted = already_emitted
        for label in titles[already_emitted:_MAX_PREVIEW_LABELS]:
            try:
                await asyncio.to_thread(
                    db.append_worker_event,
                    run_id,
                    self.worker_id,
                    "phase.progress",
                    {"phase": phase, "callId": call_id, "label": label},
                )
            except Exception:
                logger.debug("research.phase_preview_failed run_id=%s", run_id, exc_info = True)
                return emitted
            emitted += 1
        return emitted

    async def _note_phase(
        self, run_id: str, event_type: str, phase: str, call_id: str, step_position: int | None
    ) -> None:
        """Bracket one model call with a timeline event.

        Planning, per-step decisions, and the synthesis audit run with thinking disabled and
        report progress off, so they emit nothing for their whole duration. Without these the
        UI has no row to show and a multi-minute call looks like a stalled run.
        """
        try:
            await asyncio.to_thread(
                db.append_worker_event,
                run_id,
                self.worker_id,
                event_type,
                {
                    "phase": phase,
                    "callId": call_id,
                    **({"stepPosition": step_position} if step_position is not None else {}),
                },
            )
        except Exception:
            # Best effort: a progress marker must never fail the run it is reporting on.
            logger.debug("research.phase_event_failed run_id=%s", run_id, exc_info = True)

    async def _process(self, run: dict) -> None:
        # Everything this worker writes after a terminal status is only its to write while the run is still
        # on that attempt.
        attempt = int(run.get("retryCount") or 0)
        cancel_event = self._cancel_event(run["id"])
        if await asyncio.to_thread(db.is_cancel_requested, run["id"]):
            cancel_event.set()
        heartbeat = asyncio.create_task(self._heartbeat(run["id"]))
        try:
            await self._check_active(run["id"])
            if run["status"] == "planning":
                await self._plan(run)
            else:
                await self._research(run)
        except RunCancelled:
            actual_status = await asyncio.to_thread(
                db.finish, run["id"], self.worker_id, "cancelled"
            )
            fresh = await asyncio.to_thread(db.get_run, run["id"])
            if actual_status == "cancelled" and not _run_moved_on(fresh, attempt):
                await asyncio.to_thread(
                    _update_assistant, fresh, "Research cancelled.", "cancelled"
                )
        except LeaseLost:
            logger.warning("research.lease_lost run_id=%s", run["id"])
            actual_status = await self._finish_after_lease_loss(run["id"])
            fresh = await asyncio.to_thread(db.get_run, run["id"])
            if actual_status == "cancelled" and not _run_moved_on(fresh, attempt):
                await asyncio.to_thread(
                    _update_assistant,
                    fresh,
                    "Research cancelled.",
                    "cancelled",
                )
            elif actual_status == "failed" and not _run_moved_on(fresh, attempt):
                await asyncio.to_thread(
                    _update_assistant,
                    fresh,
                    "Research paused because its worker lease expired. Retry to continue.",
                    "failed",
                )
        except Exception as exc:
            error = _safe_error(exc)
            logger.warning("research.run_failed run_id=%s error=%s", run["id"], error)
            try:
                actual_status = await asyncio.to_thread(
                    db.finish, run["id"], self.worker_id, "failed", error
                )
            except sqlite3.OperationalError:
                actual_status = await self._finish_after_lease_loss(run["id"])
            if actual_status is None:
                actual_status = await self._finish_after_lease_loss(run["id"])
            fresh = await asyncio.to_thread(db.get_run, run["id"])
            if actual_status == "cancelled" and not _run_moved_on(fresh, attempt):
                await asyncio.to_thread(
                    _update_assistant, fresh, "Research cancelled.", "cancelled"
                )
            elif actual_status == "failed" and not _run_moved_on(fresh, attempt):
                await asyncio.to_thread(
                    _update_assistant, fresh, f"Research failed: {error}", "failed"
                )
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            self._cancel_events.pop(run["id"], None)
            self._lost_leases.discard(run["id"])

    async def _heartbeat(self, run_id: str) -> None:
        delay = 30.0
        consecutive_errors = 0
        while True:
            await asyncio.sleep(delay)
            delay = 30.0
            try:
                renewed = await asyncio.to_thread(db.heartbeat, run_id, self.worker_id)
            except Exception:
                logger.warning("research.heartbeat_failed run_id=%s", run_id, exc_info = True)
                # A busy SQLite writer is not proof that ownership was lost; retry briefly, but stop well before the
                # 120-second lease expires.
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    self._lost_leases.add(run_id)
                    self.cancel(run_id)
                    return
                delay = 1.0
                continue
            consecutive_errors = 0
            if not renewed:
                self._lost_leases.add(run_id)
                self.cancel(run_id)
                return

    async def _plan(self, run: dict) -> None:
        question, conversation_context = await asyncio.to_thread(
            _research_question_context,
            run["threadId"],
            run["userMessageId"],
            str(run["config"].get("question") or ""),
        )
        if not question:
            raise ValueError("User message has no text to research")
        max_steps = int(run["config"]["budgets"]["maxSteps"])
        planner_system = _system_prompt_with_instructions(
            _planner_system_prompt(max_steps, run["config"].get("websitePolicy")),
            run["config"],
        )
        # The question is budgeted before the history but is unbounded on its own (a pasted document arrives
        # verbatim) and would overflow before planning.
        planning_total = _prompt_char_budget(_SYNTHESIS_CONTEXT_RESERVE_TOKENS)
        planning_question = question[
            : max(
                _MIN_QUESTION_CHARS,
                _trimmable_budget(
                    planning_total, len(planner_system), _MAX_SYNTHESIS_EVIDENCE_CHARS
                ),
            )
        ]
        planning_context = conversation_context[
            : _trimmable_budget(
                planning_total, len(planner_system) + len(planning_question), _MAX_CONTEXT_CHARS
            )
        ]
        response, planning_reasoning, _finish_reason, _usage = await self._stream_completion(
            run,
            [
                {
                    "role": "system",
                    "content": planner_system,
                },
                {
                    "role": "user",
                    "content": (
                        "Prior conversation context as JSON (oldest to newest; use it only to "
                        "resolve references in the latest request):\n"
                        f"{_shield_untrusted(planning_context)}\n\n"
                        f"Latest research request:\n{_shield_untrusted(planning_question)}"
                    ),
                },
            ],
            json_mode = True,
            report_progress = False,
            phase = "planning",
            max_tokens = 4096,
            enable_thinking = False,
            preview_labels = True,
        )
        plan = _parse_and_validate_plan(response, planning_reasoning, max_steps)
        try:
            result = await asyncio.to_thread(
                db.set_plan,
                run["id"],
                plan,
                None,
                self.worker_id,
            )
        except db.ResearchConflictError:
            if await asyncio.to_thread(db.is_cancel_requested, run["id"]):
                raise RunCancelled()
            await self._check_active(run["id"])
            raise
        run.update(result)

    # The structured inline card renders the plan; no second markdown copy below it.
    async def _research(self, run: dict) -> None:
        resuming = run.get("claimedFromStatus") == "running"
        fresh = await asyncio.to_thread(db.get_run, run["id"])
        if not fresh or not fresh.get("plan"):
            raise ValueError("Approved plan is missing")
        run = fresh
        # The attempt this pass belongs to, kept because ``run`` is re-read below.
        research_attempt = int(run.get("retryCount") or 0)
        budgets = run["config"]["budgets"]
        max_steps = int(budgets["maxSteps"])
        max_sources = int(budgets["maxSources"])
        tool_timeout = int(budgets["toolTimeoutSeconds"])
        # Absent for runs created before auto-scrape: default 0 keeps their behavior unchanged.
        max_auto_scrape = int(budgets.get("maxAutoScrape", 0))
        # On a tiny context the prompt overhead alone fills the window, so fall back to snippet-only.
        if max_auto_scrape > 0:
            loaded_ctx = _loaded_context_length()
            if loaded_ctx is not None and loaded_ctx < _AUTO_SCRAPE_MIN_CONTEXT_TOKENS:
                logger.info(
                    "research.auto_scrape_disabled_small_context run_id=%s context=%s",
                    run["id"],
                    loaded_ctx,
                )
                max_auto_scrape = 0
        website_policy = run["config"].get("websitePolicy")
        policy_prompt = website_policy_prompt(website_policy)
        notes: list[str] = []
        decision_notes: list[str] = []
        research_state: dict[str, Any] = {}
        sources: list[dict] = []
        document_sources: list[dict] = []
        used_queries: set[str] = set()
        fetched_urls: set[str] = set()
        question, conversation_context = await asyncio.to_thread(
            _research_question_context,
            run["threadId"],
            run["userMessageId"],
            str(run["config"].get("question") or ""),
        )
        reset = db.prepare_execution_resume if resuming else db.reset_execution_steps
        written = await asyncio.to_thread(reset, run["id"], self.worker_id)
        await self._check_worker_write(run["id"], written)
        run = await asyncio.to_thread(db.get_run, run["id"])
        if not run:
            raise LeaseLost()
        if resuming:
            sources = list(run.get("sources") or [])[:max_sources]
            remaining = max(0, max_sources - len(sources))
            document_sources = list(run.get("documentSources") or [])[:remaining]

        for step in run.get("steps") or []:
            result = step.get("result") if isinstance(step.get("result"), dict) else {}
            action = str(result.get("action") or "search")
            argument = str(result.get("input") or step.get("query") or "")
            if action == "fetch":
                fetched_urls.add(argument)
            elif argument:
                used_queries.add(argument)
            if step.get("status") != "completed":
                continue
            restored_state = _normalize_research_state(result.get("researchState"))
            if restored_state:
                research_state = restored_state
            step_sources = [
                source for source in sources if source.get("stepPosition") == step.get("position")
            ]
            web_evidence = str(result.get("excerpt") or "")
            if not web_evidence and step_sources:
                web_evidence = "\n\n---\n\n".join(
                    f"Title: {source.get('title') or source['url']}\n"
                    f"URL: {source['url']}\n"
                    f"Snippet: {source.get('snippet') or ''}"
                    for source in step_sources
                )
            restored_rag_sources = [
                item for item in result.get("evidenceSources") or [] if isinstance(item, dict)
            ]
            document_source_keys = {
                str(
                    source.get("chunkId")
                    or f"{source.get('documentId') or source.get('filename')}:{source.get('page') or ''}"
                )
                for source in document_sources
            }
            # Evidence must hold only chunks that reached the catalog, else the validator strips citations to
            # the rest and synthesis builds claims on uncataloged text.
            accepted_rag_sources = []
            for source in restored_rag_sources:
                source_key = str(
                    source.get("chunkId")
                    or f"{source.get('documentId') or source.get('filename')}:{source.get('page') or ''}"
                )
                if source_key not in document_source_keys:
                    if len(sources) + len(document_sources) >= max_sources:
                        continue
                    written = await asyncio.to_thread(
                        db.upsert_document_source,
                        run["id"],
                        int(step["position"]),
                        source,
                        self.worker_id,
                    )
                    await self._check_worker_write(run["id"], written)
                    document_source_keys.add(source_key)
                    document_sources.append({**source, "stepPosition": step["position"]})
                accepted_rag_sources.append(source)
            rag_evidence = "\n".join(
                f"{item.get('filename') or 'Document'}: "
                f"{item.get('text') or item.get('snippet') or ''}"
                for item in accepted_rag_sources
            )
            title = str(step.get("title") or "Recovered research step")
            notes.append(
                f"### {title} ({action})\nInput: {argument}\nResult:\n{web_evidence}\n\n"
                f"Knowledge base:\n{rag_evidence}"
            )
            decision_notes.append(
                f"### {title} ({action})\nInput: {argument}\nResult:\n{web_evidence}"
            )

        start_position = (
            max(
                (int(step["position"]) for step in run.get("steps") or []),
                default = -1,
            )
            + 1
        )
        for position in range(start_position, max_steps):
            await self._check_active(run["id"])
            source_catalog = "\n".join(
                f"- {_citation_title(source, source['url'])} | {source['url']} | "
                f"{source.get('snippet') or ''}"
                for source in sources
            )
            evidence = "\n\n".join(decision_notes)
            decision_system = _system_prompt_with_instructions(
                _AGENT_SYSTEM_PROMPT + (f"\n\n{policy_prompt}" if policy_prompt else ""),
                run["config"],
            )
            # A fixed 60k evidence tail is many times a small context and this runs every step, so an overflow
            # here kills the run before it can synthesize.
            decision_total = _prompt_char_budget(_SYNTHESIS_CONTEXT_RESERVE_TOKENS)
            decision_question, decision_plan_json = _fit_decision_inputs(
                question,
                run["plan"],
                len(decision_system),
                decision_total,
            )
            # The catalog is unbounded too (maxSources entries, snippets up to 4000 chars), so it is fitted
            # before the sections that depend on what it leaves.
            decision_catalog = _fit_source_catalog(
                source_catalog,
                _trimmable_budget(
                    decision_total,
                    len(decision_system)
                    + len(decision_question)
                    + len(decision_plan_json)
                    + _MIN_SYNTHESIS_EVIDENCE_CHARS,
                    len(source_catalog),
                ),
            )
            decision_query_history_json = json.dumps(
                sorted(used_queries),
                ensure_ascii = False,
            )
            decision_state_json = json.dumps(research_state, ensure_ascii = False)
            decision_scaffold = (
                len(decision_system)
                + len(decision_question)
                + len(decision_plan_json)
                + len(decision_catalog)
                + len(decision_query_history_json)
                + len(decision_state_json)
            )
            evidence_chars = _trimmable_budget(
                decision_total, decision_scaffold, _MAX_SYNTHESIS_EVIDENCE_CHARS
            )
            decision_context = conversation_context[
                : _trimmable_budget(
                    decision_total, decision_scaffold + evidence_chars, _MAX_CONTEXT_CHARS
                )
            ]
            decision, decision_reasoning, _finish_reason, _usage = await self._stream_completion(
                run,
                [
                    {
                        "role": "system",
                        "content": decision_system,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Conversation context JSON:\n{_shield_untrusted(decision_context)}\n\n"
                            f"Question:\n{_shield_untrusted(decision_question)}\n\n"
                            f"Approved plan (guidance only):\n"
                            f"{_shield_untrusted(decision_plan_json)}\n\n"
                            f"Actions remaining after this one: {max_steps - position - 1}\n"
                            f"<untrusted_query_history_json>\n"
                            f"{_shield_untrusted(decision_query_history_json)}\n"
                            f"</untrusted_query_history_json>\n\n"
                            f"<untrusted_research_state_json>\n"
                            f"{_shield_untrusted(decision_state_json) or '{}'}\n"
                            f"</untrusted_research_state_json>\n\n"
                            f"<untrusted_web_evidence>\n"
                            f"Gathered sources:\n{_shield_untrusted(decision_catalog) or '(none)'}\n\n"
                            f"{_shield_untrusted(evidence[-evidence_chars:] if evidence_chars else '') or '(none)'}\n"
                            f"</untrusted_web_evidence>"
                        ),
                    },
                ],
                json_mode = True,
                report_progress = False,
                phase = "decision",
                step_position = position,
                max_tokens = 2048,
                enable_thinking = False,
            )
            try:
                action = _parse_and_validate_action(
                    decision,
                    decision_reasoning,
                    {source["url"] for source in sources},
                    website_policy,
                )
            except (ValueError, json.JSONDecodeError):
                action = _next_unused_seed_action(run["plan"], used_queries)
                if action is None:
                    break
            if action["action"] == "finish":
                if notes:
                    next_state = _normalize_research_state(action.get("researchState"))
                    if next_state:
                        research_state = next_state
                    break
                action = _next_unused_seed_action(run["plan"], used_queries)
                if action is None:
                    break
            argument = action.get("query") or action.get("url") or ""
            if action["action"] == "search":
                try:
                    argument = _sanitize_public_query(argument)
                    action["query"] = argument
                except ValueError:
                    replacement = _next_unused_seed_action(run["plan"], used_queries)
                    if replacement is None:
                        break
                    action = replacement
                    argument = action["query"]
            duplicate = (action["action"] == "search" and argument in used_queries) or (
                action["action"] == "fetch" and argument in fetched_urls
            )
            if duplicate:
                action = _next_unused_seed_action(run["plan"], used_queries)
                if action is None:
                    break
                argument = action["query"]
            # Persist model-derived state only after the action is final, so rejected decisions cannot leak
            # stale notes into the executed step, resume state, or synthesis.
            next_state = _normalize_research_state(action.get("researchState"))
            if next_state:
                research_state = next_state
            written = await asyncio.to_thread(
                db.upsert_execution_step,
                run["id"],
                position,
                action["title"],
                argument,
                "running",
                None,
                self.worker_id,
            )
            await self._check_worker_write(run["id"], written)
            seq = await asyncio.to_thread(
                db.append_worker_event,
                run["id"],
                self.worker_id,
                "step.started",
                {
                    "position": position,
                    "stepPosition": position,
                    "title": action["title"],
                    "action": action["action"],
                    "input": argument,
                },
            )
            await self._check_worker_write(run["id"], seq is not None)
            if action["action"] == "fetch":
                fetched_urls.add(argument)
                result = await asyncio.to_thread(
                    execute_tool,
                    "web_search",
                    {"url": argument},
                    cancel_event = self._cancel_event(run["id"]),
                    timeout = tool_timeout,
                    website_policy = website_policy,
                )
                rag_result = ""
            else:
                used_queries.add(argument)
                result = await asyncio.to_thread(
                    execute_tool,
                    "web_search",
                    {"query": argument},
                    cancel_event = self._cancel_event(run["id"]),
                    timeout = tool_timeout,
                    website_policy = website_policy,
                )
                rag_result = ""
                if run["config"].get("ragScope"):
                    rag_result = await asyncio.to_thread(
                        execute_tool,
                        "search_knowledge_base",
                        {"query": argument},
                        cancel_event = self._cancel_event(run["id"]),
                        timeout = tool_timeout,
                        rag_scope = run["config"]["ragScope"],
                    )
            rag_result, rag_sources = _split_rag_result(rag_result)
            await self._check_active(run["id"])
            document_source_keys = {
                str(
                    source.get("chunkId")
                    or f"{source.get('documentId') or source.get('filename')}:{source.get('page') or ''}"
                )
                for source in document_sources
            }
            accepted_rag_sources = []
            for source in rag_sources:
                source_key = str(
                    source.get("chunkId")
                    or f"{source.get('documentId') or source.get('filename')}:{source.get('page') or ''}"
                )
                if source_key not in document_source_keys:
                    if len(sources) + len(document_sources) >= max_sources:
                        continue
                    written = await asyncio.to_thread(
                        db.upsert_document_source,
                        run["id"],
                        position,
                        source,
                        self.worker_id,
                    )
                    await self._check_worker_write(run["id"], written)
                    document_source_keys.add(source_key)
                    document_sources.append({**source, "stepPosition": position})
                accepted_rag_sources.append(source)
            if accepted_rag_sources:
                rag_result = "\n\n".join(
                    f"Document: {source.get('filename') or 'Document'}"
                    f"{', page ' + str(source.get('page')) if source.get('page') is not None else ''}\n"
                    f"{source.get('text') or source.get('snippet') or ''}"
                    for source in accepted_rag_sources
                )
            elif rag_sources:
                # Chunks refused by the source cap have no catalog entry and the validator would strip every
                # citation to them; gated on rag_sources so a text-only KB reply still passes through.
                rag_result = ""
            rag_sources = accepted_rag_sources
            step_sources = []
            for match in _URL_BLOCK.finditer(result if action["action"] == "search" else ""):
                if len(sources) + len(document_sources) >= max_sources:
                    break
                source = {k: match.group(k).strip() for k in ("title", "url", "snippet")}
                allowed, _reason, _hostname = check_url_access(
                    source["url"],
                    website_policy,
                )
                if not allowed:
                    continue
                if source["url"] in {s["url"] for s in sources}:
                    continue
                sources.append(source)
                step_sources.append(source)
                await self._check_active(run["id"])
                written = await asyncio.to_thread(
                    db.upsert_source,
                    run["id"],
                    position,
                    source["url"],
                    source["title"],
                    source["snippet"],
                    self.worker_id,
                )
                await self._check_worker_write(run["id"], written)
            tool_failed = is_tool_error(result)
            step_failed = _research_step_failed(result, rag_sources)
            scraped_section = ""
            if (
                action["action"] == "search"
                and step_sources
                and not tool_failed
                and max_auto_scrape > 0
            ):
                scraped_section, scraped_urls = await self._auto_scrape_sources(
                    run,
                    question,
                    step_sources,
                    fetched_urls,
                    limit = max_auto_scrape,
                    tool_timeout = tool_timeout,
                    website_policy = website_policy,
                )
                fetched_urls.update(scraped_urls)
                await self._check_active(run["id"])
                if scraped_section:
                    # Additive, not replace: see _merge_scraped_evidence for why replacing the snippets regressed
                    # accuracy.
                    result = _merge_scraped_evidence(result, scraped_section)
            note = (
                f"### {action['title']} ({action['action']})\n"
                f"Input: {argument}\nResult:\n{result[:12000]}\n\n"
                f"Knowledge base:\n{rag_result[:6000]}"
            )
            notes.append(note)
            decision_notes.append(
                f"### {action['title']} ({action['action']})\n"
                f"Input: {argument}\nResult:\n{result[:12000]}"
            )
            clean_result = strip_result_for_model(result, "web_search")
            step_result = {
                "action": action["action"],
                "input": argument,
                "sourceCount": len(step_sources) + len(rag_sources),
                "sourceUrls": [source["url"] for source in step_sources],
                "evidenceSources": rag_sources,
                **(
                    {"excerpt": clean_result[:12000]}
                    if action["action"] == "fetch" or scraped_section
                    else {}
                ),
                **({"researchState": research_state} if research_state else {}),
                # tool_failed as well as step_failed: a tool error RAG rescued still records why.
                **({"error": clean_result[:500]} if tool_failed or step_failed else {}),
            }
            await self._check_active(run["id"])
            written = await asyncio.to_thread(
                db.upsert_execution_step,
                run["id"],
                position,
                action["title"],
                argument,
                "failed" if step_failed else "completed",
                step_result,
                self.worker_id,
            )
            await self._check_worker_write(run["id"], written)
            seq = await asyncio.to_thread(
                db.append_worker_event,
                run["id"],
                self.worker_id,
                "step.failed" if step_failed else "step.completed",
                {
                    "position": position,
                    "stepPosition": position,
                    "title": action["title"],
                    "action": action["action"],
                    "input": argument,
                    "sourceCount": len(step_sources) + len(rag_sources),
                    **({"error": clean_result[:500]} if step_failed else {}),
                },
            )
            await self._check_worker_write(run["id"], seq is not None)
        await self._check_active(run["id"])
        source_catalog = "\n".join(
            f"{index}. Title: {_citation_title(source, source['url'])}\n   URL: {source['url']}"
            for index, source in enumerate(sources, 1)
        )
        document_source_catalog = "\n".join(
            f"{index}. Filename: {source.get('filename') or 'Document'}\n"
            f"   Page: {source.get('page') if source.get('page') is not None else '(unknown)'}\n"
            f"   Citation: {_document_source_citation(source)}\n"
            f"   Document ID: {source.get('documentId') or '(unknown)'}\n"
            f"   Chunk ID: {source.get('chunkId') or '(unknown)'}"
            for index, source in enumerate(document_sources, 1)
        )
        # Model-derived JSON shares the evidence budget, and conversation history receives only what the
        # fixed scaffold leaves.
        total_budget = _prompt_char_budget(_SYNTHESIS_CONTEXT_RESERVE_TOKENS)
        plan_json = json.dumps(run["plan"], ensure_ascii = False)
        audit_system = _system_prompt_with_instructions(
            _SYNTHESIS_AUDIT_SYSTEM_PROMPT,
            run["config"],
        )
        audit_scaffold_chars = (
            len(audit_system)
            + len(question)
            + len(plan_json)
            + len(source_catalog)
            + len(document_source_catalog)
        )
        audit_evidence_text, [audit_state_json] = _fit_synthesis_context(
            notes,
            [research_state],
            audit_scaffold_chars,
        )
        audit_conversation_context = conversation_context[
            : _trimmable_budget(
                total_budget,
                audit_scaffold_chars + len(audit_evidence_text) + len(audit_state_json),
                _MAX_CONTEXT_CHARS,
            )
        ]
        (
            audit_response,
            audit_reasoning,
            _audit_finish_reason,
            _audit_usage,
        ) = await self._stream_completion(
            run,
            [
                {
                    "role": "system",
                    "content": audit_system,
                },
                {
                    "role": "user",
                    "content": (
                        f"<conversation_context_json>\n"
                        f"{_shield_untrusted(audit_conversation_context)}\n"
                        f"</conversation_context_json>\n\n"
                        f"<research_question>\n{_shield_untrusted(question)}\n"
                        f"</research_question>\n\n"
                        f"<approved_plan>\n"
                        f"{_shield_untrusted(plan_json)}\n"
                        f"</approved_plan>\n\n"
                        f"<source_catalog>\n"
                        f"{_shield_untrusted(source_catalog) or '(no web sources gathered)'}\n"
                        f"</source_catalog>\n\n"
                        f"<document_source_catalog>\n"
                        f"{_shield_untrusted(document_source_catalog) or '(no document sources gathered)'}\n"
                        f"</document_source_catalog>\n\n"
                        f"<untrusted_research_state_json>\n"
                        f"{_shield_untrusted(audit_state_json)}\n"
                        f"</untrusted_research_state_json>\n\n"
                        f"<untrusted_evidence>\n{_shield_untrusted(audit_evidence_text)}\n"
                        f"</untrusted_evidence>"
                    ),
                },
            ],
            json_mode = True,
            report_progress = False,
            phase = "synthesis_audit",
            max_tokens = 2048,
            enable_thinking = False,
        )
        synthesis_audit: dict[str, Any] = {}
        for candidate in (audit_response, audit_reasoning):
            if not candidate.strip():
                continue
            try:
                synthesis_audit = _normalize_synthesis_audit(
                    _parse_json_object(candidate),
                    {source["url"] for source in sources},
                    _allowed_document_citations(document_sources),
                )
                if synthesis_audit:
                    break
            except (ValueError, json.JSONDecodeError):
                continue
        report_system = _system_prompt_with_instructions(_REPORT_SYSTEM_PROMPT, run["config"])
        report_scaffold_chars = (
            len(report_system)
            + len(question)
            + len(plan_json)
            + len(source_catalog)
            + len(document_source_catalog)
        )
        evidence_text, [synthesis_audit_json, synthesis_state_json] = _fit_synthesis_context(
            notes,
            [synthesis_audit, research_state],
            report_scaffold_chars,
        )
        synthesis_conversation_context = conversation_context[
            : _trimmable_budget(
                total_budget,
                report_scaffold_chars
                + len(evidence_text)
                + len(synthesis_audit_json)
                + len(synthesis_state_json),
                _MAX_CONTEXT_CHARS,
            )
        ]
        synthesis_messages = [
            {
                "role": "system",
                "content": report_system,
            },
            {
                "role": "user",
                "content": (
                    f"<conversation_context_json>\n"
                    f"{_shield_untrusted(synthesis_conversation_context)}\n"
                    f"</conversation_context_json>\n\n"
                    f"<research_question>\n{_shield_untrusted(question)}\n"
                    f"</research_question>\n\n"
                    f"<approved_plan>\n{_shield_untrusted(plan_json)}\n"
                    f"</approved_plan>\n\n"
                    f"<source_catalog>\n{_shield_untrusted(source_catalog) or '(no web sources gathered)'}\n"
                    f"</source_catalog>\n\n"
                    f"<document_source_catalog>\n"
                    f"{_shield_untrusted(document_source_catalog) or '(no document sources gathered)'}\n"
                    f"</document_source_catalog>\n\n"
                    f"<untrusted_research_state_json>\n"
                    f"{_shield_untrusted(synthesis_state_json)}\n"
                    f"</untrusted_research_state_json>\n\n"
                    f"<untrusted_synthesis_audit_json>\n"
                    f"{_shield_untrusted(synthesis_audit_json)}\n"
                    f"</untrusted_synthesis_audit_json>\n\n"
                    f"<untrusted_evidence>\n{_shield_untrusted(evidence_text)}\n"
                    f"</untrusted_evidence>"
                ),
            },
        ]
        (
            report,
            synthesis_reasoning,
            synthesis_finish_reason,
            synthesis_usage,
        ) = await self._stream_completion(
            run,
            synthesis_messages,
            phase = "synthesis",
            max_tokens = 16384,
        )
        await self._check_active(run["id"])
        report = _select_synthesis_report(report, synthesis_reasoning)
        if _synthesis_needs_recovery(report, synthesis_finish_reason):
            recovery_reason = (
                "exhausted its output budget"
                if synthesis_finish_reason == "length"
                else "did not return a safely identifiable final report"
            )
            recovery_messages = [
                {
                    **synthesis_messages[0],
                    "content": (
                        synthesis_messages[0]["content"]
                        + f"\nThe previous synthesis {recovery_reason}. Write the report "
                        "directly without exposing analysis or reconstructing source URLs. Copy "
                        "citation titles and URLs only from the supplied catalogs. Begin with the "
                        "required final-report boundary on its own line."
                    ),
                },
                synthesis_messages[1],
            ]
            recovery_max_tokens = _resolve_max_tokens(
                16384,
                run["config"].get("inferenceRequest") or {},
                recovery_messages,
            )
            (
                recovered_report,
                recovery_reasoning,
                recovery_finish_reason,
                recovery_usage,
            ) = await self._stream_completion(
                run,
                recovery_messages,
                phase = "synthesis_recovery",
                max_tokens = 16384,
                enable_thinking = False,
            )
            synthesis_reasoning += recovery_reasoning
            report = _select_synthesis_report(recovered_report, recovery_reasoning)
            synthesis_finish_reason = recovery_finish_reason
            synthesis_usage = recovery_usage
            await self._check_active(run["id"])
            if synthesis_finish_reason == "length":
                raise ValueError(
                    _synthesis_length_limit_error(
                        synthesis_usage,
                        requested_max_tokens = recovery_max_tokens,
                    )
                )
        if not report:
            raise ValueError(
                "Local model returned no safely identifiable final report. Disable thinking or "
                "use a compatible chat template and retry."
            )
        report = _validate_report_sources(report, sources)
        report = _validate_report_document_sources(report, document_sources)
        reasoning = await asyncio.to_thread(db.get_reasoning_text, run["id"])
        if synthesis_reasoning and synthesis_reasoning not in reasoning:
            reasoning += synthesis_reasoning
        # Renew ownership before syncing the discoverable chat message; a restarted worker can safely overwrite it.
        renewed = await asyncio.to_thread(db.heartbeat, run["id"], self.worker_id)
        if not renewed:
            await self._check_active(run["id"])
            raise LeaseLost()
        await asyncio.to_thread(
            _update_assistant,
            run,
            report,
            "completed",
            sources,
            reasoning,
            self.worker_id,
        )
        actual_status = await asyncio.to_thread(
            db.finish, run["id"], self.worker_id, "completed", None, {"report": report}
        )
        if actual_status is None:
            raise LeaseLost()
        run = await asyncio.to_thread(db.get_run, run["id"])
        if actual_status == "cancelled" and not _run_moved_on(run, research_attempt):
            await asyncio.to_thread(_update_assistant, run, "Research cancelled.", "cancelled")
