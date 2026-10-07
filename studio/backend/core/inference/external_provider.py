# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async HTTP client proxying chat completions to external LLM providers. Most use OpenAI-compatible
/v1/chat/completions; Anthropic uses the native Messages API, translated here."""

import asyncio
import base64
import contextlib
import io
import json as _json
import math
import mimetypes
import random
import re
import threading
import time
import weakref
import wave
from typing import Any, AsyncGenerator, Awaitable, Callable, Literal, NamedTuple, Optional, Union
from urllib.parse import urlparse, urlsplit, urlunsplit

import httpx
import structlog


from core.inference.openai_responses_shared import (
    normalize_function_schema,
    responses_function_call,
    responses_function_output,
    responses_usage_to_chat,
    response_event_type,
)
from core.inference.sse_control_frames import sanitize_provider_sse_line
from models.providers import (
    normalize_provider_reasoning_config,
    validate_provider_reasoning_contract,
)

# Local servers apply the chat template themselves, so prompts here need delimiter sweeping;
# an unknown endpoint is assumed templated (a forged turn costs more than a lost space).
_TEMPLATE_APPLYING_PROVIDERS = frozenset({"vllm", "llama_cpp", "ollama", "custom", "lemonade"})

_CONTINUATION_FLAG_PROVIDERS = frozenset({"vllm", "llama_cpp"})

# Providers documenting stream_options.include_usage; strict endpoints 400 on unknown fields.
# openai is absent: /v1/responses reports usage on its own.
_USAGE_STREAM_OPTION_PROVIDERS = frozenset({"vllm", "llama_cpp", "openrouter", "kimi", "lemonade"})

# llama-server reads repeat_penalty, not repetition_penalty (as routes/inference does).
_REPETITION_PENALTY_BODY_KEY = {"llama_cpp": "repeat_penalty"}

logger = structlog.get_logger(__name__)

_MAX_CONCATENATED_WAV_BYTES = 64 * 1024 * 1024
_MAX_CONCATENATED_WAV_SEGMENTS = 1_024

CompactionFallback = Callable[
    [list[dict[str, Any]]],
    Awaitable[tuple[list[dict[str, Any]], Optional[int], Optional[str]]],
]


def _merge_concatenated_wav_segments(
    audio: bytes, cancelled: Optional[threading.Event] = None
) -> bytes:
    """Join a byte stream containing multiple complete WAV files into one WAV."""
    cancelled = cancelled or threading.Event()
    if len(audio) > _MAX_CONCATENATED_WAV_BYTES:
        return audio

    def _segment_offsets():
        offset = 0
        while offset < len(audio):
            if audio[offset : offset + 4] != b"RIFF" or audio[offset + 8 : offset + 12] != b"WAVE":
                raise ValueError("not a concatenated WAV stream")
            segment_end = offset + 8 + int.from_bytes(audio[offset + 4 : offset + 8], "little")
            if segment_end <= offset + 12 or segment_end > len(audio):
                raise ValueError("invalid RIFF segment length")
            yield offset, segment_end
            offset = segment_end

    try:
        params = None
        source = io.BytesIO(audio)
        segment_count = 0
        for segment_start, _segment_end in _segment_offsets():
            if cancelled.is_set():
                return audio
            segment_count += 1
            if segment_count > _MAX_CONCATENATED_WAV_SEGMENTS:
                return audio
            source.seek(segment_start)
            with wave.open(source, "rb") as reader:
                current = (
                    reader.getnchannels(),
                    reader.getsampwidth(),
                    reader.getframerate(),
                    reader.getcomptype(),
                    reader.getcompname(),
                )
                if params is None:
                    params = current
                elif current != params:
                    return audio
        if segment_count < 2:
            return audio
        assert params is not None
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(params[0])
            writer.setsampwidth(params[1])
            writer.setframerate(params[2])
            writer.setcomptype(params[3], params[4])
            for segment_start, _segment_end in _segment_offsets():
                if cancelled.is_set():
                    return audio
                source.seek(segment_start)
                with wave.open(source, "rb") as reader:
                    expected_bytes = (
                        reader.getnframes() * reader.getnchannels() * reader.getsampwidth()
                    )
                    observed_bytes = 0
                    while frames := reader.readframes(65_536):
                        if cancelled.is_set():
                            return audio
                        observed_bytes += len(frames)
                        writer.writeframesraw(frames)
                    if observed_bytes != expected_bytes:
                        return audio
        return output.getvalue()
    except MemoryError:
        raise
    except Exception:
        return audio


def _append_provider_path(base_url: str, endpoint: str) -> str:
    """Append an API path without moving it behind a base URL's query string."""
    parts = urlsplit(base_url)
    path = f"{parts.path.rstrip('/')}/{endpoint.lstrip('/')}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def caches_at_the_last_block(
    provider_type: Optional[str], model: Optional[str], enable_prompt_caching: Optional[bool]
) -> bool:
    if enable_prompt_caching is False:
        return False
    if provider_type == "anthropic":
        return True
    return provider_type == "openrouter" and (model or "").strip().lower().lstrip("~").startswith(
        "anthropic/"
    )


def _is_azure_openai_host(host: str) -> bool:
    return host.endswith((".openai.azure.com", ".services.ai.azure.com"))


def _is_openai_family_cloud(base_url: Optional[str]) -> bool:
    """True iff ``base_url`` points at OpenAI cloud or Azure OpenAI Foundry. Anchored to the URL
    host so a path/subdomain like ``https://api.openai.com.attacker.com/v1`` cannot bypass it
    (CodeQL py/incomplete-url-substring-sanitization). Scopes cloud-only Responses-API extensions
    that 400 on non-cloud OAI-compat servers. Azure Foundry resources live at
    ``<resource>.openai.azure.com`` and ``<resource>.services.ai.azure.com``; the leading
    dots on `endswith` stop the apexes from matching."""
    if not base_url:
        return False
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return host == "api.openai.com" or _is_azure_openai_host(host)


# Claude Opus 4.7+ / Claude 5 / Mythos 400 on sampling params. Family is [a-z]+ so 3-5 ids
# do not parse as 5; minor capped at 2 digits so snapshot dates are not read as minors.
_ANTHROPIC_MODEL_VERSION = re.compile(
    r"^claude-(?P<family>[a-z]+)-(?P<major>\d+)(?:[-.](?P<minor>\d{1,2}))?(?:[-.]|$)",
    re.IGNORECASE,
)
_OPENAI_REASONING_SUMMARY_UNSUPPORTED = re.compile(r"^o3(?:[-.]|$)")
_OPENAI_FIXED_SAMPLING_MODEL = re.compile(
    r"^(?:gpt-5(?:[.-]|$)|gpt-4\.5(?:[.-]|$)|o\d+(?:[.-]|$)|codex-mini(?:[.-]|$)|gpt-6-astra(?:[.-]|$))"
)
_OPENAI_NON_REASONING_CHAT_ALIAS = re.compile(r"-chat(?:-latest)?$")


def _openai_fixed_sampling_model(model: str) -> bool:
    normalized = model.strip().lower()
    return bool(
        _OPENAI_FIXED_SAMPLING_MODEL.match(normalized)
        and not _OPENAI_NON_REASONING_CHAT_ALIAS.search(normalized)
    )


_GEMINI3_FAMILY = re.compile(r"^gemini-(?:[3-9]|\d{2,})(?:\.\d+)?-")
_GEMINI3_PRO = re.compile(r"^gemini-(?:[3-9]|\d{2,})(?:\.\d+)?-pro")


def _anthropic_text_is_sendable(value: Any) -> bool:
    """Anthropic rejects empty and whitespace-only text; the composer joins with "\\n"."""
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _anthropic_sampling_params_removed(model: str) -> bool:
    """Whether Anthropic rejects non-default sampling params for ``model``."""
    normalized = model.strip().lower()
    if normalized == "claude-mythos-preview" or normalized.startswith("claude-mythos-preview-"):
        return True

    match = _ANTHROPIC_MODEL_VERSION.match(normalized)
    if match is None:
        return False

    family = match.group("family")
    version = (int(match.group("major")), int(match.group("minor") or 0))
    return version[0] >= 5 or (family == "opus" and version >= (4, 7))


def _openai_response_error_message(event: Any) -> str:
    """Extract a useful message from a Responses failure event."""
    if not isinstance(event, dict):
        return "OpenAI response failed without error details."

    response = event.get("response")
    if not isinstance(response, dict):
        response = {}
    for candidate in (response.get("error"), event.get("error")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, dict):
            message = candidate.get("message")
            code = candidate.get("code") or candidate.get("type")
            if isinstance(message, str) and message.strip():
                return f"{message.strip()} ({code})" if code else message.strip()

    message = event.get("message")
    code = event.get("code")
    if isinstance(message, str) and message.strip():
        return f"{message.strip()} ({code})" if code else message.strip()

    details = response.get("incomplete_details")
    if isinstance(details, dict):
        reason = details.get("reason") or details.get("message")
        if isinstance(reason, str) and reason.strip():
            return f"OpenAI response failed: {reason.strip()}"

    response_id = response.get("id")
    suffix = f" (response {response_id})" if isinstance(response_id, str) else ""
    return f"OpenAI response failed without error details{suffix}."


def _openai_response_finish_reason(response: Any, *, has_tool_calls: bool = False) -> str:
    """Map a completed Responses object to the closest Chat Completions finish reason."""
    if isinstance(response, dict) and response.get("status") == "incomplete":
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        return "content_filter" if reason == "content_filter" else "length"
    return "tool_calls" if has_tool_calls else "stop"


def _openai_image_replay_requires_reasoning(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def _sanitize_openai_reasoning_replay_item(item: Any) -> Optional[dict[str, Any]]:
    """Return a Responses input-safe reasoning item, if ``item`` is one. OpenAI image-generation
    docs allow follow-up edits via the previous ``image_generation_call`` id, and reasoning
    models can also require the paired ``reasoning`` output item in manually managed context, so
    keep only the public replay fields and drop everything else."""
    if not isinstance(item, dict) or item.get("type") != "reasoning":
        return None
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        return None
    summary_parts: list[dict[str, str]] = []
    summary = item.get("summary")
    if isinstance(summary, list):
        for part in summary:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "summary_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                summary_parts.append({"type": "summary_text", "text": text})
    # `id` and `summary` only: Responses rejects `status` on an input item.
    replay: dict[str, Any] = {"type": "reasoning", "id": item_id, "summary": summary_parts}
    # ZDR orgs get encrypted_content; carry it whenever present or the replay id resolves to nothing.
    encrypted = item.get("encrypted_content")
    if isinstance(encrypted, str) and encrypted:
        replay["encrypted_content"] = encrypted
    return replay


# OpenAI Responses citation markers use private-use codepoints; unresolved tokens are dropped.
_OPENAI_CITE_OPEN = "cite"
_OPENAI_CITE_STOP = ""
_OPENAI_CITE_DELIM = ""
_OPENAI_CITATION_MARKER = re.compile(
    f"{_OPENAI_CITE_OPEN}([^{_OPENAI_CITE_STOP}]+){_OPENAI_CITE_STOP}"
)


def _build_citation_lookup(url_citations: list[dict[str, Any]]) -> dict[str, tuple[int, str]]:
    """Map every known ``source_id`` alias to ``(citation_index, url)``. Accepts singular
    ``source_id`` and plural ``source_ids``. First-seen wins on collision so an earlier citation
    keeps its number."""
    by_source: dict[str, tuple[int, str]] = {}
    for idx, cit in enumerate(url_citations, start = 1):
        url = cit.get("url")
        if not isinstance(url, str) or not url:
            continue
        aliases: list[str] = []
        sid = cit.get("source_id")
        if isinstance(sid, str) and sid:
            aliases.append(sid)
        sids = cit.get("source_ids")
        if isinstance(sids, list):
            aliases.extend(s for s in sids if isinstance(s, str) and s)
        for alias in aliases:
            by_source.setdefault(alias, (idx, url))
    return by_source


def _replace_openai_citation_markers(text: str, url_citations: list[dict[str, Any]]) -> str:
    """Rewrite `\\ue200cite\\ue202SOURCE_ID[\\ue202LOCATOR]\\ue201` markers into `[[N]](URL)` per
    resolvable id. Multi-source markers expand to one link per id; unresolved tokens drop.
    Idempotent on text without private-use codepoints."""
    if not text or _OPENAI_CITE_STOP not in text:
        return text
    by_source = _build_citation_lookup(url_citations)

    def _sub(match: re.Match[str]) -> str:
        rendered: list[str] = []
        for tok in match.group(1).split(_OPENAI_CITE_DELIM):
            if not tok:
                continue
            hit = by_source.get(tok)
            if hit is None:
                continue
            idx, url = hit
            rendered.append(f"[[{idx}]]({url})")
        return "".join(rendered)

    return _OPENAI_CITATION_MARKER.sub(_sub, text)


def _rewrite_citation_markers_partial(
    text: str, url_citations: list[dict[str, Any]]
) -> tuple[str, bool]:
    """Like ``_replace_openai_citation_markers`` but also reports whether any marker referenced a
    source_id not yet in ``url_citations``. A url_citation's ``annotation.added`` event typically
    arrives AFTER the delta carrying the marker that references it, so callers buffer the segment
    until a later event records the annotation; unresolved markers are left verbatim so a
    follow-up pass still parses cleanly."""
    if not text or _OPENAI_CITE_STOP not in text:
        return text, False
    by_source = _build_citation_lookup(url_citations)
    has_unresolved = False

    def _sub(match: re.Match[str]) -> str:
        nonlocal has_unresolved
        tokens = [t for t in match.group(1).split(_OPENAI_CITE_DELIM) if t]
        rendered: list[str] = []
        any_unresolved = False
        for tok in tokens:
            hit = by_source.get(tok)
            if hit is None:
                any_unresolved = True
                continue
            idx, url = hit
            rendered.append(f"[[{idx}]]({url})")
        # Keep the marker verbatim if any token is unresolved so a late annotation can resolve it.
        if any_unresolved:
            has_unresolved = True
            return match.group(0)
        return "".join(rendered)

    return _OPENAI_CITATION_MARKER.sub(_sub, text), has_unresolved


def _split_pending_citation_tail(text: str) -> tuple[str, str]:
    """Split ``text`` into ``(head, pending_tail)`` for streamed deltas. A citation marker can
    straddle two SSE deltas, so the unterminated tail is buffered and prepended onto the next
    delta and the rewriter sees a complete marker. ``pending_tail`` is the longest suffix
    starting with ``\\ue200`` and lacking ``\\ue201``; ``head`` is safe to emit. Empty tail when
    ``text`` has no open or a fully closed marker."""
    if not text:
        return text, ""
    last_open = text.rfind("")
    if last_open == -1:
        return text, ""
    if _OPENAI_CITE_STOP in text[last_open:]:
        return text, ""
    return text[:last_open], text[last_open:]


def _record_openai_url_citation(
    url_citations: list[dict[str, Any]], payload: dict[str, Any]
) -> None:
    """Normalize and append one Responses ``url_citation`` annotation.

    API revisions have used ``source_id``, ``id``, ``locator``, and
    ``source_ids`` for the marker aliases. Citations sharing a URL retain one
    display index while accumulating every alias that can reference it.
    """
    if payload.get("type") != "url_citation":
        return
    url = payload.get("url")
    if not isinstance(url, str) or not url:
        return
    aliases: list[str] = []
    source_id = payload.get("source_id") or payload.get("id") or payload.get("locator")
    if isinstance(source_id, str) and source_id:
        aliases.append(source_id)
    source_ids = payload.get("source_ids")
    if isinstance(source_ids, list):
        aliases.extend(alias for alias in source_ids if isinstance(alias, str) and alias)

    for citation in url_citations:
        if citation["url"] != url:
            continue
        existing_aliases = citation.setdefault("source_ids", [])
        for alias in aliases:
            if alias not in existing_aliases:
                existing_aliases.append(alias)
        return

    url_citations.append(
        {
            "url": url,
            "title": payload.get("title") or url,
            "snippet": payload.get("snippet") or payload.get("quote") or "",
            "source_ids": aliases,
        }
    )


def _extract_web_search_action(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAI web_search_call action into card arguments. gpt-5.x agentic search emits
    three action types discriminated by `action.type`: `search` carries queries, `open_page` a
    url, `find_in_page` a url and a pattern. Reading only `action.query` renders the last two as
    an empty `Searching ""` card. Shapes per WebSearchToolCall in
    https://github.com/openai/openai-openapi."""
    if not isinstance(item, dict):
        return {}
    action = item.get("action") if isinstance(item.get("action"), dict) else {}
    action_type = action.get("type") if isinstance(action.get("type"), str) else ""
    query = ""
    for source in (action.get("queries"), item.get("queries")):
        if isinstance(source, list):
            joined = ", ".join(q for q in source if isinstance(q, str) and q)
            if joined:
                query = joined
                break
    if not query:
        for legacy in (action.get("query"), item.get("query")):
            if isinstance(legacy, str) and legacy:
                query = legacy
                break
    url = action.get("url") if isinstance(action.get("url"), str) else ""
    pattern = action.get("pattern") if isinstance(action.get("pattern"), str) else ""
    arguments: dict[str, Any] = {}
    if query:
        arguments["query"] = query
    if url:
        arguments["url"] = url
    if pattern:
        arguments["pattern"] = pattern
    if action_type:
        arguments["action_type"] = action_type
    return arguments


# Only these accept 24h prompt_cache_retention; others 400 (openai/codex#39397), so guess narrow.
_OPENAI_EXTENDED_CACHE_FAMILY = re.compile(r"^(?:gpt-5(?:\.\d+)?(?:[-.]|$)|gpt-4\.1$)")


class _AnthropicThinkingSpec(NamedTuple):
    prefixes: tuple[str, ...]
    kind: Literal["adaptive", "manual"]
    efforts: tuple[str, ...]
    # Claude 5 thinks by default so off needs an explicit disable; Fable/Mythos 5 400 on it.
    thinking_default_on: bool = False
    can_disable: bool = True


_ANTHROPIC_THINKING_SPECS = (
    _AnthropicThinkingSpec(
        prefixes = ("claude-fable-5", "claude-mythos-5"),
        kind = "adaptive",
        efforts = ("none", "low", "medium", "high", "xhigh", "max"),
        thinking_default_on = True,
        can_disable = False,
    ),
    _AnthropicThinkingSpec(
        prefixes = ("claude-opus-5", "claude-sonnet-5"),
        kind = "adaptive",
        efforts = ("none", "low", "medium", "high", "xhigh", "max"),
        thinking_default_on = True,
    ),
    _AnthropicThinkingSpec(
        prefixes = ("claude-opus-4-8", "claude-opus-4-7"),
        kind = "adaptive",
        efforts = ("none", "low", "medium", "high", "xhigh", "max"),
    ),
    _AnthropicThinkingSpec(
        prefixes = ("claude-opus-4-6", "claude-sonnet-4-6"),
        kind = "adaptive",
        efforts = ("none", "low", "medium", "high", "xhigh", "max"),
    ),
    _AnthropicThinkingSpec(
        prefixes = ("claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"),
        kind = "manual",
        efforts = ("none", "low", "medium", "high"),
    ),
    # Earlier Claude 4 models and 3.7 Sonnet only take manual budget_tokens; adaptive thinking returns a 400 there.
    _AnthropicThinkingSpec(
        prefixes = (
            "claude-opus-4-1",
            "claude-opus-4-0",
            "claude-opus-4-2025",
            "claude-sonnet-4-0",
            "claude-sonnet-4-2025",
            "claude-3-7-sonnet",
        ),
        kind = "manual",
        efforts = ("none", "low", "medium", "high"),
    ),
)


def _anthropic_spec_prefix_matches(model_lc: str, prefix: str) -> bool:
    """A version prefix ("claude-opus-4-1") must stop at a boundary or it swallows
    "claude-opus-4-15"; a truncated release date ("claude-opus-4-2025") has to run on, so only a
    4-digit tail may."""
    if model_lc == prefix:
        return True
    if not model_lc.startswith(prefix):
        return False
    rest = model_lc[len(prefix) :]
    if rest.startswith("-"):
        return True
    trailing_digits = re.search(r"\d+$", prefix)
    return bool(trailing_digits and len(trailing_digits.group()) >= 4 and rest[0].isdigit())


def _anthropic_thinking_spec(model: str) -> Optional[_AnthropicThinkingSpec]:
    model_lc = model.strip().lower()
    for spec in _ANTHROPIC_THINKING_SPECS:
        if any(_anthropic_spec_prefix_matches(model_lc, p) for p in spec.prefixes):
            return spec
    return None


def _anthropic_model_newer_than_specs(model: str) -> bool:
    match = _ANTHROPIC_MODEL_VERSION.match(model.strip().lower())
    if match is None:
        return False
    major, minor = int(match.group("major")), int(match.group("minor") or 0)
    return major >= 5 or (major == 4 and minor >= 6)


# Tool versions are date-pinned per model family; pick the newest the model accepts.
_ANTHROPIC_5_PREFIXES = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-8",
)
_ANTHROPIC_NEW_WEB_PREFIXES = _ANTHROPIC_5_PREFIXES + (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)
_ANTHROPIC_NEW_CODE_EXEC_PREFIXES = _ANTHROPIC_NEW_WEB_PREFIXES + (
    "claude-opus-4-5",
    "claude-sonnet-4-5",
)


def _anthropic_web_search_version(model: str) -> str:
    return (
        "web_search_20260209"
        if model.startswith(_ANTHROPIC_NEW_WEB_PREFIXES)
        else "web_search_20250305"
    )


def _anthropic_web_fetch_version(model: str) -> str:
    return (
        "web_fetch_20260209"
        if model.startswith(_ANTHROPIC_NEW_WEB_PREFIXES)
        else "web_fetch_20250910"
    )


def _anthropic_code_execution_version(model: str) -> str:
    return (
        "code_execution_20260120"
        if model.startswith(_ANTHROPIC_NEW_CODE_EXEC_PREFIXES)
        else "code_execution_20250825"
    )


_ANTHROPIC_CODE_EXECUTION_BETA = "code-execution-2025-08-25"


# Anthropic compaction uses compact-2026-01-12 and ignores unsupported models to avoid 400s.
_ANTHROPIC_COMPACTION_PREFIXES = _ANTHROPIC_5_PREFIXES + (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-mythos-preview",
)
_ANTHROPIC_COMPACTION_BETA = "compact-2026-01-12"
_ANTHROPIC_COMPACTION_TYPE = "compact_20260112"
# thresholds below 50K tokens return an upstream 400.
_ANTHROPIC_COMPACTION_MIN = 50_000
# compaction above this is slow and costly on 1M-token windows; Anthropic defaults to 150K.
_SERVER_COMPACTION_MAX = 200_000


# Anthropic fast mode is limited to Opus 5/4.8; Opus 4.7 rejects speed, 4.6 reports standard, and Priority conflicts.
_ANTHROPIC_FAST_MODE_BETA = "fast-mode-2026-02-01"
_ANTHROPIC_FAST_MODE_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-8",
)


def _anthropic_supports_compaction(model: str) -> bool:
    return model.startswith(_ANTHROPIC_COMPACTION_PREFIXES)


def compacts_server_side(
    provider_type: Optional[str], base_url: Optional[str], api_type: Optional[str], model: str
) -> bool:
    if provider_type == "anthropic":
        return _anthropic_supports_compaction(model)
    return (provider_type == "openai" or api_type == "responses") and _is_openai_family_cloud(
        base_url
    )


def _anthropic_supports_fast_mode(model: str) -> bool:
    # require a family boundary so IDs such as claude-opus-4-70 do not match.
    return any(model == p or model.startswith(f"{p}-") for p in _ANTHROPIC_FAST_MODE_PREFIXES)


# cap cited_text to bound SSE size; the frontend trims it to 240 characters.
_CITED_TEXT_MAX_LEN = 512


def _anthropic_citation_key(citation: dict[str, Any]) -> tuple:
    """build a stable Anthropic citation key that preserves range ends and distinct search results."""
    ctype = citation.get("type")
    doc = citation.get("document_index")
    title = citation.get("document_title") or ""
    if ctype == "char_location":
        return (
            ctype,
            doc,
            title,
            citation.get("start_char_index"),
            citation.get("end_char_index"),
        )
    if ctype == "page_location":
        return (
            ctype,
            doc,
            title,
            citation.get("start_page_number"),
            citation.get("end_page_number"),
        )
    if ctype == "content_block_location":
        return (
            ctype,
            doc,
            title,
            citation.get("start_block_index"),
            citation.get("end_block_index"),
        )
    if ctype == "search_result_location":
        return (
            ctype,
            citation.get("search_result_index"),
            citation.get("source"),
            citation.get("title") or "",
            citation.get("start_block_index"),
            citation.get("end_block_index"),
        )
    return (ctype, _json.dumps(citation, sort_keys = True))


class _MistralThinkingSpec(NamedTuple):
    models: tuple[str, ...]
    style: Literal["prompt_mode", "reasoning_effort", "disabled"]
    efforts: tuple[str, ...] = ()


_MISTRAL_THINKING_SPECS = (
    _MistralThinkingSpec(
        models = ("magistral-medium-latest",),
        style = "prompt_mode",
    ),
    _MistralThinkingSpec(
        models = (
            "mistral-small-latest",
            "mistral-small-2603",
            "mistral-medium-latest",
            "mistral-medium-2604",
            "mistral-medium-3-5",
            "mistral-vibe-cli-latest",
            "zai-glm-5-2",
        ),
        style = "reasoning_effort",
        efforts = ("none", "high"),
    ),
)

_OPENROUTER_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})
_OPENROUTER_MANDATORY_REASONING_MODELS = frozenset(
    {
        "~google/gemini-pro-latest",
        "baidu/cobuddy:free",
        "inclusionai/ring-2.6-1t:free",
        "deepseek/deepseek-r1",
    }
)


def _mistral_thinking_spec(model: str) -> _MistralThinkingSpec:
    for spec in _MISTRAL_THINKING_SPECS:
        if model in spec.models:
            return spec
    return _MistralThinkingSpec(models = (), style = "disabled")


def _apply_mistral_reasoning_controls(
    body: dict[str, Any],
    model: str,
    enable_thinking: Optional[bool],
    reasoning_effort: Optional[str],
) -> None:
    """Translate generic reasoning controls into Mistral's model-specific shape:
    magistral-medium-latest takes baseline or `prompt_mode="reasoning"`; mistral-small-latest /
    mistral-vibe-cli-latest / mistral-medium-3-5 take `reasoning_effort` in {"none", "high"}; all
    other tested Mistral models take no reasoning params."""
    model_for_matching = model.rsplit("/", 1)[-1].strip().lower()
    spec = _mistral_thinking_spec(model_for_matching)
    body.pop("prompt_mode", None)
    body.pop("reasoning_effort", None)

    if spec.style == "prompt_mode":
        if enable_thinking is True or reasoning_effort == "high":
            body["prompt_mode"] = "reasoning"
        return

    if spec.style == "reasoning_effort":
        if reasoning_effort in spec.efforts:
            body["reasoning_effort"] = reasoning_effort
        elif reasoning_effort in _REASONING_EFFORT_LEVELS or enable_thinking is True:
            body["reasoning_effort"] = "high"
        elif enable_thinking is False:
            body["reasoning_effort"] = "none"
        return

    # Other Mistral models reject the parameter, so a mis-cataloged effort is dropped here.


_REASONING_EFFORT_LEVELS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})
_DEEPSEEK_EFFORT_ALIASES = {"minimal": "low", "medium": "high", "xhigh": "high"}
_LOCAL_SERVER_EFFORT_ALIASES = {"minimal": "low", "xhigh": "high", "max": "high"}


def _apply_deepseek_reasoning_controls(
    body: dict[str, Any], enable_thinking: Optional[bool], reasoning_effort: Optional[str]
) -> None:
    effort = (reasoning_effort or "").strip().lower()
    if effort == "none" or (not effort and enable_thinking is False):
        body["thinking"] = {"type": "disabled"}
        return
    if effort in _REASONING_EFFORT_LEVELS:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = _DEEPSEEK_EFFORT_ALIASES.get(effort, effort)
        return
    if enable_thinking is True:
        body["thinking"] = {"type": "enabled"}


def _apply_qwen_reasoning_controls(body: dict[str, Any], enable_thinking: Optional[bool]) -> None:
    if enable_thinking is not None:
        body["enable_thinking"] = bool(enable_thinking)


def _apply_passthrough_reasoning_effort(
    body: dict[str, Any],
    enable_thinking: Optional[bool],
    reasoning_effort: Optional[str],
    aliases: dict[str, str] | None = None,
) -> None:
    effort = (reasoning_effort or "").strip().lower()
    if aliases:
        effort = aliases.get(effort, effort)
    if effort == "none" or (not effort and enable_thinking is False):
        body["reasoning_effort"] = "none"
    elif effort in _REASONING_EFFORT_LEVELS:
        body["reasoning_effort"] = effort


# ollama's openai-compatible /v1/chat/completions accepts these five values.
# https://docs.ollama.com/api/openai-compatibility
_OLLAMA_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "max"})
_OLLAMA_REASONING_EFFORT_ALIASES = {
    "minimal": "low",
    "xhigh": "max",
}


def _apply_ollama_reasoning_controls(
    body: dict[str, Any], enable_thinking: Optional[bool], reasoning_effort: Optional[str]
) -> None:
    """Map API thinking controls onto Ollama's ``reasoning_effort`` field. Requests with neither
    control remain unchanged. An explicit off is ``none``; an on without a level is ``medium``.
    #9649"""
    effort = (reasoning_effort or "").strip().lower()
    if effort in _OLLAMA_REASONING_EFFORT_ALIASES:
        effort = _OLLAMA_REASONING_EFFORT_ALIASES[effort]
    if effort in _OLLAMA_REASONING_EFFORTS:
        body["reasoning_effort"] = effort
        return
    if enable_thinking is False:
        body["reasoning_effort"] = "none"
        return
    if enable_thinking is True:
        body["reasoning_effort"] = "medium"


def _create_shared_http_client() -> httpx.AsyncClient:
    # Unsupported env proxy schemes (socks:// etc) raise at construction and would crash Unsloth startup (#6090);
    # retry ignoring env proxies instead.
    try:
        return httpx.AsyncClient()
    except (ImportError, ValueError) as exc:
        exc_str = str(exc)
        if "Unknown scheme for proxy URL" not in exc_str and "socksio" not in exc_str:
            raise
        logger.warning(
            "Ignoring unsupported environment proxy for the shared HTTP client: %s", exc_str
        )
        return httpx.AsyncClient(trust_env = False)


_http_client = _create_shared_http_client()
# Studio's own loopback runtime: an env proxy would receive its key and prompts.
_loopback_http_client = httpx.AsyncClient(trust_env = False)


class _PinnedPublicTransport(httpx.AsyncBaseTransport):
    """Managed-account egress: re-resolve per connection, dial one validated public address."""

    def __init__(self):
        self._transports: dict[tuple, httpx.AsyncHTTPTransport] = {}

    def _pool(self, origin: tuple) -> httpx.AsyncHTTPTransport:
        if origin not in self._transports:
            self._transports[origin] = httpx.AsyncHTTPTransport(trust_env = False)
        return self._transports[origin]

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        from core.inference.providers import _public_registry_hostname, public_provider_address

        host = request.url.host
        if _public_registry_hostname(host):
            return await self._pool(("registry",)).handle_async_request(request)
        try:
            address = await asyncio.to_thread(public_provider_address, str(request.url))
        except ValueError as exc:
            raise httpx.ConnectError(str(exc), request = request) from exc
        pinned = httpx.Request(
            method = request.method,
            url = request.url.copy_with(host = address),
            headers = request.headers,
            stream = request.stream,
            extensions = {**request.extensions, "sni_hostname": host},
        )
        origin = (request.url.scheme, host, request.url.port)
        return await self._pool(origin).handle_async_request(pinned)

    async def aclose(self) -> None:
        for transport in self._transports.values():
            await transport.aclose()


class _PinnedNonMetadataTransport(_PinnedPublicTransport):
    """Managed-account egress once the owner has allowed private addresses.

    The public pin, one rule looser: the dialled address may be private, never metadata. The
    re-resolve stays, or a name that validated as public could answer 169.254.169.254 by connect
    time, which is the one destination this switch may not open.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        from core.inference.providers import (
            _public_registry_hostname,
            provider_address_excluding_metadata,
        )

        host = request.url.host
        if _public_registry_hostname(host):
            return await self._pool(("registry",)).handle_async_request(request)
        try:
            address = await asyncio.to_thread(provider_address_excluding_metadata, str(request.url))
        except ValueError as exc:
            raise httpx.ConnectError(str(exc), request = request) from exc
        pinned = httpx.Request(
            method = request.method,
            url = request.url.copy_with(host = address),
            headers = request.headers,
            stream = request.stream,
            extensions = {**request.extensions, "sni_hostname": host},
        )
        origin = (request.url.scheme, host, request.url.port)
        return await self._pool(origin).handle_async_request(pinned)


# Keyed by account: AsyncClient persists cookies, so sharing would cross gateway sessions.
_managed_clients: dict[tuple[str, bool], httpx.AsyncClient] = {}
_managed_clients_lock = threading.Lock()
_retired_accounts: set[str] = set()
_RETIRED_ACCOUNTS_MAX = 1024
_client_loops: "weakref.WeakKeyDictionary[httpx.AsyncClient, asyncio.AbstractEventLoop]" = (
    weakref.WeakKeyDictionary()
)


def retire_account_clients(account_id: str) -> int:
    """Drop and close a retired account's provider clients, returning how many were held.

    Without it the cache is bounded by accounts ever CREATED, and a deleted account's cookie jar
    and idle sockets outlive it for the life of the process.
    """
    with _managed_clients_lock:
        retired = [
            _managed_clients.pop(key) for key in list(_managed_clients) if key[0] == account_id
        ]
        # Tombstoned: a request authenticated before deactivation may still reach _client() after this.
        _retired_accounts.add(account_id)
        while len(_retired_accounts) > _RETIRED_ACCOUNTS_MAX:
            _retired_accounts.pop()
    for client in retired:
        loop = _client_loops.get(client)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and (loop is None or loop is running):
            running.create_task(client.aclose())
        elif loop is not None and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(client.aclose(), loop)
    return len(retired)


def restore_account_clients(account_id: str) -> None:
    """Lift the retirement tombstone: a failed delete leaves the account reactivatable, and one
    reactivated under a tombstone would never be cached again, so no pooling and no cookies."""
    with _managed_clients_lock:
        _retired_accounts.discard(account_id)


def _rejects_max_tokens(status_code: int, error_text: str) -> bool:
    """400 from an upstream that wants `max_completion_tokens` (Azure gpt-5.x / o-series behind custom gateways, #10787)."""
    if status_code != 400:
        return False
    try:
        err = _json.loads(error_text).get("error")
    except Exception:
        err = None
    if isinstance(err, dict) and err.get("param") == "max_tokens":
        return err.get("code") == "unsupported_parameter" or "max_completion_tokens" in str(
            err.get("message", "")
        )
    return "max_tokens" in error_text and "max_completion_tokens" in error_text


def _with_max_completion_tokens(body: dict[str, Any]) -> dict[str, Any]:
    body = dict(body)
    body["max_completion_tokens"] = body.pop("max_tokens")
    return body


@contextlib.asynccontextmanager
async def _stream_post_retrying_max_tokens(
    http: httpx.AsyncClient, url: str, body: dict[str, Any], **kwargs
):
    """`http.stream("POST", ...)` that resends once with `max_completion_tokens` if the upstream rejects `max_tokens`.
    Nothing has been yielded to the caller at the status check, so the retry is invisible."""
    async with http.stream("POST", url, json = body, **kwargs) as response:
        retry = (
            response.status_code == 400
            and "max_tokens" in body
            and _rejects_max_tokens(400, (await response.aread()).decode("utf-8", errors = "replace"))
        )
        if not retry:
            yield response
            return
    logger.info("Upstream rejected max_tokens; retrying with max_completion_tokens")
    async with http.stream(
        "POST", url, json = _with_max_completion_tokens(body), **kwargs
    ) as response:
        yield response


def _client() -> httpx.AsyncClient:
    """The shared client for the owner; a screening client of its own for each managed account."""
    from utils.account_context import current_account_id, is_owner_context
    from utils.managed_provider_url_settings import get_managed_private_provider_urls_allowed

    if is_owner_context():
        return _http_client
    allowed = get_managed_private_provider_urls_allowed()
    account_id = current_account_id()
    key = (account_id, allowed)
    with _managed_clients_lock:
        client = _managed_clients.get(key)
        if client is not None:
            return client
        transport = _PinnedNonMetadataTransport() if allowed else _PinnedPublicTransport()
        client = httpx.AsyncClient(transport = transport, trust_env = False)
        try:
            _client_loops[client] = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if account_id not in _retired_accounts:
            _managed_clients[key] = client
        return client


# Cap per-image fetch well below Gemini's ~20 MB total request budget.
_GEMINI_REMOTE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
_GEMINI_REMOTE_IMAGE_TIMEOUT_S = 15.0
_REMOTE_IMAGE_FETCH_DEADLINE_S = 30.0


def safe_fetch_remote_image_sync(
    url: str,
    fallback_mime: str,
    max_bytes: int = _GEMINI_REMOTE_IMAGE_MAX_BYTES,
    label: str = "Remote image fetch",
    deadline: Optional[float] = None,
    require_image_content_type: bool = True,
) -> Optional[tuple[str, str]]:
    """Fetch an HTTPS image through a validated, pinned public IP.

    Redirects are revalidated and the response is capped by ``max_bytes``. ``deadline`` is a
    ``time.monotonic`` cutoff for the whole fetch. A caller that decodes the bytes itself can
    pass ``require_image_content_type=False`` to accept any declared type as ``fallback_mime``.
    All failures return ``None`` so callers do not expose details about the host network.
    """
    import http.client
    import urllib.error
    import urllib.request
    from urllib.parse import quote, urljoin, urlunparse

    _byte_limit = min(max(0, int(max_bytes)), _GEMINI_REMOTE_IMAGE_MAX_BYTES)
    if _byte_limit <= 0:
        return None

    from .tools import (
        _IRI_PATH_SAFE,
        _IRI_QUERY_SAFE,
        _explicit_proxy_applies,
        _fetch_budget_exceeded,
        _fetch_hop_timeout,
        _NoRedirect,
        _pinned_netloc,
        _read_capped_body,
        _resolve_with_budget,
        _SNIHTTPSHandler,
        _USER_AGENTS,
    )

    user_agent = random.choice(_USER_AGENTS)
    if deadline is None:
        deadline = time.monotonic() + _REMOTE_IMAGE_FETCH_DEADLINE_S

    def _safe_parse_https(raw_url: str) -> Optional[tuple[Any, str, int]]:
        """Return a parsed HTTPS URL, hostname and port, or ``None``."""
        try:
            parsed_url = urlparse(raw_url)
            host_value = parsed_url.hostname
            port_value = parsed_url.port or 443
        except (ValueError, UnicodeError) as _err:
            logger.info(
                f"{label}: refusing malformed url err=%s",
                type(_err).__name__,
            )
            return None
        scheme_value = (parsed_url.scheme or "").lower()
        if scheme_value != "https":
            logger.info(
                f"{label}: refusing non-https scheme=%s",
                scheme_value,
            )
            return None
        if not host_value:
            logger.info(f"{label}: refusing url with no hostname")
            return None
        if not host_value.isascii():
            try:
                host_value = host_value.encode("idna").decode("ascii")
            except UnicodeError:
                logger.info(f"{label}: refusing unencodable hostname")
                return None
        return parsed_url, host_value, port_value

    parsed_info = _safe_parse_https(url)
    if parsed_info is None:
        return None
    parsed, current_host, current_port = parsed_info
    current_url = url
    ok, reason, pinned_ips = _resolve_with_budget(current_host, current_port, deadline, None)
    if not ok:
        logger.warning(
            f"{label}: refusing host=%s reason=%s",
            current_host,
            reason,
        )
        return None

    for _hop in range(4):
        budget_error = _fetch_budget_exceeded(deadline, None)
        if budget_error is not None:
            logger.info(f"{label}: {budget_error} host=%s", current_host)
            return None
        # Pin to validated IP; SNI + cert still use the hostname via _SNIHTTPSHandler.
        cp_info = _safe_parse_https(current_url)
        if cp_info is None:
            return None
        cp, _cp_host, _cp_port = cp_info
        try:
            cp = cp._replace(
                path = quote(cp.path, safe = _IRI_PATH_SAFE),
                params = quote(cp.params, safe = _IRI_PATH_SAFE),
                query = quote(cp.query, safe = _IRI_QUERY_SAFE),
            )
        except UnicodeError:
            logger.info(f"{label}: refusing unencodable url host=%s", current_host)
            return None
        pinned_url = urlunparse(cp._replace(netloc = _pinned_netloc(pinned_ips[0], cp.port)))

        # Route on hostname: NO_PROXY entries never match the pinned IP.
        authority = _pinned_netloc(current_host, cp.port)
        proxied = _explicit_proxy_applies("https", authority)
        handlers = [_NoRedirect, _SNIHTTPSHandler(current_host, () if proxied else pinned_ips)]
        if not proxied:
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        req = urllib.request.Request(
            pinned_url,
            headers = {"Host": authority, "User-Agent": user_agent},
            method = "GET",
        )

        try:
            resp = opener.open(
                req, timeout = _fetch_hop_timeout(_GEMINI_REMOTE_IMAGE_TIMEOUT_S, deadline)
            )
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                logger.info(
                    f"{label}: status=%d host=%s",
                    e.code,
                    current_host,
                )
                return None
            location = e.headers.get("Location")
            if not location:
                return None
            try:
                current_url = urljoin(current_url, location)
            except (ValueError, UnicodeError) as _err:
                logger.info(
                    f"{label}: refusing malformed redirect err=%s",
                    type(_err).__name__,
                )
                return None
            rp_info = _safe_parse_https(current_url)
            if rp_info is None:
                return None
            _rp, current_host, current_port = rp_info
            ok2, reason2, pinned_ips = _resolve_with_budget(
                current_host, current_port, deadline, None
            )
            if not ok2:
                logger.warning(
                    f"{label}: refusing redirect host=%s reason=%s",
                    current_host,
                    reason2,
                )
                return None
            continue
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as _err:
            logger.warning(
                f"{label} failed host=%s err=%s",
                current_host,
                type(_err).__name__,
            )
            return None

        with resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status != 200:
                logger.info(f"{label}: status=%s host=%s", status, current_host)
                return None
            _hdr_mime = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            if _hdr_mime and not _hdr_mime.startswith("image/") and not require_image_content_type:
                _hdr_mime = ""
            if _hdr_mime and not _hdr_mime.startswith("image/"):
                logger.info(
                    f"{label}: non-image content-type=%s host=%s",
                    _hdr_mime,
                    current_host,
                )
                return None
            _final_mime_pre = _hdr_mime if _hdr_mime else fallback_mime
            if not isinstance(_final_mime_pre, str) or not _final_mime_pre.startswith("image/"):
                logger.info(
                    f"{label}: missing content-type and no image fallback host=%s",
                    current_host,
                )
                return None
            _hdr_len = resp.headers.get("content-length")
            if _hdr_len and _hdr_len.isdigit() and int(_hdr_len) > _byte_limit:
                logger.info(
                    f"{label}: declared %s bytes exceeds cap=%s host=%s",
                    _hdr_len,
                    _byte_limit,
                    current_host,
                )
                return None
            try:
                body_error, raw = _read_capped_body(
                    resp, _byte_limit + 1, _GEMINI_REMOTE_IMAGE_TIMEOUT_S, deadline, None
                )
            except (OSError, http.client.HTTPException, ValueError) as _err:
                logger.warning(
                    f"{label} failed host=%s err=%s",
                    current_host,
                    type(_err).__name__,
                )
                return None
            if body_error is not None:
                logger.info(f"{label}: {body_error} host=%s", current_host)
                return None
            if len(raw) > _byte_limit:
                logger.info(
                    f"{label}: streamed bytes exceed cap=%s host=%s",
                    _byte_limit,
                    current_host,
                )
                return None
            return _final_mime_pre, base64.b64encode(raw).decode("ascii")

    logger.info(f"{label}: too many redirects host=%s", current_host)
    return None


_GEMINI_IMAGE_FETCH_LABEL = "Gemini image fetch"


def _safe_fetch_image_for_gemini_sync(
    url: str,
    fallback_mime: str,
    max_bytes: int = _GEMINI_REMOTE_IMAGE_MAX_BYTES,
) -> Optional[tuple[str, str]]:
    return safe_fetch_remote_image_sync(
        url, fallback_mime, max_bytes, label = _GEMINI_IMAGE_FETCH_LABEL
    )


async def _safe_fetch_image_for_gemini(
    url: str,
    fallback_mime: str,
    max_bytes: int = _GEMINI_REMOTE_IMAGE_MAX_BYTES,
) -> Optional[tuple[str, str]]:
    """Async wrapper running the IP-pinned fetch on a worker thread. SSRF guards (https only, pinned
    IP, per-hop redirect re-check, size cap, image/* content-type) live in the sync helper.
    `max_bytes` carries the remaining per-request budget so over-budget URLs are rejected up
    front."""
    import asyncio
    return await asyncio.to_thread(_safe_fetch_image_for_gemini_sync, url, fallback_mime, max_bytes)


_SERVER_SIDE_BUILTIN_TOOL_NAMES = frozenset(
    {"web_search", "web_fetch", "code_execution", "image_generation"}
)


def _stamp_server_tool_marker(payload: dict[str, Any]) -> None:
    """Tag synthetic provider-side tool events so the frontend can tell them from real user-declared
    / local function tools of the same name. The marker rides on `arguments._server_tool` and is
    only added for known server-side builtin names, so user-supplied tool calls echoed back
    through these helpers (e.g. Kimi `$web_search`) keep their shape."""
    if not isinstance(payload, dict):
        return
    if payload.get("type") != "tool_start":
        return
    name = payload.get("tool_name")
    if not isinstance(name, str) or name not in _SERVER_SIDE_BUILTIN_TOOL_NAMES:
        return
    args = payload.get("arguments")
    if not isinstance(args, dict):
        args = {}
        payload["arguments"] = args
    args["_server_tool"] = True


def _build_kimi_tool_end(
    synthetic_chunk_fn: Any, tool_call_id: str, citations: list[dict[str, str]]
) -> str:
    """Format Kimi web_search citations into the tool_end payload, in the shape the frontend's
    parseSourcesFromResult expects for the other built-in web_search providers. With no
    citations, fall back to a generic "(search complete)" string so the UI still transitions the
    tool card to completed."""
    blocks: list[str] = []
    for cit in citations:
        line = f"Title: {cit['title']}\nURL: {cit['url']}"
        if cit.get("snippet"):
            line += f"\nSnippet: {cit['snippet']}"
        blocks.append(line)
    return synthetic_chunk_fn(
        {
            "type": "tool_end",
            "tool_call_id": tool_call_id,
            "result": "\n---\n".join(blocks) if blocks else "(search complete)",
        }
    )


def _apply_custom_reasoning_controls(
    body: dict[str, Any],
    config: Optional[dict],
    enable_thinking: Optional[bool],
    reasoning_effort: Optional[str],
) -> None:
    """Emit exactly one explicitly configured dialect. Disabled/corrupt config emits nothing."""
    if not config or not config["enabled"]:
        return
    off = enable_thinking is False or reasoning_effort == "none"
    # Explicit off must still reach the server when a previous model left a wider effort level.
    if (
        not off
        and reasoning_effort is not None
        and reasoning_effort not in {"low", "medium", "high"}
    ):
        return
    effort = "none" if off else (reasoning_effort or "medium")
    style = config["style"]
    if style == "reasoning_effort":
        body["reasoning_effort"] = effort
    elif style == "reasoning":
        body["reasoning"] = {"enabled": False} if off else {"effort": effort}
    elif style == "thinking":
        body["thinking"] = {"type": "disabled" if off else "enabled"}
    elif style == "chat_template_kwargs.enable_thinking":
        body["chat_template_kwargs"] = {"enable_thinking": not off}


class ExternalProviderClient:
    """Async proxy for OpenAI-compatible external APIs."""

    def __init__(
        self,
        provider_type: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        *,
        api_type: str = "chat_completions",
        reasoning_config: Optional[dict] = None,
        managed_loopback: bool = False,
    ):
        self.provider_type = provider_type
        self.api_type = api_type if provider_type == "custom" else "chat_completions"
        self.reasoning_config = (
            normalize_provider_reasoning_config(reasoning_config)
            if provider_type == "custom"
            else None
        )
        validate_provider_reasoning_contract(provider_type, self.api_type, self.reasoning_config)
        from core.inference.providers import validate_provider_base_url

        self.base_url = (
            base_url.rstrip("/") if managed_loopback else validate_provider_base_url(base_url)
        )
        self._managed_loopback = managed_loopback
        if self.provider_type == "gemini":
            _parsed_base = urlparse(self.base_url)
            if (
                _parsed_base.hostname or ""
            ).lower() == "generativelanguage.googleapis.com" and _parsed_base.path.rstrip(
                "/"
            ) == "/v1beta/openai":
                self.base_url = self.base_url[: -len("/openai")]
        self.api_key = api_key
        self._timeout = httpx.Timeout(timeout, connect = 10.0)
        # Generous read timeout: reasoning models pause tens of seconds between bytes.
        self._stream_timeout = httpx.Timeout(timeout, connect = 10.0, read = 300.0)
        # Some OpenAI/Azure deployments expose /responses but reject its optional context_management field. The
        # client lives for the whole Studio tool loop, so remember that capability result instead of paying for the
        # same guaranteed 400 on every provider turn.
        self._responses_compaction_rejected = False

    def _auth_headers(self) -> dict[str, str]:
        """Build authentication headers using the provider's registry config."""
        from core.inference.providers import get_provider_info

        provider_info = get_provider_info(self.provider_type) or {}
        auth_header = provider_info.get("auth_header", "Authorization")
        auth_prefix = provider_info.get("auth_prefix", "Bearer ")

        if self.provider_type == "gemini":
            _host = (urlparse(self.base_url).hostname or "").lower()
            if _host != "generativelanguage.googleapis.com":
                auth_header = "Authorization"
                auth_prefix = "Bearer "

        headers = {"Content-Type": "application/json"}
        # Azure: `api-key` or Entra Bearer token; raw JWTs and `Bearer <token>` select bearer.
        # Never send both credentials.
        azure_custom_responses = (
            self.provider_type == "custom"
            and self.api_type == "responses"
            and _is_azure_openai_host((urlparse(self.base_url).hostname or "").lower())
        )
        if azure_custom_responses and self.api_key:
            if self.api_key[:7].lower() == "bearer ":
                bearer_token = self.api_key[7:].strip()
            elif re.fullmatch(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", self.api_key):
                bearer_token = self.api_key
            else:
                bearer_token = None
            if bearer_token:
                headers["Authorization"] = f"Bearer {bearer_token}"
            elif bearer_token is None:
                headers["api-key"] = self.api_key
        elif self.api_key:
            headers[auth_header] = f"{auth_prefix}{self.api_key}"
        headers.update(provider_info.get("extra_headers", {}))
        return headers

    def _is_openai_compatible(self) -> bool:
        """Return False for providers needing request/response translation (e.g. Anthropic)."""
        from core.inference.providers import get_provider_info

        info = get_provider_info(self.provider_type) or {}
        if self.provider_type == "gemini":
            _host = (urlparse(self.base_url).hostname or "").lower()
            if _host != "generativelanguage.googleapis.com":
                return True
        return info.get("openai_compatible", True)

    async def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.95,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        enabled_tools: Optional[list[str]] = None,
        enable_prompt_caching: Optional[Union[bool, str]] = None,
        openai_code_exec_container_id: Optional[str] = None,
        anthropic_code_exec_container_id: Optional[str] = None,
        prompt_cache_ttl: Optional[str] = None,
        compaction_threshold: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        fast_mode: Optional[bool] = None,
        continue_final_message: Optional[bool] = None,
        response_format: Optional[dict[str, Any]] = None,
        stream: bool = True,
        preserve_thinking: Optional[bool] = None,
        thread_id: Optional[str] = None,
        compaction_fallback: Optional[CompactionFallback] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield OpenAI-format SSE lines from the external provider. OpenAI-compatible providers
        forward lines verbatim; for Anthropic the native Messages API SSE is translated.
        ``top_k``, ``min_p``, ``repetition_penalty`` and ``presence_penalty`` are opt-in, forwarded
        only when supplied, since the frontend's capability map already filters them per provider.
        ``fast_mode`` only applies to Anthropic Opus 5 / Opus 4.8 (silently dropped elsewhere); it
        adds the beta header and ``speed: "fast"``."""
        tool_choice_disabled = (
            isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"
        )

        managed_custom_responses = (
            self.provider_type == "custom"
            and self.api_type == "responses"
            and _is_openai_family_cloud(self.base_url)
        )
        if self.provider_type in _TEMPLATE_APPLYING_PROVIDERS and not managed_custom_responses:
            from core.inference.chat_template_helpers import (
                neutralize_control_markup_in_messages,
                neutralize_tool_descriptions,
                reconciled_tool_choice,
            )
            messages = neutralize_control_markup_in_messages(messages)
            if tools:
                safe_tools = neutralize_tool_descriptions(tools)
                tool_choice = reconciled_tool_choice(tool_choice, tools, safe_tools)
                if not safe_tools:
                    tool_choice = None
                tools = safe_tools

        if not self._is_openai_compatible():
            if self.provider_type == "gemini":
                async for line in self._stream_gemini(
                    messages,
                    model,
                    temperature,
                    top_p,
                    max_tokens,
                    top_k,
                    presence_penalty,
                    enabled_tools,
                    enable_prompt_caching,
                    enable_thinking,
                    reasoning_effort,
                    tools,
                    tool_choice,
                    response_format,
                ):
                    yield line
                return
            async for line in self._stream_anthropic(
                messages,
                model,
                temperature,
                top_p,
                max_tokens,
                top_k,
                enable_thinking,
                reasoning_effort,
                enabled_tools,
                enable_prompt_caching,
                anthropic_code_exec_container_id,
                prompt_cache_ttl,
                compaction_threshold,
                tool_choice,
                fast_mode = fast_mode,
                tools = tools,
            ):
                yield line
            return

        # gpt-5.x 404 on /v1/chat/completions, so all OpenAI traffic goes via /v1/responses.
        if self.provider_type == "openai" or self.api_type == "responses":
            async for line in self._stream_openai_responses(
                messages,
                model,
                temperature,
                top_p,
                max_tokens,
                enable_thinking,
                reasoning_effort,
                enabled_tools,
                enable_prompt_caching,
                openai_code_exec_container_id,
                compaction_threshold,
                tools,
                tool_choice,
                response_format,
                stream = stream if self.provider_type == "custom" else True,
                compaction_fallback = compaction_fallback,
            ):
                yield line
            return

        _kimi_tool_choice_forced_function = (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
            and bool(tool_choice["function"].get("name"))
        )
        if (
            self.provider_type == "kimi"
            and not tool_choice_disabled
            and not _kimi_tool_choice_forced_function
            and enabled_tools
            and "web_search" in enabled_tools
        ):
            async for line in self._stream_kimi_web_search(
                messages,
                model,
                max_tokens,
            ):
                yield line
            return

        # Both set: servers reject continuing while a generation prompt is asked for. Strict
        # endpoints 400 on unknown fields, so only the two documented providers get it.
        _continue_body = (
            {"continue_final_message": True, "add_generation_prompt": False}
            if continue_final_message and self.provider_type in _CONTINUATION_FLAG_PROVIDERS
            else {}
        )

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            **_continue_body,
        }
        if top_p is not None:
            body["top_p"] = top_p
        if stream and self.provider_type in _USAGE_STREAM_OPTION_PROVIDERS:
            body["stream_options"] = {"include_usage": True}
        if max_tokens is not None:
            if self.provider_type == "openai":
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens
        if top_k is not None:
            body["top_k"] = top_k
        if min_p is not None:
            body["min_p"] = min_p
        if repetition_penalty is not None:
            body[_REPETITION_PENALTY_BODY_KEY.get(self.provider_type, "repetition_penalty")] = (
                repetition_penalty
            )

        from core.inference.providers import get_provider_info

        provider_info = get_provider_info(self.provider_type) or {}
        for field in provider_info.get("body_omit", ()):
            body.pop(field, None)

        if self.provider_type == "llama_cpp" and preserve_thinking is not None:
            body["chat_template_kwargs"] = {"preserve_thinking": preserve_thinking}

        if self.provider_type == "kimi" and enable_thinking is not None:
            if model == "kimi-k2-thinking":
                pass
            elif enable_thinking:
                body["thinking"] = {"type": "enabled", "keep": "all"}
            else:
                body["thinking"] = {"type": "disabled"}
        elif self.provider_type == "mistral":
            _apply_mistral_reasoning_controls(body, model, enable_thinking, reasoning_effort)
        elif self.provider_type == "deepseek":
            _apply_deepseek_reasoning_controls(body, enable_thinking, reasoning_effort)
        elif self.provider_type == "qwen":
            _apply_qwen_reasoning_controls(body, enable_thinking)
        elif self.provider_type == "huggingface":
            _apply_passthrough_reasoning_effort(body, enable_thinking, reasoning_effort)
        elif provider_info.get("supports_chat_template_kwargs"):
            # chat_template_kwargs is opt-in per registry entry (strict gateways 400 on it); vLLM <=0.16
            # 400s on reasoning_effort=none, so off rides on the kwarg alone.
            effort = (reasoning_effort or "").strip().lower()
            effort = _LOCAL_SERVER_EFFORT_ALIASES.get(effort, effort)
            thinking = False if effort == "none" else enable_thinking
            if thinking is not None:
                tpl_kw = body.get("chat_template_kwargs")
                if not isinstance(tpl_kw, dict):
                    tpl_kw = {}
                tpl_kw["enable_thinking"] = bool(thinking)
                body["chat_template_kwargs"] = tpl_kw
            if effort in ("low", "medium", "high"):
                body["reasoning_effort"] = effort
        elif self.provider_type == "ollama":
            _apply_ollama_reasoning_controls(body, enable_thinking, reasoning_effort)
        elif self.provider_type == "lemonade":
            _apply_fastflowlm_reasoning_controls(body, enable_thinking, reasoning_effort)
        elif self.provider_type == "custom":
            _apply_custom_reasoning_controls(
                body, self.reasoning_config, enable_thinking, reasoning_effort
            )

        if self.provider_type == "openrouter":
            normalized_or_model = model.strip().lower()
            if reasoning_effort in _OPENROUTER_REASONING_EFFORTS:
                body["reasoning"] = {"effort": reasoning_effort}
            elif reasoning_effort == "none" or enable_thinking is False:
                if normalized_or_model in _OPENROUTER_MANDATORY_REASONING_MODELS:
                    body.pop("reasoning", None)
                else:
                    body["reasoning"] = {"enabled": False}
            elif enable_thinking is True:
                body["reasoning"] = {"enabled": True}

            # Claude caches only behind cache_control; the top-level form moves the breakpoint each turn.
            if caches_at_the_last_block("openrouter", model, enable_prompt_caching):
                cache_control = {"type": "ephemeral"}
                if prompt_cache_ttl == "1h":
                    cache_control["ttl"] = "1h"
                body["cache_control"] = cache_control
            if thread_id:
                body["session_id"] = str(thread_id)[:256]

            _or_tool_choice_forced_function = (
                isinstance(tool_choice, dict)
                and tool_choice.get("type") == "function"
                and isinstance(tool_choice.get("function"), dict)
                and bool(tool_choice["function"].get("name"))
            )
            if (
                not tool_choice_disabled
                and not _or_tool_choice_forced_function
                and enabled_tools
                and "web_search" in enabled_tools
            ):
                plugins = list(body.get("plugins") or [])
                if not any(isinstance(p, dict) and p.get("id") == "web" for p in plugins):
                    plugins.append({"id": "web"})
                body["plugins"] = plugins
                logger.info(
                    "OpenRouter web_search: attached plugins=[{id: 'web'}] (model=%s)",
                    body.get("model"),
                )

        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format

        url = f"{self.base_url}/chat/completions"
        logger.info(
            "Proxying chat completion to %s (provider=%s, model=%s)",
            url,
            self.provider_type,
            model,
        )

        try:
            async with _stream_post_retrying_max_tokens(
                _loopback_http_client if getattr(self, "_managed_loopback", False) else _client(),
                url,
                body,
                headers = self._auth_headers(),
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    error_text = _friendly_provider_error_text(
                        self.provider_type,
                        response.status_code,
                        error_text,
                        model = model,
                    )
                    logger.error(
                        "External provider returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    yield _error_sse_line(
                        response.status_code,
                        error_text,
                        self.provider_type,
                        response.headers.get("Retry-After"),
                    )
                    return

                # Manual __anext__ so the response closes BEFORE lines_gen: avoids httpcore 1.0
                # GeneratorExit -> RuntimeError on Python 3.13.
                from .http_stream import closing_response_lines

                lines_gen = closing_response_lines(response)
                event_counts: dict[str, int] = {}
                chosen_model: Optional[str] = None
                web_search_active = (
                    self.provider_type == "openrouter"
                    and not tool_choice_disabled
                    and not _or_tool_choice_forced_function
                    and bool(enabled_tools)
                    and "web_search" in (enabled_tools or [])
                )
                web_search_tool_id = "openrouter_web_search"
                web_search_citations: list[dict[str, str]] = []
                web_search_tool_started = False
                web_search_tool_ended = False

                def _emit_synthetic_tool_event(payload: dict[str, Any]) -> str:
                    _stamp_server_tool_marker(payload)
                    chunk = {
                        "id": f"chatcmpl-{self.provider_type}-synthetic",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": None,
                            }
                        ],
                        "_toolEvent": payload,
                    }
                    return f"data: {_json.dumps(chunk)}"

                def _record_or_url_citation(payload: Any) -> None:
                    if not isinstance(payload, dict):
                        return
                    if payload.get("type") != "url_citation":
                        return
                    cit = payload.get("url_citation")
                    if not isinstance(cit, dict):
                        cit = payload
                    url = cit.get("url", "") if isinstance(cit, dict) else ""
                    if not url or not isinstance(url, str):
                        return
                    if any(c["url"] == url for c in web_search_citations):
                        return
                    title = cit.get("title") or url
                    snippet = cit.get("content") or cit.get("snippet") or ""
                    web_search_citations.append(
                        {
                            "url": url,
                            "title": title,
                            "snippet": snippet if isinstance(snippet, str) else "",
                        }
                    )

                def _build_web_search_tool_end() -> str:
                    blocks: list[str] = []
                    for cit in web_search_citations:
                        line = f"Title: {cit['title']}\nURL: {cit['url']}"
                        if cit.get("snippet"):
                            line += f"\nSnippet: {cit['snippet']}"
                        blocks.append(line)
                    return _emit_synthetic_tool_event(
                        {
                            "type": "tool_end",
                            "tool_call_id": web_search_tool_id,
                            "result": ("\n---\n".join(blocks) if blocks else "(search complete)"),
                        }
                    )

                if web_search_active:
                    yield _emit_synthetic_tool_event(
                        {
                            "type": "tool_start",
                            "tool_name": "web_search",
                            "tool_call_id": web_search_tool_id,
                            "arguments": {},
                        }
                    )
                    web_search_tool_started = True

                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip():
                            continue
                        if line.startswith("data:"):
                            data_str = line[len("data:") :].strip()
                            if data_str == "[DONE]":
                                event_counts["done"] = event_counts.get("done", 0) + 1
                                if (
                                    web_search_active
                                    and web_search_tool_started
                                    and not web_search_tool_ended
                                ):
                                    yield _build_web_search_tool_end()
                                    web_search_tool_ended = True
                            elif data_str:
                                try:
                                    parsed = _json.loads(data_str)
                                except Exception:
                                    parsed = None
                                if isinstance(parsed, dict):
                                    if "error" in parsed:
                                        event_counts["error"] = event_counts.get("error", 0) + 1
                                        logger.warning(
                                            "%s SSE error event: %s",
                                            self.provider_type,
                                            parsed.get("error"),
                                        )
                                    else:
                                        event_counts["delta"] = event_counts.get("delta", 0) + 1
                                    if chosen_model is None and isinstance(
                                        parsed.get("model"), str
                                    ):
                                        chosen_model = parsed["model"]
                                    if web_search_active:
                                        choices = parsed.get("choices") or []
                                        if isinstance(choices, list):
                                            for choice in choices:
                                                if not isinstance(choice, dict):
                                                    continue
                                                for envelope in (
                                                    choice.get("delta"),
                                                    choice.get("message"),
                                                ):
                                                    if not isinstance(envelope, dict):
                                                        continue
                                                    for ann in envelope.get("annotations") or []:
                                                        _record_or_url_citation(ann)
                        if self.provider_type == "lemonade":
                            if line.startswith("{"):
                                line = _bare_json_error_as_sse(line) or line
                            else:
                                line = _with_fastflowlm_timings(line)
                        # Strip Unsloth's UI control frames so an echoing endpoint cannot forge a tool card.
                        relayed = sanitize_provider_sse_line(line)
                        if relayed is None:
                            continue
                        yield relayed
                    if web_search_active and web_search_tool_started and not web_search_tool_ended:
                        yield _build_web_search_tool_end()
                        web_search_tool_ended = True
                except GeneratorExit:
                    await response.aclose()
                    await lines_gen.aclose()
                    raise
                finally:
                    logger.info(
                        "%s stream complete (model=%s, chosen=%s, "
                        "web_search_requested=%s, citations=%s, events=%s)",
                        self.provider_type,
                        model,
                        chosen_model,
                        web_search_active,
                        len(web_search_citations),
                        event_counts,
                    )
                    await response.aclose()
                    await lines_gen.aclose()

        except httpx.ConnectError as exc:
            logger.error("Connection error to %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Failed to connect to {self.provider_type}: {exc}",
                self.provider_type,
            )
        except httpx.ReadTimeout as exc:
            logger.error("Read timeout from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                504,
                f"Timeout waiting for {self.provider_type} response",
                self.provider_type,
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP error from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Error communicating with {self.provider_type}: {exc}",
                self.provider_type,
            )

    async def _stream_kimi_web_search(
        self, messages: list[dict[str, Any]], model: str, max_tokens: Optional[int]
    ) -> AsyncGenerator[str, None]:
        """Kimi $web_search round-trip, per https://platform.kimi.ai/docs/guide/use-web-search: POST
        messages with tools=[{type: "builtin_function", function: {name: "$web_search"}}] and
        thinking=disabled; stream the first response, accumulating function.arguments across
        tool_call deltas until finish_reason="tool_calls" and forwarding none of those chunks to the
        client (internal protocol step, not user-visible output); build a second request of the
        original messages plus the assistant message carrying the tool_calls plus a role=tool
        message echoing the same arguments verbatim (the server actually runs the search); stream
        the second response, which is the final answer with search results incorporated.

        We synthesize tool_start (with the parsed query) when the first call completes, and tool_end
        (with any url_citation annotations the second stream emits) before [DONE], so the chat UI
        shows the same web-search tool card as other providers.
        """
        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "thinking": {"type": "disabled"},
            "tools": [{"type": "builtin_function", "function": {"name": "$web_search"}}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        from core.inference.providers import get_provider_info

        provider_info = get_provider_info(self.provider_type) or {}
        for field in provider_info.get("body_omit", ()):
            body.pop(field, None)

        tool_call_id = "kimi_web_search"
        synthetic_id = f"chatcmpl-{self.provider_type}-synthetic"

        def _synthetic_chunk(payload: dict[str, Any]) -> str:
            _stamp_server_tool_marker(payload)
            chunk = {
                "id": synthetic_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "_toolEvent": payload,
            }
            return f"data: {_json.dumps(chunk)}"

        logger.info(
            "Kimi $web_search round-trip starting (model=%s, url=%s)",
            model,
            url,
        )

        tool_calls_acc: dict[int, dict[str, Any]] = {}
        try:
            async with _client().stream(
                "POST",
                url,
                json = body,
                headers = self._auth_headers(),
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    logger.error(
                        "Kimi first-call returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    yield _error_sse_line(
                        response.status_code,
                        error_text,
                        self.provider_type,
                        response.headers.get("Retry-After"),
                    )
                    return

                lines_gen = response.aiter_lines().__aiter__()
                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip() or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            parsed = _json.loads(data_str)
                        except Exception:
                            continue
                        for choice in parsed.get("choices") or []:
                            if not isinstance(choice, dict):
                                continue
                            delta = choice.get("delta") or {}
                            for tc in delta.get("tool_calls") or []:
                                if not isinstance(tc, dict):
                                    continue
                                idx = tc.get("index", 0)
                                slot = tool_calls_acc.setdefault(
                                    idx,
                                    {
                                        "id": tc.get("id") or f"call_{idx}",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if tc.get("id"):
                                    slot["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot["function"]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    slot["function"]["arguments"] += fn["arguments"]
                            if choice.get("finish_reason") == "tool_calls":
                                break
                except GeneratorExit:
                    await response.aclose()
                    await lines_gen.aclose()
                    raise
                finally:
                    await response.aclose()
                    await lines_gen.aclose()
        except httpx.HTTPError as exc:
            logger.error("Kimi first-call HTTP error: %s", exc)
            yield _error_sse_line(
                502,
                f"Error communicating with kimi: {exc}",
                self.provider_type,
            )
            return

        search_calls = [
            tc for tc in tool_calls_acc.values() if tc["function"]["name"] == "$web_search"
        ]
        if not search_calls:
            logger.info(
                "Kimi $web_search: model did not invoke search; falling back to plain stream"
            )
            fallback_body = dict(body)
            fallback_body.pop("tools", None)
            fallback_body["stream_options"] = {"include_usage": True}
            try:
                async with _client().stream(
                    "POST",
                    url,
                    json = fallback_body,
                    headers = self._auth_headers(),
                    timeout = self._stream_timeout,
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_text = error_body.decode("utf-8", errors = "replace")
                        logger.error(
                            "Kimi fallback returned %d: %s",
                            response.status_code,
                            error_text[:500],
                        )
                        yield _error_sse_line(
                            response.status_code,
                            error_text,
                            self.provider_type,
                            response.headers.get("Retry-After"),
                        )
                        return
                    lines_gen = response.aiter_lines().__aiter__()
                    try:
                        while True:
                            try:
                                line = await lines_gen.__anext__()
                            except StopAsyncIteration:
                                break
                            if line.strip():
                                relayed = sanitize_provider_sse_line(line)
                                if relayed is not None:
                                    yield relayed
                    except GeneratorExit:
                        await response.aclose()
                        await lines_gen.aclose()
                        raise
                    finally:
                        await response.aclose()
                        await lines_gen.aclose()
            except httpx.HTTPError as exc:
                logger.error("Kimi fallback HTTP error: %s", exc)
                yield _error_sse_line(
                    502,
                    f"Error communicating with kimi: {exc}",
                    self.provider_type,
                )
            return

        first_args_raw = search_calls[0]["function"]["arguments"] or "{}"
        try:
            first_args = _json.loads(first_args_raw)
        except Exception:
            first_args = {}
        logger.info(
            "Kimi $web_search: %d tool_call(s), args[0]=%s",
            len(search_calls),
            first_args_raw[:500],
        )
        first_args_search_tokens: Optional[int] = None
        if isinstance(first_args, dict):
            usage_block = first_args.get("usage")
            if isinstance(usage_block, dict):
                tok = usage_block.get("total_tokens")
                if isinstance(tok, int):
                    first_args_search_tokens = tok
        yield _synthetic_chunk(
            {
                "type": "tool_start",
                "tool_name": "web_search",
                "tool_call_id": tool_call_id,
                "arguments": first_args if isinstance(first_args, dict) else {},
            }
        )
        yield _build_kimi_tool_end(_synthetic_chunk, tool_call_id, [])

        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": list(tool_calls_acc.values()),
        }
        tool_msgs = [
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["function"]["name"],
                "content": tc["function"]["arguments"],
            }
            for tc in tool_calls_acc.values()
        ]
        followup_body = dict(body)
        followup_body["messages"] = list(messages) + [assistant_msg] + tool_msgs
        followup_body["stream_options"] = {"include_usage": True}

        try:
            async with _client().stream(
                "POST",
                url,
                json = followup_body,
                headers = self._auth_headers(),
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    logger.error(
                        "Kimi second-call returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    yield _error_sse_line(
                        response.status_code,
                        error_text,
                        self.provider_type,
                        response.headers.get("Retry-After"),
                    )
                    return

                lines_gen = response.aiter_lines().__aiter__()
                last_usage: Optional[dict[str, Any]] = None
                annotation_shapes: set[str] = set()
                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip():
                            continue
                        if line.startswith("data:"):
                            data_str = line[len("data:") :].strip()
                            if data_str and data_str != "[DONE]":
                                try:
                                    parsed = _json.loads(data_str)
                                except Exception:
                                    parsed = None
                                if isinstance(parsed, dict):
                                    usage = parsed.get("usage")
                                    if isinstance(usage, dict):
                                        last_usage = usage
                                    for choice in parsed.get("choices") or []:
                                        if not isinstance(choice, dict):
                                            continue
                                        for envelope in (
                                            choice.get("delta"),
                                            choice.get("message"),
                                        ):
                                            if not isinstance(envelope, dict):
                                                continue
                                            for ann in envelope.get("annotations") or []:
                                                if isinstance(ann, dict):
                                                    annotation_shapes.add(
                                                        str(ann.get("type") or "?")
                                                    )
                        relayed = sanitize_provider_sse_line(line)
                        if relayed is None:
                            continue
                        yield relayed
                except GeneratorExit:
                    await response.aclose()
                    await lines_gen.aclose()
                    raise
                finally:
                    logger.info(
                        "Kimi $web_search complete (model=%s, "
                        "search_ctx_tokens=%s, annotation_types=%s, "
                        "prompt_tokens=%s, completion_tokens=%s)",
                        model,
                        first_args_search_tokens,
                        sorted(annotation_shapes) or None,
                        (last_usage or {}).get("prompt_tokens"),
                        (last_usage or {}).get("completion_tokens"),
                    )
                    await response.aclose()
                    await lines_gen.aclose()
        except httpx.HTTPError as exc:
            logger.error("Kimi second-call HTTP error: %s", exc)
            yield _error_sse_line(
                502,
                f"Error communicating with kimi: {exc}",
                self.provider_type,
            )

    async def _stream_anthropic(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        top_k: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        enabled_tools: Optional[list[str]] = None,
        enable_prompt_caching: Optional[bool] = None,
        anthropic_code_exec_container_id: Optional[str] = None,
        prompt_cache_ttl: Optional[str] = None,
        compaction_threshold: Optional[int] = None,
        tool_choice: Optional[Any] = None,
        *,
        fast_mode: Optional[bool] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """call Anthropic Messages and translate its SSE into OpenAI stream events."""
        import json as _json

        system: Optional[str] = None
        filtered: list[dict[str, Any]] = []
        compaction_replayed = False
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                system = (
                    content
                    if isinstance(content, str)
                    else "\n".join(p["text"] for p in content if p.get("type") == "text")
                )
                continue

            content = msg.get("content")
            extra = msg.get("extra_content") or {}
            native_content = (extra.get("anthropic") or {}).get("content")
            if msg.get("role") == "assistant" and isinstance(native_content, list):
                # replay signed native blocks because display text has synthetic <think> markers.
                content = []
            # role=tool with list content must also become a tool_result, else Anthropic rejects it.
            if msg.get("role") == "tool":
                _tr_id = msg.get("tool_call_id") or ""
                if isinstance(content, list):
                    _flat_parts: list[str] = []
                    _result_images: list[dict[str, Any]] = []
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "text"
                            and part.get("text")
                        ):
                            _flat_parts.append(str(part["text"]))
                        elif isinstance(part, dict) and part.get("type") == "image_url":
                            image_url = part.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                header, _, data = image_url.partition(",")
                                source = {
                                    "type": "base64",
                                    "media_type": header[5:].split(";")[0],
                                    "data": data,
                                }
                            else:
                                source = {"type": "url", "url": image_url}
                            _result_images.append({"type": "image", "source": source})
                    _flat_result = "".join(_flat_parts)
                    if _result_images:
                        _flat_result = (
                            [{"type": "text", "text": _flat_result}] if _flat_result else []
                        ) + _result_images
                elif content is None:
                    _flat_result = ""
                elif isinstance(content, str):
                    _flat_result = content
                else:
                    _flat_result = _json.dumps(content)
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": _tr_id,
                    "content": _flat_result,
                }
                if (
                    filtered
                    and filtered[-1]["role"] == "user"
                    and isinstance(filtered[-1]["content"], list)
                    and all(part.get("type") == "tool_result" for part in filtered[-1]["content"])
                ):
                    filtered[-1]["content"].append(result_block)
                else:
                    filtered.append({"role": "user", "content": [result_block]})
                continue
            if isinstance(content, list):
                # preserve Unsloth input_document support when translating to Anthropic blocks.
                anthropic_parts: list[dict[str, Any]] = (
                    [
                        block
                        for block in native_content
                        if (
                            block.get("type") != "text"
                            or _anthropic_text_is_sendable(block.get("text"))
                        )
                        and (
                            block.get("type") != "compaction"
                            or _anthropic_supports_compaction(model)
                        )
                    ]
                    if msg.get("role") == "assistant" and isinstance(native_content, list)
                    else []
                )
                if any(part.get("type") == "compaction" for part in anthropic_parts):
                    compaction_replayed = True
                for part in content:
                    if part.get("type") == "text" and _anthropic_text_is_sendable(part.get("text")):
                        anthropic_parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "compaction":
                        # replay compaction here to avoid compacting history twice.
                        summary = part.get("content") or ""
                        if (
                            isinstance(summary, str)
                            and summary
                            and _anthropic_supports_compaction(model)
                        ):
                            compaction = {"type": "compaction", "content": summary}
                            encrypted = part.get("encrypted_content")
                            if isinstance(encrypted, str) and encrypted:
                                compaction["encrypted_content"] = encrypted
                            anthropic_parts.append(compaction)
                            compaction_replayed = True
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            header, _, b64data = url.partition(",")
                            media_type = header.split(";")[0].replace("data:", "") or "image/jpeg"
                            anthropic_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64data,
                                    },
                                }
                            )
                        else:
                            anthropic_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url,
                                    },
                                }
                            )
                    elif part.get("type") == "input_document":
                        url = part.get("file_url") or ""
                        data_uri = part.get("file_data") or ""
                        title = part.get("filename")
                        data_uri_valid = False
                        b64data = ""
                        header = ""
                        if data_uri.startswith("data:"):
                            header, _, b64data = data_uri.partition(",")
                            data_uri_valid = bool(b64data.strip())
                        if data_uri_valid:
                            media_type = (
                                part.get("media_type")
                                or header.split(";")[0].replace("data:", "")
                                or "application/pdf"
                            )
                            doc_block: dict[str, Any] = {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64data,
                                },
                                "citations": {"enabled": True},
                            }
                            if title:
                                doc_block["title"] = title
                            anthropic_parts.append(doc_block)
                        elif url:
                            doc_block = {
                                "type": "document",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                },
                                "citations": {"enabled": True},
                            }
                            if title:
                                doc_block["title"] = title
                            anthropic_parts.append(doc_block)
                native_calls = {
                    block.get("id"): i
                    for i, block in enumerate(anthropic_parts)
                    if block.get("type") == "tool_use"
                }
                if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
                    for _tc in msg["tool_calls"]:
                        if not isinstance(_tc, dict):
                            continue
                        _fn = _tc.get("function") or {}
                        if not isinstance(_fn, dict) or not _fn.get("name"):
                            continue
                        _raw = _fn.get("arguments") or "{}"
                        try:
                            _input = _json.loads(_raw) if isinstance(_raw, str) else _raw
                        except Exception:
                            _input = {"_raw": _raw}
                        if not isinstance(_input, dict):
                            _input = {"value": _input}
                        _tool_use = {
                            "type": "tool_use",
                            "id": _tc.get("id") or f"toolu_{time.time_ns()}",
                            "name": _fn["name"],
                            "input": _input,
                        }
                        _slot = native_calls.pop(_tool_use["id"], None)
                        if _slot is None:
                            anthropic_parts.append(_tool_use)
                        else:
                            anthropic_parts[_slot] = _tool_use
                if native_calls:
                    _withheld = set(native_calls.values())
                    anthropic_parts = [
                        part for i, part in enumerate(anthropic_parts) if i not in _withheld
                    ]
                if anthropic_parts:
                    filtered.append({"role": msg["role"], "content": anthropic_parts})
            else:
                if msg.get("role") == "tool":
                    _tr_id = msg.get("tool_call_id") or ""
                    _tr_content = msg.get("content")
                    if _tr_content is None:
                        _tr_content = ""
                    filtered.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": _tr_id,
                                    "content": (
                                        _tr_content
                                        if isinstance(_tr_content, str)
                                        else _json.dumps(_tr_content)
                                    ),
                                }
                            ],
                        }
                    )
                    continue
                if (
                    msg.get("role") == "assistant"
                    and isinstance(msg.get("tool_calls"), list)
                    and msg["tool_calls"]
                ):
                    _text_content = msg.get("content")
                    _blocks: list[dict[str, Any]] = []
                    if isinstance(_text_content, str) and _anthropic_text_is_sendable(
                        _text_content
                    ):
                        _blocks.append({"type": "text", "text": _text_content})
                    for _tc in msg["tool_calls"]:
                        if not isinstance(_tc, dict):
                            continue
                        _fn = _tc.get("function") or {}
                        if not isinstance(_fn, dict) or not _fn.get("name"):
                            continue
                        _raw = _fn.get("arguments") or "{}"
                        try:
                            _input = _json.loads(_raw) if isinstance(_raw, str) else _raw
                        except Exception:
                            _input = {"_raw": _raw}
                        if not isinstance(_input, dict):
                            _input = {"value": _input}
                        _blocks.append(
                            {
                                "type": "tool_use",
                                "id": _tc.get("id") or f"toolu_{time.time_ns()}",
                                "name": _fn["name"],
                                "input": _input,
                            }
                        )
                    if _blocks:
                        filtered.append({"role": "assistant", "content": _blocks})
                    continue
                # A plain string is one text block, so an empty one 400s too.
                if isinstance(content, str) and not content.strip():
                    continue
                filtered.append(msg)

        pending_server_calls: dict[str, str] = {}
        for message in filtered:
            if message["role"] != "assistant" or not isinstance(message["content"], list):
                continue
            for block in message["content"]:
                if block.get("type") == "server_tool_use":
                    pending_server_calls[block["id"]] = block["name"]
                elif block.get("tool_use_id"):
                    pending_server_calls.pop(block["tool_use_id"], None)
        pending_hosted_tools = set(pending_server_calls.values())
        if pending_hosted_tools & {"bash_code_execution", "text_editor_code_execution"}:
            pending_hosted_tools.add("code_execution")

        # Newer Claude models 400 on temperature/top_p/top_k, including the thinking override.
        sampling_removed = _anthropic_sampling_params_removed(model)

        body: dict[str, Any] = {
            "model": model,
            "messages": filtered,
            "max_tokens": max_tokens or 1024,
            "stream": True,
        }
        if not sampling_removed:
            body["temperature"] = temperature
        if top_k is not None and top_k > 0 and not sampling_removed:
            body["top_k"] = top_k
        prompt_caching_enabled = enable_prompt_caching is not False
        # 1h TTL: 2x write vs 1.25x, reads 0.1x for both, so 1h wins after one extra hit.
        cache_marker: dict[str, Any] = {"type": "ephemeral"}
        if prompt_cache_ttl in ("5m", "1h"):
            cache_marker["ttl"] = prompt_cache_ttl

        if system:
            if prompt_caching_enabled:
                body["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": dict(cache_marker),
                    }
                ]
            else:
                body["system"] = system

        if prompt_caching_enabled and filtered:
            # Second breakpoint on the tail covers system prompts under the ~1024-token cache floor (max 4).
            last_msg = filtered[-1]
            content = last_msg.get("content")
            if isinstance(content, str):
                last_msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": dict(cache_marker),
                    }
                ]
            elif isinstance(content, list) and content:
                head = list(content[:-1])
                tail = content[-1]
                if isinstance(tail, dict):
                    head.append({**tail, "cache_control": dict(cache_marker)})
                else:
                    head.append(tail)
                last_msg["content"] = head
        thinking_spec = _anthropic_thinking_spec(model)
        allowed_efforts = (
            thinking_spec.efforts
            if thinking_spec
            else ("none", "low", "medium", "high", "xhigh", "max")
        )
        effort = reasoning_effort if reasoning_effort in allowed_efforts else None
        if effort == "xhigh" and model.strip().lower().startswith(
            ("claude-opus-4-6", "claude-sonnet-4-6")
        ):
            effort = "max"
        if effort is None:
            if enable_thinking is False:
                effort = "none"
            elif enable_thinking is True:
                effort = "medium"
        # Default-on thinking needs explicit disable; only valid at effort <= high, so send it alone.
        if (
            effort == "none"
            and thinking_spec
            and thinking_spec.thinking_default_on
            and thinking_spec.can_disable
        ):
            body["thinking"] = {"type": "disabled"}
        if effort and effort != "none":
            # Anthropic rejects top_k whenever thinking is enabled.
            body.pop("top_k", None)
            # 4.5/4.6 need temperature=1 with thinking and forbid top_p; 4.7 removed temperature.
            if not sampling_removed:
                body["temperature"] = 1
            body.pop("top_p", None)
            adaptive = (thinking_spec is not None and thinking_spec.kind == "adaptive") or (
                thinking_spec is None and _anthropic_model_newer_than_specs(model)
            )
            if adaptive:
                # display=summarized: defaults to omitted on Opus 4.7 (blank panel). Claude <=4.5 rejects adaptive.
                body["thinking"] = {"type": "adaptive", "display": "summarized"}
                # Effort goes under output_config.effort; top-level 400s.
                body["output_config"] = {"effort": effort}
            elif thinking_spec and thinking_spec.kind == "manual":
                budget_tokens = {"low": 1024, "medium": 2048, "high": 4096}[effort]
                body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
                if body.get("max_tokens", 0) <= budget_tokens:
                    body["max_tokens"] = budget_tokens + 1024

        _anthropic_tool_choice_disabled = (
            isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"
        )
        _anthropic_tool_choice_forced_function = (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
            and bool(tool_choice["function"].get("name"))
        )
        _anthropic_hosted_builtins_allowed = (
            not _anthropic_tool_choice_disabled and not _anthropic_tool_choice_forced_function
        )
        if _anthropic_tool_choice_disabled and pending_hosted_tools:
            body["tool_choice"] = {"type": "none"}

        # Anthropic 400s a history holding tool blocks without `tools`, so a withdrawn catalog stays declared.
        history_has_tool_blocks = any(
            isinstance(message["content"], list)
            and any(
                block.get("type") in ("tool_use", "tool_result") for block in message["content"]
            )
            for message in filtered
        )
        if tools and (not _anthropic_tool_choice_disabled or history_has_tool_blocks):
            client_tools = []
            for tool in tools:
                if tool.get("type") != "function":
                    continue
                function = tool.get("function") or {}
                if not function.get("name"):
                    continue
                entry = {
                    "name": function["name"],
                    "input_schema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
                for key in ("description", "strict"):
                    if key in function:
                        entry[key] = function[key]
                client_tools.append(entry)
            if client_tools:
                body["tools"] = client_tools
                if _anthropic_tool_choice_disabled:
                    body["tool_choice"] = {"type": "none"}
                elif _anthropic_tool_choice_forced_function:
                    body["tool_choice"] = {"type": "tool", "name": tool_choice["function"]["name"]}
                elif tool_choice == "required":
                    body["tool_choice"] = {"type": "any"}
                else:
                    body["tool_choice"] = {"type": "auto"}
                if (
                    body["tool_choice"]["type"] in ("any", "tool")
                    and (body.get("thinking") or {}).get("type") == "enabled"
                ):
                    body.pop("thinking")

        client_tool_names = {tool["name"] for tool in body.get("tools", [])}

        if "web_search" in pending_hosted_tools or (
            _anthropic_hosted_builtins_allowed
            and enabled_tools
            and "web_search" in enabled_tools
            and "web_search" not in client_tool_names
        ):
            anthropic_tools = list(body.get("tools") or [])
            anthropic_tools.append(
                {
                    "type": _anthropic_web_search_version(model),
                    "name": "web_search",
                    "max_uses": 5,
                }
            )
            body["tools"] = anthropic_tools

        web_fetch_enabled = bool(
            "web_fetch" in pending_hosted_tools
            or (
                _anthropic_hosted_builtins_allowed
                and enabled_tools
                and "web_fetch" in enabled_tools
                and "web_fetch" not in client_tool_names
            )
        )
        if web_fetch_enabled:
            anthropic_tools = list(body.get("tools") or [])
            anthropic_tools.append(
                {
                    "type": _anthropic_web_fetch_version(model),
                    "name": "web_fetch",
                    "max_uses": 5,
                }
            )
            body["tools"] = anthropic_tools

        code_execution_enabled = bool(
            "code_execution" in pending_hosted_tools
            or (
                _anthropic_hosted_builtins_allowed
                and enabled_tools
                and "code_execution" in enabled_tools
                and "code_execution" not in client_tool_names
            )
        )
        if code_execution_enabled:
            anthropic_tools = list(body.get("tools") or [])
            anthropic_tools.append(
                {
                    "type": _anthropic_code_execution_version(model),
                    "name": "code_execution",
                }
            )
            body["tools"] = anthropic_tools
            # reuse the prior container for filesystem state; stale IDs emit container_invalidated.
            if anthropic_code_exec_container_id:
                body["container"] = anthropic_code_exec_container_id

        # clamp Anthropic compaction thresholds to the supported range to avoid upstream 400s.
        compaction_active = (
            compaction_threshold is not None
            and compaction_threshold > 0
            and _anthropic_supports_compaction(model)
        )
        if compaction_active and compaction_threshold is not None:
            trigger_value = min(
                max(int(compaction_threshold), _ANTHROPIC_COMPACTION_MIN),
                _SERVER_COMPACTION_MAX,
            )
            body["context_management"] = {
                "edits": [
                    {
                        "type": _ANTHROPIC_COMPACTION_TYPE,
                        "trigger": {
                            "type": "input_tokens",
                            "value": trigger_value,
                        },
                    }
                ]
            }

        # fast mode is limited to Opus 5/4.8 and conflicts with the Priority service tier.
        fast_mode_active = bool(fast_mode) and _anthropic_supports_fast_mode(model)
        if fast_mode_active:
            body["speed"] = "fast"

        url = f"{self.base_url}/messages"
        completion_id = f"chatcmpl-anthropic-{model.replace('/', '-')}"

        logger.info(
            "Anthropic request shape (model=%s, has_thinking=%s, thinking=%s, "
            "output_config=%s, temperature=%s, has_top_p=%s, has_top_k=%s, "
            "max_tokens=%s)",
            model,
            "thinking" in body,
            body.get("thinking"),
            body.get("output_config"),
            body.get("temperature"),
            "top_p" in body,
            "top_k" in body,
            body.get("max_tokens"),
        )

        _finish_reason_map: dict[str, Optional[str]] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "refusal": "content_filter",
            "model_context_window_exceeded": "length",
            "pause_turn": None,
        }

        logger.info("Proxying Anthropic Messages API to %s (model=%s)", url, model)

        request_headers = self._auth_headers()
        # preserve registry beta flags when adding request-specific flags.
        existing_beta = request_headers.get("anthropic-beta", "").strip()
        beta_parts = (
            [p.strip() for p in existing_beta.split(",") if p.strip()] if existing_beta else []
        )
        if code_execution_enabled and _ANTHROPIC_CODE_EXECUTION_BETA not in beta_parts:
            beta_parts.append(_ANTHROPIC_CODE_EXECUTION_BETA)
        if (
            compaction_active or compaction_replayed
        ) and _ANTHROPIC_COMPACTION_BETA not in beta_parts:
            beta_parts.append(_ANTHROPIC_COMPACTION_BETA)
        if fast_mode_active and _ANTHROPIC_FAST_MODE_BETA not in beta_parts:
            beta_parts.append(_ANTHROPIC_FAST_MODE_BETA)
        if beta_parts:
            request_headers["anthropic-beta"] = ",".join(beta_parts)

        try:
            async with _client().stream(
                "POST",
                url,
                json = body,
                headers = request_headers,
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    logger.error(
                        "Anthropic returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    if anthropic_code_exec_container_id and 400 <= response.status_code < 500:
                        lowered = error_text.lower()
                        if "container" in lowered and (
                            "expired" in lowered
                            or "not_found" in lowered
                            or "not found" in lowered
                            or "no such container" in lowered
                            or "invalid" in lowered
                        ):
                            yield (
                                f"data: "
                                f"{_json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': None}], '_toolEvent': {'type': 'container_invalidated'}})}"
                            )
                    yield _error_sse_line(
                        response.status_code,
                        error_text,
                        self.provider_type,
                        response.headers.get("Retry-After"),
                    )
                    return

                lines_gen = response.aiter_lines().__aiter__()
                thinking_open = False
                client_tool_indices: dict[int, int] = {}
                client_tool_chunks: list[str] = []
                replay_blocks: dict[int, dict[str, Any]] = {}
                replay_inputs: dict[int, str] = {}
                event_counts: dict[str, int] = {}
                current_server_tool_use: Optional[dict[str, Any]] = None
                current_result_block: Optional[dict[str, Any]] = None
                web_search_calls: dict[str, dict[str, Any]] = {}
                current_code_exec_use: Optional[dict[str, Any]] = None
                current_code_exec_result: Optional[dict[str, Any]] = None
                code_execution_calls: dict[str, dict[str, Any]] = {}
                current_web_fetch_use: Optional[dict[str, Any]] = None
                current_web_fetch_result: Optional[dict[str, Any]] = None
                web_fetch_calls: dict[str, dict[str, Any]] = {}
                current_compaction: Optional[dict[str, Any]] = None
                compaction_blocks_seen = 0
                document_citations: list[dict[str, Any]] = []
                code_execution_generated_files = 0
                latched_container_id: Optional[str] = None
                container_id_emitted = False
                last_usage: dict[str, Any] = {}

                def _content_chunk(text: str) -> str:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    return f"data: {_json.dumps(chunk)}"

                def _delta_chunk(delta: dict[str, Any]) -> str:
                    return "data: " + _json.dumps(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                    )

                def _emit_tool_event(payload: dict[str, Any]) -> str:
                    _stamp_server_tool_marker(payload)
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": None,
                            }
                        ],
                        "_toolEvent": payload,
                    }
                    return f"data: {_json.dumps(chunk)}"

                def _format_web_search_results(results: list[Any] | dict[str, Any]) -> str:
                    if isinstance(results, dict):
                        if results.get("type") == "web_search_tool_result_error":
                            return f"Error: {results.get('error_code') or 'unknown'}"
                        return ""
                    blocks: list[str] = []
                    for r in results:
                        if not isinstance(r, dict):
                            continue
                        if r.get("type") != "web_search_result":
                            continue
                        url = r.get("url", "")
                        title = r.get("title") or url
                        if not url:
                            continue
                        blocks.append(f"Title: {title}\nURL: {url}")
                    return "\n---\n".join(blocks)

                def _format_web_fetch_result(inner: dict[str, Any]) -> str:
                    """Render a `web_fetch_tool_result.content` payload as the Title / URL / snippet
                    block CodeExecutionToolUI and parseSourcesFromResult expect from the
                    web_search path. Success (text): {type: web_fetch_result, url, retrieved_at,
                    content: {type: document, source: {type: text, media_type, data}, title?}}.
                    Success (pdf): source.type=base64 + media_type=application/pdf, whose base64
                    bytes are not surfaced -- title + url is enough for the source pill and the
                    model still sees the document. Error: {type: web_fetch_tool_error,
                    error_code}."""
                    inner_type = inner.get("type") or ""
                    if inner_type == "web_fetch_tool_error":
                        return f"Error: {inner.get('error_code', 'unknown')}"
                    url = inner.get("url", "")
                    document = inner.get("content") or {}
                    title = ""
                    snippet = ""
                    if isinstance(document, dict):
                        title = document.get("title") or ""
                        source = document.get("source") or {}
                        if isinstance(source, dict):
                            media_type = source.get("media_type") or ""
                            data = source.get("data") or ""
                            if media_type.startswith("text/") and isinstance(data, str) and data:
                                snippet = data[:240].strip()
                    if not title and url:
                        title = url
                    parts: list[str] = []
                    if title:
                        parts.append(f"Title: {title}")
                    if url:
                        parts.append(f"URL: {url}")
                    if snippet:
                        parts.append(f"Snippet: {snippet}")
                    return "\n".join(parts) if parts else "(fetch complete)"

                def _format_code_execution_result(inner: dict[str, Any]) -> str:
                    """Render an Anthropic code-execution result block as the preformatted text
                    payload the frontend's CodeExecutionToolUI displays inside a <pre>. Handles
                    bash, text_editor (view/create/str_replace), and the matching error variants."""
                    inner_type = inner.get("type") or ""
                    if inner_type.endswith("_error"):
                        return f"Error: {inner.get('error_code', 'unknown')}"
                    if inner_type == "bash_code_execution_result":
                        stdout = inner.get("stdout") or ""
                        stderr = inner.get("stderr") or ""
                        return_code = inner.get("return_code")
                        parts: list[str] = []
                        if stdout:
                            parts.append(stdout)
                        if stderr:
                            parts.append(f"--- stderr ---\n{stderr}")
                        if isinstance(return_code, int) and return_code != 0:
                            parts.append(f"return_code: {return_code}")
                        return "\n".join(parts) if parts else "(no output)"
                    if inner_type == "text_editor_code_execution_result":
                        if "lines" in inner and isinstance(inner.get("lines"), list):
                            return "\n".join(str(line) for line in inner["lines"])
                        if "is_file_update" in inner:
                            return "Updated" if inner.get("is_file_update") else "Created"
                        content_field = inner.get("content")
                        if isinstance(content_field, str):
                            return content_field
                        return "(file operation complete)"
                    return "(code execution complete)"

                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line or line.startswith("event:"):
                            continue
                        if not line.startswith("data:"):
                            continue

                        data_str = line[len("data:") :].strip()
                        if not data_str:
                            continue

                        try:
                            event = _json.loads(data_str)
                        except _json.JSONDecodeError:
                            continue

                        event_type = event.get("type")
                        if event_type == "content_block_delta":
                            delta_kind = (event.get("delta") or {}).get("type")
                            key = f"{event_type}:{delta_kind}"
                        else:
                            key = event_type or "<unknown>"
                        event_counts[key] = event_counts.get(key, 0) + 1

                        if event_type == "message_start":
                            start_usage = (event.get("message") or {}).get("usage")
                            if isinstance(start_usage, dict):
                                last_usage.update(start_usage)

                        if event_type == "content_block_start":
                            content_block = event.get("content_block") or {}
                            block_type = content_block.get("type")
                            block_name = content_block.get("name")
                            block_index = event.get("index", 0)
                            if tools:
                                replay_blocks[block_index] = dict(content_block)
                            if block_type == "tool_use":
                                client_tool_indices[block_index] = len(client_tool_indices)
                                client_tool_chunks.append(
                                    _delta_chunk(
                                        {
                                            "tool_calls": [
                                                {
                                                    "index": client_tool_indices[block_index],
                                                    "id": content_block["id"],
                                                    "type": "function",
                                                    "function": {
                                                        "name": block_name,
                                                        "arguments": "",
                                                    },
                                                }
                                            ]
                                        }
                                    )
                                )
                            elif block_type == "server_tool_use" and block_name == "web_search":
                                tool_use_id = content_block.get("id", "") or (
                                    f"ws_{len(web_search_calls)}"
                                )
                                current_server_tool_use = {
                                    "id": tool_use_id,
                                    "buffer": "",
                                }
                                web_search_calls[tool_use_id] = {
                                    "query": "",
                                    "results": [],
                                }
                            elif block_type == "web_search_tool_result":
                                tool_use_id = content_block.get("tool_use_id", "")
                                content = content_block.get("content") or []
                                current_result_block = {
                                    "tool_use_id": tool_use_id,
                                    "results": content if isinstance(content, (list, dict)) else [],
                                }
                            elif block_type == "server_tool_use" and block_name == "web_fetch":
                                tool_use_id = content_block.get("id", "") or (
                                    f"wf_{len(web_fetch_calls)}"
                                )
                                current_web_fetch_use = {
                                    "id": tool_use_id,
                                    "buffer": "",
                                }
                                web_fetch_calls[tool_use_id] = {
                                    "url": "",
                                    "result": None,
                                }
                            elif block_type == "web_fetch_tool_result":
                                tool_use_id = content_block.get("tool_use_id", "")
                                inner = content_block.get("content") or {}
                                current_web_fetch_result = {
                                    "tool_use_id": tool_use_id,
                                    "inner": inner if isinstance(inner, dict) else {},
                                }
                            elif block_type == "server_tool_use" and block_name in (
                                "bash_code_execution",
                                "text_editor_code_execution",
                            ):
                                tool_use_id = content_block.get("id", "") or (
                                    f"ce_{len(code_execution_calls)}"
                                )
                                kind = (
                                    "bash" if block_name == "bash_code_execution" else "text_editor"
                                )
                                current_code_exec_use = {
                                    "id": tool_use_id,
                                    "kind": kind,
                                    "buffer": "",
                                }
                                code_execution_calls[tool_use_id] = {
                                    "kind": kind,
                                    "arguments": {},
                                    "result": None,
                                }
                            elif block_type in (
                                "bash_code_execution_tool_result",
                                "text_editor_code_execution_tool_result",
                            ):
                                tool_use_id = content_block.get("tool_use_id", "")
                                inner = content_block.get("content") or {}
                                current_code_exec_result = {
                                    "tool_use_id": tool_use_id,
                                    "inner": inner if isinstance(inner, dict) else {},
                                }
                            elif block_type == "compaction":
                                seed = content_block.get("content") or ""
                                current_compaction = {
                                    "content": seed if isinstance(seed, str) else "",
                                }
                                encrypted = content_block.get("encrypted_content")
                                if isinstance(encrypted, str) and encrypted:
                                    current_compaction["encrypted_content"] = encrypted

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            delta_type = delta.get("type")
                            block_index = event.get("index", 0)
                            replay = replay_blocks.get(block_index)
                            if replay is not None:
                                field = {
                                    "text_delta": "text",
                                    "thinking_delta": "thinking",
                                    "signature_delta": "signature",
                                }.get(delta_type)
                                if field == "text" and replay.get("type") == "compaction":
                                    field = "content"
                                if field:
                                    replay[field] = replay.get(field, "") + delta.get(
                                        "text" if field == "content" else field, ""
                                    )
                                elif delta_type == "input_json_delta":
                                    replay_inputs[block_index] = replay_inputs.get(
                                        block_index, ""
                                    ) + delta.get("partial_json", "")
                                elif delta_type == "citations_delta":
                                    replay.setdefault("citations", []).append(delta["citation"])
                                elif delta_type == "compaction_delta":
                                    content = delta.get("content")
                                    if isinstance(content, str):
                                        prior = replay.get("content")
                                        replay["content"] = (
                                            prior if isinstance(prior, str) else ""
                                        ) + content
                                    replay.update(
                                        {
                                            key: value
                                            for key, value in delta.items()
                                            if key not in ("type", "content")
                                        }
                                    )
                            if delta_type == "thinking_delta":
                                thinking_text = delta.get("thinking", "")
                                if thinking_text:
                                    if not thinking_open:
                                        thinking_text = f"<think>{thinking_text}"
                                        thinking_open = True
                                    yield _content_chunk(thinking_text)
                            elif delta_type == "text_delta":
                                text = delta.get("text", "")
                                if current_compaction is not None:
                                    if text:
                                        current_compaction["content"] += text
                                else:
                                    # close thinking on text_delta to tolerate out-of-order events.
                                    if thinking_open:
                                        yield _content_chunk("</think>")
                                        thinking_open = False
                                    if text:
                                        yield _content_chunk(text)
                            elif (
                                delta_type == "compaction_delta" and current_compaction is not None
                            ):
                                summary = delta.get("content")
                                if isinstance(summary, str):
                                    current_compaction["content"] += summary
                                encrypted = delta.get("encrypted_content")
                                if isinstance(encrypted, str) and encrypted:
                                    current_compaction["encrypted_content"] = encrypted
                            elif delta_type == "citations_delta":
                                cit = delta.get("citation")
                                if isinstance(cit, dict):
                                    key = _anthropic_citation_key(cit)
                                    idx_for_marker: Optional[int] = None
                                    for idx, existing in enumerate(document_citations, start = 1):
                                        if existing.get("_key") == key:
                                            idx_for_marker = idx
                                            break
                                    if idx_for_marker is None:
                                        document_citations.append({**cit, "_key": key})
                                        idx_for_marker = len(document_citations)
                                    yield _content_chunk(f"[{idx_for_marker}]")
                            elif delta_type == "input_json_delta":
                                partial = delta.get("partial_json", "")
                                if block_index in client_tool_indices:
                                    client_tool_chunks.append(
                                        _delta_chunk(
                                            {
                                                "tool_calls": [
                                                    {
                                                        "index": client_tool_indices[block_index],
                                                        "function": {"arguments": partial},
                                                    }
                                                ]
                                            }
                                        )
                                    )
                                elif current_server_tool_use is not None:
                                    current_server_tool_use["buffer"] += partial
                                elif current_code_exec_use is not None:
                                    current_code_exec_use["buffer"] += partial
                                elif current_web_fetch_use is not None:
                                    current_web_fetch_use["buffer"] += partial

                        elif event_type == "content_block_stop":
                            block_index = event.get("index", 0)
                            if block_index in replay_inputs:
                                try:
                                    replay_blocks[block_index]["input"] = _json.loads(
                                        replay_inputs.pop(block_index)
                                    )
                                except _json.JSONDecodeError:
                                    replay_blocks.pop(block_index)
                            if current_server_tool_use is not None:
                                buffer = current_server_tool_use["buffer"]
                                query = ""
                                if buffer:
                                    try:
                                        parsed = _json.loads(buffer)
                                        if isinstance(parsed, dict):
                                            q = parsed.get("query", "")
                                            if isinstance(q, str):
                                                query = q
                                    except Exception:
                                        query = ""
                                tool_use_id = current_server_tool_use["id"]
                                if tool_use_id in web_search_calls:
                                    web_search_calls[tool_use_id]["query"] = query
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_start",
                                        "tool_name": "web_search",
                                        "tool_call_id": tool_use_id,
                                        "arguments": ({"query": query} if query else {}),
                                    }
                                )
                                current_server_tool_use = None
                            elif current_result_block is not None:
                                tool_use_id = current_result_block["tool_use_id"]
                                results = current_result_block["results"]
                                if tool_use_id in web_search_calls:
                                    web_search_calls[tool_use_id]["results"] = results
                                result_text = _format_web_search_results(results)
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_end",
                                        "tool_call_id": tool_use_id,
                                        "result": (result_text or "(search complete)"),
                                    }
                                )
                                current_result_block = None
                            elif current_code_exec_use is not None:
                                buffer = current_code_exec_use["buffer"]
                                parsed_args: dict[str, Any] = {}
                                if buffer:
                                    try:
                                        parsed_obj = _json.loads(buffer)
                                        if isinstance(parsed_obj, dict):
                                            parsed_args = parsed_obj
                                    except Exception:
                                        parsed_args = {}
                                tool_use_id = current_code_exec_use["id"]
                                kind = current_code_exec_use["kind"]
                                emit_args = {"kind": kind, **parsed_args}
                                if tool_use_id in code_execution_calls:
                                    code_execution_calls[tool_use_id]["arguments"] = emit_args
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_start",
                                        "tool_name": "code_execution",
                                        "tool_call_id": tool_use_id,
                                        "arguments": emit_args,
                                    }
                                )
                                current_code_exec_use = None
                            elif current_compaction is not None:
                                # Anthropic can fail compaction when tools are defined and returns content:null. That
                                # block is a no-op even when it carries encrypted state: replaying it would discard
                                # the live history. OpenAI's encrypted-only item uses a different translator below.
                                compaction_blocks_seen += 1
                                summary = current_compaction.get("content")
                                if isinstance(summary, str) and summary:
                                    yield _emit_tool_event(
                                        {
                                            "type": "compaction_block",
                                            **current_compaction,
                                        }
                                    )
                                else:
                                    # Do not let message_delta replay the failed native block through
                                    # extra_content either; the rest of this turn still has to follow full history.
                                    replay_blocks.pop(block_index, None)
                                current_compaction = None
                            elif current_code_exec_result is not None:
                                tool_use_id = current_code_exec_result["tool_use_id"]
                                inner = current_code_exec_result["inner"]
                                if isinstance(inner, dict):
                                    file_blocks = inner.get("content")
                                    if isinstance(file_blocks, list):
                                        for entry in file_blocks:
                                            if isinstance(entry, dict) and entry.get("file_id"):
                                                code_execution_generated_files += 1
                                result_text = _format_code_execution_result(
                                    inner if isinstance(inner, dict) else {}
                                )
                                if tool_use_id in code_execution_calls:
                                    code_execution_calls[tool_use_id]["result"] = result_text
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_end",
                                        "tool_call_id": tool_use_id,
                                        "result": result_text,
                                    }
                                )
                                current_code_exec_result = None
                            elif current_web_fetch_use is not None:
                                buffer = current_web_fetch_use["buffer"]
                                url = ""
                                if buffer:
                                    try:
                                        parsed = _json.loads(buffer)
                                        if isinstance(parsed, dict):
                                            probe = parsed.get("url", "")
                                            if isinstance(probe, str):
                                                url = probe
                                    except Exception:
                                        logger.debug(
                                            "Failed to parse web_fetch input_json",
                                            buffer = buffer,
                                        )
                                        url = ""
                                tool_use_id = current_web_fetch_use["id"]
                                if tool_use_id in web_fetch_calls:
                                    web_fetch_calls[tool_use_id]["url"] = url
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_start",
                                        "tool_name": "web_fetch",
                                        "tool_call_id": tool_use_id,
                                        "arguments": ({"url": url} if url else {}),
                                    }
                                )
                                current_web_fetch_use = None
                            elif current_web_fetch_result is not None:
                                tool_use_id = current_web_fetch_result["tool_use_id"]
                                result_text = _format_web_fetch_result(
                                    current_web_fetch_result["inner"]
                                )
                                if tool_use_id in web_fetch_calls:
                                    web_fetch_calls[tool_use_id]["result"] = result_text
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_end",
                                        "tool_call_id": tool_use_id,
                                        "result": result_text,
                                    }
                                )
                                current_web_fetch_result = None
                            elif thinking_open:
                                yield _content_chunk("</think>")
                                thinking_open = False

                        elif event_type == "message_delta":
                            if replay_blocks:
                                yield _delta_chunk(
                                    {
                                        "extra_content": {
                                            "anthropic": {"content": list(replay_blocks.values())}
                                        }
                                    }
                                )
                            delta_usage = event.get("usage")
                            if isinstance(delta_usage, dict):
                                last_usage.update(delta_usage)
                                iterations = delta_usage.get("iterations")
                                if isinstance(iterations, list):
                                    c_in = 0
                                    c_out = 0
                                    for it in iterations:
                                        if isinstance(it, dict) and it.get("type") == "compaction":
                                            c_in += int(it.get("input_tokens") or 0)
                                            c_out += int(it.get("output_tokens") or 0)
                                    if c_in or c_out:
                                        last_usage["compaction_input_tokens"] = c_in
                                        last_usage["compaction_output_tokens"] = c_out
                            delta_obj = event.get("delta") or {}
                            container_obj = delta_obj.get("container")
                            if isinstance(container_obj, dict) and latched_container_id is None:
                                probe = container_obj.get("id")
                                if isinstance(probe, str) and probe:
                                    latched_container_id = probe
                            if (
                                latched_container_id
                                and not container_id_emitted
                                and latched_container_id != anthropic_code_exec_container_id
                            ):
                                yield _emit_tool_event(
                                    {
                                        "type": "container_ready",
                                        "container_id": latched_container_id,
                                    }
                                )
                                container_id_emitted = True
                            stop_reason = event.get("delta", {}).get("stop_reason")
                            if stop_reason:
                                if thinking_open:
                                    yield _content_chunk("</think>")
                                    thinking_open = False
                                # pause_turn is not terminal: no finish_reason=stop chunk, which would truncate the UI.
                                if stop_reason not in _finish_reason_map:
                                    logger.warning(
                                        "Unmapped Anthropic stop_reason %r (model=%s); "
                                        "reporting the turn as finished",
                                        stop_reason,
                                        model,
                                    )
                                mapped = _finish_reason_map.get(stop_reason, "stop")
                                if stop_reason == "refusal":
                                    logger.warning(
                                        "Anthropic refusal stop_reason (model=%s)",
                                        model,
                                    )
                                    # Drop signal rides _toolEvent so assistant text cannot spoof a context reset.
                                    yield _content_chunk(
                                        "\n\n_The response was stopped by "
                                        "Anthropic's safety classifier. Edit "
                                        "or remove the previous turn and try "
                                        "again._"
                                    )
                                    yield _emit_tool_event({"type": "anthropic_refusal"})
                                if stop_reason == "model_context_window_exceeded":
                                    logger.warning(
                                        "Anthropic context window exhausted (model=%s)",
                                        model,
                                    )
                                    yield _emit_tool_event({"type": "context_window_exceeded"})
                                if mapped is not None:
                                    chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": mapped,
                                            }
                                        ],
                                    }
                                    if client_tool_indices:
                                        client_tool_chunks.append(f"data: {_json.dumps(chunk)}")
                                    else:
                                        yield f"data: {_json.dumps(chunk)}"

                        elif event_type == "message_stop":
                            for client_chunk in client_tool_chunks:
                                yield client_chunk
                            if thinking_open:
                                yield _content_chunk("</think>")
                                thinking_open = False
                            if document_citations:
                                clean_cits = []
                                for c in document_citations:
                                    entry = {k: v for k, v in c.items() if k != "_key"}
                                    cited = entry.get("cited_text")
                                    if isinstance(cited, str) and len(cited) > _CITED_TEXT_MAX_LEN:
                                        entry["cited_text"] = cited[:_CITED_TEXT_MAX_LEN] + "…"
                                    clean_cits.append(entry)
                                yield _emit_tool_event(
                                    {
                                        "type": "document_citations",
                                        "citations": clean_cits,
                                    }
                                )
                            usage_line = _build_usage_chunk(
                                completion_id,
                                "anthropic",
                                last_usage,
                            )
                            if usage_line:
                                yield usage_line
                            yield "data: [DONE]"
                            await response.aclose()
                            break

                        elif event_type == "error":
                            if thinking_open:
                                yield _content_chunk("</think>")
                            error = event.get("error")
                            error_type = error.get("type") if isinstance(error, dict) else None
                            if not isinstance(error_type, str):
                                error_type = None
                            yield _error_sse_line(
                                _ANTHROPIC_ERROR_STATUS.get(error_type, 502),
                                _json.dumps(event),
                                self.provider_type,
                            )
                            break
                except GeneratorExit:
                    await response.aclose()
                    await lines_gen.aclose()
                    raise
                finally:
                    web_search_requested = bool(enabled_tools and "web_search" in enabled_tools)
                    web_search_invocations = len(web_search_calls)
                    total_results = sum(
                        len(sc["results"])
                        for sc in web_search_calls.values()
                        if isinstance(sc.get("results"), list)
                    )
                    web_search_errors = [
                        sc["results"].get("error_code") or "unknown"
                        for sc in web_search_calls.values()
                        if isinstance(sc.get("results"), dict)
                        and sc["results"].get("type") == "web_search_tool_result_error"
                    ]
                    queries = [sc["query"] for sc in web_search_calls.values() if sc.get("query")]
                    code_execution_invocations = len(code_execution_calls)
                    code_execution_results = sum(
                        1 for c in code_execution_calls.values() if c.get("result") is not None
                    )
                    web_fetch_requested = web_fetch_enabled
                    web_fetch_invocations = len(web_fetch_calls)
                    web_fetch_urls = [wf["url"] for wf in web_fetch_calls.values() if wf.get("url")]
                    logger.info(
                        "Anthropic stream complete (model=%s, "
                        "web_search_requested=%s, web_search_invocations=%s, "
                        "results=%s, web_search_errors=%s, queries=%s, "
                        "web_fetch_requested=%s, web_fetch_invocations=%s, "
                        "web_fetch_urls=%s, "
                        "code_execution_requested=%s, "
                        "code_execution_invocations=%s, "
                        "code_execution_results=%s, "
                        "code_execution_generated_files=%s, "
                        "container_id_in=%s, container_id_out=%s, "
                        "input_tokens=%s, output_tokens=%s, "
                        "cache_creation_input_tokens=%s, "
                        "cache_read_input_tokens=%s, "
                        "compaction_input_tokens=%s, "
                        "compaction_output_tokens=%s, "
                        "compaction_blocks_seen=%s, events=%s)",
                        model,
                        web_search_requested,
                        web_search_invocations,
                        total_results,
                        web_search_errors,
                        queries,
                        web_fetch_requested,
                        web_fetch_invocations,
                        web_fetch_urls,
                        code_execution_enabled,
                        code_execution_invocations,
                        code_execution_results,
                        code_execution_generated_files,
                        anthropic_code_exec_container_id,
                        latched_container_id,
                        last_usage.get("input_tokens"),
                        last_usage.get("output_tokens"),
                        last_usage.get("cache_creation_input_tokens"),
                        last_usage.get("cache_read_input_tokens"),
                        last_usage.get("compaction_input_tokens"),
                        last_usage.get("compaction_output_tokens"),
                        compaction_blocks_seen,
                        event_counts,
                    )
                    await response.aclose()
                    await lines_gen.aclose()

        except httpx.ConnectError as exc:
            logger.error("Connection error to %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Failed to connect to {self.provider_type}: {exc}",
                self.provider_type,
            )
        except httpx.ReadTimeout as exc:
            logger.error("Read timeout from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                504,
                f"Timeout waiting for {self.provider_type} response",
                self.provider_type,
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP error from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Error communicating with {self.provider_type}: {exc}",
                self.provider_type,
            )

    async def _stream_gemini(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        top_k: Optional[int] = None,
        presence_penalty: float = 0.0,
        enabled_tools: Optional[list[str]] = None,
        enable_prompt_caching: Optional[Any] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Call Google's native Gemini API and translate its streaming ``streamGenerateContent``
        response into OpenAI Chat Completions chunks.

        Gemini does not speak the OpenAI Chat Completions contract on its primary endpoint. The
        request is POST /v1beta/models/{model}:streamGenerateContent?alt=sse with `contents` (role
        user|model, `parts`), `systemInstruction`, `generationConfig`
        (temperature/topP/topK/maxOutputTokens), `tools` (googleSearch, codeExecution) and an
        optional `cachedContent`. Streamed responses are SSE frames carrying partial
        ``GenerateContentResponse`` objects: `candidates[].content.parts[]`, `finishReason` and
        `usageMetadata`.

        Image generation uses the same endpoint with an image model (Nano Banana); the response
        carries an ``inlineData`` part with base64 bytes and a ``mimeType``, surfaced through the
        same ``tool_start`` / ``tool_end`` ``image_b64`` envelope the OpenAI image_generation path
        uses, so the chat UI renders it inline with no extra plumbing. Refs:
        https://ai.google.dev/gemini-api/docs/text-generation, function-calling, grounding, caching
        and image-generation.
        """
        import json as _json

        # Validate the model id first: `../cachedContents/x` is path traversal.
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
            yield _error_sse_line(
                400,
                f"Invalid Gemini model id: {model!r}",
                self.provider_type,
            )
            return

        system_text_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        tool_call_names: dict[str, str] = {}
        _gemini_skip_tool_result_ids: set[str] = set()
        # Decoded-byte cap ~14 MB: base64 expansion + prompt must fit Gemini's ~20 MB limit.
        _GEMINI_REMOTE_IMAGE_MAX_COUNT = 8
        _GEMINI_REMOTE_IMAGE_MAX_TOTAL_BYTES = 14 * 1024 * 1024
        _remote_image_count = 0
        _remote_image_total_bytes = 0
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                if isinstance(content, str):
                    if content:
                        system_text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "text"
                            and part.get("text")
                        ):
                            system_text_parts.append(part["text"])
                continue
            gemini_role = "model" if role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        text = part.get("text", "")
                        if text:
                            parts.append({"text": text})
                    elif ptype == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            header, _, b64data = url.partition(",")
                            media_type = (
                                header.split(";")[0].replace("data:", "").strip().lower()
                                or "image/jpeg"
                            )
                            if not media_type.startswith("image/"):
                                logger.info(
                                    "Gemini inlineData: refusing non-image data URL media_type=%s",
                                    media_type,
                                )
                            elif b64data:
                                _data_approx_bytes = (len(b64data) * 3) // 4
                                if _remote_image_count >= _GEMINI_REMOTE_IMAGE_MAX_COUNT:
                                    logger.info(
                                        "Gemini inlineData: per-request count cap %d reached, dropping image",
                                        _GEMINI_REMOTE_IMAGE_MAX_COUNT,
                                    )
                                elif (
                                    _remote_image_total_bytes + _data_approx_bytes
                                    > _GEMINI_REMOTE_IMAGE_MAX_TOTAL_BYTES
                                ):
                                    logger.info(
                                        "Gemini inlineData: per-request byte cap reached, dropping image",
                                    )
                                else:
                                    _remote_image_count += 1
                                    _remote_image_total_bytes += _data_approx_bytes
                                    parts.append(
                                        {
                                            "inlineData": {
                                                "mimeType": media_type,
                                                "data": b64data,
                                            }
                                        }
                                    )
                        elif url:
                            # fileUri takes only Files-API and YouTube URIs; parse the host, not the path.
                            try:
                                _parsed_image_url = urlparse(url)
                            except (ValueError, UnicodeError):
                                _parsed_image_url = None
                            if _parsed_image_url is None:
                                _img_scheme = ""
                                _img_host = ""
                                _img_path = ""
                            else:
                                _img_scheme = (_parsed_image_url.scheme or "").lower()
                                _img_host = (_parsed_image_url.hostname or "").lower()
                                _img_path = _parsed_image_url.path or ""
                            _is_native_uri = (
                                _img_scheme == "https"
                                and _img_host == "generativelanguage.googleapis.com"
                                and _img_path.startswith("/v1beta/files/")
                            )
                            _is_youtube = _img_scheme == "https" and (
                                _img_host == "youtu.be"
                                or _img_host == "youtube.com"
                                or _img_host.endswith(".youtube.com")
                            )
                            _guessed, _ = mimetypes.guess_type(_img_path)
                            _media_type = (
                                _guessed
                                if isinstance(_guessed, str) and _guessed.startswith("image/")
                                else "image/jpeg"
                            )
                            if _is_youtube:
                                # YouTube URIs must use video/mp4; the default image/jpeg yields a 400.
                                parts.append(
                                    {
                                        "fileData": {
                                            "fileUri": url,
                                            "mimeType": "video/mp4",
                                        }
                                    }
                                )
                            elif _is_native_uri:
                                parts.append(
                                    {
                                        "fileData": {
                                            "fileUri": url,
                                            "mimeType": _media_type,
                                        }
                                    }
                                )
                            elif _remote_image_count >= _GEMINI_REMOTE_IMAGE_MAX_COUNT:
                                logger.info(
                                    "Gemini image fetch: per-request count cap %d reached, dropping image",
                                    _GEMINI_REMOTE_IMAGE_MAX_COUNT,
                                )
                            else:
                                _remaining_bytes = (
                                    _GEMINI_REMOTE_IMAGE_MAX_TOTAL_BYTES - _remote_image_total_bytes
                                )
                                if _remaining_bytes <= 0:
                                    logger.info(
                                        "Gemini image fetch: per-request byte cap already reached, dropping image",
                                    )
                                else:
                                    _remote_image_count += 1
                                    _fetched = await _safe_fetch_image_for_gemini(
                                        url,
                                        _media_type,
                                        max_bytes = _remaining_bytes,
                                    )
                                    if _fetched is not None:
                                        _final_mime, _b64 = _fetched
                                        _approx_bytes = (len(_b64) * 3) // 4
                                        if (
                                            _remote_image_total_bytes + _approx_bytes
                                            > _GEMINI_REMOTE_IMAGE_MAX_TOTAL_BYTES
                                        ):
                                            logger.info(
                                                "Gemini image fetch: per-request byte cap reached, dropping image",
                                            )
                                        else:
                                            _remote_image_total_bytes += _approx_bytes
                                            parts.append(
                                                {
                                                    "inlineData": {
                                                        "mimeType": _final_mime,
                                                        "data": _b64,
                                                    }
                                                }
                                            )
            # Gemini 3 strict function-calling needs the text-part thoughtSignature replayed.
            if role == "assistant" and parts:
                _msg_extra = msg.get("extra_content") if isinstance(msg, dict) else None
                if isinstance(_msg_extra, dict):
                    _msg_g = _msg_extra.get("google") or {}
                    if isinstance(_msg_g, dict):
                        _msg_sig = _msg_g.get("thought_signature") or _msg_g.get("thoughtSignature")
                        if isinstance(_msg_sig, str) and _msg_sig:
                            for _idx in range(len(parts) - 1, -1, -1):
                                if "text" in parts[_idx]:
                                    parts[_idx] = {
                                        **parts[_idx],
                                        "thoughtSignature": _msg_sig,
                                    }
                                    break
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    if not isinstance(fn, dict):
                        continue
                    args_raw = fn.get("arguments") or "{}"
                    if isinstance(args_raw, str):
                        try:
                            args = _json.loads(args_raw)
                        except Exception:
                            args = {"_raw": args_raw}
                    elif isinstance(args_raw, dict):
                        args = args_raw
                    else:
                        args = {}
                    fn_name = fn.get("name", "")
                    tc_id = tc.get("id")
                    if fn_name and isinstance(tc_id, str) and tc_id:
                        tool_call_names[tc_id] = fn_name

                    _extra = tc.get("extra_content")
                    _native_part = None
                    _google_extra: dict[str, Any] = {}
                    if isinstance(_extra, dict):
                        _ge = _extra.get("google") or {}
                        if isinstance(_ge, dict):
                            _google_extra = _ge
                            _native_part = _ge.get("native_part")
                    if _native_part is None and isinstance(args, dict):
                        _args_google = args.get("google")
                        if isinstance(_args_google, dict):
                            _args_np = _args_google.get("native_part")
                            if isinstance(_args_np, dict):
                                _native_part = _args_np
                                if not _google_extra:
                                    _google_extra = _args_google

                    _name_lc = fn_name.lower() if isinstance(fn_name, str) else ""
                    _is_synthetic_server_builtin = (
                        _name_lc
                        in (
                            "web_search",
                            "web_fetch",
                            "code_execution",
                            "image_generation",
                        )
                        and isinstance(args, dict)
                        and (
                            args.get("_server_tool") is True
                            or isinstance((args.get("google") or {}).get("native_part"), dict)
                        )
                    )
                    if _is_synthetic_server_builtin and not (
                        _name_lc in ("code_execution", "image_generation")
                        and isinstance(_native_part, dict)
                    ):
                        if isinstance(tc_id, str) and tc_id:
                            _gemini_skip_tool_result_ids.add(tc_id)
                            tool_call_names.pop(tc_id, None)
                        continue
                    if fn_name in ("code_execution", "image_generation") and isinstance(
                        _native_part, dict
                    ):
                        # Skip the matching role=tool, else Gemini sees an undeclared functionResponse and 400s.
                        if isinstance(tc_id, str) and tc_id:
                            _gemini_skip_tool_result_ids.add(tc_id)
                        _native_parts_list = _native_part.get("parts")
                        if isinstance(_native_parts_list, list):
                            for _entry in _native_parts_list:
                                if isinstance(_entry, dict):
                                    parts.append(_entry)
                            continue
                        _legacy_sig = _native_part.get("thoughtSignature") or _native_part.get(
                            "thought_signature"
                        )
                        _legacy_subparts = [
                            _k
                            for _k in (
                                "executableCode",
                                "codeExecutionResult",
                                "inlineData",
                            )
                            if isinstance(_native_part.get(_k), dict)
                        ]
                        for _native_key in (
                            "executableCode",
                            "codeExecutionResult",
                            "inlineData",
                        ):
                            _sub = _native_part.get(_native_key)
                            if not isinstance(_sub, dict):
                                continue
                            _replay_part: dict[str, Any] = {_native_key: _sub}
                            if isinstance(_legacy_sig, str) and _legacy_sig:
                                if len(_legacy_subparts) == 1:
                                    _replay_part["thoughtSignature"] = _legacy_sig
                                elif _native_key == "executableCode":
                                    _replay_part["thoughtSignature"] = _legacy_sig
                            parts.append(_replay_part)
                        continue

                    function_call_part: dict[str, Any] = {
                        "name": fn_name,
                        "args": args,
                    }
                    if isinstance(tc_id, str) and tc_id:
                        function_call_part["id"] = tc_id
                    fc_part: dict[str, Any] = {"functionCall": function_call_part}
                    sig = _google_extra.get("thought_signature") or _google_extra.get(
                        "thoughtSignature"
                    )
                    if isinstance(sig, str) and sig:
                        fc_part["thoughtSignature"] = sig
                    parts.append(fc_part)
            if role == "tool":
                _tc_id_for_skip = msg.get("tool_call_id")
                if (
                    isinstance(_tc_id_for_skip, str)
                    and _tc_id_for_skip in _gemini_skip_tool_result_ids
                ):
                    continue
                tool_name = msg.get("name") or msg.get("tool_name") or ""
                if not tool_name:
                    tc_id = msg.get("tool_call_id")
                    if isinstance(tc_id, str) and tc_id in tool_call_names:
                        tool_name = tool_call_names[tc_id]
                response_payload: Any
                if isinstance(content, list):
                    _flat_parts: list[str] = []
                    for _cpart in content:
                        if (
                            isinstance(_cpart, dict)
                            and _cpart.get("type") == "text"
                            and isinstance(_cpart.get("text"), str)
                        ):
                            _flat_parts.append(_cpart["text"])
                    _flat_text = "".join(_flat_parts)
                    try:
                        response_payload = _json.loads(_flat_text)
                    except Exception:
                        response_payload = {"result": _flat_text}
                elif isinstance(content, str):
                    try:
                        response_payload = _json.loads(content)
                    except Exception:
                        response_payload = {"result": content}
                else:
                    response_payload = content or {}
                function_response_part: dict[str, Any] = {
                    "name": tool_name,
                    "response": (
                        response_payload
                        if isinstance(response_payload, dict)
                        else {"result": response_payload}
                    ),
                }
                tc_id = msg.get("tool_call_id")
                if isinstance(tc_id, str) and tc_id:
                    function_response_part["id"] = tc_id
                parts = [{"functionResponse": function_response_part}]
                gemini_role = "user"
            if parts:
                # Merge consecutive functionResponse-only user blocks: Gemini wants parallel tool responses grouped
                # into one user turn.
                if (
                    role == "tool"
                    and contents
                    and contents[-1].get("role") == "user"
                    and all(
                        isinstance(p, dict) and "functionResponse" in p
                        for p in (contents[-1].get("parts") or [])
                    )
                ):
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append({"role": gemini_role, "parts": parts})

        body: dict[str, Any] = {"contents": contents}
        if system_text_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_text_parts)}]}

        gen_config: dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["topP"] = top_p
        if top_k is not None and top_k > 0:
            gen_config["topK"] = top_k
        if presence_penalty:
            gen_config["presencePenalty"] = presence_penalty
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens

        # Only image-capable models accept TEXT+IMAGE modalities; text models 400.
        model_lc = model.lower()
        is_image_picker_model = "-image" in model_lc or "nano-banana" in model_lc
        _tool_choice_disabled = (
            isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"
        )
        _tool_choice_forced_function = (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
            and bool(tool_choice["function"].get("name"))
        )
        _hosted_builtins_allowed = not _tool_choice_disabled and not _tool_choice_forced_function
        # Image-tier models reject tools and thinkingConfig regardless of the pill.
        image_tool_requested = bool(
            _hosted_builtins_allowed and enabled_tools and "image_generation" in enabled_tools
        )
        is_image_model_strict = is_image_picker_model
        is_image_model = is_image_picker_model and image_tool_requested
        if is_image_model:
            gen_config["responseModalities"] = ["TEXT", "IMAGE"]
        elif is_image_picker_model:
            gen_config["responseModalities"] = ["TEXT"]

        # Gemini 3 uses thinkingLevel (no full off), 2.5 uses thinkingBudget; match 3.x by pattern
        # so a new minor does not get an int budget, which Gemini 3 rejects.
        _GEMINI3_ALIASES = (
            "gemini-pro-latest",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
        )
        _PRO_THINKING_PREFIXES = ("gemini-2.5-pro",)
        is_gemini3_thinking = bool(_GEMINI3_FAMILY.match(model_lc)) or model_lc.startswith(
            _GEMINI3_ALIASES
        )
        is_gemini3_pro = bool(_GEMINI3_PRO.match(model_lc)) or model_lc.startswith(
            "gemini-pro-latest"
        )
        _is_pro_thinking_only = any(
            model_lc == p or model_lc.startswith(p + "-") for p in _PRO_THINKING_PREFIXES
        )
        effort_lc = (reasoning_effort or "").strip().lower()
        is_gemma_thinking = bool(re.match(r"^gemma-(?:[4-9]|\d{2,})(?:\.\d+)?-", model_lc))
        if not is_image_model_strict and is_gemma_thinking:
            if effort_lc in ("none", "off") or enable_thinking is False:
                gen_config["thinkingConfig"] = {"thinkingLevel": "minimal"}
            elif effort_lc or enable_thinking is True:
                gen_config["thinkingConfig"] = {"thinkingLevel": "high"}
        elif not is_image_model_strict and is_gemini3_thinking:
            # 3.1+ Pro low/medium/high; 3 Pro low/high; Flash minimal..high. Coerce to the allowed set.
            _G3_LEVELS = {"minimal", "low", "medium", "high"}
            level: Optional[str] = None
            if effort_lc in ("none", "off"):
                level = "low" if is_gemini3_pro else "minimal"
            elif effort_lc == "max":
                level = "high"
            elif effort_lc in _G3_LEVELS:
                _is_legacy_gemini3_pro = model_lc.startswith("gemini-3-pro")
                if is_gemini3_pro and effort_lc == "minimal":
                    level = "low"
                elif _is_legacy_gemini3_pro and effort_lc == "medium":
                    level = "high"
                else:
                    level = effort_lc
            elif enable_thinking is True:
                level = "high"
            elif enable_thinking is False:
                level = "low" if is_gemini3_pro else "minimal"
            if level is not None:
                gen_config["thinkingConfig"] = {"thinkingLevel": level}
        elif not is_image_model_strict:
            # flash-lite rejects positive budgets below 512, so minimal=512.
            _EFFORT_TO_BUDGET: dict[str, int] = {
                "minimal": 512,
                "low": 2048,
                "medium": 8192,
                "high": 24576,
                "xhigh": -1,
                "max": -1,
            }
            thinking_budget: Optional[int] = None
            if effort_lc == "none" or enable_thinking is False:
                # Pro-tier 2.5 rejects budget=0 (400 "only works in thinking mode"), so coerce to a small positive
                # value.
                thinking_budget = 128 if _is_pro_thinking_only else 0
            elif effort_lc in _EFFORT_TO_BUDGET:
                thinking_budget = _EFFORT_TO_BUDGET[effort_lc]
            elif enable_thinking is True:
                thinking_budget = -1
            if thinking_budget is not None:
                gen_config["thinkingConfig"] = {
                    "thinkingBudget": thinking_budget,
                }

        if gen_config:
            body["generationConfig"] = gen_config

        def _gemini_image_model_allows_google_search(_m: str) -> bool:
            return (
                _m.startswith("gemini-3-pro-image")
                or _m.startswith("gemini-3.1-flash-image")
                or _m.startswith("nano-banana-pro")
                or _m.startswith("nano-banana-2")
            )

        google_search_allowed = (
            not is_image_model_strict or _gemini_image_model_allows_google_search(model_lc)
        )
        code_execution_allowed = not is_image_model_strict
        text_tools_allowed = not is_image_model_strict
        tools_array: list[dict[str, Any]] = []
        if (
            _hosted_builtins_allowed
            and enabled_tools
            and "web_search" in enabled_tools
            and google_search_allowed
        ):
            tools_array.append({"googleSearch": {}})
        if (
            _hosted_builtins_allowed
            and enabled_tools
            and "code_execution" in enabled_tools
            and code_execution_allowed
        ):
            tools_array.append({"codeExecution": {}})
        # Gemini Schema is an OpenAPI 3.0 subset; strip other keys or it 400s INVALID_ARGUMENT.
        _GEMINI_ALLOWED_SCHEMA_KEYS = frozenset(
            {
                "type",
                "format",
                "title",
                "description",
                "nullable",
                "enum",
                "maxItems",
                "minItems",
                "properties",
                "required",
                "minProperties",
                "maxProperties",
                "items",
                "minimum",
                "maximum",
                "minLength",
                "maxLength",
                "pattern",
                "default",
                "anyOf",
                "propertyOrdering",
            }
        )

        def _resolve_local_schema_ref(root: Optional[dict[str, Any]], ref: str) -> Optional[Any]:
            if not isinstance(root, dict) or not isinstance(ref, str):
                return None
            if not ref.startswith("#/"):
                return None
            node: Any = root
            for raw_part in ref[2:].split("/"):
                if not raw_part:
                    continue
                part = raw_part.replace("~1", "/").replace("~0", "~")
                if not isinstance(node, dict) or part not in node:
                    return None
                node = node[part]
            return node

        def _sanitize_gemini_schema(
            node: Any,
            root: Optional[dict[str, Any]] = None,
            _seen_refs: Optional[frozenset[str]] = None,
        ) -> Any:
            if root is None and isinstance(node, dict):
                root = node
            if _seen_refs is None:
                _seen_refs = frozenset()
            if isinstance(node, dict):
                _ref = node.get("$ref")
                if isinstance(_ref, str):
                    if _ref in _seen_refs:
                        return {}
                    _target = _resolve_local_schema_ref(root, _ref)
                    if isinstance(_target, dict):
                        _merged = {
                            **_target,
                            **{k: v for k, v in node.items() if k != "$ref"},
                        }
                        return _sanitize_gemini_schema(_merged, root, _seen_refs | {_ref})
                cleaned: dict[str, Any] = {}
                _nullable_from_union = False
                _flattened_type: Optional[str] = None
                _union_any_of: Optional[list[dict[str, Any]]] = None
                _raw_type = node.get("type")
                if isinstance(_raw_type, list):
                    _non_null = [t for t in _raw_type if t != "null"]
                    if len(_non_null) < len(_raw_type):
                        _nullable_from_union = True
                    if len(_non_null) == 1:
                        _flattened_type = _non_null[0]
                    elif len(_non_null) > 1:
                        _union_any_of = [{"type": _t} for _t in _non_null if isinstance(_t, str)]
                for _k, _v in node.items():
                    if _k == "type" and isinstance(_v, list):
                        continue
                    if _k not in _GEMINI_ALLOWED_SCHEMA_KEYS:
                        continue
                    if _k == "properties" and isinstance(_v, dict):
                        cleaned[_k] = {
                            _name: _sanitize_gemini_schema(_subschema, root, _seen_refs)
                            for _name, _subschema in _v.items()
                        }
                    elif _k == "items":
                        cleaned[_k] = _sanitize_gemini_schema(_v, root, _seen_refs)
                    elif _k == "anyOf" and isinstance(_v, list):
                        # Gemini rejects type null inside anyOf: drop it and set nullable instead.
                        _saw_null = any(
                            isinstance(_entry, dict) and _entry.get("type") == "null"
                            for _entry in _v
                        )
                        _non_null_entries = [
                            _entry
                            for _entry in _v
                            if not (isinstance(_entry, dict) and _entry.get("type") == "null")
                        ]
                        if len(_non_null_entries) == 1 and _saw_null:
                            _inner = _sanitize_gemini_schema(_non_null_entries[0], root, _seen_refs)
                            if isinstance(_inner, dict):
                                for _ik, _iv in _inner.items():
                                    cleaned.setdefault(_ik, _iv)
                                cleaned.setdefault("nullable", True)
                        else:
                            cleaned[_k] = [
                                _sanitize_gemini_schema(_entry, root, _seen_refs)
                                for _entry in _non_null_entries
                            ]
                            if _saw_null:
                                cleaned.setdefault("nullable", True)
                    elif _k in ("required", "enum", "propertyOrdering"):
                        cleaned[_k] = _v
                    else:
                        cleaned[_k] = _v
                if _union_any_of is not None and "anyOf" not in cleaned:
                    cleaned["anyOf"] = [
                        _sanitize_gemini_schema(_s, root, _seen_refs) for _s in _union_any_of
                    ]
                elif _flattened_type is not None:
                    cleaned["type"] = _flattened_type
                if _nullable_from_union and "nullable" not in cleaned:
                    cleaned["nullable"] = True
                return cleaned
            return node

        function_declarations: list[dict[str, Any]] = []
        if tools and text_tools_allowed and not _tool_choice_disabled:
            for _tool in tools:
                if not isinstance(_tool, dict) or _tool.get("type") != "function":
                    continue
                _fn = _tool.get("function")
                if not isinstance(_fn, dict) or not _fn.get("name"):
                    continue
                _decl: dict[str, Any] = {
                    "name": _fn["name"],
                    "description": _fn.get("description") or "",
                }
                _params = _fn.get("parameters")
                if isinstance(_params, dict):
                    _decl["parameters"] = _sanitize_gemini_schema(_params)
                function_declarations.append(_decl)
        if function_declarations:
            tools_array.append({"functionDeclarations": function_declarations})
        if tools_array:
            body["tools"] = tools_array
        if tool_choice is not None and function_declarations and text_tools_allowed:
            _mode: Optional[str] = None
            _allowed: Optional[list[str]] = None
            if isinstance(tool_choice, str):
                _tc_lc = tool_choice.strip().lower()
                if _tc_lc == "auto":
                    _mode = "AUTO"
                elif _tc_lc == "none":
                    _mode = "NONE"
                elif _tc_lc in ("required", "any"):
                    _mode = "ANY"
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                _fn_pick = tool_choice.get("function") or {}
                _name = _fn_pick.get("name") if isinstance(_fn_pick, dict) else None
                if isinstance(_name, str) and _name:
                    _mode = "ANY"
                    _allowed = [_name]
            if _mode is not None:
                _fcc: dict[str, Any] = {"mode": _mode}
                if _allowed:
                    _fcc["allowedFunctionNames"] = _allowed
                body["toolConfig"] = {"functionCallingConfig": _fcc}

        # JSON mode only on tool-free turns: Gemini 400s on response mime type plus function calling.
        _rf_type = response_format.get("type") if isinstance(response_format, dict) else None
        if _rf_type in ("json_object", "json_schema") and "tools" not in body:
            _gen_cfg = body.setdefault("generationConfig", {})
            _gen_cfg["responseMimeType"] = "application/json"
            if _rf_type == "json_schema":
                _rf_schema = response_format.get("json_schema")
                if isinstance(_rf_schema, dict) and isinstance(_rf_schema.get("schema"), dict):
                    _gen_cfg["responseSchema"] = _sanitize_gemini_schema(_rf_schema["schema"])

        if isinstance(enable_prompt_caching, str) and enable_prompt_caching:
            body["cachedContent"] = enable_prompt_caching

        url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse"
        completion_id = f"chatcmpl-gemini-{model.replace('/', '-')}"

        logger.info(
            "Proxying Gemini streamGenerateContent to %s (model=%s, tools=%s, image=%s)",
            url,
            model,
            [list(t.keys())[0] for t in tools_array] if tools_array else [],
            is_image_model,
        )

        def _emit_tool_event(payload: dict[str, Any]) -> str:
            _stamp_server_tool_marker(payload)
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None,
                    }
                ],
                "_toolEvent": payload,
            }
            return f"data: {_json.dumps(chunk)}"

        def _text_chunk(text: str, extra_content: Optional[dict[str, Any]] = None) -> str:
            delta: dict[str, Any] = {"content": text}
            if extra_content:
                delta["extra_content"] = extra_content
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {_json.dumps(chunk)}"

        def _gemini_part_extra(part: dict[str, Any]) -> Optional[dict[str, Any]]:
            """Return ``{"google": {"thought_signature": ...}}`` when the Gemini stream part carries
            a `thoughtSignature` we must replay on a follow-up turn (Gemini 3 image editing and
            tool contexts both require an exact signature echo)."""
            sig = part.get("thoughtSignature") or part.get("thought_signature")
            if isinstance(sig, str) and sig:
                return {"google": {"thought_signature": sig}}
            return None

        _finish_reason_map: dict[str, Optional[str]] = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "PROHIBITED_CONTENT": "content_filter",
            "BLOCKLIST": "content_filter",
            "MALFORMED_FUNCTION_CALL": "stop",
            "OTHER": "stop",
            "FINISH_REASON_UNSPECIFIED": None,
        }

        last_usage: Optional[dict[str, Any]] = None
        emitted_function_call_ids: set[str] = set()
        # Swap STOP -> tool_calls once a functionCall is emitted, or OAI clients never run the tool.
        emitted_any_function_call = False
        web_search_active = any("googleSearch" in t for t in tools_array)
        web_search_tool_id = "gemini_web_search"
        web_search_tool_started = False
        web_search_tool_ended = False
        web_search_citations: list[dict[str, str]] = []
        gemini_code_exec_pending_id: Optional[str] = None
        last_code_exec_tool_id: Optional[str] = None
        last_code_exec_result_text: str = ""

        try:
            async with _client().stream(
                "POST",
                url,
                json = body,
                headers = self._auth_headers(),
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    logger.error(
                        "Gemini returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    yield _error_sse_line(
                        response.status_code,
                        error_text,
                        self.provider_type,
                        response.headers.get("Retry-After"),
                    )
                    return

                if web_search_active:
                    yield _emit_tool_event(
                        {
                            "type": "tool_start",
                            "tool_name": "web_search",
                            "tool_call_id": web_search_tool_id,
                            "arguments": {},
                        }
                    )
                    web_search_tool_started = True

                lines_gen = response.aiter_lines().__aiter__()
                final_finish_reason: Optional[str] = None
                bare_json = ""
                stream_error: Optional[str] = None
                stream_error_message = ""
                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip():
                            continue
                        if line.startswith("data:"):
                            data_str = line[len("data:") :].strip()
                        elif bare_json or line.lstrip().startswith("{"):
                            # Gemini sends a mid-stream error as bare multi-line JSON, not as a `data:` frame.
                            bare_json += line
                            try:
                                _json.loads(bare_json)
                            except ValueError:
                                continue
                            data_str, bare_json = bare_json, ""
                        else:
                            continue
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            event = _json.loads(data_str)
                        except Exception:
                            logger.warning(
                                "Gemini: failed to parse SSE chunk: %s",
                                data_str[:200],
                            )
                            continue
                        if not isinstance(event, dict):
                            continue

                        error = event.get("error")
                        if isinstance(error, dict):
                            code = error.get("code")
                            stream_error = _error_sse_line(
                                code if isinstance(code, int) else 502,
                                _json.dumps(event),
                                self.provider_type,
                            )
                            stream_error_message = str(
                                error.get("message") or error.get("status") or code
                            )
                            break

                        usage_meta = event.get("usageMetadata")
                        if isinstance(usage_meta, dict):
                            last_usage = usage_meta

                        prompt_feedback = event.get("promptFeedback")
                        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
                            block_reason = str(prompt_feedback.get("blockReason"))
                            if (
                                web_search_active
                                and web_search_tool_started
                                and not web_search_tool_ended
                            ):
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_end",
                                        "tool_call_id": web_search_tool_id,
                                        "result": (
                                            "(search aborted: Gemini blocked "
                                            f"prompt: {block_reason})"
                                        ),
                                    }
                                )
                                web_search_tool_ended = True
                            yield _error_sse_line(
                                400,
                                f"Gemini blocked prompt: {block_reason}",
                                self.provider_type,
                            )
                            return

                        candidates = event.get("candidates") or []
                        if not isinstance(candidates, list):
                            continue
                        for cand in candidates:
                            if not isinstance(cand, dict):
                                continue
                            gm = cand.get("groundingMetadata")
                            if isinstance(gm, dict) and web_search_active:
                                chunks_list = gm.get("groundingChunks") or []
                                if isinstance(chunks_list, list):
                                    for ch in chunks_list:
                                        if not isinstance(ch, dict):
                                            continue
                                        web = ch.get("web") or {}
                                        if not isinstance(web, dict):
                                            continue
                                        u = web.get("uri") or ""
                                        if not u or not isinstance(u, str):
                                            continue
                                        if any(c["url"] == u for c in web_search_citations):
                                            continue
                                        web_search_citations.append(
                                            {
                                                "url": u,
                                                "title": (web.get("title") or u),
                                                "snippet": "",
                                            }
                                        )

                            content_obj = cand.get("content") or {}
                            parts = (
                                content_obj.get("parts") if isinstance(content_obj, dict) else None
                            )
                            if isinstance(parts, list):
                                for part in parts:
                                    if not isinstance(part, dict):
                                        continue
                                    text = part.get("text")
                                    _part_extra = _gemini_part_extra(part)
                                    if isinstance(text, str) and text:
                                        yield _text_chunk(
                                            text,
                                            extra_content = _part_extra,
                                        )
                                    elif _part_extra is not None and not any(
                                        k in part
                                        for k in (
                                            "functionCall",
                                            "executableCode",
                                            "codeExecutionResult",
                                            "inlineData",
                                        )
                                    ):
                                        yield _text_chunk(
                                            "",
                                            extra_content = _part_extra,
                                        )
                                    fc = part.get("functionCall")
                                    if isinstance(fc, dict):
                                        fc_name = fc.get("name") or ""
                                        fc_args = fc.get("args") or {}
                                        fc_id = fc.get("id") or f"call_{fc_name}_{time.time_ns()}"
                                        if fc_id in emitted_function_call_ids:
                                            continue
                                        emitted_function_call_ids.add(fc_id)
                                        # Distinct index per functionCall, else consumers collapse parallel calls.
                                        tc_index = len(emitted_function_call_ids) - 1
                                        tool_call_delta: dict[str, Any] = {
                                            "index": tc_index,
                                            "id": fc_id,
                                            "type": "function",
                                            "function": {
                                                "name": fc_name,
                                                "arguments": _json.dumps(fc_args),
                                            },
                                        }
                                        thought_sig = part.get("thoughtSignature") or part.get(
                                            "thought_signature"
                                        )
                                        if isinstance(thought_sig, str) and thought_sig:
                                            tool_call_delta["extra_content"] = {
                                                "google": {
                                                    "thought_signature": thought_sig,
                                                }
                                            }
                                        emitted_any_function_call = True
                                        tool_chunk = {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"tool_calls": [tool_call_delta]},
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                        yield f"data: {_json.dumps(tool_chunk)}"
                                    exec_code = part.get("executableCode")
                                    if isinstance(exec_code, dict):
                                        code_str = exec_code.get("code") or ""
                                        if code_str:
                                            code_tool_id = (
                                                exec_code.get("id")
                                                or f"gemini_code_exec_{time.time_ns()}"
                                            )
                                            gemini_code_exec_pending_id = code_tool_id
                                            _exec_thought_sig = part.get(
                                                "thoughtSignature"
                                            ) or part.get("thought_signature")
                                            # Gemini 3 rejects shared thoughtSignatures across parts.
                                            _exec_part_entry: dict[str, Any] = {
                                                "executableCode": exec_code,
                                            }
                                            if (
                                                isinstance(_exec_thought_sig, str)
                                                and _exec_thought_sig
                                            ):
                                                _exec_part_entry["thoughtSignature"] = (
                                                    _exec_thought_sig
                                                )
                                            _exec_native: dict[str, Any] = {
                                                "parts": [_exec_part_entry],
                                            }
                                            yield _emit_tool_event(
                                                {
                                                    "type": "tool_start",
                                                    "tool_name": "code_execution",
                                                    "tool_call_id": code_tool_id,
                                                    "arguments": {
                                                        "kind": "code_execution",
                                                        "language": (
                                                            (
                                                                exec_code.get("language")
                                                                or "PYTHON"
                                                            ).lower()
                                                        ),
                                                        "code": code_str,
                                                        "google": {
                                                            "native_part": _exec_native,
                                                        },
                                                    },
                                                }
                                            )
                                    exec_result = part.get("codeExecutionResult")
                                    if isinstance(exec_result, dict):
                                        outcome = exec_result.get("outcome") or ""
                                        output = exec_result.get("output") or ""
                                        if outcome and outcome != "OUTCOME_OK":
                                            result_text = f"[{outcome}]\n{output}".rstrip()
                                        else:
                                            result_text = output
                                        pair_id = (
                                            gemini_code_exec_pending_id
                                            or exec_result.get("id")
                                            or f"gemini_code_exec_{time.time_ns()}"
                                        )
                                        if gemini_code_exec_pending_id is None:
                                            yield _emit_tool_event(
                                                {
                                                    "type": "tool_start",
                                                    "tool_name": "code_execution",
                                                    "tool_call_id": pair_id,
                                                    "arguments": {
                                                        "kind": "code_execution",
                                                        "code": "",
                                                    },
                                                }
                                            )
                                        _result_thought_sig = part.get(
                                            "thoughtSignature"
                                        ) or part.get("thought_signature")
                                        _result_part_entry: dict[str, Any] = {
                                            "codeExecutionResult": exec_result,
                                        }
                                        if (
                                            isinstance(_result_thought_sig, str)
                                            and _result_thought_sig
                                        ):
                                            _result_part_entry["thoughtSignature"] = (
                                                _result_thought_sig
                                            )
                                        _result_native: dict[str, Any] = {
                                            "parts": [_result_part_entry],
                                        }
                                        yield _emit_tool_event(
                                            {
                                                "type": "tool_end",
                                                "tool_call_id": pair_id,
                                                "result": result_text,
                                                "google": {
                                                    "native_part": _result_native,
                                                },
                                            }
                                        )
                                        last_code_exec_tool_id = pair_id
                                        last_code_exec_result_text = result_text
                                        gemini_code_exec_pending_id = None
                                    inline = part.get("inlineData")
                                    if isinstance(inline, dict):
                                        b64 = inline.get("data") or ""
                                        mime = inline.get("mimeType") or "image/png"
                                        if b64:
                                            image_uri = f"data:{mime};base64,{b64}"
                                            attached_to_code_exec = (
                                                not is_image_model
                                                and last_code_exec_tool_id is not None
                                                and bool(enabled_tools)
                                                and "code_execution" in (enabled_tools or [])
                                            )
                                            if attached_to_code_exec:
                                                updated_result = (
                                                    last_code_exec_result_text
                                                    + "\n__IMAGES__:"
                                                    + _json.dumps([image_uri])
                                                )
                                                _plot_thought_sig = part.get(
                                                    "thoughtSignature"
                                                ) or part.get("thought_signature")
                                                _plot_part_entry: dict[str, Any] = {
                                                    "inlineData": {
                                                        "mimeType": mime,
                                                        "data": b64,
                                                    },
                                                }
                                                if (
                                                    isinstance(_plot_thought_sig, str)
                                                    and _plot_thought_sig
                                                ):
                                                    _plot_part_entry["thoughtSignature"] = (
                                                        _plot_thought_sig
                                                    )
                                                yield _emit_tool_event(
                                                    {
                                                        "type": "tool_end",
                                                        "tool_call_id": (last_code_exec_tool_id),
                                                        "result": updated_result,
                                                        "google": {
                                                            "native_part": {
                                                                "parts": [_plot_part_entry],
                                                            },
                                                        },
                                                    }
                                                )
                                                last_code_exec_result_text = updated_result
                                            else:
                                                img_id = f"img_{time.time_ns()}"
                                                yield _emit_tool_event(
                                                    {
                                                        "type": "tool_start",
                                                        "tool_name": "image_generation",
                                                        "tool_call_id": img_id,
                                                        "arguments": {
                                                            "kind": "image",
                                                            "prompt": "",
                                                        },
                                                    }
                                                )
                                                _img_thought_sig = part.get(
                                                    "thoughtSignature"
                                                ) or part.get("thought_signature")
                                                _img_tool_end: dict[str, Any] = {
                                                    "type": "tool_end",
                                                    "tool_call_id": img_id,
                                                    "result": "",
                                                    "image_b64": b64,
                                                    "image_mime": mime,
                                                }
                                                _img_part_entry: dict[str, Any] = {
                                                    "inlineData": {
                                                        "mimeType": mime,
                                                        "data": b64,
                                                    },
                                                }
                                                if (
                                                    isinstance(_img_thought_sig, str)
                                                    and _img_thought_sig
                                                ):
                                                    _img_part_entry["thoughtSignature"] = (
                                                        _img_thought_sig
                                                    )
                                                _img_native: dict[str, Any] = {
                                                    "parts": [_img_part_entry],
                                                }
                                                _img_google: dict[str, Any] = {
                                                    "native_part": _img_native,
                                                }
                                                if (
                                                    isinstance(_img_thought_sig, str)
                                                    and _img_thought_sig
                                                ):
                                                    _img_google["thought_signature"] = (
                                                        _img_thought_sig
                                                    )
                                                _img_tool_end["google"] = _img_google
                                                yield _emit_tool_event(_img_tool_end)
                            finish_reason = cand.get("finishReason")
                            if isinstance(finish_reason, str):
                                mapped = _finish_reason_map.get(finish_reason, "stop")
                                if mapped is not None:
                                    final_finish_reason = mapped

                    if web_search_active and web_search_tool_started and not web_search_tool_ended:
                        blocks: list[str] = []
                        for cit in web_search_citations:
                            line_out = f"Title: {cit['title']}\nURL: {cit['url']}"
                            if cit.get("snippet"):
                                line_out += f"\nSnippet: {cit['snippet']}"
                            blocks.append(line_out)
                        yield _emit_tool_event(
                            {
                                "type": "tool_end",
                                "tool_call_id": web_search_tool_id,
                                "result": (
                                    f"(search aborted: {stream_error_message})"
                                    if stream_error
                                    else "\n---\n".join(blocks)
                                    if blocks
                                    else "(search complete)"
                                ),
                            }
                        )
                        web_search_tool_ended = True

                    if stream_error:
                        yield stream_error
                        return

                    if final_finish_reason:
                        # Gemini says STOP for a pure functionCall turn; OAI clients need tool_calls.
                        if emitted_any_function_call and final_finish_reason == "stop":
                            final_finish_reason = "tool_calls"
                        finish_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": final_finish_reason,
                                }
                            ],
                        }
                        yield f"data: {_json.dumps(finish_chunk)}"

                    if isinstance(last_usage, dict):
                        thought_tokens = last_usage.get("thoughtsTokenCount") or 0
                        candidate_tokens = last_usage.get("candidatesTokenCount") or 0
                        prompt_tokens = last_usage.get("promptTokenCount") or 0
                        tool_use_prompt_tokens = last_usage.get("toolUsePromptTokenCount") or 0
                        translated_usage = {
                            "input_tokens": prompt_tokens + tool_use_prompt_tokens,
                            "output_tokens": candidate_tokens + thought_tokens,
                            "input_tokens_details": {
                                "cached_tokens": (last_usage.get("cachedContentTokenCount") or 0),
                                "tool_use_prompt_tokens": tool_use_prompt_tokens,
                            },
                            "output_tokens_details": {
                                "reasoning_tokens": thought_tokens,
                            },
                        }
                        usage_line = _build_usage_chunk(completion_id, "openai", translated_usage)
                        if usage_line:
                            yield usage_line

                    yield "data: [DONE]"
                finally:
                    await response.aclose()
                    await lines_gen.aclose()

        except httpx.ConnectError as exc:
            logger.error("Connection error to %s: %s", self.provider_type, exc)
            if web_search_tool_started and not web_search_tool_ended:
                yield _emit_tool_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": web_search_tool_id,
                        "result": f"(search aborted: connection error: {exc})",
                    }
                )
                web_search_tool_ended = True
            yield _error_sse_line(
                502,
                f"Failed to connect to {self.provider_type}: {exc}",
                self.provider_type,
            )
        except httpx.ReadTimeout as exc:
            logger.error("Read timeout from %s: %s", self.provider_type, exc)
            if web_search_tool_started and not web_search_tool_ended:
                yield _emit_tool_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": web_search_tool_id,
                        "result": "(search aborted: read timeout)",
                    }
                )
                web_search_tool_ended = True
            yield _error_sse_line(
                504,
                f"Timeout waiting for {self.provider_type} response",
                self.provider_type,
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP error from %s: %s", self.provider_type, exc)
            if web_search_tool_started and not web_search_tool_ended:
                yield _emit_tool_event(
                    {
                        "type": "tool_end",
                        "tool_call_id": web_search_tool_id,
                        "result": f"(search aborted: transport error: {exc})",
                    }
                )
                web_search_tool_ended = True
            yield _error_sse_line(
                502,
                f"Error communicating with {self.provider_type}: {exc}",
                self.provider_type,
            )

    async def _stream_openai_responses(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        enable_thinking: Optional[bool],
        reasoning_effort: Optional[str],
        enabled_tools: Optional[list[str]] = None,
        enable_prompt_caching: Optional[bool] = None,
        openai_code_exec_container_id: Optional[str] = None,
        compaction_threshold: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[dict[str, Any]] = None,
        stream: bool = True,
        compaction_fallback: Optional[CompactionFallback] = None,
    ) -> AsyncGenerator[str, None]:
        """Call OpenAI's /v1/responses endpoint and translate its SSE stream back into OpenAI Chat
        Completions chunk format. The Responses API uses a different request shape (``input`` not
        ``messages``, ``instructions`` for system prompts, ``max_output_tokens`` for the budget)
        and emits event-typed SSE frames rather than chat-completion chunks. ``presence_penalty``
        / ``top_k`` are not part of the Responses contract and are dropped here."""
        import json as _json

        # A deployment that rejected context_management once will reject it on every later Studio-tool turn too.
        # Apply the same local fitting fallback up front and suppress the unsupported field for the rest of this
        # client's run, rather than deliberately incurring another 400 before each continuation.
        if (
            self._responses_compaction_rejected
            and compaction_threshold is not None
            and compaction_threshold > 0
        ):
            if compaction_fallback is not None:
                messages, max_tokens, truncation_line = await compaction_fallback(messages)
                if truncation_line:
                    yield truncation_line
            compaction_threshold = None
            compaction_fallback = None

        is_openai_cloud = _is_openai_family_cloud(self.base_url)
        _responses_tool_choice_none = (
            isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"
        )
        _responses_tool_choice_forced_function = (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
            and bool(tool_choice["function"].get("name"))
        )
        _responses_hosted_builtins_allowed = (
            not _responses_tool_choice_none and not _responses_tool_choice_forced_function
        )
        image_generation_requested = bool(
            _responses_hosted_builtins_allowed
            and enabled_tools
            and "image_generation" in enabled_tools
            and is_openai_cloud
        )

        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []
        skipped_server_builtin_call_ids: set[str] = set()
        openai_replay_items: list[dict[str, Any]] = []
        previous_response_id: Optional[str] = None
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    if content:
                        instructions_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text" and part.get("text"):
                            instructions_parts.append(part["text"])
                continue

            if role == "assistant" and is_openai_cloud:
                # A saved turn carries its compaction item as a content part, the tool loop on extra_content.
                extra = msg.get("extra_content")
                replayed = (
                    [extra.get("openai_responses_compaction")] if isinstance(extra, dict) else []
                )
                if isinstance(content, list):
                    replayed += [
                        part.get("encrypted_content")
                        for part in content
                        if part.get("type") == "compaction"
                    ]
                replayed = [item for item in replayed if isinstance(item, str) and item]
                if replayed:
                    # The item carries everything before it, and resending that would compact it again.
                    input_items = [{"type": "compaction", "encrypted_content": replayed[-1]}]
                    openai_replay_items = []
                    previous_response_id = None
                    if not content and not msg.get("tool_calls"):
                        continue

            # Responses requires function_call items paired with function_call_output by call_id.
            if role == "tool":
                _call_id = msg.get("tool_call_id") or ""
                # drop outputs for omitted builtins to avoid orphan function_call_output items.
                if _call_id and _call_id in skipped_server_builtin_call_ids:
                    continue
                if isinstance(content, list):
                    _flat_parts: list[str] = []
                    for part in content:
                        if part.get("type") == "text" and part.get("text"):
                            _flat_parts.append(part["text"])
                    _output_text = "".join(_flat_parts)
                else:
                    _output_text = content if isinstance(content, str) else ""
                if _call_id:
                    input_items.append(responses_function_output(_call_id, _output_text))
                continue

            _tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
            if role == "assistant" and isinstance(_tool_calls, list):
                # Collected: reasoning items lead the turn, known only after the builtin filter below.
                _turn_items: list[dict[str, Any]] = []
                if isinstance(content, str) and content:
                    _turn_items.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    _asst_parts: list[dict[str, Any]] = []
                    for _part in content:
                        if not isinstance(_part, dict):
                            continue
                        _pt = _part.get("type")
                        if _pt == "text" and _part.get("text"):
                            _asst_parts.append(
                                {
                                    "type": "input_text",
                                    "text": _part.get("text", ""),
                                }
                            )
                        elif _pt == "image_url":
                            _u = _part.get("image_url", {}).get("url", "")
                            if _u:
                                _asst_parts.append({"type": "input_image", "image_url": _u})
                    if _asst_parts:
                        _turn_items.append({"role": "assistant", "content": _asst_parts})

                for _tc in _tool_calls:
                    if not isinstance(_tc, dict):
                        continue
                    _fn = _tc.get("function") or {}
                    if not isinstance(_fn, dict) or not _fn.get("name"):
                        continue
                    _args_raw = _fn.get("arguments") or ""
                    if not isinstance(_args_raw, str):
                        try:
                            _args_raw = _json.dumps(_args_raw)
                        except Exception:
                            _args_raw = ""
                    _fn_name_lc = (_fn.get("name") or "").lower()
                    _is_server_builtin = False
                    if _fn_name_lc in _SERVER_SIDE_BUILTIN_TOOL_NAMES:
                        try:
                            _args_obj = _json.loads(_args_raw) if _args_raw else {}
                        except Exception:
                            _args_obj = None
                        if isinstance(_args_obj, dict):
                            if _args_obj.get("_server_tool") is True:
                                _is_server_builtin = True
                            else:
                                _g = _args_obj.get("google")
                                if isinstance(_g, dict) and isinstance(_g.get("native_part"), dict):
                                    _is_server_builtin = True
                    _call_id_out = _tc.get("id") or f"call_{time.time_ns()}"
                    if _is_server_builtin:
                        skipped_server_builtin_call_ids.add(_call_id_out)
                        continue
                    _turn_items.append(
                        responses_function_call(_call_id_out, _fn["name"], _args_raw)
                    )
                # Reasoning items returned with tool calls must be replayed with them (OpenAI docs);
                # a trailing reasoning item with nothing after it is a hard 400, so replay none then.
                if _turn_items:
                    _msg_extra = msg.get("extra_content") if isinstance(msg, dict) else None
                    _reasoning_replay = (
                        _msg_extra.get("openai_responses_reasoning")
                        if isinstance(_msg_extra, dict)
                        else None
                    )
                    if isinstance(_reasoning_replay, list):
                        for _r_item in _reasoning_replay:
                            _replay = _sanitize_openai_reasoning_replay_item(_r_item)
                            if _replay:
                                input_items.append(_replay)
                input_items.extend(_turn_items)
                continue

            if isinstance(content, str):
                input_items.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                translated_parts: list[dict[str, Any]] = []
                used_previous_response_id = False
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                        translated_parts.append(
                            {"type": "input_text", "text": part.get("text", "")}
                        )
                    elif part_type == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url:
                            translated_parts.append({"type": "input_image", "image_url": url})
                    elif (
                        part_type == "reasoning"
                        and role == "assistant"
                        and image_generation_requested
                    ):
                        replay_item = _sanitize_openai_reasoning_replay_item(part)
                        if replay_item:
                            openai_replay_items.append(replay_item)
                    elif (
                        part_type == "image_generation_call"
                        and role == "assistant"
                        and image_generation_requested
                    ):
                        response_id = (
                            part.get("response_id")
                            or part.get("openai_response_id")
                            or part.get("previous_response_id")
                        )
                        call_id = part.get("id") or part.get("image_generation_call_id")
                        if isinstance(call_id, str) and call_id:
                            if isinstance(response_id, str) and response_id:
                                previous_response_id = response_id
                                input_items = []
                                translated_parts = []
                                used_previous_response_id = True
                            else:
                                previous_response_id = None
                            openai_replay_items.append(
                                {"type": "image_generation_call", "id": call_id}
                            )
                    elif part_type == "input_document":
                        file_url = part.get("file_url")
                        file_data = part.get("file_data")
                        filename = part.get("filename")
                        # Empty base64 payload counts as missing (file_data="" 400s); fall back to file_url.
                        file_data_valid = bool(
                            isinstance(file_data, str)
                            and file_data
                            and (
                                not file_data.startswith("data:")
                                or file_data.partition(",")[2].strip()
                            )
                        )
                        block: dict[str, Any] = {"type": "input_file"}
                        if file_data_valid:
                            block["file_data"] = file_data
                        elif file_url:
                            block["file_url"] = file_url
                        else:
                            continue
                        if filename:
                            block["filename"] = filename
                        translated_parts.append(block)
                if translated_parts and not used_previous_response_id:
                    input_items.append({"role": role, "content": translated_parts})

        if previous_response_id:
            openai_replay_items = []
        elif (
            _openai_image_replay_requires_reasoning(model)
            and reasoning_effort != "none"
            and enable_thinking is not False
        ):
            filtered_replay_items: list[dict[str, Any]] = []
            has_reasoning_replay = False
            dropped_image_replay_without_reasoning = False
            for item in openai_replay_items:
                if item.get("type") == "reasoning":
                    has_reasoning_replay = True
                    filtered_replay_items.append(item)
                elif item.get("type") == "image_generation_call":
                    if has_reasoning_replay:
                        filtered_replay_items.append(item)
                    else:
                        dropped_image_replay_without_reasoning = True
                else:
                    filtered_replay_items.append(item)
            openai_replay_items = filtered_replay_items
            if dropped_image_replay_without_reasoning:
                error_line = _error_sse_line(
                    400,
                    "OpenAI image edit reference is missing paired reasoning state. "
                    "Regenerate the image, then retry the edit.",
                    self.provider_type,
                )
                yield error_line if stream else error_line.removeprefix("data: ")
                return
        image_generation_has_reference = bool(
            previous_response_id
            or any(
                isinstance(item, dict) and item.get("type") == "image_generation_call"
                for item in openai_replay_items
            )
        )
        if openai_replay_items:
            insert_at = len(input_items)
            for index in range(len(input_items) - 1, -1, -1):
                if input_items[index].get("role") == "user":
                    insert_at = index
                    break
            input_items[insert_at:insert_at] = openai_replay_items

        body: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": stream,
        }
        # Azure deployment names hide the model, so omit sampling controls there.
        is_azure_openai = _is_azure_openai_host((urlparse(self.base_url).hostname or "").lower())
        forward_custom_sampling = self.provider_type == "custom" and not (
            is_azure_openai or (is_openai_cloud and _openai_fixed_sampling_model(model))
        )
        if forward_custom_sampling:
            if temperature is not None:
                body["temperature"] = temperature
            if top_p is not None:
                body["top_p"] = top_p
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        summary_unsupported = bool(
            _OPENAI_REASONING_SUMMARY_UNSUPPORTED.match(model.strip().lower())
        )
        if reasoning_effort in (
            "minimal",
            "low",
            "medium",
            "high",
            "max",
            "xhigh",
        ):
            body["reasoning"] = {"effort": reasoning_effort}
            if not summary_unsupported:
                body["reasoning"]["summary"] = "auto"
        elif reasoning_effort == "none" or enable_thinking is False:
            body["reasoning"] = {"effort": "none"}
        elif enable_thinking is True:
            body["reasoning"] = {"effort": "medium"}
            if not summary_unsupported:
                body["reasoning"]["summary"] = "auto"
        if instructions_parts:
            body["instructions"] = "\n\n".join(instructions_parts)
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens

        # Responses uses text.format with name/schema/strict flattened beside `type`.
        _rf_type = response_format.get("type") if isinstance(response_format, dict) else None
        if _rf_type == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        elif _rf_type == "json_schema":
            _rf_schema = response_format.get("json_schema")
            if isinstance(_rf_schema, dict) and isinstance(_rf_schema.get("schema"), dict):
                body["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": str(_rf_schema.get("name") or "response"),
                        "schema": _rf_schema["schema"],
                        "strict": bool(_rf_schema.get("strict", True)),
                    }
                }

        # 24h retention requires an OpenAI cloud model that accepts prompt_cache_retention.
        if (
            is_openai_cloud
            and enable_prompt_caching is not False
            and _OPENAI_EXTENDED_CACHE_FAMILY.match(model.strip().lower())
        ):
            body["prompt_cache_retention"] = "24h"

        # server-side context compaction is available only on OpenAI cloud.
        if is_openai_cloud and compaction_threshold is not None and compaction_threshold > 0:
            body["context_management"] = [
                {
                    "type": "compaction",
                    "compact_threshold": min(int(compaction_threshold), _SERVER_COMPACTION_MAX),
                }
            ]

        # only OpenAI cloud accepts Responses API server tools.
        code_execution_enabled_openai = bool(
            enabled_tools and "code_execution" in enabled_tools and is_openai_cloud
        )
        image_generation_enabled_openai = bool(
            enabled_tools and "image_generation" in enabled_tools and is_openai_cloud
        )

        def _openai_image_generation_tool() -> dict[str, Any]:
            tool: dict[str, Any] = {"type": "image_generation"}
            if image_generation_has_reference:
                # force edit mode so the prior call ID remains available as context.
                tool["action"] = "edit"
            return tool

        responses_user_function_tools: list[dict[str, Any]] = []
        if tools:
            for _tool in tools:
                if not isinstance(_tool, dict) or _tool.get("type") != "function":
                    continue
                _fn = _tool.get("function")
                if not isinstance(_fn, dict) or not _fn.get("name"):
                    continue
                _entry: dict[str, Any] = {
                    "type": "function",
                    "name": _fn["name"],
                }
                if _fn.get("description"):
                    _entry["description"] = _fn["description"]
                if isinstance(_fn.get("parameters"), dict):
                    _entry["parameters"] = normalize_function_schema(_fn["parameters"])
                if _fn.get("strict") is not None:
                    _entry["strict"] = bool(_fn["strict"])
                responses_user_function_tools.append(_entry)

        _responses_tc_string: Optional[str] = None
        if isinstance(tool_choice, str):
            _tc_lc = tool_choice.strip().lower()
            if _tc_lc in ("auto", "none", "required"):
                _responses_tc_string = _tc_lc
        responses_tool_choice: Optional[Any] = None
        _has_responses_tools = bool(enabled_tools or responses_user_function_tools)
        if _responses_tc_string is not None and _has_responses_tools:
            responses_tool_choice = _responses_tc_string
        elif (
            tool_choice is not None
            and responses_user_function_tools
            and isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
        ):
            _fn_pick = tool_choice.get("function") or {}
            _name = _fn_pick.get("name") if isinstance(_fn_pick, dict) else None
            if isinstance(_name, str) and _name:
                responses_tool_choice = {"type": "function", "name": _name}

        _responses_image_generation_enabled = (
            _responses_hosted_builtins_allowed and image_generation_enabled_openai
        )
        if _responses_image_generation_enabled and not stream:
            yield _json.dumps(
                {
                    "error": {
                        "message": (
                            "image_generation is not supported for non-streaming Responses "
                            "requests; set stream=true."
                        ),
                        "type": "invalid_request_error",
                        "code": "400",
                        "provider": self.provider_type,
                    }
                }
            )
            return

        if (enabled_tools or responses_user_function_tools) and not _responses_tool_choice_none:
            tools_array: list[dict[str, Any]] = list(responses_user_function_tools)
            if (
                _responses_hosted_builtins_allowed
                and enabled_tools
                and "web_search" in enabled_tools
            ):
                tools_array.append({"type": "web_search"})
            if _responses_hosted_builtins_allowed and code_execution_enabled_openai:
                shell_env: dict[str, Any]
                if openai_code_exec_container_id:
                    shell_env = {
                        "type": "container_reference",
                        "container_id": openai_code_exec_container_id,
                    }
                else:
                    shell_env = {"type": "container_auto"}
                tools_array.append({"type": "shell", "environment": shell_env})
            if _responses_image_generation_enabled:
                tools_array.append(_openai_image_generation_tool())
            if tools_array:
                body["tools"] = tools_array
        if responses_tool_choice is not None:
            body["tool_choice"] = responses_tool_choice

        url = _append_provider_path(self.base_url, "/responses")
        completion_id = f"chatcmpl-openai-{model.replace('/', '-')}"

        logger.info("Proxying OpenAI Responses API to %s (model=%s)", url, model)

        def _build_body(container_id_for_this_attempt: Optional[str]) -> dict[str, Any]:
            """Snapshot of the request body. Called once for the initial attempt and again with
            ``None`` for the post-expiry retry. Returns a fresh dict so the retry does not share
            state with the first attempt."""
            attempt_body = dict(body)
            if (enabled_tools or responses_user_function_tools) and not _responses_tool_choice_none:
                tools_array_attempt: list[dict[str, Any]] = list(responses_user_function_tools)
                if (
                    _responses_hosted_builtins_allowed
                    and enabled_tools
                    and "web_search" in enabled_tools
                ):
                    tools_array_attempt.append({"type": "web_search"})
                if _responses_hosted_builtins_allowed and code_execution_enabled_openai:
                    if container_id_for_this_attempt:
                        env_attempt: dict[str, Any] = {
                            "type": "container_reference",
                            "container_id": container_id_for_this_attempt,
                        }
                    else:
                        env_attempt = {"type": "container_auto"}
                    tools_array_attempt.append({"type": "shell", "environment": env_attempt})
                if _responses_image_generation_enabled:
                    tools_array_attempt.append(_openai_image_generation_tool())
                if tools_array_attempt:
                    attempt_body["tools"] = tools_array_attempt
                else:
                    attempt_body.pop("tools", None)
            if responses_tool_choice is not None:
                attempt_body["tool_choice"] = responses_tool_choice
            return attempt_body

        def _is_openai_container_expired_error(error_text: str) -> bool:
            """Substring-match OpenAI's expired/missing code-exec container errors (no official
            error code exists)."""
            lowered = error_text.lower()
            if "container" not in lowered:
                return False
            return (
                "expired" in lowered
                or "not_found" in lowered
                or "not found" in lowered
                or "no such container" in lowered
            )

        try:
            retried = False
            attempt_container_id = openai_code_exec_container_id
            while True:
                attempt_body = _build_body(attempt_container_id)
                async with _client().stream(
                    "POST",
                    url,
                    json = attempt_body,
                    headers = self._auth_headers(),
                    timeout = self._stream_timeout,
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_text = error_body.decode("utf-8", errors = "replace")
                        logger.error(
                            "OpenAI Responses returned %d: %s",
                            response.status_code,
                            error_text[:500],
                        )
                        expired_container_4xx = (
                            attempt_container_id
                            and 400 <= response.status_code < 500
                            and _is_openai_container_expired_error(error_text)
                        )
                        if (
                            response.status_code == 400
                            and (
                                "compact_threshold" in error_text
                                or "context_management" in error_text
                            )
                            and body.pop("context_management", None) is not None
                        ):
                            self._responses_compaction_rejected = True
                            if compaction_fallback is not None:
                                (
                                    fallback_messages,
                                    fallback_max_tokens,
                                    truncation_line,
                                ) = await compaction_fallback(messages)
                                await response.aclose()
                                if truncation_line:
                                    yield truncation_line
                                async for fallback_line in self._stream_openai_responses(
                                    messages = fallback_messages,
                                    model = model,
                                    temperature = temperature,
                                    top_p = top_p,
                                    max_tokens = fallback_max_tokens,
                                    enable_thinking = enable_thinking,
                                    reasoning_effort = reasoning_effort,
                                    enabled_tools = enabled_tools,
                                    enable_prompt_caching = enable_prompt_caching,
                                    openai_code_exec_container_id = openai_code_exec_container_id,
                                    compaction_threshold = None,
                                    compaction_fallback = None,
                                    tools = tools,
                                    tool_choice = tool_choice,
                                    response_format = response_format,
                                    stream = stream,
                                ):
                                    yield fallback_line
                                return
                            continue
                        if expired_container_4xx and not retried:
                            if stream:
                                yield (
                                    f"data: "
                                    f"{_json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': None}], '_toolEvent': {'type': 'container_invalidated'}})}"
                                )
                            retried = True
                            attempt_container_id = None
                            continue
                        error_line = _error_sse_line(
                            response.status_code,
                            error_text,
                            self.provider_type,
                            response.headers.get("Retry-After"),
                        )
                        yield error_line if stream else error_line.removeprefix("data: ")
                        return

                    if not stream:
                        response_payload = _json.loads(await response.aread())
                        status = response_payload.get("status")
                        if status == "failed" or response_payload.get("error"):
                            yield _json.dumps(
                                {
                                    "error": {
                                        "message": _openai_response_error_message(response_payload),
                                        "type": "server_error",
                                        "provider": self.provider_type,
                                    }
                                }
                            )
                            return
                        text_parts: list[str] = []
                        reasoning_summary_parts: list[str] = []
                        refusal_parts: list[str] = []
                        tool_calls: list[dict[str, Any]] = []
                        reasoning_replay_items: list[dict[str, Any]] = []
                        url_citations: list[dict[str, Any]] = []
                        compaction_part: dict[str, Any] | None = None
                        for item in response_payload.get("output") or []:
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") == "reasoning":
                                summary = item.get("summary")
                                if isinstance(summary, list):
                                    for part in summary:
                                        if not isinstance(part, dict):
                                            continue
                                        if part.get("type") != "summary_text":
                                            continue
                                        summary_text = part.get("text")
                                        if isinstance(summary_text, str) and summary_text:
                                            reasoning_summary_parts.append(summary_text)
                                reasoning_item = _sanitize_openai_reasoning_replay_item(item)
                                if reasoning_item:
                                    reasoning_replay_items.append(reasoning_item)
                            elif item.get("type") == "compaction":
                                encrypted = item.get("encrypted_content")
                                if isinstance(encrypted, str) and encrypted:
                                    compaction_part = {
                                        "type": "compaction",
                                        "encrypted_content": encrypted,
                                    }
                            elif item.get("type") == "message":
                                content = item.get("content") or []
                                if isinstance(content, str):
                                    text_parts.append(content)
                                    continue
                                for part in content:
                                    if not isinstance(part, dict):
                                        continue
                                    if part.get("type") in ("output_text", "text"):
                                        text = part.get("text")
                                        if isinstance(text, str):
                                            text_parts.append(text)
                                        for annotation in part.get("annotations") or []:
                                            if isinstance(annotation, dict):
                                                _record_openai_url_citation(
                                                    url_citations,
                                                    annotation,
                                                )
                                    elif part.get("type") == "refusal":
                                        refusal = part.get("refusal")
                                        if isinstance(refusal, str):
                                            refusal_parts.append(refusal)
                            elif item.get("type") == "function_call":
                                name = item.get("name")
                                if not isinstance(name, str) or not name:
                                    continue
                                arguments = item.get("arguments", "")
                                if not isinstance(arguments, str):
                                    arguments = _json.dumps(arguments)
                                tool_calls.append(
                                    {
                                        "id": item.get("call_id") or item.get("id") or "call_0",
                                        "type": "function",
                                        "function": {"name": name, "arguments": arguments},
                                    }
                                )
                        if not text_parts and isinstance(response_payload.get("output_text"), str):
                            text_parts.append(response_payload["output_text"])

                        visible_text = "".join(text_parts)
                        if reasoning_summary_parts:
                            visible_text = (
                                f"<think>{''.join(reasoning_summary_parts)}</think>{visible_text}"
                            )
                        visible_text = _replace_openai_citation_markers(
                            visible_text,
                            url_citations,
                        )

                        message: dict[str, Any] = {
                            "role": "assistant",
                            "content": visible_text or None,
                        }
                        if compaction_part is not None:
                            message["content"] = [compaction_part]
                            if visible_text:
                                message["content"].append({"type": "text", "text": visible_text})
                        if refusal_parts:
                            message["refusal"] = "".join(refusal_parts)
                        if tool_calls:
                            message["tool_calls"] = tool_calls
                            if reasoning_replay_items:
                                message["extra_content"] = {
                                    "openai_responses_reasoning": reasoning_replay_items
                                }

                        completion: dict[str, Any] = {
                            "id": response_payload.get("id") or completion_id,
                            "object": "chat.completion",
                            "created": int(response_payload.get("created_at") or time.time()),
                            "model": response_payload.get("model") or model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": message,
                                    "finish_reason": _openai_response_finish_reason(
                                        response_payload,
                                        has_tool_calls = bool(tool_calls),
                                    ),
                                }
                            ],
                        }
                        usage = responses_usage_to_chat(response_payload.get("usage"))
                        if usage is not None:
                            completion["usage"] = usage
                        yield _json.dumps(completion)
                        return

                    lines_gen = response.aiter_lines().__aiter__()
                    done_emitted = False
                    reasoning_open = False
                    reasoning_emitted = False
                    saw_function_call = False
                    function_call_index = 0
                    last_usage: Optional[dict[str, Any]] = None
                    web_search_calls: dict[str, dict[str, Any]] = {}
                    all_url_citations: list[dict[str, Any]] = []
                    shell_calls: dict[str, dict[str, Any]] = {}
                    latched_container_id: Optional[str] = None
                    container_id_emitted = False
                    current_openai_response_id: Optional[str] = None
                    last_openai_reasoning_replay_item: Optional[dict[str, Any]] = None
                    openai_reasoning_replay_items: dict[str, dict[str, Any]] = {}
                    image_generation_calls_started: set[str] = set()
                    pending_marker_tail: str = ""
                    pending_citation_segments: list[str] = []

                    def _record_openai_response_id(payload: dict[str, Any]) -> None:
                        nonlocal current_openai_response_id
                        response_obj = payload.get("response")
                        candidates: list[Any] = []
                        if isinstance(response_obj, dict):
                            candidates.append(response_obj.get("id"))
                        candidates.append(payload.get("response_id"))
                        for candidate in candidates:
                            if isinstance(candidate, str) and candidate:
                                current_openai_response_id = candidate
                                return

                    def _drain_pending_segments(force: bool) -> str:
                        """Re-attempt resolution on buffered segments in order. Stops at the first
                        still-unresolved segment unless ``force`` (end-of-stream), where
                        lingering markers drop."""
                        out: list[str] = []
                        while pending_citation_segments:
                            seg = pending_citation_segments[0]
                            rewritten, unresolved = _rewrite_citation_markers_partial(
                                seg,
                                all_url_citations,
                            )
                            if unresolved and not force:
                                pending_citation_segments[0] = rewritten
                                break
                            if unresolved and force:
                                rewritten = _replace_openai_citation_markers(
                                    rewritten,
                                    all_url_citations,
                                )
                            pending_citation_segments.pop(0)
                            if rewritten:
                                out.append(rewritten)
                        return "".join(out)

                    def _flush_pending_marker_tail(tail: str) -> str:
                        """Render any leftover citation tail at end-of-stream. Unterminated tails
                        drop (no annotation to bind to). If the close byte arrived concatenated,
                        rewrite then scrub any residual private-use bytes and any orphan
                        ``cite<sid>`` literal so the renderer never sees raw markup.
                        url_citations are aggregated separately and applied to web_search
                        tool_end."""
                        if not tail:
                            return ""
                        if _OPENAI_CITE_STOP not in tail:
                            return ""
                        rendered = _replace_openai_citation_markers(tail, all_url_citations)
                        for ch in ("", "", ""):
                            rendered = rendered.replace(ch, "")
                        rendered = re.sub(r"^cite\S*", "", rendered)
                        return rendered

                    def _emit_tool_event(payload: dict[str, Any]) -> str:
                        _stamp_server_tool_marker(payload)
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": None,
                                }
                            ],
                            "_toolEvent": payload,
                        }
                        return f"data: {_json.dumps(chunk)}"

                    def _format_shell_output(output: Any) -> str:
                        """Render `shell_call_output.output` (stdout/stderr/outcome per entry) as
                        the preformatted text CodeExecutionToolUI shows; append
                        return_code/(timeout) only when informative."""
                        if not isinstance(output, list):
                            return ""
                        parts: list[str] = []
                        for entry in output:
                            if not isinstance(entry, dict):
                                continue
                            stdout = entry.get("stdout") or ""
                            stderr = entry.get("stderr") or ""
                            outcome = entry.get("outcome") or {}
                            chunk_parts: list[str] = []
                            if stdout:
                                chunk_parts.append(stdout)
                            if stderr:
                                chunk_parts.append(f"--- stderr ---\n{stderr}")
                            if isinstance(outcome, dict):
                                outcome_type = outcome.get("type")
                                if outcome_type == "exit":
                                    exit_code = outcome.get("exit_code")
                                    if isinstance(exit_code, int) and exit_code != 0:
                                        chunk_parts.append(f"return_code: {exit_code}")
                                elif outcome_type == "timeout":
                                    chunk_parts.append("(timeout)")
                            if chunk_parts:
                                parts.append("\n".join(chunk_parts))
                        return "\n--- next command ---\n".join(parts) if parts else "(no output)"

                    def _record_url_citation(payload: dict[str, Any]) -> None:
                        _record_openai_url_citation(all_url_citations, payload)

                    def _record_openai_reasoning_replay_item(
                        payload: Any,
                    ) -> Optional[dict[str, Any]]:
                        if not isinstance(payload, dict):
                            return None
                        item_id = payload.get("id") or payload.get("item_id")
                        if not isinstance(item_id, str) or not item_id:
                            return None
                        existing = openai_reasoning_replay_items.setdefault(
                            item_id,
                            {
                                "type": "reasoning",
                                "id": item_id,
                                "summary": [],
                                "status": "completed",
                            },
                        )
                        if payload.get("type") == "reasoning":
                            sanitized = _sanitize_openai_reasoning_replay_item(payload)
                            if sanitized:
                                existing.update(sanitized)
                                return existing
                        summary_text = ""
                        part = payload.get("part")
                        if isinstance(part, dict) and part.get("type") == "summary_text":
                            text = part.get("text")
                            if isinstance(text, str):
                                summary_text = text
                        elif payload.get("type") == "response.reasoning_summary_text.done":
                            text = payload.get("text")
                            if isinstance(text, str):
                                summary_text = text
                        if summary_text:
                            summary_index = payload.get("summary_index")
                            summary = existing.setdefault("summary", [])
                            if isinstance(summary, list):
                                summary_part = {
                                    "type": "summary_text",
                                    "text": summary_text,
                                }
                                if isinstance(summary_index, int) and summary_index >= 0:
                                    while len(summary) <= summary_index:
                                        summary.append({"type": "summary_text", "text": ""})
                                    summary[summary_index] = summary_part
                                else:
                                    summary.append(summary_part)
                        return existing

                    def _image_generation_arguments(
                        prompt: str, raw_item_id: Any
                    ) -> dict[str, Any]:
                        arguments: dict[str, Any] = {"kind": "image", "prompt": prompt}
                        if isinstance(raw_item_id, str) and raw_item_id:
                            arguments["openai_image_generation_call_id"] = raw_item_id
                        if current_openai_response_id:
                            arguments["openai_response_id"] = current_openai_response_id
                        if last_openai_reasoning_replay_item:
                            arguments["openai_reasoning_item"] = last_openai_reasoning_replay_item
                        return arguments

                    def _extract_reasoning_text(payload: Any) -> str:
                        if payload is None:
                            return ""
                        if isinstance(payload, str):
                            return payload
                        if isinstance(payload, list):
                            out: list[str] = []
                            for item in payload:
                                text = _extract_reasoning_text(item)
                                if text:
                                    out.append(text)
                            return "".join(out)
                        if isinstance(payload, dict):
                            for key in ("text", "delta", "content", "summary"):
                                if key in payload:
                                    text = _extract_reasoning_text(payload.get(key))
                                    if text:
                                        return text
                            if payload.get("type") == "summary_text":
                                return _extract_reasoning_text(payload.get("text"))
                        return ""

                    def _chunk_with_text(text: str) -> str:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        return f"data: {_json.dumps(chunk)}"

                    sse_event_name = ""
                    try:
                        while True:
                            try:
                                line = await lines_gen.__anext__()
                            except StopAsyncIteration:
                                break
                            if not line:
                                # An event name ends at its blank line, else a stale response.failed fails later frames.
                                sse_event_name = ""
                                continue
                            if line.startswith("event:"):
                                sse_event_name = line[len("event:") :].strip()
                                continue
                            if not line.startswith("data:"):
                                continue

                            data_str = line[len("data:") :].strip()
                            if not data_str:
                                continue
                            if data_str == "[DONE]":
                                if pending_marker_tail:
                                    flushed = _flush_pending_marker_tail(pending_marker_tail)
                                    pending_marker_tail = ""
                                    if flushed:
                                        if reasoning_open:
                                            yield _chunk_with_text("</think>")
                                            reasoning_open = False
                                        yield _chunk_with_text(flushed)
                                tail_flushed = _drain_pending_segments(
                                    force = True,
                                )
                                if tail_flushed:
                                    if reasoning_open:
                                        yield _chunk_with_text("</think>")
                                        reasoning_open = False
                                    yield _chunk_with_text(tail_flushed)
                                if not done_emitted:
                                    yield "data: [DONE]"
                                    done_emitted = True
                                break

                            try:
                                event = _json.loads(data_str)
                            except _json.JSONDecodeError:
                                continue

                            try:
                                event_type = response_event_type(event, sse_event_name)
                            except ValueError:
                                # Compatible endpoints may send Chat Completions frames: skip unknown ones, but
                                # surface a bare {"error": ...} payload. The ChatGPT path stays strict.
                                if isinstance(event, dict) and isinstance(
                                    event.get("error"), (dict, str)
                                ):
                                    yield _error_sse_line(
                                        502,
                                        _openai_response_error_message(event),
                                        self.provider_type,
                                    )
                                    break
                                continue
                            _record_openai_response_id(event)

                            if event_type == "response.output_text.delta":
                                delta_text = event.get("delta", "")
                                for ann in event.get("annotations") or []:
                                    if isinstance(ann, dict):
                                        _record_url_citation(ann)
                                if delta_text or pending_marker_tail:
                                    combined = pending_marker_tail + delta_text
                                    head, pending_marker_tail = _split_pending_citation_tail(
                                        combined
                                    )
                                    if head:
                                        if reasoning_open:
                                            yield _chunk_with_text("</think>")
                                            reasoning_open = False
                                        flushed = _drain_pending_segments(
                                            force = False,
                                        )
                                        if flushed:
                                            yield _chunk_with_text(flushed)
                                        head_rewritten, has_unresolved = (
                                            _rewrite_citation_markers_partial(
                                                head,
                                                all_url_citations,
                                            )
                                        )
                                        if has_unresolved or pending_citation_segments:
                                            pending_citation_segments.append(head_rewritten)
                                        elif head_rewritten:
                                            yield _chunk_with_text(head_rewritten)

                            elif event_type == "response.refusal.delta":
                                refusal_delta = event.get("delta", "")
                                if isinstance(refusal_delta, str) and refusal_delta:
                                    if reasoning_open:
                                        yield _chunk_with_text("</think>")
                                        reasoning_open = False
                                    yield _chunk_with_text(refusal_delta)

                            elif event_type == "response.output_text.annotation.added":
                                ann = event.get("annotation")
                                if isinstance(ann, dict):
                                    _record_url_citation(ann)
                                flushed = _drain_pending_segments(
                                    force = False,
                                )
                                if flushed:
                                    if reasoning_open:
                                        yield _chunk_with_text("</think>")
                                        reasoning_open = False
                                    yield _chunk_with_text(flushed)

                            elif event_type == "response.output_item.added":
                                item = event.get("item", {})
                                if isinstance(item, dict) and item.get("type") == "web_search_call":
                                    item_id = item.get("id", "") or (f"ws_{len(web_search_calls)}")
                                    web_search_calls.setdefault(
                                        item_id,
                                        _extract_web_search_action(item),
                                    )
                                if isinstance(item, dict) and item.get("type") == "shell_call":
                                    item_id = item.get("id", "") or (f"sc_{len(shell_calls)}")
                                    shell_calls.setdefault(
                                        item_id,
                                        {"commands": [], "output": None},
                                    )
                                    env = item.get("environment")
                                    if isinstance(env, dict):
                                        probe = env.get("container_id") or env.get("id")
                                        if (
                                            isinstance(probe, str)
                                            and probe.startswith("cntr_")
                                            and latched_container_id is None
                                        ):
                                            latched_container_id = probe
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "image_generation_call"
                                ):
                                    raw_item_id = item.get("id")
                                    if isinstance(raw_item_id, str) and raw_item_id:
                                        arguments = _image_generation_arguments(
                                            "",
                                            raw_item_id,
                                        )
                                        image_generation_calls_started.add(raw_item_id)
                                        yield _emit_tool_event(
                                            {
                                                "type": "tool_start",
                                                "tool_name": "image_generation",
                                                "tool_call_id": raw_item_id,
                                                "arguments": arguments,
                                            }
                                        )

                            elif event_type == "response.output_item.done":
                                item = event.get("item", {})
                                if not isinstance(item, dict):
                                    continue
                                if item.get("type") == "reasoning":
                                    last_openai_reasoning_replay_item = (
                                        _record_openai_reasoning_replay_item(item)
                                    )
                                    summary_text = _extract_reasoning_text(item.get("summary"))
                                    if summary_text and not reasoning_emitted:
                                        if not reasoning_open:
                                            summary_text = f"<think>{summary_text}"
                                            reasoning_open = True
                                        yield _chunk_with_text(summary_text)
                                        reasoning_emitted = True
                                elif item.get("type") == "compaction":
                                    encrypted = item.get("encrypted_content")
                                    if isinstance(encrypted, str) and encrypted:
                                        yield _emit_tool_event(
                                            {
                                                "type": "compaction_block",
                                                "encrypted_content": encrypted,
                                            }
                                        )
                                elif item.get("type") == "web_search_call":
                                    # response.completed replaces this result with citations.
                                    item_id = item.get("id", "") or (f"ws_{len(web_search_calls)}")
                                    # merge partial done events so fields from added events survive.
                                    arguments = {
                                        **web_search_calls.get(item_id, {}),
                                        **_extract_web_search_action(item),
                                    }
                                    web_search_calls[item_id] = dict(arguments)
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_start",
                                            "tool_name": "web_search",
                                            "tool_call_id": item_id,
                                            "arguments": arguments,
                                        }
                                    )
                                    query = arguments.get("query") or ""
                                    per_call_result = f"Searching: {query}" if query else ""
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": item_id,
                                            "result": per_call_result,
                                        }
                                    )
                                elif item.get("type") == "shell_call":
                                    item_id = item.get("id", "") or (f"sc_{len(shell_calls)}")
                                    action = item.get("action") or {}
                                    commands = (
                                        action.get("commands") if isinstance(action, dict) else None
                                    ) or []
                                    joined_command = (
                                        "\n".join(str(c) for c in commands)
                                        if isinstance(commands, list)
                                        else ""
                                    )
                                    shell_calls.setdefault(
                                        item_id,
                                        {
                                            "commands": [],
                                            "output": None,
                                            "tool_end_emitted": False,
                                        },
                                    )
                                    shell_calls[item_id]["commands"] = (
                                        list(commands) if isinstance(commands, list) else []
                                    )
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_start",
                                            "tool_name": "code_execution",
                                            "tool_call_id": item_id,
                                            "arguments": {
                                                "kind": "bash",
                                                "command": joined_command,
                                            },
                                        }
                                    )
                                    embedded_output = item.get("output")
                                    if isinstance(embedded_output, list) and embedded_output:
                                        shell_calls[item_id]["output"] = embedded_output
                                        shell_calls[item_id]["tool_end_emitted"] = True
                                        yield _emit_tool_event(
                                            {
                                                "type": "tool_end",
                                                "tool_call_id": item_id,
                                                "result": _format_shell_output(embedded_output),
                                            }
                                        )
                                elif item.get("type") == "shell_call_output":
                                    call_id = item.get("call_id") or item.get("id") or ""
                                    output = item.get("output") or []
                                    if shell_calls.get(call_id, {}).get("tool_end_emitted"):
                                        continue
                                    if call_id in shell_calls:
                                        shell_calls[call_id]["output"] = output
                                        shell_calls[call_id]["tool_end_emitted"] = True
                                    result_text = _format_shell_output(output)
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": call_id,
                                            "result": result_text,
                                        }
                                    )
                                elif item.get("type") == "image_generation_call":
                                    raw_item_id = item.get("id")
                                    item_id = raw_item_id or f"img_{time.time_ns()}"
                                    prompt_in = (
                                        item.get("revised_prompt") or item.get("prompt") or ""
                                    )
                                    done_arguments = _image_generation_arguments(
                                        prompt_in,
                                        raw_item_id,
                                    )
                                    if item_id not in image_generation_calls_started:
                                        yield _emit_tool_event(
                                            {
                                                "type": "tool_start",
                                                "tool_name": "image_generation",
                                                "tool_call_id": item_id,
                                                "arguments": done_arguments,
                                            }
                                        )
                                    b64 = item.get("result") or item.get("b64_json") or ""
                                    output_format = item.get("output_format") or "png"
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": item_id,
                                            "result": "",
                                            "arguments": done_arguments,
                                            "image_b64": b64,
                                            "image_mime": (f"image/{output_format}"),
                                            "size": item.get("size"),
                                            "quality": item.get("quality"),
                                            "background": item.get("background"),
                                            "prompt": prompt_in,
                                        }
                                    )
                                elif item.get("type") == "function_call":
                                    fn_call_id = (
                                        item.get("call_id")
                                        or item.get("id")
                                        or f"call_{time.time_ns()}"
                                    )
                                    fn_name = item.get("name") or ""
                                    fn_args = item.get("arguments") or ""
                                    if not isinstance(fn_args, str):
                                        try:
                                            fn_args = _json.dumps(fn_args)
                                        except Exception:
                                            fn_args = ""
                                    _tc_index = function_call_index
                                    function_call_index += 1
                                    yield (
                                        "data: "
                                        + _json.dumps(
                                            {
                                                "id": completion_id,
                                                "object": "chat.completion.chunk",
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": {
                                                            "tool_calls": [
                                                                {
                                                                    "index": _tc_index,
                                                                    "id": fn_call_id,
                                                                    "type": "function",
                                                                    "function": {
                                                                        "name": fn_name,
                                                                        "arguments": (fn_args),
                                                                    },
                                                                }
                                                            ],
                                                        },
                                                        "finish_reason": None,
                                                    }
                                                ],
                                            }
                                        )
                                    )
                                    saw_function_call = True

                            elif isinstance(event_type, str) and "reasoning" in event_type:
                                recorded_reasoning = _record_openai_reasoning_replay_item(event)
                                if recorded_reasoning:
                                    last_openai_reasoning_replay_item = recorded_reasoning
                                reasoning_delta = _extract_reasoning_text(event)
                                if reasoning_delta:
                                    if not reasoning_open:
                                        reasoning_delta = f"<think>{reasoning_delta}"
                                        reasoning_open = True
                                    yield _chunk_with_text(reasoning_delta)
                                    reasoning_emitted = True

                            elif event_type == "response.completed":
                                completed_usage = (event.get("response") or {}).get("usage")
                                if isinstance(completed_usage, dict):
                                    last_usage = completed_usage
                                if pending_marker_tail:
                                    flushed = _flush_pending_marker_tail(pending_marker_tail)
                                    pending_marker_tail = ""
                                    if flushed:
                                        if reasoning_open:
                                            yield _chunk_with_text("</think>")
                                            reasoning_open = False
                                        yield _chunk_with_text(flushed)
                                tail_flushed = _drain_pending_segments(
                                    force = True,
                                )
                                if tail_flushed:
                                    if reasoning_open:
                                        yield _chunk_with_text("</think>")
                                        reasoning_open = False
                                    yield _chunk_with_text(tail_flushed)
                                if reasoning_open:
                                    yield _chunk_with_text("</think>")
                                    reasoning_open = False
                                response_obj = event.get("response") or {}
                                if isinstance(response_obj, dict):
                                    probe_id = response_obj.get("container_id")
                                    if not probe_id:
                                        container_field = response_obj.get("container")
                                        if isinstance(container_field, dict):
                                            probe_id = container_field.get("id")
                                    if (
                                        isinstance(probe_id, str)
                                        and probe_id.startswith("cntr_")
                                        and latched_container_id is None
                                    ):
                                        latched_container_id = probe_id
                                if (
                                    latched_container_id
                                    and not container_id_emitted
                                    and latched_container_id != openai_code_exec_container_id
                                ):
                                    yield _emit_tool_event(
                                        {
                                            "type": "container_ready",
                                            "container_id": latched_container_id,
                                        }
                                    )
                                    container_id_emitted = True
                                if web_search_calls and all_url_citations:
                                    last_id = list(web_search_calls.keys())[-1]
                                    blocks: list[str] = []
                                    for cit in all_url_citations:
                                        line = f"Title: {cit['title']}\nURL: {cit['url']}"
                                        if cit.get("snippet"):
                                            line += f"\nSnippet: {cit['snippet']}"
                                        blocks.append(line)
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": last_id,
                                            "result": "\n---\n".join(blocks),
                                        }
                                    )
                                for sc_id, sc_state in shell_calls.items():
                                    if sc_state.get("tool_end_emitted"):
                                        continue
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": sc_id,
                                            "result": _format_shell_output(
                                                sc_state.get("output") or []
                                            ),
                                        }
                                    )
                                    sc_state["tool_end_emitted"] = True
                                # Return reasoning items for replay beside their function_call (tool-calling turns only).
                                _terminal_delta: dict[str, Any] = {}
                                if saw_function_call and openai_reasoning_replay_items:
                                    _terminal_delta["extra_content"] = {
                                        "openai_responses_reasoning": list(
                                            openai_reasoning_replay_items.values()
                                        )
                                    }
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": _terminal_delta,
                                            "finish_reason": (
                                                "tool_calls" if saw_function_call else "stop"
                                            ),
                                        }
                                    ],
                                }
                                yield f"data: {_json.dumps(chunk)}"
                                usage_line = _build_usage_chunk(
                                    completion_id,
                                    "openai",
                                    last_usage,
                                )
                                if usage_line:
                                    yield usage_line

                            elif event_type == "response.incomplete":
                                incomplete_response = dict(event.get("response") or {})
                                incomplete_response.setdefault("status", "incomplete")
                                incomplete_usage = incomplete_response.get("usage")
                                if isinstance(incomplete_usage, dict):
                                    last_usage = incomplete_usage
                                if pending_marker_tail:
                                    flushed = _flush_pending_marker_tail(pending_marker_tail)
                                    pending_marker_tail = ""
                                    if flushed:
                                        if reasoning_open:
                                            yield _chunk_with_text("</think>")
                                            reasoning_open = False
                                        yield _chunk_with_text(flushed)
                                tail_flushed = _drain_pending_segments(
                                    force = True,
                                )
                                if tail_flushed:
                                    if reasoning_open:
                                        yield _chunk_with_text("</think>")
                                        reasoning_open = False
                                    yield _chunk_with_text(tail_flushed)
                                if reasoning_open:
                                    yield _chunk_with_text("</think>")
                                    reasoning_open = False
                                if web_search_calls and all_url_citations:
                                    last_id = list(web_search_calls.keys())[-1]
                                    blocks = []
                                    for cit in all_url_citations:
                                        line = f"Title: {cit['title']}\nURL: {cit['url']}"
                                        if cit.get("snippet"):
                                            line += f"\nSnippet: {cit['snippet']}"
                                        blocks.append(line)
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": last_id,
                                            "result": "\n---\n".join(blocks),
                                        }
                                    )
                                for sc_id, sc_state in shell_calls.items():
                                    if sc_state.get("tool_end_emitted"):
                                        continue
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": sc_id,
                                            "result": _format_shell_output(
                                                sc_state.get("output") or []
                                            ),
                                        }
                                    )
                                    sc_state["tool_end_emitted"] = True
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": _openai_response_finish_reason(
                                                incomplete_response
                                            ),
                                        }
                                    ],
                                }
                                yield f"data: {_json.dumps(chunk)}"
                                usage_line = _build_usage_chunk(
                                    completion_id,
                                    "openai",
                                    last_usage,
                                )
                                if usage_line:
                                    yield usage_line

                            elif event_type in ("response.failed", "error"):
                                yield _error_sse_line(
                                    502,
                                    _openai_response_error_message(event),
                                    self.provider_type,
                                )
                                break
                    except GeneratorExit:
                        await response.aclose()
                        await lines_gen.aclose()
                        raise
                    finally:
                        web_search_requested = bool(enabled_tools and "web_search" in enabled_tools)
                        web_search_invocations = len(web_search_calls)
                        total_citations = len(all_url_citations)
                        queries = [
                            sc["query"] for sc in web_search_calls.values() if sc.get("query")
                        ]
                        cached_input_tokens = None
                        if isinstance(last_usage, dict):
                            details = last_usage.get("input_tokens_details")
                            if isinstance(details, dict):
                                cached_input_tokens = details.get("cached_tokens")
                        code_execution_requested = code_execution_enabled_openai
                        code_execution_invocations = len(shell_calls)
                        code_execution_results = sum(
                            1 for sc in shell_calls.values() if sc.get("output") is not None
                        )
                        logger.info(
                            "OpenAI Responses stream complete (model=%s, "
                            "web_search_requested=%s, web_search_invocations=%s, "
                            "citations=%s, queries=%s, reasoning_emitted=%s, "
                            "code_execution_requested=%s, "
                            "code_execution_invocations=%s, "
                            "code_execution_results=%s, "
                            "container_id_in=%s, container_id_out=%s, "
                            "input_tokens=%s, output_tokens=%s, "
                            "cached_input_tokens=%s)",
                            model,
                            web_search_requested,
                            web_search_invocations,
                            total_citations,
                            queries,
                            reasoning_emitted,
                            code_execution_requested,
                            code_execution_invocations,
                            code_execution_results,
                            openai_code_exec_container_id,
                            latched_container_id,
                            (last_usage or {}).get("input_tokens"),
                            (last_usage or {}).get("output_tokens"),
                            cached_input_tokens,
                        )
                        await response.aclose()
                        await lines_gen.aclose()
                    return

        except httpx.ConnectError as exc:
            logger.error("Connection error to %s: %s", self.provider_type, exc)
            error_line = _error_sse_line(
                502,
                f"Failed to connect to {self.provider_type}: {exc}",
                self.provider_type,
            )
            yield error_line if stream else error_line.removeprefix("data: ")
        except httpx.ReadTimeout as exc:
            logger.error("Read timeout from %s: %s", self.provider_type, exc)
            error_line = _error_sse_line(
                504,
                f"Timeout waiting for {self.provider_type} response",
                self.provider_type,
            )
            yield error_line if stream else error_line.removeprefix("data: ")
        except httpx.HTTPError as exc:
            logger.error("HTTP error from %s: %s", self.provider_type, exc)
            error_line = _error_sse_line(
                502,
                f"Error communicating with {self.provider_type}: {exc}",
                self.provider_type,
            )
            yield error_line if stream else error_line.removeprefix("data: ")

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        top_p: Optional[float] = 0.95,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full response dict. Only valid for
        OpenAI-compatible providers: Anthropic requires its own Messages API, so use
        stream_chat_completion (with stream=False) if a non-streaming Anthropic path is needed
        later."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
        }
        if top_p is not None:
            body["top_p"] = top_p
        if max_tokens is not None:
            if self.provider_type == "openai":
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens

        url = f"{self.base_url}/chat/completions"
        response = await _client().post(
            url,
            json = body,
            headers = self._auth_headers(),
            timeout = self._timeout,
        )
        if "max_tokens" in body and _rejects_max_tokens(response.status_code, response.text):
            response = await _client().post(
                url,
                json = _with_max_completion_tokens(body),
                headers = self._auth_headers(),
                timeout = self._timeout,
            )
        response.raise_for_status()
        return response.json()

    async def create_speech(
        self,
        text: str,
        model: str,
        voice: Optional[str] = None,
        response_format: str = "wav",
        speed: Optional[float] = None,
        instructions: Optional[str] = None,
    ) -> tuple[bytes, str]:
        """POST /audio/speech (OpenAI CreateSpeech). Returns (audio_bytes, media_type)."""
        body: dict[str, Any] = {
            "model": model,
            "input": text,
            "response_format": response_format,
        }
        if voice:
            body["voice"] = voice
        if speed is not None:
            body["speed"] = speed
        if instructions is not None:
            body["instructions"] = instructions
        response = await _client().post(
            _append_provider_path(self.base_url, "/audio/speech"),
            headers = self._auth_headers(),
            json = body,
            timeout = self._timeout,
        )
        response.raise_for_status()
        media_type = (response.headers.get("content-type") or "").split(";")[0].strip()
        audio = response.content
        if response_format.strip().lower() == "wav":
            merge_cancelled = threading.Event()
            merge_task = asyncio.create_task(
                asyncio.to_thread(_merge_concatenated_wav_segments, audio, merge_cancelled)
            )
            try:
                audio = await asyncio.shield(merge_task)
            except asyncio.CancelledError:
                merge_cancelled.set()
                while not merge_task.done():
                    try:
                        await asyncio.shield(merge_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                try:
                    merge_task.result()
                except BaseException:
                    pass
                raise asyncio.CancelledError
        return audio, media_type or f"audio/{response_format}"

    async def create_transcription(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        model: str,
        language: Optional[str] = None,
        response_format: str = "json",
        timestamp_granularities: Optional[list[str]] = None,
    ) -> tuple[bytes, str]:
        """Post audio to an OpenAI-compatible transcription endpoint."""
        data = {
            "model": model,
            "response_format": response_format,
        }
        if language:
            data["language"] = language
        if timestamp_granularities:
            data["timestamp_granularities[]"] = list(timestamp_granularities)
        headers = self._auth_headers()
        headers.pop("Content-Type", None)
        response = await _client().post(
            f"{self.base_url}/audio/transcriptions",
            headers = headers,
            files = {"file": (filename, audio, content_type)},
            data = data,
            timeout = self._timeout,
        )
        response.raise_for_status()
        media_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip()
        if not media_type:
            media_type = "text/plain" if response_format == "text" else "application/json"
        return response.content, media_type

    async def create_decision(
        self, model: str, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        def as_text(value: Any) -> Any:
            return (
                _json.dumps(value, ensure_ascii = False) if isinstance(value, (dict, list)) else value
            )

        sent = {}
        for name, question in questions.items():
            question = dict(question)
            if "instructions" in question:
                question["instructions"] = as_text(question["instructions"])
            criteria = question.get("criteria")
            if isinstance(criteria, dict):
                question["criteria"] = {key: as_text(value) for key, value in criteria.items()}
            elif isinstance(criteria, list):
                question["criteria"] = [as_text(value) for value in criteria]
            sent[name] = question
        response = await _client().post(
            re.sub(r"/systemone$", "", self.base_url.rstrip("/")) + "/systemone",
            headers = self._auth_headers(),
            json = {"model": model, "state": state, "questions": sent},
            timeout = self._timeout,
        )
        response.raise_for_status()
        return response.json()

    async def list_decision_models(self) -> list[str]:
        response = await _client().get(
            re.sub(r"/systemone$", "", self.base_url.rstrip("/")) + "/models",
            params = {"output_modalities": "decisions"},
            headers = self._auth_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        models = data.get("data") if isinstance(data, dict) else None
        return [
            model["id"]
            for model in (models if isinstance(models, list) else [])
            if isinstance(model, dict)
            and isinstance(model.get("id"), str)
            and isinstance(model.get("architecture"), dict)
            and "decisions" in (model["architecture"].get("output_modalities") or [])
        ]

    async def list_models(self) -> list[dict[str, Any]]:
        """GET /models to discover available models. Returns dicts with at least 'id'. All providers
        expose /models with the OpenAI {"data": [...]} shape, Anthropic included."""
        try:
            response = await _client().get(
                f"{self.base_url}/models",
                headers = self._auth_headers(),
                timeout = self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            models: list[dict[str, Any]] = []
            if isinstance(data, dict):
                raw_models = data.get("data") or []
                if isinstance(raw_models, list):
                    models = [model for model in raw_models if isinstance(model, dict)]
            if self.provider_type == "ollama":
                if not models:
                    models = await self._list_ollama_native_models()
                else:
                    models = await self._with_ollama_capabilities(models)
            if not models and self.provider_type == "gemini":
                models = self._parse_gemini_models(data)
            return models
        except httpx.HTTPError as exc:
            logger.error("Failed to list models from %s: %s", self.provider_type, exc)
            raise

    @staticmethod
    def _parse_gemini_models(payload: Any) -> list[dict[str, Any]]:
        """Translate Gemini's native /v1beta/models payload to OpenAI shape, keeping only entries
        advertising generateContent / streamGenerateContent so embedding-only models do not reach
        the chat picker."""
        if not isinstance(payload, dict):
            return []
        entries = payload.get("models") or []
        if not isinstance(entries, list):
            return []
        out: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            methods = entry.get("supportedGenerationMethods") or []
            if (
                isinstance(methods, list)
                and methods
                and not any(m in methods for m in ("generateContent", "streamGenerateContent"))
            ):
                continue
            base_id = entry.get("baseModelId")
            name = entry.get("name") or ""
            short_id = (
                base_id
                if isinstance(base_id, str) and base_id
                else (name.split("/", 1)[1] if "/" in name else name)
            )
            if not short_id:
                continue
            out.append(
                {
                    "id": short_id,
                    "owned_by": "google",
                    "display_name": entry.get("displayName") or short_id,
                }
            )
        return out

    @staticmethod
    def _ollama_capability_names(entry: dict[str, Any]) -> Optional[list[str]]:
        # None = the row is silent (older Ollama), not "no capabilities".
        raw = entry.get("capabilities")
        if not isinstance(raw, list):
            return None
        return [name for name in raw if isinstance(name, str) and name]

    async def _list_ollama_native_models(self) -> list[dict[str, Any]]:
        """Ollama's /api/tags catalog, with per-model capabilities when reported."""
        root = self.base_url.removesuffix("/v1").rstrip("/")
        response = await _client().get(
            f"{root}/api/tags",
            headers = self._auth_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return []
        raw_models = payload.get("models") or []
        if not isinstance(raw_models, list):
            return []
        models: list[dict[str, Any]] = []
        for entry in raw_models:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("name", "").strip()
            if not model_id:
                continue
            model: dict[str, Any] = {"id": model_id, "owned_by": "ollama"}
            capabilities = self._ollama_capability_names(entry)
            if capabilities is not None:
                model["capabilities"] = capabilities
            models.append(model)
        return models

    async def _with_ollama_capabilities(self, models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            native = await self._list_ollama_native_models()
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("Ollama /api/tags capabilities unavailable: %s", exc)
            return models
        capabilities = {
            entry["id"]: entry["capabilities"]
            for entry in native
            if entry.get("capabilities") is not None
        }
        if not capabilities:
            return models
        merged: list[dict[str, Any]] = []
        for model in models:
            names = capabilities.get(model.get("id", ""))
            merged.append(model if names is None else {**model, "capabilities": names})
        return merged

    async def verify_models_endpoint_lightweight(self) -> None:
        """Confirm GET /models returns 200 without buffering the full response body. Used for
        providers with enormous catalogs (e.g. OpenRouter, Hugging Face router) where downloading
        the full JSON would be prohibitive."""
        url = f"{self.base_url}/models"
        try:
            async with _client().stream(
                "GET",
                url,
                headers = self._auth_headers(),
                timeout = self._timeout,
            ) as response:
                if response.status_code != 200:
                    response.raise_for_status()
                async for _chunk in response.aiter_bytes(chunk_size = 2048):
                    break
        except httpx.HTTPError as exc:
            logger.error(
                "Lightweight /models check failed for %s: %s",
                self.provider_type,
                exc,
            )
            raise

    def _container_headers(self) -> dict[str, str]:
        """Auth headers plus the required ``OpenAI-Beta: containers=v1`` opt-in; without it DELETE
        silently no-ops (returns deleted:true but keeps the container, verified 2026-05-15)."""
        headers = self._auth_headers()
        headers["OpenAI-Beta"] = "containers=v1"
        return headers

    async def list_openai_containers(self) -> list[dict[str, Any]]:
        """GET /v1/containers; returns raw container records (the route reshapes
        them). Only valid against api.openai.com (caller guards is_openai_cloud).
        """
        response = await _client().get(
            f"{self.base_url}/containers",
            headers = self._container_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        containers = data.get("data") if isinstance(data, dict) else None
        result = list(containers) if isinstance(containers, list) else []
        logger.info(
            "openai_container_list.response count=%s items=%s",
            len(result),
            [{"id": c.get("id"), "status": c.get("status")} for c in result if isinstance(c, dict)],
        )
        return result

    async def create_openai_container(self, name: str, ttl_minutes: int) -> dict[str, Any]:
        """POST /v1/containers with ``expires_after.anchor="last_active_at"``. ``ttl_minutes`` is
        the idle timeout -- every API call touching the container resets the timer."""
        body = {
            "name": name,
            "expires_after": {
                "anchor": "last_active_at",
                "minutes": ttl_minutes,
            },
        }
        response = await _client().post(
            f"{self.base_url}/containers",
            json = body,
            headers = self._container_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        return response.json()

    async def delete_openai_container(self, container_id: str) -> None:
        """DELETE /v1/containers/{id}. 404s surface as HTTPError. Uses a fresh httpx client
        (shared-pool DELETEs returned deleted:true but left the container alive), and verifies
        the body reports deleted:true, since OpenAI 2xx-returns that even when silently rejecting
        the request."""
        url = f"{self.base_url}/containers/{container_id}"
        headers = self._container_headers()
        logger.info(
            "openai_container_delete.outbound url=%s has_auth=%s openai_beta=%s",
            url,
            "Authorization" in headers,
            headers.get("OpenAI-Beta"),
        )
        async with httpx.AsyncClient(timeout = self._timeout) as fresh_client:
            response = await fresh_client.delete(url, headers = headers)
        logger.info(
            "openai_container_delete.response status=%s cf_ray=%s "
            "request_id=%s organization=%s project=%s processing_ms=%s body=%s",
            response.status_code,
            response.headers.get("cf-ray"),
            response.headers.get("x-request-id"),
            response.headers.get("openai-organization"),
            response.headers.get("openai-project"),
            response.headers.get("openai-processing-ms"),
            response.text[:300],
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not (isinstance(payload, dict) and payload.get("deleted") is True):
            raise httpx.HTTPError(
                f"OpenAI did not confirm container deletion: {response.text[:200]}"
            )

    async def close(self) -> None:
        """No-op — the underlying client is shared across requests."""


def _provider_display_name(provider_type: str) -> str:
    from core.inference.providers import get_provider_info
    info = get_provider_info(provider_type) or {}
    return str(info.get("display_name") or provider_type)


def _friendly_provider_error_text(
    provider_type: str,
    status_code: int,
    raw_message: str,
    *,
    model: str | None = None,
) -> str:
    """Rewrite common provider errors into actionable Unsloth copy."""
    if status_code == 404 and model:
        lowered = raw_message.lower()
        if "not found" in lowered or "not_found" in lowered:
            if provider_type == "ollama":
                label = _provider_display_name(provider_type)
                return (
                    f"Model '{model}' is not installed in {label}. "
                    f"Run `ollama pull {model}` in a terminal, then retry."
                )
            if provider_type in ("vllm", "llama_cpp"):
                label = _provider_display_name(provider_type)
                return (
                    f"Model '{model}' is not available on the {label} server. "
                    "Check that the server is running and the model is loaded, "
                    "then retry."
                )
    return raw_message


def _readable_provider_error(status_code: int, message: str, provider_type: str) -> str:
    """Reduce an upstream error body to the sentence it carries. OpenAI, Anthropic and Gemini all
    nest the text under `error`, so the raw body would otherwise reach the UI as a JSON blob.
    Non-JSON input (already friendly text) passes through unchanged."""
    import json

    try:
        payload = json.loads(message)
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = None

    text = code = None
    if isinstance(payload, dict):
        error = payload.get("error")
        source = error if isinstance(error, dict) else payload
        text = error if isinstance(error, str) else source.get("message")
        code = next(
            (
                c
                for c in (source.get("code"), source.get("type"), source.get("status"))
                if isinstance(c, str) and c
            ),
            None,
        )
        if not isinstance(text, str) or not text.strip():
            detail = payload.get("detail")
            if isinstance(detail, str):
                text = detail
            elif isinstance(detail, list):
                msgs = [
                    d["msg"]
                    for d in detail
                    if isinstance(d, dict) and isinstance(d.get("msg"), str)
                ]
                text = "; ".join(msgs) or None

    if not isinstance(text, str) or not text.strip():
        raw = " ".join(message.split())[:500] if isinstance(message, str) else ""
        if isinstance(payload, dict) or not raw:
            return f"{provider_type} returned HTTP {status_code} with no error details."
        return raw

    text = text.strip()
    return f"{text} ({code})" if code and code not in text else text


_ANTHROPIC_ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "conflict_error": 409,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


def _apply_fastflowlm_reasoning_controls(
    body: dict[str, Any], enable_thinking: Optional[bool], reasoning_effort: Optional[str]
) -> None:
    """Translate reasoning controls to FastFlowLM's ``think`` field.

    Explicit thinking preserves reasoning on length cutoffs; effort ``none`` disables it.
    """
    effort = (reasoning_effort or "").strip().lower()
    if effort == "none":
        body["think"] = False
        return
    if enable_thinking is not None:
        body["think"] = bool(enable_thinking)
    if effort in ("low", "medium", "high") and body.get("think", True):
        body["reasoning_effort"] = effort


def _bare_json_error_as_sse(line: str) -> Optional[str]:
    try:
        parsed = _json.loads(line)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or "error" not in parsed:
        return None
    error = parsed["error"]
    if not isinstance(error, dict):
        error = {"message": str(error), "type": "provider_error"}
    return "data: " + _json.dumps({"error": error})


def _seconds_or_rate(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _count(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _fastflowlm_timings(usage: Any) -> Optional[dict[str, Any]]:
    """Convert FastFlowLM usage metrics to llama-server timings for the UI and API monitor."""
    if not isinstance(usage, dict):
        return None
    prefill_s = _seconds_or_rate(usage.get("prefill_duration_ttft"))
    decode_s = _seconds_or_rate(usage.get("decoding_duration"))
    if prefill_s is None and decode_s is None:
        return None
    timings: dict[str, Any] = {}
    prompt_tokens = _count(usage.get("prompt_tokens"))
    details = usage.get("prompt_tokens_details")
    cached = _count(details.get("cached_tokens")) if isinstance(details, dict) else None
    if prefill_s is not None:
        timings["prompt_ms"] = prefill_s * 1000.0
        if prompt_tokens is not None:
            # FastFlowLM counts a cached prefix in prompt_tokens but not in its prefill speed.
            timings["prompt_n"] = max(prompt_tokens - (cached or 0), 0)
        rate = _seconds_or_rate(usage.get("prefill_speed_tps"))
        if rate is not None:
            timings["prompt_per_second"] = rate
        if cached is not None:
            timings["cache_n"] = cached
    if decode_s is not None:
        timings["predicted_ms"] = decode_s * 1000.0
        completion_tokens = _count(usage.get("completion_tokens"))
        if completion_tokens is not None:
            timings["predicted_n"] = completion_tokens
        rate = _seconds_or_rate(usage.get("decoding_speed_tps"))
        if rate is not None:
            timings["predicted_per_second"] = rate
    return timings


def _with_fastflowlm_timings(line: str) -> str:
    """Add timings to FastFlowLM's final usage chunk."""
    if '"decoding_duration"' not in line and '"prefill_duration_ttft"' not in line:
        return line
    if not line.startswith("data:"):
        return line
    try:
        chunk = _json.loads(line[len("data:") :])
    except ValueError:
        return line
    if not isinstance(chunk, dict) or "timings" in chunk:
        return line
    timings = _fastflowlm_timings(chunk.get("usage"))
    if timings is None:
        return line
    chunk["timings"] = timings
    return "data: " + _json.dumps(chunk)


def _error_sse_line(
    status_code: int,
    message: str,
    provider_type: str,
    retry_after: str | None = None,
) -> str:
    """Format an error as an SSE data line in OpenAI error format. ``retry_after`` carries the
    upstream Retry-After through: this stream is delivered under a 200, so a client that backs
    off has nowhere else to read the delay from."""
    import json

    error: dict[str, str] = {
        "message": _readable_provider_error(status_code, message, provider_type),
        "type": "provider_error",
        "code": str(status_code),
        "provider": provider_type,
    }
    if retry_after:
        error["retry_after"] = retry_after
    return f"data: {json.dumps({'error': error})}"


def _build_usage_chunk(
    completion_id: str, provider: Literal["anthropic", "openai"], last_usage: Optional[dict]
) -> Optional[str]:
    """Build an OpenAI ``include_usage``-style SSE chunk carrying upstream prompt-cache accounting back
    to the client.

    Emits the standard chunk shape (``choices: []`` + ``usage`` block) so
    ``stream_options={"include_usage": true}`` clients keep working, plus the Anthropic-native
    counts as extra keys: usage.prompt_tokens_details.cached_tokens (both providers),
    usage.cache_creation_input_tokens and usage.cache_read_input_tokens (Anthropic-only).
    Anthropic's ``input_tokens`` excludes the cache buckets, so prompt_tokens sums all three (OpenAI
    Responses already folds cached tokens in). Returns ``None`` when there are no usage numbers to
    report.
    """
    if not isinstance(last_usage, dict):
        return None

    completion_tokens = last_usage.get("output_tokens") or 0

    if provider == "anthropic":
        uncached_input = last_usage.get("input_tokens") or 0
        cache_creation = last_usage.get("cache_creation_input_tokens") or 0
        cache_read = last_usage.get("cache_read_input_tokens") or 0
        # Anthropic reports provider-compaction work as separate usage.iterations entries and excludes
        # it from the top-level input/output counts. Fold those billed iterations into the standard
        # OpenAI-shaped totals so monitors and cost consumers do not undercount the turn.
        compaction_input = last_usage.get("compaction_input_tokens") or 0
        compaction_output = last_usage.get("compaction_output_tokens") or 0
        prompt_tokens = uncached_input + cache_creation + cache_read + compaction_input
        completion_tokens += compaction_output
        if not (prompt_tokens or completion_tokens):
            return None
        usage_block: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cache_read},
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        }
        if compaction_input or compaction_output:
            usage_block["compaction_input_tokens"] = compaction_input
            usage_block["compaction_output_tokens"] = compaction_output
        # Forward the 5m/1h cache-write breakdown so cost calc applies the 2x 1h premium instead of defaulting to 5m
        # on chat-style.
        cc_breakdown = last_usage.get("cache_creation")
        if isinstance(cc_breakdown, dict) and cc_breakdown:
            usage_block["cache_creation"] = cc_breakdown
        speed = last_usage.get("speed")
        if speed in ("fast", "standard"):
            usage_block["speed"] = speed
    else:
        prompt_tokens = last_usage.get("input_tokens") or 0
        details = last_usage.get("input_tokens_details")
        prompt_details = dict(details) if isinstance(details, dict) else {}
        cached = prompt_details.get("cached_tokens") or 0
        prompt_details.setdefault("cached_tokens", cached)
        if not (prompt_tokens or completion_tokens or cached):
            return None
        usage_block = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": prompt_details,
        }
        out_details = last_usage.get("output_tokens_details")
        if isinstance(out_details, dict) and out_details:
            usage_block["completion_tokens_details"] = dict(out_details)
            usage_block["completion_tokens_details"].setdefault("reasoning_tokens", 0)
            usage_block["output_tokens_details"] = out_details

    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": usage_block,
    }
    return f"data: {_json.dumps(chunk)}"
