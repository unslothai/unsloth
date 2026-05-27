# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Async HTTP client for proxying chat completions to external LLM providers.

Most registry providers expose OpenAI-compatible /v1/chat/completions endpoints;
Anthropic uses native Messages API with translation in this client.
"""

import base64
import json as _json
import mimetypes
import re
import time
from typing import Any, AsyncGenerator, Literal, NamedTuple, Optional, Union
from urllib.parse import urlparse

import httpx
import structlog

# Use structlog so INFO-level diagnostics actually surface in the
# studio backend's JSON log stream. The stdlib root logger defaults to
# WARNING and is not configured with handlers, so plain
# `logging.getLogger(__name__).info(...)` was being silently dropped —
# only WARNING/ERROR made it through (because they bypassed the root
# level threshold via uvicorn's stderr capture). All existing call
# sites use printf-style positional args, which structlog accepts.
logger = structlog.get_logger(__name__)


# Claude 4.7 (Opus/Sonnet/Haiku) removed temperature, top_p, and top_k —
# the API returns 400 "<param> is deprecated for this model" if any of
# them is set to a non-default value. The "Sampling parameters removed"
# section of the 4.7 release notes is the authoritative reference:
#   https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7
# 3.x and 4.5/4.6 still accept all three; match the 4-7 line strictly so
# the knobs keep working on earlier families. The trailing -4-7[-.]/EOL
# anchor keeps future versions (e.g. claude-opus-5) unaffected.
def _is_openai_family_cloud(base_url: Optional[str]) -> bool:
    """True iff ``base_url`` points at OpenAI cloud or Azure OpenAI Foundry.

    Anchored to the URL host so an attacker can't bypass the gate with a
    path or subdomain like ``https://evil.com/api.openai.com/v1`` or
    ``https://api.openai.com.attacker.com/v1`` (CodeQL py/incomplete-url-
    substring-sanitization). Used to scope cloud-only Responses-API
    extensions (prompt_cache_retention, context_management compaction,
    container shell tool) that 400 on non-cloud OpenAI-compatible
    servers (ollama / llama.cpp / vLLM).

    Azure Foundry resources are scoped to
    ``<resource-name>.openai.azure.com``; match any subdomain via an
    `endswith` on the lowercased hostname, with the leading dot so
    `openai.azure.com` itself doesn't slip through (there is no
    apex-hosted Azure Foundry endpoint).
    """
    if not base_url:
        return False
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return host == "api.openai.com" or host.endswith(".openai.azure.com")


_ANTHROPIC_4_7_SAMPLING_REMOVED = re.compile(
    r"^claude-(?:opus|sonnet|haiku)-4-7(?:[-.]|$)"
)
_OPENAI_REASONING_SUMMARY_UNSUPPORTED = re.compile(r"^o3(?:[-.]|$)")
_OPENAI_REASONING_STATUSES = {"in_progress", "completed", "incomplete"}


def _openai_image_replay_requires_reasoning(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def _sanitize_openai_reasoning_replay_item(
    item: Any,
) -> Optional[dict[str, Any]]:
    """Return a Responses input-safe reasoning item, if ``item`` is one.

    OpenAI's image-generation docs allow follow-up edits by sending the
    previous ``image_generation_call`` id. Reasoning models can additionally
    require the paired ``reasoning`` output item in manually managed context,
    so keep the public replay fields only and drop everything else.
    """
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
    replay_item: dict[str, Any] = {
        "type": "reasoning",
        "id": item_id,
        "summary": summary_parts,
    }
    status = item.get("status")
    if isinstance(status, str) and status in _OPENAI_REASONING_STATUSES:
        replay_item["status"] = status
    return replay_item


# OpenAI Responses inline citation markers: `citeSOURCE_ID[id2...][LOCATOR]`
# using private-use codepoints (see
# https://developers.openai.com/api/docs/guides/citation-formatting).
# Group 1 holds the delim-separated tokens; each resolvable token expands
# to `[[N]](URL)`, unresolved tokens (locators, unknown ids) drop silently
# so no garbled glyph reaches the renderer.
_OPENAI_CITE_OPEN = "cite"
_OPENAI_CITE_STOP = ""
_OPENAI_CITE_DELIM = ""
_OPENAI_CITATION_MARKER = re.compile(
    f"{_OPENAI_CITE_OPEN}([^{_OPENAI_CITE_STOP}]+){_OPENAI_CITE_STOP}"
)


def _build_citation_lookup(
    url_citations: list[dict[str, Any]],
) -> dict[str, tuple[int, str]]:
    """Map every known ``source_id`` alias to ``(citation_index, url)``.

    Accepts singular ``source_id`` and plural ``source_ids``. First-seen
    wins on alias collision so an earlier citation keeps its number.
    """
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


def _replace_openai_citation_markers(
    text: str,
    url_citations: list[dict[str, Any]],
) -> str:
    """Rewrite `\\ue200cite\\ue202SOURCE_ID[\\ue202LOCATOR]\\ue201` markers into
    `[[N]](URL)` per resolvable id. Multi-source markers expand to one link
    per id; unresolved tokens drop silently. Idempotent on text without
    private-use codepoints.
    """
    if not text or _OPENAI_CITE_STOP not in text:
        return text
    by_source = _build_citation_lookup(url_citations)

    def _sub(match: re.Match[str]) -> str:
        # Try every delim-split token; unresolved tokens drop silently.
        # Handles multi-source (all resolve) and source+locator (only the
        # id resolves, locator drops). Empty result strips the marker.
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
    text: str,
    url_citations: list[dict[str, Any]],
) -> tuple[str, bool]:
    """Like ``_replace_openai_citation_markers`` but also reports whether
    any marker referenced a source_id not yet in ``url_citations``.

    The ``annotation.added`` event for a url_citation typically arrives
    AFTER the delta carrying the marker referencing it. Callers buffer the
    segment until a later event records the annotation; unresolved markers
    are left verbatim so a follow-up pass still parses cleanly.
    """
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
        # Leave the whole marker verbatim if any token is unresolved so the
        # caller can re-run once the late annotation lands; partial emission
        # would lose the unresolved ids once the source text is dropped.
        if any_unresolved:
            has_unresolved = True
            return match.group(0)
        return "".join(rendered)

    return _OPENAI_CITATION_MARKER.sub(_sub, text), has_unresolved


def _split_pending_citation_tail(text: str) -> tuple[str, str]:
    """Split ``text`` into ``(head, pending_tail)`` for streamed deltas.

    A citation marker can straddle two SSE deltas (e.g. delta-1 ends with
    ``\\ue200citetu`` and delta-2 starts with ``rn0view0\\ue201``); the
    unterminated tail is buffered and prepended onto the next delta so the
    rewriter sees a complete marker. ``pending_tail`` is the longest suffix
    starting with ``\\ue200`` and lacking ``\\ue201``; ``head`` is safe to
    emit. Empty tail when ``text`` has no open marker or a fully closed one.
    """
    if not text:
        return text, ""
    last_open = text.rfind("")
    if last_open == -1:
        return text, ""
    # Stop byte after the last open byte means the marker closed in this delta.
    if _OPENAI_CITE_STOP in text[last_open:]:
        return text, ""
    return text[:last_open], text[last_open:]


class _AnthropicThinkingSpec(NamedTuple):
    prefixes: tuple[str, ...]
    kind: Literal["adaptive", "manual"]
    efforts: tuple[str, ...]


_ANTHROPIC_THINKING_SPECS = (
    _AnthropicThinkingSpec(
        prefixes = ("claude-opus-4-7",),
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
)


def _anthropic_thinking_spec(model: str) -> Optional[_AnthropicThinkingSpec]:
    for spec in _ANTHROPIC_THINKING_SPECS:
        if model.startswith(spec.prefixes):
            return spec
    return None


# Anthropic ships date-pinned tool versions per model family. Per the
# tool-reference docs (https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)
# the newer `_20260209` / `_20260120` variants only run on Opus 4.6/4.7
# and Sonnet 4.6 (web_search / web_fetch) or Opus 4.5+ and Sonnet 4.5+
# (code_execution). Sending the new versions to an older model returns
# 400 "tool not supported", and sending the old versions on a new model
# misses the dynamic-filtering and free-with-search pricing path. Pick
# the newest combination the model accepts, falling back to the GA
# (`_20250305` / `_20250910` / `_20250825`) defaults for everything else.
_ANTHROPIC_NEW_WEB_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)
_ANTHROPIC_NEW_CODE_EXEC_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
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


# Anthropic's beta-header flag for code execution does NOT change with
# the tool version -- both `_20250825` and `_20260120` are unlocked by
# the same `code-execution-2025-08-25` header per the upstream docs.
_ANTHROPIC_CODE_EXECUTION_BETA = "code-execution-2025-08-25"


# Anthropic server-side context compaction (beta as of compact-2026-01-12).
# Per the docs, the compaction tool is currently supported on Opus 4.6,
# Opus 4.7, Sonnet 4.6 and Mythos Preview. The beta header is the same
# for every supported model; the dated `compact_20260112` type lives in
# the body's `context_management.edits` array. Anything sent to a model
# outside this prefix list is silently ignored so we don't 400 upstream.
_ANTHROPIC_COMPACTION_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-mythos-preview",
)
_ANTHROPIC_COMPACTION_BETA = "compact-2026-01-12"
_ANTHROPIC_COMPACTION_TYPE = "compact_20260112"
# The docs require the threshold to be at least 50K tokens; lower values
# would 400. We clamp on the way out so a UI slider can't underflow.
_ANTHROPIC_COMPACTION_MIN = 50_000


# Anthropic fast-mode beta (Opus 4.6 / 4.7 only, per
# https://platform.claude.com/docs/en/build-with-claude/fast-mode).
# Mutually exclusive with the Priority service tier.
_ANTHROPIC_FAST_MODE_BETA = "fast-mode-2026-02-01"
_ANTHROPIC_FAST_MODE_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-6",
)


def _anthropic_supports_compaction(model: str) -> bool:
    return model.startswith(_ANTHROPIC_COMPACTION_PREFIXES)


def _anthropic_supports_fast_mode(model: str) -> bool:
    # Require a family boundary ("" or "-") after the prefix so IDs like
    # "claude-opus-4-70" / "claude-opus-4-7b" do not match.
    return any(
        model == p or model.startswith(f"{p}-") for p in _ANTHROPIC_FAST_MODE_PREFIXES
    )


# Cap on ``cited_text`` forwarded in document_citations tool_events;
# keeps SSE bytes bounded on multi-KB cited spans (frontend trims to
# 240 chars anyway).
_CITED_TEXT_MAX_LEN = 512


def _anthropic_citation_key(citation: dict[str, Any]) -> tuple:
    """Stable dedup key for an Anthropic ``citations_delta.citation``.

    Anchor fields vary per type (char_location, page_location,
    content_block_location, search_result_location); both start AND
    exclusive end indices are part of the key so same-start /
    different-end pairs stay distinct. search_result_location keys on
    ``search_result_index`` + ``source`` instead of document_index so
    distinct results with the same source don't collapse. Unknown
    shapes fall back to a stringified copy (more entries, never
    collisions). See
    https://platform.claude.com/docs/en/build-with-claude/citations
    and https://platform.claude.com/docs/en/build-with-claude/search-results.
    """
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
        models = ("mistral-small-latest", "mistral-vibe-cli-latest"),
        style = "reasoning_effort",
        efforts = ("none", "high"),
    ),
)

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
    """
    Translate generic reasoning controls into Mistral's model-specific shape.

    Current contract:
      - magistral-medium-latest: baseline (no extra field) or
        `prompt_mode="reasoning"` for the explicit reasoning mode.
      - mistral-small-latest / mistral-vibe-cli-latest:
        `reasoning_effort` in {"none", "high"}.
      - all other tested Mistral models: no reasoning/thinking params.
    """
    model_for_matching = model.rsplit("/", 1)[-1].strip().lower()
    spec = _mistral_thinking_spec(model_for_matching)
    body.pop("prompt_mode", None)
    body.pop("reasoning_effort", None)

    if spec.style == "prompt_mode":
        # Magistral baseline is already reasoning-capable. The explicit
        # prompt_mode path is only used for the "high" UI selection.
        if enable_thinking is True or reasoning_effort == "high":
            body["prompt_mode"] = "reasoning"
        return

    if spec.style == "reasoning_effort":
        if reasoning_effort in spec.efforts:
            body["reasoning_effort"] = reasoning_effort
        elif enable_thinking is False:
            body["reasoning_effort"] = "none"
        elif enable_thinking is True:
            body["reasoning_effort"] = "high"


# Shared client reused across all requests for HTTP connection pooling.
# Auth headers and timeouts are passed per-request, so a single client
# handles every provider without storing credentials.
_http_client = httpx.AsyncClient()


# Cap per-image fetch well below Gemini's ~20 MB total request budget.
_GEMINI_REMOTE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
_GEMINI_REMOTE_IMAGE_TIMEOUT_S = 15.0


def _safe_fetch_image_for_gemini_sync(
    url: str,
    fallback_mime: str,
    max_bytes: int = _GEMINI_REMOTE_IMAGE_MAX_BYTES,
) -> Optional[tuple[str, str]]:
    """Synchronous IP-pinned HTTPS image fetch with SSRF guards.

    Uses the same pinned-IP + SNI pattern as `tools._fetch_page_text` so
    DNS rebinding between validation and the actual connection cannot
    redirect us to a private/metadata address. Follows up to 4 hops,
    re-validating each redirect target. Returns (mime, base64) or None.

    `max_bytes` is clamped to the per-image cap and additionally lets
    the caller pass the remaining per-request budget so an over-budget
    URL is rejected via Content-Length (or read short-circuit) instead
    of being fully downloaded then discarded after the fact.
    """
    import urllib.error
    import urllib.request
    from urllib.parse import urljoin, urlunparse

    # Refuse upfront if the per-request budget is already spent.
    _byte_limit = min(max(0, int(max_bytes)), _GEMINI_REMOTE_IMAGE_MAX_BYTES)
    if _byte_limit <= 0:
        return None

    # Share tools.py's pinned-IP hardening: validate-once-then-pin.
    from .tools import (
        _NoRedirect,
        _SNIHTTPSHandler,
        _validate_and_resolve_host,
    )

    def _safe_parse_https(raw_url: str) -> Optional[tuple[Any, str, int]]:
        """Validate https + hostname + port. Returns (parsed, host, port) or
        None. Handles malformed-port and malformed-bracketed-IPv6 URLs that
        would otherwise raise ValueError mid-build.
        """
        try:
            parsed_url = urlparse(raw_url)
            host_value = parsed_url.hostname
            port_value = parsed_url.port or 443
        except (ValueError, UnicodeError) as _err:
            logger.info(
                "Gemini image fetch: refusing malformed url err=%s",
                type(_err).__name__,
            )
            return None
        scheme_value = (parsed_url.scheme or "").lower()
        if scheme_value != "https":
            logger.info(
                "Gemini image fetch: refusing non-https scheme=%s",
                scheme_value,
            )
            return None
        if not host_value:
            logger.info("Gemini image fetch: refusing url with no hostname")
            return None
        return parsed_url, host_value, port_value

    parsed_info = _safe_parse_https(url)
    if parsed_info is None:
        return None
    parsed, current_host, current_port = parsed_info
    current_url = url
    ok, reason, pinned_ip = _validate_and_resolve_host(current_host, current_port)
    if not ok:
        logger.warning(
            "Gemini image fetch: refusing host=%s reason=%s",
            current_host,
            reason,
        )
        return None

    for _hop in range(4):
        # Pin to validated IP; SNI + cert still use hostname via _SNIHTTPSHandler.
        cp_info = _safe_parse_https(current_url)
        if cp_info is None:
            return None
        cp, _cp_host, _cp_port = cp_info
        ip_str = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
        ip_netloc = f"{ip_str}:{cp.port}" if cp.port else ip_str
        pinned_url = urlunparse(cp._replace(netloc = ip_netloc))

        opener = urllib.request.build_opener(
            _NoRedirect,
            _SNIHTTPSHandler(current_host),
        )
        req = urllib.request.Request(
            pinned_url,
            headers = {"Host": current_host},
            method = "GET",
        )

        try:
            resp = opener.open(req, timeout = _GEMINI_REMOTE_IMAGE_TIMEOUT_S)
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                logger.info(
                    "Gemini image fetch: status=%d host=%s",
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
                    "Gemini image fetch: refusing malformed redirect err=%s",
                    type(_err).__name__,
                )
                return None
            rp_info = _safe_parse_https(current_url)
            if rp_info is None:
                return None
            _rp, current_host, current_port = rp_info
            ok2, reason2, pinned_ip = _validate_and_resolve_host(
                current_host, current_port
            )
            if not ok2:
                logger.warning(
                    "Gemini image fetch: refusing redirect host=%s reason=%s",
                    current_host,
                    reason2,
                )
                return None
            continue
        except (urllib.error.URLError, OSError) as _err:
            logger.warning(
                "Gemini image fetch failed host=%s err=%s",
                current_host,
                type(_err).__name__,
            )
            return None

        with resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status != 200:
                logger.info(
                    "Gemini image fetch: status=%s host=%s", status, current_host
                )
                return None
            _hdr_mime = (
                (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            )
            # Declared non-image MIME is a refusal; missing MIME falls back to caller's.
            if _hdr_mime and not _hdr_mime.startswith("image/"):
                logger.info(
                    "Gemini image fetch: non-image content-type=%s host=%s",
                    _hdr_mime,
                    current_host,
                )
                return None
            _final_mime_pre = _hdr_mime if _hdr_mime else fallback_mime
            if not isinstance(_final_mime_pre, str) or not _final_mime_pre.startswith(
                "image/"
            ):
                logger.info(
                    "Gemini image fetch: missing content-type and no image fallback host=%s",
                    current_host,
                )
                return None
            _hdr_len = resp.headers.get("content-length")
            if _hdr_len and _hdr_len.isdigit() and int(_hdr_len) > _byte_limit:
                logger.info(
                    "Gemini image fetch: declared %s bytes exceeds cap=%s host=%s",
                    _hdr_len,
                    _byte_limit,
                    current_host,
                )
                return None
            # Read cap+1 to detect oversize without buffering unbounded data.
            raw = resp.read(_byte_limit + 1)
            if len(raw) > _byte_limit:
                logger.info(
                    "Gemini image fetch: streamed bytes exceed cap=%s host=%s",
                    _byte_limit,
                    current_host,
                )
                return None
            return _final_mime_pre, base64.b64encode(raw).decode("ascii")

    logger.info("Gemini image fetch: too many redirects host=%s", current_host)
    return None


async def _safe_fetch_image_for_gemini(
    url: str,
    fallback_mime: str,
    max_bytes: int = _GEMINI_REMOTE_IMAGE_MAX_BYTES,
) -> Optional[tuple[str, str]]:
    """Async wrapper running the IP-pinned fetch on a worker thread.

    SSRF guards (https only, pinned IP, per-hop redirect re-check, size
    cap, image/* content-type) live in the sync helper. `max_bytes`
    carries the remaining per-request budget so over-budget URLs are
    rejected up front.
    """
    import asyncio

    return await asyncio.to_thread(
        _safe_fetch_image_for_gemini_sync, url, fallback_mime, max_bytes
    )


# Synthetic-tool names stamped onto outbound _toolEvent.arguments so the
# frontend can distinguish provider-side cards from real user-declared
# tools of the same name. Mirrored on the TS side.
_SERVER_SIDE_BUILTIN_TOOL_NAMES = frozenset(
    {"web_search", "web_fetch", "code_execution", "image_generation"}
)


def _stamp_server_tool_marker(payload: dict[str, Any]) -> None:
    """Tag synthetic provider-side tool events so the frontend can
    distinguish them from real user-declared / local function tools of
    the same name. The marker rides on `arguments._server_tool` and is
    only added for known server-side builtin names; user-supplied
    tool calls echoed back through these helpers (e.g. Kimi
    `$web_search`) keep their existing shape because we keep this scoped
    to the canonical builtin names.
    """
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
    synthetic_chunk_fn: Any,
    tool_call_id: str,
    citations: list[dict[str, str]],
) -> str:
    """Format Kimi web_search citations into the tool_end payload.

    Same shape parseSourcesFromResult on the frontend expects for the
    other built-in web_search providers: `Title: ...\\nURL: ...\\n
    Snippet: ...\\n---\\n...`. If no citations were emitted, fall back
    to a generic "(search complete)" string so the UI still shows the
    tool card transitioning to a completed state.
    """
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


class ExternalProviderClient:
    """Async proxy for OpenAI-compatible external LLM APIs."""

    def __init__(
        self,
        provider_type: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
    ):
        self.provider_type = provider_type
        self.base_url = base_url.rstrip("/")
        # Strip a legacy `/openai` suffix from Google-hosted bases so
        # configs saved before the native switch still route correctly.
        # Custom proxy paths ending in `/openai` are left untouched.
        if self.provider_type == "gemini":
            _parsed_base = urlparse(self.base_url)
            if (
                (_parsed_base.hostname or "").lower()
                == "generativelanguage.googleapis.com"
                and _parsed_base.path.rstrip("/") == "/v1beta/openai"
            ):
                self.base_url = self.base_url[: -len("/openai")]
        self.api_key = api_key
        self._timeout = httpx.Timeout(timeout, connect = 10.0)
        # Disable read timeout on SSE streams: reasoning-heavy models
        # pause tens of seconds between bytes while thinking, and httpx's
        # read timeout is the per-byte gap, not wall clock. connect/write
        # bounds still surface real network failures.
        self._stream_timeout = httpx.Timeout(timeout, connect = 10.0, read = None)

    def _auth_headers(self) -> dict[str, str]:
        """Build authentication headers using the provider's registry config."""
        from core.inference.providers import get_provider_info

        provider_info = get_provider_info(self.provider_type) or {}
        auth_header = provider_info.get("auth_header", "Authorization")
        auth_prefix = provider_info.get("auth_prefix", "Bearer ")

        # Non-Google Gemini bases (LiteLLM, custom gateways) use OAI-compat
        # Bearer auth, not Google's x-goog-api-key. Override the registry default.
        if self.provider_type == "gemini":
            _host = (urlparse(self.base_url).hostname or "").lower()
            if _host != "generativelanguage.googleapis.com":
                auth_header = "Authorization"
                auth_prefix = "Bearer "

        headers = {"Content-Type": "application/json"}
        # Skip auth header when api_key is empty (optional for local providers);
        # httpx rejects an empty `Bearer ` value as "Illegal header value".
        if self.api_key:
            headers[auth_header] = f"{auth_prefix}{self.api_key}"
        # Merge any provider-specific extra headers (e.g. anthropic-version, OpenRouter attribution)
        headers.update(provider_info.get("extra_headers", {}))
        return headers

    def _is_openai_compatible(self) -> bool:
        """Return False for providers that need request/response translation (e.g. Anthropic)."""
        from core.inference.providers import get_provider_info

        info = get_provider_info(self.provider_type) or {}
        # Google-hosted Gemini uses the native translator; non-Google
        # bases stay on OAI-compat so LiteLLM / custom proxies still work.
        if self.provider_type == "gemini":
            _host = (urlparse(self.base_url).hostname or "").lower()
            if _host != "generativelanguage.googleapis.com":
                return True
        return info.get("openai_compatible", True)

    async def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        top_k: Optional[int] = None,
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
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """
        Yield OpenAI-format SSE lines from the external provider.

        For OpenAI-compatible providers, lines are forwarded verbatim.
        For Anthropic, the native Messages API SSE is translated to OpenAI format.

        ``top_k`` and ``presence_penalty`` are forwarded only when the caller
        supplies a value the provider accepts — the frontend's
        provider-capability map already filters these per provider, so we
        treat them as opt-in here.

        ``fast_mode`` only applies to Anthropic Opus 4.6 / 4.7 (silently
        dropped elsewhere); adds the beta header and ``speed: "fast"``.
        """
        # tool_choice="none" hard-disables hosted/builtin tools across
        # every provider so enabled_tools cannot accidentally bill or leak.
        tool_choice_disabled = (
            isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"
        )

        if not self._is_openai_compatible():
            # Gemini speaks its own native REST shape (contents/parts);
            # `_stream_gemini` translates request/response into the OpenAI
            # Chat Completions chunk format the rest of Studio expects.
            # API reference: https://ai.google.dev/gemini-api/docs
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
            ):
                yield line
            return

        # OpenAI moved their flagship models (gpt-5.x) off /v1/chat/completions
        # — those endpoints return 404 with "This is not a chat model" for the
        # new families. Route all OpenAI traffic through /v1/responses instead;
        # we translate the Responses SSE back into Chat Completions chunks so
        # the frontend stays endpoint-agnostic.
        if self.provider_type == "openai":
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
            ):
                yield line
            return

        # Kimi $web_search needs a 2-call round-trip + thinking off; route
        # to a helper. Forced-function tool_choice suppresses it.
        # https://platform.kimi.ai/docs/guide/use-web-search
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

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
        }
        if max_tokens is not None:
            # OpenAI newer models (gpt-4o, gpt-5.x) reject max_tokens
            if self.provider_type == "openai":
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens

        # Drop fields the registry flags as unusable so reasoning-class
        # models with fixed defaults (Kimi k2.6 etc) don't 400 on pydantic
        # default values that the route layer still fills in.
        from core.inference.providers import get_provider_info

        provider_info = get_provider_info(self.provider_type) or {}
        for field in provider_info.get("body_omit", ()):
            body.pop(field, None)

        # Kimi thinking is a top-level body field. kimi-k2-thinking is
        # always on (ignore the toggle); kimi-k2.6 defaults on, can be
        # disabled. `keep: all` preserves every chunk for the UI panel.
        if self.provider_type == "kimi" and enable_thinking is not None:
            if model == "kimi-k2-thinking":
                # Always on; ignore client toggle to avoid an API-level reject.
                pass
            elif enable_thinking:
                body["thinking"] = {"type": "enabled", "keep": "all"}
            else:
                body["thinking"] = {"type": "disabled"}
        elif self.provider_type == "mistral":
            _apply_mistral_reasoning_controls(
                body, model, enable_thinking, reasoning_effort
            )
        elif self.provider_type == "vllm" and enable_thinking is not None:
            # vLLM gates thinking via chat_template_kwargs.enable_thinking.
            tpl_kw = body.get("chat_template_kwargs")
            if not isinstance(tpl_kw, dict):
                tpl_kw = {}
            tpl_kw["enable_thinking"] = bool(enable_thinking)
            body["chat_template_kwargs"] = tpl_kw

        # OpenRouter's unified `reasoning` field gates per-model thinking.
        # Some routes (`*_MANDATORY_REASONING_MODELS`) 400 on explicit off.
        # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
        if self.provider_type == "openrouter":
            normalized_or_model = model.strip().lower()
            if reasoning_effort in ("low", "medium", "high"):
                body["reasoning"] = {"effort": reasoning_effort}
            elif enable_thinking is True:
                body["reasoning"] = {"enabled": True}
            elif enable_thinking is False:
                if normalized_or_model in _OPENROUTER_MANDATORY_REASONING_MODELS:
                    body.pop("reasoning", None)
                else:
                    body["reasoning"] = {"enabled": False}

            # OpenRouter web plugin works on every model id including
            # meta-routers (unlike the `:online` suffix). Forced-function
            # tool_choice suppresses it, matching Gemini/Anthropic.
            # https://openrouter.ai/docs/guides/features/plugins/web-search
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
                if not any(
                    isinstance(p, dict) and p.get("id") == "web" for p in plugins
                ):
                    plugins.append({"id": "web"})
                body["plugins"] = plugins
                logger.info(
                    "OpenRouter web_search: attached plugins=[{id: 'web'}] (model=%s)",
                    body.get("model"),
                )

        # Forward OpenAI-style function tools / tool_choice on every
        # OAI-compat route (incl. custom Gemini OpenAI proxies like
        # LiteLLM). Without this, callers that wire user-defined tools
        # silently lose function-calling on non-native providers.
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        url = f"{self.base_url}/chat/completions"
        logger.info(
            "Proxying chat completion to %s (provider=%s, model=%s)",
            url,
            self.provider_type,
            model,
        )

        try:
            async with _http_client.stream(
                "POST",
                url,
                json = body,
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
                        response.status_code, error_text, self.provider_type
                    )
                    return

                # Manual __anext__ (not `async for`) so we can close
                # the response BEFORE lines_gen, avoiding the httpcore
                # 1.0 GeneratorExit -> RuntimeError path on Python 3.13.
                lines_gen = response.aiter_lines().__aiter__()
                # Diagnostic counters for the OAI-compat path; surfaces
                # OpenRouter mid-stream errors that would otherwise be
                # invisible server-side.
                event_counts: dict[str, int] = {}
                chosen_model: Optional[str] = None
                # OpenRouter has no web_search_call events — citations
                # arrive as url_citation annotations. Synthesise a
                # tool_start/tool_end pair to match the OpenAI/Anthropic UX.
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
                    # OpenRouter (and OpenAI Chat Completions web_search)
                    # nest the citation under url_citation; some variants
                    # ship the fields flat on the annotation itself. Accept
                    # both.
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
                            "result": (
                                "\n---\n".join(blocks)
                                if blocks
                                else "(search complete)"
                            ),
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
                                # Emit synthetic tool_end with collected
                                # citations BEFORE forwarding [DONE], so the
                                # tool-card transitions to "complete" in the
                                # UI before the stream closes.
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
                                    # Mid-stream provider error event. OpenRouter
                                    # in particular returns 200 then surfaces the
                                    # actual failure as an SSE error event.
                                    if "error" in parsed:
                                        event_counts["error"] = (
                                            event_counts.get("error", 0) + 1
                                        )
                                        logger.warning(
                                            "%s SSE error event: %s",
                                            self.provider_type,
                                            parsed.get("error"),
                                        )
                                    else:
                                        event_counts["delta"] = (
                                            event_counts.get("delta", 0) + 1
                                        )
                                    # OpenRouter (and most OAI-compat providers)
                                    # report the underlying model that handled
                                    # the request in every chunk's `model` field.
                                    # Latch the first non-empty value so the
                                    # router-picked model surfaces in logs and
                                    # is available to the proxy caller.
                                    if chosen_model is None and isinstance(
                                        parsed.get("model"), str
                                    ):
                                        chosen_model = parsed["model"]
                                    # When the user has web_search on, scan
                                    # every chunk's delta and message
                                    # objects for url_citation annotations.
                                    # Different OpenRouter upstreams place
                                    # them in different spots.
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
                                                    for ann in (
                                                        envelope.get("annotations")
                                                        or []
                                                    ):
                                                        _record_or_url_citation(ann)
                        yield line
                    # Stream ended without [DONE] (some upstreams just close
                    # the connection). Emit tool_end so the card doesn't
                    # stay in "running" forever.
                    if (
                        web_search_active
                        and web_search_tool_started
                        and not web_search_tool_ended
                    ):
                        yield _build_web_search_tool_end()
                        web_search_tool_ended = True
                except GeneratorExit:
                    await response.aclose()  # set PoolByteStream._closed=True FIRST
                    await lines_gen.aclose()  # now safe — aclose() is a no-op
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
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: Optional[int],
    ) -> AsyncGenerator[str, None]:
        """
        Kimi $web_search round-trip.

        Wire flow (per https://platform.kimi.ai/docs/guide/use-web-search):
          1. POST messages with tools=[{type: "builtin_function",
             function: {name: "$web_search"}}] and thinking=disabled.
          2. Stream the first response — accumulate function.arguments
             across tool_call deltas until finish_reason="tool_calls".
             Do NOT forward those tool_call chunks to the client (they
             are an internal protocol step, not user-visible output).
          3. Build a second request: original messages + the assistant
             message carrying the tool_calls + a role=tool message that
             echoes the same arguments back verbatim (per Kimi docs,
             the caller "just needs to submit tool_call.function.arguments
             to Kimi as they are" — the server actually runs the search).
          4. Stream the second response — that is the final answer the
             user sees, with search results already incorporated.

        We synthesize tool_start (with the parsed query) when step (2)
        completes, and tool_end (with any url_citation annotations the
        second stream emits) before [DONE], so the chat UI shows the
        same web-search tool card as the other providers.
        """
        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            # $web_search forbids thinking; sending the toggle silently
            # would have the server reject the request with 400.
            "thinking": {"type": "disabled"},
            "tools": [
                {"type": "builtin_function", "function": {"name": "$web_search"}}
            ],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        # Strip body fields the Kimi registry declares unusable
        # (temperature/top_p — see body_omit in providers.py).
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

        # ---- First call: collect the model's $web_search tool_call ----
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        try:
            async with _http_client.stream(
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
                        response.status_code, error_text, self.provider_type
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

        # If the model decided not to search, fall back to a plain
        # streaming call without the builtin tool. That mirrors the UX
        # of every other provider when web_search is on but the model
        # didn't actually need it.
        search_calls = [
            tc
            for tc in tool_calls_acc.values()
            if tc["function"]["name"] == "$web_search"
        ]
        if not search_calls:
            logger.info(
                "Kimi $web_search: model did not invoke search; "
                "falling back to plain stream"
            )
            fallback_body = dict(body)
            fallback_body.pop("tools", None)
            try:
                async with _http_client.stream(
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
                            response.status_code, error_text, self.provider_type
                        )
                        return
                    # Manual __anext__ loop instead of `async for` — see the
                    # comment in stream_chat_completion for the Python 3.13 +
                    # httpcore 1.0.x GeneratorExit interaction this avoids.
                    lines_gen = response.aiter_lines().__aiter__()
                    try:
                        while True:
                            try:
                                line = await lines_gen.__anext__()
                            except StopAsyncIteration:
                                break
                            if line.strip():
                                yield line
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

        # Synthesize tool_start with the parsed search query so the
        # chat UI's web-search card shows "Searching for: ...".
        first_args_raw = search_calls[0]["function"]["arguments"] or "{}"
        try:
            first_args = _json.loads(first_args_raw)
        except Exception:
            first_args = {}
        # Log the raw arguments so we can confirm the server actually
        # ran the search. The shape is documented loosely but in practice
        # the model emits `{"search_result":{"search_id":...},
        # "usage":{"total_tokens":N}}` — an opaque receipt where N is the
        # token cost of the injected search context. The query string is
        # NOT present; Kimi runs the search server-side during the first
        # call and bakes the results straight into the model's context.
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
        # Kimi's search has already executed server-side by the time the
        # first call returns (the tool_call envelope encodes the search
        # result reference, not a query for us to dispatch). Emit
        # tool_end NOW so the UI's web-search card transitions to
        # "complete" before the second call starts streaming the
        # answer, instead of after — otherwise the card sits in
        # "running" all the way through the answer streaming and the
        # user perceives the model answering before search finishes.
        yield _build_kimi_tool_end(_synthetic_chunk, tool_call_id, [])

        # ---- Second call: echo the tool_calls back and stream answer ----
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
        # Ask the SSE stream to include a final `usage` block so we can
        # see prompt_tokens (which jumps to thousands when the server
        # injects search context). Without this, OpenAI-compat streams
        # omit usage entirely. Kimi follows the same convention.
        followup_body["stream_options"] = {"include_usage": True}
        # Keep the tool definition on the second call so the model can
        # decide to search again mid-turn if needed. Kimi's doc shows
        # the same tools array on every step.

        try:
            async with _http_client.stream(
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
                        response.status_code, error_text, self.provider_type
                    )
                    return

                lines_gen = response.aiter_lines().__aiter__()
                # Diagnostics: latch usage.prompt_tokens from the final
                # chunk. The Kimi docs say search results count toward
                # prompt_tokens, so a big value here is direct evidence
                # the server actually injected results into context.
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
                                    # Scan annotations only for diagnostics —
                                    # Kimi today doesn't emit url_citation, but
                                    # if a future model version starts to we'll
                                    # see the type name in the final log line
                                    # and can wire it into the tool_end payload.
                                    for choice in parsed.get("choices") or []:
                                        if not isinstance(choice, dict):
                                            continue
                                        for envelope in (
                                            choice.get("delta"),
                                            choice.get("message"),
                                        ):
                                            if not isinstance(envelope, dict):
                                                continue
                                            for ann in (
                                                envelope.get("annotations") or []
                                            ):
                                                if isinstance(ann, dict):
                                                    annotation_shapes.add(
                                                        str(ann.get("type") or "?")
                                                    )
                        yield line
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
    ) -> AsyncGenerator[str, None]:
        """
        Call the Anthropic Messages API and translate its SSE to OpenAI format.

        Anthropic SSE event types:
          content_block_delta  → OpenAI chunk with delta.content
          message_delta        → OpenAI chunk with finish_reason
          message_stop         → data: [DONE]
          (all others skipped)
        """
        import json as _json

        # Extract system prompt and translate image_url parts to Anthropic format
        system: Optional[str] = None
        filtered: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                system = (
                    content
                    if isinstance(content, str)
                    else "\n".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
                )
                continue

            content = msg.get("content")
            # OpenAI role="tool" with list content -> Anthropic native
            # tool_result block on a user message. Translating in the
            # string-content branch only (below) leaves the list-content
            # form forwarded as an invalid `role:"tool"` message that
            # Anthropic rejects. Handle both upfront.
            if msg.get("role") == "tool":
                _tr_id = msg.get("tool_call_id") or ""
                if isinstance(content, list):
                    _flat_parts: list[str] = []
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "text"
                            and part.get("text")
                        ):
                            _flat_parts.append(str(part["text"]))
                    _flat_result = "".join(_flat_parts)
                elif content is None:
                    _flat_result = ""
                elif isinstance(content, str):
                    _flat_result = content
                else:
                    _flat_result = _json.dumps(content)
                filtered.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": _tr_id,
                                "content": _flat_result,
                            }
                        ],
                    }
                )
                continue
            if isinstance(content, list):
                # Translate OpenAI multimodal parts -> Anthropic native shapes.
                # - `image_url`     -> `{type:"image", source:...}`
                # - `input_document` -> `{type:"document", source:...}`
                #   (Studio extension; mirrors Anthropic's document block,
                #   which supports PDFs as base64 or URL per
                #   https://platform.claude.com/docs/en/build-with-claude/vision)
                anthropic_parts: list[dict[str, Any]] = []
                for part in content:
                    if part.get("type") == "text":
                        anthropic_parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "compaction":
                        # Round-trip the compaction block. When the
                        # prior assistant turn ran server-side
                        # compaction, that block must land back on this
                        # turn's assistant message so Anthropic skips
                        # re-compaction from scratch. Forward verbatim
                        # under the {type:"compaction", content:"..."}
                        # shape the API expects. See
                        #   https://platform.claude.com/docs/en/build-with-claude/compaction
                        summary = part.get("content") or ""
                        if isinstance(summary, str) and summary:
                            anthropic_parts.append(
                                {"type": "compaction", "content": summary}
                            )
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # data:image/png;base64,<DATA> -> split header and data
                            header, _, b64data = url.partition(",")
                            media_type = (
                                header.split(";")[0].replace("data:", "")
                                or "image/jpeg"
                            )
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
                            # Remote URL -- Anthropic supports url source type natively.
                            # See: https://docs.anthropic.com/en/docs/build-with-claude/vision#url-based-images
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
                        # `input_document` is Studio's normalised content type
                        # for PDFs / docs. The frontend sends either
                        # `{type:"input_document", file_data:"data:application/pdf;base64,..."}`
                        # or `{type:"input_document", file_url:"https://..."}`,
                        # plus optional `filename` and `media_type`.
                        # Translate to Anthropic's native `document` block.
                        url = part.get("file_url") or ""
                        data_uri = part.get("file_data") or ""
                        title = part.get("filename")
                        # Treat any "data:" URI with no actual base64
                        # payload (`data:application/pdf;base64,` or
                        # whitespace-only) as missing so the file_url
                        # branch below can take over. Matches the
                        # OpenAI-side fallback so a malformed inline
                        # payload + valid remote URL still attaches.
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
                                # Opt into Anthropic's natural-citation
                                # pipeline; without this no citations_delta
                                # events fire. See
                                # https://platform.claude.com/docs/en/build-with-claude/citations
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
                # Assistant tool_calls -> Anthropic tool_use blocks
                # appended to the same message. Anthropic native
                # Messages API does not accept OpenAI's top-level
                # `tool_calls` field; the call lives inside a content
                # block with `{type:"tool_use", id, name, input}`.
                if msg.get("role") == "assistant" and isinstance(
                    msg.get("tool_calls"), list
                ):
                    for _tc in msg["tool_calls"]:
                        if not isinstance(_tc, dict):
                            continue
                        _fn = _tc.get("function") or {}
                        if not isinstance(_fn, dict) or not _fn.get("name"):
                            continue
                        _raw = _fn.get("arguments") or "{}"
                        try:
                            _input = (
                                _json.loads(_raw) if isinstance(_raw, str) else _raw
                            )
                        except Exception:
                            _input = {"_raw": _raw}
                        if not isinstance(_input, dict):
                            _input = {"value": _input}
                        anthropic_parts.append(
                            {
                                "type": "tool_use",
                                "id": _tc.get("id") or f"toolu_{time.time_ns()}",
                                "name": _fn["name"],
                                "input": _input,
                            }
                        )
                # Skip whole-message append when nothing usable survived.
                # An empty content array (e.g. user dropped only an unparseable
                # `input_document`) would 400 the Anthropic API with
                # "messages.N.content: at least one block is required".
                if anthropic_parts:
                    filtered.append({"role": msg["role"], "content": anthropic_parts})
            else:
                # role="tool" follow-up -> Anthropic native tool_result
                # block on a `user` message. The OpenAI shape
                # (role=tool, content=string, tool_call_id) is not a
                # valid Anthropic role.
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
                # Assistant turn whose content is a plain string but
                # also carries OpenAI `tool_calls`: convert into a
                # content-array message with a text block + tool_use
                # blocks. Without this, the top-level tool_calls leaks
                # through unchanged.
                if (
                    msg.get("role") == "assistant"
                    and isinstance(msg.get("tool_calls"), list)
                    and msg["tool_calls"]
                ):
                    _text_content = msg.get("content")
                    _blocks: list[dict[str, Any]] = []
                    if isinstance(_text_content, str) and _text_content:
                        _blocks.append({"type": "text", "text": _text_content})
                    for _tc in msg["tool_calls"]:
                        if not isinstance(_tc, dict):
                            continue
                        _fn = _tc.get("function") or {}
                        if not isinstance(_fn, dict) or not _fn.get("name"):
                            continue
                        _raw = _fn.get("arguments") or "{}"
                        try:
                            _input = (
                                _json.loads(_raw) if isinstance(_raw, str) else _raw
                            )
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
                filtered.append(msg)

        # Claude 4.7 family removed temperature / top_p / top_k entirely.
        # The earlier guard only handled top_k; temperature is now also
        # rejected with 400 "temperature is deprecated for this model".
        # Latch the match once and reuse it everywhere temperature or
        # top_k would otherwise be set — including the thinking-mode
        # override below, which used to force temperature=1.
        sampling_removed = bool(_ANTHROPIC_4_7_SAMPLING_REMOVED.match(model))

        body: dict[str, Any] = {
            "model": model,
            "messages": filtered,
            "max_tokens": max_tokens or 1024,  # required by Anthropic
            "stream": True,
        }
        if not sampling_removed:
            body["temperature"] = temperature
        if top_k is not None and top_k > 0 and not sampling_removed:
            body["top_k"] = top_k
        # Anthropic only caches a prefix when at least one cache_control
        # marker is attached to it — the frontend defaults
        # enable_prompt_caching to True for Anthropic, so treat `None` the
        # same as True here (callers that don't set the flag still get
        # caching). Pass False explicitly to opt out.
        prompt_caching_enabled = enable_prompt_caching is not False
        # Optional 1h cache TTL is GA as of 2026-05 (no beta header). 1h
        # writes are 2x vs 5m's 1.25x but reads are 0.1x for both, so 1h
        # wins after a single extra hit. Unknown TTL strings drop.
        cache_marker: dict[str, Any] = {"type": "ephemeral"}
        if prompt_cache_ttl in ("5m", "1h"):
            cache_marker["ttl"] = prompt_cache_ttl

        if system:
            if prompt_caching_enabled:
                # System is the most stable cross-turn prefix; own breakpoint.
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
            # Second breakpoint at the end of the conversation. Anthropic
            # caches the longest matching prefix up to a cache_control
            # marker; placing one on the latest message means turn N+1
            # rehydrates everything up through turn N from cache instead
            # of recomputing it. This is what makes caching actually work
            # when the system prompt is empty or shorter than Anthropic's
            # ~1024-token cache floor — the conversation history carries
            # the bulk of the input tokens. Anthropic allows up to 4
            # breakpoints per request; we use at most 2 (system + tail).
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
                # Don't mutate the caller's list. Rebuild the tail with
                # cache_control attached to the final block so an
                # upstream image-bearing turn still cleanly slots into
                # the cache as part of the conversational prefix.
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
            else ("none", "low", "medium", "high")
        )
        effort = reasoning_effort if reasoning_effort in allowed_efforts else None
        # Claude 4.6 Opus/Sonnet accept top-tier adaptive effort as "max" only;
        # "xhigh" is rejected (supported on Claude 4.7). Map our shared "xhigh"
        # semantic to "max" for 4.6 outbound requests while still accepting
        # both in ``allowed_efforts`` for persisted / cross-provider UI state.
        if effort == "xhigh" and model.startswith(
            ("claude-opus-4-6", "claude-sonnet-4-6")
        ):
            effort = "max"
        if effort is None:
            if enable_thinking is False:
                effort = "none"
            elif enable_thinking is True:
                effort = "medium"
        # Normalize one semantic Thinking control into Anthropic's two model-era
        # APIs: adaptive effort on Claude 4.6/4.7, manual budget_tokens on 4.5.
        if effort and effort != "none":
            # Anthropic rejects top_k whenever thinking is enabled.
            body.pop("top_k", None)
            # Earlier families (4.5/4.6) require temperature=1 when
            # thinking is enabled and forbid top_p in the same request:
            #   "temperature and top_p cannot both be specified for this
            #    model. Please use only one."
            # On Claude 4.7, temperature was removed entirely — sending
            # any value (including 1) returns 400 — so skip the override
            # there and let the model use its default sampling.
            if not sampling_removed:
                body["temperature"] = 1
            body.pop("top_p", None)
            if thinking_spec and thinking_spec.kind == "adaptive":
                # `display` defaults to "omitted" on Claude Opus 4.7 (per the
                # adaptive-thinking docs) — without an explicit opt-in the
                # API emits an empty thinking block plus a signature_delta,
                # so our SSE handler would surface a stray <think></think>
                # and the reasoning panel would stay blank. Force
                # "summarized" so 4.7 streams thinking_delta events like
                # 4.6 does. On 4.6 / Sonnet 4.6 this is the default, so
                # setting it explicitly is harmless.
                body["thinking"] = {"type": "adaptive", "display": "summarized"}
                # Per the Messages API reference, the effort knob for
                # adaptive thinking lives under `output_config.effort` —
                # NOT as a top-level field. Sending `effort: ...` directly
                # produces a 400 "effort: Extra inputs are not permitted".
                # Allowed values: low | medium | high | xhigh | max. See:
                # https://platform.claude.com/docs/en/api/messages
                body["output_config"] = {"effort": effort}
            elif thinking_spec and thinking_spec.kind == "manual":
                budget_tokens = {"low": 1024, "medium": 2048, "high": 4096}[effort]
                body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
                # Anthropic requires max_tokens to be strictly greater than
                # thinking.budget_tokens on the manual-thinking path.
                if body.get("max_tokens", 0) <= budget_tokens:
                    body["max_tokens"] = budget_tokens + 1024

        # tool_choice="none" or pinned-function suppresses hosted tools
        # so a stale UI toggle can't fire server-side search/code-exec.
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
            not _anthropic_tool_choice_disabled
            and not _anthropic_tool_choice_forced_function
        )

        # Anthropic web_search (date-pinned per model family).
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
        if (
            _anthropic_hosted_builtins_allowed
            and enabled_tools
            and "web_search" in enabled_tools
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

        # Anthropic web_fetch: only URLs already in conversation. Date-pinned.
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
        web_fetch_enabled = bool(
            _anthropic_hosted_builtins_allowed
            and enabled_tools
            and "web_fetch" in enabled_tools
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

        # Anthropic server-side code execution — see
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
        # Date-pinned tool type per model; both unlock via the same
        # `code-execution-2025-08-25` beta header set below.
        code_execution_enabled = bool(
            _anthropic_hosted_builtins_allowed
            and enabled_tools
            and "code_execution" in enabled_tools
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
            # Reuse the thread's prior container so filesystem state
            # persists. Stale ids 4xx and clear via container_invalidated.
            if anthropic_code_exec_container_id:
                body["container"] = anthropic_code_exec_container_id

        # Server-side compaction (beta `compact-2026-01-12`). Clamps
        # below-min thresholds to 50K so the request doesn't 400.
        # https://platform.claude.com/docs/en/build-with-claude/compaction
        compaction_active = (
            compaction_threshold is not None
            and compaction_threshold > 0
            and _anthropic_supports_compaction(model)
        )
        if compaction_active and compaction_threshold is not None:
            trigger_value = max(
                int(compaction_threshold),
                _ANTHROPIC_COMPACTION_MIN,
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

        # fast_mode is Opus 4.6/4.7 only; silently drop elsewhere.
        # Incompatible with the Priority service_tier (frontend gate
        # prevents both at once; backend lets Anthropic 400 if combined).
        fast_mode_active = bool(fast_mode) and _anthropic_supports_fast_mode(model)
        if fast_mode_active:
            body["speed"] = "fast"

        url = f"{self.base_url}/messages"
        completion_id = f"chatcmpl-anthropic-{model.replace('/', '-')}"

        # Log outgoing config keys (not messages) to prove which thinking /
        # effort fields actually reached the wire.
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

        # Anthropic stop_reason -> OpenAI finish_reason. `pause_turn`
        # maps to None so the UI doesn't treat a paused server-tool turn
        # as final. `refusal` -> "content_filter" (closest match).
        # https://platform.claude.com/docs/en/api/messages#response-stop-reason
        _finish_reason_map: dict[str, Optional[str]] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "refusal": "content_filter",
            "pause_turn": None,
        }

        logger.info("Proxying Anthropic Messages API to %s (model=%s)", url, model)

        request_headers = self._auth_headers()
        # Merge new beta flags onto whatever the registry contributed.
        existing_beta = request_headers.get("anthropic-beta", "").strip()
        beta_parts = (
            [p.strip() for p in existing_beta.split(",") if p.strip()]
            if existing_beta
            else []
        )
        if code_execution_enabled and _ANTHROPIC_CODE_EXECUTION_BETA not in beta_parts:
            beta_parts.append(_ANTHROPIC_CODE_EXECUTION_BETA)
        if compaction_active and _ANTHROPIC_COMPACTION_BETA not in beta_parts:
            beta_parts.append(_ANTHROPIC_COMPACTION_BETA)
        if fast_mode_active and _ANTHROPIC_FAST_MODE_BETA not in beta_parts:
            beta_parts.append(_ANTHROPIC_FAST_MODE_BETA)
        if beta_parts:
            request_headers["anthropic-beta"] = ",".join(beta_parts)

        try:
            async with _http_client.stream(
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
                    # Stale container detection (mirror of the OpenAI
                    # path). When we sent a `container` field and the
                    # response is 4xx with any hint that the id is
                    # expired / missing, emit container_invalidated so
                    # the chat adapter clears the stored id and the
                    # next turn falls back to auto-create.
                    if (
                        anthropic_code_exec_container_id
                        and 400 <= response.status_code < 500
                    ):
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
                        response.status_code, error_text, self.provider_type
                    )
                    return

                # NOTE: same manual __anext__ loop as stream_chat_completion — see comment there.
                lines_gen = response.aiter_lines().__aiter__()
                thinking_open = False
                # Diagnostic counters for the next time the user reports
                # "no thinking content" — distinguishes "Anthropic never sent
                # thinking_delta" from "frontend didn't render the chunks".
                event_counts: dict[str, int] = {}
                # web_search state. Query streams via input_json_delta
                # on a server_tool_use block; results land in a separate
                # web_search_tool_result block. Per-call citations.
                current_server_tool_use: Optional[dict[str, Any]] = None
                current_result_block: Optional[dict[str, Any]] = None
                web_search_calls: dict[str, dict[str, Any]] = {}
                # code_execution state (bash / text_editor sub-tools);
                # kept parallel to web_search so concurrent pills don't collide.
                current_code_exec_use: Optional[dict[str, Any]] = None
                current_code_exec_result: Optional[dict[str, Any]] = None
                code_execution_calls: dict[str, dict[str, Any]] = {}
                # web_fetch state. Same server_tool_use → *_tool_result
                # block shape as web_search but the server_tool_use
                # carries name="web_fetch" and the result block is
                # `web_fetch_tool_result` with content.type=
                # `web_fetch_result` (success) or `web_fetch_tool_error`
                # (failure). Kept separate from web_search state so a
                # turn that uses both does not collide.
                current_web_fetch_use: Optional[dict[str, Any]] = None
                current_web_fetch_result: Optional[dict[str, Any]] = None
                web_fetch_calls: dict[str, dict[str, Any]] = {}
                # Compaction state. Server-side compaction emits a
                # `{type:"compaction", content:"..."}` content block
                # whenever it runs. The summary text can land on the
                # start event AND/OR via text_delta events on the same
                # block (Anthropic's wire format is permissive here).
                # Accumulate in `current_compaction["content"]` and emit
                # on content_block_stop so the chat-adapter can persist
                # it onto the assistant message for round-tripping on
                # the next turn.
                current_compaction: Optional[dict[str, Any]] = None
                compaction_blocks_seen = 0
                # Document citations from ``citations_delta`` events.
                # Deduped by type-specific anchor key; inline [N] is
                # injected after each cited run, and the full list is
                # forwarded as a synthetic document_citations tool_event
                # on message_stop for the Sources panel.
                document_citations: list[dict[str, Any]] = []
                # Counts surfaced in the final log line so reports of
                # "Code execution did nothing" can be triaged at a
                # glance. generated_files_count is interesting for the
                # future Files API PR — when bash creates files inside
                # the container, they show up as file_id entries on
                # bash_code_execution_result.content, and v1 drops
                # them. Track the count so we know how often it would
                # have mattered.
                code_execution_generated_files = 0
                # Container id captured from `message_start.message.container.id`
                # when code_execution is enabled. Emit a `container_ready`
                # _toolEvent on first sight so the chat adapter persists it
                # on the thread record. Only emitted when the value differs
                # from the inbound id — no churn on reuse.
                latched_container_id: Optional[str] = None
                container_id_emitted = False
                # Cache usage tracking. message_start carries the input
                # accounting (incl. cache_creation_input_tokens and
                # cache_read_input_tokens); message_delta carries cumulative
                # output_tokens. Both are surfaced in the "stream complete"
                # log so prompt caching can be verified per-request without
                # opening the Anthropic dashboard.
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

                def _format_web_search_results(
                    results: list[Any],
                ) -> str:
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
                    """Render a `web_fetch_tool_result.content` payload
                    as the Title / URL / snippet block CodeExecutionToolUI
                    and parseSourcesFromResult already expect from the
                    web_search path.

                    Success shape (text):
                        {type: web_fetch_result, url, retrieved_at,
                         content: {type: document, source: {type: text,
                                   media_type, data}, title?}}
                    Success shape (pdf): source.type=base64 + media_type=
                        application/pdf. We do not surface the base64
                        bytes; the title + url is enough for the source
                        pill, and the model still sees the document
                        contents on its side.
                    Error shape: {type: web_fetch_tool_error, error_code}.
                    """
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
                            # Inline a short text preview so the source
                            # pill carries usable context; skip for PDFs
                            # since the body is base64-encoded.
                            if (
                                media_type.startswith("text/")
                                and isinstance(data, str)
                                and data
                            ):
                                snippet = data[:240].strip()
                    # Frontend parseSourcesFromResult only emits a source
                    # pill when both `Title:` and `URL:` are present, so
                    # fall back to the URL when Anthropic omits the
                    # document title (matches the web_search formatter).
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

                def _format_code_execution_result(
                    inner: dict[str, Any],
                ) -> str:
                    """Render an Anthropic code-execution result block as
                    the preformatted text payload the frontend's
                    CodeExecutionToolUI displays inside a <pre>. Handles
                    bash, text_editor (view/create/str_replace), and the
                    matching error variants.
                    """
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
                        # view: file content; create: is_file_update flag;
                        # str_replace: diff `lines` list. The matching
                        # server_tool_use carries the command + path, but
                        # that's encoded into the tool_start arguments
                        # already — here we only format the result body.
                        if "lines" in inner and isinstance(inner.get("lines"), list):
                            return "\n".join(str(line) for line in inner["lines"])
                        if "is_file_update" in inner:
                            return (
                                "Updated" if inner.get("is_file_update") else "Created"
                            )
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

                        # message_start carries the input-side usage block
                        # including cache_creation_input_tokens and
                        # cache_read_input_tokens. message_delta updates
                        # output_tokens (and may overwrite the input fields
                        # with final values). Merge both into last_usage.
                        if event_type == "message_start":
                            start_usage = (event.get("message") or {}).get("usage")
                            if isinstance(start_usage, dict):
                                last_usage.update(start_usage)

                        if event_type == "content_block_start":
                            content_block = event.get("content_block") or {}
                            block_type = content_block.get("type")
                            block_name = content_block.get("name")
                            if (
                                block_type == "server_tool_use"
                                and block_name == "web_search"
                            ):
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
                                # Anthropic sometimes ships the full results
                                # list on the start event; sometimes deltas
                                # follow. Capture whatever is present and
                                # finalize on content_block_stop.
                                content = content_block.get("content") or []
                                current_result_block = {
                                    "tool_use_id": tool_use_id,
                                    "results": list(content)
                                    if isinstance(content, list)
                                    else [],
                                }
                            elif (
                                block_type == "server_tool_use"
                                and block_name == "web_fetch"
                            ):
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
                                    "bash"
                                    if block_name == "bash_code_execution"
                                    else "text_editor"
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
                                # Anthropic ships the full result content
                                # on the start event for code-exec result
                                # blocks (unlike web_search, which can
                                # split across deltas). Capture it and
                                # finalize on content_block_stop so the
                                # ordering matches the web_search path.
                                tool_use_id = content_block.get("tool_use_id", "")
                                inner = content_block.get("content") or {}
                                current_code_exec_result = {
                                    "tool_use_id": tool_use_id,
                                    "inner": inner if isinstance(inner, dict) else {},
                                }
                            elif block_type == "compaction":
                                # Summary may arrive on start AND/OR via
                                # text_delta. Capture both; emit on stop.
                                seed = content_block.get("content") or ""
                                current_compaction = {
                                    "content": seed if isinstance(seed, str) else "",
                                }

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            delta_type = delta.get("type")
                            if delta_type == "thinking_delta":
                                # Wrap as <think>...</think> for parseAssistantContent.
                                thinking_text = delta.get("thinking", "")
                                if thinking_text:
                                    if not thinking_open:
                                        thinking_text = f"<think>{thinking_text}"
                                        thinking_open = True
                                    yield _content_chunk(thinking_text)
                            elif delta_type == "text_delta":
                                text = delta.get("text", "")
                                # text_deltas inside a compaction block
                                # carry the summary chunks; route them
                                # into the compaction buffer and DON'T
                                # yield them to the user-visible stream
                                # -- the summary is opaque internal
                                # state, not assistant prose.
                                if current_compaction is not None:
                                    if text:
                                        current_compaction["content"] += text
                                else:
                                    # First text after a thinking block closes the
                                    # <think> tag we opened above. Anthropic emits
                                    # a content_block_stop between blocks, but
                                    # closing on the text_delta transition is more
                                    # forgiving if events arrive out of order.
                                    if thinking_open:
                                        yield _content_chunk("</think>")
                                        thinking_open = False
                                    if text:
                                        yield _content_chunk(text)
                                    # web_search citations: web_search_tool_result.
                                    # User-doc citations: citations_delta below.
                            elif delta_type == "citations_delta":
                                # One citation per event; collapse onto a
                                # numbered footnote list and inject [N]
                                # inline. See
                                # https://platform.claude.com/docs/en/build-with-claude/citations
                                cit = delta.get("citation")
                                if isinstance(cit, dict):
                                    key = _anthropic_citation_key(cit)
                                    idx_for_marker: Optional[int] = None
                                    for idx, existing in enumerate(
                                        document_citations, start = 1
                                    ):
                                        if existing.get("_key") == key:
                                            idx_for_marker = idx
                                            break
                                    if idx_for_marker is None:
                                        document_citations.append({**cit, "_key": key})
                                        idx_for_marker = len(document_citations)
                                    yield _content_chunk(f"[{idx_for_marker}]")
                            elif delta_type == "input_json_delta":
                                # Streamed partial_json carrying tool inputs
                                # — the search query for web_search, or the
                                # command/path/etc. for code execution.
                                # Route to whichever buffer is open. The two
                                # state slots are exclusive in practice
                                # (Anthropic doesn't interleave tool input
                                # streams), but checking both keeps the
                                # dispatch robust if that ever changes.
                                partial = delta.get("partial_json", "")
                                if current_server_tool_use is not None:
                                    current_server_tool_use["buffer"] += partial
                                elif current_code_exec_use is not None:
                                    current_code_exec_use["buffer"] += partial
                                elif current_web_fetch_use is not None:
                                    current_web_fetch_use["buffer"] += partial
                            # signature_delta and any other delta types are
                            # intentionally skipped — they carry trust /
                            # verification metadata, not user-visible content.

                        elif event_type == "content_block_stop":
                            if current_server_tool_use is not None:
                                # End of the server_tool_use block — parse the
                                # accumulated input_json into a query and
                                # emit tool_start. The matching tool_end fires
                                # later when the web_search_tool_result block
                                # closes with the actual results.
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
                                        "arguments": (
                                            {"query": query} if query else {}
                                        ),
                                    }
                                )
                                current_server_tool_use = None
                            elif current_result_block is not None:
                                # End of a web_search_tool_result — emit
                                # tool_end carrying the search results as
                                # Title:/URL: blocks. parseSourcesFromResult
                                # on the frontend lifts these into source
                                # pills at message tail.
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
                                # End of a code-execution server_tool_use —
                                # parse the buffered input_json into a
                                # {command, path, ...} dict and emit
                                # tool_start. The matching tool_end fires
                                # on the result block's content_block_stop.
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
                                    code_execution_calls[tool_use_id]["arguments"] = (
                                        emit_args
                                    )
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
                                # End of a compaction block. Emit it as a
                                # synthetic tool_event so the chat-adapter
                                # can persist the {type:"compaction",
                                # content:"..."} payload onto the
                                # assistant message. The next turn's
                                # request body forwards the content_part
                                # verbatim and Anthropic recognises it
                                # as the prior compaction state.
                                compaction_blocks_seen += 1
                                yield _emit_tool_event(
                                    {
                                        "type": "compaction_block",
                                        "content": current_compaction["content"],
                                    }
                                )
                                current_compaction = None
                            elif current_code_exec_result is not None:
                                # End of a code-execution result block —
                                # format the inner result into the text
                                # payload CodeExecutionToolUI renders.
                                tool_use_id = current_code_exec_result["tool_use_id"]
                                inner = current_code_exec_result["inner"]
                                # Track generated-file count for the
                                # follow-up Files API PR. v1 drops them.
                                if isinstance(inner, dict):
                                    file_blocks = inner.get("content")
                                    if isinstance(file_blocks, list):
                                        for entry in file_blocks:
                                            if isinstance(entry, dict) and entry.get(
                                                "file_id"
                                            ):
                                                code_execution_generated_files += 1
                                result_text = _format_code_execution_result(
                                    inner if isinstance(inner, dict) else {}
                                )
                                if tool_use_id in code_execution_calls:
                                    code_execution_calls[tool_use_id]["result"] = (
                                        result_text
                                    )
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_end",
                                        "tool_call_id": tool_use_id,
                                        "result": result_text,
                                    }
                                )
                                current_code_exec_result = None
                            elif current_web_fetch_use is not None:
                                # End of the web_fetch server_tool_use —
                                # parse the buffered input_json into the
                                # URL the model asked Anthropic to fetch
                                # and emit tool_start. The matching
                                # tool_end fires on the result block's
                                # content_block_stop just below.
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
                                # End of the web_fetch_tool_result —
                                # format Title / URL / snippet for the
                                # frontend source pill and emit tool_end.
                                # `inner` is sanitised to a dict at the
                                # matching content_block_start, and the
                                # formatter always returns a non-empty
                                # string (defaulting to "(fetch complete)"
                                # when no fields are present), so no
                                # extra fallback is needed here.
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
                                # Close the <think> tag when the thinking block
                                # ends, in case no text_delta follows (e.g.
                                # display=omitted on Claude 4.7, or thinking-
                                # only turns).
                                yield _content_chunk("</think>")
                                thinking_open = False

                        elif event_type == "message_delta":
                            delta_usage = event.get("usage")
                            if isinstance(delta_usage, dict):
                                last_usage.update(delta_usage)
                                # Compaction iterations aren't in top-level
                                # input/output_tokens; fold them into
                                # compaction_{input,output}_tokens for billing.
                                iterations = delta_usage.get("iterations")
                                if isinstance(iterations, list):
                                    c_in = 0
                                    c_out = 0
                                    for it in iterations:
                                        if (
                                            isinstance(it, dict)
                                            and it.get("type") == "compaction"
                                        ):
                                            c_in += int(it.get("input_tokens") or 0)
                                            c_out += int(it.get("output_tokens") or 0)
                                    if c_in or c_out:
                                        last_usage["compaction_input_tokens"] = c_in
                                        last_usage["compaction_output_tokens"] = c_out
                            # Anthropic reports the code_execution container
                            # id on `message_delta.delta.container.{id,
                            # expires_at}` (NOT on message_start — at start
                            # the container hasn't been provisioned yet).
                            # Latch on first sight and emit container_ready
                            # only when the value differs from the inbound
                            # id, so steady-state reuse doesn't re-write
                            # the same id to the thread record every turn.
                            delta_obj = event.get("delta") or {}
                            container_obj = delta_obj.get("container")
                            if (
                                isinstance(container_obj, dict)
                                and latched_container_id is None
                            ):
                                probe = container_obj.get("id")
                                if isinstance(probe, str) and probe:
                                    latched_container_id = probe
                            if (
                                latched_container_id
                                and not container_id_emitted
                                and latched_container_id
                                != anthropic_code_exec_container_id
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
                                # `pause_turn` is in-progress, not terminal:
                                # the SSE stream still ends with [DONE] via
                                # message_stop but we skip emitting a
                                # finish_reason="stop" chunk that would
                                # truncate the rendered message in the UI.
                                mapped = _finish_reason_map.get(stop_reason, "stop")
                                # Streaming refusal: emit a visible notice
                                # plus an out-of-band _toolEvent so the
                                # frontend can prune the refused turn.
                                # The mapped finish_reason is
                                # "content_filter" per OpenAI spec.
                                # https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals
                                if stop_reason == "refusal":
                                    logger.warning(
                                        "Anthropic refusal stop_reason (model=%s)",
                                        model,
                                    )
                                    # Drop signal rides _toolEvent (not
                                    # text) so assistant content cannot
                                    # spoof a context reset.
                                    yield _content_chunk(
                                        "\n\n_The response was stopped by "
                                        "Anthropic's safety classifier. Edit "
                                        "or remove the previous turn and try "
                                        "again._"
                                    )
                                    yield _emit_tool_event(
                                        {"type": "anthropic_refusal"}
                                    )
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
                                    yield f"data: {_json.dumps(chunk)}"

                        elif event_type == "message_stop":
                            if thinking_open:
                                yield _content_chunk("</think>")
                                thinking_open = False
                            # Forward document_citations so the Sources
                            # panel can render the inline [N] footnotes.
                            # ``cited_text`` is truncated server-side to
                            # keep SSE bytes bounded on long spans.
                            if document_citations:
                                clean_cits = []
                                for c in document_citations:
                                    entry = {k: v for k, v in c.items() if k != "_key"}
                                    cited = entry.get("cited_text")
                                    if (
                                        isinstance(cited, str)
                                        and len(cited) > _CITED_TEXT_MAX_LEN
                                    ):
                                        entry["cited_text"] = (
                                            cited[:_CITED_TEXT_MAX_LEN] + "…"
                                        )
                                    clean_cits.append(entry)
                                yield _emit_tool_event(
                                    {
                                        "type": "document_citations",
                                        "citations": clean_cits,
                                    }
                                )
                            # Final include_usage-style chunk so callers can
                            # see cache_creation / cache_read without
                            # scraping the server log.
                            usage_line = _build_usage_chunk(
                                completion_id,
                                "anthropic",
                                last_usage,
                            )
                            if usage_line:
                                yield usage_line
                            yield "data: [DONE]"
                            await (
                                response.aclose()
                            )  # set PoolByteStream._closed=True FIRST
                            break
                except GeneratorExit:
                    await response.aclose()  # set PoolByteStream._closed=True FIRST
                    await lines_gen.aclose()  # now safe — aclose() is a no-op
                    raise
                finally:
                    # Surface per-event-type counts + web_search summary so
                    # reports of "no reasoning panel content" / "Search
                    # didn't do anything" can be triaged at a glance.
                    web_search_requested = bool(
                        enabled_tools and "web_search" in enabled_tools
                    )
                    web_search_invocations = len(web_search_calls)
                    total_results = sum(
                        len(sc.get("results") or []) for sc in web_search_calls.values()
                    )
                    queries = [
                        sc["query"]
                        for sc in web_search_calls.values()
                        if sc.get("query")
                    ]
                    # cache_read_input_tokens > 0 on turn N proves the
                    # cache_control marker on the system block is doing
                    # its job — turn 1 will show cache_creation > 0
                    # instead. cache_creation tokens are billed at a
                    # small premium; cache_read tokens are billed at a
                    # discount.
                    code_execution_invocations = len(code_execution_calls)
                    code_execution_results = sum(
                        1
                        for c in code_execution_calls.values()
                        if c.get("result") is not None
                    )
                    web_fetch_requested = web_fetch_enabled
                    web_fetch_invocations = len(web_fetch_calls)
                    web_fetch_urls = [
                        wf["url"] for wf in web_fetch_calls.values() if wf.get("url")
                    ]
                    logger.info(
                        "Anthropic stream complete (model=%s, "
                        "web_search_requested=%s, web_search_invocations=%s, "
                        "results=%s, queries=%s, "
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
    ) -> AsyncGenerator[str, None]:
        """
        Call Google's native Gemini API and translate its streaming
        ``streamGenerateContent`` response into OpenAI Chat Completions
        chunk format.

        Gemini does NOT speak the OpenAI Chat Completions contract on
        its primary endpoint. The wire shape is:

          POST /v1beta/models/{model}:streamGenerateContent?alt=sse
          {
            "contents": [{"role": "user|model", "parts": [{"text": "..."}]}],
            "systemInstruction": {"parts": [{"text": "..."}]},
            "generationConfig": {"temperature": 0.7, "topP": 0.95, "topK": 40,
                                  "maxOutputTokens": 1024},
            "tools": [{"googleSearch": {}}, {"codeExecution": {}}],
            "cachedContent": "<cache name>"  // optional, see caching docs
          }

        Streamed responses are SSE frames carrying partial
        ``GenerateContentResponse`` objects:

          {"candidates": [{"content": {"parts": [{"text": "Hello"}]},
                            "finishReason": "STOP"}],
           "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}

        Image generation uses the same endpoint with model
        ``gemini-2.5-flash-image`` (also called Nano Banana); the
        response carries an ``inlineData`` part with the base64 PNG
        bytes and a ``mimeType``. We surface that through the same
        ``tool_start`` / ``tool_end`` ``image_b64`` envelope the OpenAI
        image_generation path uses, so the chat UI renders the image
        inline with no extra plumbing.

        References:
          - https://ai.google.dev/gemini-api/docs/text-generation
          - https://ai.google.dev/gemini-api/docs/function-calling
          - https://ai.google.dev/gemini-api/docs/grounding
          - https://ai.google.dev/gemini-api/docs/caching
          - https://ai.google.dev/gemini-api/docs/image-generation
        """
        import json as _json

        # Validate the user-controlled model id BEFORE any message
        # translation. A model like `../cachedContents/x` is path-
        # traversal that lands in `/v1beta/cachedContents/...`; rejecting
        # it here also avoids triggering user-controlled outbound fetches
        # (remote image_url inlining) on a request we'll error out
        # anyway. Documented catalog ids match `[A-Za-z0-9._-]+`.
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
            yield _error_sse_line(
                400,
                f"Invalid Gemini model id: {model!r}",
                self.provider_type,
            )
            return

        # Translate OpenAI messages -> Gemini contents. system role
        # promotes to top-level systemInstruction.
        system_text_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        # OpenAI may drop `name` from role="tool" follow-ups. Remember
        # prior function names so functionResponse isn't sent name-less
        # (Gemini 400s on empty names).
        tool_call_names: dict[str, str] = {}
        # tool_call_ids whose assistant card was dropped (synthetic
        # builtin) or already replayed as native parts. Their role="tool"
        # follow-up must be skipped to avoid orphan/duplicate responses.
        _gemini_skip_tool_result_ids: set[str] = set()
        # Per-request image caps. The byte cap counts DECODED bytes; we
        # set it to ~14 MB because base64 expansion + prompt overhead
        # must fit Gemini's ~20 MB request limit.
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
            # Map OpenAI roles to Gemini's two-role contract.
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
                                header.split(";")[0]
                                .replace("data:", "")
                                .strip()
                                .lower()
                                or "image/jpeg"
                            )
                            # Symmetry with the fetched remote image
                            # path, which already rejects non-image
                            # Content-Type. A `data:text/html;base64,...`
                            # URL otherwise lands as Gemini inlineData
                            # with mimeType="text/html" and 400s the
                            # whole request.
                            if not media_type.startswith("image/"):
                                logger.info(
                                    "Gemini inlineData: refusing non-image data URL media_type=%s",
                                    media_type,
                                )
                            elif b64data:
                                # data: URLs share the same caps as fetched
                                # URLs so inline payloads don't bypass them.
                                _data_approx_bytes = (len(b64data) * 3) // 4
                                if (
                                    _remote_image_count
                                    >= _GEMINI_REMOTE_IMAGE_MAX_COUNT
                                ):
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
                            # fileData.fileUri only accepts Files-API URIs
                            # and YouTube; everything else must be downloaded
                            # and inlined. Parse fields explicitly so
                            # attacker URLs like https://evil.com/youtube.com/x
                            # aren't misclassified as YouTube.
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
                                if isinstance(_guessed, str)
                                and _guessed.startswith("image/")
                                else "image/jpeg"
                            )
                            if _is_youtube:
                                # YouTube URIs must use video/mp4; the
                                # default image/jpeg yields a 400.
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
                                # Refuse pre-fetch when the per-request
                                # byte budget is spent; pass the remainder
                                # so over-budget URLs reject on Content-Length.
                                _remaining_bytes = (
                                    _GEMINI_REMOTE_IMAGE_MAX_TOTAL_BYTES
                                    - _remote_image_total_bytes
                                )
                                if _remaining_bytes <= 0:
                                    logger.info(
                                        "Gemini image fetch: per-request byte cap already reached, dropping image",
                                    )
                                else:
                                    # Count attempts before awaiting so
                                    # slow URLs don't each burn the timeout.
                                    _remote_image_count += 1
                                    _fetched = await _safe_fetch_image_for_gemini(
                                        url,
                                        _media_type,
                                        max_bytes = _remaining_bytes,
                                    )
                                    if _fetched is not None:
                                        _final_mime, _b64 = _fetched
                                        # base64 expands ~4/3 — recover bytes from len(_b64).
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
            # Gemini 3 strict function-calling requires text-part
            # thoughtSignatures to be replayed on history; the frontend
            # stows the latest one as
            # extra_content.google.thought_signature on the assistant
            # message and we pin it onto the last text part here.
            if role == "assistant" and parts:
                _msg_extra = msg.get("extra_content") if isinstance(msg, dict) else None
                if isinstance(_msg_extra, dict):
                    _msg_g = _msg_extra.get("google") or {}
                    if isinstance(_msg_g, dict):
                        _msg_sig = _msg_g.get("thought_signature") or _msg_g.get(
                            "thoughtSignature"
                        )
                        if isinstance(_msg_sig, str) and _msg_sig:
                            for _idx in range(len(parts) - 1, -1, -1):
                                if "text" in parts[_idx]:
                                    parts[_idx] = {
                                        **parts[_idx],
                                        "thoughtSignature": _msg_sig,
                                    }
                                    break
            # Translate OpenAI tool_calls into Gemini functionCall parts.
            # code_execution / image_generation replay their native parts
            # (executableCode / codeExecutionResult / inlineData) stowed
            # on extra_content.google.native_part.
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

                    # Replay native Gemini code_execution / image_generation parts
                    # from extra_content.google.native_part, with fallback to
                    # args.google.native_part for OAI-compat round-trips.
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

                    # Synthetic builtin cards (web_search/web_fetch) must
                    # not become fake functionCalls; drop them. Native
                    # code_execution / image_generation replay below.
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
                            or isinstance(
                                (args.get("google") or {}).get("native_part"), dict
                            )
                        )
                    )
                    if _is_synthetic_server_builtin and not (
                        _name_lc in ("code_execution", "image_generation")
                        and isinstance(_native_part, dict)
                    ):
                        # No replayable Gemini native part -- skip
                        # entirely rather than send a fake functionCall.
                        # Also remember this tool_call_id so a matching
                        # role="tool" follow-up does not become an
                        # orphan functionResponse below.
                        if isinstance(tc_id, str) and tc_id:
                            _gemini_skip_tool_result_ids.add(tc_id)
                            tool_call_names.pop(tc_id, None)
                        continue
                    if fn_name in ("code_execution", "image_generation") and isinstance(
                        _native_part, dict
                    ):
                        # code_execution/image_generation history is
                        # replayed as native parts; the matching
                        # role="tool" must be skipped or Gemini sees a
                        # functionResponse with no declared function
                        # name and 400s the turn.
                        if isinstance(tc_id, str) and tc_id:
                            _gemini_skip_tool_result_ids.add(tc_id)
                        # New shape: `native_part.parts` is an ordered list
                        # of full part wrappers, each carrying its own
                        # `thoughtSignature`. This preserves Gemini 3's
                        # strict per-part replay requirement when the
                        # frontend has merged executableCode +
                        # codeExecutionResult + inlineData into the same
                        # tool-call card.
                        _native_parts_list = _native_part.get("parts")
                        if isinstance(_native_parts_list, list):
                            for _entry in _native_parts_list:
                                if isinstance(_entry, dict):
                                    parts.append(_entry)
                            continue
                        # Legacy single-object native_part: fan the shared
                        # thoughtSignature only when one subpart exists;
                        # for code+result, prefer executableCode and drop
                        # the signature elsewhere.
                        _legacy_sig = _native_part.get(
                            "thoughtSignature"
                        ) or _native_part.get("thought_signature")
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

                    # Forward the OpenAI tool_call id into Gemini's
                    # functionCall.id so a follow-up turn that issues
                    # multiple calls to the same function (different
                    # args, same name) can be disambiguated on the
                    # response side. Gemini accepts the field per
                    # https://ai.google.dev/gemini-api/docs/function-calling.
                    function_call_part: dict[str, Any] = {
                        "name": fn_name,
                        "args": args,
                    }
                    if isinstance(tc_id, str) and tc_id:
                        function_call_part["id"] = tc_id
                    # Gemini 3 function-calling requires the prior
                    # thoughtSignature to be echoed back as a sibling
                    # of the functionCall part. The translator stows
                    # it on the assistant tool_call via
                    # `extra_content.google.thought_signature` (see
                    # the inbound emit below).
                    fc_part: dict[str, Any] = {"functionCall": function_call_part}
                    sig = _google_extra.get("thought_signature") or _google_extra.get(
                        "thoughtSignature"
                    )
                    if isinstance(sig, str) and sig:
                        fc_part["thoughtSignature"] = sig
                    parts.append(fc_part)
            if role == "tool":
                # If the matching assistant-side tool_call was either
                # dropped (synthetic server-tool with no native part)
                # or already replayed as Gemini-native parts
                # (code_execution/image_generation native_part), drop
                # the follow-up too. Emitting it as a functionResponse
                # would be orphaned or duplicate the native result.
                _tc_id_for_skip = msg.get("tool_call_id")
                if (
                    isinstance(_tc_id_for_skip, str)
                    and _tc_id_for_skip in _gemini_skip_tool_result_ids
                ):
                    continue
                # OpenAI's role="tool" follow-up carries the function
                # result. Gemini's matching shape is a role="user" turn
                # with a functionResponse part. When the caller dropped
                # ``name``, recover it from the matching assistant
                # tool_call so Gemini doesn't 400 on an empty name.
                tool_name = msg.get("name") or msg.get("tool_name") or ""
                if not tool_name:
                    tc_id = msg.get("tool_call_id")
                    if isinstance(tc_id, str) and tc_id in tool_call_names:
                        tool_name = tool_call_names[tc_id]
                response_payload: Any
                if isinstance(content, list):
                    # OpenAI tool messages may carry list-form content
                    # (`[{"type":"text","text":"..."}]`). Forwarding the
                    # content-part objects verbatim into Gemini's
                    # `functionResponse.response.result` yields
                    # `result:[{"type":"text","text":"..."}]` instead of
                    # the actual tool output text; flatten text parts so
                    # the result mirrors the string-content path.
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
                # Mirror tool_call_id onto functionResponse.id so
                # Gemini can match the result to the originating
                # functionCall when multiple parallel calls were made.
                tc_id = msg.get("tool_call_id")
                if isinstance(tc_id, str) and tc_id:
                    function_response_part["id"] = tc_id
                parts = [{"functionResponse": function_response_part}]
                gemini_role = "user"
            if parts:
                # Gemini expects parallel functionResponses (multiple
                # OpenAI role="tool" messages in a row) to ride on a
                # single user content with multiple functionResponse
                # parts -- the docs show parallel responses grouped
                # together in the next turn. Merge consecutive
                # functionResponse-only user blocks so realistic
                # parallel tool loops round-trip correctly.
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
            body["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_text_parts)}]
            }

        # Generation config -- temperature / topP / topK / maxOutputTokens
        # map straight across. The frontend capability matrix restricts
        # the sliders the UI exposes for Gemini to this set.
        gen_config: dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["topP"] = top_p
        if top_k is not None and top_k > 0:
            gen_config["topK"] = top_k
        # Gemini accepts ``presencePenalty`` on generationConfig with the
        # same sign convention as the OpenAI knob (positive discourages
        # repetition). Forward when the caller bothers to set it.
        if presence_penalty:
            gen_config["presencePenalty"] = presence_penalty
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens

        # Nano Banana image generation. Gemini only accepts
        # `responseModalities: ["TEXT","IMAGE"]` on the image-capable
        # model family (id contains `-image` or `nano-banana`). Text-
        # only models such as `gemini-2.5-flash` 400 on the same body,
        # so only force image mode when the selected model actually
        # supports it -- a stale `enabled_tools=["image_generation"]`
        # on a text model is silently treated as a regular turn.
        # https://ai.google.dev/gemini-api/docs/image-generation
        model_lc = model.lower()
        is_image_picker_model = "-image" in model_lc or "nano-banana" in model_lc
        # tool_choice="none" / forced-function tool_choice must also
        # suppress the implicit image-generation hosted tool. Otherwise
        # an explicit OpenAI-style opt-out (or an explicit user-function
        # pin) still flips `responseModalities=["TEXT","IMAGE"]` on
        # image-tier models and bills for image output.
        _tool_choice_disabled = (
            isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"
        )
        _tool_choice_forced_function = (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
            and bool(tool_choice["function"].get("name"))
        )
        _hosted_builtins_allowed = (
            not _tool_choice_disabled and not _tool_choice_forced_function
        )
        # Image-tier model IDs reject text-only tools (code_execution,
        # user functions) and thinkingConfig regardless of whether the
        # Images pill is on -- those are model-level constraints
        # documented by Google. The pill only controls whether we ask
        # Gemini to actually emit image output via
        # `responseModalities: ["TEXT","IMAGE"]`. Decoupling the two
        # avoids the case where Images is off + Code/Search is on
        # forwards `tools: [{codeExecution: {}}]` plus
        # `thinkingConfig` to an image model and 400s.
        image_tool_requested = bool(
            _hosted_builtins_allowed
            and enabled_tools
            and "image_generation" in enabled_tools
        )
        # Strict tool / thinking strip uses the model-id check.
        is_image_model_strict = is_image_picker_model
        # The actual modality flip only happens when the user opted in.
        is_image_model = is_image_picker_model and image_tool_requested
        if is_image_model:
            gen_config["responseModalities"] = ["TEXT", "IMAGE"]
        elif is_image_picker_model:
            # Force TEXT-only so an image-capable model with Images OFF
            # doesn't still bill for image output.
            gen_config["responseModalities"] = ["TEXT"]

        # Thinking control. Gemini 3 uses thinkingLevel (str), 2.5 uses
        # thinkingBudget (int). Gemini 3 has no full-off; minimum is
        # "minimal" on Flash, "low" on Pro.
        # https://ai.google.dev/gemini-api/docs/thinking
        _GEMINI3_THINKING_PREFIXES = (
            "gemini-3.5-",
            "gemini-3.1-",
            "gemini-3-",
            "gemini-pro-latest",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
        )
        _GEMINI3_PRO_PREFIXES = (
            "gemini-3.5-pro",
            "gemini-3.1-pro",
            "gemini-3-pro",
            "gemini-pro-latest",
        )
        _PRO_THINKING_PREFIXES = ("gemini-2.5-pro",)
        is_gemini3_thinking = any(
            model_lc.startswith(p) for p in _GEMINI3_THINKING_PREFIXES
        )
        is_gemini3_pro = any(model_lc.startswith(p) for p in _GEMINI3_PRO_PREFIXES)
        _is_pro_thinking_only = any(
            model_lc == p or model_lc.startswith(p + "-")
            for p in _PRO_THINKING_PREFIXES
        )
        effort_lc = (reasoning_effort or "").strip().lower()
        if not is_image_model_strict and is_gemini3_thinking:
            # Gemini 3.x thinkingLevel matrix:
            #   3.1+ Pro:    low/medium/high
            #   3 Pro:       low/high (deprecated 2026-03-09)
            #   3.x Flash*:  minimal/low/medium/high
            # Coerce minimal->low on Pro; medium->high on legacy 3-Pro.
            _G3_LEVELS = {"minimal", "low", "medium", "high"}
            level: Optional[str] = None
            if effort_lc in ("none", "off"):
                level = "low" if is_gemini3_pro else "minimal"
            elif effort_lc == "max":
                level = "high"
            elif effort_lc in _G3_LEVELS:
                # Coerce legacy 3-Pro (low/high only) inputs.
                _is_legacy_gemini3_pro = model_lc.startswith(
                    ("gemini-3-pro-preview", "gemini-3-pro")
                ) and not model_lc.startswith(("gemini-3.1-pro", "gemini-3.5-pro"))
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
            # Gemini 2.5 / older: thinkingBudget int. Effort -> budget
            # mirrors the OpenAI minimal/low/medium/high ladder so the
            # existing frontend picker maps cleanly.
            # NOTE: gemini-2.5-flash-lite rejects positive budgets below
            # 512 with HTTP 400, so minimal=512 sits at that floor.
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
                # Pro-tier 2.5 rejects budget=0 (400 "only works in
                # thinking mode"), so coerce to a small positive value.
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

        # Hosted tools: googleSearch (grounding) and codeExecution.
        # Image-mode rejects codeExecution; only Gemini 3 image models
        # accept googleSearch.
        # https://ai.google.dev/gemini-api/docs/grounding
        # https://ai.google.dev/gemini-api/docs/code-execution
        def _gemini_image_model_allows_google_search(_m: str) -> bool:
            return (
                _m.startswith("gemini-3-pro-image")
                or _m.startswith("gemini-3.1-flash-image")
                or _m.startswith("nano-banana-pro")
                or _m.startswith("nano-banana-2")
            )

        google_search_allowed = (
            not is_image_model_strict
            or _gemini_image_model_allows_google_search(model_lc)
        )
        code_execution_allowed = not is_image_model_strict
        text_tools_allowed = not is_image_model_strict
        # tool_choice="none" / forced-function suppresses hosted builtins
        # too, matching the Anthropic / OpenRouter gates.
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
        # OpenAI-style function declarations -> Gemini functionDeclarations.
        # https://ai.google.dev/gemini-api/docs/function-calling#step_1
        # Gemini's Schema accepts only the OpenAPI 3.0 subset documented
        # at https://ai.google.dev/api/caching#Schema; OpenAI's strict
        # tool definitions routinely include `additionalProperties`,
        # `$schema`, `$defs`, `strict`, `examples`, and similar keys
        # which 400 the request as INVALID_ARGUMENT. Strip them
        # recursively before forwarding.
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

        def _resolve_local_schema_ref(
            root: Optional[dict[str, Any]], ref: str
        ) -> Optional[Any]:
            # Walk a `#/foo/bar` JSON pointer against the schema root.
            # Returns None if the pointer doesn't resolve to a dict, so
            # the caller can fall back to the unresolved node.
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
            # Recursively filter to Gemini's OpenAPI 3.0 subset. At a
            # Schema-keyword dict layer we drop keys not in the
            # allowlist; under `properties` the keys are user-defined
            # field names and the values are themselves Schemas; under
            # `items` / `anyOf` the values are also Schemas.
            # OpenAI strict tools commonly use JSON Schema's
            # `"type": ["string", "null"]` form for nullable fields;
            # Gemini's OpenAPI Schema uses `"type": "string"` plus
            # `"nullable": true`. Translate that here.
            if root is None and isinstance(node, dict):
                root = node
            if _seen_refs is None:
                _seen_refs = frozenset()
            if isinstance(node, dict):
                # Pydantic / OpenAI strict tools commonly hoist nested
                # object schemas into `$defs` and reference them via
                # `{"$ref": "#/$defs/Address"}`. Gemini's OpenAPI subset
                # has no $ref and drops anything not in the allowlist,
                # so the referenced shape would vanish if we didn't
                # inline it here. Recurse into the resolved target with
                # local siblings overriding the reference (normal JSON
                # Schema composition), guarding against ref cycles.
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
                        return _sanitize_gemini_schema(
                            _merged, root, _seen_refs | {_ref}
                        )
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
                        # Preserve multi-type unions as anyOf; flattening
                        # to the first non-null type silently drops the
                        # other branches and changes the tool contract.
                        _union_any_of = [
                            {"type": _t} for _t in _non_null if isinstance(_t, str)
                        ]
                for _k, _v in node.items():
                    if _k == "type" and isinstance(_v, list):
                        # Handled below via _flattened_type.
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
                        # Optional[X] / Union[A, B, None]: Pydantic emits
                        # `anyOf: [..., {"type":"null"}]`. Gemini's
                        # OpenAPI subset rejects `"type": "null"` inside
                        # anyOf, so drop the null variant and surface it
                        # via `nullable: true`. If exactly one non-null
                        # branch remains, collapse it inline; otherwise
                        # keep the slim anyOf and mark the field
                        # nullable.
                        _saw_null = any(
                            isinstance(_entry, dict) and _entry.get("type") == "null"
                            for _entry in _v
                        )
                        _non_null_entries = [
                            _entry
                            for _entry in _v
                            if not (
                                isinstance(_entry, dict)
                                and _entry.get("type") == "null"
                            )
                        ]
                        if len(_non_null_entries) == 1 and _saw_null:
                            _inner = _sanitize_gemini_schema(
                                _non_null_entries[0], root, _seen_refs
                            )
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
                        # Lists of plain strings; copy verbatim.
                        cleaned[_k] = _v
                    else:
                        cleaned[_k] = _v
                if _union_any_of is not None and "anyOf" not in cleaned:
                    cleaned["anyOf"] = [
                        _sanitize_gemini_schema(_s, root, _seen_refs)
                        for _s in _union_any_of
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
        # Tool-choice mapping: OpenAI "auto"/"none"/"required"/{name=...}
        # -> Gemini toolConfig.functionCallingConfig.mode + allowedFunctionNames.
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
            elif (
                isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
            ):
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

        # Prompt caching. The Gemini caching contract is "create a
        # CachedContent resource, then pass its name on
        # `cachedContent`". The cache itself is created out of band by
        # the caller via POST /cachedContents; here we forward an
        # explicit cache id when the dispatcher hands us one (a string
        # value on enable_prompt_caching means "use this cache name").
        # https://ai.google.dev/gemini-api/docs/caching
        if isinstance(enable_prompt_caching, str) and enable_prompt_caching:
            body["cachedContent"] = enable_prompt_caching

        # Model id is already validated at the top of _stream_gemini so
        # we never reach a path-traversed URL segment here.
        url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse"
        completion_id = f"chatcmpl-gemini-{model.replace('/', '-')}"

        logger.info(
            "Proxying Gemini streamGenerateContent to %s (model=%s, "
            "tools=%s, image=%s)",
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

        def _text_chunk(
            text: str, extra_content: Optional[dict[str, Any]] = None
        ) -> str:
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
            """Return ``{"google": {"thought_signature": ...}}`` when the
            Gemini stream part carries a `thoughtSignature` we need to
            replay on a follow-up turn (Gemini 3 image editing + tool
            contexts both require an exact signature echo)."""
            sig = part.get("thoughtSignature") or part.get("thought_signature")
            if isinstance(sig, str) and sig:
                return {"google": {"thought_signature": sig}}
            return None

        # Gemini finish reasons -> OpenAI vocabulary. Reference:
        # https://ai.google.dev/api/rest/v1beta/Candidate#FinishReason
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
        # True once any Gemini functionCall part has been emitted so the
        # final finish_reason swaps STOP -> tool_calls (matches the
        # OpenAI Chat Completions contract; an OAI client that sees a
        # tool_calls delta followed by finish_reason="stop" never
        # executes the tool).
        emitted_any_function_call = False
        # web_search_active drives the tool_start / tool_end envelope.
        # Track on whether `googleSearch` was actually forwarded above,
        # not the raw caller intent -- image-mode requests filter the
        # tool out, and emitting a phantom "search complete" card on a
        # turn where Gemini was never told to search confuses the UI.
        web_search_active = any("googleSearch" in t for t in tools_array)
        web_search_tool_id = "gemini_web_search"
        web_search_tool_started = False
        web_search_tool_ended = False
        web_search_citations: list[dict[str, str]] = []
        # Tracks the tool_call_id minted on the most recent
        # executableCode part so the matching codeExecutionResult can
        # close out the same envelope. None between rounds.
        gemini_code_exec_pending_id: Optional[str] = None
        # The most recently emitted code_execution id + result text. Kept
        # *after* the tool_end so a following inline image (matplotlib
        # plot rendered by codeExecution) can attach to the same card
        # via a `__IMAGES__:` marker instead of spawning a separate
        # image_generation event.
        last_code_exec_tool_id: Optional[str] = None
        last_code_exec_result_text: str = ""

        try:
            async with _http_client.stream(
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
                        response.status_code, error_text, self.provider_type
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

                # NOTE: same manual __anext__ loop pattern as the other
                # streaming helpers (see stream_chat_completion for the
                # Python 3.13 + httpcore 1.0.x GeneratorExit ordering).
                lines_gen = response.aiter_lines().__aiter__()
                final_finish_reason: Optional[str] = None
                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip():
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
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

                        # Latch usageMetadata across deltas -- the final
                        # fragment carries the complete totals.
                        usage_meta = event.get("usageMetadata")
                        if isinstance(usage_meta, dict):
                            last_usage = usage_meta

                        # Prompt-level safety block: Gemini ships zero
                        # candidates plus a `promptFeedback.blockReason`
                        # (e.g. SAFETY). The downstream OAI client would
                        # otherwise see an empty successful assistant
                        # response. Surface as a content_filter error
                        # event so the UI can render the block reason.
                        prompt_feedback = event.get("promptFeedback")
                        if isinstance(prompt_feedback, dict) and prompt_feedback.get(
                            "blockReason"
                        ):
                            block_reason = str(prompt_feedback.get("blockReason"))
                            # Close out the synthetic web_search start so
                            # the UI does not show a spinner stuck on
                            # "searching..." after the error toast lands.
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
                            # Citations / grounding metadata.
                            # `groundingMetadata.groundingChunks[].web`
                            # carries `uri` + `title`. Collect for the
                            # tool_end emission at stream close.
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
                                        if any(
                                            c["url"] == u for c in web_search_citations
                                        ):
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
                                content_obj.get("parts")
                                if isinstance(content_obj, dict)
                                else None
                            )
                            if isinstance(parts, list):
                                for part in parts:
                                    if not isinstance(part, dict):
                                        continue
                                    # Text delta. Stow part-level
                                    # `thoughtSignature` on the delta so
                                    # Gemini 3 turns that need an exact
                                    # signature echo round-trip cleanly.
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
                                        # Empty-content part carrying a
                                        # thoughtSignature: emit an empty delta
                                        # so the signature is preserved.
                                        yield _text_chunk(
                                            "",
                                            extra_content = _part_extra,
                                        )
                                    # functionCall -> OpenAI tool_calls
                                    # delta envelope.
                                    fc = part.get("functionCall")
                                    if isinstance(fc, dict):
                                        fc_name = fc.get("name") or ""
                                        fc_args = fc.get("args") or {}
                                        fc_id = (
                                            fc.get("id")
                                            or f"call_{fc_name}_{time.time_ns()}"
                                        )
                                        if fc_id in emitted_function_call_ids:
                                            continue
                                        emitted_function_call_ids.add(fc_id)
                                        # Each distinct functionCall in an
                                        # assistant turn needs its own
                                        # tool_calls[*].index. Consumers
                                        # that reassemble tool_calls by
                                        # index collapse all calls onto
                                        # the same slot when this is
                                        # hardcoded to 0, breaking
                                        # parallel/multi-tool turns.
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
                                        # Gemini 3 function-calling: the
                                        # part-level `thoughtSignature`
                                        # must be echoed back on the
                                        # next turn or the model rejects
                                        # the tool-result envelope. Stow
                                        # it on `extra_content.google`
                                        # so the frontend can persist it
                                        # and our outbound translator
                                        # (below) can replay it.
                                        thought_sig = part.get(
                                            "thoughtSignature"
                                        ) or part.get("thought_signature")
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
                                                    "delta": {
                                                        "tool_calls": [tool_call_delta]
                                                    },
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                        yield f"data: {_json.dumps(tool_chunk)}"
                                    # executableCode + codeExecutionResult
                                    # parts surface as the standard
                                    # code_execution tool_start/tool_end
                                    # envelope (same shape OpenAI and
                                    # Anthropic emit) so the chat
                                    # adapter can render Gemini sandbox
                                    # output through CodeExecutionToolUI.
                                    # https://ai.google.dev/gemini-api/docs/code-execution
                                    exec_code = part.get("executableCode")
                                    if isinstance(exec_code, dict):
                                        code_str = exec_code.get("code") or ""
                                        if code_str:
                                            code_tool_id = (
                                                exec_code.get("id")
                                                or f"gemini_code_exec_{time.time_ns()}"
                                            )
                                            gemini_code_exec_pending_id = code_tool_id
                                            # Stow the raw Gemini part so
                                            # follow-up turns can replay
                                            # the native `executableCode`
                                            # (Gemini rejects a generic
                                            # functionCall echo for code
                                            # execution history).
                                            _exec_thought_sig = part.get(
                                                "thoughtSignature"
                                            ) or part.get("thought_signature")
                                            # Per-part thoughtSignature stays
                                            # bound to its own part (Gemini 3
                                            # rejects shared signatures).
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
                                                                exec_code.get(
                                                                    "language"
                                                                )
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
                                        # Gemini returns
                                        # OUTCOME_OK / OUTCOME_FAILED /
                                        # OUTCOME_DEADLINE_EXCEEDED. Treat
                                        # non-OK outcomes as stderr so the
                                        # UI surfaces the error.
                                        if outcome and outcome != "OUTCOME_OK":
                                            result_text = (
                                                f"[{outcome}]\n{output}".rstrip()
                                            )
                                        else:
                                            result_text = output
                                        # Pair tool_end with the most recent
                                        # executableCode tool_start; fall back
                                        # to exec_result.id then a fresh id.
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
                                    # inlineData: either a Nano Banana
                                    # generation (own card) or a sandbox
                                    # plot attached to the code_execution
                                    # card via the __IMAGES__: marker.
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
                                                and "code_execution"
                                                in (enabled_tools or [])
                                            )
                                            if attached_to_code_exec:
                                                updated_result = (
                                                    last_code_exec_result_text
                                                    + "\n__IMAGES__:"
                                                    + _json.dumps([image_uri])
                                                )
                                                # Stow inlineData so a follow-up
                                                # turn can replay the plot with
                                                # its per-part thoughtSignature.
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
                                                    _plot_part_entry[
                                                        "thoughtSignature"
                                                    ] = _plot_thought_sig
                                                yield _emit_tool_event(
                                                    {
                                                        "type": "tool_end",
                                                        "tool_call_id": (
                                                            last_code_exec_tool_id
                                                        ),
                                                        "result": updated_result,
                                                        "google": {
                                                            "native_part": {
                                                                "parts": [
                                                                    _plot_part_entry
                                                                ],
                                                            },
                                                        },
                                                    }
                                                )
                                                last_code_exec_result_text = (
                                                    updated_result
                                                )
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
                                                # Gemini 3 image edit needs
                                                # the prior thoughtSignature
                                                # echoed on the inline image part.
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
                                                # Stow inlineData so multi-turn
                                                # edits replay the original
                                                # image as native history.
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
                                                    _img_part_entry[
                                                        "thoughtSignature"
                                                    ] = _img_thought_sig
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

                    # End-of-stream emission order: web_search tool_end
                    # (with citations) -> finish_reason chunk -> usage
                    # chunk -> [DONE]. Matches the Anthropic / OpenAI
                    # helpers' contract so the frontend handler does
                    # not need provider-specific ordering knowledge.
                    if (
                        web_search_active
                        and web_search_tool_started
                        and not web_search_tool_ended
                    ):
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
                                    "\n---\n".join(blocks)
                                    if blocks
                                    else "(search complete)"
                                ),
                            }
                        )
                        web_search_tool_ended = True

                    if final_finish_reason:
                        # OpenAI clients trigger tool execution when
                        # finish_reason="tool_calls". Gemini emits
                        # "STOP" even when the turn was a pure
                        # functionCall request, so override after the
                        # fact to match the OAI contract.
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

                    # Map Gemini usageMetadata onto OpenAI include_usage.
                    # thoughtsTokenCount is billed output too — fold it in
                    # so cost calculators don't undercount.
                    if isinstance(last_usage, dict):
                        thought_tokens = last_usage.get("thoughtsTokenCount") or 0
                        candidate_tokens = last_usage.get("candidatesTokenCount") or 0
                        prompt_tokens = last_usage.get("promptTokenCount") or 0
                        # Gemini bills tool-call prompt slices separately
                        # via `toolUsePromptTokenCount`. Fold into input
                        # so total_tokens does not undercount tool turns.
                        tool_use_prompt_tokens = (
                            last_usage.get("toolUsePromptTokenCount") or 0
                        )
                        translated_usage = {
                            "input_tokens": prompt_tokens + tool_use_prompt_tokens,
                            "output_tokens": candidate_tokens + thought_tokens,
                            "input_tokens_details": {
                                "cached_tokens": (
                                    last_usage.get("cachedContentTokenCount") or 0
                                ),
                                "tool_use_prompt_tokens": tool_use_prompt_tokens,
                            },
                            "output_tokens_details": {
                                "reasoning_tokens": thought_tokens,
                            },
                        }
                        usage_line = _build_usage_chunk(
                            completion_id, "openai", translated_usage
                        )
                        if usage_line:
                            yield usage_line

                    yield "data: [DONE]"
                finally:
                    # Close response first so lines_gen.aclose() becomes
                    # a no-op (avoids the httpcore 1.0 GeneratorExit
                    # path and the aclose-never-awaited RuntimeWarning).
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
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        enable_thinking: Optional[bool],
        reasoning_effort: Optional[str],
        enabled_tools: Optional[list[str]] = None,
        enable_prompt_caching: Optional[bool] = None,
        openai_code_exec_container_id: Optional[str] = None,
        compaction_threshold: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Call OpenAI's /v1/responses endpoint and translate its SSE stream back
        into OpenAI Chat Completions chunk format.

        The Responses API uses a different request shape (``input`` instead of
        ``messages``, ``instructions`` for system prompts, ``max_output_tokens``
        for the budget) and emits event-typed SSE frames (e.g.
        ``response.output_text.delta``) rather than chat-completion chunks.
        ``presence_penalty`` / ``top_k`` are not part of the Responses contract
        and are dropped here intentionally.
        """
        import json as _json

        is_openai_cloud = _is_openai_family_cloud(self.base_url)
        image_generation_requested = bool(
            enabled_tools and "image_generation" in enabled_tools and is_openai_cloud
        )

        # Split system messages out into a single `instructions` string and
        # translate user/assistant messages into the Responses input shape.
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []
        # When we drop a server-side builtin `function_call` here, the
        # matching `role="tool"` follow-up must also be dropped --
        # otherwise the outbound body contains an orphan
        # `function_call_output` with no matching `function_call`, which
        # OpenAI Responses can reject or mis-associate.
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

            # OpenAI Responses uses item-shape history for function
            # calling: assistant turns that invoked user tools must
            # serialize each call as a `function_call` input item, and
            # each role="tool" follow-up as a `function_call_output`
            # item keyed by the matching `call_id`. Without this the
            # second turn after a function call sends Chat Completions
            # shape and Responses 400s the request.
            if role == "tool":
                _call_id = msg.get("tool_call_id") or ""
                # If the matching assistant `function_call` was a
                # server-side builtin we already dropped, drop the
                # follow-up too to avoid emitting an orphan
                # `function_call_output`.
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
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": _call_id,
                            "output": _output_text,
                        }
                    )
                continue

            # Assistant turns that returned tool_calls translate each
            # call as a `function_call` item (carrying name + JSON
            # arguments + call_id). Skip builtin server-side cards
            # (canonical builtin name + `args._server_tool` marker)
            # which never round-trip as user functions. We require both
            # checks so a user function literally named `_server_tool`
            # in its argument schema is not dropped.
            _tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
            if role == "assistant" and isinstance(_tool_calls, list):
                # Preserve the prior `response.output` ordering: the
                # model's text precedes its function_call items, and
                # the matching role=tool follow-up arrives AFTER the
                # call. Without this guard, history replay puts
                # function_call -> assistant text -> function_call_output,
                # which can put the tool output after an unrelated
                # assistant message and confuse multi-turn function
                # calling.
                if isinstance(content, str) and content:
                    input_items.append({"role": "assistant", "content": content})
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
                                _asst_parts.append(
                                    {"type": "input_image", "image_url": _u}
                                )
                    if _asst_parts:
                        input_items.append(
                            {"role": "assistant", "content": _asst_parts}
                        )

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
                                if isinstance(_g, dict) and isinstance(
                                    _g.get("native_part"), dict
                                ):
                                    _is_server_builtin = True
                    _call_id_out = _tc.get("id") or f"call_{time.time_ns()}"
                    if _is_server_builtin:
                        skipped_server_builtin_call_ids.add(_call_id_out)
                        continue
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": _call_id_out,
                            "name": _fn["name"],
                            "arguments": _args_raw,
                        }
                    )
                # Assistant text already emitted above (in order) so we
                # don't fall through to the generic content branches.
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
                            # Responses takes image_url as a flat string (both
                            # https:// URLs and data: URLs are accepted).
                            translated_parts.append(
                                {"type": "input_image", "image_url": url}
                            )
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
                        # OpenAI Responses accepts PDFs / docs as
                        # `{type:"input_file", file_data:"data:application/pdf;base64,..."}`
                        # or `{type:"input_file", file_url:"https://..."}`,
                        # with optional `filename`. See
                        # https://developers.openai.com/api/docs/guides/images-vision
                        # Map Studio's normalised `input_document` shape
                        # straight onto Responses' `input_file`.
                        file_url = part.get("file_url")
                        file_data = part.get("file_data")
                        filename = part.get("filename")
                        # Mirror the Anthropic-side guard: any "data:" URI
                        # without an actual base64 payload (`data:application/pdf;base64,`
                        # or whitespace-only) would otherwise be forwarded
                        # to OpenAI as `file_data=""`, which 400s the whole
                        # turn. Treat such payloads as missing AND fall
                        # back to file_url if one is also present, so a
                        # recoverable remote URL doesn't get discarded in
                        # favour of a malformed inline payload.
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
            # OpenAI's documented multi-turn image generation path can use
            # `previous_response_id` to carry the prior generated image and
            # paired reasoning state. Prefer that over manual item replay when
            # we captured the response id; keep replay below as a fallback for
            # older stored turns that only have an image_generation_call id.
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
                yield _error_sse_line(
                    400,
                    "OpenAI image edit reference is missing paired reasoning state. "
                    "Regenerate the image, then retry the edit.",
                    self.provider_type,
                )
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

        # NOTE: gpt-5.x / o3 / gpt-4.5 are reasoning-class models. They reject
        # temperature and top_p with `Unsupported parameter` 400s on
        # /v1/responses (and on /v1/chat/completions for the same families).
        # The PROVIDER_REGISTRY['openai'] model_id_allowlist already scopes
        # the picker to those families, so we never need to send sampling
        # knobs here. ``reasoning.effort`` defaults to "medium" server-side
        # if omitted — surface it in a future commit if a knob is wanted.
        del temperature, top_p  # explicit drop — params are accepted for
        # API symmetry with the other stream methods but not forwarded.

        body: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
        }
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        # `summary: "auto"` is what makes /v1/responses emit reasoning
        # summary events — without it OpenAI returns no thinking text on
        # most reasoning models, the SSE handler has no <think>…</think>
        # to wrap, and the chat reasoning panel stays blank. Always pair
        # an explicit effort with summary except for the explicit "off"
        # case (effort: "none"), where summaries are pointless.
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

        # Opt into 24h prompt-cache retention (free, vs the default
        # ~5-10 min). Gated on the OpenAI cloud host because ollama /
        # llama.cpp / "custom" presets reach this code path too and
        # would 400 on the unknown field.
        if is_openai_cloud and enable_prompt_caching is not False:
            body["prompt_cache_retention"] = "24h"

        # Server-side context compaction (OpenAI cloud only).
        # https://developers.openai.com/api/docs/guides/compaction
        if (
            is_openai_cloud
            and compaction_threshold is not None
            and compaction_threshold > 0
        ):
            body["context_management"] = [
                {
                    "type": "compaction",
                    "compact_threshold": int(compaction_threshold),
                }
            ]

        # Map enabled_tools onto Responses-API server tools (cloud only;
        # local OAI-compat backends 400 on these).
        # https://developers.openai.com/api/docs/guides/tools
        code_execution_enabled_openai = bool(
            enabled_tools and "code_execution" in enabled_tools and is_openai_cloud
        )
        image_generation_enabled_openai = bool(
            enabled_tools and "image_generation" in enabled_tools and is_openai_cloud
        )

        def _openai_image_generation_tool() -> dict[str, Any]:
            tool: dict[str, Any] = {"type": "image_generation"}
            if image_generation_has_reference:
                # Force edit mode so the prior call id is used as context.
                tool["action"] = "edit"
            return tool

        # Translate Chat-Completions function tools into the Responses
        # function-tool shape (flattened name/description/parameters).
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
                    _entry["parameters"] = _fn["parameters"]
                responses_user_function_tools.append(_entry)

        # Translate tool_choice into the Responses shape.
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

        _responses_tool_choice_none = _responses_tc_string == "none"
        # A pinned user function suppresses hosted builtins (privacy +
        # billing), matching the Gemini / Anthropic / OpenRouter gates.
        _responses_tool_choice_forced_function = (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
            and bool(tool_choice["function"].get("name"))
        )
        _responses_hosted_builtins_allowed = (
            not _responses_tool_choice_none
            and not _responses_tool_choice_forced_function
        )

        if (
            enabled_tools or responses_user_function_tools
        ) and not _responses_tool_choice_none:
            tools_array: list[dict[str, Any]] = list(responses_user_function_tools)
            if (
                _responses_hosted_builtins_allowed
                and enabled_tools
                and "web_search" in enabled_tools
            ):
                tools_array.append({"type": "web_search"})
            if _responses_hosted_builtins_allowed and code_execution_enabled_openai:
                # Reuse the thread's container so filesystem state
                # persists; auto-create when there isn't one yet. Stale
                # ids 400 and are cleared via container_invalidated.
                shell_env: dict[str, Any]
                if openai_code_exec_container_id:
                    shell_env = {
                        "type": "container_reference",
                        "container_id": openai_code_exec_container_id,
                    }
                else:
                    shell_env = {"type": "container_auto"}
                tools_array.append({"type": "shell", "environment": shell_env})
            if _responses_hosted_builtins_allowed and image_generation_enabled_openai:
                tools_array.append(_openai_image_generation_tool())
            if tools_array:
                body["tools"] = tools_array
        if responses_tool_choice is not None:
            body["tool_choice"] = responses_tool_choice

        url = f"{self.base_url}/responses"
        completion_id = f"chatcmpl-openai-{model.replace('/', '-')}"

        logger.info("Proxying OpenAI Responses API to %s (model=%s)", url, model)

        def _build_body(container_id_for_this_attempt: Optional[str]) -> dict[str, Any]:
            """Snapshot of the request body. Called once for the initial
            attempt and again with ``None`` for the post-expiry retry.
            Returns a fresh dict so the retry doesn't share state with the
            first attempt.
            """
            attempt_body = dict(body)
            if (
                enabled_tools or responses_user_function_tools
            ) and not _responses_tool_choice_none:
                tools_array_attempt: list[dict[str, Any]] = list(
                    responses_user_function_tools
                )
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
                    tools_array_attempt.append(
                        {"type": "shell", "environment": env_attempt}
                    )
                if (
                    _responses_hosted_builtins_allowed
                    and image_generation_enabled_openai
                ):
                    tools_array_attempt.append(_openai_image_generation_tool())
                if tools_array_attempt:
                    attempt_body["tools"] = tools_array_attempt
                else:
                    attempt_body.pop("tools", None)
            if responses_tool_choice is not None:
                attempt_body["tool_choice"] = responses_tool_choice
            return attempt_body

        def _is_openai_container_expired_error(error_text: str) -> bool:
            """Match the substring patterns OpenAI uses for expired / missing
            code-exec containers. There's no official error code in the public
            docs, so we substring-match a small set.
            """
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
                async with _http_client.stream(
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
                        if expired_container_4xx and not retried:
                            yield (
                                f"data: "
                                f"{_json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': None}], '_toolEvent': {'type': 'container_invalidated'}})}"
                            )
                            retried = True
                            attempt_container_id = None
                            continue
                        yield _error_sse_line(
                            response.status_code, error_text, self.provider_type
                        )
                        return

                    # NOTE: same manual __anext__ loop as stream_chat_completion —
                    # see comment there for the GeneratorExit / aclose ordering.
                    lines_gen = response.aiter_lines().__aiter__()
                    done_emitted = False
                    reasoning_open = False
                    reasoning_emitted = False
                    # Per-call function-tool indexing; distinct slots so
                    # parallel calls don't collide on delta.tool_calls[].index.
                    saw_function_call = False
                    function_call_index = 0
                    # Latched from response.completed/incomplete; surfaces
                    # input_tokens_details.cached_tokens to prove cache hits.
                    last_usage: Optional[dict[str, Any]] = None
                    # web_search state. Citations are emitted on text deltas
                    # (not per call), so the aggregate list is shared and
                    # applied to the LAST web_search tool_end (parseSourcesFromResult
                    # flatmaps every call, one non-empty is enough).
                    web_search_calls: dict[str, dict[str, Any]] = {}
                    all_url_citations: list[dict[str, Any]] = []
                    # shell_calls (code execution): { call_id -> {commands, output} }.
                    # shell_call <-> shell_call_output match by call_id; emit
                    # tool_start/tool_end like the Anthropic UX.
                    shell_calls: dict[str, dict[str, Any]] = {}
                    # Container id latched from response.container_id or
                    # item.environment.container_id; emit container_ready
                    # when it differs from the inbound id.
                    latched_container_id: Optional[str] = None
                    container_id_emitted = False
                    current_openai_response_id: Optional[str] = None
                    last_openai_reasoning_replay_item: Optional[dict[str, Any]] = None
                    openai_reasoning_replay_items: dict[str, dict[str, Any]] = {}
                    image_generation_calls_started: set[str] = set()
                    # Buffer for a citation marker straddling two delta events;
                    # prepended onto the next delta. See _split_pending_citation_tail.
                    pending_marker_tail: str = ""
                    # Segments deferred while their markers reference unseen
                    # source_ids; held in arrival order so output never
                    # leapfrogs an earlier deferred segment. Flushed on
                    # annotation events and force-flushed at end-of-stream
                    # with leftover private-use codepoints stripped.
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
                        """Re-attempt resolution on buffered segments in order.
                        Stops at the first still-unresolved segment unless
                        ``force`` (end-of-stream), where lingering markers are stripped."""
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
                        """Render any leftover citation tail at end-of-stream.

                        Unterminated tails drop (no annotation to bind to). If the
                        close byte arrived concatenated, rewrite then scrub any
                        residual private-use bytes and any orphan ``cite<sid>``
                        literal so the renderer never sees raw markup. url_citations
                        are aggregated separately and applied to web_search tool_end.
                        """
                        if not tail:
                            return ""
                        if _OPENAI_CITE_STOP not in tail:
                            # Unterminated: drop the whole tail, otherwise the
                            # residual ``cite<sid>`` would leak as plain text.
                            return ""
                        rendered = _replace_openai_citation_markers(
                            tail, all_url_citations
                        )
                        # Scrub residual private-use bytes (e.g. a partial opener).
                        for ch in ("", "", ""):
                            rendered = rendered.replace(ch, "")
                        # Drop any orphan ``cite<sid>`` literal -- meaningless
                        # without its closing byte and matching url_citation.
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
                        """Render an OpenAI `shell_call_output.output` list
                        as the preformatted text payload the frontend's
                        CodeExecutionToolUI displays inside a <pre>. Each
                        entry has stdout/stderr/outcome — concatenate them
                        with a separator block per entry and append
                        `return_code` / `(timeout)` annotations only when
                        they convey information beyond "succeeded".
                        """
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
                        return (
                            "\n--- next command ---\n".join(parts)
                            if parts
                            else "(no output)"
                        )

                    def _record_url_citation(payload: dict[str, Any]) -> None:
                        """Append a url_citation onto the shared all_url_citations
                        list. Dedup by URL — the same URL can be cited many
                        times under different ``source_id`` aliases (one per
                        span/locator), so collect every alias we see onto
                        the matching entry's ``source_ids`` list. The
                        delta-text rewriter resolves any of those aliases
                        back to this entry's URL. The id may live under
                        ``source_id``, ``id``, or ``locator`` across the
                        Responses API revisions."""
                        if payload.get("type") != "url_citation":
                            return
                        url = payload.get("url", "")
                        if not url:
                            return
                        source_id = (
                            payload.get("source_id")
                            or payload.get("id")
                            or payload.get("locator")
                            or ""
                        )
                        # Single pass: either backfill aliases onto an
                        # existing URL entry (and return) or fall through
                        # to append a fresh one.
                        for c in all_url_citations:
                            if c["url"] != url:
                                continue
                            if source_id:
                                aliases = c.setdefault("source_ids", [])
                                if source_id not in aliases:
                                    aliases.append(source_id)
                            return
                        title = payload.get("title") or url
                        snippet = payload.get("snippet") or payload.get("quote") or ""
                        all_url_citations.append(
                            {
                                "url": url,
                                "title": title,
                                "snippet": snippet,
                                "source_ids": [source_id] if source_id else [],
                            }
                        )

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
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "summary_text"
                        ):
                            text = part.get("text")
                            if isinstance(text, str):
                                summary_text = text
                        elif (
                            payload.get("type")
                            == "response.reasoning_summary_text.done"
                        ):
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
                                if (
                                    isinstance(summary_index, int)
                                    and summary_index >= 0
                                ):
                                    while len(summary) <= summary_index:
                                        summary.append(
                                            {"type": "summary_text", "text": ""}
                                        )
                                    summary[summary_index] = summary_part
                                else:
                                    summary.append(summary_part)
                        return existing

                    def _image_generation_arguments(
                        prompt: str,
                        raw_item_id: Any,
                    ) -> dict[str, Any]:
                        arguments: dict[str, Any] = {"kind": "image", "prompt": prompt}
                        if isinstance(raw_item_id, str) and raw_item_id:
                            arguments["openai_image_generation_call_id"] = raw_item_id
                        if current_openai_response_id:
                            arguments["openai_response_id"] = current_openai_response_id
                        if last_openai_reasoning_replay_item:
                            arguments["openai_reasoning_item"] = (
                                last_openai_reasoning_replay_item
                            )
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
                            # OpenAI responses may carry reasoning summaries in
                            # different envelope fields across event variants.
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
                            if data_str == "[DONE]":
                                # Flush any held-over partial marker; strip
                                # private-use bytes so garbled glyphs don't leak.
                                if pending_marker_tail:
                                    flushed = _flush_pending_marker_tail(
                                        pending_marker_tail
                                    )
                                    pending_marker_tail = ""
                                    if flushed:
                                        if reasoning_open:
                                            yield _chunk_with_text("</think>")
                                            reasoning_open = False
                                        yield _chunk_with_text(flushed)
                                # Force-drain any segment still awaiting an
                                # annotation; lingering codepoints are stripped.
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

                            event_type = event.get("type")
                            _record_openai_response_id(event)

                            if event_type == "response.output_text.delta":
                                delta_text = event.get("delta", "")
                                # Process inline annotations first so source_ids
                                # referenced by same-delta markers are in the lookup
                                # before the rewriter runs. Some API versions inline
                                # url citations on the delta event itself.
                                for ann in event.get("annotations") or []:
                                    if isinstance(ann, dict):
                                        _record_url_citation(ann)
                                if delta_text or pending_marker_tail:
                                    # Prepend any held-over tail so a marker
                                    # straddling two SSE events resolves cleanly.
                                    combined = pending_marker_tail + delta_text
                                    head, pending_marker_tail = (
                                        _split_pending_citation_tail(combined)
                                    )
                                    if head:
                                        if reasoning_open:
                                            yield _chunk_with_text("</think>")
                                            reasoning_open = False
                                        # Re-attempt earlier deferred segments first
                                        # so output stays in order; the needed
                                        # annotation may have arrived inline above.
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
                                            pending_citation_segments.append(
                                                head_rewritten
                                            )
                                        elif head_rewritten:
                                            yield _chunk_with_text(head_rewritten)

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
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "web_search_call"
                                ):
                                    item_id = item.get("id", "") or (
                                        f"ws_{len(web_search_calls)}"
                                    )
                                    web_search_calls.setdefault(item_id, {"query": ""})
                                # Register shell_call eagerly so out-of-order
                                # output links back. Probe env.container_id
                                # to emit container_ready before response.completed.
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "shell_call"
                                ):
                                    item_id = item.get("id", "") or (
                                        f"sc_{len(shell_calls)}"
                                    )
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
                                    summary_text = _extract_reasoning_text(
                                        item.get("summary")
                                    )
                                    if summary_text and not reasoning_emitted:
                                        if not reasoning_open:
                                            summary_text = f"<think>{summary_text}"
                                            reasoning_open = True
                                        yield _chunk_with_text(summary_text)
                                        reasoning_emitted = True
                                elif item.get("type") == "web_search_call":
                                    # done is the canonical place to read the
                                    # query, so emit both tool_start and tool_end
                                    # here. Frontend then renders a card per call
                                    # with the proper "Searching: <query>" label.
                                    # Citations are aggregated separately and the
                                    # *last* call's result is overwritten at
                                    # response.completed with the citation list
                                    # (so the source-pill extraction at message
                                    # tail surfaces them once).
                                    item_id = item.get("id", "") or (
                                        f"ws_{len(web_search_calls)}"
                                    )
                                    action = item.get("action")
                                    query = (
                                        action.get("query", "")
                                        if isinstance(action, dict)
                                        else ""
                                    )
                                    web_search_calls[item_id] = {"query": query}
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_start",
                                            "tool_name": "web_search",
                                            "tool_call_id": item_id,
                                            "arguments": (
                                                {"query": query} if query else {}
                                            ),
                                        }
                                    )
                                    # Per-card text; last call gets overwritten
                                    # with citations at response.completed.
                                    per_call_result = (
                                        f"Searching: {query}" if query else ""
                                    )
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": item_id,
                                            "result": per_call_result,
                                        }
                                    )
                                elif item.get("type") == "shell_call":
                                    # OpenAI ships the commands array on the
                                    # action field. Join them onto one
                                    # command string for the tool card —
                                    # the renderer is shared with Anthropic
                                    # bash, which only carries a single
                                    # `command`. Multiple commands in one
                                    # shell_call get joined with newlines so
                                    # they still render as one card.
                                    item_id = item.get("id", "") or (
                                        f"sc_{len(shell_calls)}"
                                    )
                                    action = item.get("action") or {}
                                    commands = (
                                        action.get("commands")
                                        if isinstance(action, dict)
                                        else None
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
                                        list(commands)
                                        if isinstance(commands, list)
                                        else []
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
                                    # Fallback: output may be bundled on the
                                    # shell_call done event itself.
                                    embedded_output = item.get("output")
                                    if (
                                        isinstance(embedded_output, list)
                                        and embedded_output
                                    ):
                                        shell_calls[item_id]["output"] = embedded_output
                                        shell_calls[item_id]["tool_end_emitted"] = True
                                        yield _emit_tool_event(
                                            {
                                                "type": "tool_end",
                                                "tool_call_id": item_id,
                                                "result": _format_shell_output(
                                                    embedded_output
                                                ),
                                            }
                                        )
                                elif item.get("type") == "shell_call_output":
                                    # `call_id` links back to the shell_call's
                                    # `id`, which is what we used as the
                                    # tool_call_id on tool_start. Match on
                                    # call_id when present so the matching
                                    # card transitions to complete.
                                    call_id = (
                                        item.get("call_id") or item.get("id") or ""
                                    )
                                    output = item.get("output") or []
                                    # Skip if bundled-output path already
                                    # finalised this card.
                                    if shell_calls.get(call_id, {}).get(
                                        "tool_end_emitted"
                                    ):
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
                                    # Base64 image on `result` (or `b64_json`),
                                    # `revised_prompt` for the rewritten prompt.
                                    # ns-resolution id so concurrent gens stay unique.
                                    raw_item_id = item.get("id")
                                    item_id = raw_item_id or f"img_{time.time_ns()}"
                                    prompt_in = (
                                        item.get("revised_prompt")
                                        or item.get("prompt")
                                        or ""
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
                                    b64 = (
                                        item.get("result") or item.get("b64_json") or ""
                                    )
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
                                    # Translate to Chat-Completions delta.tool_calls.
                                    # https://platform.openai.com/docs/guides/function-calling?api-mode=responses
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
                                                                        "arguments": (
                                                                            fn_args
                                                                        ),
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

                            elif (
                                isinstance(event_type, str)
                                and "reasoning" in event_type
                            ):
                                recorded_reasoning = (
                                    _record_openai_reasoning_replay_item(event)
                                )
                                if recorded_reasoning:
                                    last_openai_reasoning_replay_item = (
                                        recorded_reasoning
                                    )
                                reasoning_delta = _extract_reasoning_text(event)
                                if reasoning_delta:
                                    if not reasoning_open:
                                        reasoning_delta = f"<think>{reasoning_delta}"
                                        reasoning_open = True
                                    yield _chunk_with_text(reasoning_delta)
                                    reasoning_emitted = True

                            elif event_type == "response.completed":
                                completed_usage = (event.get("response") or {}).get(
                                    "usage"
                                )
                                if isinstance(completed_usage, dict):
                                    last_usage = completed_usage
                                # Flush any unterminated citation tail
                                # held over from the last delta. By
                                # the time we get here every annotation
                                # has been recorded so a late-arriving
                                # source_id may resolve cleanly; if it
                                # still doesn't, the helper strips the
                                # private-use bytes so no garbled
                                # glyph reaches the user.
                                if pending_marker_tail:
                                    flushed = _flush_pending_marker_tail(
                                        pending_marker_tail
                                    )
                                    pending_marker_tail = ""
                                    if flushed:
                                        if reasoning_open:
                                            yield _chunk_with_text("</think>")
                                            reasoning_open = False
                                        yield _chunk_with_text(flushed)
                                # Force-drain any segment still awaiting an
                                # annotation; lingering codepoints are stripped.
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
                                # Probe response.container_id (top-level) and
                                # response.container.id for the shell-tool
                                # container id. OpenAI's docs don't pin the
                                # exact field, so we scan both. Emit
                                # `container_ready` only when the value
                                # differs from the inbound one — no churn on
                                # reuse.
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
                                    and latched_container_id
                                    != openai_code_exec_container_id
                                ):
                                    yield _emit_tool_event(
                                        {
                                            "type": "container_ready",
                                            "container_id": latched_container_id,
                                        }
                                    )
                                    container_id_emitted = True
                                # Overwrite the last web_search call with the
                                # citation list; the source-pill extractor
                                # flatMaps across cards. Earlier cards keep
                                # their per-call "Searching:" text.
                                if web_search_calls and all_url_citations:
                                    last_id = list(web_search_calls.keys())[-1]
                                    blocks: list[str] = []
                                    for cit in all_url_citations:
                                        line = (
                                            f"Title: {cit['title']}\nURL: {cit['url']}"
                                        )
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
                                # Final flush: finalise any orphan shell_call
                                # so the card stops spinning.
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
                                            "finish_reason": (
                                                "tool_calls"
                                                if saw_function_call
                                                else "stop"
                                            ),
                                        }
                                    ],
                                }
                                yield f"data: {_json.dumps(chunk)}"
                                # Emit include_usage-style chunk after the
                                # finish_reason so callers can surface
                                # cached_tokens in their UI.
                                usage_line = _build_usage_chunk(
                                    completion_id,
                                    "openai",
                                    last_usage,
                                )
                                if usage_line:
                                    yield usage_line

                            elif event_type == "response.incomplete":
                                incomplete_usage = (event.get("response") or {}).get(
                                    "usage"
                                )
                                if isinstance(incomplete_usage, dict):
                                    last_usage = incomplete_usage
                                # Same flush as response.completed --
                                # truncated streams can leave a half-
                                # marker in the buffer.
                                if pending_marker_tail:
                                    flushed = _flush_pending_marker_tail(
                                        pending_marker_tail
                                    )
                                    pending_marker_tail = ""
                                    if flushed:
                                        if reasoning_open:
                                            yield _chunk_with_text("</think>")
                                            reasoning_open = False
                                        yield _chunk_with_text(flushed)
                                # Force-drain any segment still awaiting an
                                # annotation; lingering codepoints are stripped.
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
                                # Same backfill as response.completed — apply
                                # whatever citations we managed to gather
                                # before truncation onto the last call. All
                                # earlier tool cards already have their proper
                                # query + empty placeholder result from the
                                # output_item.done emissions above.
                                if web_search_calls and all_url_citations:
                                    last_id = list(web_search_calls.keys())[-1]
                                    blocks = []
                                    for cit in all_url_citations:
                                        line = (
                                            f"Title: {cit['title']}\nURL: {cit['url']}"
                                        )
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
                                # Mirror the response.completed flush so
                                # truncated streams also finalise orphan
                                # shell_calls.
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
                                            "finish_reason": "length",
                                        }
                                    ],
                                }
                                yield f"data: {_json.dumps(chunk)}"
                                # Emit include_usage-style chunk after the
                                # length-truncated finish_reason too, so
                                # incomplete responses still report
                                # cached_tokens.
                                usage_line = _build_usage_chunk(
                                    completion_id,
                                    "openai",
                                    last_usage,
                                )
                                if usage_line:
                                    yield usage_line

                            elif event_type in ("response.failed", "error"):
                                # Surface the failure to the client; let the
                                # outer route emit [DONE] as part of its cleanup.
                                error_payload = event.get("response", {}).get(
                                    "error", {}
                                ) or {
                                    "message": event.get("message", "Unknown error"),
                                    "code": event.get("code"),
                                }
                                yield _error_sse_line(
                                    502,
                                    _json.dumps(error_payload),
                                    self.provider_type,
                                )
                                break
                    except GeneratorExit:
                        await response.aclose()
                        await lines_gen.aclose()
                        raise
                    finally:
                        # Summarise what the model actually did this turn so
                        # support reports of "I clicked Search and got nothing"
                        # can be triaged at a glance: was the tool requested,
                        # did OpenAI invoke it, and how many sources came back?
                        web_search_requested = bool(
                            enabled_tools and "web_search" in enabled_tools
                        )
                        web_search_invocations = len(web_search_calls)
                        total_citations = len(all_url_citations)
                        queries = [
                            sc["query"]
                            for sc in web_search_calls.values()
                            if sc.get("query")
                        ]
                        # cached_input_tokens > 0 on turn N proves
                        # prompt_cache_retention="24h" is letting the previous
                        # turn's prefix hit the cache instead of being
                        # recomputed. On /v1/responses the field is nested as
                        # usage.input_tokens_details.cached_tokens (not
                        # prompt_tokens_details, which is the /v1/chat/completions
                        # shape).
                        cached_input_tokens = None
                        if isinstance(last_usage, dict):
                            details = last_usage.get("input_tokens_details")
                            if isinstance(details, dict):
                                cached_input_tokens = details.get("cached_tokens")
                        code_execution_requested = code_execution_enabled_openai
                        code_execution_invocations = len(shell_calls)
                        code_execution_results = sum(
                            1
                            for sc in shell_calls.values()
                            if sc.get("output") is not None
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

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full response dict.

        Note: only valid for OpenAI-compatible providers. Anthropic requires its
        own Messages API; use stream_chat_completion (with stream=False) instead
        if a non-streaming Anthropic path is needed in the future.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
        }
        if max_tokens is not None:
            if self.provider_type == "openai":
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens

        response = await _http_client.post(
            f"{self.base_url}/chat/completions",
            json = body,
            headers = self._auth_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        return response.json()

    async def list_models(self) -> list[dict[str, Any]]:
        """
        Call GET /models on the provider to discover available models.

        Returns a list of model dicts with at least 'id' and optionally
        'created', 'owned_by', etc.

        All supported providers expose a /models endpoint:
        - OpenAI-compatible: standard {"data": [...]} response
        - Anthropic: https://api.anthropic.com/v1/models — same {"data": [...]} shape
        """
        try:
            response = await _http_client.get(
                f"{self.base_url}/models",
                headers = self._auth_headers(),
                timeout = self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI format: {"data": [{"id": "...", ...}, ...]}
            # Some local servers (Ollama with no models) return data: null.
            models: list[dict[str, Any]] = []
            if isinstance(data, dict):
                raw_models = data.get("data") or []
                if isinstance(raw_models, list):
                    models = [model for model in raw_models if isinstance(model, dict)]
            if not models and self.provider_type == "ollama":
                models = await self._list_ollama_native_models()
            # Gemini's native /v1beta/models returns
            # {"models": [{"name": "models/gemini-2.5-flash", ...}]}
            # -- repackage into the OpenAI-compatible shape the rest
            # of Studio expects so dynamic model discovery works.
            if not models and self.provider_type == "gemini":
                models = self._parse_gemini_models(data)
            return models
        except httpx.HTTPError as exc:
            logger.error("Failed to list models from %s: %s", self.provider_type, exc)
            raise

    @staticmethod
    def _parse_gemini_models(payload: Any) -> list[dict[str, Any]]:
        """Translate Gemini's native /v1beta/models payload to OpenAI shape.

        Native response:
          {"models": [{"name": "models/gemini-2.5-flash",
                       "baseModelId": "gemini-2.5-flash",
                       "displayName": "Gemini 2.5 Flash",
                       "supportedGenerationMethods": [...]}]}

        We only keep entries that advertise
        ``generateContent`` / ``streamGenerateContent`` so the picker
        does not surface embedding-only models the chat path can't
        drive.
        """
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
                and not any(
                    m in methods for m in ("generateContent", "streamGenerateContent")
                )
            ):
                continue
            base_id = entry.get("baseModelId")
            name = entry.get("name") or ""
            # ``name`` arrives as ``"models/gemini-2.5-flash"``; the
            # chat path uses the bare id.
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

    async def _list_ollama_native_models(self) -> list[dict[str, Any]]:
        """Fallback when Ollama's /v1/models returns an empty or null catalog."""
        root = self.base_url.removesuffix("/v1").rstrip("/")
        response = await _http_client.get(
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
        return [
            {"id": entry.get("name", "").strip(), "owned_by": "ollama"}
            for entry in raw_models
            if isinstance(entry, dict) and entry.get("name", "").strip()
        ]

    async def verify_models_endpoint_lightweight(self) -> None:
        """
        Confirm GET /models returns 200 without buffering the full response body.

        Used for providers with enormous catalogs (e.g. OpenRouter, Hugging Face router)
        where downloading the full JSON would be prohibitive.
        """
        url = f"{self.base_url}/models"
        try:
            async with _http_client.stream(
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
        """Auth headers plus the OpenAI-Beta opt-in for /v1/containers.

        OpenAI's containers API requires ``OpenAI-Beta: containers=v1``.
        Without it, DELETE silently no-ops: the API returns 200 with a
        ``{"deleted": true}`` body but does not actually remove the
        container (verified 2026-05-15). The header is required for
        list / create / delete to behave consistently.
        """
        headers = self._auth_headers()
        headers["OpenAI-Beta"] = "containers=v1"
        return headers

    async def list_openai_containers(self) -> list[dict[str, Any]]:
        """
        GET /v1/containers on the user's OpenAI account.

        Returns the raw container records (id, name, created_at,
        last_active_at, expires_after, status). The route layer
        reshapes these into the UI summary shape.

        Only valid against api.openai.com — non-cloud OpenAI-compat
        servers don't implement /v1/containers and would 404 here.
        Caller is responsible for the is_openai_cloud guard.
        """
        response = await _http_client.get(
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
            [
                {"id": c.get("id"), "status": c.get("status")}
                for c in result
                if isinstance(c, dict)
            ],
        )
        return result

    async def create_openai_container(
        self,
        name: str,
        ttl_minutes: int,
    ) -> dict[str, Any]:
        """
        POST /v1/containers with ``expires_after.anchor="last_active_at"``.
        ``ttl_minutes`` is the idle timeout — every API call that
        touches the container resets the timer.
        """
        body = {
            "name": name,
            "expires_after": {
                "anchor": "last_active_at",
                "minutes": ttl_minutes,
            },
        }
        response = await _http_client.post(
            f"{self.base_url}/containers",
            json = body,
            headers = self._container_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        return response.json()

    async def delete_openai_container(self, container_id: str) -> None:
        """DELETE /v1/containers/{id}. 404s are surfaced as HTTPError.

        Uses a fresh httpx client (not the shared ``_http_client``) so
        connection-pool state from earlier chat requests cannot
        interfere — observed in the wild that DELETEs over the shared
        pool returned ``deleted: true`` while the container persisted
        in subsequent /containers list calls, even though the same
        DELETE issued from a fresh client genuinely removed it.

        Verifies the response body reports ``deleted: true``. OpenAI
        returns a 2xx ``deleted: true`` body even when the request is
        silently rejected (e.g. missing OpenAI-Beta header), so a
        status-only check is not sufficient.
        """
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
    """Rewrite common provider errors into actionable Studio copy."""
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


def _error_sse_line(status_code: int, message: str, provider_type: str) -> str:
    """Format an error as an SSE data line in OpenAI error format."""
    import json

    error_obj = {
        "error": {
            "message": message,
            "type": "provider_error",
            "code": str(status_code),
            "provider": provider_type,
        }
    }
    return f"data: {json.dumps(error_obj)}"


def _build_usage_chunk(
    completion_id: str,
    provider: Literal["anthropic", "openai"],
    last_usage: Optional[dict],
) -> Optional[str]:
    """Build an OpenAI ``include_usage``-style SSE chunk that carries the
    upstream prompt-cache accounting back to the client.

    Until now Studio captured ``cache_creation_input_tokens`` /
    ``cache_read_input_tokens`` (Anthropic) and
    ``input_tokens_details.cached_tokens`` (OpenAI Responses) on
    ``last_usage`` and only wrote them to the structlog stream.
    Browser / SDK clients had no way to see how many tokens hit the cache
    -- so the "you saved $X" UX in the chat panel was impossible without
    scraping the server log.

    This helper emits the standard OpenAI chunk shape -- ``choices: []``
    with a populated ``usage`` block -- so any client that already
    consumes ``stream_options={"include_usage": true}`` keeps working,
    and the Anthropic-native counts are surfaced as extra keys on the
    same ``usage`` dict:

        usage.prompt_tokens_details.cached_tokens
            normalised cache-read count, present for both providers.
        usage.cache_creation_input_tokens
            Anthropic-only; tokens billed at the cache-write premium.
        usage.cache_read_input_tokens
            Anthropic-only; same value as cached_tokens, kept for
            callers that already key off the native Anthropic name.

    Anthropic's ``input_tokens`` excludes the cache buckets -- the
    real prompt size is ``input_tokens + cache_creation_input_tokens
    + cache_read_input_tokens``. Emitting ``input_tokens`` alone as
    ``prompt_tokens`` undercounts cache-heavy turns and breaks
    downstream context / cost displays, so we add all three input
    buckets together. OpenAI Responses already folds cached tokens
    into ``input_tokens`` so no extra arithmetic is needed there.

    Returns ``None`` when there are no usage numbers to report (e.g. an
    upstream error before ``message_start`` / ``response.completed``).
    """
    if not isinstance(last_usage, dict):
        return None

    completion_tokens = last_usage.get("output_tokens") or 0

    if provider == "anthropic":
        uncached_input = last_usage.get("input_tokens") or 0
        cache_creation = last_usage.get("cache_creation_input_tokens") or 0
        cache_read = last_usage.get("cache_read_input_tokens") or 0
        prompt_tokens = uncached_input + cache_creation + cache_read
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
        # Forward 5m/1h cache-write breakdown so cost calc applies the
        # 2x 1h premium instead of defaulting to 5m on chat-style.
        cc_breakdown = last_usage.get("cache_creation")
        if isinstance(cc_breakdown, dict) and cc_breakdown:
            usage_block["cache_creation"] = cc_breakdown
        # Propagate fast-mode `usage.speed` so the cost ledger can apply
        # the 6x multiplier without re-derivation (Anthropic falls back
        # to "standard" when fast-mode is unsupported or rate-limited).
        speed = last_usage.get("speed")
        if speed in ("fast", "standard"):
            usage_block["speed"] = speed
    else:
        prompt_tokens = last_usage.get("input_tokens") or 0
        cached = 0
        details = last_usage.get("input_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens") or 0
        if not (prompt_tokens or completion_tokens or cached):
            return None
        usage_block = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        # Surface OpenAI Responses / Gemini reasoning-token detail. The
        # caller pre-populates last_usage["output_tokens_details"] with
        # at least {"reasoning_tokens": ...}; mirror it into the OAI
        # `completion_tokens_details` shape so SDKs can render the
        # hidden-thoughts slice.
        out_details = last_usage.get("output_tokens_details")
        if isinstance(out_details, dict) and out_details:
            usage_block["completion_tokens_details"] = {
                "reasoning_tokens": out_details.get("reasoning_tokens") or 0,
            }
            usage_block["output_tokens_details"] = out_details

    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": usage_block,
    }
    return f"data: {_json.dumps(chunk)}"
