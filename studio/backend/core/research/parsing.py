# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reading structured decisions out of a local model's free-form output.

Plans, agent actions, research state, and the synthesis audit arrive as JSON that a small
model routinely wraps in prose, truncates, or emits from its reasoning channel instead. Every
parser here recovers what it can and validates the result against what the run actually holds.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.inference.web_access_policy import check_url_access
from core.research.redaction import _sanitize_public_query

# Completed "title": "..." pairs in a partially streamed planner response.
_STREAMED_TITLE = re.compile(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"')
# One more than the plan-step cap, since the plan's own title matches too.
_MAX_PREVIEW_LABELS = 31
# Allow a list-item or block-quote container marker before the fence, else the open is missed and a
# marker quoted inside is taken for the real boundary.
_MARKDOWN_FENCE = re.compile(r"^ {0,3}(?:(?:[-*+]|\d{1,9}[.)])[ \t]+|>[ \t]?)*(`{3,}|~{3,})")


def _validate_agent_action(
    value: dict,
    allowed_urls: set[str],
    website_policy: dict | None = None,
) -> dict[str, Any]:
    action = str(value.get("action") or "").strip().lower()
    title = str(value.get("title") or "Researching").strip()[:200]
    research_state = _normalize_research_state(value.get("researchState"))
    if action == "search":
        query = str(value.get("query") or "").strip()
        if not query:
            raise ValueError("Research agent returned an empty search query")
        query = _sanitize_public_query(query)
        return {
            "action": action,
            "title": title,
            "query": query,
            **({"researchState": research_state} if research_state else {}),
        }
    if action == "fetch":
        url = str(value.get("url") or "").strip()
        if url not in allowed_urls:
            raise ValueError("Research agent selected an unknown URL")
        allowed, reason, _hostname = check_url_access(url, website_policy)
        if not allowed:
            raise ValueError(reason)
        return {
            "action": action,
            "title": title,
            "url": url,
            **({"researchState": research_state} if research_state else {}),
        }
    if action == "finish":
        return {
            "action": action,
            "title": title,
            **({"researchState": research_state} if research_state else {}),
        }
    raise ValueError("Research agent returned an unsupported action")


def _normalize_research_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    def short_list(name: str, limit: int) -> list[str]:
        raw = value.get(name)
        if not isinstance(raw, list):
            return []
        return [str(item).strip()[:400] for item in raw[:limit] if str(item).strip()]

    state = {
        "summary": str(value.get("summary") or "").strip()[:4000],
        "gaps": short_list("gaps", 8),
        "unsupportedClaims": short_list("unsupportedClaims", 8),
        "nextBridge": str(value.get("nextBridge") or "").strip()[:800],
    }
    return {key: item for key, item in state.items() if item}


def _normalize_synthesis_audit(
    value: Any, allowed_source_urls: set[str], allowed_document_citations: set[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    def short_list(
        name: str,
        limit: int,
        item_limit: int = 500,
    ) -> list[str]:
        raw = value.get(name)
        if not isinstance(raw, list):
            return []
        return [str(item).strip()[:item_limit] for item in raw[:limit] if str(item).strip()]

    def allowed_list(raw: Any, allowed: set[str]) -> list[str]:
        values: list[str] = []
        if not isinstance(raw, list):
            return values
        for raw_value in raw:
            item = str(raw_value).strip()
            if item in allowed and item not in values:
                values.append(item)
            if len(values) == 8:
                break
        return values

    supported_claims = []
    raw_claims = value.get("supportedClaims")
    if isinstance(raw_claims, list):
        for item in raw_claims[:20]:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()[:500]
            urls = allowed_list(item.get("sourceUrls"), allowed_source_urls)
            document_citations = allowed_list(
                item.get("documentCitations"),
                allowed_document_citations,
            )
            # A claim is supported only when the audit maps it to web or document evidence gathered in this run.
            if claim and (urls or document_citations):
                supported_claims.append(
                    {
                        "claim": claim,
                        **({"sourceUrls": urls} if urls else {}),
                        **({"documentCitations": document_citations} if document_citations else {}),
                    }
                )

    audit = {
        "thesis": str(value.get("thesis") or "").strip()[:2000],
        "outline": short_list("outline", 16),
        "supportedClaims": supported_claims,
        "designInferences": short_list("designInferences", 16),
        "unsupportedPrecision": short_list("unsupportedPrecision", 16),
        "contradictions": short_list("contradictions", 12),
        "missingDimensions": short_list("missingDimensions", 12),
    }
    return {key: item for key, item in audit.items() if item}


def _streamed_titles(streamed: str) -> list[str]:
    """Plan step titles already complete in a partially streamed planner response.

    Only closed JSON strings match, so a title still being written is never published half
    formed. Escapes are decoded per match; the surrounding object is still incomplete, so the
    response as a whole cannot be parsed yet.
    """
    titles: list[str] = []
    for match in _STREAMED_TITLE.finditer(streamed):
        try:
            title = json.loads(f'"{match.group(1)}"')
        except ValueError:
            continue
        title = " ".join(str(title).split())[:120]
        if title:
            titles.append(title)
    return titles


def _next_unused_seed_action(plan: dict, used_queries: set[str]) -> dict[str, str] | None:
    for seed in plan.get("steps") or []:
        try:
            query = _sanitize_public_query(str(seed.get("query") or seed.get("title") or ""))
        except ValueError:
            continue
        if query in used_queries:
            continue
        return {
            "action": "search",
            "title": str(seed.get("title") or "Plan follow-up")[:200],
            "query": query,
        }
    return None


def _parse_and_validate_action(
    response: str,
    reasoning: str,
    allowed_urls: set[str],
    website_policy: dict | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    decoder = json.JSONDecoder()
    for candidate in (response, reasoning):
        valid_actions = []
        for match in re.finditer(r"\{", candidate):
            try:
                value, _end = decoder.raw_decode(candidate[match.start() :])
                if isinstance(value, dict):
                    valid_actions.append(
                        _validate_agent_action(value, allowed_urls, website_policy)
                    )
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        if valid_actions:
            return valid_actions[-1]
    if last_error is not None:
        raise last_error
    raise ValueError("Research agent did not return a JSON action")


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags = re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Planner did not return a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Planner response must be an object")
    return value


def _validate_plan(value: dict, max_steps: int) -> dict:
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Planner returned no steps")
    steps = []
    for raw in raw_steps[:max_steps]:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()[:200]
        raw_query = str(raw.get("query") or title).strip()
        if title and raw_query:
            try:
                query = _sanitize_public_query(raw_query)
            except ValueError:
                continue
            steps.append({"title": title, "query": query})
    if not steps:
        raise ValueError("Planner returned no valid steps")
    return {"title": str(value.get("title") or "Research plan").strip()[:200], "steps": steps}


def _parse_and_validate_plan(response: str, reasoning: str, max_steps: int) -> dict:
    last_error: Exception | None = None
    for candidate in (response, reasoning):
        if not candidate.strip():
            continue
        valid_plans: list[dict] = []
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", candidate):
            try:
                value, _end = decoder.raw_decode(candidate[match.start() :])
                if isinstance(value, dict):
                    valid_plans.append(_validate_plan(value, max_steps))
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        if valid_plans:
            return valid_plans[-1]
    if last_error is not None:
        raise last_error
    raise ValueError("Planner did not return a JSON object")


def _recover_report_from_reasoning(reasoning: str) -> str:
    text = reasoning.strip()
    marker = re.search(
        r"(?m)^(?:#{1,2}\s+(?:Executive\s+)?Summary\b|\*\*(?:Executive\s+)?Summary\*\*)",
        text,
        flags = re.IGNORECASE,
    )
    if marker is None:
        return ""
    report = text[marker.start() :].strip()
    return report if len(report) >= 500 else ""


def _report_after_boundary(text: str, boundary: str) -> str | None:
    lines = text.splitlines(keepends = True)
    fence_char: str | None = None
    fence_length = 0
    boundary_line: int | None = None
    for index, line in enumerate(lines):
        content = line.rstrip("\r\n")
        fence = _MARKDOWN_FENCE.match(content)
        if fence_char is not None:
            if fence is not None:
                token = fence.group(1)
                remainder = content[fence.end() :]
                if token[0] == fence_char and len(token) >= fence_length and not remainder.strip():
                    fence_char = None
                    fence_length = 0
            continue
        if fence is not None:
            token = fence.group(1)
            if token[0] == "`" and "`" in content[fence.end() :]:
                continue
            fence_char = token[0]
            fence_length = len(token)
            continue
        # CommonMark measures indentation in columns with a four-column tab stop, so one tab opens an
        # indented code block just as four spaces do.
        prefix = content[: len(content) - len(content.lstrip(" \t"))]
        indentation = len(prefix.expandtabs(4))
        # splitlines breaks on whitespace forms rstrip("\r\n") leaves behind, so strip every one rather than
        # let a stray one hide the boundary.
        if indentation <= 3 and content[len(prefix) :].strip().strip("`").strip() == boundary:
            boundary_line = index
    if boundary_line is None:
        return None
    return "".join(lines[boundary_line + 1 :]).strip()
