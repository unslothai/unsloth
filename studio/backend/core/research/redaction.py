# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keeping private data out of Deep Research web queries and prompts.

Search queries leave the machine, so credentials, personal identifiers, and private network
addresses are stripped before a query is issued. Prompt-delimiter tags in gathered evidence
are escaped so untrusted text cannot close a wrapper block and inject instructions, and a
source URL cannot close its own citation to inject a link.
"""

from __future__ import annotations

import ipaddress
import re


# Any occurrence inside untrusted evidence is escaped so gathered content cannot close a prompt block early.
_PROMPT_DELIMITER_TAGS = re.compile(
    r"</?\s*(?:untrusted_web_evidence|untrusted_evidence|source_catalog"
    r"|document_source_catalog|conversation_context_json|research_question"
    r"|approved_plan|untrusted_research_state_json|research_state_json"
    r"|untrusted_query_history_json|query_history_json"
    r"|untrusted_synthesis_audit_json|synthesis_audit_json)\s*>",
    re.IGNORECASE,
)
# An unescaped copy could move the synthesis boundary and truncate the report; spelled out to avoid
# importing prompts, and pinned by the hardening test.
_REPORT_BOUNDARY_TAG = re.compile(r"<!--\s*UNSLOTH_FINAL_REPORT\s*-->")
_QUERY_CREDENTIAL = re.compile(
    r"""(?ix)(?<![A-Za-z0-9])(?:api[\s_-]?key|access[\s_-]?(?:key|token)
    |auth[\s_-]?token|bearer[\s_-]?token|client[\s_-]?secret|private[\s_-]?key
    |refresh[\s_-]?token|session[\s_-]?token|authorization|password|secret|token)\s*[:=]\s*
    (?:"[^"]*"|'[^']*'|“[^”]*”|‘[^’]*’|[^\s,;]+)"""
)
_QUERY_NAMED_ASSIGNMENT = re.compile(
    r"""(?x)(?<![A-Za-z0-9])(?P<label>[A-Za-z][A-Za-z0-9_-]{0,100})\s*[:=]\s*
    (?P<value>"[^"]*"|'[^']*'|“[^”]*”|‘[^’]*’|[^\s,;]+)"""
)
_QUERY_CREDENTIAL_SUFFIXES = (
    "apikey",
    "accesskey",
    "accesstoken",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "privatekey",
    "refreshtoken",
    "secretkey",
    "sessiontoken",
    "authorization",
    "password",
    "token",
)
_QUERY_PUBLIC_ASSIGNMENT_SUFFIXES = ("designtoken", "cancellationtoken")
# Bearer tokens carry no key=value label; the length floor keeps ordinary prose ("bearer of bad news") from matching.
_QUERY_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_QUERY_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_QUERY_PRIVATE_ID = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_QUERY_OPAQUE_TOKEN = re.compile(
    r"\b(?:eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r"|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{16,}"
    r"|hf_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{20,}"
    r"|AKIA[A-Z0-9]{16})\b"
)
# International (+CC) or NANP-formatted phone numbers: requires separators or a leading "+" so
# bare numeric research terms are not redacted.
_QUERY_PHONE = re.compile(
    r"(?<!\w)\+\d[\d\s().-]{7,17}\d(?!\w)|(?<!\w)\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w)"
)
_QUERY_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_QUERY_IPV6 = re.compile(
    r"(?<![0-9A-Fa-f:])\[?(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f.]*(?:%[A-Za-z0-9_.-]+)?\]?"
    r"(?![0-9A-Fa-f:])"
)
_QUERY_LABELED_PRIVATE_ID = re.compile(
    r"(?ix)\b(?:passport|driver(?:'s)?[\s_-]?licen[cs]e|national[\s_-]?id"
    r"|tax[\s_-]?id|account[\s_-]?(?:number|no))\s*[:=#-]?\s*[A-Za-z0-9][A-Za-z0-9_-]{4,24}\b"
)
_QUERY_PAYMENT_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def _luhn_valid(candidate: str) -> bool:
    digits = [int(character) for character in candidate if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _redact_nonpublic_ip(match: "re.Match[str]") -> str:
    try:
        return " " if not ipaddress.ip_address(match.group(0)).is_global else match.group(0)
    except ValueError:
        return match.group(0)


def _redact_nonpublic_ipv6(match: "re.Match[str]") -> str:
    # Strip brackets and any zone id before validating; redact non-global addresses.
    candidate = match.group(0).strip("[]").split("%", 1)[0]
    try:
        return " " if not ipaddress.ip_address(candidate).is_global else match.group(0)
    except ValueError:
        return match.group(0)


def _escape_link_destination(url: str) -> str:
    # Escape an unbalanced ")" so a source URL cannot close the citation and inject a link.
    out: list[str] = []
    depth = 0
    for char in url:
        if char == "\\":
            out.append("\\\\")
        elif char == "(":
            depth += 1
            out.append(char)
        elif char == ")" and depth == 0:
            out.append("\\)")
        else:
            if char == ")":
                depth -= 1
            out.append(char)
    return "".join(out)


def _shield_untrusted(text: str) -> str:
    """Escape prompt-delimiter tags and the report boundary marker in untrusted evidence, so
    gathered content cannot close a wrapper block to inject model instructions, nor move the
    boundary that selects the published report."""
    if not text:
        return text

    def escape(match: re.Match) -> str:
        return match.group(0).replace("<", "&lt;").replace(">", "&gt;")

    return _REPORT_BOUNDARY_TAG.sub(escape, _PROMPT_DELIMITER_TAGS.sub(escape, text))


def _sanitize_public_query(query: str) -> str:
    def redact_named_assignment(match: re.Match) -> str:
        label = re.sub(r"[^a-z0-9]", "", match.group("label").lower())
        if label.endswith(_QUERY_CREDENTIAL_SUFFIXES) and not label.endswith(
            _QUERY_PUBLIC_ASSIGNMENT_SUFFIXES
        ):
            return " "
        return match.group(0)

    query = _QUERY_CREDENTIAL.sub(" ", query)
    query = _QUERY_NAMED_ASSIGNMENT.sub(redact_named_assignment, query)
    query = _QUERY_BEARER.sub(" ", query)
    query = _QUERY_EMAIL.sub(" ", query)
    query = _QUERY_PRIVATE_ID.sub(" ", query)
    query = _QUERY_OPAQUE_TOKEN.sub(" ", query)
    query = _QUERY_PHONE.sub(" ", query)
    query = _QUERY_LABELED_PRIVATE_ID.sub(" ", query)
    query = _QUERY_IPV4.sub(_redact_nonpublic_ip, query)
    query = _QUERY_IPV6.sub(_redact_nonpublic_ipv6, query)
    query = _QUERY_PAYMENT_CARD.sub(
        lambda match: " " if _luhn_valid(match.group(0)) else match.group(0),
        query,
    )
    query = " ".join(query.split()).strip(" ,;:-")[:500]
    if not any(character.isalnum() for character in query):
        raise ValueError("Research query contained only private or credential-like data")
    return query
