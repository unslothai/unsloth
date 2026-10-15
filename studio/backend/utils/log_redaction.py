# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mask credentials in log text before it leaves the process.

Nothing redacts secrets today: loggers/handlers.py:filter_sensitive_data only masks native path leases, and raw output (faulthandler dumps, uvicorn, third party prints) never passes through a structlog processor at all. The log viewer invites users to copy lines into a bug report, so the masking happens on read.

Every pattern is anchored on a known credential prefix or a key name. There is deliberately NO generic "long high entropy string" rule: that would eat sha256 blob digests, HF revisions, snapshot paths and GGUF tensor names, exactly the content someone opened the log to read.
"""

from __future__ import annotations

import re

REDACTED = "<redacted>"

# Strip ANSI first: escapes between key and value defeat the anchored rules.
# _strip_ansi must stay output-identical to the old regex (differential fuzzed).
_CSI_7BIT_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CSI_8BIT_RE = re.compile(r"\x9b[0-?]*[ -/]*[@-~]")
# Fe also claims an unterminated "]" or "P", hence tried last.
_FE_RE = re.compile(r"\x1b[@-Z\\-_]")
_ANSI_INTRODUCER_RE = re.compile(r"[\x1b\x90\x98\x9b\x9d-\x9f]")
_C1_STRING_INTRODUCERS = "\x9d\x90\x98\x9e\x9f"


def _strip_ansi(text: str) -> str:
    """Remove terminal control sequences. Same output as the lazy alternation this replaces, in linear time."""
    first = _ANSI_INTRODUCER_RE.search(text)
    if first is None:
        return text
    out: list[str] = []
    written = 0
    index = first.start()
    length = len(text)
    # index only moves forward, so cached hits stay valid; keeps the scan linear.
    found: dict[str, int] = {}

    def next_index(needle: str, start: int) -> int:
        cached = found.get(needle, -2)
        if cached == -1:
            return -1
        if cached == -2 or cached < start:
            cached = text.find(needle, start)
            found[needle] = cached
        return cached

    def string_end(terminators: tuple[str, ...], start: int) -> int:
        """End of the shortest body, which is what a lazy quantifier picks."""
        best, best_length = -1, 0
        for terminator in terminators:
            at = next_index(terminator, start)
            if at >= 0 and (best < 0 or at < best):
                best, best_length = at, len(terminator)
        return best + best_length if best >= 0 else -1

    while index < length:
        char = text[index]
        end = -1
        if char == "\x1b" and index + 1 < length and text[index + 1] == "]":
            end = string_end(("\x07", "\x1b\\", "\x9c"), index + 2)
        elif char == "\x1b" and index + 1 < length and text[index + 1] in "P^_X":
            end = string_end(("\x1b\\", "\x9c"), index + 2)
        elif char in _C1_STRING_INTRODUCERS:
            end = string_end(("\x07", "\x9c"), index + 1)
        if end < 0:
            if char == "\x1b":
                match = _CSI_7BIT_RE.match(text, index) or _FE_RE.match(text, index)
            elif char == "\x9b":
                match = _CSI_8BIT_RE.match(text, index)
            else:
                match = None
            end = match.end() if match else -1
        if end < 0:
            # Jump to the next introducer, not the next character, to stay linear.
            following = _ANSI_INTRODUCER_RE.search(text, index + 1)
            if following is None:
                break
            index = following.start()
            continue
        out.append(text[written:index])
        written = end
        following = _ANSI_INTRODUCER_RE.search(text, end)
        index = following.start() if following else length
    out.append(text[written:])
    return "".join(out)


# Bare "token" is absent on purpose so n_tokens and token_id survive.
_SECRET_KEYS = (
    "authorization|x-api-key|api[-_]?key|apikey|hf[-_]?token|access[-_]?token|"
    "refresh[-_]?token|auth[-_]?token|bearer[-_]?token|client[-_]?secret|"
    "aws_secret_access_key|aws_session_token|wandb[-_]?token|hub[-_]?token|"
    # Unsloth's S3 field and camelCase alias; AWS secrets have no shape prefix.
    "secret[-_]?access[-_]?key|"
    "password|passwd|secret"
)

# No leading \b since "_" is a word char (OPENAI_API_KEY); trailing \b stays.
_KEY_START = r"(?<![A-Za-z0-9])"

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bhf_(?:oauth_[A-Za-z0-9._~+/=-]{20,}|[A-Za-z0-9]{20,})"), "hf_" + REDACTED),
    # No \b: it would fire after a hyphen, eating checkpoint-sk-... filenames.
    (
        re.compile(r"(?<![A-Za-z0-9-])sk-(?:proj-|ant-api\d{2}-|or-v1-)?[A-Za-z0-9_-]{16,}"),
        "sk-" + REDACTED,
    ),
    (
        re.compile(
            r"\b(?:gsk_|xai-|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|"
            r"xox[abpsr]-|ya29\.)[A-Za-z0-9_.-]{16,}"
        ),
        REDACTED,
    ),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), REDACTED),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTED),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"), REDACTED),
    (re.compile(r"://[^/\s:@]+:[^/\s@]+@"), "://" + REDACTED + "@"),
    # Bare "key" omitted: in object storage URLs it names the object.
    (
        re.compile(
            r"(?i)([?&](?:token|api[-_]key|apikey|sig|signature|x-amz-signature|"
            r"x-amz-credential|x-amz-security-token|access_token)=)[^&\s\"']+"
        ),
        r"\1" + REDACTED,
    ),
)

# Quoted branch wins so multi-word values are fully masked; newlines end the value.
_QUOTED_VALUE = r"(?:[^\"'\\\n]|\\.){6,}"
_KV_RE = re.compile(
    r"(?i)" + _KEY_START + r"(?P<key>" + _SECRET_KEYS + r")\b"
    r"(?P<sep>[\"']?\s*[:=]\s*(?P<q>[\"'])?)"
    r"(?P<val>(?(q)" + _QUOTED_VALUE + r"|[^\"'\s,}\]]{6,}))"
)
_FLAG_RE = re.compile(
    r"(?i)(?P<key>--(?:" + _SECRET_KEYS + r"))"
    r"(?P<sep>\s+(?P<q>[\"'])?)"
    r"(?P<val>(?(q)" + _QUOTED_VALUE + r"|[^\s\"']{6,}))"
)

# Mask the full Authorization/Cookie value whatever the scheme.
_SCHEMES = ("bearer", "basic", "digest", "token", "apikey")
# Scheme words count only after an Authorization header; stop at delimiters.
_CREDENTIAL = r"[^\s\"',}\]]+"
_AUTH_HEADER_RE = re.compile(
    r"(?i)((?:proxy-)?authorization[\"']?\s*[:=]\s*[\"']?"
    r"(?:" + "|".join(_SCHEMES) + r"))(\s+)(" + _CREDENTIAL + r")"
)
# Bearer keeps a header-less rule; the shape guard spares plain English.
_SCHEME_RE = re.compile(r"(?i)\b(Bearer)(\s+)(" + _CREDENTIAL + r")")
# MULTILINE: this also runs over exception text.
_COOKIE_RE = re.compile(
    r"(?i)\b(?P<key>(?:set-)?cookie)(?P<sep>[\"']?\s*[:=]\s*(?P<q>[\"'])?)(?P<val>\S.*)$",
    re.MULTILINE,
)

# Numeric values stay secret only for password/secret keys.
_NUMERIC_IS_STILL_SECRET = re.compile(r"(?i)pass(word|wd)?$|secret$")


def _looks_like_credential(value: str) -> bool:
    """Token-shaped rather than an English word. Guards the rules keyed on a weak name: "Bearer credentials were not accepted" and "Cookie: disabled" are log content, and blanking them hides the failure being diagnosed."""
    if len(value) < 8:
        return False
    if len(value) >= 20:
        return True
    has_digit = any(char.isdigit() for char in value)
    has_symbol = any(char in "._-+/=~" for char in value)
    mixed_case = any(char.isupper() for char in value) and any(char.islower() for char in value)
    return has_digit or has_symbol or mixed_case


def _redact_kv(match: re.Match[str]) -> str:
    # Named groups: the quoted branch shifts positional numbering.
    value = match.group("val")
    if value.isdigit() and not _NUMERIC_IS_STILL_SECRET.search(match.group("key")):
        return match.group(0)
    # A quoted value may include the scheme; skip it and mask the rest.
    scheme, sep, rest = value.partition(" ")
    if scheme.lower() in _SCHEMES:
        if not sep or not rest.strip():
            return match.group(0)
        return f"{match.group('key')}{match.group('sep')}{scheme}{sep}{REDACTED}"
    return f"{match.group('key')}{match.group('sep')}{REDACTED}"


def _redact_shaped(match: re.Match[str]) -> str:
    if not _looks_like_credential(match.group(3)):
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}{REDACTED}"


_COOKIE_PAIR_RE = re.compile(r"^[A-Za-z0-9_.\-]+=\S")


def _redact_cookie(match: re.Match[str]) -> str:
    value, tail = match.group("val"), ""
    # A quoted value ends at its closing quote so following fields survive.
    quote = match.group("q")
    if quote:
        end = value.find(quote)
        if end != -1:
            value, tail = value[:end], value[end:]
    if not _COOKIE_PAIR_RE.match(value.strip()):
        return match.group(0)
    return f"{match.group('key')}{match.group('sep')}{REDACTED}{tail}"


def redact_log_text(text: str) -> str:
    """Mask credentials. Idempotent, and a no-op on ordinary log content."""
    if not text:
        return text
    if _ANSI_INTRODUCER_RE.search(text):
        text = _strip_ansi(text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    # Before the key/value rules, which would mask only the scheme word.
    text = _AUTH_HEADER_RE.sub(_redact_shaped, text)
    text = _SCHEME_RE.sub(_redact_shaped, text)
    text = _COOKIE_RE.sub(_redact_cookie, text)
    text = _KV_RE.sub(_redact_kv, text)
    text = _FLAG_RE.sub(_redact_kv, text)
    try:
        from utils.native_path_leases import redact_native_paths
        text = redact_native_paths(text)
    except Exception:
        pass
    return text
