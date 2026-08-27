# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mask credentials in log text before it leaves the process.

Nothing redacts secrets today: loggers/handlers.py:filter_sensitive_data only
masks native path leases, and raw output (faulthandler dumps, uvicorn, third
party prints) never passes through a structlog processor at all. The log viewer
invites users to copy lines into a bug report, so the masking happens on read.

Every pattern is anchored on a known credential prefix or a key name. There is
deliberately NO generic "long high entropy string" rule: that would eat sha256
blob digests, HF revisions, snapshot paths and GGUF tensor names, which is
exactly the content someone opened the log to read.
"""

from __future__ import annotations

import re

REDACTED = "<redacted>"

# Terminal control sequences, stripped BEFORE anything is matched. A colorized
# writer puts an escape between the key and its value (ConsoleRenderer emits
# "\x1b[36mapi_key\x1b[0m=\x1b[35m<secret>\x1b[0m", and colors default on even
# off-terminal); the "m" ending "\x1b[36m" is a word character, so every anchored
# rule below stops matching and the credential goes out untouched.
#
# Order matters: OSC (\x1b]) comes before the single-character Fe class, which
# covers 0x5C-0x5F and would otherwise swallow the "]" and leave the payload.
# ECMA-48 5.4 (CSI) and 5.6 (OSC / DCS / SOS / PM / APC).
_ANSI_RE = re.compile(
    r"\x1b\][\s\S]*?(?:\x07|\x1b\\|\x9c)"  # OSC ... BEL / ST
    r"|\x1b[P^_X][\s\S]*?(?:\x1b\\|\x9c)"  # DCS / PM / APC / SOS ... ST
    r"|\x1b\[[0-?]*[ -/]*[@-~]"  # CSI (colors, cursor moves)
    r"|\x1b[@-Z\\-_]"  # other two-character Fe escapes
    r"|\x9b[0-?]*[ -/]*[@-~]"  # 8-bit CSI
    r"|[\x9d\x90\x98\x9e\x9f][\s\S]*?(?:\x07|\x9c)"  # 8-bit OSC / DCS / SOS / PM / APC
)
_ANSI_INTRODUCER_RE = re.compile(r"[\x1b\x90\x98\x9b\x9d-\x9f]")

# Key names whose VALUE is a secret. "token" alone is absent on purpose, so
# n_tokens = 4096 and token_id=128009 survive.
_SECRET_KEYS = (
    "authorization|x-api-key|api[-_]?key|apikey|hf[-_]?token|access[-_]?token|"
    "refresh[-_]?token|auth[-_]?token|bearer[-_]?token|client[-_]?secret|"
    "aws_secret_access_key|aws_session_token|wandb[-_]?token|hub[-_]?token|"
    # Unsloth's own S3 field (models/training.py:60) and its camelCase alias.
    # Neither is reachable through the bare "secret" alternative (the trailing \b
    # cannot fire before "_access" or "Access"), and an AWS secret key has no
    # prefix of its own for a shape rule to catch.
    "secret[-_]?access[-_]?key|"
    "password|passwd|secret"
)

# No leading \b: "_" is a word character, so \b never fires inside
# OPENAI_API_KEY / db_password, the shape an env dump or argv line carries. The
# trailing \b stays, so eos_token_id and secret_sauce_path are left alone.
_KEY_START = r"(?<![A-Za-z0-9])"

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Hugging Face
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "hf_" + REDACTED),
    # OpenAI and other sk- keys (project, Anthropic, OpenRouter). Not \b: that
    # also fires after a hyphen, eating checkpoint-sk-9f8a... in a filename.
    (
        re.compile(r"(?<![A-Za-z0-9-])sk-(?:proj-|ant-api\d{2}-|or-v1-)?[A-Za-z0-9_-]{16,}"),
        "sk-" + REDACTED,
    ),
    # Other vendor prefixes
    (
        re.compile(
            r"\b(?:gsk_|xai-|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|"
            r"xox[abpsr]-|ya29\.)[A-Za-z0-9_.-]{16,}"
        ),
        REDACTED,
    ),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), REDACTED),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTED),
    # JWTs, including the desktop access token
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"), REDACTED),
    # user:password@host in a URL
    (re.compile(r"://[^/\s:@]+:[^/\s@]+@"), "://" + REDACTED + "@"),
    # Presigned URL parameters. Bare "key" is deliberately absent: in an object
    # storage URL it names the object, and blanking it hides WHICH download
    # failed. Google's ?key=AIza... is caught by the AIza rule above.
    (
        re.compile(
            r"(?i)([?&](?:token|api[-_]key|apikey|sig|signature|x-amz-signature|"
            r"x-amz-credential|x-amz-security-token|access_token)=)[^&\s\"']+"
        ),
        r"\1" + REDACTED,
    ),
)

# key = value / "key": "value" / --api-key value
#
# The QUOTED branch wins whenever an opening quote is there, so a quoted
# credential is consumed to its CLOSING quote. Stopping at whitespace turned
# password="correct horse battery staple" into password="<redacted> horse
# battery staple", which reads as masked while leaking all but the first word.
#
# "[^\"'\\\n]|\\." rather than a lazy ".*?" so an escaped quote does not end the
# value early; \n is excluded so an unterminated quote cannot run the mask past
# its own line.
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

# An Authorization value, whatever the scheme. The key/value rule cannot reach
# it: for "Authorization: Basic dXNlcjpwdw==" the value it captures is "Basic",
# leaving the credential behind it. Same for a Cookie, which for Unsloth is the
# UI session that gates these very endpoints.
_SCHEMES = ("bearer", "basic", "digest", "token", "apikey")
# A scheme word only introduces a credential when an Authorization header put it
# there. Bare "digest sha256:..." and "token hf_..." are ordinary log content,
# and firing on the word alone blanked the digest a user came here to read.
# The credential stops at a quote or a structural delimiter, not at the next
# space: \S+ swallowed the closing quote and every field behind it, so a compact
# header dict came back as {"Authorization":"Bearer <redacted> with the request
# id and status gone with it.
_CREDENTIAL = r"[^\s\"',}\]]+"
_AUTH_HEADER_RE = re.compile(
    r"(?i)((?:proxy-)?authorization[\"']?\s*[:=]\s*[\"']?"
    r"(?:" + "|".join(_SCHEMES) + r"))(\s+)(" + _CREDENTIAL + r")"
)
# Bearer is not an English word that shows up in a log on its own, so it keeps
# a header-less rule; the shape guard still spares "Bearer credentials expired".
_SCHEME_RE = re.compile(r"(?i)\b(Bearer)(\s+)(" + _CREDENTIAL + r")")
# MULTILINE: this also runs over exception text, where the header is not on the
# last line. The optional quote matters: headers are usually logged as a dict,
# and the pair test never matched a value that opened with a quote, so the
# session cookie gating these very endpoints went out in the clear.
_COOKIE_RE = re.compile(
    r"(?i)\b(?P<key>(?:set-)?cookie)(?P<sep>[\"']?\s*[:=]\s*(?P<q>[\"'])?)(?P<val>\S.*)$",
    re.MULTILINE,
)

# Keys whose value is a secret even when it is all digits (a numeric password is
# still a password); everywhere else a bare number is a count or an id.
_NUMERIC_IS_STILL_SECRET = re.compile(r"(?i)pass(word|wd)?$|secret$")


def _looks_like_credential(value: str) -> bool:
    """Token-shaped rather than an English word.

    Guards the rules keyed on a weak name: "Bearer credentials were not
    accepted" and "Cookie: disabled" are log content, and blanking them hides
    the failure being diagnosed.
    """
    if len(value) < 8:
        return False
    if len(value) >= 20:
        return True
    has_digit = any(char.isdigit() for char in value)
    has_symbol = any(char in "._-+/=~" for char in value)
    mixed_case = any(char.isupper() for char in value) and any(char.islower() for char in value)
    return has_digit or has_symbol or mixed_case


def _redact_kv(match: re.Match[str]) -> str:
    # Named groups: the quoted/unquoted branch adds a group, so positional
    # numbering is not stable.
    value = match.group("val")
    if value.isdigit() and not _NUMERIC_IS_STILL_SECRET.search(match.group("key")):
        return match.group(0)
    # Quoting puts the scheme inside the value ('authorization': 'Basic abc').
    # Step over it rather than abandon the match: the rest is still the
    # credential, and blanking the scheme reads as if the header were the secret.
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


# A cookie header is name=value pairs. _COOKIE_RE takes the rest of the line, so
# without this the length shortcut reads "Cookie: not sent because the origin is
# cross-site" as a token and masks the diagnosis.
_COOKIE_PAIR_RE = re.compile(r"^[A-Za-z0-9_.\-]+=\S")


def _redact_cookie(match: re.Match[str]) -> str:
    value, tail = match.group("val"), ""
    # A quoted value ends at its closing quote, so the fields behind it in a
    # header dict survive instead of disappearing into the mask.
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
    # Nothing anchored below survives an escape between a key and its value, so
    # strip first, guarded by one introducer scan: ordinary content is untouched.
    if _ANSI_INTRODUCER_RE.search(text):
        text = _ANSI_RE.sub("", text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    # Before the key/value rules: _KV_RE captures "Basic" from "Authorization:
    # Basic dXNlcjpwdw==", masking the scheme and leaving the credential clear.
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
