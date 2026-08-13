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
# writer puts an escape between the key and its value (structlog's
# ConsoleRenderer renders "\x1b[36mapi_key\x1b[0m=\x1b[35m<secret>\x1b[0m", and
# its `colors` default is on whether or not the sink is a terminal), and every
# rule here is anchored: the "m" that ends "\x1b[36m" is a word character, so
# _KEY_START's lookbehind and the \b in front of \bhf_ both stop matching and
# the credential goes out untouched. The viewer's own pane strips these before
# it renders, so the escapes are never what the reader sees anyway.
#
# Order matters: OSC (\x1b]) is listed before the single-character Fe class,
# which covers 0x5C-0x5F and would otherwise swallow the "]" and leave the
# payload behind. ECMA-48 5.4 (CSI) and 5.6 (OSC / DCS / SOS / PM / APC).
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
    "password|passwd|secret"
)

# The key name is matched WITHOUT a leading \b. "_" is a word character, so \b
# never fires inside OPENAI_API_KEY / WANDB_API_KEY / db_password, which is the
# exact shape an env dump or a subprocess argv line carries. The trailing \b
# stays, so eos_token_id and secret_sauce_path are still left alone.
_KEY_START = r"(?<![A-Za-z0-9])"

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Hugging Face
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "hf_" + REDACTED),
    # OpenAI and the sk- style keys (project, Anthropic, OpenRouter). Not \b:
    # that also fires after a hyphen, so checkpoint-sk-9f8a... in a filename
    # would be eaten.
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
    # storage URL that names the object, and blanking it hides WHICH download
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
# The value has two branches and the QUOTED one wins whenever an opening quote
# is there, so a credential the writer quoted is consumed to its CLOSING quote
# instead of stopping at the first space. A value that stopped at whitespace
# turned password="correct horse battery staple" into
# password="<redacted> horse battery staple", which reads as masked while
# leaking all but the first word, and left --api-key "abc defgh" untouched
# because the value class rejected the leading quote outright. Passphrases with
# spaces are the realistic shape here; a partially masked one is worse than an
# unmasked one, because it invites the user to paste the line into a bug report.
#
# "[^\"'\\\n]|\\." rather than a lazy ".*?" so an escaped quote inside the value
# does not end it early, and \n is excluded so an unterminated quote cannot run
# the mask past the end of its own line.
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

# An Authorization value, whatever the scheme. The key/value rule alone cannot
# reach it: for "Authorization: Basic dXNlcjpwdw==" the value it captures is
# "Basic", so the credential right after it survived untouched. Same for a
# Cookie, which for Studio is the UI session that gates these very endpoints.
_SCHEMES = ("bearer", "basic", "digest", "token", "apikey")
# The scheme word is only trusted to introduce a credential when an
# Authorization header put it there. Bare "digest sha256:..." and "token
# hf_..." are ordinary log content, and a rule that fired on the word alone
# blanked the digest a user came here to read.
_AUTH_HEADER_RE = re.compile(
    r"(?i)((?:proxy-)?authorization[\"']?\s*[:=]\s*[\"']?"
    r"(?:" + "|".join(_SCHEMES) + r"))(\s+)(\S+)"
)
# Bearer is not an English word that shows up in a log on its own, so it keeps
# a header-less rule; the shape guard still spares "Bearer credentials expired".
_SCHEME_RE = re.compile(r"(?i)\b(Bearer)(\s+)(\S+)")
# MULTILINE: this also runs over exception text, where the header is not on the
# last line.
_COOKIE_RE = re.compile(r"(?i)\b((?:set-)?cookie)(\s*[:=]\s*)(\S.*)$", re.MULTILINE)

# Keys whose value is a secret even when it is all digits (a numeric password is
# still a password); everywhere else a bare number is a count or an id.
_NUMERIC_IS_STILL_SECRET = re.compile(r"(?i)pass(word|wd)?$|secret$")


def _looks_like_credential(value: str) -> bool:
    """Token-shaped rather than an English word.

    Guards the rules keyed on a weak name. "Bearer credentials were not
    accepted" and "Cookie: disabled" are log content someone opened the viewer
    to read, and blanking them hides the failure being diagnosed.
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
    # Named groups: the quoted/unquoted branch adds a group of its own, so the
    # positional numbering these three used to carry is no longer stable.
    value = match.group("val")
    # A numeric value is a count or an id, never a credential.
    if value.isdigit() and not _NUMERIC_IS_STILL_SECRET.search(match.group("key")):
        return match.group(0)
    # "Authorization: Bearer <redacted>": the scheme word is not the secret, and
    # blanking it too would read as if the header itself were the credential.
    # Quoting puts the scheme INSIDE the value ('authorization': 'Basic abc'),
    # so the scheme is stepped over rather than the whole match abandoned: the
    # rest of the value is still the credential.
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


# A cookie header is name=value pairs. _COOKIE_RE takes the whole rest of the
# line, so without this "Cookie: not sent because the origin is cross-site" is
# 20-odd characters and the length shortcut reads it as a token. That line is a
# diagnosis, and masking it hides the thing being diagnosed.
_COOKIE_PAIR_RE = re.compile(r"^[A-Za-z0-9_.\-]+=\S")


def _redact_cookie(match: re.Match[str]) -> str:
    if not _COOKIE_PAIR_RE.match(match.group(3).strip()):
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}{REDACTED}"


def redact_log_text(text: str) -> str:
    """Mask credentials. Idempotent, and a no-op on ordinary log content."""
    if not text:
        return text
    # Nothing anchored below survives an escape sequence sitting between a key
    # and its value, so normalize first. Guarded by one scan for an introducer:
    # ordinary log content carries none and is left exactly as it was.
    if _ANSI_INTRODUCER_RE.search(text):
        text = _ANSI_RE.sub("", text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    # Before the key/value rules: for "Authorization: Basic dXNlcjpwdw==" the
    # value _KV_RE captures is the word "Basic", so running it first would mask
    # the scheme and leave the credential behind it in the clear.
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
