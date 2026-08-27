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

from utils.secret_env import SECRET_ENV_NAMES, SECRET_ENV_PREFIXES, is_secret_env_name

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
    "session[-_]?token|"
    "aws_secret_access_key|aws_session_token|wandb[-_]?token|hub[-_]?token|"
    # Unsloth's own S3 field (models/training.py:60) and its camelCase alias.
    # Neither is reachable through the bare "secret" alternative (the trailing \b
    # cannot fire before "_access" or "Access"), and an AWS secret key has no
    # prefix of its own for a shape rule to catch.
    "secret[-_]?access[-_]?key|shared[-_]?access[-_]?key|access[-_]?key|"
    "account[-_]?key|private[-_]?key(?:[-_]?data)?|pwd|"
    "password|passwd|passphrase|secret"
)
_FLAG_SECRET_KEYS = _SECRET_KEYS + "|token"

# No leading \b: "_" is a word character, so \b never fires inside
# OPENAI_API_KEY / db_password, the shape an env dump or argv line carries. The
# trailing \b stays, so eos_token_id and secret_sauce_path are left alone.
_KEY_START = r"(?<![A-Za-z0-9])"
_QUERY_SECRET_KEYS = (
    r"token|api[-_]key|apikey|sig|signature|x-amz-signature|"
    r"x-amz-credential|x-amz-security-token|x-goog-signature|access_token"
)

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
    # URL userinfo may be user:password, :password (Redis), or a token alone.
    (re.compile(r"://[^/?#\s]+@"), "://" + REDACTED + "@"),
    # Presigned URL parameters. Bare "key" is deliberately absent: in an object
    # storage URL it names the object, and blanking it hides WHICH download
    # failed. Google's ?key=AIza... is caught by the AIza rule above.
    (
        re.compile(r"(?i)([?&](?:" + _QUERY_SECRET_KEYS + r")=)[^&\s\"']+"),
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
# A backreference closes the same quote that opened the value. This lets a
# single-quoted secret contain double quotes (and vice versa) without exposing
# the suffix, while an escaped matching quote does not end the value early.
_QUOTED_VALUE = r"(?:\\.|(?!(?P=quote))[^\\\n])*"
_UNTERMINATED_QUOTED_VALUE = _QUOTED_VALUE + r"\\?"
_PYTHON_BYTES_PREFIX = r"(?:[bB][rR]?|[rR][bB])"
_SHELL_WORD_SUFFIX = r"(?:\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|\\[^\r\n]|[^\s\\'\";&|<>()])*"
_ENV_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<sep>=)(?:(?P<quote>[\"'])(?P<quoted>"
    + _QUOTED_VALUE
    + r")(?P=quote)(?P<suffix>"
    + _SHELL_WORD_SUFFIX
    + r")|(?P<val>(?:\\[^\r\n]|[^\s\\])+))"
)
_STRUCTURED_ENV_KV_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<key_text>(?:(?P<key_quote>[\"'])(?P<quoted_key>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P=key_quote)|(?P<plain_key>[A-Za-z_][A-Za-z0-9_]*)))"
    r"(?P<sep>\s*:\s*)"
    r"(?:(?P<value_bytes>" + _PYTHON_BYTES_PREFIX + r")?(?P<quote>[\"'])"
    r"(?P<quoted>" + _QUOTED_VALUE + r")(?P=quote)|"
    r"(?P<val>[^\"'\s,}\]]+))"
)
_QUOTED_KV_RE = re.compile(
    r"(?i)" + _KEY_START + r"(?P<key>" + _SECRET_KEYS + r")\b"
    r"(?P<sep>[\"']?\s*[:=]\s*)(?P<value_bytes>" + _PYTHON_BYTES_PREFIX + r")?"
    r"(?P<quote>[\"'])"
    r"(?P<val>" + _QUOTED_VALUE + r")(?P=quote)"
)
_UNTERMINATED_QUOTED_KV_RE = re.compile(
    r"(?i)" + _KEY_START + r"(?P<key>" + _SECRET_KEYS + r")\b"
    r"(?P<sep>[\"']?\s*[:=]\s*)(?P<value_bytes>" + _PYTHON_BYTES_PREFIX + r")?"
    r"(?P<quote>[\"'])"
    r"(?P<val>" + _UNTERMINATED_QUOTED_VALUE + r")(?=\r?$)",
    re.MULTILINE,
)
_CONTAINER_KV_START_RE = re.compile(
    r"(?i)" + _KEY_START + r"(?P<key>" + _SECRET_KEYS + r")\b"
    r"(?P<sep>[\"']?\s*[:=]\s*)(?P<open>[\[({])"
)
_PLAIN_SCALAR_KV_RE = re.compile(
    r"(?i)" + _KEY_START + r"(?P<key>" + _SECRET_KEYS + r")\b"
    r"(?P<sep>[\"']?\s*[:=]\s*)(?!<redacted>)"
    r"(?!" + _PYTHON_BYTES_PREFIX + r"[\"'])"
    r"(?P<val>[^\"'\s|>,}\]][^\"'\r\n,}\]]*)"
)
_ESCAPED_QUOTED_KV_RE = re.compile(
    r"(?i)" + _KEY_START + r"(?P<key>(?:" + _SECRET_KEYS + r"|(?:set-)?cookie))\b"
    r"(?P<sep>\\(?P<key_quote>[\"'])\s*[:=]\s*\\(?P<quote>[\"']))"
    r"(?P<rest>[^\r\n]*)"
)
_QUOTED_HEADER_PAIR_RE = re.compile(
    r"(?i)(?P<key_bytes>b)?(?P<key_quote>[\"'])(?P<key>(?:" + _SECRET_KEYS + r"|(?:set-)?cookie))"
    r"(?P=key_quote)(?P<sep>\s*,\s*)(?P<value_bytes>b)?(?P<quote>[\"'])"
    r"(?P<val>" + _QUOTED_VALUE + r")(?P=quote)"
)
_KV_RE = re.compile(
    r"(?i)" + _KEY_START + r"(?P<key>" + _SECRET_KEYS + r")\b"
    r"(?P<sep>[\"']?\s*[:=]\s*)(?!<redacted>)(?!" + _PYTHON_BYTES_PREFIX + r"[\"'])"
    r"(?P<val>[^\"'\s,}\]]+)"
)
_QUOTED_FLAG_RE = re.compile(
    r"(?i)(?P<key>--(?:" + _FLAG_SECRET_KEYS + r"))"
    r"(?P<sep>\s+)(?P<quote>[\"'])(?P<val>" + _QUOTED_VALUE + r")(?P=quote)"
)
_UNTERMINATED_QUOTED_FLAG_RE = re.compile(
    r"(?i)(?P<key>--(?:" + _FLAG_SECRET_KEYS + r"))"
    r"(?P<sep>\s+)(?P<quote>[\"'])(?P<val>" + _UNTERMINATED_QUOTED_VALUE + r")(?=\r?$)",
    re.MULTILINE,
)
_FLAG_RE = re.compile(
    r"(?i)(?P<key>--(?:" + _FLAG_SECRET_KEYS + r"))(?P<sep>\s+)(?P<val>[^\s\"']+)"
)

# YAML and similar structured logs may put a credential value on the next
# physical line. ``redact_log_text`` can mask that shape when it receives both
# lines together, but bounded streaming readers deliberately process one record
# at a time. Keep only structural state: either the next non-empty record is
# sensitive, or indented records belong to a YAML block scalar. Never retain a
# credential value itself.
_CONTINUED_SECRET_RE = re.compile(
    r"(?i)^(?P<indent>[ \t]*)" + _KEY_START + r"(?:" + _SECRET_KEYS + r")\b[\"']?\s*[:=]\s*"
    r"(?P<block>[|>](?:[1-9][+-]?|[+-][1-9]?)?)?\s*(?:#.*)?$"
)
_CONTINUED_ENV_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*$")
_CONTINUED_COOKIE_RE = re.compile(r"(?i)^(?P<indent>[ \t]*)(?:set-)?cookie[\"']?\s*[:=]\s*$")
_CONTINUED_COOKIE_PAIR_RE = re.compile(
    r"(?i)^(?P<indent>[ \t]*).*?(?P<quote>[\"'])(?:set-)?cookie(?P=quote)\s*,\s*$"
)
_INLINE_COOKIE_RE = re.compile(
    r"(?i)^(?P<indent>[ \t]*)(?:set-)?cookie[\"']?\s*[:=]\s*[\"']?(?P<value>\S.*)$"
)
_UNTERMINATED_QUOTED_SECRET_RE = re.compile(
    r"(?i)^(?P<indent>[ \t]*)" + _KEY_START + r"(?:" + _SECRET_KEYS + r")\b[\"']?\s*[:=]\s*"
    r"(?:(?P<double>\")(?:\\.|[^\"\\])*\\?|(?P<single>')(?:\\.|[^'\\])*\\?)$"
)
_INLINE_PLAIN_SECRET_RE = re.compile(
    r"(?i)^(?P<indent>[ \t]*)"
    + _KEY_START
    + r"(?:"
    + _SECRET_KEYS
    + r")\b[\"']?\s*[:=]\s*(?![|>\"'])(?P<value>\S.*)$"
)
_CONTINUED_QUERY_SECRET_RE = re.compile(r"(?i)[?&](?:" + _QUERY_SECRET_KEYS + r")=\s*$")
_OMITTED_CONTEXT_RE = re.compile(
    r"(?i)(?:"
    + _KEY_START
    + r"(?:(?:"
    + _SECRET_KEYS
    + r")|(?:set-)?cookie)\b[\"']?\s*[:=]|[?&](?:"
    + _QUERY_SECRET_KEYS
    + r")=)"
)
_OMITTED_QUOTED_START_RE = re.compile(
    r"(?i)(?:"
    + _KEY_START
    + r"(?:(?:"
    + _SECRET_KEYS
    + r")|(?:set-)?cookie)\b[\"']?\s*[:=]\s*|--(?:"
    + _FLAG_SECRET_KEYS
    + r")\s+)(?P<quote>[\"'])"
)

# An Authorization value, whatever the scheme. The key/value rule cannot reach
# it: for "Authorization: Basic dXNlcjpwdw==" the value it captures is "Basic",
# leaving the credential behind it. Same for a Cookie, which for Unsloth is the
# UI session that gates these very endpoints.
_SCHEMES = ("bearer", "basic", "digest", "token", "apikey", "aws4-hmac-sha256")
_PARAMETERIZED_AUTH_SCHEMES = frozenset({"digest", "aws4-hmac-sha256"})
# A scheme word only introduces a credential when an Authorization header put it
# there. Bare "digest sha256:..." and "token hf_..." are ordinary log content,
# and firing on the word alone blanked the digest a user came here to read.
# The credential stops at a quote or a structural delimiter, not at the next
# space: \S+ swallowed the closing quote and every field behind it, so a compact
# header dict came back as {"Authorization":"Bearer <redacted> with the request
# id and status gone with it.
_CREDENTIAL = r"[^\s\"',}\]]+"
_QUOTED_AUTH_RE = re.compile(
    r"(?i)(?P<key>(?:proxy-)?authorization)(?P<sep>[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])(?P<val>" + _QUOTED_VALUE + r")(?P=quote)"
)
_UNQUOTED_AUTH_RE = re.compile(
    r"(?i)(?P<key>(?:proxy-)?authorization)(?P<sep>[\"']?\s*[:=]\s*)"
    r"(?P<val>[^\"'\s}\]][^\"'\r\n}\]]*)"
)
_AUTH_ADJACENT_FIELD_RE = re.compile(r"[,;]\s*[A-Za-z_][\w-]*\s*[:=]")
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

# Exact secret keys mask every non-empty value. Preserve only explicit null
# sentinels, which communicate that no credential was configured.
_NON_SECRET_SENTINELS = frozenset({"none", "null"})
_SEMICOLON_FIELD_BOUNDARY_RE = re.compile(r";(?=\s*[A-Za-z_][A-Za-z0-9_.-]*\s*[:=])")
_YAML_BLOCK_MARKER_RE = re.compile(r"[|>](?:[1-9][+-]?|[+-][1-9]?|[+-])?")
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----.*?"
    r"-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----",
    re.IGNORECASE | re.DOTALL,
)
_PRIVATE_KEY_BEGIN_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----",
    re.IGNORECASE,
)
_PRIVATE_KEY_END_RE = re.compile(
    r"-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----",
    re.IGNORECASE,
)


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
    value = match.group("val")
    boundary = _SEMICOLON_FIELD_BOUNDARY_RE.search(value)
    tail = ""
    if boundary is not None:
        value, tail = value[: boundary.start()], value[boundary.start() :]
    if value.lower() in _NON_SECRET_SENTINELS or _YAML_BLOCK_MARKER_RE.fullmatch(value):
        return match.group(0)
    # Quoting puts the scheme inside the value ('authorization': 'Basic abc').
    # Step over it rather than abandon the match: the rest is still the
    # credential, and blanking the scheme reads as if the header were the secret.
    scheme, sep, rest = value.partition(" ")
    if scheme.lower() in _SCHEMES:
        if not sep or not rest.strip():
            return match.group(0)
        return f"{match.group('key')}{match.group('sep')}{scheme}{sep}{REDACTED}{tail}"
    return f"{match.group('key')}{match.group('sep')}{REDACTED}{tail}"


def _redact_env_assignment(match: re.Match[str]) -> str:
    """Mask Studio-recognized secret env vars without consuming a command."""
    key = match.group("key")
    if not _is_shell_secret_env_name(key):
        return match.group(0)
    quote = match.group("quote") or ""
    return f"{key}{match.group('sep')}{quote}{REDACTED}{quote}"


def _redact_structured_env_kv(match: re.Match[str]) -> str:
    """Mask env-style credentials in JSON, Python mappings, and YAML."""
    key = match.group("quoted_key") or match.group("plain_key")
    # Structured records also contain ordinary application fields. Limit the
    # broad shared env classifier to conventional uppercase env names, while
    # still matching every explicitly inventoried name case-insensitively.
    upper = key.upper()
    if key != upper and upper not in SECRET_ENV_NAMES:
        return match.group(0)
    if not _is_shell_secret_env_name(key):
        return match.group(0)
    quote = match.group("quote") or ""
    value_bytes = match.groupdict().get("value_bytes") or ""
    tail = ""
    value = match.group("val")
    if value is not None:
        boundary = _SEMICOLON_FIELD_BOUNDARY_RE.search(value)
        if boundary is not None:
            tail = value[boundary.start() :]
    return (
        f"{match.group('key_text')}{match.group('sep')}"
        f"{value_bytes}{quote}{REDACTED}{quote}{tail}"
    )


def _redact_container_values(text: str) -> str:
    """Mask balanced list, mapping, and tuple values for exact secret keys."""
    pairs = {"[": "]", "{": "}", "(": ")"}
    cursor = 0
    chunks: list[str] = []
    while match := _CONTAINER_KV_START_RE.search(text, cursor):
        start = match.start()
        index = match.end()
        stack = [pairs[match.group("open")]]
        quote: str | None = None
        escaped = False
        while index < len(text) and stack:
            char = text[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char in pairs:
                stack.append(pairs[char])
            elif char == stack[-1]:
                stack.pop()
            index += 1

        if stack:
            line_end = text.find("\n", match.end())
            index = len(text) if line_end == -1 else line_end
        chunks.append(text[cursor:start])
        chunks.append(f"{match.group('key')}{match.group('sep')}{REDACTED}")
        cursor = index
    chunks.append(text[cursor:])
    return "".join(chunks)


def _is_shell_secret_env_name(name: str) -> bool:
    """Apply the env classifier without treating model token metadata as env."""
    if not is_secret_env_name(name):
        return False
    upper = name.upper()
    if name != upper and name.lower() in {"password", "passwd", "pwd", "secret"}:
        return False
    if name == upper or upper in SECRET_ENV_NAMES:
        return True
    if any(upper.startswith(prefix) for prefix in SECRET_ENV_PREFIXES):
        return True
    strong_markers = (
        "API_KEY",
        "APIKEY",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "PASSPHRASE",
        "CREDENTIAL",
        "PRIVATE_KEY",
        "AUTH",
        "CONNSTR",
        "CONNECTIONSTRING",
    )
    return any(marker in upper for marker in strong_markers)


def _redact_quoted_kv(match: re.Match[str]) -> str:
    value = match.group("val")
    scheme, sep, rest = value.partition(" ")
    masked = (
        f"{scheme}{sep}{REDACTED}"
        if scheme.lower() in _SCHEMES and sep and rest.strip()
        else REDACTED
    )
    quote = match.group("quote")
    value_bytes = match.groupdict().get("value_bytes") or ""
    return f"{match.group('key')}{match.group('sep')}{value_bytes}{quote}{masked}{quote}"


def _redact_unterminated_quoted_kv(match: re.Match[str]) -> str:
    quote = match.group("quote")
    value_bytes = match.groupdict().get("value_bytes") or ""
    return f"{match.group('key')}{match.group('sep')}{value_bytes}{quote}{REDACTED}"


def _redact_escaped_quoted_kv(match: re.Match[str]) -> str:
    rest = match.group("rest")
    quote = match.group("quote")
    is_cookie = match.group("key").lower().endswith("cookie")
    for index, char in enumerate(rest):
        if char != quote:
            continue
        slash_count = 0
        cursor = index - 1
        while cursor >= 0 and rest[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        # One serialization layer turns 1, 5, 9, ... backslashes before a
        # quote into an unescaped nested quote. Runs of 3, 7, 11, ... encode a
        # quote inside the credential and must remain part of the masked value.
        if slash_count % 4 == 1:
            value = rest[: index - 1]
            if is_cookie and not _COOKIE_PAIR_RE.match(value.strip()):
                return match.group(0)
            return (
                f"{match.group('key')}{match.group('sep')}{REDACTED}" f"\\{quote}{rest[index + 1:]}"
            )
    # An exact secret key with an unterminated serialized value is still
    # sensitive. Mask the remainder rather than leaking it for malformed logs.
    if is_cookie and not _COOKIE_PAIR_RE.match(rest.strip()):
        return match.group(0)
    return f"{match.group('key')}{match.group('sep')}{REDACTED}"


def _redact_quoted_header_pair(match: re.Match[str]) -> str:
    value = match.group("val")
    if match.group("key").lower().endswith("cookie"):
        if not _COOKIE_PAIR_RE.match(value.strip()):
            return match.group(0)
        masked = REDACTED
    else:
        scheme, separator, credential = value.partition(" ")
        masked = (
            f"{scheme}{separator}{REDACTED}"
            if scheme.lower() in _SCHEMES and separator and credential.strip()
            else REDACTED
        )
    key_quote = match.group("key_quote")
    value_quote = match.group("quote")
    return (
        f"{match.group('key_bytes') or ''}{key_quote}{match.group('key')}{key_quote}"
        f"{match.group('sep')}{match.group('value_bytes') or ''}"
        f"{value_quote}{masked}{value_quote}"
    )


def _redact_shaped(match: re.Match[str]) -> str:
    if not _looks_like_credential(match.group(3)):
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}{REDACTED}"


def _redact_auth_assignment(match: re.Match[str]) -> str:
    """Mask every exact Authorization value, preserving only known scheme names."""
    value = match.group("val").strip()
    scheme, sep, rest = value.partition(" ")
    if not sep and scheme.lower() in _SCHEMES:
        return match.group(0)
    quote = match.groupdict().get("quote") or ""
    tail = ""
    if not quote and sep and scheme.lower() not in _PARAMETERIZED_AUTH_SCHEMES:
        boundary = _AUTH_ADJACENT_FIELD_RE.search(rest)
        if boundary is not None:
            rest, tail = rest[: boundary.start()].rstrip(), rest[boundary.start() :]
    masked = f"{scheme}{sep}{REDACTED}" if scheme.lower() in _SCHEMES and rest.strip() else REDACTED
    return f"{match.group('key')}{match.group('sep')}{quote}{masked}{quote}{tail}"


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
    text = _PRIVATE_KEY_BLOCK_RE.sub(REDACTED, text)
    # Nothing anchored below survives an escape between a key and its value, so
    # strip first, guarded by one introducer scan: ordinary content is untouched.
    if _ANSI_INTRODUCER_RE.search(text):
        text = _ANSI_RE.sub("", text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    text = _redact_container_values(text)
    # Use the same env-name classifier as the tool sandbox. Shell assignments
    # end at whitespace, so later command arguments remain useful diagnostics.
    text = _ENV_ASSIGNMENT_RE.sub(_redact_env_assignment, text)
    text = _STRUCTURED_ENV_KV_RE.sub(_redact_structured_env_kv, text)
    # Exact Authorization assignments are unambiguous even when the scheme is
    # uncommon (Negotiate, AWS SigV4, or a provider-specific extension).
    text = _QUOTED_AUTH_RE.sub(_redact_auth_assignment, text)
    text = _UNQUOTED_AUTH_RE.sub(_redact_auth_assignment, text)
    text = _SCHEME_RE.sub(_redact_shaped, text)
    text = _COOKIE_RE.sub(_redact_cookie, text)
    text = _ESCAPED_QUOTED_KV_RE.sub(_redact_escaped_quoted_kv, text)
    text = _QUOTED_HEADER_PAIR_RE.sub(_redact_quoted_header_pair, text)
    text = _QUOTED_KV_RE.sub(_redact_quoted_kv, text)
    text = _UNTERMINATED_QUOTED_KV_RE.sub(_redact_unterminated_quoted_kv, text)
    text = _PLAIN_SCALAR_KV_RE.sub(_redact_kv, text)
    text = _KV_RE.sub(_redact_kv, text)
    text = _QUOTED_FLAG_RE.sub(_redact_quoted_kv, text)
    text = _UNTERMINATED_QUOTED_FLAG_RE.sub(_redact_unterminated_quoted_kv, text)
    text = _FLAG_RE.sub(_redact_kv, text)
    try:
        from utils.native_path_leases import redact_native_paths
        text = redact_native_paths(text)
    except Exception:
        pass
    return text


class StreamingLogRedactor:
    """Redact independent records while preserving key/value context."""

    def __init__(self) -> None:
        self._plain_key_indent: int | None = None
        self._plain_has_value = False
        self._plain_explicit_continuation = False
        self._block_key_indent: int | None = None
        self._block_value_indent: int | None = None
        self._cookie_key_indent: int | None = None
        self._cookie_has_value = False
        self._quoted_secret: str | None = None
        self._private_key_block = False

    @staticmethod
    def _masked_record(text: str) -> str:
        newline = (
            "\r\n"
            if text.endswith("\r\n")
            else "\n"
            if text.endswith("\n")
            else "\r"
            if text.endswith("\r")
            else ""
        )
        indent = re.match(r"[ \t]*", text).group(0)
        return f"{indent}{REDACTED}{newline}"

    @staticmethod
    def _has_unescaped_quote(text: str, quote: str) -> bool:
        escaped = False
        index = 0
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif quote == "'" and char == "'" and text[index : index + 2] == "''":
                index += 2
                continue
            elif char == quote:
                return True
            index += 1
        return False

    @staticmethod
    def omitted_record_chunk_has_sensitive_context(text: str) -> bool:
        if _OMITTED_CONTEXT_RE.search(text) is not None:
            return True
        return any(
            _is_shell_secret_env_name(match.group("key"))
            for match in re.finditer(
                r"(?<![A-Za-z0-9_])(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=",
                text,
            )
        )

    @staticmethod
    def omitted_record_private_key_state(text: str, active: bool = False) -> bool:
        """Track private-key armor while an oversized record is discarded."""
        events = [(match.start(), True) for match in _PRIVATE_KEY_BEGIN_RE.finditer(text)] + [
            (match.start(), False) for match in _PRIVATE_KEY_END_RE.finditer(text)
        ]
        for _, is_begin in sorted(events):
            active = is_begin
        return active

    @classmethod
    def omitted_record_continuation_kind(
        cls,
        text: str,
        previous: str | None = None,
        sensitive_context: bool = False,
    ) -> str | None:
        """Track whether an omitted physical record opens a later value."""
        matches = list(_OMITTED_CONTEXT_RE.finditer(text))
        if matches:
            suffix = text[matches[-1].end() :].rstrip("\r\n")
            stripped = suffix.strip()
            if not stripped:
                return "plain"
            before_comment = stripped.split("#", 1)[0].rstrip()
            if _YAML_BLOCK_MARKER_RE.fullmatch(before_comment):
                return "block"
            return "plain" if cls._ends_with_unescaped_backslash(suffix.rstrip()) else None

        if previous == "block":
            stripped = text.strip()
            return "block" if not stripped or stripped.startswith("#") else None
        if previous == "plain":
            if not text.strip():
                return "plain"
            return "plain" if cls._ends_with_unescaped_backslash(text.rstrip()) else None
        if sensitive_context and cls._ends_with_unescaped_backslash(text.rstrip()):
            return "plain"
        return None

    def mark_omitted_sensitive_record(
        self,
        quoted_secret: str | None = None,
        continuation_kind: str | None = None,
    ) -> None:
        """Conservatively mask the continuation after a discarded sensitive record."""
        if quoted_secret is not None:
            self._quoted_secret = quoted_secret
            return
        if continuation_kind == "block":
            self._block_key_indent = 0
            self._block_value_indent = None
            return
        self._plain_key_indent = 0
        self._plain_has_value = False
        self._plain_explicit_continuation = False

    def mark_omitted_private_key_block(self) -> None:
        self._private_key_block = True

    @staticmethod
    def omitted_record_quote_state(
        text: str,
        quote: str | None,
        escaped: bool = False,
    ) -> tuple[str | None, bool]:
        """Track only quote structure while an oversized record is discarded."""
        index = 0
        while index < len(text):
            if quote is None:
                start = _OMITTED_QUOTED_START_RE.search(text, index)
                if start is None:
                    return None, False
                quote = start.group("quote")
                escaped = False
                index = start.end()
                continue

            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif quote == "'" and text[index : index + 2] == "''":
                index += 2
                continue
            elif char == quote:
                quote = None
            index += 1
        return quote, escaped

    @staticmethod
    def _ends_with_unescaped_backslash(text: str) -> bool:
        return (len(text) - len(text.rstrip("\\"))) % 2 == 1

    @staticmethod
    def _context_view(text: str) -> str:
        """Start at the last credential assignment, preserving YAML indentation.

        Requiring the assignment separator keeps words such as ``secret`` in a
        credential value from being mistaken for a second key marker.
        """
        matches = list(_OMITTED_CONTEXT_RE.finditer(text))
        if not matches:
            return text
        start = matches[-1].start()
        prefix = text[:start]
        if not prefix.strip():
            return prefix + text[start:]
        sequence = re.fullmatch(r"(?P<indent>[ \t]*)-\s+", prefix)
        indent = re.match(r"[ \t]*", prefix).group(0)
        if sequence:
            whitespace = sequence.group("indent")
            indent = whitespace + " " * (len(prefix) - len(whitespace))
        return indent + text[start:]

    def redact_record(self, text: str) -> str:
        physical = text.rstrip("\r\n")
        physical_context = self._context_view(physical)
        if self._private_key_block:
            if _PRIVATE_KEY_END_RE.search(physical):
                self._private_key_block = False
            return self._masked_record(text)
        begin = _PRIVATE_KEY_BEGIN_RE.search(physical)
        if begin:
            self._private_key_block = _PRIVATE_KEY_END_RE.search(physical, begin.end()) is None
            return self._masked_record(text)
        if self._quoted_secret is not None:
            quote = self._quoted_secret
            if self._has_unescaped_quote(physical, quote):
                self._quoted_secret = None
            return self._masked_record(text)

        redacted = redact_log_text(text)
        redacted_context = self._context_view(redacted.rstrip("\r\n"))
        if self._block_key_indent is not None:
            if not redacted.strip():
                return redacted
            indent = len(re.match(r"[ \t]*", redacted).group(0))
            if self._block_value_indent is None and indent > self._block_key_indent:
                self._block_value_indent = indent
            if self._block_value_indent is not None and indent >= self._block_value_indent:
                return self._masked_record(redacted)
            self._block_key_indent = None
            self._block_value_indent = None

        if self._plain_key_indent is not None:
            if not redacted.strip():
                return redacted
            indent = len(re.match(r"[ \t]*", redacted).group(0))
            if indent > self._plain_key_indent:
                self._plain_has_value = True
                return self._masked_record(redacted)
            if indent == self._plain_key_indent and re.match(r"-\s", physical.lstrip()):
                self._plain_has_value = True
                return self._masked_record(redacted)
            if self._plain_explicit_continuation:
                self._plain_explicit_continuation = self._ends_with_unescaped_backslash(physical)
                if not self._plain_explicit_continuation:
                    self._plain_key_indent = None
                    self._plain_has_value = False
                return self._masked_record(redacted)
            if not self._plain_has_value:
                self._plain_explicit_continuation = self._ends_with_unescaped_backslash(physical)
                if self._plain_explicit_continuation:
                    self._plain_has_value = True
                else:
                    self._plain_key_indent = None
                return self._masked_record(redacted)
            self._plain_key_indent = None
            self._plain_has_value = False
            self._plain_explicit_continuation = False

        if self._cookie_key_indent is not None:
            if not redacted.strip():
                return redacted
            indent = len(re.match(r"[ \t]*", redacted).group(0))
            if self._cookie_has_value and indent <= self._cookie_key_indent:
                self._cookie_key_indent = None
                self._cookie_has_value = False
            elif self._cookie_has_value or _COOKIE_PAIR_RE.match(redacted.lstrip().lstrip("'\"")):
                self._cookie_has_value = True
                return self._masked_record(redacted)
            else:
                self._cookie_key_indent = None

        quoted_assignment = _UNTERMINATED_QUOTED_SECRET_RE.search(physical_context)
        quoted_flag = _UNTERMINATED_QUOTED_FLAG_RE.search(physical_context)
        if quoted_assignment:
            self._quoted_secret = '"' if quoted_assignment.group("double") else "'"
            return self._masked_record(redacted)
        if quoted_flag:
            self._quoted_secret = quoted_flag.group("quote")
            return self._masked_record(redacted)

        inline_plain = _INLINE_PLAIN_SECRET_RE.search(physical_context)
        if inline_plain and (
            REDACTED in redacted or inline_plain.group("value").strip().lower() in _SCHEMES
        ):
            self._plain_key_indent = len(inline_plain.group("indent"))
            self._plain_has_value = True
            self._plain_explicit_continuation = self._ends_with_unescaped_backslash(
                physical_context
            )
            return redacted

        if _CONTINUED_QUERY_SECRET_RE.search(physical_context):
            self._plain_key_indent = len(re.match(r"[ \t]*", physical_context).group(0))
            self._plain_has_value = False
            self._plain_explicit_continuation = False
            return redacted

        inline_flag = _FLAG_RE.search(physical_context)
        if (
            inline_flag
            and REDACTED in redacted
            and self._ends_with_unescaped_backslash(inline_flag.group("val"))
        ):
            self._plain_key_indent = len(re.match(r"[ \t]*", physical_context).group(0))
            self._plain_has_value = True
            self._plain_explicit_continuation = True
            return redacted

        continued = _CONTINUED_SECRET_RE.search(redacted_context)
        if continued:
            if continued.group("block"):
                self._block_key_indent = len(continued.group("indent"))
                explicit = next(
                    (char for char in continued.group("block") if char.isdigit()),
                    None,
                )
                self._block_value_indent = (
                    self._block_key_indent + int(explicit) if explicit else None
                )
            else:
                self._plain_key_indent = len(continued.group("indent"))
                self._plain_has_value = False
                self._plain_explicit_continuation = False
            return redacted

        continued_env = _CONTINUED_ENV_RE.search(redacted_context)
        if continued_env and _is_shell_secret_env_name(continued_env.group("key")):
            self._plain_key_indent = len(continued_env.group("indent"))
            self._plain_has_value = False
            self._plain_explicit_continuation = False
            return redacted

        cookie = _CONTINUED_COOKIE_RE.search(redacted_context)
        cookie_pair = _CONTINUED_COOKIE_PAIR_RE.search(redacted_context)
        if cookie or cookie_pair:
            self._cookie_key_indent = len((cookie or cookie_pair).group("indent"))
            self._cookie_has_value = False
            return redacted

        inline_cookie = _INLINE_COOKIE_RE.search(physical_context)
        if inline_cookie and _COOKIE_PAIR_RE.match(inline_cookie.group("value").lstrip("'\"")):
            self._cookie_key_indent = len(inline_cookie.group("indent"))
            self._cookie_has_value = True
        return redacted
