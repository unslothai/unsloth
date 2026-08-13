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

# Key names whose VALUE is a secret. "token" alone is absent on purpose, so
# n_tokens = 4096 and token_id=128009 survive.
_SECRET_KEYS = (
    "authorization|x-api-key|api[-_]?key|apikey|hf[-_]?token|access[-_]?token|"
    "refresh[-_]?token|auth[-_]?token|bearer[-_]?token|client[-_]?secret|"
    "aws_secret_access_key|password|passwd|secret"
)

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Hugging Face
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "hf_" + REDACTED),
    # OpenAI and the sk- style keys (project, Anthropic, OpenRouter)
    (re.compile(r"\bsk-(?:proj-|ant-api\d{2}-|or-v1-)?[A-Za-z0-9_-]{16,}"), "sk-" + REDACTED),
    # Other vendor prefixes
    (
        re.compile(r"\b(?:gsk_|xai-|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{16,}"),
        REDACTED,
    ),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), REDACTED),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTED),
    # JWTs, including the desktop access token
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"), REDACTED),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]{8,}"), "Bearer " + REDACTED),
    # user:password@host in a URL
    (re.compile(r"://[^/\s:@]+:[^/\s@]+@"), "://" + REDACTED + "@"),
    # Presigned URL parameters
    (
        re.compile(
            r"(?i)([?&](?:token|api_key|key|sig|signature|"
            r"x-amz-signature|x-amz-credential|access_token)=)[^&\s\"']+"
        ),
        r"\1" + REDACTED,
    ),
)

# key = value / "key": "value" / --api-key value
_KV_RE = re.compile(
    r"(?i)\b(" + _SECRET_KEYS + r")\b([\"']?\s*[:=]\s*[\"']?)([^\"'\s,}\]]{6,})"
)
_FLAG_RE = re.compile(
    r"(?i)(--(?:" + _SECRET_KEYS + r"))(\s+)([^\s\"']{6,})"
)


def _redact_kv(match: re.Match[str]) -> str:
    value = match.group(3)
    # A numeric value is a count or an id, never a credential.
    if value.isdigit():
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}{REDACTED}"


def redact_log_text(text: str) -> str:
    """Mask credentials. Idempotent, and a no-op on ordinary log content."""
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    text = _KV_RE.sub(_redact_kv, text)
    text = _FLAG_RE.sub(_redact_kv, text)
    try:
        from utils.native_path_leases import redact_native_paths

        text = redact_native_paths(text)
    except Exception:
        pass
    return text
