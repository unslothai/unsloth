# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The literal gates in redact_log_text skip only rules that cannot match, so the output must equal running every rule."""

from __future__ import annotations

import random
import re

import pytest

from utils import log_redaction as lr


def _reference(text: str) -> str:
    """Every rule, ungated, in the same order as redact_log_text."""
    if not text:
        return text
    if lr._ANSI_INTRODUCER_RE.search(text):
        text = lr._strip_ansi(text)
    for pattern, replacement in lr._PATTERNS:
        text = pattern.sub(replacement, text)
    text = lr._AUTH_HEADER_RE.sub(lr._redact_shaped, text)
    text = lr._SCHEME_RE.sub(lr._redact_shaped, text)
    text = lr._COOKIE_RE.sub(lr._redact_cookie, text)
    text = lr._KV_RE.sub(lr._redact_kv, text)
    text = lr._FLAG_RE.sub(lr._redact_kv, text)
    try:
        from utils.native_path_leases import redact_native_paths
        text = redact_native_paths(text)
    except Exception:
        pass
    return text


_FRAGMENTS = [
    "hf_",
    "hf_oauth_",
    "sk-",
    "sk-proj-",
    "sk-ant-api03-",
    "sk-or-v1-",
    "gsk_",
    "xai-",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "glpat-",
    "xoxb-",
    "xoxp-",
    "ya29.",
    "AIza",
    "AKIA",
    "ASIA",
    "eyJ",
    ".",
    "://",
    "user:pw@",
    "@",
    "?",
    "&",
    "token=",
    "api_key=",
    "apikey=",
    "sig=",
    "signature=",
    "X-Amz-Signature=",
    "x-amz-credential=",
    "x-amz-security-token=",
    "access_token=",
    "Authorization",
    "authorization: ",
    "Proxy-Authorization",
    "Bearer ",
    "bearer ",
    "Basic ",
    "Digest ",
    "Token ",
    "Cookie: ",
    "set-cookie=",
    "api_key",
    "API-KEY",
    "x-api-key",
    "hf_token",
    "HF-TOKEN",
    "refresh-token",
    "client_secret",
    "aws_secret_access_key",
    "aws_session_token",
    "wandb_token",
    "hub_token",
    "secretAccessKey",
    "password",
    "passwd",
    "PASSWORD",
    "secret",
    "--hf-token",
    "--password",
    "--api_key",
    "--",
    "-",
    ":",
    "=",
    '"',
    "'",
    "\\",
    " ",
    "\t",
    ",",
    "}",
    "]",
    "{",
    "\n",
    "\r",
    # Characters IGNORECASE folds onto ASCII letters: lowercasing alone would miss these.
    "K",
    "ſ",
    "İ",
    "ı",
    "paſſword",
    "Key",
    "toKen",
    "é",
    "█",
    "\x1b[31m",
    "\x1b]0;t\x07",
    "\x9b1m",
    "\x1b",
    "0123456789",
    "ABCDEFGHIJKLMNOPQRST",
    "abcdefghijklmnopqrstuvwxyz0123",
    "AbCdEfGh12",
    "n_tokens = 4096",
    "token_id=128009",
    "eos_token_id",
    "checkpoint-sk-9f8a",
]
# One real-shaped secret per alternative of every gated rule: a gate missing a literal leaves its sample clear.
_SHAPED_SECRETS = [
    "hf_abcdefghijklmnopqrstuvwx",
    "hf_oauth_abcdefghijklmnopqrstuvwx",
    "sk-abcdefghijklmnop1234",
    "sk-proj-abcdefghijklmnop1234",
    *(
        prefix + "abcdefghijklmnop1234"
        for prefix in (
            "gsk_",
            "xai-",
            "ghp_",
            "gho_",
            "ghu_",
            "ghs_",
            "ghr_",
            "github_pat_",
            "glpat-",
            "ya29.",
        )
    ),
    *("xox" + c + "-abcdefghijklmnop1234" for c in "abpsr"),
    "AIzaabcdefghijklmnopqrstuvwxyz0123456",
    "AKIAABCDEFGHIJKLMNOP",
    "ASIAABCDEFGHIJKLMNOP",
    "eyJabcdefghijk.abcdefghijk.abcdef",
    "https://h/x?sig=abcdef",
]
_FRAGMENTS += _SHAPED_SECRETS
_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./=+~"


def _random_line(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(1, 14)):
        roll = rng.random()
        if roll < 0.6:
            parts.append(rng.choice(_FRAGMENTS))
        elif roll < 0.9:
            parts.append("".join(rng.choice(_ALNUM) for _ in range(rng.randint(1, 40))))
        else:
            parts.append(chr(rng.randint(1, 0x2FFF)))
    line = "".join(parts)
    if rng.random() < 0.3:
        line = "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in line)
    return line


@pytest.mark.parametrize("seed", range(4))
def test_gated_output_equals_every_rule(seed):
    rng = random.Random(seed)
    redacted = 0
    for _ in range(20_000):
        line = _random_line(rng)
        expected = _reference(line)
        assert lr.redact_log_text(line) == expected, repr(line)
        redacted += expected != line
    assert redacted > 2_000


@pytest.mark.parametrize(
    "line",
    [
        "paſſword=abcdef12",
        "api_Key=hunter22hunter",
        "██ HF_TOKEN=hf_abcdefghijklmnopqrstuvwx",
        "AUTHORIZATION: Bearer abcdef1234567890",
        "Cookie: session=abcdef123",
        "--PASSWORD hunter22",
        "https://u:p@host/x?X-Amz-Signature=abc",
        "line one\npassword = 'hunter22'\nCookie: a=b",
        "plain training step 12 loss=0.91",
        "",
    ],
)
def test_edge_lines_match_reference(line):
    assert lr.redact_log_text(line) == _reference(line)


def test_non_ascii_key_still_redacted():
    # U+017F folds to "s" under IGNORECASE but is not "s" after lower(), so the gate must ask the regex engine.
    assert lr.redact_log_text("paſſword=abcdef12") == "paſſword=" + lr.REDACTED


def test_every_secret_key_has_a_gate_word():
    for key in lr._SECRET_KEYS.split("|"):
        assert any(word in key for word in lr._SECRET_KEY_WORDS), key


def test_every_pattern_gate_literal_is_in_its_rule():
    assert len(lr._PATTERN_LITERALS) == len(lr._PATTERNS) == len(lr._PATTERN_GATES)
    for literals, (pattern, _) in zip(lr._PATTERN_LITERALS, lr._PATTERNS):
        if literals is None:
            continue
        for literal in literals:
            assert literal in pattern.pattern or literal.replace(".", r"\.") in pattern.pattern, (
                literal,
                pattern.pattern,
            )
            # A case-insensitive rule could match the literal in another case, which the plain substring gate misses.
            assert not pattern.flags & re.IGNORECASE or literal.upper() == literal.lower(), literal


@pytest.mark.parametrize("secret", _SHAPED_SECRETS)
def test_every_gated_alternative_is_redacted(secret):
    line = f"log {secret} end"
    assert lr.redact_log_text(line) == _reference(line) != line
