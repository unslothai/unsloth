# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Credentials must not reach the log viewer, and ordinary log content must
survive untouched. The negative cases carry the weight here: over-redaction
hides the failure the user opened the log to read."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.log_redaction import redact_log_text

_SLACK_SHAPED = "xox" + "b-" + "1234567890" + "-ABCDEFGHIJKLMNOP"

# (line, the substring that must be gone)
SECRETS = [
    (
        "Downloading with token hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
    ),
    (
        'GET /v1/chat -H "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg"',
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg",
    ),
    (
        "llama-server --api-key sk-proj-AbCdEf0123456789AbCdEf --port 8080",
        "sk-proj-AbCdEf0123456789AbCdEf",
    ),
    ("HF_TOKEN=hf_zzzzzzzzzzzzzzzzzzzzzzzzzzz", "hf_zzzzzzzzzzzzzzzzzzzzzzzzzzz"),
    ('{"event":"auth","api_key":"abcdef123456","model":"gpt-4o"}', "abcdef123456"),
    ("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    (
        "https://cdn.example.com/m.gguf?X-Amz-Signature=deadbeef0123456789&X-Amz-Expires=900",
        "deadbeef0123456789",
    ),
    (
        "git clone https://dan:ghp_ABCDEFGHIJKLMNOPQRST0123@github.com/x/y",
        "ghp_ABCDEFGHIJKLMNOPQRST0123",
    ),
    ("password: hunter2hunter2", "hunter2hunter2"),
    # "_" is a word character, so a \b before the key name never fires inside an
    # env-style name. Every one of these survived in the clear before.
    ("OPENAI_API_KEY=opaquevalue123456", "opaquevalue123456"),
    ("TOGETHER_API_KEY=abc123def456ghi789", "abc123def456ghi789"),
    (
        "WANDB_API_KEY=0123456789abcdef0123456789abcdef01234567",
        "0123456789abcdef0123456789abcdef01234567",
    ),
    ("DATABASE_PASSWORD=hunter2hunter2", "hunter2hunter2"),
    ("training config: wandb_token='local-9f8e7d6c5b4a3210'", "local-9f8e7d6c5b4a3210"),
    # The key/value rule captures the scheme word as the "value", so the
    # credential after it was never looked at.
    ("Authorization: Basic dXNlcm5hbWU6c3VwZXJzZWNyZXQ=", "dXNlcm5hbWU6c3VwZXJzZWNyZXQ="),
    ("headers={'authorization': 'Basic dXNlcjpwdw=='}", "dXNlcjpwdw=="),
    # Studio's UI session cookie is the credential that gates these very
    # endpoints.
    ("Cookie: unsloth_session=8f3c9d1ab77e4f0a9c2b3d4e", "8f3c9d1ab77e4f0a9c2b3d4e"),
    ("set-cookie: refresh=8f3c9d1ab77e4f0a9c2b; HttpOnly", "8f3c9d1ab77e4f0a9c2b"),
    ("CI token glpat-ABCDEFGHIJKLMNOPQRST", "glpat-ABCDEFGHIJKLMNOPQRST"),
    # Assembled rather than written out: a literal in the shape of a real Slack
    # token, even an invented one, trips GitHub push protection.
    ("posting with " + _SLACK_SHAPED, _SLACK_SHAPED),
    ("refreshed ya29.a0ARrdaM9xQZ1lKjHgFdSaQwErTyUiOp", "ya29.a0ARrdaM9xQZ1lKjHgFdSaQwErTyUiOp"),
    # A numeric password is still a password, whatever the general rule says
    # about numbers being counts and ids.
    ("password=1234567890123", "1234567890123"),
    # A quoted value that contains spaces. A passphrase is the everyday shape
    # here, and a value that stopped at the first space left every word but the
    # first in the clear while still printing <redacted>, which reads as safe.
    (
        '{"event":"login_failed","password":"correct horse battery staple"}',
        "horse battery staple",
    ),
    ("password='correct horse battery staple'", "horse battery staple"),
    # The flag rule's value class rejected a leading quote outright, so this
    # whole line survived untouched.
    ('llama-server --api-key "abcdef ghijklmnop" --port 8080', "abcdef ghijklmnop"),
]

# Real log lines. Each one must come back byte for byte.
KEEP = [
    "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    "blk.31.attn_q.weight  q4_K  [ 3072,  3072 ]",
    "/home/dan/.unsloth/studio/cache/models/models--unsloth--gemma-3-4b-it/snapshots/9a2f1c8b7e6d5c4b3a2918f7e6d5c4b3a2918f7e",
    "sha256:4f3c9a1b2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c",
    "n_tokens = 4096",
    "token_id=128009",
    "slot 0 released, 512 tokens in cache",
    "revision=a1b2c3d4e5f6",
    "CUDA error: out of memory (device 0, 23.6 GiB free)",
    "| Traceback (most recent call last):",
    "|   RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same",
    "Bearer",
    '  File "/opt/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518 in _call_impl',
    "llama-server --port 8080 --n-gpu-layers 99 --ctx-size 32768",
    '{"timestamp":"2026-08-13T09:00:00Z","level":"error","event":"llama_start_failed"}',
    # The words a credential rule is tempted by, in the places Studio actually
    # writes them. Blanking any of these hides the failure being diagnosed.
    "provider rejected the request: Bearer credentials expired",
    "Authorization header missing, expected Bearer authentication",
    "manifest digest sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "tokenizer: eos_token = <|eot_id|>, bos_token = <|begin_of_text|>",
    "pad_token_id=128004 set from config",
    "note: cookie support is disabled in this webview",
    "reading secret_sauce_path from the recipe",
    "hint: password authentication is not configured for this endpoint",
    "downloaded checkpoint-sk-9f8a7b6c5d4e3f2a1b0c9d8e7f.safetensors",
    "i18n: falling back from sk-SK to en",
    # "key" in an object storage URL names the object, so it stays readable.
    "https://cdn-lfs.hf.co/repos/ab/cd/model.gguf?download=true&key=publicfilename",
    "provider config: api_key = None",
]


@pytest.mark.parametrize("line,secret", SECRETS, ids = [s[1][:18] for s in SECRETS])
def test_a_credential_never_survives(line, secret):
    out = redact_log_text(line)
    assert secret not in out
    assert "<redacted>" in out


@pytest.mark.parametrize("line", KEEP, ids = [k[:28] for k in KEEP])
def test_ordinary_log_content_is_untouched(line):
    assert redact_log_text(line) == line


@pytest.mark.parametrize("line", [s[0] for s in SECRETS] + KEEP)
def test_redaction_is_idempotent(line):
    once = redact_log_text(line)
    assert redact_log_text(once) == once


QUOTED = [
    # (line, what the whole line must come back as)
    ('password="correct horse battery staple"', 'password="<redacted>"'),
    ("password='correct horse battery staple'", "password='<redacted>'"),
    ('llama-server --api-key "abcdef ghijklmnop"', 'llama-server --api-key "<redacted>"'),
    # The value ends at its own closing quote, so the fields after it survive.
    (
        '{"password": "correct horse battery staple", "model": "gpt-4o"}',
        '{"password": "<redacted>", "model": "gpt-4o"}',
    ),
    # An escaped quote inside the value does not end it early.
    ('password="corr\\"ect horse staple"', 'password="<redacted>"'),
    # Quoting puts the scheme word inside the value; it stays readable, and the
    # credential behind it does not.
    ('password: "Basic dXNlcjpwdw=="', 'password: "Basic <redacted>"'),
]


@pytest.mark.parametrize("line,expected", QUOTED, ids = [q[0][:24] for q in QUOTED])
def test_a_quoted_credential_is_masked_whole(line, expected):
    """The value patterns used to stop at whitespace, so a quoted credential
    containing spaces was masked only up to its first space and the rest of the
    secret was printed next to the <redacted> marker."""
    assert redact_log_text(line) == expected


def test_an_unterminated_quote_does_not_mask_the_next_line():
    """\\n is outside the quoted value class, so a writer that opened a quote
    and never closed it cannot blank the log lines that follow it."""
    text = 'password="correct horse battery\nloading model from /models/x.gguf\n'
    out = redact_log_text(text)
    assert "loading model from /models/x.gguf" in out
    assert "correct horse battery" not in out


def test_an_empty_line_is_safe():
    assert redact_log_text("") == ""


COOKIE_PROSE = [
    "Cookie: disabled by the browser",
    "Cookie: not sent because the origin is cross-site",
    "Set-Cookie: cleared on logout",
]


@pytest.mark.parametrize("line", COOKIE_PROSE)
def test_a_cookie_diagnosis_is_not_mistaken_for_a_cookie(line):
    """The cookie rule takes the whole rest of the line, so prose longer than
    the token-length shortcut was being masked. A line explaining why a cookie
    was not sent is the diagnosis, not the secret."""
    assert redact_log_text(line) == line


@pytest.mark.parametrize(
    "line,secret",
    [
        ("Cookie: unsloth_ui_session=abc123def456xyz", "abc123def456xyz"),
        ("Set-Cookie: session=abc123def456xyz; HttpOnly", "abc123def456xyz"),
    ],
)
def test_a_real_cookie_pair_is_still_masked(line, secret):
    assert secret not in redact_log_text(line)


# A colorized writer puts an escape sequence between the key and its value.
# Every rule here is anchored on a word boundary or a lookbehind, and the "m"
# that ends "\x1b[36m" is a word character, so the anchor stopped matching and
# the credential went out in the clear -- and the viewer's pane strips the
# escapes before rendering, so the reader saw a clean, unmasked token.
ANSI_SECRETS = [
    # structlog's ConsoleRenderer, verbatim: colors are on by default whether or
    # not the sink is a terminal, so this is what lands in the session log.
    (
        "\x1b[36mapi_key\x1b[0m=\x1b[35msk_live_abcdef123456\x1b[0m",
        "sk_live_abcdef123456",
    ),
    (
        "\x1b[36mhf_token\x1b[0m=\x1b[35mhf_AbCdEfGhIjKlMnOpQrStUvWxYz012345\x1b[0m",
        "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
    ),
    (
        "\x1b[31mAuthorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg\x1b[0m",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg",
    ),
    # An escape INSIDE the token, and the 8-bit CSI form.
    ("hf_\x1b[31mAbCdEfGhIjKlMnOpQrStUvWxYz012345\x1b[0m", "AbCdEfGhIjKlMnOpQrStUvWxYz012345"),
    ("\x9b36mapi_key\x9b0m=abcdef123456", "abcdef123456"),
]


@pytest.mark.parametrize("line,secret", ANSI_SECRETS, ids = ["kv", "hf", "auth", "mid", "c1"])
def test_a_colorized_credential_is_still_masked(line, secret):
    assert secret not in redact_log_text(line)


def test_ordinary_colorized_content_keeps_its_text():
    """Stripping the control sequences must not eat the log line with them."""
    out = redact_log_text("\x1b[32mmodel loaded\x1b[0m from /models/qwen3-4b.gguf")
    assert out == "model loaded from /models/qwen3-4b.gguf"


def test_a_hyperlink_escape_does_not_swallow_the_line():
    """OSC is matched before the two-character Fe class, which covers "]" and
    would otherwise consume only the introducer and leave the payload behind."""
    out = redact_log_text("open \x1b]8;;https://example.com\x07docs\x1b]8;;\x07 for help")
    assert out == "open docs for help"
