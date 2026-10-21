# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Credentials must not reach the log viewer, and ordinary log content must
survive untouched. The negative cases carry the weight here: over-redaction
hides the failure the user opened the log to read."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.log_redaction import REDACTED, redact_log_text

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
    # env-style name; all of these used to survive in the clear.
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
    # Unsloth's UI session cookie gates these very endpoints.
    ("Cookie: unsloth_session=8f3c9d1ab77e4f0a9c2b3d4e", "8f3c9d1ab77e4f0a9c2b3d4e"),
    ("set-cookie: refresh=8f3c9d1ab77e4f0a9c2b; HttpOnly", "8f3c9d1ab77e4f0a9c2b"),
    ("CI token glpat-ABCDEFGHIJKLMNOPQRST", "glpat-ABCDEFGHIJKLMNOPQRST"),
    # Assembled, not written out: even an invented Slack-shaped literal trips
    # GitHub push protection.
    ("posting with " + _SLACK_SHAPED, _SLACK_SHAPED),
    ("refreshed ya29.a0ARrdaM9xQZ1lKjHgFdSaQwErTyUiOp", "ya29.a0ARrdaM9xQZ1lKjHgFdSaQwErTyUiOp"),
    # A numeric password is still a password, unlike numbers elsewhere.
    ("password=1234567890123", "1234567890123"),
    # A quoted passphrase with spaces: stopping at the first space left every
    # word but the first in the clear while still printing <redacted>.
    (
        '{"event":"login_failed","password":"correct horse battery staple"}',
        "horse battery staple",
    ),
    ("password='correct horse battery staple'", "horse battery staple"),
    # The flag rule's value class rejected a leading quote, so this line
    # survived untouched.
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
    # Words a credential rule is tempted by, as Unsloth actually writes them.
    # Blanking any of these hides the failure being diagnosed.
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
    # Quoting puts the scheme inside the value; it stays, the credential goes.
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
        # Headers are normally logged as a dict, so the value opens with a
        # quote. Anchoring the pair test on the bare value never matched it.
        ('headers={"Cookie": "session=abc123def456xyz"}', "abc123def456xyz"),
        ("Cookie: 'session=abc123def456xyz'", "abc123def456xyz"),
    ],
)
def test_a_real_cookie_pair_is_still_masked(line, secret):
    assert secret not in redact_log_text(line)


# A credential that ran to the next space swallowed the closing quote and every
# field behind it, so the pane lost the status and request id it was opened for.
@pytest.mark.parametrize(
    "line,expected",
    [
        (
            '{"Authorization":"Bearer abcdef123456","x-request-id":"req-42"}',
            '{"Authorization":"Bearer <redacted>","x-request-id":"req-42"}',
        ),
        ("authorization: 'Basic dXNlcjpwdw=='", "authorization: 'Basic <redacted>'"),
        (
            'headers={"Cookie": "session=abc123def456xyz", "accept": "*/*"}',
            'headers={"Cookie": "<redacted>", "accept": "*/*"}',
        ),
    ],
)
def test_the_fields_after_a_masked_header_survive(line, expected):
    assert redact_log_text(line) == expected


# A colorized writer puts an escape between the key and its value. Every rule is
# anchored on a word boundary or lookbehind, and the "m" ending "\x1b[36m" is a
# word character, so the anchor stopped matching and the credential went out in
# the clear -- and the pane strips escapes, so the reader saw a clean token.
ANSI_SECRETS = [
    # structlog's ConsoleRenderer verbatim: colors default on even off-terminal,
    # so this is what lands in the session log.
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


def test_studio_s3_secret_key_spellings_are_masked():
    """models/training.py:60 takes secret_access_key, alias secretAccessKey.

    Neither reaches the bare "secret" alternative: its trailing \\b cannot fire
    before "_access" or "Access", and an AWS secret key has no prefix of its own
    for a shape rule to catch, so both spellings went out in the clear.
    """
    secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    for line in (
        f"secret_access_key={secret}",
        f'{{"secretAccessKey":"{secret}"}}',
        f"secret-access-key: {secret}",
        f"--secret-access-key {secret}",
        f"aws_secret_access_key={secret}",
    ):
        masked = redact_log_text(line)
        assert secret not in masked, line
        assert REDACTED in masked, line


def test_talking_about_the_s3_key_without_a_value_survives():
    line = "secret_access_key is required when use_iam_role is false"
    assert redact_log_text(line) == line


# The lazy alternation this file's subject replaced. Kept verbatim as the oracle:
# the change is a performance fix, so the contract is that _strip_ansi returns
# exactly what this returns, on every input.
_LAZY_ANSI_RE = re.compile(
    r"\x1b\][\s\S]*?(?:\x07|\x1b\\|\x9c)"
    r"|\x1b[P^_X][\s\S]*?(?:\x1b\\|\x9c)"
    r"|\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b[@-Z\\-_]"
    r"|\x9b[0-?]*[ -/]*[@-~]"
    r"|[\x9d\x90\x98\x9e\x9f][\s\S]*?(?:\x07|\x9c)"
)

_ANSI_SHAPES = [
    # Well formed, every introducer and every terminator.
    "\x1b[36m",
    "\x1b[0m",
    "\x1b[38;5;196m",
    "\x1b[?25l",
    "\x1b[2K",
    "\x9b0m",
    "\x1b]0;title\x07",
    "\x1b]0;title\x1b\\",
    "\x1b]0;title\x9c",
    "\x9d0;title\x9c",
    "\x1bPx\x1b\\",
    "\x1b^x\x9c",
    "\x1b_x\x1b\\",
    "\x1bXx\x9c",
    "\x90x\x07",
    "\x98x\x9c",
    "\x9ex\x07",
    "\x9fx\x9c",
    "\x1bM",
    "\x1b7",
    "\x1b(B",
    # Truncated: an introducer whose sequence never terminates.
    "\x1b]",
    "\x1b]cut",
    "\x1bP",
    "\x1bPcut",
    "\x1b^cut",
    "\x1b_cut",
    "\x1bXcut",
    "\x9d",
    "\x9dcut",
    "\x90cut",
    "\x98cut",
    "\x9ecut",
    "\x9fcut",
    "\x1b",
    "\x1b[",
    "\x1b[38;5",
    "\x9b",
    "\x9b38;5",
    "\x1b(",
    # Cut, then a well formed sequence later in the same record.
    "\x1b]cut\x1b]t\x07",
    "\x9dcut\x9dt\x07",
    "\x1bPcut\x1bPt\x1b\\",
    "\x1b]cut\x1b[36m",
    "\x1b]cut\x9b36m",
    "\x1b\x1b[0m",
    # Stray terminators with no introducer, and interleaved introducers.
    "\x07",
    "\x9c",
    "\x1b\\",
    "\x1b]\x9d\x07",
    "\x9d\x1b]\x07",
    "\x1b]\x1b]\x1b]\x07",
]


def test_the_strip_is_unchanged_by_the_rewrite():
    """The contract. The lazy alternation was replaced because it backtracked,
    not because its answers were wrong, so the walk has to agree with it
    everywhere: same alternatives, same order, same lazy shortest match, same
    fallthrough when a control string never terminates.

    A redactor is the wrong place to smuggle a behaviour change into a
    performance fix, and every way of "improving" the truncated cases that was
    tried here moved a leak rather than removing one: consuming an aborted body
    ate the separator out of "api_key<cut>=value", and dropping a lone escape
    welded "prefix" onto "api_key".
    """
    from utils.log_redaction import _strip_ansi

    lines = [
        "api_key=abcdef123456",
        "Authorization: Bearer abcdef123456",
        "Cookie: session=abcdef123456",
        "?token=abcdef123456&next=1",
        "--password hunter2secret",
        "INFO loading unsloth/Llama-3.2-1B revision 8f3a2b1c in 13ms",
        "n_tokens = 4096, token_id=128009",
    ]
    checked = 0
    for shape in _ANSI_SHAPES:
        for line in lines:
            middle = len(line) // 2
            for text in (
                shape + line,
                line + shape,
                shape + line + shape,
                line[:middle] + shape + line[middle:],
            ):
                checked += 1
                assert _strip_ansi(text) == _LAZY_ANSI_RE.sub("", text), text
    assert checked > 1000


def test_the_strip_is_unchanged_on_random_records():
    """The shapes above are the ones someone thought of. This covers the ones
    nobody did, which is where every finding on this change actually came
    from."""
    import random

    from utils.log_redaction import _strip_ansi

    rng = random.Random(20260913)
    alphabet = (
        list("\x1b\x07\x9c\x9b\x9d\x9e\x9f\x90\x98") * 4
        + list("[]P^_X0123456789;?m\\ ") * 2
        + list("abcdefghijklmnopqrstuvwxyzABCDEF =:\"'-_/.&?\n\r")
    )
    for _ in range(20000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
        assert _strip_ansi(text) == _LAZY_ANSI_RE.sub("", text), repr(text)


def test_an_unterminated_ansi_introducer_does_not_cost_quadratic_time():
    """A lazy scan for the terminator backtracks: the introducer with no
    terminator scans to end of string, fails, and falls through to the single
    character Fe branch, so the cost grows with the square of the record.

    Before the negated body classes, 40k of these took ~15.8s against ~0.005s
    for the same length of ordinary text, and the log viewer hands whole lines
    to this function once a second. An unterminated introducer is not exotic; a
    rotated log or a writer cut mid sequence leaves one behind.

    Timing is asserted loosely, as a shape rather than a number: quadratic here
    is seconds and linear is milliseconds, so any threshold in between separates
    them on any host.
    """
    import time

    shapes = [
        "\x9d",
        "\x90",
        "\x98",
        "\x9e",
        "\x9f",
        "\x9b",
        "\x1b",
        "\x1b]",
        "\x1bP",
        "\x1b[",
        # Interleaved: no single body class can run to the end, but each start
        # still offers the next one a fresh full scan under a lazy body.
        "\x1b]\x9d",
        "\x1bP\x9e\x1b[",
        # A terminator that belongs to nobody, and a key in front of it, which
        # is what a colorized line cut at a page boundary actually looks like.
        "api_key\x9d",
        "\x1b]title\x9d",
    ]
    for shape in shapes:
        text = shape * (40000 // len(shape))
        started = time.monotonic()
        redact_log_text(text)
        elapsed = time.monotonic() - started
        assert elapsed < 2.0, f"{shape!r} took {elapsed:.1f}s"


# The seven introducers _ANSI_INTRODUCER_RE recognises, which is the set that
# decides whether the strip runs at all.
_INTRODUCERS = ("\x1b", "\x90", "\x98", "\x9b", "\x9d", "\x9e", "\x9f")


def test_terminated_ansi_sequences_are_still_stripped():
    """The walk must not cost the stripping the rules depend on: an escape
    between a key and its value stops every anchored rule matching, and a
    terminated sequence has to disappear whichever of the six forms it is."""
    for text in (
        "\x1b[36mpassword\x1b[0m=hunter2secret",
        "\x1b]0;title\x1b\\api_key=abcdef123456",
        "\x1b]0;title\x07api_key=abcdef123456",
        "\x1b]0;title\x9capi_key=abcdef123456",
        "\x1bPsome dcs\x1b\\api_key=abcdef123456",
        "\x9dbody\x9capi_key=abcdef123456",
        "\x9bmapi_key=abcdef123456",
        "api\x1b[36m_key=abcdef123456",
        "api\x1b[36mkey=abcdef123456",
    ):
        masked = redact_log_text(text)
        assert "abcdef123456" not in masked and "hunter2secret" not in masked, text
        assert REDACTED in masked, text
