# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``--no-context-shift`` launch-flag contract.

With llama-server's default context-shift behavior, the UI cannot tell the user
the KV cache was rotated -- earlier turns silently vanish from the conversation.
The Unsloth backend always passes ``--no-context-shift`` so the server returns a
clean error instead, and the chat adapter can point the user at the
``Context Length`` input in the settings panel.

This file statically reads the launch command: we ask ``LlamaCppBackend`` to
assemble its ``cmd`` list and assert the flag is present. Testing via the real
subprocess would need an actual GGUF on disk, out of scope for the fast suite.
"""

from __future__ import annotations

import inspect
import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Same external-dep stubs as the other llama_cpp tests.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

_httpx_stub = _types.ModuleType("httpx")
for _exc in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
_httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
_httpx_stub.Client = type(
    "C",
    (),
    {
        "__init__": lambda s, **kw: None,
        "__enter__": lambda s: s,
        "__exit__": lambda s, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference import llama_cpp as llama_cpp_module


def _load_model_source() -> str:
    """Return the source of ``LlamaCppBackend.load_model``.

    Using ``inspect.getsource`` instead of reading the file scopes the assertions
    to the function that launches llama-server, so neither the presence nor the
    location check can be fooled by a stray ``"--no-context-shift"`` elsewhere in
    the module.
    """
    return inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)


def test_no_context_shift_is_in_load_model():
    """The flag is part of the static launch-command template.

    We check the source of ``load_model`` rather than mocking the whole call
    chain (GPU probing, GGUF stat, etc.): the flag is a literal in one place and
    any regression must delete it, which a text search catches.
    """
    assert '"--no-context-shift"' in _load_model_source(), (
        "llama-server must be launched with --no-context-shift so the "
        "UI can surface a clean 'context full' error instead of silently "
        "losing old turns to a KV-cache rotation."
    )


def test_the_flag_is_emitted_unless_the_build_lacks_it():
    """The gate replaces the old "must be a literal in the base list" pin.

    It used to sit unconditionally inside ``cmd = [...]``, which meant a stale
    or user-supplied LLAMA_SERVER_PATH without the flag got it anyway and
    exited on an unknown argument. It is now gated, but the gate FAILS OPEN:
    the capability defaults to True everywhere, so an unreadable --help keeps
    today's command and only a build whose help positively lacks the flag
    drops it.
    """
    source = _load_model_source()
    assert 'cmd.append("--no-context-shift")' in source
    assert 'if _caps.get("supports_no_context_shift", True):' in source, (
        "the gate must default to True, so a failed probe still emits the flag"
    )
    # And the default really is True in both places the probe can return.
    probe_src = inspect.getsource(llama_cpp_module.LlamaCppBackend.probe_server_capabilities)
    assert '"supports_no_context_shift": True' in probe_src
    assert "supports_no_context_shift = True" in probe_src


def test_the_base_cmd_list_still_leads_straight_into_the_context_flag():
    """-c must stay grouped with the base list.

    auto-fit must omit -c entirely, because "-c 0" pins the full native context
    and disables --fit's VRAM-based sizing, so the emission needs to stay where
    that reasoning is visible.
    """
    source = _load_model_source()
    start = source.find("cmd = [")
    assert start >= 0, "could not find the base cmd = [...] block"
    rest = source[start:]
    end_rel = -1
    for line_start, line in _iter_lines_with_offset(rest):
        if line_start == 0:
            continue
        if line.strip() == "]":
            end_rel = line_start
            break
    assert end_rel > 0, "could not find end of cmd = [...] block"
    after = rest[end_rel : end_rel + 1400]
    assert '"-c"' in after, (
        "-c must still be emitted near the base cmd list (omitted only in "
        "auto-fit, where --fit sizes context)."
    )


def test_flash_attention_drops_its_value_only_for_a_boolean_build():
    """Older builds take -fa as a bare boolean and read "on" as a positional.

    That is an immediate "invalid argument" exit, not a degraded launch.
    """
    value_form = "-fa, --flash-attn [on|off|auto]   set flash attention"
    boolean_form = "-fa, --flash-attn                 enable flash attention"
    assert llama_cpp_module.LlamaCppBackend._flash_attn_takes_value(value_form) is True
    assert llama_cpp_module.LlamaCppBackend._flash_attn_takes_value(boolean_form) is False
    # Fail open when the help says nothing about it, since the pinned prebuilt
    # is the value form and guessing wrong there breaks the supported path.
    assert llama_cpp_module.LlamaCppBackend._flash_attn_takes_value("-m, --model FNAME") is True
    assert llama_cpp_module.LlamaCppBackend._flash_attn_takes_value("") is True


def _iter_lines_with_offset(text: str):
    """Yield (offset, line) pairs over ``text`` without losing offsets."""
    offset = 0
    for line in text.splitlines(keepends = True):
        yield offset, line
        offset += len(line)
