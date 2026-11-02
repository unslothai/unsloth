# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pass --no-context-shift so a full KV cache errors cleanly instead of silently dropping old turns."""

from __future__ import annotations

import inspect
import sys
import types as _types
from pathlib import Path

import pytest

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
# Stub only if httpx is not installed: a stub without Response breaks later
# starlette.testclient imports for the whole session.
try:
    import httpx  # noqa: F401
except ImportError:
    sys.modules.setdefault("httpx", _httpx_stub)

from core.inference import llama_cpp as llama_cpp_module


def _load_model_source() -> str:
    """Scope to load_model's source so a stray flag string elsewhere cannot satisfy the check."""
    return inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)


def test_no_context_shift_is_in_load_model():
    """Checked as source text: the flag is a literal, so deleting it is what a regression looks like."""
    assert '"--no-context-shift"' in _load_model_source(), (
        "llama-server must be launched with --no-context-shift so the "
        "UI can surface a clean 'context full' error instead of silently "
        "losing old turns to a KV-cache rotation."
    )


def test_the_flag_is_emitted_unless_the_build_lacks_it():
    """The flag is gated on supports_no_context_shift and fails open: only a help that lacks it drops it."""
    source = _load_model_source()
    assert 'cmd.append("--no-context-shift")' in source
    assert (
        'if _caps.get("supports_no_context_shift", True):' in source
    ), "the gate must default to True, so a failed probe still emits the flag"
    probe_src = inspect.getsource(llama_cpp_module.LlamaCppBackend.probe_server_capabilities)
    assert '"supports_no_context_shift": True' in probe_src
    assert "supports_no_context_shift = True" in probe_src


def test_the_base_cmd_list_still_leads_straight_into_the_context_flag():
    """Auto-fit must omit -c entirely, since -c 0 pins the full native context and disables --fit sizing."""
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
    # Wide enough to span the gated flags (and comments) between the base list and -c.
    after = rest[end_rel : end_rel + 2400]
    assert '"-c"' in after, (
        "-c must still be emitted near the base cmd list (omitted only in "
        "auto-fit, where --fit sizes context)."
    )


def test_flash_attention_drops_its_value_only_for_a_boolean_build():
    """Builds with boolean -fa must not get a value, or they exit with an invalid argument error."""
    value_form = "-fa, --flash-attn [on|off|auto]   set flash attention"
    boolean_form = "-fa, --flash-attn                 enable flash attention"
    assert llama_cpp_module.LlamaCppBackend._flash_attn_takes_value(value_form) is True
    assert llama_cpp_module.LlamaCppBackend._flash_attn_takes_value(boolean_form) is False
    # Fail open: the pinned prebuilt is the value form.
    assert llama_cpp_module.LlamaCppBackend._flash_attn_takes_value("-m, --model FNAME") is True
    assert llama_cpp_module.LlamaCppBackend._flash_attn_takes_value("") is True


def _iter_lines_with_offset(text: str):
    """Yield (offset, line) pairs over ``text`` without losing offsets."""
    offset = 0
    for line in text.splitlines(keepends = True):
        yield offset, line
        offset += len(line)
