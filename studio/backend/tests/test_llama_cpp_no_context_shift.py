# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``--no-context-shift`` launch-flag contract.

When llama-server runs with its default context-shift behavior, the UI
has no way to tell the user that the KV cache has been rotated --
earlier turns silently vanish from the conversation. The Studio
backend always passes ``--no-context-shift`` so the server returns a
clean error instead, and the chat adapter can point the user at the
``Context Length`` input in the settings panel.

This file is a static read of the launch command: we ask
``LlamaCppBackend`` to assemble its ``cmd`` list and assert the flag
is always present. Testing via the real subprocess would require an
actual GGUF on disk, which is out of scope for the fast test suite.
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

    Using ``inspect.getsource`` instead of reading the file directly
    scopes the assertions to the function that actually launches
    llama-server, so neither the presence check nor the location check
    can be fooled by a stray occurrence of ``"--no-context-shift"``
    elsewhere in the module.
    """
    return inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)


def test_no_context_shift_is_in_load_model():
    """The flag is part of the static launch-command template.

    We check the source of ``load_model`` rather than mocking the whole
    call chain (GPU probing, GGUF stat, etc.): the flag is written as
    a literal in one place and any regression has to delete it, which
    a text search will catch.
    """
    assert '"--no-context-shift"' in _load_model_source(), (
        "llama-server must be launched with --no-context-shift so the "
        "UI can surface a clean 'context full' error instead of silently "
        "losing old turns to a KV-cache rotation."
    )


def test_flag_sits_inside_the_base_cmd_list():
    """Pin the flag's location so a future refactor can't accidentally
    move it into a branch that only fires on some code paths.

    We slice from ``cmd = [`` to the first ``]`` at the same indent.
    Using ``inspect.getsource`` means the function lives in its own
    string and there are no siblings to worry about, so a plain
    bracket search would also work -- anchoring on the trailing indent
    just keeps the slice from wandering into a later expression if the
    opening literal ever grows an in-line comment trailing it.
    """
    source = _load_model_source()
    start = source.find("cmd = [")
    assert start >= 0, "could not find the base cmd = [...] block"
    # Find the first line containing only ``]`` (possibly indented).
    # Works for any indentation style the formatter picks.
    rest = source[start:]
    end_rel = -1
    for line_start, line in _iter_lines_with_offset(rest):
        if line_start == 0:
            # Skip the opening ``cmd = [`` line itself.
            continue
        if line.strip() == "]":
            end_rel = line_start
            break
    assert end_rel > 0, "could not find end of cmd = [...] block"
    block = rest[:end_rel]
    assert '"--no-context-shift"' in block, (
        "--no-context-shift must be in the base cmd list, not in a "
        "conditional branch -- otherwise some code paths would still "
        "run with silent context shift enabled."
    )
    # Also pin that it is next to -c / --ctx so the grouping makes sense.
    assert '"-c"' in block
    assert '"--flash-attn"' in block


def _iter_lines_with_offset(text: str):
    """Yield (offset, line) pairs over ``text`` without losing offsets."""
    offset = 0
    for line in text.splitlines(keepends = True):
        yield offset, line
        offset += len(line)
