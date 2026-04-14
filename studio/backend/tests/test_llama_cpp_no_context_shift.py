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


def test_no_context_shift_is_in_source():
    """The flag is part of the static launch-command template.

    We intentionally check the source rather than mocking up the whole
    ``load_model`` call chain (GPU probing, GGUF stat, etc.): the flag
    is written as a literal in one place and any regression would have
    to delete it, which a text search will catch.
    """
    source = Path(llama_cpp_module.__file__).read_text()
    assert '"--no-context-shift"' in source, (
        "llama-server must be launched with --no-context-shift so the "
        "UI can surface a clean 'context full' error instead of silently "
        "losing old turns to a KV-cache rotation."
    )


def test_flag_sits_inside_the_base_cmd_list():
    """Pin the flag's location so a future refactor can't accidentally
    move it into a branch that only fires on some code paths."""
    source = Path(llama_cpp_module.__file__).read_text()
    # The base ``cmd = [ ... ]`` list opens with ``binary,\n  "-m",`` and
    # closes before the first ``if use_fit``. The flag must be inside
    # that block.
    start = source.find("cmd = [")
    assert start >= 0, "could not find the base cmd = [...] block"
    end = source.find("            ]", start)
    assert end > start, "could not find end of cmd = [...] block"
    block = source[start:end]
    assert '"--no-context-shift"' in block, (
        "--no-context-shift must be in the base cmd list, not in a "
        "conditional branch -- otherwise some code paths would still "
        "run with silent context shift enabled."
    )
    # Also pin that it is next to -c / --ctx so the grouping makes sense.
    assert '"-c"' in block
    assert '"--flash-attn"' in block
