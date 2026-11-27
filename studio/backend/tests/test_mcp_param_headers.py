# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#19876: Streamable HTTP tools that mark params with x-mcp-header need Mcp-Param-* on tools/call.
When the call session has never listed tools, GitHub (and peers) reject with header mismatch; Studio
must tools/list and retry once instead of returning that error to the model."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mcp_client


class _HeaderMismatch(Exception):
    def __init__(
        self,
        message: str,
        code: int = -32020,
    ):
        super().__init__(message)
        self.error = SimpleNamespace(code = code, message = message)


def test_header_mismatch_detection_matches_github_style_messages():
    assert mcp_client._is_header_mismatch(
        _HeaderMismatch('header mismatch: missing Mcp-Param-owner header for parameter "owner"')
    )
    assert mcp_client._is_header_mismatch(_HeaderMismatch("HeaderMismatch", code = -32020))
    # Session-gone uses -32001 with a different message; must not be treated as header mismatch.
    assert not mcp_client._is_header_mismatch(_HeaderMismatch("Session not found", code = -32001))
    assert not mcp_client._is_header_mismatch(RuntimeError("tool blew up"))


def test_is_error_result_with_header_mismatch_text_is_detected():
    result = SimpleNamespace(
        is_error = True,
        content = [
            SimpleNamespace(
                text = 'header mismatch: missing Mcp-Param-owner header for parameter "owner"'
            )
        ],
    )
    assert mcp_client._result_header_mismatch(result)
    assert not mcp_client._result_header_mismatch(
        SimpleNamespace(is_error = True, content = [SimpleNamespace(text = "permission denied")])
    )
    assert not mcp_client._result_header_mismatch(
        SimpleNamespace(is_error = False, content = [SimpleNamespace(text = "header mismatch")])
    )


def test_call_tool_retries_once_after_header_mismatch_exception():
    calls: list[tuple[str, dict]] = []
    lists = 0

    class _Client:
        async def call_tool(
            self,
            name,
            args,
            raise_on_error = True,
        ):
            calls.append((name, dict(args)))
            if len(calls) == 1:
                raise _HeaderMismatch(
                    'header mismatch: missing Mcp-Param-owner header for parameter "owner"'
                )
            return SimpleNamespace(
                content = [SimpleNamespace(text = "wrote file", type = "text")],
                is_error = False,
                structured_content = None,
            )

        async def list_tools(self):
            nonlocal lists
            lists += 1
            return []

    out = asyncio.run(
        mcp_client._call_tool_respecting_headers(
            _Client(), "create_or_update_file", {"owner": "acme", "path": "index.html"}
        )
    )
    assert lists == 1
    assert calls == [
        ("create_or_update_file", {"owner": "acme", "path": "index.html"}),
        ("create_or_update_file", {"owner": "acme", "path": "index.html"}),
    ]
    assert getattr(out.content[0], "text") == "wrote file"


def test_call_tool_retries_once_after_is_error_header_mismatch():
    calls = 0
    lists = 0

    class _Client:
        async def call_tool(
            self,
            name,
            args,
            raise_on_error = True,
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                return SimpleNamespace(
                    is_error = True,
                    content = [
                        SimpleNamespace(
                            text = (
                                "header mismatch: missing Mcp-Param-owner header "
                                'for parameter "owner"'
                            )
                        )
                    ],
                    structured_content = None,
                )
            return SimpleNamespace(
                is_error = False,
                content = [SimpleNamespace(text = "ok", type = "text")],
                structured_content = None,
            )

        async def list_tools(self):
            nonlocal lists
            lists += 1
            return []

    out = asyncio.run(
        mcp_client._call_tool_respecting_headers(_Client(), "get_file_contents", {"owner": "o"})
    )
    assert lists == 1 and calls == 2
    assert getattr(out.content[0], "text") == "ok"


def test_non_header_errors_are_not_retried():
    calls = 0
    lists = 0

    class _Client:
        async def call_tool(
            self,
            name,
            args,
            raise_on_error = True,
        ):
            nonlocal calls
            calls += 1
            raise RuntimeError("rate limited")

        async def list_tools(self):
            nonlocal lists
            lists += 1
            return []

    with pytest.raises(RuntimeError, match = "rate limited"):
        asyncio.run(mcp_client._call_tool_respecting_headers(_Client(), "push_files", {}))
    assert calls == 1 and lists == 0


def test_call_tool_sync_recovers_from_header_mismatch(monkeypatch):
    calls = 0
    lists = 0

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def call_tool(
            self,
            name,
            args,
            raise_on_error = True,
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _HeaderMismatch(
                    'header mismatch: missing Mcp-Param-owner header for parameter "owner"'
                )
            return SimpleNamespace(
                content = [SimpleNamespace(text = "created", type = "text")],
                is_error = False,
                structured_content = None,
            )

        async def list_tools(self):
            nonlocal lists
            lists += 1
            return []

    monkeypatch.setattr(mcp_client, "_client", lambda *a, **kw: _Client())
    monkeypatch.setattr(mcp_client, "validate_mcp_address", lambda url: None)

    out = mcp_client.call_tool_sync(
        url = "https://api.githubcopilot.com/mcp/",
        headers = {"Authorization": "Bearer x"},
        name = "create_or_update_file",
        args = {"owner": "acme", "content": "<html/>"},
        timeout = 5.0,
    )
    assert out == "created"
    assert calls == 2 and lists == 1
