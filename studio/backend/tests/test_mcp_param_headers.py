# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#19876: tools/call to a server whose params carry x-mcp-header (GitHub's) must send Mcp-Param-*."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

pytest.importorskip("fastmcp")
pytest.importorskip("uvicorn")

from core.inference import mcp_client
from core.inference.mcp_client import call_tool_sync, close_mcp_sessions

_SERVER = """
import socket, sys
from typing import Annotated
import uvicorn
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from pydantic import Field

mcp = FastMCP("gh-like")
_runs = []


@mcp.tool
def create_or_update_file(
    owner: Annotated[str, Field(json_schema_extra={"x-mcp-header": "owner"})],
    path: str,
    content: str,
) -> str:
    _runs.append(path)
    seen = get_http_headers(include_all=True).get("mcp-param-owner")
    return f"wrote {path} header={seen} runs={len(_runs)}"


if __name__ == "__main__":
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    print(f"PORT {sock.getsockname()[1]}", flush=True)
    uvicorn.Server(uvicorn.Config(mcp.http_app(), log_level="warning")).run(sockets=[sock])
"""


@pytest.fixture(scope = "module")
def server(tmp_path_factory):
    script = tmp_path_factory.mktemp("mcp") / "gh_like_server.py"
    script.write_text(textwrap.dedent(_SERVER))
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        env = dict(os.environ, PYTHONUNBUFFERED = "1"),
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
    )
    first = proc.stdout.readline()
    if not first.startswith("PORT "):
        proc.kill()
        raise RuntimeError(f"server died:\n{first}{proc.stdout.read()}")
    url = f"http://127.0.0.1:{int(first.split()[1])}/mcp/"
    port = int(first.split()[1])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), 0.5).close()
            break
        except OSError:
            time.sleep(0.2)
    try:
        yield url
    finally:
        close_mcp_sessions()
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(autouse = True)
def clean_cache():
    yield
    close_mcp_sessions()


def _write(url, path, scope):
    args = {"owner": "acme", "path": path, "content": "<html>" + "x" * 10_000 + "</html>"}
    return call_tool_sync(url, None, "create_or_update_file", args, timeout = 60.0, scope = scope)


def _runs(text):
    return int(text.rsplit("runs=", 1)[1])


def test_first_call_on_a_fresh_chat_session_sends_the_header(server):
    out = _write(server, "index.html", "s=sess1:t=threadA")
    assert "header=acme" in out, out


def test_a_call_after_the_idle_session_was_reaped_sends_the_header(server):
    scope = "s=sess1:t=threadB"
    assert "header=acme" in _write(server, "a.html", scope)
    # A long generation outlives _SESSION_IDLE_TTL; the reaper closes the session.
    mcp_client._reap_idle_sessions(time.monotonic() + mcp_client._SESSION_IDLE_TTL + 1)
    out = _write(server, "b.html", scope)
    assert "header=acme" in out, out


def test_one_shot_call_recovers_and_runs_the_tool_once(server):
    before = _runs(_write(server, "c.html", "s=sess1:t=threadC"))
    out = _write(server, "d.html", None)
    assert "header=acme" in out, out
    assert _runs(out) == before + 1


def test_only_header_mismatch_errors_are_retried():
    from mcp.shared.exceptions import MCPError

    def err(code):
        return MCPError(code, "x")

    assert mcp_client._is_header_mismatch(err(-32020))
    assert not mcp_client._is_header_mismatch(err(-32001))
    assert not mcp_client._is_header_mismatch(
        RuntimeError("header mismatch: missing Mcp-Param-owner")
    )
