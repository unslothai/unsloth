# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A re-authorization must keep the registered OAuth client: a new one means a new consent screen."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

pytest.importorskip("fastmcp")
pytest.importorskip("uvicorn")

import httpx2

from core.inference import mcp_client
from core.inference.mcp_client import call_tool_sync

_SERVER = '''
import json, socket, sys
import uvicorn
from fastmcp import FastMCP
from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
from mcp.server.auth.settings import ClientRegistrationOptions

sock = socket.socket(socket.AF_INET)
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
auth = InMemoryOAuthProvider(
    base_url = f"http://127.0.0.1:{port}",
    client_registration_options = ClientRegistrationOptions(enabled = True),
)
registered, authorized = [], []

register_client = auth.register_client
async def _register(info):
    registered.append(info.client_id)
    return await register_client(info)
auth.register_client = _register

authorize = auth.authorize
async def _authorize(client, params):
    authorized.append([client.client_id, str(params.redirect_uri)])
    return await authorize(client, params)
auth.authorize = _authorize

mcp = FastMCP("oauth", auth = auth)


@mcp.tool
def stats() -> str:
    return json.dumps({"registered": registered, "authorized": authorized})


@mcp.custom_route("/revoke", methods = ["POST"])
async def revoke(request):
    """What a provider does when it ends a grant: the saved access and refresh tokens stop working.
    A plain route, not a tool, so the revoking call cannot itself trigger the next sign-in."""
    from starlette.responses import PlainTextResponse

    auth.access_tokens.clear()
    auth.refresh_tokens.clear()
    return PlainTextResponse("revoked")


print(f"PORT {port}", flush = True)
uvicorn.Server(uvicorn.Config(mcp.http_app(), log_level = "warning")).run(sockets = [sock])
'''


@pytest.fixture
def server(tmp_path):
    script = tmp_path / "oauth_server.py"
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
        proc.wait(15)
        raise RuntimeError(f"server died:\n{first}{proc.stdout.read()}")
    port = int(first.split()[1])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout = 0.5):
                break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("server never came up")
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def signed_in(tmp_path, monkeypatch):
    """Persistent Studio token store, and a browser whose user approves every sign-in."""
    import fastmcp.client.auth.oauth as fastmcp_oauth

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    opened = []

    def approve(url, *args, **kwargs):
        opened.append(url)
        # The callback server starts right after the browser opens; follow the redirect to it then.
        threading.Timer(0.3, lambda: httpx2.get(url, follow_redirects = True, timeout = 30)).start()
        return True

    monkeypatch.setattr(fastmcp_oauth.webbrowser, "open", approve)
    return opened


def _call(url, name):
    return call_tool_sync(url, None, name, {}, timeout = 120.0, use_oauth = True)


def _revoke(url):
    response = httpx2.post(url.rsplit("/mcp", 1)[0] + "/revoke", timeout = 30)
    assert response.text == "revoked"


def _stats(url):
    import json

    return json.loads(_call(url, "stats"))


def test_reauthorization_keeps_the_registered_client(server, signed_in):
    first = _stats(server)
    assert len(signed_in) == 1
    assert len(first["registered"]) == 1
    client_id = first["registered"][0]

    # Reconnecting with a saved token signs nobody in.
    assert _stats(server)["registered"] == [client_id]
    assert len(signed_in) == 1

    _revoke(server)
    after = _stats(server)

    assert len(signed_in) == 2
    assert after["registered"] == [client_id]
    (_, first_redirect), (again_id, again_redirect) = after["authorized"][0], after["authorized"][-1]
    assert again_id == client_id
    assert again_redirect == first_redirect


def test_busy_registered_port_falls_back_to_a_new_client(server, signed_in):
    from urllib.parse import urlsplit

    first = _stats(server)
    port = urlsplit(first["authorized"][0][1]).port
    _revoke(server)

    with socket.socket(socket.AF_INET) as squatter:
        squatter.bind(("localhost", port))
        squatter.listen()
        after = _stats(server)

    assert len(signed_in) == 2
    assert len(after["registered"]) == 2
    assert urlsplit(after["authorized"][-1][1]).port != port
