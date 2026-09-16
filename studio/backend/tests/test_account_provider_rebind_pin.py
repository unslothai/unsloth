# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's provider hostname is re-resolved and pinned per connection, so a
name that validated as public cannot rebind to loopback before the request is sent."""

import asyncio
import http.server
import socket
import threading

import pytest

from core.inference import external_provider, providers
from core.inference.external_provider import ExternalProviderClient
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
REBINDING_HOST = "rebind.example"
PUBLIC_ADDRESS = "93.184.216.34"


@pytest.fixture
def owner_local():
    """A loopback server counting the requests that reached it."""
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = b'{"data": [{"id": "owner-local-model"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        yield server.server_address[1], hits
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def rebinding_dns(monkeypatch, owner_local):
    """The first lookup of the name answers a public address; every later one answers loopback."""
    port, _ = owner_local
    real = socket.getaddrinfo
    answers = []

    def fake(host, *args, **kwargs):
        if host != REBINDING_HOST:
            return real(host, *args, **kwargs)
        address = PUBLIC_ADDRESS if not answers else "127.0.0.1"
        answers.append(address)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    monkeypatch.setattr(providers, "_dns_cache", {})
    monkeypatch.setattr(external_provider, "_managed_http_client", None, raising = False)
    return answers


def _list_models_as(account, base_url):
    token = bind_account(account)
    try:
        client = ExternalProviderClient("custom", base_url, "key")
        return asyncio.run(client.list_models())
    finally:
        reset_account(token)


def test_managed_account_request_does_not_follow_a_rebound_name(owner_local, rebinding_dns):
    port, hits = owner_local
    with pytest.raises(Exception):
        _list_models_as(ALICE, f"http://{REBINDING_HOST}:{port}/v1")
    assert hits == [], hits
    assert rebinding_dns[0] == PUBLIC_ADDRESS and "127.0.0.1" in rebinding_dns[1:]


def test_owner_request_still_reaches_a_local_provider(owner_local):
    port, hits = owner_local
    models = _list_models_as(OWNER, f"http://127.0.0.1:{port}/v1")
    assert [m["id"] for m in models] == ["owner-local-model"]
    assert hits == ["/v1/models"]
