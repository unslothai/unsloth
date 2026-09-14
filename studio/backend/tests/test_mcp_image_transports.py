# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import json
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from starlette.requests import ClientDisconnect

from core.inference import mcp_client
from core.inference.mcp_image_redaction import PRIVATE_CALL_COMPLETE, PRIVATE_CALL_ERROR
from models.inference import ChatCompletionRequest
from routes import inference
from studio.backend.tests.test_mcp_image_redaction import ENCODED, PUBLIC, make_context


def _response(read):
    return SimpleNamespace(
        status = 200,
        read = read,
        close = lambda: None,
        getheader = lambda name, default = None: "application/json"
        if name == "Content-Type"
        else default,
    )


def _connection(response = None, **kwargs):
    def getresponse():
        if isinstance(response, Exception):
            raise response
        return response

    return SimpleNamespace(
        sock = kwargs.get("sock"),
        getresponse = getresponse,
        request = kwargs.get("request", lambda *args, **kw: None),
        close = kwargs.get("close", lambda: None),
    )


def _serve(
    *,
    calls = None,
    error = False,
    blocked = None,
):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            if blocked:
                data = b'{"jsonrpc":"2.0","id":1,"result":{}}'
            else:
                message = json.loads(raw)
                if message["method"] == "notifications/initialized":
                    self.send_response(202)
                    self.end_headers()
                    return
                result = {"protocolVersion": "2024-11-05", "capabilities": {}}
                if message["method"] != "initialize":
                    calls.append(message)
                    image = message["params"]["arguments"]["picture"]
                    result = {
                        "content": [
                            {"type": "text", "text": "private server detail" if error else image}
                        ],
                        **({"isError": True} if error else {}),
                    }
                data = json.dumps(
                    {"jsonrpc": "2.0", "id": message["id"], "result": result}
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if blocked:
                self.send_header("Connection", "close")
            else:
                self.send_header("Mcp-Session-Id", "fixed-session")
            self.end_headers()
            if blocked:
                blocked[0].set()
                blocked[1].wait(5)
            try:
                self.wfile.write(data)
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    return server, thread


def _stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join()


def _assert_cancelled(transport, cancel):
    started = time.monotonic()
    with pytest.raises(Exception):
        transport.exchange("initialize", {}, cancel_event = cancel)
    assert cancel.is_set()
    assert time.monotonic() - started < 2


@pytest.fixture(params = [False, True], ids = ["success", "error"])
def http_recipient(request):
    calls = []
    error = request.param
    server, thread = _serve(calls = calls, error = error)
    yield f"http://127.0.0.1:{server.server_port}/mcp", calls, error
    _stop_server(server, thread)


@pytest.fixture(autouse = True)
def cleanup_recipients():
    yield
    with mcp_client._private_recipients_lock:
        recipients = list(mcp_client._private_recipients.values())
        mcp_client._private_recipients.clear()
        mcp_client._private_recipient_connects.clear()
    for recipient in recipients:
        recipient.close()


def test_recipient_capacity_is_reserved_before_initialization(monkeypatch):
    entered, release = threading.Semaphore(0), threading.Event()
    constructed = []

    class SlowTransport:
        def __init__(self, url, headers, timeout):
            self.url, self.account = url, mcp_client.current_account_id()
            self.identity, self.created_at = f"recipient-{len(constructed)}", time.monotonic()
            self.closed = threading.Event()
            constructed.append(self)

        def exchange(self, method, *args, **kwargs):
            if method == "initialize":
                entered.release()
                release.wait(5)
                return {"protocolVersion": "2024-11-05"}

        def close(self):
            self.closed.set()

    monkeypatch.setattr(mcp_client, "_DEFAULT_MAX_SESSIONS", 2)
    monkeypatch.setattr(mcp_client, "_PrivateMcpTransport", SlowTransport)
    monkeypatch.setattr(mcp_client, "validate_mcp_address", lambda _: None)
    results = []

    def prepare():
        try:
            results.append(mcp_client.prepare_mcp_image_recipient("https://recipient.test/mcp"))
        except Exception:
            results.append("rejected")

    workers = [threading.Thread(target = prepare) for _ in range(2)]
    for worker in workers:
        worker.start()
    assert entered.acquire(timeout = 2) and entered.acquire(timeout = 2)
    started = time.monotonic()
    prepare()
    assert time.monotonic() - started < 1
    assert results == ["rejected"] and len(constructed) == 2
    release.set()
    for worker in workers:
        worker.join(timeout = 2)
        assert not worker.is_alive()
    assert len(results) == 3
    assert mcp_client._private_recipient_connects == {}
    mcp_client._private_recipients.clear()
    mcp_client._private_recipients["other"] = SimpleNamespace(
        account = "other", created_at = time.monotonic(), closed = threading.Event(), close = lambda: None
    )
    monkeypatch.setattr(mcp_client, "_DEFAULT_MAX_SESSIONS", 1)
    monkeypatch.setattr(mcp_client, "current_account_id", lambda: "mine")
    reservation = mcp_client._reserve_private_recipient()
    assert reservation == "mine"
    mcp_client._release_private_recipient(reservation)


def test_private_tool_call_can_be_explicitly_unbounded(monkeypatch):
    transport = mcp_client._PrivateMcpTransport("http://example.test/mcp", {}, 0.01)
    response = _response(
        lambda _limit: json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
        ).encode()
    )
    monkeypatch.setattr(
        transport,
        "_connection",
        lambda _: _connection(response, request = lambda *args, **kwargs: time.sleep(0.05)),
    )
    try:
        assert transport.exchange("initialize", {}, timeout = None)["protocolVersion"] == "2024-11-05"
    finally:
        transport.close()


def test_http_send_is_one_use_and_withholds_all_results(http_recipient, caplog, monkeypatch):
    url, calls, error = http_recipient
    cancel, initialization_events = threading.Event(), []
    original_exchange = mcp_client._PrivateMcpTransport.exchange

    def exchange(transport, *args, **kwargs):
        initialization_events.append(kwargs.get("cancel_event"))
        return original_exchange(transport, *args, **kwargs)

    monkeypatch.setattr(mcp_client._PrivateMcpTransport, "exchange", exchange)
    identity = mcp_client.prepare_mcp_image_recipient(url, cancel_event = cancel)
    assert initialization_events == [cancel, cancel]
    context = make_context(recipient = identity)
    result = mcp_client.call_tool_sync(
        url, None, "inspect_picture", PUBLIC, disclosure_context = context, config_check = lambda: True
    )
    assert len(calls) == 1 and calls[0]["params"]["arguments"]["picture"] == ENCODED
    assert result == (PRIVATE_CALL_ERROR if error else PRIVATE_CALL_COMPLETE)
    assert ENCODED not in result + caplog.text
    assert "private server detail" not in result
    assert (
        mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC, disclosure_context = context)
        == PRIVATE_CALL_ERROR
    )
    assert len(calls) == 1


def test_http_revalidates_configuration_in_worker_before_write(monkeypatch):
    transport = mcp_client._PrivateMcpTransport("http://example.test/mcp", {}, 5)
    worker_started, release_worker = threading.Event(), threading.Event()
    valid, writes = [True], []
    connection = _connection(
        AssertionError("an invalid configuration must not send"),
        request = lambda *args, **kwargs: writes.append(1),
    )

    def delayed_account_thread(*, target, daemon):
        def delayed_target():
            worker_started.set()
            release_worker.wait(2)
            target()

        return threading.Thread(target = delayed_target, daemon = daemon)

    def revoke():
        assert worker_started.wait(2)
        valid[0] = False
        release_worker.set()

    monkeypatch.setattr(transport, "_connection", lambda _: connection)
    monkeypatch.setattr(mcp_client, "account_thread", delayed_account_thread)
    revoker = threading.Thread(target = revoke, daemon = True)
    revoker.start()
    context = make_context(recipient = transport.identity)
    with pytest.raises(Exception):
        transport.exchange(
            "tools/call",
            {"name": "inspect_picture"},
            context = context,
            arguments = PUBLIC,
            config_check = lambda: valid[0],
        )
    assert writes == [] and context.spent is False
    revoker.join(timeout = 2)
    assert not revoker.is_alive()


@pytest.mark.parametrize("boundary", ["connection", "response"])
def test_http_cancel_interrupts_mocked_boundary(monkeypatch, boundary):
    transport = mcp_client._PrivateMcpTransport("http://example.test/mcp", {}, 30)
    cancel = threading.Event()
    if boundary == "connection":
        entered, release, worker_done = (threading.Event() for _ in range(3))
        writes = []

        def blocked_connection(_deadline):
            entered.set()
            release.wait(5)
            return _connection(
                request = lambda *args, **kwargs: writes.append(1), close = worker_done.set
            )

        monkeypatch.setattr(transport, "_connection", blocked_connection)
        threading.Thread(target = lambda: entered.wait(2) and cancel.set(), daemon = True).start()
    else:
        left, right = socket.socketpair()
        connection = _connection(_response(lambda *_: left.recv(1)), sock = left, close = left.close)
        monkeypatch.setattr(
            transport, "_connection", lambda _: setattr(transport, "http", connection) or connection
        )
        threading.Timer(0.1, cancel.set).start()
    try:
        _assert_cancelled(transport, cancel)
    finally:
        if boundary == "connection":
            release.set()
            assert worker_done.wait(2)
            assert writes == []
        else:
            right.close()


def test_http_cancel_interrupts_connection_close_response_body():
    headers_sent, release = threading.Event(), threading.Event()
    server, thread = _serve(blocked = (headers_sent, release))
    cancel = threading.Event()
    transport = mcp_client._PrivateMcpTransport(f"http://127.0.0.1:{server.server_port}/mcp", {}, 5)
    threading.Thread(target = lambda: headers_sent.wait(2) and cancel.set(), daemon = True).start()
    try:
        _assert_cancelled(transport, cancel)
    finally:
        release.set()
        transport.close()
        _stop_server(server, thread)


def test_stdio_send_redacts_results_and_side_channels(tmp_path, monkeypatch, capfd):
    script = tmp_path / "server.py"
    script.write_text(
        """import json, sys
for line in sys.stdin:
    message = json.loads(line)
    method = message['method']
    if method == 'notifications/initialized': continue
    if method == 'initialize': result = {'protocolVersion': '2024-11-05', 'capabilities': {}}
    else:
        image = message['params']['arguments']['picture']
        print(image, file=sys.stderr, flush=True); print(json.dumps({'jsonrpc':'2.0','method':'notifications/message','params':{'data':image}}), flush=True)
        result = {'content':[{'type':'text','text':image}, {'type':'text','text':'safe stdio label'}]}
    print(json.dumps({'jsonrpc':'2.0','id':message['id'],'result':result}), flush=True)
""",
        encoding = "utf-8",
    )
    monkeypatch.setattr(mcp_client, "stdio_mcp_enabled", lambda: True)
    url = mcp_client.join_stdio_command([sys.executable, "-u", str(script)])
    identity = mcp_client.prepare_mcp_image_recipient(url)
    result = mcp_client.call_tool_sync(
        url,
        None,
        "classify_frame",
        PUBLIC,
        timeout = 5,
        disclosure_context = make_context(recipient = identity, tool_name = "classify_frame"),
    )
    assert result == PRIVATE_CALL_COMPLETE
    assert ENCODED not in result + "".join(capfd.readouterr())


@pytest.mark.parametrize("allowed", [True, False])
def test_direct_dispatch_cannot_bypass_server_permission(monkeypatch, allowed):
    from core.inference import tools

    server = {
        "id": "server-1",
        "url": "https://example.test/mcp",
        "is_enabled": True,
        "allow_image_attachments": allowed,
        "config_revision": 1,
        "image_input_mappings_json": json.dumps(
            [{"tool": "inspect_picture", "field": "picture", "encoding": "base64"}]
        ),
    }
    calls = []
    monkeypatch.setattr(tools.mcp_servers_db, "get_server_for_tool", lambda _: server)
    monkeypatch.setattr(
        tools, "call_tool_sync", lambda **kwargs: calls.append(kwargs) or "ordinary result"
    )
    monkeypatch.setattr(tools, "_fit_result_to_room", lambda result, *args: result)
    result = tools.execute_tool("mcp__server-1__inspect_picture", {"picture": ENCODED})
    assert (result.startswith("Error:") and not calls) if allowed else bool(calls)
    if allowed:
        assert (
            tools.execute_tool("mcp__server-1__inspect_picture", {"threshold": 0.5})
            == "ordinary result"
        )


def _assert_recipient_unavailable():
    for operation in (
        mcp_client.mcp_image_recipient_location,
        mcp_client.mcp_image_recipient_remaining_ms,
    ):
        with pytest.raises(mcp_client._PrivateTransportUnavailable):
            operation("opaque")


def test_recipient_is_account_isolated_and_concurrent_close_claims_once(monkeypatch):
    closes = []
    transport = SimpleNamespace(
        account = "owner",
        location = "https://private-recipient.example/mcp",
        created_at = time.monotonic(),
        close = lambda: closes.append("closed"),
    )
    monkeypatch.setattr(mcp_client, "_private_recipients", {"opaque": transport})
    monkeypatch.setattr(mcp_client, "_private_recipients_lock", threading.Lock())
    monkeypatch.setattr(mcp_client, "current_account_id", lambda: "other")
    _assert_recipient_unavailable()
    mcp_client.close_mcp_image_recipient("opaque")
    assert (mcp_client._private_recipients, closes) == ({"opaque": transport}, [])
    monkeypatch.setattr(mcp_client, "current_account_id", lambda: "owner")
    assert mcp_client.mcp_image_recipient_location("opaque") == transport.location
    assert 0 < mcp_client.mcp_image_recipient_remaining_ms("opaque") <= 300_000
    barrier = threading.Barrier(2)

    def close(_):
        barrier.wait()
        mcp_client.close_mcp_image_recipient("opaque")

    with ThreadPoolExecutor(max_workers = 2) as executor:
        list(executor.map(close, range(2)))
    assert (closes, mcp_client._private_recipients) == (["closed"], {})
    mcp_client.close_mcp_image_recipient("opaque")
    _assert_recipient_unavailable()
    assert closes == ["closed"]


@pytest.mark.parametrize("provider", ["custom", "openai_codex"])
def test_provider_disconnect_before_body_closes_eager_image_run(monkeypatch, provider):
    from core.inference import external_provider, openai_codex_auth, openai_codex_client
    from core.inference.openai_codex_auth import OPENAI_CODEX_API_BASE
    from core.inference.providers import get_provider_info

    closed, clients = [], []
    image_run = SimpleNamespace(close = lambda: closed.append("image"))

    class Client:
        def __init__(self, *args, **kwargs):
            clients.append(self)
            self.closed = False

        async def close(self):
            self.closed = True

    for target, name in (
        (external_provider, "ExternalProviderClient"),
        (inference, "ExternalProviderClient"),
        (openai_codex_client, "OpenAICodexClient"),
    ):
        monkeypatch.setattr(target, name, Client)

    async def prepare(payload, subject, tools, cancel_event, ui_events):
        return (image_run, tools) if tools is not None else (None, None)

    monkeypatch.setattr(inference, "_prepare_mcp_image_for_route", prepare)
    if provider == "openai_codex":
        model = get_provider_info(provider)["default_models"][0]
        provider_row = {
            "provider_type": provider,
            "base_url": OPENAI_CODEX_API_BASE,
            "display_name": "ChatGPT",
            "is_enabled": True,
            "models": [model],
        }
        monkeypatch.setattr(
            inference.providers_db, "get_provider", lambda pid: {"id": pid, **provider_row}
        )
        monkeypatch.setattr(
            openai_codex_auth, "load_oauth_bundle", lambda pid: {"account_id": "account"}
        )
        for name, value in {
            "subscription_catalog_matches_account": True,
            "subscription_catalog_known": False,
            "subscription_catalog_stale": False,
            "saved_models_proven_for": True,
        }.items():
            monkeypatch.setattr(openai_codex_client, name, lambda *args, value = value: value)

        async def resolve(*args, **kwargs):
            return "token", "account"

        monkeypatch.setattr(openai_codex_auth, "resolve_access", resolve)
        provider_fields = {"provider_id": "codex", "external_model": model}
    else:
        provider_fields = {
            "provider_type": "custom",
            "provider_base_url": "https://example.test/v1",
            "external_model": "test",
        }

    async def disconnected():
        return False

    request = SimpleNamespace(
        headers = {}, state = SimpleNamespace(skip_api_monitor = True), is_disconnected = disconnected
    )
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hello"}], stream = True, **provider_fields
    )

    async def run():
        response = await inference._proxy_to_external_provider(payload, request, "user")
        assert isinstance(response, inference._SameTaskStreamingResponse)

        async def send(message):
            assert message["type"] == "http.response.start"
            raise OSError("client disconnected")

        async def receive():
            return {"type": "http.disconnect"}

        with pytest.raises(ClientDisconnect):
            await response({}, receive, send)

    asyncio.run(run())
    assert closed == ["image"]
    assert (
        (clients and all(client.closed for client in clients))
        if provider == "custom"
        else not clients
    )
