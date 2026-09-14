# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import base64
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from core.inference import mcp_client
from core.inference.mcp_image_redaction import PRIVATE_CALL_COMPLETE, PRIVATE_CALL_ERROR
from studio.backend.tests.test_mcp_image_redaction import ENCODED, PUBLIC, make_context


@pytest.fixture
def http_recipient():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if message["method"] == "notifications/initialized":
                self.send_response(202)
                self.end_headers()
                return
            if message["method"] == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "test", "version": "1"},
                }
            else:
                calls.append(message)
                image = message["params"]["arguments"]["picture"]
                double_encoded = base64.b64encode(image.encode()).decode()
                result = {
                    "content": [
                        {"type": "text", "text": "#".join(image)},
                        {"type": "image", "data": image, "mimeType": "image/png"},
                    ],
                    "structuredContent": {"echo": double_encoded},
                }
            data = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Mcp-Session-Id", "fixed-session")
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/mcp", calls
    server.shutdown()
    server.server_close()
    thread.join()


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
    entered = threading.Semaphore(0)
    release = threading.Event()
    constructed = []

    class SlowTransport:
        def __init__(self, url, headers, timeout):
            self.url = url
            self.account = mcp_client.current_account_id()
            self.identity = f"recipient-{len(constructed)}"
            self.created_at = time.monotonic()
            self.closed = threading.Event()
            constructed.append(self)

        def exchange(self, method, *args, **kwargs):
            if method == "initialize":
                entered.release()
                release.wait(5)
                return {"protocolVersion": "2024-11-05"}
            return None

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
    assert results == ["rejected"]
    assert len(constructed) == 2

    release.set()
    for worker in workers:
        worker.join(timeout = 2)
        assert not worker.is_alive()
    assert len(results) == 3
    assert mcp_client._private_recipient_connects == {}


def test_recipient_capacity_is_isolated_per_account(monkeypatch):
    other = SimpleNamespace(
        account = "other",
        created_at = time.monotonic(),
        closed = threading.Event(),
        close = lambda: None,
    )
    mcp_client._private_recipients["other"] = other
    monkeypatch.setattr(mcp_client, "_DEFAULT_MAX_SESSIONS", 1)
    monkeypatch.setattr(mcp_client, "current_account_id", lambda: "mine")

    reservation = mcp_client._reserve_private_recipient()
    assert reservation == "mine"
    mcp_client._release_private_recipient(reservation)


def test_private_tool_call_can_be_explicitly_unbounded(monkeypatch):
    transport = mcp_client._PrivateMcpTransport("http://example.test/mcp", {}, 0.01)

    class Response:
        status = 200

        def getheader(self, name, default = None):
            return "application/json" if name == "Content-Type" else default

        def read(self, _limit):
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"protocolVersion": "2024-11-05"},
                }
            ).encode()

        def close(self):
            pass

    class Connection:
        sock = None

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            time.sleep(0.05)
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(transport, "_connection", lambda _: Connection())
    try:
        assert transport.exchange("initialize", {}, timeout = None)["protocolVersion"] == "2024-11-05"
    finally:
        transport.close()


def test_http_send_is_one_use_and_withholds_all_results(http_recipient, caplog, monkeypatch):
    url, calls = http_recipient
    cancel = threading.Event()
    initialization_events = []
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
    assert len(calls) == 1
    assert calls[0]["params"]["arguments"]["picture"] == ENCODED
    assert result == PRIVATE_CALL_COMPLETE and ENCODED not in result + caplog.text
    assert (
        mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC, disclosure_context = context)
        == PRIVATE_CALL_ERROR
    )
    assert len(calls) == 1


def test_private_tool_error_returns_fixed_error(monkeypatch):
    url = "http://recipient.test/mcp"
    context = make_context()

    def exchange(*args, **kwargs):
        kwargs["context"].commit_at_send(context.recipient)
        return {
            "content": [{"type": "text", "text": "private server detail"}],
            "isError": True,
        }

    transport = SimpleNamespace(
        _configuration = (url, ()),
        created_at = time.monotonic(),
        exchange = exchange,
        close = lambda: None,
    )
    monkeypatch.setattr(mcp_client, "_private_recipient", lambda *args, **kwargs: transport)

    result = mcp_client.call_tool_sync(
        url,
        None,
        "inspect_picture",
        PUBLIC,
        disclosure_context = context,
        config_check = lambda: True,
    )
    assert result == PRIVATE_CALL_ERROR
    assert "private server detail" not in result


def test_http_revalidates_configuration_in_worker_before_write(monkeypatch):
    transport = mcp_client._PrivateMcpTransport("http://example.test/mcp", {}, 5)
    worker_started = threading.Event()
    release_worker = threading.Event()
    valid = [True]
    writes = []

    class Connection:
        sock = None

        def request(self, *args, **kwargs):
            writes.append((args, kwargs))

        def getresponse(self):
            raise AssertionError("an invalid configuration must not send")

        def close(self):
            pass

    connection = Connection()

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
    assert writes == []
    assert context.spent is False
    revoker.join(timeout = 2)
    assert not revoker.is_alive()


def test_http_cancel_returns_while_connection_setup_is_blocked(monkeypatch):
    transport = mcp_client._PrivateMcpTransport("http://example.test/mcp", {}, 5)
    entered = threading.Event()
    release = threading.Event()
    worker_done = threading.Event()
    writes = []

    class Connection:
        sock = None

        def request(self, *args, **kwargs):
            writes.append((args, kwargs))

        def close(self):
            worker_done.set()

    def blocked_connection(_deadline):
        entered.set()
        release.wait(5)
        return Connection()

    cancel = threading.Event()
    monkeypatch.setattr(transport, "_connection", blocked_connection)
    threading.Thread(target = lambda: entered.wait(2) and cancel.set(), daemon = True).start()
    started = time.monotonic()
    try:
        with pytest.raises(Exception):
            transport.exchange("initialize", {}, cancel_event = cancel)
        assert cancel.is_set()
        assert time.monotonic() - started < 2
    finally:
        release.set()
        assert worker_done.wait(2)
    assert writes == []


def test_http_initialize_cancel_interrupts_blocked_response(monkeypatch):
    transport = mcp_client._PrivateMcpTransport("http://example.test/mcp", {}, 30)
    left, right = socket.socketpair()

    class Response:
        status = 200

        def getheader(
            self,
            name,
            default = None,
        ):
            return "application/json" if name == "Content-Type" else default

        def read(self, *_):
            return left.recv(1)

    class Connection:
        sock = left

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            left.close()

    monkeypatch.setattr(
        transport,
        "_connection",
        lambda _: setattr(transport, "http", Connection()) or transport.http,
    )
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(Exception):
        transport.exchange("initialize", {}, cancel_event = cancel)
    assert cancel.is_set()
    assert time.monotonic() - started < 2
    right.close()


def test_http_cancel_interrupts_connection_close_response_body():
    headers_sent = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b'{"jsonrpc":"2.0","id":1,"result":{}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            headers_sent.set()
            release.wait(5)
            try:
                self.wfile.write(body)
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    cancel = threading.Event()
    transport = mcp_client._PrivateMcpTransport(f"http://127.0.0.1:{server.server_port}/mcp", {}, 5)
    timer = threading.Thread(target = lambda: headers_sent.wait(2) and cancel.set(), daemon = True)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(Exception):
            transport.exchange("initialize", {}, cancel_event = cancel)
        assert cancel.is_set()
        assert time.monotonic() - started < 2
    finally:
        release.set()
        transport.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_stdio_send_redacts_results_and_side_channels(tmp_path, monkeypatch, capfd):
    script = tmp_path / "server.py"
    script.write_text(
        """import json, sys
for line in sys.stdin:
    message = json.loads(line)
    if message['method'] == 'notifications/initialized': continue
    if message['method'] == 'initialize': result = {'protocolVersion': '2024-11-05', 'capabilities': {}}
    else:
        image = message['params']['arguments']['picture']
        print(image, file=sys.stderr, flush=True)
        print(json.dumps({'jsonrpc':'2.0','method':'notifications/message','params':{'data':image}}), flush=True)
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
        disclosure_context = make_context(recipient = identity, tool_name = "classify_frame"),
        timeout = 5,
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
        "image_input_mappings_json": json.dumps(
            [{"tool": "inspect_picture", "field": "picture", "encoding": "base64"}]
        ),
        "config_revision": 1,
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
        assert tools.execute_tool("mcp__server-1__inspect_picture", {"threshold": 0.5}) == (
            "ordinary result"
        )
