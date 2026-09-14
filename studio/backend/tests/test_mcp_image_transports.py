# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core.inference import mcp_client
from core.inference.mcp_image_redaction import PRIVATE_CALL_ERROR, REDACTED_IMAGE
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
                result = {
                    "content": [
                        {"type": "text", "text": "safe label"},
                        {"type": "image", "data": image, "mimeType": "image/png"},
                    ],
                    "structuredContent": {"echo": image},
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
    for recipient in recipients:
        recipient.close()


def test_http_send_is_one_use_and_redacts_echoes(http_recipient, caplog, monkeypatch):
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
    assert REDACTED_IMAGE in result and ENCODED not in result + caplog.text
    assert (
        mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC, disclosure_context = context)
        == PRIVATE_CALL_ERROR
    )
    assert len(calls) == 1


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
    assert "safe stdio label" in result and REDACTED_IMAGE in result
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
