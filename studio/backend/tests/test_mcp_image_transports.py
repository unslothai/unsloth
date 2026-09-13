# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core.inference import mcp_client
from core.inference.mcp_image_redaction import PRIVATE_CALL_ERROR, REDACTED_IMAGE
from studio.backend.tests.test_mcp_image_redaction import ENCODED, PUBLIC, make_context


@pytest.fixture
def http_recipient():
    calls = []
    state = {"echo": True, "redirect": False, "events": False, "disconnect": False, "rpc_error": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = message["method"]
            if method == "notifications/initialized":
                self.send_response(202)
                self.end_headers()
                return
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05", "capabilities": {},
                          "serverInfo": {"name": "test", "version": "1"}}
            else:
                calls.append(message)
                if state["disconnect"]:
                    self.connection.close()
                    return
                if state["redirect"]:
                    self.send_response(307)
                    self.send_header("Location", "/unapproved")
                    self.end_headers()
                    return
                image = message["params"]["arguments"]["picture"]
                result = {"content": [{"type": "text", "text": "safe label"},
                                      {"type": "image", "data": image, "mimeType": "image/png"}],
                          "structuredContent": {"echo": image}}
            response = {"jsonrpc": "2.0", "id": message["id"], "result": result}
            if state["rpc_error"] and method == "tools/call":
                response = {"jsonrpc": "2.0", "id": message["id"],
                            "error": {"code": -32603, "message": ENCODED}}
            data = json.dumps(response).encode()
            if state["events"] and method == "tools/call":
                # Logging/sampling side channels are swallowed before the result.
                data = (b'data: {"jsonrpc":"2.0","method":"notifications/message","params":'
                        + json.dumps({"data": ENCODED}).encode() + b'}\n\n'
                        + b"data: " + data + b"\n\n")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if state["events"] and method == "tools/call"
                             else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Mcp-Session-Id", "fixed-session")
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/mcp", calls, state
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


@pytest.mark.parametrize("events", [False, True])
def test_real_http_private_send_is_exact_one_use_and_redacted(http_recipient, events, caplog):
    url, calls, state = http_recipient
    state["events"] = events
    identity = mcp_client.prepare_mcp_image_recipient(url)
    assert mcp_client.mcp_image_recipient_location(identity) == url
    commits = []
    context = make_context(recipient = identity, commit = lambda value: commits.append(value) is None)
    result = mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC,
                                      disclosure_context = context, config_check = lambda: True)
    assert len(calls) == 1
    assert calls[0]["params"]["arguments"]["picture"] == ENCODED
    assert calls[0]["params"]["arguments"]["options"] == PUBLIC["options"]
    assert PUBLIC["picture"].startswith("mcp-image-ref-")
    assert commits == [identity]
    assert "safe label" in result and REDACTED_IMAGE in result
    assert ENCODED not in result and ENCODED not in caplog.text
    retry = mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC, disclosure_context = context)
    assert retry.startswith("Error:") and len(calls) == 1


def test_http_redirect_never_replays_image(http_recipient):
    url, calls, state = http_recipient
    identity = mcp_client.prepare_mcp_image_recipient(url)
    state["redirect"] = True
    result = mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC,
                                      disclosure_context = make_context(recipient = identity))
    assert result == PRIVATE_CALL_ERROR
    assert len(calls) == 1


def test_socket_closed_at_commit_cannot_implicitly_reconnect(http_recipient):
    url, calls, state = http_recipient
    identity = mcp_client.prepare_mcp_image_recipient(url)
    transport = mcp_client._private_recipients[identity]

    def commit(recipient):
        transport.http.close()
        return True

    result = mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC,
                                      disclosure_context = make_context(recipient = identity, commit = commit))
    assert result == PRIVATE_CALL_ERROR
    assert calls == []


@pytest.mark.parametrize("failure", ["disconnect", "rpc_error"])
def test_unknown_delivery_and_error_echoes_are_fixed_and_never_retried(http_recipient, caplog, failure):
    url, calls, state = http_recipient
    identity = mcp_client.prepare_mcp_image_recipient(url)
    state[failure] = True
    result = mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC,
                                      disclosure_context = make_context(recipient = identity), timeout = 3)
    assert result == PRIVATE_CALL_ERROR
    assert len(calls) == 1
    assert ENCODED not in caplog.text


@pytest.mark.parametrize("header", ["Host", "Content-Length", "Transfer-Encoding", "Mcp-Session-Id"])
def test_private_http_cannot_override_destination_or_framing(http_recipient, header):
    url, calls, state = http_recipient
    with pytest.raises(mcp_client._PrivateTransportUnavailable):
        mcp_client.prepare_mcp_image_recipient(url, {header: "override"})
    assert calls == []


@pytest.mark.parametrize("revocation", ["config", "consent", "recipient", "arguments", "cancel", "tool"])
def test_revocations_prevent_actual_http_invocation(http_recipient, revocation):
    url, calls, state = http_recipient
    identity = mcp_client.prepare_mcp_image_recipient(url)
    context = make_context(recipient = identity, commit = lambda _: revocation != "consent")
    cancel = threading.Event()
    if revocation == "cancel":
        cancel.set()
    result = mcp_client.call_tool_sync(
        url + ("?changed=1" if revocation == "recipient" else ""), None,
        "another_tool" if revocation == "tool" else "inspect_picture",
        {**PUBLIC, "options": {"limit": 3}} if revocation == "arguments" else PUBLIC,
        disclosure_context = context, config_check = lambda: revocation != "config", cancel_event = cancel,
    )
    assert result.startswith("Error:")
    assert calls == []


def test_http_config_revoked_while_connecting_prevents_write(http_recipient, monkeypatch):
    url, calls, state = http_recipient
    identity = mcp_client.prepare_mcp_image_recipient(url)
    reached = threading.Event()
    release = threading.Event()
    current = threading.Event()
    current.set()
    original = mcp_client._PrivateMcpTransport._connection

    def paused(self, deadline):
        result = original(self, deadline)
        reached.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(mcp_client._PrivateMcpTransport, "_connection", paused)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(mcp_client.call_tool_sync, url, None, "inspect_picture", PUBLIC,
                             disclosure_context = make_context(recipient = identity), config_check = current.is_set)
        assert reached.wait(5)
        current.clear()
        release.set()
        assert future.result(5).startswith("Error:")
    assert calls == []


def test_racing_private_calls_have_one_physical_invocation(http_recipient):
    url, calls, state = http_recipient
    identity = mcp_client.prepare_mcp_image_recipient(url)
    barrier = threading.Barrier(2)

    def call():
        barrier.wait()
        return mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC,
                                         disclosure_context = make_context(recipient = identity))

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: call(), range(2)))
    assert len(calls) == 1
    assert sum("safe label" in result for result in results) == 1


def test_sanitizer_failure_withholds_entire_http_result(http_recipient, monkeypatch):
    url, calls, state = http_recipient
    identity = mcp_client.prepare_mcp_image_recipient(url)
    context = make_context(recipient = identity)

    def fail(result):
        raise RuntimeError(ENCODED)

    monkeypatch.setattr(context, "redact_result", fail)
    result = mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC, disclosure_context = context)
    assert result == PRIVATE_CALL_ERROR
    assert len(calls) == 1


def test_real_stdio_private_send_suppresses_stderr_notifications_and_echoes(tmp_path, monkeypatch, capfd):
    script = tmp_path / "server.py"
    script.write_text('''import json, sys
for line in sys.stdin:
    message = json.loads(line)
    if message['method'] == 'notifications/initialized':
        continue
    if message['method'] == 'initialize':
        result = {'protocolVersion': '2024-11-05', 'capabilities': {}}
    else:
        image = message['params']['arguments']['picture']
        print(image, file=sys.stderr, flush=True)
        print(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/message', 'params': {'data': image}}), flush=True)
        result = {'content': [{'type': 'text', 'text': image[:20]}, {'type': 'text', 'text': image[20:]},
                              {'type': 'text', 'text': 'safe stdio label'}]}
    print(json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result}), flush=True)
''', encoding = "utf-8")
    monkeypatch.setattr(mcp_client, "stdio_mcp_enabled", lambda: True)
    url = mcp_client.join_stdio_command([sys.executable, "-u", str(script)])
    identity = mcp_client.prepare_mcp_image_recipient(url)
    result = mcp_client.call_tool_sync(url, None, "classify_frame", PUBLIC,
                                      disclosure_context = make_context(recipient = identity, tool_name = "classify_frame"), timeout = 5)
    assert REDACTED_IMAGE in result
    assert "safe stdio label" in result
    assert ENCODED not in result
    captured = capfd.readouterr()
    assert ENCODED not in captured.out + captured.err


def test_unsupported_oauth_declines_before_connection(http_recipient):
    url, calls, state = http_recipient
    with pytest.raises(mcp_client._PrivateTransportUnavailable):
        mcp_client.prepare_mcp_image_recipient(url, use_oauth = True)
    assert calls == []


@pytest.mark.parametrize("enabled", [True, False])
def test_direct_execute_cannot_bypass_enabled_mapping_with_raw_payload(monkeypatch, enabled):
    from core.inference import tools
    from storage import studio_db

    server = {"id": "server-1", "url": "https://example.test/mcp", "is_enabled": True,
              "image_input_mappings_json": json.dumps([{"tool": "inspect_picture", "field": "picture", "encoding": "base64"}]),
              "config_revision": 1}
    calls = []
    monkeypatch.setattr(tools.mcp_servers_db, "get_server_for_tool", lambda _: server)
    monkeypatch.setattr(studio_db, "get_chat_setting_with_revision", lambda _: (enabled, "r1"))
    monkeypatch.setattr(tools, "call_tool_sync", lambda **kwargs: calls.append(kwargs) or "ordinary result")
    monkeypatch.setattr(tools, "_fit_result_to_room", lambda result, *args: result)
    arguments = {"picture": ENCODED}
    result = tools.execute_tool("mcp__server-1__inspect_picture", arguments)
    if enabled:
        assert result.startswith("Error:") and calls == []
    else:
        assert result == "ordinary result"
        assert calls[0]["args"] is arguments


def test_server_invalidation_removes_prepared_recipient(http_recipient):
    url, calls, state = http_recipient
    identity = mcp_client.prepare_mcp_image_recipient(url)
    mcp_client.close_mcp_sessions(url)
    result = mcp_client.call_tool_sync(url, None, "inspect_picture", PUBLIC,
                                      disclosure_context = make_context(recipient = identity))
    assert result.startswith("Error:")
    assert calls == []
