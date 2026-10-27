# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Claude Desktop profile switching and the loopback gateway, without a real Claude or Studio."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import pytest

from utils import claude_desktop as cda

MODEL_ID = "unsloth/gemma-4-E2B-it-GGUF"
UPSTREAM_KEY = "sk-unsloth-a99a99a99a99"
TOKEN = "unsloth-claude-test-token"
IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
MMPROJ_500 = {
    "type": "error",
    "error": {
        "type": "api_error",
        "message": '{"error":{"code":500,"message":"image input is not supported - hint: if '
        'this is unexpected, you may need to provide the mmproj","type":"server_error"}}',
    },
}
RAW_LLAMA_500 = {
    "error": {
        "code": 500,
        "message": "image input is not supported - hint: you may need to provide the mmproj",
        "type": "server_error",
    }
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")


# -- paths -------------------------------------------------------------------------------------


def test_macos_paths_match_claude_desktop(tmp_path):
    targets = cda.target_paths("darwin", tmp_path)
    support = tmp_path / "Library" / "Application Support"
    assert targets.normal_configs == [support / "Claude" / "claude_desktop_config.json"]
    (third,) = targets.third_party
    assert third.desktop_config == support / "Claude-3p" / "claude_desktop_config.json"
    assert third.meta == support / "Claude-3p" / "configLibrary" / "_meta.json"
    assert third.profile == support / "Claude-3p" / "configLibrary" / f"{cda.PROFILE_ID}.json"


def test_windows_paths_cover_claude_and_claude_nest(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    targets = cda.target_paths("win32", tmp_path)
    local = tmp_path / "Local"
    assert targets.normal_configs == [
        local / "Claude" / "claude_desktop_config.json",
        local / "Claude Nest" / "claude_desktop_config.json",
    ]
    assert [t.profile.parent.parent for t in targets.third_party] == [
        local / "Claude-3p",
        local / "Claude Nest-3p",
    ]


def test_windows_falls_back_to_userprofile(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising = False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "me"))
    assert cda.local_app_data() == tmp_path / "me" / "AppData" / "Local"


def test_other_platforms_are_refused(tmp_path):
    with pytest.raises(RuntimeError):
        cda.target_paths("linux", tmp_path)
    assert not cda.can_launch("linux")
    assert not cda.can_launch("win32")
    assert cda.can_launch("darwin")


def test_app_is_found_in_the_user_applications_folder(tmp_path):
    assert cda.app_path("darwin", tmp_path) in (None, Path("/Applications/Claude.app"))
    (tmp_path / "Applications" / "Claude.app").mkdir(parents = True)
    found = cda.app_path("darwin", tmp_path)
    assert found in (Path("/Applications/Claude.app"), tmp_path / "Applications" / "Claude.app")


# -- profile switch and restore ----------------------------------------------------------------


def _writer(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")


def test_profile_has_the_same_keys_as_ollama():
    values = cda.profile_values("http://127.0.0.1:5", TOKEN)
    assert values == {
        "inferenceProvider": "gateway",
        "inferenceGatewayBaseUrl": "http://127.0.0.1:5",
        "inferenceGatewayApiKey": TOKEN,
        "inferenceGatewayAuthScheme": "bearer",
        "deploymentDisplayName": "Unsloth",
        "chatTabEnabled": True,
        "disableDeploymentModeChooser": True,
        "coworkEgressAllowedHosts": ["*"],
        "disableEssentialTelemetry": True,
        "disableNonessentialTelemetry": True,
        "autoModeEnabled": True,
    }
    assert cda.PROFILE_DROPPED == ("inferenceModels",)


def test_switch_then_restore_is_byte_identical(tmp_path):
    targets = cda.target_paths("darwin", tmp_path)
    (third,) = targets.third_party
    normal = targets.normal_configs[0]
    user_normal = '{\n    "mcpServers": {"fs": {"command": "npx"}},\n    "deploymentMode": "1p"\n}'
    user_meta = json.dumps({"appliedId": "theirs", "entries": [{"id": "theirs", "name": "Corp"}]})
    _write(normal, user_normal)
    _write(third.meta, user_meta)
    files = cda.snapshot(targets)

    cda.apply_profile(targets, files, "http://127.0.0.1:4242", TOKEN, _writer)

    assert json.loads(normal.read_text())["deploymentMode"] == "3p"
    assert json.loads(normal.read_text())["mcpServers"] == {"fs": {"command": "npx"}}
    assert json.loads(third.desktop_config.read_text()) == {"deploymentMode": "3p"}
    meta = json.loads(third.meta.read_text())
    assert meta["appliedId"] == cda.PROFILE_ID
    assert meta["entries"] == [
        {"id": "theirs", "name": "Corp"},
        {"id": cda.PROFILE_ID, "name": "Unsloth"},
    ]
    profile = json.loads(third.profile.read_text())
    assert profile["inferenceGatewayBaseUrl"] == "http://127.0.0.1:4242"
    assert profile["inferenceProvider"] == "gateway"

    # The record survives a JSON round trip, as it does through the state file.
    files = json.loads(json.dumps(files))
    assert cda.restore_profile(files, _writer) == []

    assert normal.read_text() == user_normal
    assert third.meta.read_text() == user_meta
    assert not third.profile.exists()
    assert not third.desktop_config.exists()


def test_restore_puts_back_a_pinned_model_list_and_old_values(tmp_path):
    targets = cda.target_paths("darwin", tmp_path)
    (third,) = targets.third_party
    old = {
        "inferenceModels": ["claude-x"],
        "autoModeEnabled": False,
        "inferenceProvider": "bedrock",
    }
    _write(third.profile, json.dumps(old))
    files = cda.snapshot(targets)
    cda.apply_profile(targets, files, "http://127.0.0.1:1", TOKEN, _writer)
    assert "inferenceModels" not in json.loads(third.profile.read_text())
    cda.restore_profile(files, _writer)
    assert json.loads(third.profile.read_text()) == old


def test_restore_keeps_a_key_the_user_changed_meanwhile(tmp_path):
    targets = cda.target_paths("darwin", tmp_path)
    normal = targets.normal_configs[0]
    _write(normal, '{"deploymentMode": "1p", "theme": "dark"}')
    files = cda.snapshot(targets)
    cda.apply_profile(targets, files, "http://127.0.0.1:1", TOKEN, _writer)
    changed = json.loads(normal.read_text())
    changed["deploymentMode"] = "custom"
    changed["theme"] = "light"
    normal.write_text(json.dumps(changed))

    notes = cda.restore_profile(files, _writer)

    assert json.loads(normal.read_text()) == {"deploymentMode": "custom", "theme": "light"}
    assert any("deploymentMode" in note and str(normal) in note for note in notes)


def test_a_symlinked_config_is_refused_before_anything_is_written(tmp_path):
    targets = cda.target_paths("darwin", tmp_path)
    normal = targets.normal_configs[0]
    real = tmp_path / "elsewhere.json"
    real.write_text("{}")
    normal.parent.mkdir(parents = True)
    normal.symlink_to(real)
    with pytest.raises(ValueError, match = "symlink"):
        cda.snapshot(targets)


def test_an_unparseable_config_is_refused(tmp_path):
    targets = cda.target_paths("darwin", tmp_path)
    _write(targets.normal_configs[0], "[1, 2]")
    with pytest.raises(ValueError):
        cda.snapshot(targets)


# -- image retry, as in maxkongerskov/ollama 92bbeda ------------------------------------------


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (500, MMPROJ_500, True),
        (500, RAW_LLAMA_500, True),
        (
            400,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "model does not support image input",
                },
            },
            True,
        ),
        (
            400,
            {
                "type": "error",
                "error": {"type": "api_error", "message": "model does not support image input"},
            },
            False,
        ),
        (
            500,
            {"type": "error", "error": {"type": "api_error", "message": "CUDA out of memory"}},
            False,
        ),
        (500, {"error": {"type": "server_error", "message": "context shift disabled"}}, False),
        (
            500,
            {
                "type": "error",
                "error": {"type": "api_error", "message": "vision is unsupported here"},
            },
            True,
        ),
        (502, MMPROJ_500, True),
        (500, "not json", False),
    ],
)
def test_which_errors_mean_the_model_cannot_see_images(status, body, expected):
    raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    assert cda.response_rejects_images(status, raw) is expected


def test_images_are_replaced_also_inside_tool_results():
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}, IMAGE]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t", "content": [IMAGE]}],
            },
            {"role": "assistant", "content": "plain"},
        ]
    }
    assert cda.replace_unsupported_images(payload)
    notice = {"type": "text", "text": cda.UNSUPPORTED_IMAGE_NOTICE}
    assert payload["messages"][0]["content"][1] == notice
    assert payload["messages"][1]["content"][0]["content"] == [notice]
    assert not cda.replace_unsupported_images({"messages": [{"role": "user", "content": "x"}]})


# -- the gateway against a fake Unsloth ------------------------------------------------------


class _Upstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.requests = []
        self.replies = []
        super().__init__(("127.0.0.1", 0), _UpstreamHandler)
        threading.Thread(target = self.serve_forever, args = (0.05,), daemon = True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, dict(self.headers), json.loads(body)))
        status, reply = self.server.replies.pop(0) if self.server.replies else (200, {"ok": 1})
        if status == "raw-sse":
            # Events given verbatim, so a test can send Studio's in-band error shapes.
            data = b"".join(reply)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if status == "sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for event in reply:
                data = f"event: {event}\ndata: {{}}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            return
        data = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def gateway():
    upstream = _Upstream()
    gw = cda.Gateway(
        upstream.url,
        UPSTREAM_KEY,
        cda.gateway_model({"id": MODEL_ID, "context_length": 32768}),
        TOKEN,
    ).start()
    yield gw, upstream
    gw.stop()
    upstream.shutdown()
    upstream.server_close()


def _call(
    gw,
    method,
    path,
    body = None,
    headers = None,
):
    conn = http.client.HTTPConnection("127.0.0.1", gw.server_address[1], timeout = 10)
    sent = {"Authorization": f"Bearer {TOKEN}", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, data, sent)
    response = conn.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def test_health_identifies_the_gateway(gateway):
    gw, _ = gateway
    assert cda.probe(gw.base_url)
    status, headers, _ = _call(gw, "GET", cda.HEALTH_PATH, headers = {"Authorization": ""})
    assert status == 204 and headers[cda.HEALTH_HEADER] == "1"


def test_catalog_has_the_anthropic_shape(gateway):
    gw, _ = gateway
    status, _, body = _call(gw, "GET", "/v1/models")
    assert status == 200
    catalog = json.loads(body)
    assert catalog["first_id"] == catalog["last_id"] == "claude-fable-5"
    assert catalog["has_more"] is False
    assert catalog["data"] == [
        {
            "id": "claude-fable-5",
            "type": "model",
            "display_name": MODEL_ID,
            "created_at": "2026-06-09T00:00:00Z",
            "max_tokens": 32768,
            "anthropic_family_tier": "fable",
            "is_family_default": True,
        }
    ]


def test_requests_need_the_gateway_key(gateway):
    gw, upstream = gateway
    status, _, body = _call(gw, "GET", "/v1/models", headers = {"Authorization": "Bearer nope"})
    assert status == 401 and json.loads(body)["error"]["type"] == "authentication_error"
    status, _, _ = _call(
        gw,
        "POST",
        "/v1/messages",
        {"model": "claude-fable-5"},
        {"Authorization": "", "x-api-key": TOKEN},
    )
    assert status == 200
    assert len(upstream.requests) == 1


def test_web_pages_and_rebinding_are_refused(gateway):
    gw, upstream = gateway
    assert _call(gw, "GET", "/v1/models", headers = {"Origin": "https://evil.example"})[0] == 403
    assert _call(gw, "GET", "/v1/models", headers = {"Host": "evil.example"})[0] == 403
    assert _call(gw, "GET", "/v1/models", headers = {"Host": "127.0.0.1:1"})[0] == 403
    assert _call(gw, "GET", "/anything")[0] == 404
    assert upstream.requests == []


def test_messages_are_routed_to_the_loaded_model_with_the_unsloth_key(gateway):
    gw, upstream = gateway
    status, _, body = _call(
        gw,
        "POST",
        "/v1/messages",
        {"model": "claude-fable-5", "max_tokens": 5, "messages": []},
        {"Cookie": "a=b", "Anthropic-Version": "2023-06-01"},
    )
    assert status == 200 and json.loads(body) == {"ok": 1}
    path, headers, sent = upstream.requests[0]
    assert path == "/v1/messages"
    assert sent["model"] == MODEL_ID
    assert headers["Authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert "Cookie" not in headers
    assert headers["Anthropic-Version"] == "2023-06-01"
    # Studio scopes the Desktop-only prompt handling to requests carrying this marker.
    assert headers[cda.GATEWAY_CLIENT_HEADER] == cda.GATEWAY_CLIENT_VALUE


def test_count_tokens_goes_to_unsloth(gateway):
    gw, upstream = gateway
    upstream.replies.append((200, {"input_tokens": 12}))
    status, _, body = _call(
        gw, "POST", "/v1/messages/count_tokens", {"model": "claude-fable-5", "messages": []}
    )
    assert status == 200 and json.loads(body) == {"input_tokens": 12}
    assert upstream.requests[0][0] == "/v1/messages/count_tokens"
    assert upstream.requests[0][2]["model"] == MODEL_ID


def test_an_unknown_model_is_an_anthropic_error(gateway):
    gw, upstream = gateway
    status, _, body = _call(gw, "POST", "/v1/messages", {"model": "claude-opus-5"})
    assert status == 400
    assert json.loads(body)["error"]["message"] == 'unknown Claude model "claude-opus-5"'
    assert _call(gw, "POST", "/v1/messages", {"messages": []})[0] == 400
    assert upstream.requests == []


def test_streaming_passes_through(gateway):
    gw, upstream = gateway
    upstream.replies.append(("sse", ["message_start", "content_block_delta", "message_stop"]))
    status, headers, body = _call(
        gw, "POST", "/v1/messages", {"model": "claude-fable-5", "stream": True}
    )
    assert status == 200
    assert headers["Content-Type"] == "text/event-stream"
    assert body.count(b"event: ") == 3


@pytest.mark.parametrize("error", [MMPROJ_500, RAW_LLAMA_500])
def test_a_missing_projector_500_retries_without_images(gateway, error):
    gw, upstream = gateway
    upstream.replies += [(500, error), (200, {"ok": "retried"})]
    request = {
        "model": "claude-fable-5",
        "messages": [{"role": "user", "content": [IMAGE, {"type": "text", "text": "?"}]}],
    }
    status, _, body = _call(gw, "POST", "/v1/messages", request)
    assert status == 200 and json.loads(body) == {"ok": "retried"}
    assert len(upstream.requests) == 2
    assert upstream.requests[0][2]["messages"][0]["content"][0] == IMAGE
    retried = upstream.requests[1][2]["messages"][0]["content"]
    assert retried[0] == {"type": "text", "text": cda.UNSUPPORTED_IMAGE_NOTICE}


def test_a_text_only_400_also_retries(gateway):
    gw, upstream = gateway
    rejected = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "this model does not support images"},
    }
    upstream.replies += [(400, rejected), (200, {"ok": "retried"})]
    request = {"model": "claude-fable-5", "messages": [{"role": "user", "content": [IMAGE]}]}
    assert json.loads(_call(gw, "POST", "/v1/messages", request)[2]) == {"ok": "retried"}


def test_an_unrelated_500_is_returned_as_is(gateway):
    gw, upstream = gateway
    oom = {"type": "error", "error": {"type": "api_error", "message": "CUDA out of memory"}}
    upstream.replies.append((500, oom))
    request = {"model": "claude-fable-5", "messages": [{"role": "user", "content": [IMAGE]}]}
    status, _, body = _call(gw, "POST", "/v1/messages", request)
    assert status == 500 and json.loads(body) == oom
    assert len(upstream.requests) == 1


def test_a_projector_error_without_images_is_not_retried(gateway):
    gw, upstream = gateway
    upstream.replies.append((500, MMPROJ_500))
    status, _, body = _call(
        gw,
        "POST",
        "/v1/messages",
        {"model": "claude-fable-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 500 and json.loads(body) == MMPROJ_500
    assert len(upstream.requests) == 1


def test_unsloth_down_is_a_502(gateway):
    gw, upstream = gateway
    upstream.shutdown()
    upstream.server_close()
    assert _call(gw, "POST", "/v1/messages", {"model": "claude-fable-5"})[0] == 502


def test_max_tokens_follows_a_small_context():
    assert cda.gateway_model({"id": "m", "context_length": 8192}).max_tokens == 8192
    assert cda.gateway_model({"id": "m", "context_length": 262144}).max_tokens == 64000
    assert cda.gateway_model({"id": "m"}).max_tokens == 64000


# -- a gateway that follows the loaded model (Studio) ------------------------------------------


@pytest.fixture
def following_gateway():
    upstream = _Upstream()
    loaded = {"model": None}
    gw = cda.Gateway(upstream.url, UPSTREAM_KEY, lambda: loaded["model"], TOKEN).start()
    yield gw, upstream, loaded
    gw.stop()
    upstream.shutdown()
    upstream.server_close()


def test_a_following_gateway_routes_to_whatever_is_loaded_now(following_gateway):
    gw, upstream, loaded = following_gateway
    body = {"model": "claude-fable-5", "max_tokens": 8, "messages": []}
    status, _, reply = _call(gw, "POST", "/v1/messages", body)
    reply = json.loads(reply)
    assert status == 200 and reply["content"][0]["text"] == cda.NO_MODEL_MESSAGE
    assert reply["stop_reason"] == "end_turn" and reply["model"] == "claude-fable-5"
    assert upstream.requests == []
    status, _, catalog = _call(gw, "GET", "/v1/models")
    assert status == 200 and json.loads(catalog)["first_id"] == "claude-fable-5"

    loaded["model"] = cda.gateway_model({"id": "unsloth/a-GGUF"})
    assert _call(gw, "POST", "/v1/messages", body)[0] == 200
    loaded["model"] = cda.gateway_model({"id": "unsloth/b-GGUF", "context_length": 4096})
    assert _call(gw, "POST", "/v1/messages", body)[0] == 200
    assert [r[2]["model"] for r in upstream.requests] == ["unsloth/a-GGUF", "unsloth/b-GGUF"]
    assert json.loads(_call(gw, "GET", "/v1/models")[2])["data"][0]["max_tokens"] == 4096


def test_a_broken_model_lookup_reads_as_no_model(following_gateway):
    gw, upstream, _ = following_gateway
    gw.model = lambda: 1 / 0
    body = {"model": "claude-fable-5", "max_tokens": 8, "messages": []}
    status, _, reply = _call(gw, "POST", "/v1/messages", body)
    assert status == 200 and json.loads(reply)["content"][0]["text"] == cda.NO_MODEL_MESSAGE
    assert upstream.requests == []


def test_a_loopback_https_upstream_skips_certificate_checks():
    gw = cda.Gateway("https://127.0.0.1:8443", UPSTREAM_KEY, lambda: None, TOKEN)
    try:
        conn = gw.connection()
        assert conn._context.verify_mode == cda.ssl.CERT_NONE
    finally:
        gw.server_close()
    gw = cda.Gateway("https://studio.example.com", UPSTREAM_KEY, lambda: None, TOKEN)
    try:
        assert gw.connection()._context.verify_mode == cda.ssl.CERT_REQUIRED
    finally:
        gw.server_close()


# -- the shared switch record -------------------------------------------------------------------

USER_CONFIG = '{\n  "mcpServers": {"fs": {"command": "npx"}}\n}\n'


def test_begin_switch_and_switch_back_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(cda, "is_running", lambda: False)
    targets = cda.target_paths("darwin", tmp_path)
    _write(targets.normal_configs[0], USER_CONFIG)
    targets.normal_configs[0].chmod(0o640)
    state_file = cda.state_path(tmp_path / "agents")

    state = cda.begin(
        state_file, targets, {"gateway": "http://127.0.0.1:1", "key_id": 3}, {"pid": 1}
    )
    assert cda.read_state(state_file)["key_id"] == 3
    assert oct(state_file.stat().st_mode & 0o777) == "0o600"
    backup = targets.normal_configs[0].with_name(cda.CONFIG_NAME + cda.BACKUP_SUFFIX)
    assert backup.read_text() == USER_CONFIG
    cda.switch(state_file, state, targets, "tok")
    assert json.loads(targets.normal_configs[0].read_text())["deploymentMode"] == "3p"
    assert "wrote" in cda.read_state(state_file)["files"][str(targets.normal_configs[0])]

    assert cda.switch_back(cda.read_state(state_file)) == []
    assert targets.normal_configs[0].read_text() == USER_CONFIG
    assert oct(targets.normal_configs[0].stat().st_mode & 0o777) == "0o640"
    assert not backup.exists()
    assert not targets.third_party[0].profile.exists()


def test_switch_back_quits_an_open_claude_first(tmp_path, monkeypatch):
    seen = {"running": True, "timeouts": []}
    monkeypatch.setattr(cda, "is_running", lambda: seen["running"])

    def quit_app(timeout = 30.0):
        seen["timeouts"].append(timeout)
        seen["running"] = False

    monkeypatch.setattr(cda, "quit_app", quit_app)
    notes = cda.switch_back({"files": {}}, quit_timeout = 2.5)
    assert seen["timeouts"] == [2.5]
    assert "Quit Claude Desktop" in notes[0]


def test_abort_undoes_a_half_done_switch(tmp_path):
    targets = cda.target_paths("darwin", tmp_path)
    _write(targets.normal_configs[0], USER_CONFIG)
    state_file = cda.state_path(tmp_path / "agents")
    state = cda.begin(state_file, targets, {"gateway": "http://127.0.0.1:1"}, {"pid": 1})
    cda.switch(state_file, state, targets, "tok")
    cda.abort(state_file, state)
    assert targets.normal_configs[0].read_text() == USER_CONFIG
    assert not state_file.exists()
    assert not targets.normal_configs[0].with_name(cda.CONFIG_NAME + cda.BACKUP_SUFFIX).exists()


def test_write_file_refuses_a_symlink(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ValueError, match = "symlink"):
        cda.write_file(link, "x")
    assert real.read_text() == "{}"


def test_read_state_ignores_missing_empty_and_broken_records(tmp_path):
    path = tmp_path / "s.json"
    assert cda.read_state(path) is None
    path.write_text("{nope")
    assert cda.read_state(path) is None
    path.write_text('{"files": {}}')
    assert cda.read_state(path) is None
    path.write_text('{"files": {"a": {"original": null}}}')
    assert cda.read_state(path) == {"files": {"a": {"original": None}}}


def test_owner_alive_tells_a_reused_pid_apart():
    me = cda.owner_record("studio")
    assert me["kind"] == "studio"
    assert cda.owner_alive(me)
    if me["started"] is not None:
        assert not cda.owner_alive({**me, "started": me["started"] - 3600})
    assert not cda.owner_alive({"pid": 2**22 + 12345, "started": None})
    assert not cda.owner_alive(None)


def test_no_model_streams_a_turn_that_says_so(following_gateway):
    gw, upstream, _ = following_gateway
    body = {"model": "claude-fable-5", "max_tokens": 8, "messages": [], "stream": True}
    status, headers, reply = _call(gw, "POST", "/v1/messages", body)
    assert status == 200 and headers["Content-Type"] == "text/event-stream"
    events = [
        json.loads(line[len("data: ") :])
        for line in reply.decode().splitlines()
        if line.startswith("data: ")
    ]
    assert [e["type"] for e in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2]["delta"]["text"] == cda.NO_MODEL_MESSAGE
    assert events[4]["delta"]["stop_reason"] == "end_turn"
    status, _, counted = _call(gw, "POST", "/v1/messages/count_tokens", body)
    assert status == 200 and json.loads(counted)["input_tokens"] > 0
    assert upstream.requests == []


# -- launching Claude: the account's real home and a clean environment ---------------------------


@pytest.mark.skipif(os.name != "posix", reason = "pwd is POSIX-only")
def test_user_home_ignores_a_redirected_home(monkeypatch, tmp_path):
    import pwd

    real = Path(pwd.getpwuid(os.getuid()).pw_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert Path.home() == tmp_path
    assert cda.user_home() == real
    normal, third_party = cda.profile_roots("darwin")
    assert normal == [real / "Library" / "Application Support" / "Claude"]
    assert third_party == [real / "Library" / "Application Support" / "Claude-3p"]
    assert cda.app_candidates("darwin")[1] == real / "Applications" / "Claude.app"


def test_launch_env_keeps_only_what_a_dock_launch_has(tmp_path):
    studio_env = {
        "HOME": "/sandbox/home",
        "PATH": "/studio/venv/bin:/usr/bin",
        "VIRTUAL_ENV": "/studio/venv",
        "HF_HOME": "/models",
        "UNSLOTH_STUDIO_HOME": "/sandbox/studio",
        "PYTHONPATH": "/studio",
        "TMPDIR": "/var/folders/x/T/",
        "LANG": "da_DK.UTF-8",
        "SSH_AUTH_SOCK": "/private/tmp/agent.sock",
    }
    env = cda.launch_env(studio_env, home = tmp_path)
    assert env["HOME"] == str(tmp_path)
    assert env["PATH"] == cda.LAUNCH_PATH
    assert env["TMPDIR"] == "/var/folders/x/T/" and env["LANG"] == "da_DK.UTF-8"
    assert env["SSH_AUTH_SOCK"] == "/private/tmp/agent.sock"
    leaked = (
        set(env) - set(cda.LAUNCH_ENV_PASSTHROUGH) - {"HOME", "PATH", "USER", "LOGNAME", "SHELL"}
    )
    assert not leaked
    assert not {"VIRTUAL_ENV", "HF_HOME", "UNSLOTH_STUDIO_HOME", "PYTHONPATH"} & set(env)


def test_open_and_quit_do_not_pass_studios_environment(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(cda.subprocess, "run", fake_run)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VIRTUAL_ENV", "/studio/venv")
    cda.open_app(Path("/Applications/Claude.app"))
    cda.quit_app(timeout = 1)
    assert [c[0][0] for c in calls] == ["/usr/bin/open", "osascript"]
    for _, kwargs in calls:
        env = kwargs["env"]
        assert env["HOME"] == str(cda.user_home())
        assert "VIRTUAL_ENV" not in env
    if os.name == "posix":
        assert all(kw["env"]["HOME"] != str(tmp_path) for _, kw in calls)


# -- per-slot model mapping, like Ollama's Claude Desktop model settings ----------------------


def test_mappings_keep_known_non_empty_slots_in_claude_order():
    assert cda.normalize_mappings(
        {
            "claude-sonnet-4-6": " unsloth/b ",
            "claude-fable-5": "unsloth/a",
            "claude-opus-5": "",
            "gpt-5": "unsloth/c",
            "claude-haiku-4-5-20251001": 7,
        }
    ) == {"claude-fable-5": "unsloth/a", "claude-sonnet-4-6": "unsloth/b"}
    assert cda.normalize_mappings(None) == {}
    assert [r["id"] for r in cda.route_names()] == list(cda.ROUTE_IDS)
    assert cda.route_names()[0] == {"id": "claude-fable-5", "name": "Fable 5"}


def test_loaded_matches_understands_quant_pins():
    loaded = [{"id": "unsloth/Qwen3.5-4B-GGUF", "quant": "UD-Q4_K_XL"}]
    assert cda.loaded_matches("unsloth/qwen3.5-4b-gguf", loaded)
    assert cda.loaded_matches("unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL", loaded)
    assert not cda.loaded_matches("unsloth/Qwen3.5-4B-GGUF:Q8_0", loaded)
    assert not cda.loaded_matches("unsloth/other", loaded)


@pytest.fixture
def mapped_gateway():
    upstream = _Upstream()
    state = {
        "mappings": {
            "claude-fable-5": cda.FOLLOW_LOADED,
            "claude-opus-5": "unsloth/big-GGUF",
            "claude-sonnet-5": "unsloth/small-GGUF",
            "claude-haiku-4-5-20251001": "unsloth/small-GGUF",
        },
        "loaded": {"unsloth/small-GGUF"},
        "loads": [],
        "fail": None,
    }

    def ensure_loaded(model_id):
        state["loads"].append(model_id)
        if state["fail"]:
            raise RuntimeError(state["fail"])
        state["loaded"] = {model_id}  # "Keep multiple models loaded" off: a load replaces

    gw = cda.Gateway(
        upstream.url,
        UPSTREAM_KEY,
        lambda: cda.gateway_model({"id": next(iter(state["loaded"]))}) if state["loaded"] else None,
        TOKEN,
        mappings = lambda: state["mappings"],
        is_loaded = lambda model_id: model_id in state["loaded"],
        ensure_loaded = ensure_loaded,
    ).start()
    yield gw, upstream, state
    gw.stop()
    upstream.shutdown()
    upstream.server_close()


def test_the_catalog_lists_every_mapped_slot_under_its_model_name(mapped_gateway):
    gw, _, state = mapped_gateway
    catalog = json.loads(_call(gw, "GET", "/v1/models")[2])
    assert [(e["id"], e["display_name"]) for e in catalog["data"]] == [
        ("claude-fable-5", "unsloth/small-GGUF"),
        ("claude-opus-5", "unsloth/big-GGUF"),
        ("claude-sonnet-5", "unsloth/small-GGUF"),
        ("claude-haiku-4-5-20251001", "unsloth/small-GGUF"),
    ]
    assert catalog["first_id"] == "claude-fable-5"
    assert catalog["last_id"] == "claude-haiku-4-5-20251001"
    # Changed while connected: the next catalog read follows.
    state["mappings"] = {"claude-sonnet-4-6": "unsloth/x"}
    catalog = json.loads(_call(gw, "GET", "/v1/models")[2])
    assert [e["id"] for e in catalog["data"]] == ["claude-sonnet-4-6"]


def test_each_slot_goes_to_its_own_model_loading_it_first(mapped_gateway):
    gw, upstream, state = mapped_gateway
    for route in ("claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"):
        body = {"model": route, "max_tokens": 8, "messages": []}
        assert _call(gw, "POST", "/v1/messages", body)[0] == 200
    assert [r[2]["model"] for r in upstream.requests] == [
        "unsloth/small-GGUF",
        "unsloth/big-GGUF",
        "unsloth/small-GGUF",
    ]
    # small was loaded; big replaced it; small came back for haiku.
    assert state["loads"] == ["unsloth/big-GGUF", "unsloth/small-GGUF"]


def test_a_slot_can_be_named_by_the_model_it_shows(mapped_gateway):
    gw, upstream, _ = mapped_gateway
    body = {"model": "unsloth/big-GGUF", "max_tokens": 8, "messages": []}
    assert _call(gw, "POST", "/v1/messages", body)[0] == 200
    assert upstream.requests[-1][2]["model"] == "unsloth/big-GGUF"


def test_an_unmapped_slot_is_unknown(mapped_gateway):
    gw, upstream, _ = mapped_gateway
    body = {"model": "claude-sonnet-4-6", "max_tokens": 8, "messages": []}
    status, _, reply = _call(gw, "POST", "/v1/messages", body)
    assert status == 400 and "claude-sonnet-4-6" in json.loads(reply)["error"]["message"]
    assert upstream.requests == []


def test_count_tokens_never_loads_a_model(mapped_gateway):
    gw, upstream, state = mapped_gateway
    body = {"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi" * 40}]}
    status, _, reply = _call(gw, "POST", "/v1/messages/count_tokens", body)
    assert status == 200 and json.loads(reply)["input_tokens"] > 1
    assert state["loads"] == [] and upstream.requests == []


def test_a_failed_load_is_a_turn_that_says_why(mapped_gateway):
    gw, upstream, state = mapped_gateway
    state["fail"] = "not enough memory"
    body = {"model": "claude-opus-5", "max_tokens": 8, "messages": []}
    status, _, reply = _call(gw, "POST", "/v1/messages", body)
    text = json.loads(reply)["content"][0]["text"]
    assert status == 200 and "unsloth/big-GGUF" in text and "not enough memory" in text
    assert upstream.requests == []


def test_concurrent_slots_load_one_at_a_time(mapped_gateway):
    gw, _, state = mapped_gateway
    active, peak = [0], [0]
    plain = gw.ensure_loaded

    def slow(model_id):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        threading.Event().wait(0.05)
        plain(model_id)
        active[0] -= 1

    gw.ensure_loaded = slow
    threads = [
        threading.Thread(
            target = _call,
            args = (gw, "POST", "/v1/messages", {"model": r, "max_tokens": 8, "messages": []}),
        )
        for r in ("claude-opus-5", "claude-sonnet-5", "claude-opus-5")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def _slow_loads(gw, seconds = 0.3):
    """Loads take ``seconds``; keep-alives every 20 ms. Returns an Event set when a load ends."""
    plain, finished = gw.ensure_loaded, threading.Event()

    def slow(model_id):
        threading.Event().wait(seconds)
        try:
            plain(model_id)
        finally:
            finished.set()

    gw.ensure_loaded = slow
    gw.keepalive_seconds = 0.02
    return finished


def test_a_streaming_turn_pings_while_its_model_loads(mapped_gateway):
    gw, upstream, state = mapped_gateway
    _slow_loads(gw)
    upstream.replies.append(("sse", ["message_start", "message_stop"]))
    body = {"model": "claude-opus-5", "max_tokens": 8, "messages": [], "stream": True}
    status, headers, reply = _call(gw, "POST", "/v1/messages", body)
    assert status == 200 and headers["Content-Type"] == "text/event-stream"
    pings = reply.count(cda.PING_EVENT)
    assert pings >= 3
    # Pings first, then Studio's stream as it came.
    assert reply.index(b"event: message_start") > reply.rindex(cda.PING_EVENT)
    assert reply.endswith(b"event: message_stop\ndata: {}\n\n")
    assert state["loads"] == ["unsloth/big-GGUF"]
    assert upstream.requests[-1][2]["model"] == "unsloth/big-GGUF"


def test_a_plain_turn_is_padded_with_json_whitespace_while_its_model_loads(mapped_gateway):
    gw, upstream, _ = mapped_gateway
    _slow_loads(gw)
    body = {"model": "claude-opus-5", "max_tokens": 8, "messages": []}
    status, headers, reply = _call(gw, "POST", "/v1/messages", body)
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert reply.startswith(b"   ") and json.loads(reply) == {"ok": 1}


def test_a_failed_streaming_load_is_a_streamed_turn_that_says_why(mapped_gateway):
    gw, upstream, state = mapped_gateway
    _slow_loads(gw, 0.05)
    state["fail"] = "not enough memory"
    body = {"model": "claude-opus-5", "max_tokens": 8, "messages": [], "stream": True}
    status, _, reply = _call(gw, "POST", "/v1/messages", body)
    assert status == 200
    assert b"event: message_stop" in reply and b"not enough memory" in reply
    assert upstream.requests == []


def test_studio_failing_after_the_load_is_a_turn_that_says_why(mapped_gateway):
    gw, upstream, _ = mapped_gateway
    _slow_loads(gw, 0.05)
    upstream.replies.append((503, {"detail": "Model is still warming up"}))
    body = {"model": "claude-opus-5", "max_tokens": 8, "messages": [], "stream": True}
    status, _, reply = _call(gw, "POST", "/v1/messages", body)
    assert status == 200 and b"Model is still warming up" in reply
    assert b"event: message_stop" in reply


def test_claude_hanging_up_during_a_load_is_quiet_and_sends_nothing_upstream(mapped_gateway, capfd):
    import socket

    gw, upstream, state = mapped_gateway
    finished = _slow_loads(gw, 0.3)
    data = json.dumps(
        {"model": "claude-opus-5", "max_tokens": 8, "messages": [], "stream": True}
    ).encode()
    sock = socket.create_connection(("127.0.0.1", gw.server_address[1]), timeout = 5)
    sock.sendall(
        b"POST /v1/messages HTTP/1.1\r\n"
        + f"Host: 127.0.0.1:{gw.server_address[1]}\r\n".encode()
        + f"Authorization: Bearer {TOKEN}\r\n".encode()
        + f"Content-Length: {len(data)}\r\n\r\n".encode()
        + data
    )
    assert sock.recv(4096).startswith(b"HTTP/1.1 200")
    # RST on close, as a client giving up does.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    sock.close()
    assert finished.wait(5)
    threading.Event().wait(0.2)
    assert upstream.requests == []
    assert "Traceback" not in capfd.readouterr().err
    # The load ran on: the retry goes straight through, no second load.
    body = {"model": "claude-opus-5", "max_tokens": 8, "messages": []}
    assert _call(gw, "POST", "/v1/messages", body)[0] == 200
    assert state["loads"] == ["unsloth/big-GGUF"]


def test_count_tokens_is_estimated_while_a_load_runs(mapped_gateway):
    gw, upstream, _ = mapped_gateway
    _slow_loads(gw, 0.3)
    turn = threading.Thread(
        target = _call,
        args = (gw, "POST", "/v1/messages", {"model": "claude-opus-5", "messages": []}),
    )
    turn.start()
    threading.Event().wait(0.1)
    body = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "hi" * 40}]}
    status, _, reply = _call(gw, "POST", "/v1/messages/count_tokens", body)
    assert status == 200 and json.loads(reply)["input_tokens"] > 1
    turn.join()
    assert [r[0] for r in upstream.requests] == ["/v1/messages"]


def test_count_tokens_is_estimated_when_studio_cannot_count(gateway):
    gw, upstream = gateway
    upstream.replies.append((503, {"detail": "No model loaded"}))
    body = {"model": "claude-fable-5", "messages": [{"role": "user", "content": "hi" * 40}]}
    status, _, reply = _call(gw, "POST", "/v1/messages/count_tokens", body)
    assert status == 200 and json.loads(reply)["input_tokens"] > 1
    assert upstream.requests[-1][0] == "/v1/messages/count_tokens"


def test_upstream_error_message_reads_every_shape():
    assert cda.upstream_error_message(503, b'{"detail": "busy"}') == "busy"
    assert cda.upstream_error_message(500, b'{"detail": {"message": "oom"}}') == "oom"
    assert cda.upstream_error_message(400, b'{"error": {"message": "bad"}}') == "bad"
    assert cda.upstream_error_message(502, b"") == "HTTP 502"
    assert cda.upstream_error_message(500, b"plain text") == "plain text"


def test_without_mappings_the_gateway_keeps_one_following_slot(gateway):
    gw, _ = gateway
    assert gw.current_mappings() == cda.DEFAULT_MAPPINGS
    catalog = json.loads(_call(gw, "GET", "/v1/models")[2])
    assert [e["id"] for e in catalog["data"]] == ["claude-fable-5"]


class _LoadHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append((self.path, self.headers.get("Authorization"), body))
        status, text = self.server.reply
        data = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.mark.parametrize(
    "reply, error",
    [
        ((200, '   \n{"success": true, "model": "m"}'), None),
        ((404, '{"detail": "Model not found"}'), "isn't downloaded"),
        ((409, '{"detail": "Model is not downloaded"}'), "not downloaded"),
        ((500, '{"detail": {"message": "out of memory"}}'), "out of memory"),
        ((200, '{"success": false}'), "HTTP 200"),
    ],
)
def test_load_model_posts_to_studios_models_load(reply, error):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LoadHandler)
    server.seen, server.reply = [], reply
    threading.Thread(target = server.serve_forever, args = (0.05,), daemon = True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        if error is None:
            cda.load_model(url, "sk-unsloth-k", "unsloth/m-GGUF", timeout = 5)
        else:
            with pytest.raises(RuntimeError, match = error):
                cda.load_model(url, "sk-unsloth-k", "unsloth/m-GGUF", timeout = 5)
        assert server.seen == [("/models/load", "Bearer sk-unsloth-k", {"model": "unsloth/m-GGUF"})]
    finally:
        server.shutdown()
        server.server_close()


def test_load_model_reports_studio_down():
    with pytest.raises(RuntimeError, match = "didn't answer"):
        cda.load_model("http://127.0.0.1:9", "k", "m", timeout = 2)


# -- a model loaded with too small a window for Claude Desktop ---------------------------------

STUDIO_OVERFLOW = (
    "Prompt is too long: 61493 tokens > 8192 maximum. Even with every earlier turn dropped, this "
    "prompt would still be too long, so shortening the conversation will not help."
)
LLAMA_OVERFLOW = {
    "error": {
        "code": 400,
        "message": "request (55317 tokens) exceeds the available context size (8192 tokens), try increasing it",
        "type": "exceed_context_size_error",
        "n_prompt_tokens": 55317,
        "n_ctx": 8192,
    }
}


@pytest.mark.parametrize(
    "text, counts",
    [
        (STUDIO_OVERFLOW, (61493, 8192)),
        (LLAMA_OVERFLOW["error"]["message"], (55317, 8192)),
        (
            "Message too long: 25988 tokens exceeds the 8192-token context window. Try",
            (25988, 8192),
        ),
        ("CUDA out of memory", None),
    ],
)
def test_overflow_counts_reads_every_wording(text, counts):
    assert cda.overflow_counts(text) == counts


def test_the_suggested_context_fits_the_prompt_with_room_to_spare():
    assert cda.suggested_context(61493, cda.CLAUDE_MIN_CONTEXT) == 131072
    assert cda.suggested_context(20000, cda.CLAUDE_MIN_CONTEXT) == 65536
    assert cda.suggested_context(25988, 32768) == 32768


def test_a_window_too_small_for_claude_is_a_400_that_says_how_to_fix_it(gateway):
    gw, upstream = gateway
    error = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": STUDIO_OVERFLOW},
    }
    upstream.replies.append((400, error))
    status, _, body = _call(gw, "POST", "/v1/messages", {"model": "claude-fable-5", "messages": []})
    reply = json.loads(body)
    assert status == 400 and reply["error"]["type"] == "invalid_request_error"
    message = reply["error"]["message"]
    assert "context of 8,192 tokens" in message and "Claude Desktop needs 61,493 tokens" in message
    assert "at least 131,072" in message and "Remember for this model" in message
    assert MODEL_ID in message
    # Not Anthropic's "Prompt is too long" head: compacting cannot fit a window this small.
    assert not message.lower().startswith("prompt is too long")


def test_a_long_chat_overflowing_a_big_window_passes_through_untouched(gateway):
    gw, upstream = gateway
    long_chat = (
        "Prompt is too long: 270000 tokens > 262144 maximum. Try increasing the Context Length"
    )
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": long_chat}}
    upstream.replies.append((400, error))
    status, _, body = _call(gw, "POST", "/v1/messages", {"model": "claude-fable-5", "messages": []})
    assert status == 400 and json.loads(body) == error


def test_an_in_band_overflow_in_a_studio_stream_is_reworded(gateway):
    gw, upstream = gateway
    error = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": STUDIO_OVERFLOW},
    }
    upstream.replies.append(
        (
            "raw-sse",
            [
                b'event: ping\ndata: {"type": "ping"}\n\n',
                b"event: error\ndata: " + json.dumps(error).encode() + b"\n\n",
            ],
        )
    )
    status, _, body = _call(
        gw, "POST", "/v1/messages", {"model": "claude-fable-5", "stream": True, "messages": []}
    )
    assert status == 200
    assert body.startswith(b'event: ping\ndata: {"type": "ping"}\n\n')
    event = body.split(b"event: error\ndata: ", 1)[1].split(b"\n\n", 1)[0]
    assert "at least 131,072" in json.loads(event)["error"]["message"]


def test_other_stream_events_are_relayed_byte_for_byte(gateway):
    gw, upstream = gateway
    events = [
        b'event: message_start\ndata: {"type": "message_start", "message": {"id": "m"}}\n\n',
        b'event: content_block_delta\ndata: {"delta": {"text": "61493 tokens > 8192 maximum"}}\n\n',
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ]
    upstream.replies.append(("raw-sse", events))
    _, _, body = _call(
        gw, "POST", "/v1/messages", {"model": "claude-fable-5", "stream": True, "messages": []}
    )
    assert body == b"".join(events)


def test_a_model_loaded_too_small_for_the_turn_says_how_to_fix_it(mapped_gateway):
    gw, upstream, _ = mapped_gateway
    _slow_loads(gw, 0.05)
    upstream.replies.append((400, LLAMA_OVERFLOW))
    body = {"model": "claude-opus-5", "max_tokens": 8, "messages": [], "stream": True}
    status, _, reply = _call(gw, "POST", "/v1/messages", body)
    assert status == 200 and b"event: message_stop" in reply
    assert b"unsloth/big-GGUF is loaded with a context of 8,192 tokens" in reply
    assert b"at least 131,072" in reply
