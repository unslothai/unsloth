# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Claude Desktop profile switching and the loopback gateway, without a real Claude or Studio."""

from __future__ import annotations

import http.client
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from unsloth_cli import claude_desktop_app as cda

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
