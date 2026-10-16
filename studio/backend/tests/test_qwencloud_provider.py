# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import list_available_providers
from models.providers import ProviderModelsRequest
from routes import providers as providers_route

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_provider_sampling_over_the_wire import _Handler, _run, _Server  # noqa: E402
from test_external_provider_stream_usage_opt_in import _USAGE, _proxy, _usage_chunks  # noqa: E402


class _AuthHandler(_Handler):
    def do_POST(self) -> None:  # noqa: N802
        self.server.auth = self.headers.get("Authorization")  # type: ignore[attr-defined]
        super().do_POST()


_CHAT_IDS = [
    "qwen3.8-max",
    "qwen3.7-plus",
    "qwen3.8-flash",
    "qwen3.8-2.4t-a95b",
    "qwen3.8-omni-flash",
    "qwen3.5-omni-plus",
    "qwen3-vl-plus",
    "qwen3-coder-flash",
    "qwen-mt-plus",
    "qwen-plus-character",
    "qwq-plus-2025-03-05",
    "glm-5.3",
    "ZHIPU/GLM-5.3",
    "kimi-k3",
    "deepseek-v4-pro",
]
_NON_CHAT_IDS = [
    "pre-mock_new_gateway_session_affinity",
    "qwen3.8-omni-flash-realtime",
    "qwen3.5-livetranslate-flash-realtime",
    "qwen3-livetranslate-flash",
    "qwen-audio-3.1-realtime-plus",
    "qwen-audio-3.0-asr-flash",
    "qwen3-asr-flash-realtime",
    "qwen3-tts-flash",
    "qwen3-s2s-flash-realtime",
    "qwen-image-3.0-pro",
    "wan2.7-image-pro",
    "z-image-turbo",
    "qwen3.7-text-embedding",
    "text-embedding-v4",
    "qwen-vl-ocr-2025-11-20",
    "tongyi-tingwu-slp",
    "ccai-pro",
]


class _ModelsHandler(_Handler):
    def do_GET(self) -> None:  # noqa: N802
        ids = self.server.ids  # type: ignore[attr-defined]
        body = json.dumps({"object": "list", "data": [{"id": i} for i in ids]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_the_qwencloud_model_list_keeps_only_chat_models(monkeypatch):
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "sk-ws-test"
    )
    listed: list[str] = []
    with _Server(handler = _ModelsHandler) as server:
        server._httpd.ids = _NON_CHAT_IDS + _CHAT_IDS

        async def go() -> None:
            payload = ProviderModelsRequest(provider_type = "qwencloud", base_url = server.base_url)
            result = await providers_route.list_provider_models(payload, "tester", False)
            listed.extend(m.id for m in result)

        _run(go)
    assert listed == _CHAT_IDS


def test_qwencloud_is_offered_with_its_own_endpoint():
    rows = {row["provider_type"]: row for row in list_available_providers()}
    qwencloud = rows["qwencloud"]
    assert qwencloud["display_name"] == "QwenCloud"
    assert qwencloud["base_url"] == "https://maas.qwencloudapi.com/compatible-mode/v1"
    assert qwencloud["hidden"] is False
    assert qwencloud["supports_studio_tools"] is True
    assert {"qwen3.8-max", "qwen3.7-plus", "qwen3.8-flash"} <= set(qwencloud["default_models"])
    assert rows["qwen"]["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_a_qwencloud_connection_receives_the_usage_chunk():
    chunks, body = _proxy("qwencloud", stream_options = {"include_usage": True})
    assert body["stream_options"] == {"include_usage": True}
    assert _usage_chunks(chunks) == [_USAGE]


@pytest.mark.parametrize("enabled", [True, False])
def test_the_qwencloud_thinking_toggle_reaches_the_wire(enabled):
    with _Server(handler = _AuthHandler) as server:
        client = ExternalProviderClient(
            provider_type = "qwencloud", base_url = server.base_url, api_key = "sk-ws-test"
        )

        async def go() -> None:
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "qwen3.7-plus",
                enable_thinking = enabled,
            ):
                pass

        _run(go)
        body = server.bodies[-1]
        assert server._httpd.auth == "Bearer sk-ws-test"
    assert body["enable_thinking"] is enabled
    assert "chat_template_kwargs" not in body
    assert "thinking" not in body
