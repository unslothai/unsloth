# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ensure remote image URLs are fetched safely or refused before llama-server sees them."""

from __future__ import annotations

import base64
import copy
import socket
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from auth.authentication import get_current_subject  # noqa: E402
import core.inference.external_provider as external_provider  # noqa: E402
import routes.inference as inference_route  # noqa: E402

from .llama_backend_double import FakeLlamaCppBackend  # noqa: E402

_METADATA = "http://169.254.169.254/latest/meta-data/"


def _webp_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (2, 2), (7, 8, 9)).save(buf, format = "WEBP")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _data_url() -> str:
    buf = BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class _VisionGguf(FakeLlamaCppBackend):
    is_vision = True

    def __init__(self):
        self.dispatched: list[dict] = []
        self.counted: list[list[dict]] = []

    def generate_chat_completion(self, **kwargs):
        self.dispatched.append(kwargs)
        yield "ok"
        yield {"type": "metadata", "usage": {}, "timings": {}}

    generate_chat_completion_with_tools = generate_chat_completion

    def count_chat_tokens(self, messages, *_a, **_k):
        self.counted.append(copy.deepcopy(messages))
        return 100


def _client(monkeypatch, backend):
    async def _no_switch(*_a, **_k):
        return None

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _no_switch)

    app = FastAPI()
    app.include_router(inference_route.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app, raise_server_exceptions = False)


def _chat_body(*urls, model = "test/model.gguf"):
    parts = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    parts.append({"type": "text", "text": "what is this?"})
    return {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": parts}],
    }


def _messages_body(url):
    return {
        "model": "test/model.gguf",
        "max_tokens": 16,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": url}},
                    {"type": "text", "text": "what is this?"},
                ],
            }
        ],
    }


def _image_urls(dispatched_messages) -> list[str]:
    return [
        part["image_url"]["url"]
        for msg in dispatched_messages
        for part in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


@pytest.fixture
def canary():
    """A loopback HTTP server that records every request it is asked to serve."""
    hits: list[str] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            body = base64.b64decode(_webp_b64())
            self.send_response(200)
            self.send_header("Content-Type", "image/webp")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        yield SimpleNamespace(port = server.server_address[1], hits = hits)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout = 5)


class TestRemoteUrlNeverReachesLlamaServer:
    def test_public_https_url_is_fetched_here_and_dispatched_as_bytes(self, monkeypatch):
        monkeypatch.setattr(
            external_provider,
            "safe_fetch_remote_image_sync",
            lambda *_a, **_k: ("image/webp", _webp_b64()),
        )
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/chat/completions", json = _chat_body("https://images.example/cat.webp")
        )

        assert r.status_code == 200, r.text
        urls = _image_urls(backend.dispatched[-1]["messages"])
        assert len(urls) == 1
        assert urls[0].startswith("data:image/png;base64,")

    def test_anthropic_url_source_is_fetched_here_too(self, monkeypatch):
        monkeypatch.setattr(
            external_provider,
            "safe_fetch_remote_image_sync",
            lambda *_a, **_k: ("image/webp", _webp_b64()),
        )
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/messages", json = _messages_body("https://images.example/cat.webp")
        )

        assert r.status_code == 200, r.text
        urls = _image_urls(backend.dispatched[-1]["messages"])
        assert urls and all(u.startswith("data:image/png;base64,") for u in urls)

    def test_responses_input_image_lands_on_the_same_guard(self, monkeypatch):
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/responses",
            json = {
                "model": "test/model.gguf",
                "stream": False,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": _METADATA},
                            {"type": "input_text", "text": "what is this?"},
                        ],
                    }
                ],
            },
        )

        assert r.status_code == 400, r.text
        assert backend.dispatched == []

    def test_bare_base64_is_payload_and_still_passes_through(self, monkeypatch):
        def _never(*_a, **_k):
            raise AssertionError("payload must not be treated as a URL")

        monkeypatch.setattr(external_provider, "safe_fetch_remote_image_sync", _never)
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post("/v1/chat/completions", json = _chat_body(_webp_b64()))

        assert r.status_code == 200, r.text
        assert _image_urls(backend.dispatched[-1]["messages"]) == [_webp_b64()]

    def test_a_data_url_still_goes_through_untouched_by_the_fetch(self, monkeypatch):
        def _never(*_a, **_k):
            raise AssertionError("a data: url needs no fetch")

        monkeypatch.setattr(external_provider, "safe_fetch_remote_image_sync", _never)
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post("/v1/chat/completions", json = _chat_body(_data_url()))

        assert r.status_code == 200, r.text
        assert _image_urls(backend.dispatched[-1]["messages"])[0].startswith("data:image/png")


class TestRefusalsAreRealRefusals:
    @pytest.mark.parametrize(
        "url_for",
        [
            pytest.param(lambda p: f"http://127.0.0.1:{p}/x.png", id = "http-loopback"),
            pytest.param(lambda p: f"https://127.0.0.1:{p}/x.png", id = "https-loopback"),
            pytest.param(lambda p: f"https://localhost:{p}/x.png", id = "https-resolves-loopback"),
        ],
    )
    def test_loopback_is_refused_and_never_contacted(self, monkeypatch, canary, url_for):
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/chat/completions", json = _chat_body(url_for(canary.port))
        )

        assert r.status_code == 400, r.text
        assert backend.dispatched == []
        assert canary.hits == [], "the guard let a request through to the loopback server"

    def test_the_metadata_endpoint_is_refused(self, monkeypatch):
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post("/v1/chat/completions", json = _chat_body(_METADATA))

        assert r.status_code == 400, r.text
        assert backend.dispatched == []

    def test_a_failed_fetch_refuses_rather_than_dropping_the_image(self, monkeypatch):
        monkeypatch.setattr(
            external_provider, "safe_fetch_remote_image_sync", lambda *_a, **_k: None
        )
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/chat/completions", json = _chat_body("https://images.example/gone.png")
        )

        assert r.status_code == 400, r.text
        assert backend.dispatched == []

    def test_one_bad_url_refuses_the_whole_request(self, monkeypatch, canary):
        monkeypatch.setattr(
            external_provider,
            "safe_fetch_remote_image_sync",
            lambda *_a, **_k: ("image/webp", _webp_b64()),
        )
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/chat/completions",
            json = _chat_body(
                "https://images.example/ok.webp", f"http://127.0.0.1:{canary.port}/x.png"
            ),
        )

        assert r.status_code == 400, r.text
        assert backend.dispatched == []
        assert canary.hits == []

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x.example/a.png"])
    def test_another_scheme_is_refused_by_name(self, monkeypatch, url):
        def _never(*_a, **_k):
            raise AssertionError("only https reaches the fetcher")

        monkeypatch.setattr(external_provider, "safe_fetch_remote_image_sync", _never)
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post("/v1/chat/completions", json = _chat_body(url))

        assert r.status_code == 400, r.text
        assert backend.dispatched == []

    def test_the_vision_guard_still_runs_before_any_fetch(self, monkeypatch):
        def _never(*_a, **_k):
            raise AssertionError("a text-only model must refuse before fetching")

        monkeypatch.setattr(external_provider, "safe_fetch_remote_image_sync", _never)

        class _TextGguf(_VisionGguf):
            is_vision = False

        r = _client(monkeypatch, _TextGguf()).post(
            "/v1/chat/completions", json = _chat_body("https://images.example/cat.webp")
        )
        assert r.status_code == 400, r.text


class TestBudget:
    def test_the_per_request_byte_budget_bounds_a_long_thread(self, monkeypatch):
        b64 = _webp_b64()
        each = (len(b64) * 3) // 4
        monkeypatch.setattr(inference_route, "_REMOTE_IMAGE_REQUEST_BUDGET_BYTES", 2 * each + 1)
        calls = []

        def _fetch(
            url,
            _mime,
            max_bytes = 0,
            **_k,
        ):
            calls.append((url, max_bytes))
            return ("image/webp", b64) if each <= max_bytes else None

        monkeypatch.setattr(external_provider, "safe_fetch_remote_image_sync", _fetch)
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/chat/completions",
            json = _chat_body(*[f"https://images.example/{i}.webp" for i in range(4)]),
        )

        assert r.status_code == 400, r.text
        assert [c[1] for c in calls] == [2 * each + 1, each + 1, 1], calls
        assert backend.dispatched == []

    def test_the_per_request_count_cap_bounds_the_round_trips(self, monkeypatch):
        calls = []

        def _fetch(url, *_a, **_k):
            calls.append(url)
            return "image/webp", _webp_b64()

        monkeypatch.setattr(external_provider, "safe_fetch_remote_image_sync", _fetch)
        backend = _VisionGguf()
        over = inference_route._REMOTE_IMAGE_MAX_COUNT + 1
        r = _client(monkeypatch, backend).post(
            "/v1/chat/completions",
            json = _chat_body(*[f"https://images.example/{i}.webp" for i in range(over)]),
        )

        assert r.status_code == 400, r.text
        assert "Too many remote image URLs" in r.text
        assert len(calls) == inference_route._REMOTE_IMAGE_MAX_COUNT, calls
        assert backend.dispatched == []

    def test_every_fetch_in_a_request_shares_one_deadline(self, monkeypatch):
        deadlines = []

        def _fetch(
            url,
            *_a,
            deadline = None,
            **_k,
        ):
            deadlines.append(deadline)
            return "image/webp", _webp_b64()

        monkeypatch.setattr(external_provider, "safe_fetch_remote_image_sync", _fetch)
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/chat/completions",
            json = _chat_body("https://images.example/a.webp", "https://images.example/b.webp"),
        )

        assert r.status_code == 200, r.text
        assert len(deadlines) == 2 and deadlines[0] is not None, deadlines
        assert deadlines[0] == deadlines[1]


def _resolve_publicly(monkeypatch):
    real = socket.getaddrinfo

    def _fake(host, *args, **kwargs):
        if host == "images.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
        if host == "2606:4700::1111":
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (host, 0, 0, 0))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _fake)


class TestFetcher:
    def test_a_server_dripping_bytes_is_cut_off_at_the_deadline(self, monkeypatch):
        _resolve_publicly(monkeypatch)

        class _DripResponse:
            status = 200
            headers = {"content-type": "image/png"}

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def close(self):
                pass

            def read(self, _n = -1):
                time.sleep(0.02)
                return b"x"

        class _Opener:
            def open(
                self,
                _req,
                timeout = None,
            ):
                return _DripResponse()

        monkeypatch.setattr("urllib.request.build_opener", lambda *_a, **_k: _Opener())
        started = time.monotonic()
        result = external_provider.safe_fetch_remote_image_sync(
            "https://images.example/slow.png",
            "image/png",
            deadline = started + 0.3,
        )

        assert result is None
        assert time.monotonic() - started < 5

    def test_a_body_read_timeout_is_a_refusal_not_an_error(self, monkeypatch):
        _resolve_publicly(monkeypatch)

        class _StalledResponse:
            status = 200
            headers = {"content-type": "image/png"}

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def read(self, _n = -1):
                raise TimeoutError("timed out")

        class _Opener:
            def open(
                self,
                _req,
                timeout = None,
            ):
                return _StalledResponse()

        monkeypatch.setattr("urllib.request.build_opener", lambda *_a, **_k: _Opener())
        assert (
            external_provider.safe_fetch_remote_image_sync(
                "https://images.example/stalled.png", "image/png"
            )
            is None
        )

    def test_a_non_ascii_or_spaced_path_is_percent_encoded(self, monkeypatch):
        _resolve_publicly(monkeypatch)
        sent = []

        class _Opener:
            def open(
                self,
                req,
                timeout = None,
            ):
                sent.append(req.full_url)
                raise urllib.error.URLError("stop after the request is built")

        monkeypatch.setattr("urllib.request.build_opener", lambda *_a, **_k: _Opener())
        external_provider.safe_fetch_remote_image_sync(
            "https://images.example/caf\u00e9 1.png?q=\u00e9", "image/png"
        )

        assert sent == ["https://93.184.216.34/caf%C3%A9%201.png?q=%C3%A9"]

    def test_a_request_the_client_cannot_build_is_a_400(self, monkeypatch):
        import http.client

        _resolve_publicly(monkeypatch)

        class _Opener:
            def open(
                self,
                _req,
                timeout = None,
            ):
                raise http.client.InvalidURL("URL can't contain control characters")

        monkeypatch.setattr("urllib.request.build_opener", lambda *_a, **_k: _Opener())
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/chat/completions", json = _chat_body("https://images.example/a.png")
        )

        assert r.status_code == 400, r.text
        assert backend.dispatched == []

    @pytest.mark.parametrize(
        "url,authority",
        [
            ("https://images.example:8443/a.png", "images.example:8443"),
            ("https://[2606:4700::1111]/a.png", "[2606:4700::1111]"),
        ],
    )
    def test_the_host_header_keeps_the_url_authority(self, monkeypatch, url, authority):
        _resolve_publicly(monkeypatch)
        sent = []

        class _Opener:
            def open(
                self,
                req,
                timeout = None,
            ):
                sent.append(req.get_header("Host"))
                raise urllib.error.URLError("stop after the request is built")

        monkeypatch.setattr("urllib.request.build_opener", lambda *_a, **_k: _Opener())
        assert external_provider.safe_fetch_remote_image_sync(url, "image/png") is None
        assert sent == [authority]


class TestCountingIsNotAFetchSurface:
    def test_openai_count_tokens_still_refuses_an_image(self, monkeypatch):
        def _never(*_a, **_k):
            raise AssertionError("counting must not spend a request")

        monkeypatch.setattr(external_provider, "safe_fetch_remote_image_sync", _never)
        backend = _VisionGguf()
        backend.supports_tools = True
        r = _client(monkeypatch, backend).post(
            "/v1/chat/count_tokens", json = _chat_body("https://images.example/cat.webp")
        )

        assert r.status_code == 503, r.text
        assert "images" in r.text

    def test_anthropic_count_tokens_prices_a_remote_image_without_dialling_it(
        self, monkeypatch, canary
    ):
        def _never(*_a, **_k):
            raise AssertionError("counting must not spend a request")

        monkeypatch.setattr(external_provider, "safe_fetch_remote_image_sync", _never)
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/messages/count_tokens",
            json = _messages_body(f"https://127.0.0.1:{canary.port}/counted.png"),
        )

        assert r.status_code == 200, r.text
        assert canary.hits == []
        counted_urls = _image_urls(backend.counted[-1])
        assert counted_urls and all(u.startswith("data:image/png;base64,") for u in counted_urls)

    def test_the_counted_placeholder_prices_the_same_as_a_real_image(self, monkeypatch):
        backend = _VisionGguf()
        client = _client(monkeypatch, backend)
        remote = client.post(
            "/v1/messages/count_tokens",
            json = _messages_body("https://images.example/cat.png"),
        )
        inline = client.post(
            "/v1/messages/count_tokens",
            json = {
                "model": "test/model.gguf",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _data_url().partition(",")[2],
                                },
                            },
                            {"type": "text", "text": "what is this?"},
                        ],
                    }
                ],
            },
        )

        assert remote.status_code == 200, remote.text
        assert inline.status_code == 200, inline.text
        assert remote.json()["input_tokens"] == inline.json()["input_tokens"]

    def test_anthropic_count_tokens_refuses_the_schemes_generation_refuses(
        self, monkeypatch, canary
    ):
        backend = _VisionGguf()
        r = _client(monkeypatch, backend).post(
            "/v1/messages/count_tokens",
            json = _messages_body(f"http://127.0.0.1:{canary.port}/counted.png"),
        )

        assert r.status_code == 400, r.text
        assert backend.counted == []
        assert canary.hits == []
