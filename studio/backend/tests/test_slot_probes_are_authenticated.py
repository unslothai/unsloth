# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``/slots`` is not a public endpoint."""

from __future__ import annotations

import json
import re
import urllib.request

from core.inference import llama_stats
from core.inference.llama_stats import erase_llama_slot, fetch_llama_slots


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _capture(monkeypatch, body: bytes):
    seen: list[urllib.request.Request] = []

    def _urlopen(request, timeout = None):
        # Both helpers must send a Request, not a bare URL, or there is nowhere to put the header
        # at all.
        assert isinstance(request, urllib.request.Request), request
        seen.append(request)
        return _Response(body)

    class _Opener:
        open = staticmethod(_urlopen)

    def _proxied(*_a, **_k):
        raise AssertionError("urlopen honours HTTP_PROXY and would hand the bearer to it")

    monkeypatch.setattr(llama_stats, "_loopback_opener", _Opener)
    monkeypatch.setattr(urllib.request, "urlopen", _proxied)
    return seen


class TestTheReadCarriesTheKey:
    def test_the_authorization_header_is_sent(self, monkeypatch):
        seen = _capture(monkeypatch, json.dumps([]).encode())
        fetch_llama_slots("http://127.0.0.1:8080", headers = {"Authorization": "Bearer secret-key"})
        assert seen[0].get_header("Authorization") == "Bearer secret-key"

    def test_a_load_without_a_key_still_sends_nothing(self, monkeypatch):
        """The ordinary case. `_auth_headers` is None unless --api-key was used."""
        seen = _capture(monkeypatch, json.dumps([]).encode())
        fetch_llama_slots("http://127.0.0.1:8080")
        assert seen[0].get_header("Authorization") is None


class TestTheEraseCarriesTheKey:
    def test_the_authorization_header_is_sent(self, monkeypatch):
        seen = _capture(monkeypatch, json.dumps({"n_erased": 4096}).encode())
        freed = erase_llama_slot(
            "http://127.0.0.1:8080", 2, headers = {"Authorization": "Bearer secret-key"}
        )
        assert freed == 4096
        assert seen[0].get_header("Authorization") == "Bearer secret-key"
        assert seen[0].get_method() == "POST"


def test_a_configured_proxy_never_sees_the_loopback_call(monkeypatch):
    import http.server
    import threading

    seen = []

    class _Slots(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            body = b"[]"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Slots)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising = False)
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        assert fetch_llama_slots(url, headers = {"Authorization": "Bearer k"}) == []
    finally:
        server.shutdown()
    assert seen == ["Bearer k"]


class TestTheRouteHandsThemTheBackendsKey:
    def test_every_slot_call_goes_through_the_header_helper(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text(encoding = "utf-8")
        assert "def _llama_slot_headers(" in source
        flat = re.sub(r"\s+", " ", source)
        assert flat.count("fetch_llama_slots(") == flat.count(
            "fetch_llama_slots( base, headers = _llama_slot_headers(llama_backend) )"
        ) + flat.count("fetch_llama_slots(base, headers = _llama_slot_headers(llama_backend))")
        assert flat.count("erase_llama_slot(") == flat.count(
            "erase_llama_slot( base, slot_id, headers = _llama_slot_headers(llama_backend) )"
        ) + flat.count(
            "erase_llama_slot(base, slot_id, headers = _llama_slot_headers(llama_backend))"
        )

    def test_the_helper_reads_the_backends_own_bearer(self):
        from routes.inference import _llama_slot_headers

        class _Backend:
            _auth_headers = {"Authorization": "Bearer abc"}

        class _Unkeyed:
            _auth_headers = None

        assert _llama_slot_headers(_Backend()) == {"Authorization": "Bearer abc"}
        assert _llama_slot_headers(_Unkeyed()) == {}
        assert _llama_slot_headers(object()) == {}
