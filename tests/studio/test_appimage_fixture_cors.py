# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The AppImage E2E's fake backend has to admit the headers the real one admits.

`AppImage model download E2E` stands up a fixture backend for the webview to talk to.
Its CORS preflight used to answer with a hand-written list of three header names. The
real backend does not: studio/backend/main.py runs CORSMiddleware with
`allow_headers = ["*"]`. A fixed list is therefore a second copy of the product's
policy, and on 2026-08-28 it drifted -- #8879 began sending X-Unsloth-Timezone and
X-Unsloth-Timezone-Offset-Minutes on every authFetch, the list still named three
headers, and the browser rejected the preflight for every authed request in the test.

The visible symptom was nothing like the cause: the row for the model under test
rendered "Unsloth isn't running -- please relaunch it." (a fetch TypeError, tagged by
asTransportFailure), the request log showed no such request because it was never sent,
and the run failed as "Timed out waiting for the Q4_K_M quantization" while
/api/health kept answering. Roughly three runs in four, as a hard gate on every PR.

So these tests read the header names out of the frontend source rather than restating
them, and assert the fixture would admit them.
"""

from __future__ import annotations

import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AUTH_API = REPO / "studio" / "frontend" / "src" / "features" / "auth" / "api.ts"
FIXTURE = REPO / "tests" / "studio" / "appimage_model_download_webdriver.py"
BACKEND_MAIN = REPO / "studio" / "backend" / "main.py"


def frontend_request_headers() -> list[str]:
    """Every X-Unsloth-* header the auth layer attaches to an authed request."""
    names = re.findall(r'"(X-Unsloth-[A-Za-z0-9-]+)"', AUTH_API.read_text(encoding = "utf-8"))
    assert names, f"no X-Unsloth-* header constants found in {AUTH_API}; did they move?"
    return sorted(set(names))


def _preflight(allow_headers_for) -> str:
    """Run one real OPTIONS through http.server and return its Allow-Headers."""
    sent = ",".join(frontend_request_headers() + ["Authorization"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "tauri://localhost")
            self.send_header(
                "Access-Control-Allow-Headers",
                allow_headers_for(self.headers.get("Access-Control-Request-Headers")),
            )
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/models/gguf-variants",
            method = "OPTIONS",
        )
        request.add_header("Origin", "tauri://localhost")
        request.add_header("Access-Control-Request-Method", "GET")
        request.add_header("Access-Control-Request-Headers", sent)
        with urllib.request.urlopen(request, timeout = 10) as response:
            return (response.headers.get("Access-Control-Allow-Headers") or "").lower()
    finally:
        server.shutdown()


def blocked_headers(allowed: str) -> list[str]:
    """What a browser would refuse to send, given this Allow-Headers."""
    permitted = {h.strip() for h in allowed.split(",") if h.strip()}
    if "*" in permitted:
        return []
    return [h for h in frontend_request_headers() if h.lower() not in permitted]


def test_echoing_the_requested_headers_admits_everything_the_frontend_sends():
    allowed = _preflight(lambda requested: requested or "*")
    assert not blocked_headers(allowed), allowed


def test_the_list_that_broke_it_is_still_detected_as_broken():
    """The bug reproduced, so this file fails if the check stops being able to see it."""
    allowed = _preflight(lambda _requested: "Authorization, Content-Type, X-HF-Token")
    assert blocked_headers(allowed) == [
        "X-Unsloth-Timezone",
        "X-Unsloth-Timezone-Offset-Minutes",
    ], allowed


def test_the_fixture_echoes_rather_than_naming_headers():
    """A fixed list cannot track `allow_headers = ["*"]`, so the fixture must not carry one."""
    source = FIXTURE.read_text(encoding = "utf-8")
    options = source.split("def do_OPTIONS", 1)
    assert len(options) == 2, "the fixture no longer answers preflights"
    body = options[1].split("def do_GET", 1)[0]
    assert "Access-Control-Request-Headers" in body, (
        "the fixture's preflight no longer echoes the requested headers, so it has gone "
        "back to a hand-written list that will drift from the backend again"
    )
    # Comments stripped first: the block deliberately names the two headers when
    # explaining what drifted, and that history is worth keeping. Only code counts.
    code = "\n".join(line.split("#", 1)[0] for line in body.splitlines())
    for name in frontend_request_headers():
        assert name not in code, (
            f"{name} is hard-coded into the fixture's preflight. Echo "
            f"Access-Control-Request-Headers instead; naming headers is what broke this."
        )


def test_the_real_backend_still_allows_any_header():
    """The premise. If the product ever narrows this, echoing stops being faithful."""
    source = BACKEND_MAIN.read_text(encoding = "utf-8")
    assert re.search(r"allow_headers\s*=\s*\[\s*\"\*\"\s*\]", source), (
        "studio/backend/main.py no longer allows every request header, so the fixture "
        "echoing the request is no longer a faithful stand-in for it"
    )


@pytest.mark.parametrize("name", frontend_request_headers())
def test_each_frontend_header_is_lowercase_safe(name):
    """CORS matching is case-insensitive; the echo path must not depend on casing."""
    allowed = _preflight(lambda requested: (requested or "").upper())
    assert name.lower() in allowed.lower() or "*" in allowed, (name, allowed)
