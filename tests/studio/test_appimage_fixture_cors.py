# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The packaged-webview E2E fixture must not hold a header allowlist of its own.

`appimage_model_download_webdriver.py` stands up a stub backend that the real
WebKitGTK view talks to over `tauri://localhost` -> `http://127.0.0.1`, so every
authenticated fetch is cross-origin and preflighted. The fixture used to answer that
preflight with a fixed `Authorization, Content-Type, X-HF-Token`, which is a copy of
the app's header set that nothing kept in sync: #8879 added `X-Unsloth-Timezone` to
every authenticated request, the preflight then rejected it, and the request never
reached the server at all. The webview turned that TypeError into "Unsloth isn't
running -- please relaunch it." on every model row, and the run failed 15 seconds
later waiting for a quantization that could never load. Only `/api/health` and
`/api/liveness` kept working, because they send neither a token nor a custom header
and so are never preflighted.

Echoing `Access-Control-Request-Headers` cannot drift, so that is what is pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

DRIVER = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "studio"
    / "appimage_model_download_webdriver.py"
)
AUTH_API = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "auth"
    / "api.ts"
)


def test_the_fixture_echoes_the_requested_headers() -> None:
    source = DRIVER.read_text(encoding = "utf-8")
    assert 'self.headers.get("Access-Control-Request-Headers")' in source, (
        "the E2E fixture no longer echoes the preflight's requested headers, so the "
        "next header the app adds will silently fail every authenticated fetch"
    )
    hardcoded = re.findall(
        r'send_header\(\s*"Access-Control-Allow-Headers"\s*,\s*"([^"]*)"', source
    )
    assert not hardcoded, (
        f"the fixture answers the preflight with a fixed header list {hardcoded}, which "
        f"is a copy of the app's header set that nothing keeps in sync"
    )


def test_the_headers_the_app_actually_sends_are_still_custom() -> None:
    """If these ever became simple headers the echo would be unnecessary, not wrong."""
    api = AUTH_API.read_text(encoding = "utf-8")
    assert '"X-Unsloth-Timezone"' in api
    assert "X-Unsloth-Timezone-Offset-Minutes" in api
    assert "addBrowserTimezoneHeaders(" in api, (
        "the timezone headers are no longer attached to outgoing requests; if they are "
        "gone for good this guard and the fixture echo can go with them"
    )
