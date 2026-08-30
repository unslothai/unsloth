# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Keep the packaged-webview fixture's CORS headers aligned with each preflight."""

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
    """Keep the guard relevant while the frontend sends custom headers."""
    api = AUTH_API.read_text(encoding = "utf-8")
    assert '"X-Unsloth-Timezone"' in api
    assert "X-Unsloth-Timezone-Offset-Minutes" in api
    assert "addBrowserTimezoneHeaders(" in api, (
        "the timezone headers are no longer attached to outgoing requests; if they are "
        "gone for good this guard and the fixture echo can go with them"
    )
