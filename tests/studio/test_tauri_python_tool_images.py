# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Packaged desktop routing contract for Python tool images."""

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON_TOOL_UI = REPO / "studio/frontend/src/components/assistant-ui/tool-ui-python.tsx"
TAURI_CONFIG = REPO / "studio/src-tauri/tauri.conf.json"
PLAYWRIGHT_TEST = REPO / "tests/studio/playwright_tauri_python_tool_images.py"


def test_python_tool_images_use_authenticated_blob_urls() -> None:
    source = PYTHON_TOOL_UI.read_text(encoding = "utf-8")

    assert 'import { authFetch } from "@/features/auth";' in source
    assert "authFetch(" in source
    assert "pythonToolImagePath(sessionId, filename)" in source
    assert "new AbortController()" in source
    assert "new IntersectionObserver(" in source
    assert "URL.createObjectURL(blob)" in source
    assert "URL.revokeObjectURL(objectUrl)" in source
    assert "controller.abort()" in source
    assert "apiUrl(`/api/inference/sandbox/" not in source


def test_desktop_csp_has_no_explicit_http_loopback_image_source() -> None:
    config = json.loads(TAURI_CONFIG.read_text(encoding = "utf-8"))
    csp = config["app"]["security"]["csp"]
    directives = {
        parts[0]: parts[1:] for directive in csp.split(";") if (parts := directive.strip().split())
    }

    loopback = [
        source for source in directives["img-src"] if "127.0.0.1" in source or "localhost" in source
    ]
    assert loopback == []
    assert "blob:" in directives["img-src"]
    # The boundary is the HTTP Unsloth backend regression.
    # Ordinary remote HTTPS images remain supported, including HTTPS loopback if its certificate is trusted by the host.
    assert "https:" in directives["img-src"]

    # Trusted frontend fetches and artifact frames still use their existing backend channels.
    assert "http://127.0.0.1:*" in directives["connect-src"]
    assert "http://127.0.0.1:*" in directives["frame-src"]


def test_redirect_regression_has_a_real_browser_probe() -> None:
    source = PLAYWRIGHT_TEST.read_text(encoding = "utf-8")

    assert "sync_playwright" in source
    assert "https://redirect.invalid/attacker.png" in source
    assert '"/sensitive/redirect.png"' in source
    assert '"https://127.0.0.1:9443/sensitive/direct.png"' in source
    assert "https_loopback_requests == 1" in source
    assert "URL.createObjectURL" in source
