# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep every fetched AppImage build tool digest-pinned and off rolling release aliases."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest


REPO = Path(__file__).resolve().parents[2]
TOOL_SCRIPT = REPO / "studio/src-tauri/linux/prepare-complete-appimage-tools.sh"
ASSIGNMENT = re.compile(r'^([A-Z][A-Z0-9_]*)="([^"]+)"$', re.MULTILINE)
FETCH = re.compile(
    r'^fetch "\$([A-Z][A-Z0-9_]*)_URL" "\$\1_SHA256" (\S+)$',
    re.MULTILINE,
)
ROLLING_RELEASE_ALIASES = {"continuous", "latest"}


def _assignments(script: str) -> dict[str, str]:
    return dict(ASSIGNMENT.findall(script))


def _assert_no_rolling_release_aliases(script: str) -> None:
    values = _assignments(script)
    offenders = []
    for prefix, _filename in FETCH.findall(script):
        url_name = f"{prefix}_URL"
        url = values[url_name]
        path_parts = set(urlsplit(url).path.split("/"))
        if path_parts & ROLLING_RELEASE_ALIASES:
            offenders.append(f"{url_name}={url}")
    assert not offenders, "AppImage build tools must use versioned URLs: " + ", ".join(offenders)


def test_every_fetched_tool_has_a_sha256_digest() -> None:
    script = TOOL_SCRIPT.read_text(encoding = "utf-8")
    values = _assignments(script)
    fetches = FETCH.findall(script)
    assert fetches, "expected digest-verified AppImage tool fetches"

    for prefix, filename in fetches:
        url_name = f"{prefix}_URL"
        digest_name = f"{prefix}_SHA256"
        assert url_name in values, f"{filename} has no URL assignment"
        assert re.fullmatch(
            r"[0-9a-f]{64}", values.get(digest_name, "")
        ), f"{filename} has no full SHA-256 digest"


def test_fetched_tools_do_not_use_rolling_release_aliases() -> None:
    _assert_no_rolling_release_aliases(TOOL_SCRIPT.read_text(encoding = "utf-8"))


def test_the_former_continuous_plugin_url_is_rejected() -> None:
    former_pin = """
APPIMAGE_PLUGIN_URL="https://github.com/linuxdeploy/linuxdeploy-plugin-appimage/releases/download/continuous/linuxdeploy-plugin-appimage-x86_64.AppImage"
APPIMAGE_PLUGIN_SHA256="0441769ab38009504d2678c38cd7e526955388dd30a215b4a20afaa5471652f2"
fetch "$APPIMAGE_PLUGIN_URL" "$APPIMAGE_PLUGIN_SHA256" linuxdeploy-plugin-appimage.AppImage
"""
    with pytest.raises(AssertionError, match = "APPIMAGE_PLUGIN_URL"):
        _assert_no_rolling_release_aliases(former_pin)
