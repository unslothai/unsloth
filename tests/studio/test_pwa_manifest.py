# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Studio web UI stays installable as a PWA (#12262)."""

import json
from pathlib import Path
import re
import struct


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend"
PUBLIC = FRONTEND / "public"


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", data[16:24])


def test_index_links_the_manifest():
    html = (FRONTEND / "index.html").read_text(encoding = "utf-8")
    links = re.findall(r'<link\s+rel="manifest"\s+href="([^"]+)"', html)
    assert links == ["/manifest.webmanifest"]


def test_manifest_meets_install_criteria():
    manifest = json.loads((PUBLIC / "manifest.webmanifest").read_text(encoding = "utf-8"))
    assert manifest["name"] and manifest["short_name"]
    assert manifest["start_url"] == "/"
    assert manifest["display"] == "standalone"

    # Chromium needs a 192px and a 512px icon it can fetch from the build.
    sizes = set()
    for icon in manifest["icons"]:
        assert icon["src"].startswith("/") and icon["type"] == "image/png"
        width, height = png_size(PUBLIC / icon["src"].lstrip("/"))
        assert icon["sizes"] == f"{width}x{height}"
        sizes.add(width)
    assert {192, 512} <= sizes
