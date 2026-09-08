# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vendored Swagger UI and ReDoc bundles stay byte-identical to the releases they came from.

These files execute on the Unsloth origin, which is where session.ts keeps the access and
refresh tokens, so the point of shipping them rather than loading them from a CDN is that
their bytes are fixed at review time. A silent edit here is a script change nobody read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_DOCS_UI = _BACKEND / "assets" / "docs_ui"
_MANIFEST = json.loads((_DOCS_UI / "docs_ui_manifest.json").read_text(encoding = "utf-8"))

# Ours, not upstream's: prose we may reword, and the manifest cannot hash itself.
_UNPINNED = {"README.md", "docs_ui_manifest.json"}


def _tracked_files() -> dict[str, Path]:
    """Enumerate the real tree, so an *added* file is caught and not just an edit."""
    return {
        path.relative_to(_DOCS_UI).as_posix(): path
        for path in sorted(_DOCS_UI.rglob("*"))
        if path.is_file() and path.relative_to(_DOCS_UI).as_posix() not in _UNPINNED
    }


def test_tree_matches_the_manifest():
    found = _tracked_files()
    recorded = _MANIFEST["files"]
    assert set(found) == set(recorded), (
        "assets/docs_ui gained or lost a file; it is a static copy of the pinned releases, "
        "so update docs_ui_manifest.json in the same commit"
    )
    drifted = [
        name
        for name, path in found.items()
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded[name]
    ]
    assert not drifted, (
        f"vendored docs assets no longer match their pinned releases: {', '.join(drifted)}. "
        "A formatter or minifier most likely rewrote them"
    )


def test_no_symlinks():
    """A symlink would let the digest check pass while the served bytes differ."""
    offenders = [
        str(path.relative_to(_DOCS_UI)) for path in _DOCS_UI.rglob("*") if path.is_symlink()
    ]
    assert not offenders, f"assets/docs_ui must be plain files: {offenders}"


def test_every_package_ships_its_licence():
    names = {entry["package"] for entry in _MANIFEST["packages"]}
    assert names == {"swagger-ui-dist", "redoc"}
    assert (_DOCS_UI / "LICENSE.swagger-ui").exists()
    assert (_DOCS_UI / "LICENSE.redoc").exists()
    # Apache-2.0 section 4(d): redistributing a work that carries a NOTICE means shipping it.
    # The bundle also names its own extracted third-party banners; ship those with it.
    assert (_DOCS_UI / "NOTICE.swagger-ui").exists()
    assert (_DOCS_UI / "swagger-ui-bundle.js.LICENSE.txt").exists()
    for entry in _MANIFEST["packages"]:
        assert entry["version"] and entry["license"] and entry["source"]


def test_bundles_reference_no_remote_script_host():
    """The whole point is that nothing on the docs pages phones out for code."""
    for name in ("swagger-ui-bundle.js", "redoc.standalone.js"):
        text = (_DOCS_UI / name).read_text(encoding = "utf-8", errors = "ignore")
        assert "cdn.jsdelivr.net" not in text, f"{name} pulls from jsDelivr at runtime"
