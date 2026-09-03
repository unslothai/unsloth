# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Contract tests for clean-machine desktop release selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "resolve-desktop-release.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "desktop-app-clean-machine-ci.yml"


def _module():
    spec = importlib.util.spec_from_file_location("desktop_release_resolver", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release(
    tag: str,
    created: str,
    *assets: str,
    draft: bool = False,
):
    return {
        "tagName": tag,
        "createdAt": created,
        "isDraft": draft,
        "assets": [{"name": name} for name in assets],
    }


def test_resolver_selects_newest_semver_bundle_including_drafts():
    releases = [
        _release("v0.1.528-beta", "2026-08-10T01:00:00Z", "app.dmg"),
        _release("v0.1.529-beta", "2026-08-11T01:00:00Z", "app.dmg", draft = True),
        _release("v2026.8.11", "2026-08-12T01:00:00Z", "backend.whl"),
        _release("desktop-v0.1.530-beta", "2026-08-13T01:00:00Z", "app.dmg"),
    ]
    assert _module().resolve(releases, ".dmg") == "v0.1.529-beta"


def test_resolver_requires_the_platform_asset_and_fails_when_absent():
    releases = [
        _release("v0.1.529-beta", "2026-08-11T01:00:00Z", "app.exe"),
        _release("v0.1.528-beta", "2026-08-10T01:00:00Z", "app.AppImage"),
    ]
    resolver = _module().resolve
    assert resolver(releases, ".AppImage") == "v0.1.528-beta"
    assert resolver(releases, ".dmg") is None


def test_clean_machine_workflow_uses_resolver_but_preserves_explicit_tag():
    workflow = WORKFLOW.read_text(encoding = "utf-8")
    # One lookup per shipped-asset lane: .dmg, .deb, .AppImage, .exe.
    assert workflow.count("python3 .github/scripts/resolve-desktop-release.py") == 4
    assert "'.github/scripts/resolve-desktop-release.py'" in workflow
    assert workflow.count('if [ -z "$REL_TAG" ]; then') == 4
    assert 'startswith("desktop-v")' not in workflow
    for suffix in (".dmg", ".deb", ".AppImage", ".exe"):
        assert suffix in workflow


def test_the_resolver_stops_at_the_newest_release_holding_the_asset():
    # One lookup per release per matrix leg was the cost, and a transient failure
    # on an irrelevant older release failed the leg. Newest first, stop on match.
    looked_up: list[str] = []
    releases = [
        {"tagName": "v0.1.527-beta", "createdAt": "2026-03-01T00:00:00Z"},
        {"tagName": "v0.1.529-beta", "createdAt": "2026-05-01T00:00:00Z"},
        {"tagName": "v0.1.528-beta", "createdAt": "2026-04-01T00:00:00Z"},
        {"tagName": "not-a-release", "createdAt": "2026-06-01T00:00:00Z"},
    ]

    def fetch(tag):
        looked_up.append(tag)
        return [{"name": f"Unsloth-Desktop-{tag}-MacOS.dmg"}]

    assert _module().resolve_newest(releases, ".dmg", fetch) == "v0.1.529-beta"
    assert looked_up == ["v0.1.529-beta"]


def test_the_resolver_keeps_looking_past_a_release_without_the_asset():
    looked_up: list[str] = []
    releases = [
        {"tagName": "v0.1.528-beta", "createdAt": "2026-04-01T00:00:00Z"},
        {"tagName": "v0.1.529-beta", "createdAt": "2026-05-01T00:00:00Z"},
    ]

    def fetch(tag):
        looked_up.append(tag)
        return [] if tag == "v0.1.529-beta" else [{"name": "app-MacOS.dmg"}]

    assert _module().resolve_newest(releases, ".dmg", fetch) == "v0.1.528-beta"
    assert looked_up == ["v0.1.529-beta", "v0.1.528-beta"]
