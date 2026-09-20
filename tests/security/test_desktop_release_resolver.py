# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Contract tests for clean-machine desktop release selection."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
import yaml


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


@pytest.mark.parametrize(
    "suffix,mode,tag,returncode,unreleased",
    [
        ("Ubuntu-ARM64.deb", "absent", "", 0, True),
        ("Ubuntu.deb", "absent", "", 1, False),
        ("Linux.AppImage", "absent", "", 1, False),
        ("Ubuntu-ARM64.deb", "list-error", "", 1, False),
        ("Ubuntu-ARM64.deb", "view-error", "", 1, False),
        ("Ubuntu-ARM64.deb", "present", "", 0, False),
        ("Ubuntu-ARM64.deb", "absent", "v0.1.811-beta", 1, False),
        ("Ubuntu-ARM64.deb", "present", "v0.1.811-beta", 0, False),
    ],
)
def test_linux_download_skips_only_unreleased_arm64(
    tmp_path, suffix, mode, tag, returncode, unreleased
):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    download = next(
        step
        for step in workflow["jobs"]["linux"]["steps"]
        if step.get("name") == "Download the shipped bundle"
    )
    script = tmp_path / ".github/scripts/resolve-desktop-release.py"
    script.parent.mkdir(parents = True)
    script.write_text(SCRIPT.read_text())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text("""#!/bin/sh
set -eu
case "$2" in
  list)
    [ "$MODE" != 'list-error' ] || exit 1
    printf '%s\n' '[{"tagName":"v0.1.811-beta","createdAt":"2026-09-20T00:00:00Z"}]'
    ;;
  view)
    [ "$MODE" != 'view-error' ] || exit 1
    if [ "$MODE" = 'present' ]; then
      printf '{"assets":[{"name":"Unsloth-Desktop-%s"}]}\n' "$SUFFIX"
    else
      printf '%s\n' '{"assets":[]}'
    fi
    ;;
  download) [ "$MODE" = 'present' ] ;;
  *) exit 2 ;;
esac
""")
    gh.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", download["run"].replace("${{ matrix.asset }}", suffix)],
        cwd = tmp_path,
        env = {
            **os.environ,
            "PATH": f'{fake_bin}:{os.environ["PATH"]}',
            "MODE": mode,
            "SUFFIX": suffix,
            "REL_TAG": tag,
            "REL_REPO": "test/desktop",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_ENV": str(tmp_path / "env"),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output = True,
        text = True,
        check = False,
    )
    assert result.returncode == returncode, result.stdout + result.stderr
    assert (output.exists() and "unreleased=true" in output.read_text()) == unreleased
