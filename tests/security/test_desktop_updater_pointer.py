# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Desktop discovery must stay on a release with one complete asset set."""

import json
import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
UPDATER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-desktop-updater.yml"

REPOSITORY = "unslothai/unsloth"
STEP_NAME = "Restore latest complete Desktop release"
STABLE_ASSETS = (
    "Unsloth-Desktop-MacOS.dmg",
    "Unsloth-Desktop-Linux.AppImage",
    "Unsloth-Desktop-Ubuntu.deb",
    "Unsloth-Desktop-Windows.exe",
)

FAKE_GH = """#!/bin/sh
set -eu
printf 'gh %s\\n' "$*" >> "$COMMAND_LOG"
if [ "$1" = "api" ]; then
  case "$*" in
    *"releases?per_page=100"*) cat "$RELEASES_FIXTURE" ;;
    *"/releases/latest"*) printf '%s\\n' "$LATEST_TAG" ;;
    *"--method PATCH"*) printf '{"make_latest":true}\\n' ;;
    *) exit 1 ;;
  esac
  exit 0
fi
if [ "$1" = "release" ] && [ "$2" = "download" ]; then
  directory=""
  want_directory=0
  for argument in "$@"; do
    if [ "$want_directory" = "1" ]; then directory="$argument"; want_directory=0; continue; fi
    [ "$argument" = "--dir" ] && want_directory=1
  done
  mkdir -p "$directory"
  cp "$MANIFEST_FIXTURE" "$directory/latest.json"
  exit 0
fi
exit 0
"""


def _step_run():
    job = yaml.safe_load(UPDATER_WORKFLOW.read_text(encoding = "utf-8"))["jobs"]["publish-updater"]
    steps = {step.get("name"): step for step in job["steps"]}
    assert STEP_NAME in steps, sorted(steps)
    step = steps[STEP_NAME]
    assert "steps.gate.outputs.proceed == 'false'" in step["if"]
    assert "inputs.repair_pointer" in step["if"]
    return step["run"]


def _release(
    tag,
    *,
    release_id,
    complete = True,
    draft = False,
    prerelease = False,
    published_at = "2026-01-01T00:00:00Z",
    legacy = False,
):
    assets = [{"name": name} for name in ("latest.json", *STABLE_ASSETS)]
    if legacy:
        version = tag.removeprefix("v").replace(".", "_").replace("-", "_")
        assets = [
            {"name": "latest.json"},
            *(
                {"name": f"Unsloth-Desktop-{version}-{suffix}"}
                for suffix in ("MacOS.dmg", "Linux.AppImage", "Ubuntu.deb", "Windows.exe")
            ),
        ]
    if not complete:
        assets = [{"name": "notes.txt"}]
    return {
        "id": release_id,
        "tag_name": tag,
        "draft": draft,
        "prerelease": prerelease,
        "published_at": published_at,
        "assets": assets,
    }


def _manifest(tag):
    base = f"https://github.com/{REPOSITORY}/releases/download/{tag}/"
    return {
        "version": tag,
        "platforms": {
            "darwin-aarch64": {
                "url": f"{base}Unsloth-Desktop-ARM64.app.tar.gz",
                "signature": "c2ln",
            },
            "linux-x86_64": {
                "url": f"{base}Unsloth-Desktop-Linux.AppImage",
                "signature": "c2ln",
            },
            "windows-x86_64": {
                "url": f"{base}Unsloth-Desktop-Windows.exe",
                "signature": "c2ln",
            },
        },
    }


def _run(
    tmp_path,
    *,
    release_tag,
    releases,
    manifest,
    latest_tag = "v0.1.52-beta",
):
    tmp_path.mkdir(parents = True, exist_ok = True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(FAKE_GH, encoding = "utf-8")
    gh.chmod(0o755)

    releases_fixture = tmp_path / "releases.fixture.json"
    releases_fixture.write_text(json.dumps(releases), encoding = "utf-8")
    manifest_fixture = tmp_path / "latest.fixture.json"
    manifest_fixture.write_text(json.dumps(manifest), encoding = "utf-8")
    log = tmp_path / "commands.log"
    log.write_text("", encoding = "utf-8")
    env = os.environ.copy()
    env.update(
        {
            "COMMAND_LOG": str(log),
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_STEP_SUMMARY": str(tmp_path / "step-summary.md"),
            "LATEST_TAG": latest_tag,
            "MANIFEST_FIXTURE": str(manifest_fixture),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RELEASES_FIXTURE": str(releases_fixture),
            "RELEASE_TAG": release_tag,
            "RUNNER_TEMP": str(tmp_path),
        }
    )
    result = subprocess.run(
        ["bash", "-c", _step_run()],
        cwd = tmp_path,
        env = env,
        text = True,
        capture_output = True,
        check = False,
    )
    return result, log.read_text(encoding = "utf-8").splitlines()


def test_the_newest_complete_desktop_release_is_restored_without_copying_assets(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.52-beta", release_id = 52, published_at = "2026-03-01T00:00:00Z"),
            _release("v0.1.51-beta", release_id = 51, published_at = "2026-02-01T00:00:00Z"),
            _release(
                "v0.1.53-beta",
                release_id = 53,
                complete = False,
                published_at = "2026-04-01T00:00:00Z",
            ),
        ],
        manifest = _manifest("v0.1.52-beta"),
    )
    assert result.returncode == 0, result.stderr
    assert "gh release download v0.1.52-beta --pattern latest.json" in "\n".join(commands)
    assert any(
        line.startswith(f"gh api --method PATCH repos/{REPOSITORY}/releases/52")
        and "-f make_latest=true" in line
        for line in commands
    ), commands
    assert not [line for line in commands if line.startswith("gh release upload")]

    restored = json.loads((tmp_path / "restore-latest" / "latest.json").read_text(encoding = "utf-8"))
    assert restored == _manifest("v0.1.52-beta")
    summary = (tmp_path / "step-summary.md").read_text(encoding = "utf-8")
    assert "points back to v0.1.52-beta" in summary
    assert "404" in summary


def test_legacy_downloads_are_restored_during_migration(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.802-beta",
        releases = [
            _release("v0.1.802-beta", release_id = 802, complete = False),
            _release("v0.1.801-beta", release_id = 801, legacy = True),
        ],
        manifest = _manifest("v0.1.801-beta"),
        latest_tag = "v0.1.801-beta",
    )
    assert result.returncode == 0, result.stderr
    assert "gh release download v0.1.801-beta --pattern latest.json" in "\n".join(commands)
    assert any("releases/801 -f make_latest=true" in line for line in commands)


def test_incomplete_draft_and_prerelease_releases_are_never_restored(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52, draft = True),
            _release("v0.1.51-beta", release_id = 51, prerelease = True),
            _release("v0.1.50-beta", release_id = 50, complete = False),
        ],
        manifest = _manifest("v0.1.52-beta"),
    )
    assert result.returncode == 0, result.stderr
    assert "nothing to restore" in result.stdout
    assert not [line for line in commands if "--method PATCH" in line]


def test_a_non_semver_release_leaves_latest_alone(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "vnext",
        releases = [_release("v0.1.52-beta", release_id = 52)],
        manifest = _manifest("v0.1.52-beta"),
    )
    assert result.returncode == 0, result.stderr
    assert commands == [], commands


def test_a_manifest_naming_another_release_is_refused(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.54-beta",
        releases = [
            _release("v0.1.54-beta", release_id = 54, complete = False),
            _release("v0.1.53-beta", release_id = 53),
            _release("v0.1.52-beta", release_id = 52),
        ],
        manifest = _manifest("v0.1.52-beta"),
    )
    assert result.returncode == 1
    assert "names v0.1.52-beta" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]


def test_a_manifest_with_a_moving_bundle_url_is_refused(tmp_path):
    manifest = _manifest("v0.1.52-beta")
    manifest["platforms"]["linux-x86_64"]["url"] = (
        f"https://github.com/{REPOSITORY}/releases/latest/download/Unsloth-Desktop-Linux.AppImage"
    )
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52),
        ],
        manifest = manifest,
    )
    assert result.returncode == 1
    assert "URL is not pinned to its release" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]


def test_a_manifest_without_a_usable_version_is_refused(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52),
        ],
        manifest = {"version": "latest", "platforms": {}},
    )
    assert result.returncode == 1
    assert "declares invalid version" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]


def test_a_draft_or_prerelease_target_is_refused(tmp_path):
    for state in ("draft", "prerelease"):
        target = _release("v0.1.53-beta", release_id = 53, complete = False)
        target[state] = True
        result, commands = _run(
            tmp_path / state,
            release_tag = "v0.1.53-beta",
            releases = [target, _release("v0.1.52-beta", release_id = 52)],
            manifest = _manifest("v0.1.52-beta"),
        )
        assert result.returncode != 0, f"{state} target was accepted"
        assert f"is a {state}" in result.stderr, result.stderr
        assert not [line for line in commands if "--method PATCH" in line], commands


def test_a_target_missing_from_the_release_listing_is_refused(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [_release("v0.1.52-beta", release_id = 52)],
        manifest = _manifest("v0.1.52-beta"),
    )
    assert result.returncode != 0, result.stdout
    assert "is not among the 100 most recent releases" in result.stderr, result.stderr
    assert not [line for line in commands if "--method PATCH" in line], commands
