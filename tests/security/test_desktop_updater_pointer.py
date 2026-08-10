# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Discovery must follow desktop metadata, not whichever release is newest.

Clients poll the repo-wide `/releases/latest/download/latest.json`. Publishing a
`v...` release that carries no desktop bundles moves that pointer, so the
publish workflow carries the newest desktop manifest onto it instead.
"""

import json
import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
UPDATER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-desktop-updater.yml"

REPOSITORY = "unslothai/unsloth"
STEP_NAME = "Carry desktop metadata forward"

FAKE_GH = """#!/bin/sh
set -eu
printf 'gh %s\\n' "$*" >> "$COMMAND_LOG"
if [ "$1" = "api" ]; then
  cat "$RELEASES_FIXTURE"
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
    # Only for a release that arrived without bundles of its own.
    assert "steps.gate.outputs.proceed == 'false'" in step["if"]
    # Nothing fires this workflow automatically any more, so the repair is
    # reachable only when a maintainer asks for it by name.
    assert "inputs.repair_pointer" in step["if"]
    return step["run"]


def _release(
    tag,
    *,
    has_manifest = True,
    draft = False,
    prerelease = False,
    published_at = "2026-01-01T00:00:00Z",
):
    return {
        "tag_name": tag,
        "draft": draft,
        "prerelease": prerelease,
        "published_at": published_at,
        "assets": [{"name": "latest.json"}] if has_manifest else [{"name": "notes.txt"}],
    }


def _manifest(tag):
    base = f"https://github.com/{REPOSITORY}/releases/download/{tag}/"
    return {
        "version": tag,
        "platforms": {
            "darwin-aarch64": {"url": f"{base}app.tar.gz", "signature": "c2ln"},
            "linux-x86_64": {"url": f"{base}app.AppImage", "signature": "c2ln"},
            "windows-x86_64": {"url": f"{base}app.exe", "signature": "c2ln"},
        },
    }


def _run(tmp_path, *, release_tag, releases, manifest):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(FAKE_GH, encoding = "utf-8")
    gh.chmod(0o755)

    releases_fixture = tmp_path / "releases.fixture.json"
    releases_fixture.write_text(json.dumps(releases), encoding = "utf-8")
    manifest_fixture = tmp_path / "manifest.fixture.json"
    manifest_fixture.write_text(json.dumps(manifest), encoding = "utf-8")
    log = tmp_path / "commands.log"
    log.write_text("", encoding = "utf-8")

    env = os.environ.copy()
    env.update(
        {
            "COMMAND_LOG": str(log),
            "GITHUB_REPOSITORY": REPOSITORY,
            "MANIFEST_FIXTURE": str(manifest_fixture),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RELEASES_FIXTURE": str(releases_fixture),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "step-summary.md"),
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


def test_the_newest_desktop_manifest_is_carried_onto_a_bundleless_release(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.52-beta", published_at = "2026-03-01T00:00:00Z"),
            _release("v0.1.51-beta", published_at = "2026-02-01T00:00:00Z"),
            _release("v0.1.53-beta", has_manifest = False, published_at = "2026-04-01T00:00:00Z"),
        ],
        manifest = _manifest("v0.1.52-beta"),
    )
    assert result.returncode == 0, result.stderr
    assert "gh release download v0.1.52-beta" in "\n".join(commands)
    assert any(
        line.startswith("gh release upload v0.1.53-beta") and "--clobber" in line
        for line in commands
    ), commands
    # The manifest is copied byte for byte, so it keeps pointing at its own release.
    carried = json.loads((tmp_path / "carry-forward" / "latest.json").read_text(encoding = "utf-8"))
    assert carried == _manifest("v0.1.52-beta")

    # The 404 gap between publish and this upload is recorded, not silent.
    summary = (tmp_path / "step-summary.md").read_text(encoding = "utf-8")
    assert "v0.1.52-beta manifest" in summary
    assert "404" in summary


def test_drafts_prereleases_and_bundleless_releases_are_never_the_source(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.52-beta", draft = True, published_at = "2026-03-01T00:00:00Z"),
            _release("v0.1.51-beta", prerelease = True, published_at = "2026-02-01T00:00:00Z"),
            _release("v0.1.50-beta", has_manifest = False, published_at = "2026-01-01T00:00:00Z"),
        ],
        manifest = _manifest("v0.1.52-beta"),
    )
    assert result.returncode == 0, result.stderr
    assert "nothing to carry forward" in result.stdout
    assert not [line for line in commands if line.startswith("gh release upload")]


def test_a_non_semver_release_leaves_the_pointer_alone(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "vnext",
        releases = [_release("v0.1.52-beta")],
        manifest = _manifest("v0.1.52-beta"),
    )
    assert result.returncode == 0, result.stderr
    assert commands == [], commands


def test_an_already_carried_manifest_is_carried_forward_again(tmp_path):
    # The carried release is now the newest one holding latest.json, and the
    # manifest it holds names an older version on purpose. Forwarding it again is
    # the only thing that keeps the pointer resolving.
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.54-beta",
        releases = [
            _release("v0.1.53-beta", published_at = "2026-04-01T00:00:00Z"),
            _release("v0.1.52-beta", published_at = "2026-03-01T00:00:00Z"),
        ],
        manifest = _manifest("v0.1.52-beta"),
    )
    assert result.returncode == 0, result.stderr
    assert any(line.startswith("gh release upload v0.1.54-beta") for line in commands), commands


def test_a_manifest_that_is_not_pinned_to_its_own_version_is_refused(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [_release("v0.1.52-beta")],
        manifest = {
            "version": "v0.1.52-beta",
            "platforms": {
                "linux-x86_64": {
                    "url": f"https://github.com/{REPOSITORY}/releases/latest/download/app.AppImage",
                    "signature": "c2ln",
                }
            },
        },
    )
    assert result.returncode == 1
    assert "not pinned to v0.1.52-beta" in result.stderr
    assert not [line for line in commands if line.startswith("gh release upload")]


def test_a_manifest_without_a_usable_version_is_refused(tmp_path):
    result, _ = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [_release("v0.1.52-beta")],
        manifest = {"version": "latest", "platforms": {}},
    )
    assert result.returncode == 1
    assert "refusing to carry it forward" in result.stderr
