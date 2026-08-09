"""Checks that one desktop version tag can only ever serve one set of binaries."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"

RELEASE_TAG = "desktop-v0.1.50-beta"
SOURCE_SHA = "1f02275b86f0e0d3a5b1c9f2a4d6e8b0c2a4e6f8"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def _step_index(workflow, job, name):
    """Locate a step and report its available names on failure."""
    names = [step.get("name") for step in _steps(workflow, job)]
    assert name in names, f"{job} has no step named {name!r}; steps are {names}"
    return names.index(name)


def _step(workflow, job, name):
    return _steps(workflow, job)[_step_index(workflow, job, name)]


def _write_fake_gh(path: Path):
    """Record gh arguments and return configured statuses."""
    path.write_text(
        """#!/bin/sh
set -eu
printf 'gh %s\\n' "$*" >> "$COMMAND_LOG"
if [ "$1" = "api" ]; then
  endpoint=""
  for argument in "$@"; do
    case "$argument" in repos/*) endpoint="$argument" ;; esac
  done
  case "$endpoint" in
    */git/ref/tags/*) status="$TAG_HTTP_STATUS" ;;
    */releases/tags/*) status="$RELEASE_HTTP_STATUS" ;;
    *) exit 0 ;;
  esac
  printf 'HTTP/2.0 %s Test Response\n' "$status"
  if [ "$status" = "200" ]; then exit 0; fi
  exit 1
fi
if [ "$1" = "release" ] && [ "$2" = "list" ]; then
  if [ "$RELEASE_HTTP_STATUS" = "500" ]; then
    printf 'GraphQL test failure\n' >&2
    exit 1
  fi
  if [ "$RELEASE_HTTP_STATUS" = "200" ]; then
    printf '[{"tagName":"%s"}]\n' "$DESKTOP_RELEASE_TAG"
  else
    printf '[]\n'
  fi
  exit 0
fi

exit 0
""",
        encoding = "utf-8",
    )
    path.chmod(0o755)


def _run_step(
    workflow,
    job: str,
    name: str,
    tmp_path: Path,
    *,
    tag_http_status: int = 404,
    release_http_status: int = 404,
    extra_env: dict[str, str] | None = None,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok = True)
    _write_fake_gh(fake_bin / "gh")
    log = tmp_path / "commands.log"
    log.write_text("", encoding = "utf-8")

    env = os.environ.copy()
    env.update(
        {
            "COMMAND_LOG": str(log),
            "DESKTOP_RELEASE_TAG": RELEASE_TAG,
            "GH_REPO": "unslothai/unsloth",
            "GITHUB_OUTPUT": str(tmp_path / "github-output"),
            "GH_TOKEN": "masked-token",
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RELEASE_HTTP_STATUS": str(release_http_status),
            "RUNNER_TEMP": str(tmp_path),
            "TAG_HTTP_STATUS": str(tag_http_status),
        }
    )
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", "-c", _step(workflow, job, name)["run"]],
        cwd = tmp_path,
        env = env,
        text = True,
        capture_output = True,
        check = False,
    )
    return result, log.read_text(encoding = "utf-8").splitlines()


def _stage_assets(tmp_path: Path) -> dict[str, str]:
    """Create release assets and return their digests."""
    asset_dir = tmp_path / "desktop-release-assets"
    asset_dir.mkdir(exist_ok = True)
    digests = {}
    for name, payload in (
        ("Unsloth-Desktop-0_1_50_beta-MacOS.dmg", b"disk image"),
        ("Unsloth-Desktop-0_1_50_beta-Ubuntu.deb", b"package"),
        ("Unsloth-Desktop-0_1_50_beta-ARM64.app.tar.gz", b"mac updater"),
        ("Unsloth-Desktop-0_1_50_beta-ARM64.app.tar.gz.sig", b"mac signature"),
        ("Unsloth-Desktop-0_1_50_beta-Linux.AppImage", b"linux updater"),
        ("Unsloth-Desktop-0_1_50_beta-Linux.AppImage.sig", b"linux signature"),
        ("Unsloth-Desktop-0_1_50_beta-Windows.exe", b"installer"),
        ("Unsloth-Desktop-0_1_50_beta-Windows.exe.sig", b"windows signature"),
    ):
        (asset_dir / name).write_bytes(payload)
        if not name.endswith(".sig"):
            digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def _run_create_release(workflow, tmp_path: Path, **kwargs):
    _stage_assets(tmp_path)
    env = {
        "DESKTOP_PRERELEASE": "true",
        "DESKTOP_RELEASE_NOTES": workflow["env"]["DESKTOP_RELEASE_NOTES"],
        "APP_VERSION": "0.1.50",
        "GITHUB_SHA": SOURCE_SHA,
        "GITHUB_REPOSITORY": "unslothai/unsloth",
        "PYPI_VERSION": "2026.8.7",
        "RELEASE_DRAFT": "true",
        "STUDIO_VERSION": "v0.1.50-beta",
    }
    env.update(kwargs.pop("extra_env", None) or {})

    # Execute the production publish sequence in one shell so the notes,
    # metadata and provenance files cross the same step boundaries as Actions.
    names = (
        "Validate versioned release state",
        "Generate versioned updater metadata and provenance",
        "Create versioned release",
    )
    create_step = _step(workflow, "publish-release", "Create versioned release")
    create_step["run"] = "\n".join(
        _step(workflow, "publish-release", name)["run"] for name in names
    )
    return _run_step(
        workflow, "publish-release", "Create versioned release", tmp_path, extra_env = env, **kwargs
    )


def _upload_commands(workflow):
    commands = []
    for step in _steps(workflow, "publish-release"):
        # Join backslash continuations so a flag parked on the next line counts.
        for line in step.get("run", "").replace("\\\n", " ").splitlines():
            stripped = line.strip()
            if stripped.startswith("gh release upload"):
                commands.append(stripped)
    return commands


def test_a_used_version_fails_the_guard_before_any_build_work(tmp_path):
    workflow = _workflow()
    # Fail before the build matrix and notarization.
    assert _step_index(
        workflow, "prepare-version", "Guard against republishing an existing version"
    ) < _step_index(workflow, "prepare-version", "Verify PyPI package and Unsloth stamp")
    assert workflow["jobs"]["build"]["needs"] == "prepare-version"

    for case, expected in (
        ({"tag_http_status": 200}, 1),
        ({"release_http_status": 200}, 1),
        ({}, 0),
    ):
        case_dir = tmp_path / ("-".join(case) or "unused-version")
        case_dir.mkdir()
        result, _ = _run_step(
            workflow,
            "prepare-version",
            "Guard against republishing an existing version",
            case_dir,
            **case,
        )
        assert result.returncode == expected, (case, result.stderr)
        if expected:
            assert RELEASE_TAG in result.stderr
            # Avoid leaving a tag that fails the next attempt.
            assert f"gh release delete {RELEASE_TAG} --cleanup-tag" in result.stderr
            assert (
                f"gh api --method DELETE repos/unslothai/unsloth/git/refs/tags/{RELEASE_TAG}"
                in result.stderr
            )


def test_a_failed_guard_probe_fails_closed_before_any_build_work(tmp_path):
    workflow = _workflow()
    result, _ = _run_step(
        workflow,
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
        tag_http_status = 500,
    )
    assert result.returncode == 1
    assert "Could not verify that tag" in result.stderr


def test_publish_refuses_to_reuse_an_existing_release(tmp_path):
    workflow = _workflow()
    # The write-scoped publish job can see drafts that prepare-version cannot.
    for case in ({"tag_http_status": 200}, {"release_http_status": 200}):
        case_dir = tmp_path / "-".join(case)
        case_dir.mkdir()
        result, commands = _run_create_release(workflow, case_dir, **case)
        assert result.returncode == 1, case
        assert "Refusing to republish" in result.stderr
        assert f"gh release delete {RELEASE_TAG} --cleanup-tag" in result.stderr
        assert (
            f"gh api --method DELETE repos/unslothai/unsloth/git/refs/tags/{RELEASE_TAG}"
            in result.stderr
        )
        assert not [line for line in commands if line.startswith("gh release create")]


def test_publish_fails_closed_when_a_guard_probe_errors(tmp_path):
    workflow = _workflow()
    result, commands = _run_create_release(workflow, tmp_path, tag_http_status = 500)
    assert result.returncode == 1
    assert "Could not verify that tag" in result.stderr
    assert not [line for line in commands if line.startswith("gh release create")]


def test_publish_fails_closed_when_draft_listing_errors(tmp_path):
    workflow = _workflow()
    result, commands = _run_create_release(workflow, tmp_path, release_http_status = 500)
    assert result.returncode == 1
    assert "Could not list releases, including drafts" in result.stderr
    assert not [line for line in commands if line.startswith("gh release create")]


def test_release_body_records_provenance_the_updater_notes_do_not_carry(tmp_path):
    workflow = _workflow()
    digests = _stage_assets(tmp_path)
    result, commands = _run_create_release(workflow, tmp_path)
    assert result.returncode == 0, result.stderr

    create = next(line for line in commands if line.startswith("gh release create"))
    assert RELEASE_TAG in create
    assert "--target" not in create
    assert "--verify-tag" in create
    assert any("git/refs" in line and f"sha={SOURCE_SHA}" in line for line in commands)

    body_file = tmp_path / "desktop-release-body.md"
    assert f"--notes-file {body_file}" in create
    body = body_file.read_text(encoding = "utf-8")
    assert SOURCE_SHA in body
    for name, digest in digests.items():
        assert f"{digest}  {name}" in body
    latest = tmp_path / "latest.json"
    assert latest.is_file()
    assert f"{hashlib.sha256(latest.read_bytes()).hexdigest()}  latest.json" in body
    assert ".sig" not in body

    # Keep digests out of updater notes.
    notes = (tmp_path / "desktop-release-notes.md").read_text(encoding = "utf-8")
    assert "Build provenance" not in notes
    assert "Desktop app for Unsloth." in notes
    create_step = _step(workflow, "publish-release", "Create versioned release")
    assert "'desktop-release-notes.md'" in create_step["run"]
    assert "latest.json" in create_step["run"]


def test_versioned_uploads_never_clobber_but_the_channel_pointer_does():
    uploads = _upload_commands(_workflow())
    versioned = [line for line in uploads if "$DESKTOP_RELEASE_TAG" in line]
    channel = [line for line in uploads if "desktop-latest" in line]
    assert len(versioned) == 2, uploads
    assert len(channel) == 1, uploads

    for line in versioned:
        assert "--clobber" not in line, line
    # The channel pointer is intentionally mutable.
    assert "--clobber" in channel[0]
