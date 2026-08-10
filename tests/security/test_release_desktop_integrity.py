"""Checks that one desktop version tag can only ever serve one set of binaries."""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"

RELEASE_TAG = "v0.1.50-beta"
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
  include=0
  endpoint=""
  for argument in "$@"; do
    case "$argument" in
      --include) include=1 ;;
      repos/*) endpoint="$argument" ;;
    esac
  done
  case "$endpoint" in
    */commits/*) printf '%s\n' "$SOURCE_COMMIT_SHA"; exit 0 ;;
    */releases/tags/*) status="$TARGET_HTTP_STATUS" ;;
    *) exit 0 ;;
  esac
  if [ "$include" = "1" ]; then
    printf 'HTTP/2.0 %s Test Response\n' "$status"
  fi
  if [ "$status" = "200" ]; then
    if [ "$TARGET_HAS_DESKTOP_ASSETS" = "1" ]; then
      printf '{"tag_name":"%s","draft":false,"assets":[{"name":"latest.json"}]}\n' "$DESKTOP_RELEASE_TAG"
    else
      printf '{"tag_name":"%s","draft":false,"assets":[]}\n' "$DESKTOP_RELEASE_TAG"
    fi
    exit 0
  fi
  exit 1
fi

if [ "$1" = "release" ] && [ "$2" = "download" ]; then
  if [ "$TARGET_HAS_DESKTOP_ASSETS" != "1" ]; then
    echo "release not found" >&2
    exit 1
  fi
  directory=""
  want_directory=0
  for argument in "$@"; do
    if [ "$want_directory" = "1" ]; then directory="$argument"; want_directory=0; continue; fi
    [ "$argument" = "--dir" ] && want_directory=1
  done
  [ -n "$directory" ] || directory="."
  mkdir -p "$directory"
  printf '{"version":"%s","platforms":{}}\n' "$TARGET_MANIFEST_VERSION" > "$directory/latest.json"
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
    target_http_status: int = 200,
    target_has_desktop_assets: bool = False,
    target_manifest_version: str = RELEASE_TAG,
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
            "ASSET_VERSION": "0_1_50_beta",
            "RUNNER_TEMP": str(tmp_path),
            "SOURCE_COMMIT_SHA": SOURCE_SHA,
            "TARGET_HAS_DESKTOP_ASSETS": "1" if target_has_desktop_assets else "0",
            "TARGET_HTTP_STATUS": str(target_http_status),
            "TARGET_MANIFEST_VERSION": target_manifest_version,
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
    signature = base64.b64encode(
        b"untrusted comment: signature from tauri secret key\n"
        b"test signature bytes\n"
        b"trusted comment: timestamp:1\tfile:test\n"
        b"test global signature bytes\n"
    )
    digests = {}
    for name, payload in (
        ("Unsloth-Desktop-0_1_50_beta-MacOS.dmg", b"disk image"),
        ("Unsloth-Desktop-0_1_50_beta-Ubuntu.deb", b"package"),
        ("Unsloth-Desktop-0_1_50_beta-ARM64.app.tar.gz", b"mac updater"),
        ("Unsloth-Desktop-0_1_50_beta-ARM64.app.tar.gz.sig", signature),
        ("Unsloth-Desktop-0_1_50_beta-Linux.AppImage", b"linux updater"),
        ("Unsloth-Desktop-0_1_50_beta-Linux.AppImage.sig", signature),
        ("Unsloth-Desktop-0_1_50_beta-Windows.exe", b"installer"),
        ("Unsloth-Desktop-0_1_50_beta-Windows.exe.sig", signature),
    ):
        (asset_dir / name).write_bytes(payload)
        if not name.endswith(".sig"):
            digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def _run_create_release(
    workflow,
    tmp_path: Path,
    *,
    invalid_signature = False,
    **kwargs,
):
    _stage_assets(tmp_path)
    if invalid_signature:
        (
            tmp_path / "desktop-release-assets" / "Unsloth-Desktop-0_1_50_beta-Linux.AppImage.sig"
        ).write_text("Tauri signer diagnostic, not a signature\n", encoding = "utf-8")
    env = {
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
        "Record desktop build provenance on the release",
    )
    create_step = _step(
        workflow, "publish-release", "Record desktop build provenance on the release"
    )
    create_step["run"] = "\n".join(
        _step(workflow, "publish-release", name)["run"] for name in names
    )
    return _run_step(
        workflow,
        "publish-release",
        "Record desktop build provenance on the release",
        tmp_path,
        extra_env = env,
        **kwargs,
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
        ({"target_has_desktop_assets": True}, 1),
        ({"target_http_status": 404}, 1),
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


def test_a_missing_target_release_says_how_to_create_it(tmp_path):
    workflow = _workflow()
    result, _ = _run_step(
        workflow,
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
        target_http_status = 404,
    )
    assert result.returncode == 1
    assert f"Release {RELEASE_TAG} does not exist." in result.stderr
    assert "Tag main and publish it first" in result.stderr


def test_existing_desktop_assets_name_the_cleanup_command(tmp_path):
    workflow = _workflow()
    result, _ = _run_step(
        workflow,
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
        target_has_desktop_assets = True,
    )
    assert result.returncode == 1
    assert f"gh release delete-asset {RELEASE_TAG} latest.json --yes" in result.stderr


def test_a_failed_guard_probe_fails_closed_before_any_build_work(tmp_path):
    workflow = _workflow()
    result, _ = _run_step(
        workflow,
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
        target_http_status = 500,
    )
    assert result.returncode == 1
    assert "Could not read release" in result.stderr


def test_publish_refuses_to_reuse_an_existing_release(tmp_path):
    workflow = _workflow()
    result, commands = _run_create_release(workflow, tmp_path, target_has_desktop_assets = True)
    assert result.returncode == 1
    assert "Refusing to republish" in result.stderr
    assert f"gh release delete-asset {RELEASE_TAG} latest.json --yes" in result.stderr
    assert not [line for line in commands if line.startswith("gh release create")]


def test_publish_fails_closed_when_the_target_release_is_missing(tmp_path):
    workflow = _workflow()
    result, commands = _run_create_release(workflow, tmp_path, target_http_status = 404)
    assert result.returncode == 1
    assert f"Release {RELEASE_TAG} does not exist." in result.stderr
    assert not [line for line in commands if line.startswith("gh release create")]


def test_publish_rejects_signer_diagnostics_as_updater_signatures(tmp_path):
    workflow = _workflow()
    result, commands = _run_create_release(workflow, tmp_path, invalid_signature = True)
    assert result.returncode == 1
    assert "Invalid base64 updater signature" in result.stderr
    assert not [line for line in commands if line.startswith("gh release create")]


def test_release_body_records_provenance_the_updater_notes_do_not_carry(tmp_path):
    workflow = _workflow()
    digests = _stage_assets(tmp_path)
    result, commands = _run_create_release(workflow, tmp_path)
    assert result.returncode == 0, result.stderr

    # The release already exists, so nothing is created and no tag is reserved.
    assert not [line for line in commands if line.startswith("gh release create")]
    assert not [line for line in commands if "git/refs" in line]
    assert any(line.startswith("gh release edit") for line in commands)

    body_file = tmp_path / "desktop-release-body.md"
    body = body_file.read_text(encoding = "utf-8")
    assert SOURCE_SHA in body
    for name, digest in digests.items():
        assert f"{digest}  {name}" in body
    latest = tmp_path / "latest.json"
    metadata = yaml.safe_load(latest.read_text(encoding = "utf-8"))
    for platform in metadata["platforms"].values():
        decoded = base64.b64decode(platform["signature"], validate = True)
        assert decoded.startswith(b"untrusted comment:")
        assert b"\ntrusted comment:" in decoded
    assert latest.is_file()
    assert f"{hashlib.sha256(latest.read_bytes()).hexdigest()}  latest.json" in body
    assert ".sig" not in body

    # Keep digests out of updater notes, and out of the changelog we append to.
    notes = (tmp_path / "desktop-release-notes.md").read_text(encoding = "utf-8")
    assert "Build provenance" not in notes
    assert "Desktop app for Unsloth." in notes
    assert "Desktop app for Unsloth." not in body


def test_versioned_uploads_never_clobber_or_mutate_the_legacy_channel():
    uploads = _upload_commands(_workflow())
    versioned = [line for line in uploads if "$DESKTOP_RELEASE_TAG" in line]
    channel = [line for line in uploads if "desktop-latest" in line]
    assert len(versioned) == 2, uploads
    assert channel == [], uploads

    # latest.json is the moving updater pointer and may already hold a carried
    # forward manifest, so only it may be replaced. Bundles stay immutable.
    for line in versioned:
        if "latest.json" not in line:
            assert "--clobber" not in line, line


def test_a_carried_forward_manifest_does_not_block_the_guard(tmp_path):
    workflow = _workflow()
    result, _ = _run_step(
        workflow,
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
        target_has_desktop_assets = True,
        target_manifest_version = "v0.1.49-beta",
    )
    assert result.returncode == 0, result.stderr


def test_a_validation_only_run_touches_nothing_public():
    steps = _workflow()["jobs"]["publish-release"]["steps"]
    names = [step.get("name") for step in steps]
    mutating = (
        "Publish versioned release assets",
        "Publish versioned updater metadata",
        "Record desktop build provenance on the release",
        "Promote normal release to GitHub latest",
    )
    for name in mutating:
        step = steps[names.index(name)]
        assert step.get("if") == "${{ !inputs.draft }}", name

    # Provenance last, so a retry after a partial upload records what shipped.
    for upload in mutating[:2]:
        assert names.index(upload) < names.index(mutating[2])


def test_the_guard_rejects_a_prerelease_target_before_anything_is_built():
    workflow = _workflow()
    guard = _step(workflow, "prepare-version", "Guard against republishing an existing version")
    assert "is a prerelease" in guard["run"]
    # And again in publish-release, which is the one holding write scope.
    state = _step(workflow, "publish-release", "Validate versioned release state")
    assert "is a prerelease" in state["run"]


def test_the_build_uses_the_release_tag_not_the_dispatch_ref():
    build = _workflow()["jobs"]["build"]["steps"]
    checkout = next(s for s in build if "actions/checkout" in str(s.get("uses", "")))
    assert checkout["with"]["ref"] == "${{ needs.prepare-version.outputs.desktop_release_tag }}"


def test_the_tag_is_validated_before_it_is_checked_out(tmp_path):
    # actions/checkout resolves the free-text input, so a malformed tag would fail
    # on a generic missing-ref error and none of the corrections would be printed.
    steps = _workflow()["jobs"]["prepare-version"]["steps"]
    names = [step.get("name") or str(step.get("uses")) for step in steps]
    checkout = next(
        i for i, step in enumerate(steps) if "actions/checkout" in str(step.get("uses", ""))
    )
    assert names.index("Validate release versions") < checkout, names
    # And the checkout uses the validated value, not the raw input.
    assert steps[checkout]["with"]["ref"] == "${{ steps.prepare.outputs.studio_version }}"

    for index, (bad, expected) in enumerate(
        (
            ("v.0.1.52-beta", "did you mean v0.1.52-beta?"),
            ("0.1.52-beta", "must start with v"),
            ("2026.8.3", "not a date-style backend version"),
        )
    ):
        case_dir = tmp_path / f"case-{index}"
        case_dir.mkdir()
        result, _ = _run_step(
            _workflow(),
            "prepare-version",
            "Validate release versions",
            case_dir,
            extra_env = {"INPUT_STUDIO_VERSION": bad},
        )
        assert result.returncode == 1, bad
        assert expected in result.stderr, (bad, result.stderr)


def test_the_promotion_guard_orders_numbered_prereleases_by_number():
    guard = _step(_workflow(), "publish-release", "Promote normal release to GitHub latest")["run"]
    body = guard.split('python3 - "$latest_before"', 1)[1].split("\nPY", 1)[0]
    body = "\n".join(line[10:] if line.startswith(" " * 10) else line for line in body.split("\n"))
    body = body.split("\n", 1)[1].lstrip("\n")
    namespace: dict = {}
    exec(body.split("current = json.loads", 1)[0], namespace)
    key = namespace["key"]
    # v1.2.3-beta10 is newer than v1.2.3-beta2, and a release beats its prerelease.
    assert key("v1.2.3-beta10") > key("v1.2.3-beta2")
    assert key("v1.2.3") > key("v1.2.3-beta10")
    assert key("v0.1.527-beta") > key("v0.1.526-beta")
    assert key("not-a-tag") is None


def test_the_promotion_guard_fails_closed_on_a_failed_latest_lookup():
    guard = _step(_workflow(), "publish-release", "Promote normal release to GitHub latest")["run"]
    # A 404 means no latest yet; anything else must stop before the PATCH.
    fallback = guard.split("elif grep -Fq '(HTTP 404)'", 1)[1].split("gh api --method PATCH", 1)[0]
    assert "refusing to promote" in fallback.lower()
    assert "exit 1" in fallback
    assert "2>/dev/null" not in guard.split("releases/latest", 1)[1].split("\n", 1)[0]
