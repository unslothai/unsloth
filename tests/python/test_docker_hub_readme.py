# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The Docker Hub page is repository metadata, not something a push writes, so it
only changes when the publish workflow PATCHes it. These pin that the README in the
tree describes the images that actually ship and that the sync step cannot report
success while the page stays stale.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-publish.yml"
HUB_README = REPO_ROOT / "docker" / "DOCKERHUB.md"
REPO_README = REPO_ROOT / "README.md"


def test_the_hub_readme_describes_the_shipped_images():
    text = HUB_README.read_text(encoding = "utf-8")
    for needle in (
        "unsloth/unsloth:core",
        "`latest`",
        "linux/arm64",
        "jupyter lab --ip 0.0.0.0 --port 8888 --allow-root",
        "UNSLOTH_ALLOW_CPU=1",
        "/workspace/host",
        "/workspace/.cache/huggingface",
        "sm_75 sm_80 sm_86 sm_90 sm_100 sm_120",
    ):
        assert needle in text, f"the Hub README no longer mentions {needle!r}"
    # the previous image's conventions, none of which exist in this one
    for stale in ("USER_PASSWORD", "/workspace/work", "2222:22", "localhot"):
        assert stale not in text, f"the Hub README still carries {stale!r} from the old image"


def test_the_repo_readme_run_command_matches_the_image():
    text = REPO_README.read_text(encoding = "utf-8")
    start = text.index("#### Docker")
    section = text[start : text.index("####", start + 1)]
    assert "unsloth/unsloth:core" in section
    assert "/workspace/host" in section
    assert "--ipc=host" in section
    for stale in ("2222:22", "/workspace/work"):
        assert stale not in section, f"the README run command still has {stale!r}"


@pytest.fixture(scope = "module")
def sync_job() -> dict:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    assert "hub-readme" in doc["jobs"], "the Hub README sync job is missing"
    return doc["jobs"]["hub-readme"]


def test_the_sync_runs_only_when_latest_moved(sync_job: dict):
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    tags = [s for s in doc["jobs"]["merge-studio"]["steps"] if s.get("id") == "meta"][0]["with"][
        "tags"
    ]
    latest = [l for l in tags.splitlines() if "value=latest" in l][0]
    gate = latest.split("enable=", 1)[1].strip()
    assert sync_job["needs"] == "merge-studio"
    assert gate == sync_job["if"].strip(), (
        "the sync must be gated exactly like :latest, or a dispatch that pins refs "
        "would rewrite the public page for an image :latest does not point at"
    )


def _run_sync(
    step: str,
    tmp_path: Path,
    *,
    live_after_patch: str,
    token: str = "tok",
    secret: str = "not-a-secret",
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "curl.log"
    # what the Hub reports after the PATCH; a file, so the README's backticks and
    # dollar signs never pass through the stub's shell
    live = tmp_path / "live.json"
    live.write_text(json.dumps({"full_description": live_after_patch}), encoding = "utf-8")
    (bin_dir / "curl").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        'case "$*" in\n'
        f'  *auth/token*) printf \'{{"access_token": "{token}"}}\' ;;\n'
        "  *-X\\ PATCH*) out=''; while [ $# -gt 0 ]; do [ \"$1\" = -o ] && out=$2; shift; done; : > \"$out\"; printf '200' ;;\n"
        f"  *) cat {live} ;;\n"
        "esac\n",
        encoding = "utf-8",
    )
    (bin_dir / "curl").chmod(0o755)
    (tmp_path / "docker").mkdir()
    shutil.copy(HUB_README, tmp_path / "docker" / "DOCKERHUB.md")
    # the secrets are GitHub expressions; the test decides whether they are "set"
    script = step.replace("${{ secrets.DOCKERHUB_README_TOKEN }}", secret).replace(
        "${{ secrets.DOCKERHUB_README_USERNAME }}", "readme-user" if secret else ""
    )
    assert "${{" not in script, "unexpanded expression in the sync step"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["REGISTRY_USERNAME"] = "unsloth"
    env["IMAGE_NAME"] = "unsloth/unsloth"
    res = subprocess.run(
        ["bash", "-e", "-c", script],
        capture_output = True,
        text = True,
        env = env,
        cwd = str(tmp_path),
        timeout = 60,
    )
    return res, log.read_text(encoding = "utf-8") if log.exists() else ""


def test_the_sync_patches_the_readme_and_confirms_it(sync_job: dict, tmp_path: Path):
    step = sync_job["steps"][-1]["run"]
    res, log = _run_sync(step, tmp_path, live_after_patch = HUB_README.read_text(encoding = "utf-8"))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "-X PATCH https://hub.docker.com/v2/repositories/unsloth/unsloth/" in log
    assert "Authorization: Bearer tok" in log


def test_the_sync_fails_when_the_page_did_not_change(sync_job: dict, tmp_path: Path):
    """A 200 from PATCH is not proof. The page is read back and compared, so a token
    without description rights cannot leave the job green and the page stale."""
    step = sync_job["steps"][-1]["run"]
    res, _ = _run_sync(step, tmp_path, live_after_patch = "# the old page")
    assert res.returncode != 0, "the sync reported success while the page stayed stale"
    assert "does not match" in res.stdout + res.stderr


def test_the_sync_fails_without_a_token(sync_job: dict, tmp_path: Path):
    step = sync_job["steps"][-1]["run"]
    res, log = _run_sync(step, tmp_path, live_after_patch = "", token = "")
    assert res.returncode != 0
    assert "PATCH" not in log, "a PATCH was attempted with an empty token"


def test_the_sync_skips_visibly_without_the_personal_token(sync_job: dict, tmp_path: Path):
    """The organization token cannot edit the page (403 "token issued from
    organization access token is not allowed"), so the sync needs a personal token.
    Until one is configured the job must not fail the publish, and must not stay
    silent either: a notice names the secrets, and no PATCH is attempted."""
    step = sync_job["steps"][-1]["run"]
    res, log = _run_sync(step, tmp_path, live_after_patch = "# the old page", secret = "")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "::notice::" in res.stdout and "DOCKERHUB_README_TOKEN" in res.stdout
    assert "PATCH" not in log, "a PATCH was attempted without a personal token"
    assert "auth/token" not in log
