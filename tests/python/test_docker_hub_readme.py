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
        "UNSLOTH_STUDIO_PASSWORD",
    ):
        assert needle in text, f"the Hub README no longer mentions {needle!r}"
    # the previous image's conventions, none of which exist in this one
    # Studio writes the generated password to a file and does not print it itself
    for stale in (
        "USER_PASSWORD",
        "/workspace/work",
        "2222:22",
        "localhot",
        "prints its first-boot",
    ):
        assert stale not in text, f"the Hub README still carries {stale!r} from the old image"


def test_the_hub_readme_explains_the_studio_volume():
    """The volume keeps Studio's data and never pins its code; a volume from an image
    before the code/data split is migrated with its old code kept aside. Both facts,
    the way back to an older image, and what `docker rm` still discards have to be on
    the page, since the quick start above them mounts the volume by default."""
    text = HUB_README.read_text(encoding = "utf-8")
    for needle in (
        "-v unsloth-studio:/opt/unsloth-studio",
        "/opt/unsloth-studio-app",
        ".unsloth-studio-legacy/",
        "UNSLOTH_STUDIO_KEEP_LEGACY=0",
        "unsloth-studio-update",
        "named volume, not a bind mount of a Windows or macOS host directory",
    ):
        assert needle in text, f"the Hub README no longer mentions {needle!r}"
    # the helper is described as setting the flags of the quick start, which now
    # includes the volume: run.sh must mount it (test_docker_cpu_fallback.py checks)
    assert "including the `unsloth-studio` volume" in text
    repo = REPO_README.read_text(encoding = "utf-8")
    assert "-v unsloth-studio:/opt/unsloth-studio" in repo
    assert ".unsloth-studio-legacy/" in repo


def _docker_sections(text: str) -> list[str]:
    """Every `#### Docker` section in the README, not just the first one.

    This used to take `text.index("#### Docker")` and read to the next `####`. The README
    grew a second Docker heading above the one that carries the run command (a one-line
    pointer in the install list), and the pins below then read a section that was never
    meant to hold a `docker run` and failed on main. Which heading comes first is an
    editing accident, so key on the content instead: the section that runs the image is
    the one these assertions are about.
    """
    sections = []
    start = text.find("#### Docker")
    while start >= 0:
        end = text.find("####", start + len("#### Docker"))
        sections.append(text[start : end if end >= 0 else len(text)])
        start = text.find("#### Docker", start + 1)
    return sections


def test_the_repo_readme_run_command_matches_the_image():
    text = REPO_README.read_text(encoding = "utf-8")
    sections = _docker_sections(text)
    assert sections, "the README no longer has a `#### Docker` section"
    running = [s for s in sections if "docker run" in s]
    # Exactly one, in both directions. Zero means the run command was dropped, which is the
    # regression this test was written for and which picking by content would otherwise hide.
    # More than one means two places tell the user how to start the image and only one of
    # them is pinned here, which is how they drift apart.
    assert len(running) == 1, (
        f"expected exactly one `#### Docker` section carrying a `docker run`, found "
        f"{len(running)} of {len(sections)} Docker sections. Pin the other one too or fold "
        f"them together; a second unpinned run command is how the README goes stale."
    )
    section = running[0]
    assert "unsloth/unsloth:core" in section
    assert "/workspace/host" in section
    assert "--ipc=host" in section
    # Across every Docker section, not only the one that runs the image: a stale port or path
    # left behind in the short pointer misleads exactly as much as one in the full command.
    for stale in ("2222:22", "/workspace/work"):
        for other in sections:
            assert stale not in other, f"a README Docker section still has {stale!r}"


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
        # The auth body goes to curl on STDIN (`--data-binary @-`) so the key is never a
        # command-line argument. A stub that logs only "$*" therefore cannot see the
        # identifier at all, and the assertion that the token authenticates as the ORG
        # passed vacuously until the body moved off argv, then failed with nothing wrong.
        # Capture the body too, and only when curl was actually told to read stdin: the
        # PATCH passes its body as an argument and has no stdin to drain.
        'case "$*" in\n'
        f"  *--data-binary\\ @-*) cat >> {log} ;;\n"
        "esac\n"
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
    script = step.replace("${{ secrets.DOCKER_API_KEY }}", secret).replace(
        "${{ env.REGISTRY_USERNAME }}", "unsloth"
    )
    assert "${{" not in script, "unexpanded expression in the sync step"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["REGISTRY_USERNAME"] = "unsloth"
    env["IMAGE_NAME"] = "unsloth/unsloth"
    # The key reaches the script through the step's `env:` block, not through a `${{ }}`
    # inside the `run:`, so the replace above matches nothing and only this line supplies
    # it. Without it the body builder dies with KeyError, curl is handed an empty request,
    # and the step still exits 0 because the failure is inside a pipeline. The env block is
    # pinned by test_the_step_env_is_what_this_harness_supplies so a rename cannot put it
    # back to silently sending no credential at all.
    env["DOCKER_API_KEY"] = secret
    res = subprocess.run(
        ["bash", "-e", "-c", script],
        capture_output = True,
        text = True,
        env = env,
        cwd = str(tmp_path),
        timeout = 60,
    )
    return res, log.read_text(encoding = "utf-8") if log.exists() else ""


def test_the_step_env_is_what_this_harness_supplies(sync_job: dict):
    """The harness hands the script its credential; this pins that it hands the RIGHT one.

    The secret moved out of the `run:` body into the step's `env:` so it is never a command
    line argument. That is a real improvement, but it also means a `.replace()` on the body
    silently stops supplying anything, and the step exits 0 regardless because the builder
    fails inside a pipeline. Renaming the variable must fail here, loudly, rather than
    downgrading the assertions below to statements about an empty request.
    """
    env = sync_job["steps"][-1].get("env") or {}
    assert "DOCKER_API_KEY" in env, (
        f"the sync step no longer takes DOCKER_API_KEY from `env:` (it declares {sorted(env)}). "
        f"_run_sync supplies that exact name; update both together."
    )
    assert "secrets.DOCKER_API_KEY" in env["DOCKER_API_KEY"]
    assert (
        "${{" not in sync_job["steps"][-1]["run"]
    ), "the credential is back in the `run:` body, where it becomes a command-line argument"


def test_the_sync_patches_the_readme_and_confirms_it(sync_job: dict, tmp_path: Path):
    step = sync_job["steps"][-1]["run"]
    res, log = _run_sync(step, tmp_path, live_after_patch = HUB_README.read_text(encoding = "utf-8"))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "-X PATCH https://hub.docker.com/v2/namespaces/unsloth/repositories/unsloth" in log
    assert "Authorization: Bearer tok" in log
    assert '"identifier": "unsloth"' in log, "the organization token authenticates as the org"


def test_the_sync_never_touches_the_legacy_repository_route(sync_job: dict, tmp_path: Path):
    """Docker Hub rejects every organization access token on
    /v2/repositories/{owner}/{repo}/ with 403 "token issued from organization access
    token is not allowed", whatever its scopes; only the namespace-scoped route
    accepts it. That 403 failed the sync on every publish before this test existed."""
    step = sync_job["steps"][-1]["run"]
    _, log = _run_sync(step, tmp_path, live_after_patch = HUB_README.read_text(encoding = "utf-8"))
    assert "/v2/repositories/" not in log
    assert "/v2/namespaces/unsloth/repositories/unsloth" in log


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


def test_the_hub_readme_matches_what_each_image_ships():
    text = HUB_README.read_text(encoding = "utf-8")
    # whisper.cpp comes from Studio's setup, so only that image has it
    assert "The `latest` image adds whisper.cpp" in text
    # SYNC disables the notebooks entirely; REFRESH only skips the GitHub fetch
    assert "`UNSLOTH_SKIP_NOTEBOOK_REFRESH=1` | Do not refresh the notebooks from GitHub" in text
    assert "`UNSLOTH_SKIP_NOTEBOOK_SYNC=1` | Do not set up the notebooks at all" in text
    # Studio's services run as root and exit 1 under --user
    assert "On `core`, `--user <uid>:<gid>` is supported" in text
    assert "AGPL-3.0" in text and "Apache-2.0" in text


def test_both_images_declare_both_licenses():
    """metadata-action labels both images Apache-2.0, but they carry Studio's AGPL-3.0 code."""
    text = WORKFLOW.read_text(encoding = "utf-8")
    assert text.count("org.opencontainers.image.licenses=Apache-2.0 AND AGPL-3.0-only") == 2


def test_the_studio_image_does_not_ship_the_uv_download_cache():
    """install.sh's uv cache sits under the Studio home, which /root/.cache never reached: ~9 GB baked into :latest."""
    body = (REPO_ROOT / "docker" / "Dockerfile.studio").read_text(encoding = "utf-8")
    # an image that points uv elsewhere (the code/data split does) must drop that cache
    assert "rm -rf" in body
    cleanup = body[body.index("rm -rf") :]
    assert '"${UV_CACHE_DIR:-${UNSLOTH_STUDIO_HOME}/cache/uv}"' in cleanup
