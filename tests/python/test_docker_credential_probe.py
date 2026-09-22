# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The credential probe pushes a throwaway tag and must remove it again. Docker Hub
rejects an organization access token on the legacy /v2/repositories/... routes
with 403 whatever its scopes, and only the namespace-scoped routes accept it, so
the delete step is run here with curl stubbed and its requests inspected.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-credential-probe.yml"


@pytest.fixture(scope = "module")
def delete_step() -> str:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    steps = [
        s
        for job in doc["jobs"].values()
        for s in job["steps"]
        if s.get("name") == "Delete the probe tag"
    ]
    assert len(steps) == 1, "the delete step disappeared or was renamed"
    return steps[0]["run"]


def _run(
    step: str,
    tmp_path: Path,
    *,
    still_there: bool,
    token: str = "tok",
) -> tuple[subprocess.CompletedProcess, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "curl.log"
    (bin_dir / "curl").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        # The auth body goes to curl on STDIN (`--data-binary @-`) so the key is never a
        # command-line argument. A stub that logs only "$*" therefore cannot see the
        # identifier at all, and the assertion that the token authenticates as the ORG
        # passed vacuously until the body moved off argv, then failed with nothing wrong.
        # Capture the body too, and only when curl was actually told to read stdin.
        'case "$*" in\n'
        f"  *--data-binary\\ @-*) cat >> {log} ;;\n"
        "esac\n"
        'case "$*" in\n'
        f'  *auth/token*) printf \'{{"access_token": "{token}"}}\' ;;\n'
        "  *-X\\ DELETE*) printf '204' ;;\n"
        f"  *) printf '{200 if still_there else 404}' ;;\n"
        "esac\n",
        encoding = "utf-8",
    )
    (bin_dir / "curl").chmod(0o755)
    script = step.replace("${{ secrets.DOCKER_API_KEY }}", "not-a-secret")
    assert "${{" not in script, "unexpanded expression in the delete step"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env.update(
        REGISTRY_USERNAME = "unsloth", IMAGE_NAME = "unsloth/unsloth", PROBE_TAG = "credential-probe"
    )
    # The key reaches the script through the step's `env:` block, not through a `${{ }}`
    # inside the `run:`, so the replace above matches nothing and only this line supplies
    # it. Without it the body builder dies with KeyError, curl is handed an empty request,
    # and the step still exits 0 because the failure is inside a pipeline. The env block is
    # pinned by test_the_step_env_is_what_this_harness_supplies so a rename cannot put it
    # back to silently sending no credential at all.
    env["DOCKER_API_KEY"] = "not-a-secret"
    res = subprocess.run(
        ["bash", "-e", "-c", script],
        capture_output = True,
        text = True,
        env = env,
        cwd = str(tmp_path),
        timeout = 60,
    )
    return res, log.read_text(encoding = "utf-8") if log.exists() else ""


def test_the_step_env_is_what_this_harness_supplies():
    """The harness hands the script its credential; this pins that it hands the RIGHT one.

    The secret moved out of the `run:` body into the step's `env:` so it is never a command
    line argument. That is a real improvement, but it also means a `.replace()` on the body
    silently stops supplying anything, and the step exits 0 regardless because the builder
    fails inside a pipeline. Renaming the variable must fail here, loudly, rather than
    downgrading the assertions below to statements about an empty request.
    """
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    steps = [
        s
        for job in doc["jobs"].values()
        for s in job["steps"]
        if s.get("name") == "Delete the probe tag"
    ]
    env = steps[0].get("env") or {}
    assert "DOCKER_API_KEY" in env, (
        f"the delete step no longer takes DOCKER_API_KEY from `env:` (it declares "
        f"{sorted(env)}). _run supplies that exact name; update both together."
    )
    assert "secrets.DOCKER_API_KEY" in env["DOCKER_API_KEY"]
    assert (
        "${{" not in steps[0]["run"]
    ), "the credential is back in the `run:` body, where it becomes a command-line argument"


def test_the_delete_uses_the_namespace_route_the_org_token_is_allowed_on(
    delete_step: str, tmp_path: Path
):
    res, log = _run(delete_step, tmp_path, still_there = False)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (
        "-X DELETE https://hub.docker.com/v2/namespaces/unsloth/repositories/unsloth/tags/credential-probe"
        in log
    )
    assert (
        "/v2/repositories/" not in log
    ), "the legacy route answers every organization token with 403"
    assert '"identifier": "unsloth"' in log
    assert "Authorization: Bearer tok" in log


def test_a_tag_that_survives_the_delete_fails_the_step(delete_step: str, tmp_path: Path):
    res, _ = _run(delete_step, tmp_path, still_there = True)
    assert res.returncode != 0
    assert "still resolves" in res.stdout + res.stderr


def test_no_token_means_no_delete_and_a_failure(delete_step: str, tmp_path: Path):
    res, log = _run(delete_step, tmp_path, still_there = True, token = "")
    assert res.returncode != 0
    assert "DELETE" not in log
