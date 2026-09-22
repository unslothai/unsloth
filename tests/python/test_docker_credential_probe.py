# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The credential probe pushes a throwaway tag and must remove it again. Docker Hub
rejects an organization access token on the legacy /v2/repositories/... routes
with 403 whatever its scopes, and only the namespace-scoped routes accept it, so
the delete step is run here with curl stubbed and its requests inspected.
"""

from __future__ import annotations

import os
import re
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
    return steps[0]


# The stand-in for DOCKER_API_KEY, named so an assertion can look for it.
SECRET = "not-a-secret"


def _run(
    step: dict,
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
        # The request body does not always travel in argv. #11511 moved the token
        # request onto stdin (`--data-binary @-`) so the org secret stops showing up
        # in the process list, and a stub that logs only "$*" then records a call
        # whose payload is simply absent: every assertion about what was SENT passes
        # vacuously or fails for the wrong reason. Read it where it actually is, and
        # only when the arguments say there is one, since `cat` with no stdin hangs.
        f"case \"$*\" in *'--data-binary @-'*) cat >> {log} ;; esac\n"
        'case "$*" in\n'
        f'  *auth/token*) printf \'{{"access_token": "{token}"}}\' ;;\n'
        "  *-X\\ DELETE*) printf '204' ;;\n"
        f"  *) printf '{200 if still_there else 404}' ;;\n"
        "esac\n",
        encoding = "utf-8",
    )
    (bin_dir / "curl").chmod(0o755)
    script = step["run"].replace("${{ secrets.DOCKER_API_KEY }}", SECRET)
    assert "${{" not in script, "unexpanded expression in the delete step"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env.update(
        REGISTRY_USERNAME = "unsloth", IMAGE_NAME = "unsloth/unsloth", PROBE_TAG = "credential-probe"
    )
    # Whatever the step declares in its own `env:`, bound here too. #11511 moved the
    # secret out of the run body and into `env: DOCKER_API_KEY`, read with
    # `os.environ`, so rewriting the body alone hands the script an environment it
    # cannot run in and the request goes out with an empty payload.
    for name, value in (step.get("env") or {}).items():
        env[name] = re.sub(r"\$\{\{\s*secrets\.\w+\s*\}\}", SECRET, str(value))
        assert "${{" not in env[name], f"unexpanded expression in the step's env {name}"
    res = subprocess.run(
        ["bash", "-e", "-c", script],
        capture_output = True,
        text = True,
        env = env,
        cwd = str(tmp_path),
        timeout = 60,
    )
    return res, log.read_text(encoding = "utf-8") if log.exists() else ""


def test_the_delete_uses_the_namespace_route_the_org_token_is_allowed_on(
    delete_step: dict, tmp_path: Path
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
    # A non-empty body first: an empty request carries no identifier either, so the
    # check below cannot otherwise tell the wrong identity from no request at all.
    assert f'"secret": "{SECRET}"' in log, (
        "the token request carried no body, so this proves nothing about who it "
        "authenticates as"
    )
    assert '"identifier": "unsloth"' in log
    assert "Authorization: Bearer tok" in log


def test_a_tag_that_survives_the_delete_fails_the_step(delete_step: dict, tmp_path: Path):
    res, _ = _run(delete_step, tmp_path, still_there = True)
    assert res.returncode != 0
    assert "still resolves" in res.stdout + res.stderr


def test_no_token_means_no_delete_and_a_failure(delete_step: dict, tmp_path: Path):
    res, log = _run(delete_step, tmp_path, still_there = True, token = "")
    assert res.returncode != 0
    assert "DELETE" not in log
