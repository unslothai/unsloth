# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Docker Hub rejects org tokens on legacy /v2/repositories routes; delete via namespace routes."""

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
        # The token request body is sent on stdin (--data-binary @-), so log it from there;
        # only read stdin when that flag is present, since cat would hang.
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
    # Bind the step's env: too; expand only the expected secret so a wrong name fails.
    for name, value in (step.get("env") or {}).items():
        env[name] = re.sub(r"\$\{\{\s*secrets\.DOCKER_API_KEY\s*\}\}", SECRET, str(value))
        assert "${{" not in env[name], (
            f"the step's env {name} reads {value!r}, which is not the secret this "
            f"harness knows how to supply"
        )
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
    assert (
        f'"secret": "{SECRET}"' in log
    ), "the token request carried no body, so this proves nothing about who it authenticates as"
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


def test_every_step_that_reads_the_key_is_given_the_key():
    """Every workflow step that reads DOCKER_API_KEY from the environment must also be given it in env:."""
    offenders = []
    for path in sorted(WORKFLOW.parent.glob("docker-*.yml")):
        doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
        for job_name, job in (doc.get("jobs") or {}).items():
            job_env = set(job.get("env") or {})
            for step in job.get("steps") or []:
                body = step.get("run") or ""
                reads = (
                    'os.environ["DOCKER_API_KEY"]' in body
                    or "$DOCKER_API_KEY" in body
                    or "${DOCKER_API_KEY" in body
                )
                if reads and "DOCKER_API_KEY" not in (set(step.get("env") or {}) | job_env):
                    offenders.append(f"{path.name}:{job_name}: {step.get('name')!r}")
    assert not offenders, (
        "these steps read DOCKER_API_KEY and no env: at step or job level provides it, "
        "so the token exchange gets an empty secret and the step fails at publish "
        "time:\n  " + "\n  ".join(offenders)
    )
