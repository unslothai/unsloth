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
    # Only the secret this step is supposed to read is expanded. Standing in for any
    # `secrets.*` would make the harness agree with a workflow that names the wrong
    # one: `${{ secrets.TYPO }}` would still produce a valid payload here, while
    # Actions would hand the real step an empty value. Anything else is left for the
    # assertion below to reject by name.
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
    # A non-empty body first: an empty request carries no identifier either, so the
    # check below cannot otherwise tell the wrong identity from no request at all.
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
    """Moving a secret out of the body means putting it into `env:`. Both halves.

    Taking `${{ secrets.DOCKER_API_KEY }}` out of three `run:` bodies removed the key
    from argv, which was the point, and left two of those steps reading
    `os.environ["DOCKER_API_KEY"]` with nothing supplying it. Neither is exercised by a
    pull request: the Hub README sync and the handle-tag cleanup run after a publish, so
    the first sign would have been a released image whose page never updated and a set of
    per-run tags that never got pruned, both reported as "could not exchange the key for
    a token" -- a message that reads like an expired credential rather than a workflow
    that forgot to pass one.

    Derived by scanning, not listed, so a fourth site added later is covered too.
    """
    offenders = []
    for path in sorted(WORKFLOW.parent.glob("docker-*.yml")):
        doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
        for job_name, job in (doc.get("jobs") or {}).items():
            job_env = set(job.get("env") or {})
            for step in job.get("steps") or []:
                body = step.get("run") or ""
                # A read, not a mention. The verdict step names the key in a sentence it
                # prints for a human, which needs no value.
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
