# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A step that reads a secret from its environment has to be given it.

Secrets here are step-scoped on purpose (test_cached_paths_hold_no_credentials.py says why),
and scripts read them through the environment rather than a `${{ secrets.* }}` expression
spliced into the script, so the value never lands in the step's temporary shell file or in
argv. That leaves one way to get it wrong that nothing else sees: a script that reads
`os.environ["DOCKER_API_KEY"]` or `$DOCKER_API_KEY` in a step whose `env:` never maps it.
Python raises KeyError, the shell expands to an empty string, and the step fails or quietly
does nothing, only on the privileged run that holds the secret, which a pull request never
exercises.

That is what happened when the Docker Hub token exchanges moved to the environment: the Hub
README step was given `DOCKER_API_KEY`, the cleanup step beside it and the ROCm workflow's
README step were not, and every publish run from then on left its handle tags on Docker Hub.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).resolve().parents[2] / ".github" / "workflows").glob("*.yml"))

_SECRET = re.compile(r"secrets\.([A-Za-z_][A-Za-z0-9_]*)")


def _reads(script: str, name: str) -> bool:
    """Whether a run script reads NAME from its environment, in shell or in inline Python."""
    n = re.escape(name)
    return bool(
        re.search(rf"os\.environ\[\s*['\"]{n}['\"]\s*\]", script)
        or re.search(rf"os\.(?:environ\.get|getenv)\(\s*['\"]{n}['\"]", script)
        or re.search(rf"\$\{{?{n}(?![A-Za-z0-9_])", script)
    )


def _unmapped(path: Path) -> list[str]:
    text = path.read_text(encoding = "utf-8")
    secrets = set(_SECRET.findall(text)) - {"GITHUB_TOKEN"}
    if not secrets:
        return []
    doc = yaml.safe_load(text) or {}
    workflow_env = doc.get("env") or {}
    found = []
    for job_name, job in (doc.get("jobs") or {}).items():
        job_env = (job or {}).get("env") or {}
        for index, step in enumerate((job or {}).get("steps") or []):
            script = step.get("run")
            if not isinstance(script, str):
                continue
            env = {**workflow_env, **job_env, **(step.get("env") or {})}
            for name in sorted(secrets):
                if name not in env and _reads(script, name):
                    label = step.get("name") or f"step {index}"
                    found.append(f"{path.name} :: {job_name} :: {label} reads {name}")
    return found


def test_the_workflows_are_found():
    assert len(WORKFLOWS) > 20, "the glob stopped finding the workflows"


@pytest.mark.parametrize("path", WORKFLOWS, ids = [p.name for p in WORKFLOWS])
def test_every_secret_a_step_reads_is_in_its_env(path):
    unmapped = _unmapped(path)
    assert not unmapped, (
        "these steps read a secret from the environment without mapping it in env:, so it "
        "is empty or a KeyError on the run that holds it:\n  " + "\n  ".join(unmapped)
    )


def test_the_reader_sees_every_spelling_the_workflows_use():
    assert _reads("python3 -c 'import os; os.environ[\"K\"]'", "K")
    assert _reads("os.environ.get('K', '')", "K")
    assert _reads('os.getenv("K")', "K")
    assert _reads('curl -H "Bearer $K"', "K")
    assert _reads('echo "${K}"', "K")
    # A longer name that starts with K is not K, and an expression is not an env read.
    assert not _reads('echo "$K_OTHER"', "K")
    assert not _reads("echo ${{ secrets.K }}", "K")
