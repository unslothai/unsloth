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
_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.S)


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(key)
            yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)


def _reads(script: str, name: str) -> bool:
    """Whether a run script reads NAME from its environment, in shell or in inline Python."""
    n = re.escape(name)
    return bool(
        re.search(rf"os\.environ\[\s*['\"]{n}['\"]\s*\]", script)
        or re.search(rf"os\.(?:environ\.get|getenv)\(\s*['\"]{n}['\"]", script)
        or re.search(rf"\$\{{?{n}(?![A-Za-z0-9_])", script)
    )


def _env_blocks(doc: dict):
    yield doc.get("env") or {}
    for job in (doc.get("jobs") or {}).values():
        yield (job or {}).get("env") or {}
        for step in (job or {}).get("steps") or []:
            yield step.get("env") or {}


def _secret_backed_names() -> frozenset[str]:
    """Every name a step could be expected to receive a secret under, across all workflows.

    A secret's own name, and every env key any workflow maps from a `secrets.*` expression:
    `VT_API_KEY: ${{ secrets.VIRUS_TOTAL_API_TOKEN }}` makes VT_API_KEY secret-backed even in a
    workflow that has lost its only mapping of it, which is exactly the file a per-file scan
    would call clean.
    """
    names = set()
    for path in WORKFLOWS:
        doc = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
        # From parsed values, not the raw text: a comment that explains `secrets.A || secrets.B`
        # would otherwise make B a secret.
        for value in _strings(doc):
            for expression in _EXPRESSION.findall(value):
                names.update(_SECRET.findall(expression))
        for env in _env_blocks(doc):
            for key, value in env.items():
                if isinstance(value, str) and any(
                    _SECRET.search(e) for e in _EXPRESSION.findall(value)
                ):
                    names.add(key)
    return frozenset(names - {"GITHUB_TOKEN"})


SECRET_BACKED = _secret_backed_names()


def _unmapped(path: Path) -> list[str]:
    doc = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
    workflow_env = doc.get("env") or {}
    found = []
    for job_name, job in (doc.get("jobs") or {}).items():
        job_env = (job or {}).get("env") or {}
        for index, step in enumerate((job or {}).get("steps") or []):
            script = step.get("run")
            if not isinstance(script, str):
                continue
            env = {**workflow_env, **job_env, **(step.get("env") or {})}
            for name in sorted(SECRET_BACKED):
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


def test_an_aliased_secret_is_tracked_under_the_name_the_script_reads():
    # The repository secret is VIRUS_TOTAL_API_TOKEN and the scripts read VT_API_KEY.
    assert "VT_API_KEY" in SECRET_BACKED
    assert "VIRUS_TOTAL_API_TOKEN" in SECRET_BACKED
    # kaggle-t4-notebook-ci.yml explains `secrets.A || secrets.B` in a comment; neither is real.
    assert "A" not in SECRET_BACKED and "B" not in SECRET_BACKED


def test_an_inline_read_of_an_alias_is_caught_in_a_workflow_that_never_maps_it(tmp_path):
    """The alias is known from the workflow that maps it, so a file that lost its only mapping
    is still checked. Scripts the step merely invokes are out of scope: many read a secret
    optionally by design (the pinned-symbol suites read GH_TOKEN only when present), and from
    the file alone an intended absence and a lost mapping look the same."""
    workflow = tmp_path / "w.yml"
    workflow.write_text(
        "on: push\n"
        "jobs:\n"
        "  scan:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: Scan\n"
        "        run: |\n"
        '          curl -H "x-apikey: $VT_API_KEY" https://example.invalid\n',
        encoding = "utf-8",
    )
    assert _unmapped(workflow) == ["w.yml :: scan :: Scan reads VT_API_KEY"]
    mapped = workflow.read_text(encoding = "utf-8").replace(
        "      - name: Scan\n",
        "      - name: Scan\n        env:\n          VT_API_KEY: ${{ secrets.VIRUS_TOTAL_API_TOKEN }}\n",
    )
    workflow.write_text(mapped, encoding = "utf-8")
    assert _unmapped(workflow) == []
