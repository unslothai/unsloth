# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Backend CI installs moved from pip to uv without changing what gets installed.

That holds only while uv is told to resolve like pip: pick the best version across both
indexes (pip's --extra-index-url), compile bytecode, and keep its downloads inside the pip
cache the job already restores (no second cache family). Dropping any of these settings
changes the environment the tests run in, or adds an unsaved cache, without failing a step.
"""

import pathlib

import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
STEPS = [
    ("studio-backend-ci.yml", "pytest", "Install backend test dependencies (CPU only)"),
    (
        "studio-backend-ci.yml",
        "repo-cpu-tests",
        "Install deps (shared shape with backend pytest job)",
    ),
    (
        "distributed-cpu-ci.yml",
        "distributed-cpu-tests",
        "Install deps (shared shape with backend pytest job)",
    ),
]
PARITY_ENV = {
    "UV_INDEX_STRATEGY": "unsafe-best-match",
    "UV_COMPILE_BYTECODE": "1",
    "UV_LINK_MODE": "copy",
    "UV_CACHE_DIR": "${{ steps.pip-cache.outputs.dir }}/uv",
}
# The only pip calls left: pip upgrading itself and installing uv.
PIP_ALLOWED = ("pip install --upgrade pip", 'pip install --target "$RUNNER_TEMP/uv-tool" uv')


def _step(workflow, job, name):
    doc = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding = "utf-8"))
    steps = doc["jobs"][job]["steps"]
    found = [s for s in steps if s.get("name") == name]
    assert len(found) == 1, f"{workflow}:{job} has {len(found)} steps named {name!r}"
    return found[0], steps


@pytest.mark.parametrize("workflow,job,name", STEPS)
def test_uv_resolves_like_pip(workflow, job, name):
    step, _ = _step(workflow, job, name)
    env = {k: str(v) for k, v in (step.get("env") or {}).items()}
    for key, value in PARITY_ENV.items():
        assert env.get(key) == value, (
            f"{workflow}:{job} {name!r} sets {key}={env.get(key)!r}, expected {value!r}; "
            f"without it uv installs something other than what pip did"
        )


@pytest.mark.parametrize("workflow,job,name", STEPS)
def test_every_package_install_goes_through_uv(workflow, job, name):
    step, _ = _step(workflow, job, name)
    run = str(step.get("run", ""))
    installs = [line.strip() for line in run.splitlines() if "pip install" in line]
    via_uv = [line for line in installs if "uv pip install" in line or '"$UV" pip install' in line]
    assert via_uv, f"{workflow}:{job} {name!r} no longer installs through uv"
    for line in installs:
        if line in via_uv:
            assert (
                "--system" in line
            ), f"{workflow}:{job}: {line!r} installs outside the job's Python"
            continue
        assert line.endswith(PIP_ALLOWED), (
            f"{workflow}:{job}: {line!r} is a plain pip install next to uv ones; mixing the two "
            f"resolvers in one sequence is how the environment drifts"
        )


@pytest.mark.parametrize("workflow,job,name", STEPS)
def test_the_uv_cache_lives_in_the_restored_pip_cache(workflow, job, name):
    step, steps = _step(workflow, job, name)
    index = steps.index(step)
    restores = [
        s
        for s in steps[:index]
        if str(s.get("uses", "")).endswith("actions/pip-cache-restore")
        and s.get("id") == "pip-cache"
    ]
    assert restores, f"{workflow}:{job} lost the pip-cache restore that UV_CACHE_DIR points into"


def test_the_pytest_job_keeps_uv_off_path():
    """The backend tests never had uv importable or on PATH; some skip without it."""
    step, _ = _step(*STEPS[0])
    run = str(step.get("run", ""))
    assert '--target "$RUNNER_TEMP/uv-tool" uv' in run
    assert "uv pip install" not in run.replace('"$UV" pip install', "")
    assert str(step["env"]["UV"]) == "${{ runner.temp }}/uv-tool/bin/uv"
