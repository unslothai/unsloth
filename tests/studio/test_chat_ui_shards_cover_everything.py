# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every Playwright script in Chat UI Tests belongs to exactly one shard.

The job used to run 11 scripts in sequence against four Unsloth instances, at 22.1 minutes on
average, the largest single job in the repo. It is now four shards split on the Unsloth
boundaries.

The failure mode that matters is not a broken shard, which is loud. It is a step whose
`if:` names no shard, or names one that does not exist, or is dropped from the matrix: the
step then runs nowhere, the job is green on all four shards, and a Playwright regression
suite has silently stopped existing. Nothing else in CI would notice, because a test that
does not run cannot fail.

So this asserts coverage from the workflow itself rather than from a list kept here: every
step that invokes a Playwright script must be reachable on at least one shard in the
matrix, and every shard in the matrix must have something to do.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-ui-smoke.yml"
JOB = "ui-smoke"


def _job() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"][JOB]


def _shards() -> list[str]:
    shards = _job()["strategy"]["matrix"]["shard"]
    assert shards, "the shard matrix is empty"
    return [str(s) for s in shards]


def _named_shards(condition: str) -> set[str]:
    """Which shards a step's `if:` can be true for."""
    return set(re.findall(r"matrix\.shard\s*==\s*'([^']+)'", condition or ""))


def _driving_steps() -> list[dict]:
    """Steps that actually run a Playwright script."""
    return [
        step
        for step in _job()["steps"]
        if re.search(r"playwright[\w/]*\.py", str(step.get("run", "")))
    ]


def test_the_job_still_drives_every_script_it_used_to():
    """A dropped step is the quiet failure, so the count is pinned.

    Eleven invocations across ten scripts: the banner layout script runs twice, once for
    chromium and once for the other two engines at the viewports that reproduce.
    """
    steps = _driving_steps()
    assert len(steps) >= 11, (
        f"only {len(steps)} steps invoke a Playwright script, down from 11. If one was "
        f"deliberately removed, say which and why here; if it was lost in a shard edit, "
        f"it is now running nowhere and no shard fails."
    )


@pytest.mark.parametrize("step", _driving_steps(), ids = lambda s: str(s.get("name", "?"))[:40])
def test_every_playwright_step_runs_on_some_shard(step):
    condition = str(step.get("if", ""))
    named = _named_shards(condition)
    assert named, (
        f"{step.get('name')!r} names no shard in its `if:`, so it runs on all four and "
        f"the split does not save what it claims. Give it a shard."
    )
    live = named & set(_shards())
    assert live, (
        f"{step.get('name')!r} is gated on {sorted(named)}, none of which is in the "
        f"matrix {_shards()}. It runs on no shard at all, and every shard stays green."
    )


def test_the_studio_a_script_depends_on_boots_on_the_same_shard():
    """A script and the Unsloth it drives cannot be split across machines.

    Each boot step names its port, and so does every script that talks to it. A shard
    holding the script but not the boot fails on connection refused, which is at least
    loud; the reverse wastes a boot. Both are edits worth catching here.
    """
    steps = _job()["steps"]
    booted: dict[str, set[str]] = {}
    for step in steps:
        text = str(step.get("run", "")) + str(step.get("env", ""))
        if "boot-studio" not in text:
            continue
        for port in set(re.findall(r"\b(188\d\d)\b", text)) or {"18892"}:
            booted.setdefault(port, set()).update(_named_shards(str(step.get("if", ""))))

    for step in _driving_steps():
        text = str(step.get("run", "")) + str(step.get("env", ""))
        ports = set(re.findall(r"\b(188\d\d)\b", text))
        for port in ports & set(booted):
            missing = _named_shards(str(step.get("if", ""))) - booted[port]
            assert not missing, (
                f"{step.get('name')!r} runs on {sorted(missing)} but the Unsloth on {port} "
                f"is only booted on {sorted(booted[port])}. The script would hit a port "
                f"nothing is listening on."
            )


def test_no_shard_is_left_with_nothing_to_do():
    """An orphan shard pays 2.6 minutes of setup to run no tests, and passes."""
    covered = set()
    for step in _driving_steps():
        covered |= _named_shards(str(step.get("if", "")))
    idle = sorted(set(_shards()) - covered)
    assert not idle, (
        f"{idle} run no Playwright script. A shard that installs everything and then "
        f"tests nothing is a green tick that means nothing; remove it from the matrix or "
        f"give it work."
    )


def test_each_shard_uploads_under_its_own_artifact_name():
    """Four cells cannot upload one artifact name.

    Artifacts are immutable within a workflow run, so the first shard to finish creates
    the name and the other three fail on the conflict. The upload step carries
    `if: always()` and no `continue-on-error`, so that failure is the job's: a UI run
    where every test passed goes red, on three cells out of four, for a reason that has
    nothing to do with the UI.

    Asserted for any matrix job in this workflow rather than for this one by name, since
    the next job to be sharded inherits the same trap.
    """
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    for job_name, job in document["jobs"].items():
        dimensions = (job.get("strategy") or {}).get("matrix") or {}
        if not dimensions:
            continue
        for step in job.get("steps", []):
            if "upload-artifact" not in str(step.get("uses", "")):
                continue
            name = str((step.get("with") or {}).get("name", ""))
            assert any(f"matrix.{key}" in name for key in dimensions), (
                f"{job_name} runs a matrix over {sorted(dimensions)} but uploads its "
                f"artifacts as {name!r}, the same name in every cell. All but the first "
                f"cell fails on the conflict, and the step runs with if: always(), so a "
                f"green test run reports red."
            )


def test_every_shard_captures_its_own_server_logs():
    """Each cell is a separate machine with its own ~/.unsloth/studio/logs.

    The copy used to live inside the step that stops the last Unsloth, whose comment said
    all three Unsloth instances share the directory. True when they shared a runner; false now. A
    shard-gated copy leaves three artifacts with no server-side traceback, which is
    exactly what anyone debugging a failed shard opens first.
    """
    for step in _job()["steps"]:
        if "server-logs" not in str(step.get("run", "")):
            continue
        condition = str(step.get("if", ""))
        assert "always()" in condition, (
            f"{step.get('name')!r} copies the server logs without always(), so a failing "
            f"shard uploads an artifact with nothing in it"
        )
        assert not _named_shards(condition), (
            f"{step.get('name')!r} copies the server logs only on "
            f"{sorted(_named_shards(condition))}. Every cell has its own logs directory, "
            f"so the others upload artifacts with no server-side traceback."
        )
        return
    raise AssertionError("no step copies ~/.unsloth/studio/logs into the artifact any more")
