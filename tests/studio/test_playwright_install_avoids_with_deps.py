# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""`playwright install --with-deps` must not come back.

The flag looks like a convenience and is actually an unbounded apt step wearing
a different name: playwright runs its own `apt-get update` inside it, which is
the one apt call in this repo that cannot be restructured to try the image's
lists first. Everything CI has learned about apt -- the shared retry helper, the
20s transfer cap, `APT_ACQUIRE_RETRIES: '0'`, the archive cache -- applies to
the `install-deps` subcommand and is bypassed entirely when the work happens
inside `install --with-deps`.

Both failures that motivated this were the same shape and the same package:

  studio-ui-smoke.yml  webkit shards, 181 packages / 102 MB
  studio-frontend-ci.yml  chromium, 9 packages / 21.1 MB, and
    `fonts-wqy-zenhei [7472 kB]` alone took 5m50s off azure.archive.ubuntu.com

The supported shape is: download the engine, launch it to find out whether the
system libraries are actually missing, and run `install-deps` only if they are.
That is what this guard pins -- not the comments describing it, which drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

# `playwright install --with-deps`, however the flag is spelled or ordered, and whether invoked as `playwright`
_WITH_DEPS = re.compile(r"playwright\s+install\b[^\n]*--with-deps")


def _run_steps(path: Path):
    """(job name, step name, run body) for every step that runs a shell body."""
    doc = yaml.safe_load(path.read_text()) or {}
    for job_name, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if isinstance(run, str):
                yield job_name, step.get("name", "<unnamed>"), run


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml"))


@pytest.mark.parametrize("path", _workflow_files(), ids = lambda p: p.name)
def test_no_workflow_installs_playwright_browsers_with_deps(path: Path) -> None:
    offenders = [
        f"{path.name}::{job}::{step}"
        for job, step, run in _run_steps(path)
        if _WITH_DEPS.search(run)
    ]
    assert not offenders, (
        "`playwright install --with-deps` runs apt-get update inside itself, "
        "bypassing the shared retry helper, the 20s transfer cap and "
        "APT_ACQUIRE_RETRIES: '0'. Download the engine, probe by launching it, "
        "and call `playwright install-deps` only when the probe fails -- see the "
        "`Install Chromium for browser smokes` step in studio-frontend-ci.yml. "
        f"Offending steps: {offenders}"
    )


def test_the_guard_can_see_the_pattern_it_forbids() -> None:
    """A regex typo would make every assertion above vacuously pass."""
    for body in (
        "python3 -m playwright install --with-deps chromium",
        "python -m playwright install --with-deps chromium firefox webkit",
        "playwright install --with-deps",
        "playwright install chromium --with-deps",
    ):
        assert _WITH_DEPS.search(body), body


def test_the_guard_does_not_fire_on_the_supported_shape() -> None:
    """The split form, and the unrelated `--with-deps` on scan_packages.py."""
    for body in (
        "python3 -m playwright install chromium",
        "python -m playwright install-deps chromium",
        "python scripts/scan_packages.py --with-deps requirements.txt",
    ):
        assert not _WITH_DEPS.search(body), body


def test_at_least_one_workflow_installs_playwright_browsers() -> None:
    """Pins that the parametrisation is actually looking at something."""
    installers = [
        f"{path.name}::{step}"
        for path in _workflow_files()
        for _job, step, run in _run_steps(path)
        if re.search(r"playwright\s+install\b", run)
    ]
    assert installers, (
        "no workflow step runs `playwright install`; either the suite moved or "
        "this guard is no longer reading the workflows it thinks it is"
    )
