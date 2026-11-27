# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Playwright browser and apt caches are keyed on the OS release they were built on.

Playwright downloads a WebKit (and Firefox) build per Ubuntu release, e.g.
webkit-ubuntu-24.04.zip vs webkit-ubuntu-26.04.zip, into the same webkit-<rev> folder.
A cache saved on one release and restored on another reports a hit, `playwright install`
skips the download, and WebKit exits at launch (TargetClosedError) even after
install-deps. That is what broke the WebKit jobs when ubuntu-latest moved to 26.04 (a
24.04 build restored there) and again on 24.04 in a repo whose cache was saved on 26.04.

`${{ env.ImageOS }}` does not help: the runner's ImageOS / ImageVersion are shell
variables, not part of the `env` expression context, so they expand to nothing in a key.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

CACHED_PATHS = ("ms-playwright", ".apt-archives")
OS_OUTPUT = re.compile(r"\$\{\{\s*steps\.([\w-]+)\.outputs\.os\s*\}\}")
PRIMARY_KEY = re.compile(r"^\$\{\{\s*steps\.([\w-]+)\.outputs\.cache-primary-key\s*\}\}$")
RUNNER_ONLY_ENV = re.compile(r"\$\{\{\s*env\.(ImageOS|ImageVersion)\s*\}\}")


def _cache_steps():
    """(workflow, job, steps, index, step) for every actions/cache step over a browser or apt path."""
    out = []
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        data = yaml.safe_load(wf.read_text(encoding = "utf-8")) or {}
        for job_id, job in (data.get("jobs") or {}).items():
            steps = job.get("steps") or []
            for i, step in enumerate(steps):
                if not str(step.get("uses", "")).startswith("actions/cache"):
                    continue
                path = str((step.get("with") or {}).get("path", ""))
                if any(p in path for p in CACHED_PATHS):
                    out.append((wf.name, job_id, steps, i, step))
    return out


def _writes_os_release(step) -> bool:
    run = str(step.get("run", ""))
    return bool(re.search(r"echo\s+\"os=\$\(\.\s+/etc/os-release\b", run)) and "VERSION_ID" in run


def _key_problem(
    steps,
    i,
    key: str,
    seen = (),
):
    m = PRIMARY_KEY.match(key.strip())
    if m:
        ref = m.group(1)
        src = [(j, s) for j, s in enumerate(steps[:i]) if s.get("id") == ref]
        if not src or ref in seen:
            return (
                f"reuses the primary key of `{ref}`, which is not an earlier cache step in this job"
            )
        j, s = src[0]
        return _key_problem(steps, j, str((s.get("with") or {}).get("key", "")), seen + (ref,))
    if RUNNER_ONLY_ENV.search(key):
        return "uses ${{ env.ImageOS }} / ${{ env.ImageVersion }}, which expand to nothing in a key"
    m = OS_OUTPUT.search(key)
    if not m:
        return "does not name the OS release (expected ${{ steps.<id>.outputs.os }})"
    producers = [s for s in steps[:i] if s.get("id") == m.group(1)]
    if not producers or not _writes_os_release(producers[0]):
        return f"reads steps.{m.group(1)}.outputs.os, but no earlier step `{m.group(1)}` writes it from /etc/os-release"
    return None


CASES = _cache_steps()


def test_the_scan_still_finds_the_browser_caches():
    # studio-ui-smoke (2 jobs), studio-frontend-ci, studiobench-ci, studiobench-ui-parity.
    browser = [c for c in CASES if "ms-playwright" in str(c[4]["with"]["path"])]
    assert len({(c[0], c[1]) for c in browser}) >= 5, sorted({(c[0], c[1]) for c in browser})


@pytest.mark.parametrize(
    "wf,job,steps,i,step",
    CASES,
    ids = [f"{c[0]}::{c[1]}::{c[4].get('name', c[3])}" for c in CASES],
)
def test_the_cache_key_names_the_os_release(wf, job, steps, i, step):
    key = str((step.get("with") or {}).get("key", ""))
    problem = _key_problem(steps, i, key)
    assert problem is None, f"{wf}::{job} step {step.get('name')!r}: key {key!r} {problem}"


@pytest.mark.parametrize(
    "key,ok",
    [
        (
            "ms-playwright-${{ runner.os }}-${{ steps.pw.outputs.os }}-${{ steps.pw.outputs.version }}-cfw-v2",
            True,
        ),
        ("ms-playwright-${{ runner.os }}-${{ steps.pw.outputs.version }}-cfw-v2", False),
        ("apt-archives-${{ runner.os }}-${{ env.ImageOS }}-${{ env.ImageVersion }}-cfw-v1", False),
        ("ms-playwright-${{ runner.os }}-${{ steps.other.outputs.os }}-v2", False),
    ],
)
def test_the_key_check_itself(key, ok):
    steps = [
        {
            "id": "pw",
            "run": 'echo "os=$(. /etc/os-release && echo "${ID}-${VERSION_ID}")" >> "$GITHUB_OUTPUT"',
        },
        {"id": "other", "run": 'echo "os=linux" >> "$GITHUB_OUTPUT"'},
        {
            "uses": "actions/cache/restore@x",
            "id": "pw-cache",
            "with": {"path": "~/.cache/ms-playwright", "key": key},
        },
        {
            "uses": "actions/cache/save@x",
            "with": {
                "path": "~/.cache/ms-playwright",
                "key": "${{ steps.pw-cache.outputs.cache-primary-key }}",
            },
        },
    ]
    assert (_key_problem(steps, 2, key) is None) is ok
    assert (_key_problem(steps, 3, steps[3]["with"]["key"]) is None) is ok
