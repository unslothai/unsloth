# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Firefox leg's CI wiring. No harness imports, so the unfiltered workflow-trigger-lint job can run it."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_frontend_ci_runs_the_firefox_leg_in_its_own_job():
    ci = (REPO / ".github" / "workflows" / "studio-frontend-ci.yml").read_text(encoding = "utf-8")
    jobs = ci[ci.index("\njobs:") :]
    build = jobs[jobs.index("\n  build:") : jobs.index("\n  webkit:")]
    firefox = jobs[jobs.index("\n  firefox:") : jobs.index("\n  windows:")]
    # In parallel with the build job: none of its install or run sits on that job's critical path.
    assert "needs:" not in firefox
    for needle in (
        "LONG_THREAD_FIREFOX",
        "python3 tests/studio/_firefox_leg.py",
        "p.firefox.launch",
    ):
        assert needle not in build
    assert "python3 -m playwright install firefox" in firefox
    assert "vite.long-thread.config.ts" in firefox
    assert "LONG_THREAD_FIREFOX: '1'" in firefox
    assert "python3 tests/studio/_firefox_leg.py" in firefox
    assert "tests/studio/test_firefox_leg.py" in build
    for path in (
        "tests/studio/_firefox_leg.py",
        "tests/studio/test_firefox_leg.py",
        "tests/studio/test_firefox_leg_ci.py",
    ):
        assert ci[: ci.index("\njobs:")].count(f"- '{path}'") == 2


def test_the_wiring_test_runs_in_the_unfiltered_trigger_lint_job():
    lint = (REPO / ".github" / "workflows" / "workflow-trigger-lint.yml").read_text(
        encoding = "utf-8"
    )
    assert "tests/studio/test_firefox_leg_ci.py" in lint
