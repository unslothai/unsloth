# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The WebKit leg's CI wiring. No harness imports, so the unfiltered workflow-trigger-lint job can run it."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_frontend_ci_runs_the_webkit_leg_with_webkit_installed():
    ci = (REPO / ".github" / "workflows" / "studio-frontend-ci.yml").read_text(encoding = "utf-8")
    jobs = ci[ci.index("\njobs:") :]
    build = jobs[jobs.index("\n  build:") : jobs.index("\n  webkit:")]
    webkit = jobs[jobs.index("\n  webkit:") : jobs.index("\n  windows:")]
    # Its own job, in parallel: the build job's critical path installs no WebKit and runs no leg.
    assert "needs:" not in webkit
    for needle in (
        "install webkit",
        "chromium webkit",
        "p.webkit",
        "LONG_THREAD_WEBKIT",
        "python3 tests/studio/_webkit_leg.py",
    ):
        assert needle not in build
    assert "python3 -m playwright install webkit" in webkit
    assert "vite.long-thread.config.ts" in webkit
    assert "LONG_THREAD_WEBKIT: '1'" in webkit
    assert "python3 tests/studio/_webkit_leg.py" in webkit
    assert "tests/studio/test_webkit_leg.py" in build
    for path in (
        "tests/studio/_webkit_leg.py",
        "tests/studio/test_webkit_leg.py",
        "tests/studio/test_webkit_leg_ci.py",
    ):
        assert ci[: ci.index("\njobs:")].count(f"- '{path}'") == 2
