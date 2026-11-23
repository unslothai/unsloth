# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The openclaw legs of the agent-guides workflow resolve the newest Node 24, not the runner's cache.

openclaw@2026.10.6 raised its engines floor to node >=24.22.1 while ubuntu-latest still cached
24.21.0, so `npm install -g openclaw@latest` and the curl installer both refused and every
`connection (openclaw, ...)` leg failed before reaching Studio.
"""

from pathlib import Path

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "local-agent-guides-ci.yml"
)


def _openclaw_jobs():
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]
    for name, job in jobs.items():
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        if "openclaw" in (matrix.get("agent") or []):
            yield name, job, matrix


def test_openclaw_legs_ask_setup_node_for_the_latest_release():
    found = list(_openclaw_jobs())
    assert found, "no job in local-agent-guides-ci.yml runs the openclaw agent"
    for name, job, matrix in found:
        include = [row for row in matrix.get("include") or [] if row.get("agent") == "openclaw"]
        assert include and str(include[0].get("node", "")).startswith("24"), name
        steps = [
            s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/setup-node@")
        ]
        assert len(steps) == 1, name
        check = str((steps[0].get("with") or {}).get("check-latest", ""))
        assert "matrix.agent == 'openclaw'" in check, (
            f"{name}: setup-node must set check-latest for openclaw, or '24' resolves to the "
            "runner's cached Node, which can sit below openclaw@latest's engines floor"
        )
