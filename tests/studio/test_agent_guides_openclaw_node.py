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


def _sharded_openclaw_jobs():
    """Jobs whose matrix rows name the agents a runner loops over (`agents: "a b c"`)."""
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]
    for name, job in jobs.items():
        rows = ((job.get("strategy") or {}).get("matrix") or {}).get("include") or []
        if any("openclaw" in str(row.get("agents", "")).split() for row in rows):
            yield name, job


def test_sharded_openclaw_runs_on_the_latest_node_24_and_the_rest_on_22():
    """The connection shards loop agents on one runner: openclaw still gets the newest Node 24, every
    other agent still runs on Node 22, as each did on its own runner."""
    found = list(_sharded_openclaw_jobs())
    assert found, "the connection job no longer runs openclaw in a shard"
    for name, job in found:
        steps = job["steps"]
        nodes = [s for s in steps if str(s.get("uses", "")).startswith("actions/setup-node@")]
        versions = [str((s.get("with") or {}).get("node-version", "")) for s in nodes]
        assert versions == ["22", "24"], f"{name}: setup-node versions {versions}"
        node24 = nodes[1]
        assert "openclaw" in str(
            node24.get("if", "")
        ), f"{name}: Node 24 must be set up for the openclaw shard"
        check = str((node24.get("with") or {}).get("check-latest", ""))
        assert "openclaw" in check, (
            f"{name}: setup-node must set check-latest for openclaw, or '24' resolves to the "
            "runner's cached Node, which can sit below openclaw@latest's engines floor"
        )
        run = "\n".join(str(s.get("run", "")) for s in steps)
        assert (
            'node_bin="$NODE22_BIN"' in run
            and '[ "$agent" = openclaw ] && node_bin="$NODE24_BIN"' in run
        ), f"{name}: the agent loop must put Node 24 on PATH for openclaw only"
