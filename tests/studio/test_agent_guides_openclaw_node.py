# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The openclaw legs of the agent-guides workflow resolve the newest Node 24, not the runner's cache.

openclaw@2026.10.6 raised its engines floor to node >=24.22.1 while ubuntu-latest still cached
24.21.0, so `npm install -g openclaw@latest` and the curl installer both refused and every
`connection (openclaw, ...)` leg failed before reaching Studio.
"""

import re
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


OPENCLAW_GATE = "contains(matrix.agents, 'openclaw')"


def _contains(haystack: str, needle: str) -> bool:
    """GitHub's contains() on a string: case-insensitive substring."""
    return needle.lower() in haystack.lower()


def test_sharded_openclaw_runs_on_the_latest_node_24_and_the_rest_on_22():
    """Shards loop agents on one runner: openclaw gets the latest Node 24, the rest Node 22."""
    found = list(_sharded_openclaw_jobs())
    assert found, "the connection job no longer runs openclaw in a shard"
    for name, job in found:
        steps = job["steps"]
        rows = job["strategy"]["matrix"]["include"]
        nodes = [s for s in steps if str(s.get("uses", "")).startswith("actions/setup-node@")]
        versions = [str((s.get("with") or {}).get("node-version", "")) for s in nodes]
        assert versions == ["22", "24"], f"{name}: setup-node versions {versions}"
        node22, node24 = nodes
        assert "if" not in node22, f"{name}: Node 22 must be set up for every shard"

        assert node24.get("if") == OPENCLAW_GATE, f"{name}: Node 24 gate is {node24.get('if')!r}"
        for row in rows:
            assert _contains(row["agents"], "openclaw") == (
                "openclaw" in row["agents"].split()
            ), f"{name}: the gate does not pick exactly the openclaw shard for {row}"
        check = (node24.get("with") or {}).get("check-latest")
        assert check == "${{ " + OPENCLAW_GATE + " }}", (
            f"{name}: setup-node must set check-latest for the openclaw shard, or '24' resolves to "
            f"the runner's cached Node, which can sit below openclaw@latest's engines floor: {check!r}"
        )
        # A failed Node 24 setup fails only openclaw, not the Node 22 agents in its shard.
        assert node24.get("continue-on-error") is True, name
        assert node24.get("id") == "node24", name

        capture = [s for s in steps if "NODE24_BIN=" in str(s.get("run", ""))]
        assert len(capture) == 1, f"{name}: {len(capture)} steps capture NODE24_BIN"
        assert (
            capture[0].get("if") == OPENCLAW_GATE + " && steps.node24.outcome == 'success'"
        ), f"{name}: the Node 24 capture is gated by {capture[0].get('if')!r}"
        assert steps.index(capture[0]) == steps.index(node24) + 1, name
        assert 'NODE24_BIN=$(dirname "$(command -v node)")' in capture[0]["run"], name
        node22_capture = [s for s in steps if "NODE22_BIN=" in str(s.get("run", ""))]
        assert len(node22_capture) == 1 and "if" not in node22_capture[0], name
        assert steps.index(node22) < steps.index(node22_capture[0]) < steps.index(node24), name

        # The loop: Node 22 by default, Node 24 for openclaw only, and the selected node first on
        # the agent's PATH (after the agent's own npm bin, which holds no node).
        run = "\n".join(str(s.get("run", "")) for s in steps)
        assert 'node_bin="$NODE22_BIN"' in run, name
        assert '[ "$agent" = openclaw ] && node_bin="${NODE24_BIN:-}"' in run, name
        assert re.search(
            r'^\s+PATH="\$AGENT_DIR/npm/bin:\$node_bin:\$PATH" \\$', run, re.M
        ), f"{name}: the agent launch must put $node_bin ahead of the job PATH"
        # No node, no openclaw run: a named error instead of falling back to Node 22.
        assert 'if [ -z "$node_bin" ]; then' in run and "Node 24 setup failed" in run, name
