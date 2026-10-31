# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""cancel-closed-pr-runs.yml cancels exactly the closed PR's leftover pull_request runs.

Every test runs the workflow's real `run:` text through bash against a fake `gh` that models the
three calls the step makes (open-PR lookup, run listing, cancel) and refuses anything else. The
`--jq` programs are evaluated by the real `jq`, so a broken filter fails here, not on a runner.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "cancel-closed-pr-runs.yml"
REPOSITORY = "unslothai/unsloth"
SELF_PATH = ".github/workflows/cancel-closed-pr-runs.yml"

pytestmark = pytest.mark.skipif(
    shutil.which("jq") is None or shutil.which("bash") is None, reason = "needs bash and jq"
)

FAKE_GH_BODY = r"""
import json, os, subprocess, sys

state = json.load(open(os.environ["FAKE_STATE"]))
argv = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as handle:
    handle.write("gh " + " ".join(argv) + "\n")

fields = {}
i = 0
while i < len(argv):
    if argv[i] == "-f":
        k, _, v = argv[i + 1].partition("=")
        fields[k] = v
        i += 2
    else:
        i += 1
jq = argv[argv.index("--jq") + 1] if "--jq" in argv else None
path = next((a for a in argv if a.startswith("repos/")), "")


def emit(payload):
    if jq is None:
        print(json.dumps(payload))
        return 0
    r = subprocess.run(["jq", "-r", jq], input = json.dumps(payload), text = True, capture_output = True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


if argv[:1] == ["api"] and "GET" in argv and path == "repos/%s/pulls" % state["repo"]:
    sys.exit(emit([p for p in state["pulls"] if p["state"] == fields.get("state") and p["head_label"] == fields.get("head")]))
if argv[:1] == ["api"] and "GET" in argv and path == "repos/%s/actions/runs" % state["repo"]:
    if state.get("list_fails"):
        sys.exit(1)
    runs = [r for r in state["runs"] if r["event"] == fields.get("event") and r["status"] == fields.get("status")
            and r["head_branch"] == fields.get("branch")]
    sys.exit(emit({"total_count": len(runs), "workflow_runs": runs}))
if argv[:1] == ["api"] and "POST" in argv and path.endswith("/cancel"):
    if state.get("read_only"):
        sys.stderr.write("HTTP 403: Resource not accessible by integration\n")
        sys.exit(1)
    sys.exit(0)
sys.stderr.write("fake gh: unmodelled invocation %r\n" % (argv,))
sys.exit(2)
"""


def _script():
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    steps = document["jobs"]["cancel"]["steps"]
    assert len(steps) == 1
    return steps[0]["run"]


def _run(
    world,
    pr,
    head_repo,
    branch,
    *,
    self_run = "999",
):
    world = dict(world)
    world.setdefault("repo", REPOSITORY)
    world.setdefault("pulls", [])
    tmp = world.pop("_tmp")
    state = tmp / "state.json"
    state.write_text(json.dumps(world), encoding = "utf-8")
    fake_bin = tmp / "bin"
    fake_bin.mkdir(exist_ok = True)
    body = fake_bin / "fake_gh.py"
    body.write_text(FAKE_GH_BODY, encoding = "utf-8")
    gh = fake_bin / "gh"
    gh.write_text(f'#!/bin/sh\nexec "$FAKE_GH_PYTHON" "{body}" "$@"\n', encoding = "utf-8")
    gh.chmod(0o755)
    log = tmp / "commands.log"
    log.write_text("", encoding = "utf-8")
    summary = tmp / "summary.md"
    env = os.environ.copy()
    env.update(
        {
            "COMMAND_LOG": str(log),
            "FAKE_GH_PYTHON": sys.executable,
            "FAKE_STATE": str(state),
            "GH_TOKEN": "x",
            "GITHUB_STEP_SUMMARY": str(summary),
            "HEAD_REF": branch,
            "HEAD_REPO": head_repo,
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PR": str(pr),
            "REPO": REPOSITORY,
            "RUNNER_TEMP": str(tmp),
            "SELF_RUN": self_run,
            "SELF_WORKFLOW": f"{REPOSITORY}/{SELF_PATH}@refs/pull/{pr}/merge",
        }
    )
    result = subprocess.run(
        ["bash", "-c", _script()], cwd = tmp, env = env, text = True, capture_output = True
    )
    cancelled = sorted(
        line.split("/actions/runs/")[1].split("/")[0]
        for line in log.read_text(encoding = "utf-8").splitlines()
        if "POST" in line and line.endswith("/cancel --silent")
    )
    return result, cancelled, summary.read_text(encoding = "utf-8") if summary.exists() else ""


def _r(
    rid,
    *,
    event = "pull_request",
    status = "queued",
    branch = "feat",
    repo = REPOSITORY,
    prs = (),
    path = None,
):
    return {
        "id": rid,
        "event": event,
        "status": status,
        "head_branch": branch,
        "head_repository": {"full_name": repo},
        "pull_requests": [{"number": n} for n in prs],
        "path": path or ".github/workflows/lint-ci.yml",
    }


def test_only_this_prs_pull_request_runs_are_cancelled(tmp_path):
    world = {
        "_tmp": tmp_path,
        "runs": [
            _r(1, prs = [10]),  # queued, this PR
            _r(2, status = "in_progress", prs = [10]),  # running, this PR
            _r(3, status = "waiting", prs = [10]),
            _r(4, event = "push"),  # a push run on the same branch
            _r(5, prs = [11]),  # another PR from the same branch
            _r(6, status = "completed", prs = [10]),  # already finished
            _r(999, status = "in_progress", prs = [10], path = SELF_PATH),  # this job
            _r(7, status = "queued", prs = [10], path = SELF_PATH),  # an older close event
        ],
    }
    result, cancelled, summary = _run(world, 10, REPOSITORY, "feat")
    assert result.returncode == 0, result.stderr
    assert cancelled == ["1", "2", "3"]
    assert "3 leftover run(s), 3 cancelled, 0 refused" in summary


def test_fork_runs_match_by_head_repo_and_branch_never_another_forks(tmp_path):
    world = {
        "_tmp": tmp_path,
        "runs": [
            _r(1, branch = "main", repo = "alice/unsloth"),
            _r(2, branch = "main", repo = "alice/unsloth", status = "in_progress"),
            _r(3, branch = "main", repo = "bob/unsloth"),  # another fork, same branch name
            _r(4, branch = "main", repo = REPOSITORY),  # upstream branch of that name
        ],
    }
    result, cancelled, _ = _run(world, 20, "alice/unsloth", "main")
    assert result.returncode == 0, result.stderr
    assert cancelled == ["1", "2"]


def test_runs_shared_with_another_open_pr_of_the_same_head_are_left_alone(tmp_path):
    world = {
        "_tmp": tmp_path,
        "pulls": [{"number": 21, "state": "open", "head_label": "alice:main"}],
        "runs": [_r(1, branch = "main", repo = "alice/unsloth"), _r(2, branch = "main", prs = [20])],
    }
    result, cancelled, _ = _run(world, 20, "alice/unsloth", "main")
    assert result.returncode == 0, result.stderr
    assert cancelled == ["2"]  # the numbered run is still unambiguous


def test_a_read_only_token_reports_and_succeeds(tmp_path):
    world = {"_tmp": tmp_path, "read_only": True, "runs": [_r(1, branch = "x", repo = "alice/unsloth")]}
    result, _, summary = _run(world, 30, "alice/unsloth", "x")
    assert result.returncode == 0, result.stderr
    assert "1 leftover run(s), 0 cancelled, 1 refused" in summary
    assert "read-only" in result.stdout


def test_a_failed_listing_cancels_nothing(tmp_path):
    world = {"_tmp": tmp_path, "list_fails": True, "runs": [_r(1, prs = [10])]}
    result, cancelled, _ = _run(world, 10, REPOSITORY, "feat")
    assert result.returncode == 0 and cancelled == []
    assert "cancelling nothing" in result.stdout


def test_a_deleted_head_repo_matches_by_number_only(tmp_path):
    world = {
        "_tmp": tmp_path,
        "runs": [_r(1, branch = "main", repo = "-"), _r(2, branch = "main", prs = [40])],
    }
    result, cancelled, _ = _run(world, 40, "", "main")
    assert result.returncode == 0, result.stderr
    assert cancelled == ["2"]


def test_the_branch_name_is_data_not_shell(tmp_path):
    branch = "x$(touch pwned)`touch pwned2`;touch pwned3"
    world = {"_tmp": tmp_path, "runs": [_r(1, branch = branch, prs = [50])]}
    result, cancelled, _ = _run(world, 50, REPOSITORY, branch)
    assert result.returncode == 0, result.stderr
    assert cancelled == ["1"]
    assert not any((tmp_path / name).exists() for name in ("pwned", "pwned2", "pwned3"))


def test_the_workflow_shape():
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    on = document.get("on", document.get(True))
    assert on == {"pull_request": {"types": ["closed"]}}
    assert document["permissions"] == {}
    job = document["jobs"]["cancel"]
    assert job["permissions"] == {"actions": "write", "pull-requests": "read"}
    assert job["runs-on"] == "ubuntu-24.04-arm" and job["timeout-minutes"] <= 10
    assert not any("uses" in step for step in job["steps"])  # nothing checked out, nothing cached
    run = job["steps"][0]["run"]
    assert "force-cancel" not in run  # /cancel keeps other runs' always() cleanup
    assert "${{" not in run  # untrusted values arrive through env only
