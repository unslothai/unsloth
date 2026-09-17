# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The install workflows run a PR subset of their legs and the whole set nightly.

clean-machine-install-ci.yml (20 legs, 7 on macOS) and interrupted-install-ci.yml (10
legs, 6 on macOS) used to run every leg on every pull_request that touched an installer,
against an account capped at five concurrent macOS jobs. On the 20 PRs audited before
the split not one of those runs finished: each was cancelled by the next push while its
macOS legs were still queued, and while they queued they held the macOS pool against
every other job in the org.

The legs now live in .github/ci/*-matrix.yml with a `pr` flag each, and a `select` job
hands every matrix job the `include` list for the event
(.github/scripts/select_install_matrix.py). This file pins what that split relies on:

* the matrix files parse and every leg has the keys its job's steps read, because a
  leg missing a key no longer fails at YAML time but at `matrix.<key>` time on a runner;
* the PR subset per job is what the workflow header says it is, and the clean-machine
  subset keeps a macOS leg, so tests/studio/test_macos_slots_per_commit.py still counts
  the file as a macOS workflow;
* the `select` job is the only producer, every matrix job takes its legs from it, and a
  job whose PR subset is empty gates on the `_count` output rather than expanding an
  empty matrix, which is a workflow error;
* the emitted JSON never carries the `pr` key, so what a job sees is exactly the leg.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
SELECTOR = REPO / ".github" / "scripts" / "select_install_matrix.py"

# workflow -> (matrix file, {job: (all legs, PR legs)})
EXPECTED = {
    "clean-machine-install-ci.yml": (
        ".github/ci/clean-machine-matrix.yml",
        {"macos": (7, 3), "linux": (6, 2), "windows": (3, 1), "windows_container_install": (2, 1)},
    ),
    "interrupted-install-ci.yml": (
        ".github/ci/interrupted-install-matrix.yml",
        {"interrupt": (8, 2), "interrupt-windows": (2, 0)},
    ),
}

# Keys every leg of a job must carry, stated here rather than derived from the legs: a
# leg that drops `overlay` renders `${{ matrix.overlay }}` empty and quietly tests the
# released package, so the list that catches that cannot be computed from the legs.
# Anything else a job reads (`nonroot`, `wget_only`, `allow_working`, ...) is an optional
# flag read with a truthiness test.
REQUIRED_KEYS = {
    "macos": {"os", "mode", "delivery", "flags", "experimental", "overlay"},
    "linux": {"label", "image", "runner", "experimental", "overlay"},
    "windows": {"os", "winget", "experimental", "overlay"},
    "windows_container_install": {"overlay"},
    "interrupt": {"os", "label", "marker"},
    "interrupt-windows": {"label", "marker", "installArgs"},
}

_SELECTED = re.compile(r"^\$\{\{\s*fromJSON\(needs\.select\.outputs\.([\w-]+)\)\s*\}\}$")
_MATRIX_KEY = re.compile(r"matrix\.([\w-]+)")


def _key(jid: str) -> str:
    """The matrix-file key and select output for a job id: outputs cannot carry `-`."""
    return jid.replace("-", "_")


def _doc(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))


def _legs(matrix_file: str) -> dict:
    legs = yaml.safe_load((REPO / matrix_file).read_text(encoding = "utf-8"))
    assert isinstance(legs, dict), f"{matrix_file}: expected a mapping of job -> legs"
    return legs


def _select(matrix_file: str, event: str) -> dict[str, str]:
    """Run the selector the way the `select` job does and return its output lines."""
    out = subprocess.run(
        [sys.executable, str(SELECTOR), "--file", matrix_file, "--event", event],
        cwd = REPO,
        capture_output = True,
        text = True,
        check = True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if line)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_matrix_file_is_the_one_the_select_job_reads(name):
    matrix_file, jobs = EXPECTED[name]
    doc = _doc(name)
    select = doc["jobs"]["select"]
    files = [
        (s.get("env") or {}).get("MATRIX_FILE") for s in select["steps"] if isinstance(s, dict)
    ]
    assert matrix_file in files, f"{name}: the select job does not read {matrix_file}"
    assert set(_legs(matrix_file)) == {_key(j) for j in jobs}, (
        f"{name}: {matrix_file} lists jobs {sorted(_legs(matrix_file))}, "
        f"this test expects {sorted(jobs)}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_matrix_job_takes_its_legs_from_select_and_nothing_else_does(name):
    matrix_file, jobs = EXPECTED[name]
    doc = _doc(name)
    for jid, job in doc["jobs"].items():
        matrix = (job.get("strategy") or {}).get("matrix")
        if matrix is None:
            continue
        match = _SELECTED.match(str(matrix))
        assert match, f"{name}:{jid} carries a literal matrix; the legs belong in {matrix_file}"
        assert match.group(1) == _key(jid), (
            f"{name}:{jid} reads needs.select.outputs.{match.group(1)}; the output is "
            f"named after the job so the file and the workflow cannot drift"
        )
        assert jid in jobs, f"{name}:{jid} is a matrix job this test does not know"
        needs = job.get("needs")
        needs = needs if isinstance(needs, list) else [needs]
        assert "select" in needs, f"{name}:{jid} does not depend on the select job"
    for jid in jobs:
        assert jid in doc["jobs"], f"{name}: matrix job {jid} is gone but its legs are not"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_leg_has_the_keys_its_job_reads(name):
    """A key a step reads through `matrix.<key>` must be on every leg, or be optional.

    Keys the workflow reads with a truthiness test (`matrix.nonroot && ...`) are optional
    by construction; those read into a name, a label or an env value must be present,
    because an absent key renders as the empty string and the job runs with it.
    """
    matrix_file, jobs = EXPECTED[name]
    doc = _doc(name)
    legs = _legs(matrix_file)
    for jid in jobs:
        job = doc["jobs"][jid]
        read = set(_MATRIX_KEY.findall(json.dumps(job)))
        required = REQUIRED_KEYS[jid]
        assert required <= read, f"{name}:{jid} never reads {sorted(required - read)}"
        declared = [set(leg) - {"pr"} for leg in legs[_key(jid)]]
        for leg in legs[_key(jid)]:
            assert "pr" in leg and isinstance(
                leg["pr"], bool
            ), f"{matrix_file}:{jid}: {leg} has no bool `pr`"
            missing = required - set(leg)
            assert not missing, f"{matrix_file}:{jid}: {leg} lacks {sorted(missing)}"
        unknown = read - {k for leg in declared for k in leg}
        assert not unknown, f"{name}:{jid} reads matrix keys no leg declares: {sorted(unknown)}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_pr_subset_and_the_full_set_are_the_sizes_the_headers_claim(name):
    matrix_file, jobs = EXPECTED[name]
    legs = _legs(matrix_file)
    for jid, (total, on_pr) in jobs.items():
        rows = legs[_key(jid)]
        assert len(rows) == total, f"{matrix_file}:{jid}: {len(rows)} legs, expected {total}"
        picked = [leg for leg in rows if leg["pr"]]
        assert len(picked) == on_pr, f"{matrix_file}:{jid}: {len(picked)} PR legs, expected {on_pr}"


def test_the_clean_machine_pr_subset_keeps_a_macos_leg():
    """test_macos_slots_per_commit.py must keep counting the file as a macOS workflow."""
    legs = _legs(EXPECTED["clean-machine-install-ci.yml"][0])
    assert any(leg["pr"] and str(leg.get("os", "")).startswith("macos-") for leg in legs["macos"])


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_job_whose_pr_subset_is_empty_gates_on_the_count(name):
    """An empty `include` is a workflow error, not a job with no legs."""
    matrix_file, jobs = EXPECTED[name]
    doc = _doc(name)
    outputs = doc["jobs"]["select"].get("outputs") or {}
    for jid, (_, on_pr) in jobs.items():
        if on_pr:
            continue
        job = doc["jobs"][jid]
        gate = f"needs.select.outputs.{_key(jid)}_count != '0'"
        assert gate in str(job.get("if", "")), f"{name}:{jid} has no PR legs and no `if: {gate}`"
        assert (
            f"{_key(jid)}_count" in outputs
        ), f"{name}: the select job does not expose {_key(jid)}_count"
    # The container probe exists only to gate the container install rows.
    if name == "clean-machine-install-ci.yml":
        probe = doc["jobs"]["windows_container_probe"]
        assert "windows_container_install_count != '0'" in str(probe.get("if", ""))


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("event", ["pull_request", "schedule", "push", "workflow_dispatch"])
def test_the_selector_emits_the_legs_without_the_pr_flag(name, event):
    matrix_file, jobs = EXPECTED[name]
    out = _select(matrix_file, event)
    for jid, (total, on_pr) in jobs.items():
        matrix = json.loads(out[_key(jid)])
        assert list(matrix) == [
            "include"
        ], f"{jid}: the matrix must be a mapping with only `include`"
        expected = on_pr if event == "pull_request" else total
        assert (
            len(matrix["include"]) == expected
        ), f"{jid} on {event}: {len(matrix['include'])} legs"
        assert out[f"{_key(jid)}_count"] == str(expected)
        for leg in matrix["include"]:
            assert "pr" not in leg, f"{jid}: the `pr` flag leaked into the matrix: {leg}"
        assert "\n" not in out[_key(jid)], f"{jid}: multi-line JSON would truncate in GITHUB_OUTPUT"


def test_the_wsl_leg_is_off_on_pull_requests():
    """Not a matrix job, so it gates on the event itself; the header says why."""
    doc = _doc("clean-machine-install-ci.yml")
    assert doc["jobs"]["wsl"].get("if") == "github.event_name != 'pull_request'"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_nightly_and_the_matrix_file_are_wired_into_the_trigger(name):
    """The full set has to run somewhere, and a matrix edit has to run the workflow."""
    matrix_file, _ = EXPECTED[name]
    doc = _doc(name)
    on = doc.get(True) if True in doc else doc.get("on")
    assert on.get("schedule"), f"{name}: no schedule, so the nightly-only legs never run"
    for trigger in ("pull_request", "push"):
        paths = (on.get(trigger) or {}).get("paths")
        if paths is None:
            continue
        for needed in (matrix_file, ".github/scripts/select_install_matrix.py"):
            assert needed in paths, f"{name}: {trigger} paths do not list {needed}"
