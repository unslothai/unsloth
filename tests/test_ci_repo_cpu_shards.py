# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards that the sharded `Repo tests (CPU)` job still runs every test it used to.

`Repo tests (CPU)` was one job discovering all of `tests/`. It is now three shards on three
runners, and the whole risk of that is a test file landing in no shard: the job goes green
faster having checked less, and nothing says so.

So the split is not three allowlists. Two shards name their roots and the third is
"everything under tests/ except those roots", which makes exactly-one-shard a property of
the shape rather than of anyone remembering to edit the workflow. These tests hold that
shape: every directory under tests/ is claimed once, the catch-all really is a catch-all,
and the directories no shard runs are the ones that were already excluded, each still
excluded for the reason another job covers it.

The per-test-id partition was checked directly when the split landed, by running each shard
and the old selection and diffing their JUnit reports: 8263 + 3559 + 7234 = 19056 ids, no
overlap, no gap, and every id in the same state in both. That is a measurement of one tree;
these tests are what keeps it true of the next one.
"""

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

_BACKEND_CI = REPO_ROOT / ".github" / "workflows" / "studio-backend-ci.yml"
_TESTS_DIR = REPO_ROOT / "tests"

# Directories under tests/ that no shard runs, and what runs them instead. A shard losing a
# directory would otherwise look exactly like one of these.
_NOT_IN_ANY_SHARD = {
    "tests/qlora": "GPU-bound: needs real weights",
    "tests/saving": "GPU-bound: needs real weights",
    "tests/utils": "a helpers folder, not a suite",
    "tests/sh": "the shell suite, run by the shell-installer-tests job",
    "tests/vllm_compat": "multi-version drift canary, run by version-compat-ci.yml",
    "tests/version_compat": "multi-version drift canary, run by version-compat-ci.yml",
}


def _backend_ci() -> dict:
    return yaml.safe_load(_BACKEND_CI.read_text(encoding="utf-8"))


def _shards() -> dict:
    """{shard name: (roots, ignores)} parsed out of the job's matrix."""
    job = _backend_ci()["jobs"]["repo-cpu-tests"]
    shards = {}
    for entry in job["strategy"]["matrix"]["include"]:
        roots, ignores = [], []
        tokens = entry["selection"].split()
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.startswith("--ignore="):
                ignores.append(token.split("=", 1)[1].rstrip("/"))
            elif token == "--deselect":
                index += 1  # a test id, not a path
            elif token.startswith("-"):
                pass
            else:
                roots.append(token.rstrip("/"))
            index += 1
        shards[entry["shard"]] = (roots, ignores)
    return shards


def _covers(prefix: str, path: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _claiming_shards(path: str, shards: dict) -> list:
    """Which shards would collect `path`, by pytest's own root-and-ignore rules."""
    claiming = []
    for name, (roots, ignores) in shards.items():
        if any(_covers(ignore, path) for ignore in ignores):
            continue
        if any(_covers(root, path) for root in roots):
            claiming.append(name)
    return sorted(claiming)


def _isolated_paths() -> list:
    """Paths the job runs in their own pytest invocation instead of inside a shard.

    The spoof files mutate hardware.py module globals and tests/studio/load_freeze asserts
    wall-clock bounds, so both are ignored by their shard's discovery and named by a step of
    their own. Read out of those steps rather than listed here, so moving one between the
    two places cannot make it look dropped.
    """
    paths = []
    for step in _backend_ci()["jobs"]["repo-cpu-tests"]["steps"]:
        run = str(step.get("run", ""))
        if "${{ matrix.selection }}" in run or "pytest" not in run:
            continue
        paths.extend(token for token in run.split() if token.startswith("tests/"))
    return paths


def _test_files() -> list:
    return sorted(str(path.relative_to(REPO_ROOT)) for path in _TESTS_DIR.rglob("test_*.py"))


class TestEveryTestFileLandsInExactlyOneShard:
    def test_no_file_is_dropped_or_run_twice(self):
        shards = _shards()
        isolated = _isolated_paths()
        dropped, doubled = [], []
        for path in _test_files():
            if any(_covers(excluded, path) for excluded in _NOT_IN_ANY_SHARD):
                continue
            if any(_covers(named, path) for named in isolated):
                continue  # run by its own step, for a state or timing reason
            claiming = _claiming_shards(path, shards)
            if not claiming:
                dropped.append(path)
            elif len(claiming) > 1:
                doubled.append((path, claiming))
        assert not dropped, (
            "these files are collected by no shard, so Repo tests (CPU) no longer runs "
            f"them: {dropped}"
        )
        assert not doubled, f"these files are collected by more than one shard: {doubled}"

    def test_a_brand_new_top_level_directory_is_claimed(self):
        """The property the catch-all exists for: a directory nobody has heard of yet still
        runs, without a workflow edit."""
        claiming = _claiming_shards("tests/a_directory_added_tomorrow/test_new.py", _shards())
        assert claiming == ["rest"], (
            "a new directory under tests/ must land in the catch-all shard, and only there; "
            f"got {claiming}"
        )

    @pytest.mark.parametrize("excluded", sorted(_NOT_IN_ANY_SHARD))
    def test_the_directories_no_shard_runs_are_the_expected_ones(self, excluded):
        """Excluded on purpose, and still excluded from every shard rather than from one."""
        claiming = _claiming_shards(f"{excluded}/test_anything.py", _shards())
        assert claiming == [], (
            f"{excluded} is {_NOT_IN_ANY_SHARD[excluded]}, but shard(s) {claiming} now "
            "collect it"
        )


class TestTheSplitKeepsTheSelectionItInherited:
    def test_the_deselected_hub_tests_stay_deselected(self):
        """Both hit huggingface_hub for live model existence checks. They live in a shard-3
        file, so shard 3 is where the deselect has to be."""
        selection = _selection_text()
        for test_id in (
            "tests/test_model_registry.py::test_model_registration",
            "tests/test_model_registry.py::test_all_model_registration",
        ):
            assert (
                f"--deselect {test_id}" in selection
            ), f"{test_id} reaches the network and must stay deselected"

    def test_every_shard_keeps_the_marker_filter(self):
        """`server` needs a studio venv and `e2e` needs the network. The filter is on the
        shared step rather than per shard, so it is checked there."""
        step = _shard_pytest_step()
        assert "-m 'not server and not e2e'" in step["run"]

    def test_every_shard_keeps_the_timeouts(self):
        """A shard that loses these reports "cancelled" and names nothing, which is what
        #9515 / #9530 were about."""
        run = _shard_pytest_step()["run"]
        assert "--timeout=330" in run
        assert "timeout --signal=INT --kill-after=60" in run


class TestTheGuardIsNotVacuous:
    def test_a_gap_is_reported(self):
        broken = {"one": (["tests/studio"], []), "two": (["tests/python"], [])}
        assert _claiming_shards("tests/test_orphan.py", broken) == []

    def test_an_overlap_is_reported(self):
        broken = {"one": (["tests/"], []), "two": (["tests/studio"], [])}
        assert _claiming_shards("tests/studio/test_two_owners.py", broken) == ["one", "two"]


def _selection_text() -> str:
    return " ".join(
        " ".join(entry["selection"].split())
        for entry in _backend_ci()["jobs"]["repo-cpu-tests"]["strategy"]["matrix"]["include"]
    )


def _shard_pytest_step() -> dict:
    for step in _backend_ci()["jobs"]["repo-cpu-tests"]["steps"]:
        if "${{ matrix.selection }}" in str(step.get("run", "")):
            return step
    raise AssertionError("the sharded pytest step is gone from repo-cpu-tests")
