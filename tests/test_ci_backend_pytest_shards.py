# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every backend test file must land in exactly one shard: two letter ranges plus a catch-all shard."""

import fnmatch
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

_BACKEND_CI = REPO_ROOT / ".github" / "workflows" / "studio-backend-ci.yml"
_BACKEND_TESTS = REPO_ROOT / "studio" / "backend" / "tests"

# Named once so a rename fails here rather than silently weakening the tests.
_CATCH_ALL = "rest"

# Read from the workflow so moving a file between the parallel and serial steps is not a drop.
_SERIAL_STEP = "Backend tests that cannot share a worker"

# Files this job runs nowhere, so a lost file cannot hide among them.
_NOT_IN_ANY_SHARD = {
    "tests/test_studio_api.py": (
        "end-to-end against a live model and a GGUF download, which a GPU-less runner "
        "cannot do; the workflow's own header says so"
    ),
}


def _backend_ci() -> dict:
    return yaml.safe_load(_BACKEND_CI.read_text(encoding = "utf-8"))


def _job() -> dict:
    return _backend_ci()["jobs"]["pytest"]


def _parse_selection(tokens: list) -> tuple:
    """(roots, ignores, ignore globs) out of a pytest argument list."""
    roots, ignores, globs = [], [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--ignore-glob="):
            globs.append(token.split("=", 1)[1])
        elif token.startswith("--ignore="):
            ignores.append(token.split("=", 1)[1].rstrip("/"))
        elif token == "--deselect":
            index += 1  # a test id, not a path
        elif token.startswith("-"):
            pass
        elif token.startswith("tests/") or token == "tests":
            roots.append(token.rstrip("/"))
        index += 1
    return roots, ignores, globs


def _shards() -> dict:
    """Per-shard selection including the shared --ignore flags, which live on the step, not the matrix."""
    _, shared_ignores, shared_globs = _parse_selection(_shared_pytest_step()["run"].split())
    shards = {}
    for entry in _job()["strategy"]["matrix"]["include"]:
        if "selection" not in entry:
            continue
        roots, ignores, globs = _parse_selection(entry["selection"].split())
        shards[entry["shard"]] = (roots, ignores + shared_ignores, globs + shared_globs)
    return shards


def _covers(prefix: str, path: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _glob_hits(pattern: str, path: str) -> bool:
    """Matches as pytest's --ignore-glob does: fnmatch against the absolute path, with */ prepended."""
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"*/{pattern}")


def _claiming_shards(path: str, shards: dict) -> list:
    """Which shards would collect `path`, by pytest's own root-and-ignore rules."""
    claiming = []
    for name, (roots, ignores, globs) in shards.items():
        if any(_covers(ignore, path) for ignore in ignores):
            continue
        if any(_glob_hits(pattern, path) for pattern in globs):
            continue
        if any(_covers(root, path) for root in roots):
            claiming.append(name)
    return sorted(claiming)


def _shared_pytest_step() -> dict:
    for step in _job()["steps"]:
        if "${{ matrix.selection }}" in str(step.get("run", "")):
            return step
    raise AssertionError("the sharded pytest step is gone from the backend pytest job")


def _serial_paths() -> list:
    """The files the serial step names, read out of the step itself."""
    for step in _job()["steps"]:
        if step.get("name") == _SERIAL_STEP:
            return [token for token in str(step["run"]).split() if token.startswith("tests/")]
    raise AssertionError(f"the {_SERIAL_STEP!r} step is gone from the backend pytest job")


def _test_files() -> list:
    return sorted(
        str(path.relative_to(_BACKEND_TESTS.parent))
        for path in _BACKEND_TESTS.rglob("test_*.py")
        if "__pycache__" not in path.parts
    )


class TestEveryTestFileLandsInExactlyOneShard:
    def test_no_file_is_dropped_or_run_twice(self):
        shards = _shards()
        serial = _serial_paths()
        dropped, doubled = [], []
        for path in _test_files():
            if path in serial or path in _NOT_IN_ANY_SHARD:
                continue
            claiming = _claiming_shards(path, shards)
            if not claiming:
                dropped.append(path)
            elif len(claiming) > 1:
                doubled.append((path, claiming))
        assert not dropped, (
            "these files are collected by no shard, so the backend pytest job no longer "
            f"runs them: {dropped}"
        )
        assert not doubled, f"these files are collected by more than one shard: {doubled}"

    @pytest.mark.parametrize("path", sorted(_NOT_IN_ANY_SHARD))
    def test_the_files_no_shard_runs_are_the_expected_ones(self, path):
        """Excluded on purpose, and from every shard rather than from one: without this, a
        dropped file and a deliberately-skipped one look identical to the test above."""
        claiming = _claiming_shards(path, _shards())
        assert (
            claiming == []
        ), f"{path} is {_NOT_IN_ANY_SHARD[path]}, but shard(s) {claiming} now collect it"
        assert (
            _BACKEND_TESTS.parent / path
        ).is_file(), f"{path} no longer exists, so this exclusion is stale"

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("tests/test_apple.py", "a-k"),
            ("tests/test_kiwi.py", "a-k"),
            ("tests/test_lemon.py", "l-r"),
            ("tests/test_raisin.py", "l-r"),
            ("tests/test_squash.py", _CATCH_ALL),
            ("tests/test_yam.py", _CATCH_ALL),
        ],
    )
    def test_a_brand_new_file_is_claimed_by_exactly_one_shard(self, path, expected):
        """The property the split exists for: a file nobody has heard of yet runs, in one
        place, without a workflow edit."""
        claiming = _claiming_shards(path, _shards())
        assert claiming == [
            expected
        ], f"{path} must be collected by {expected} and only there; got {claiming}"

    @pytest.mark.parametrize(
        "path",
        [
            "tests/a_directory_added_tomorrow/test_new.py",
            "tests/multi_account/test_alice_bob_matrix.py",
            "tests/test_Zebra.py",
            "tests/test_9_regression.py",
            "tests/an_odd_name_test.py",
        ],
        ids = ["new-subdir", "existing-subdir", "uppercase", "digit", "underscore-test-suffix"],
    )
    def test_the_names_the_ranges_do_not_describe_land_in_the_catch_all(self, path):
        """Ranged shards exclude their complement, else odd names like uppercase files are claimed
        by both."""
        claiming = _claiming_shards(path, _shards())
        assert claiming == [_CATCH_ALL], (
            f"{path} is not described by either range, so it must land in the "
            f"{_CATCH_ALL!r} shard and only there; got {claiming}"
        )

    def test_a_name_matching_both_discovery_patterns_falls_out_of_every_shard(self):
        """Names matching both discovery patterns land in no shard; a known limit, forbidden by name
        below."""
        assert _claiming_shards("tests/test_api_test.py", _shards()) == [], (
            "this is a known limitation; if it now lands in a shard the patterns have "
            "changed and the naming rule below can be relaxed"
        )

    def test_a_subdirectory_named_like_a_test_file_falls_out_of_every_shard(self):
        """A subdirectory named like a test file falls out of every shard: a known limit, forbidden
        by name."""
        assert _claiming_shards("tests/test_api/test_auth.py", _shards()) == [], (
            "this is a known limitation of the split; if it now lands in a shard the "
            "patterns have changed and the naming rule below can be dropped"
        )

    def test_no_test_subdirectory_is_named_like_a_test_file(self):
        """Enforces the invariant the split depends on, so the hole above stays unreachable."""
        dirs = sorted(
            path.name
            for path in _BACKEND_TESTS.iterdir()
            if path.is_dir() and path.name.startswith("test_")
        )
        assert not dirs, (
            f"{dirs} would be collected by no shard, because the catch-all's "
            "tests/test_[a-r]*.py excludes nested paths too (fnmatch * crosses /). Rename "
            "the directory, or give the catch-all an explicit root for it."
        )
        hybrids = sorted(
            path.name
            for path in _BACKEND_TESTS.iterdir()
            if path.is_file() and path.name.startswith("test_") and path.name.endswith("_test.py")
        )
        assert not hybrids, (
            f"{hybrids} match both default discovery patterns, so the ranged shards drop "
            "them for the _test.py suffix and the catch-all drops them for the test_[a-r]* "
            "range, leaving them in no shard. Drop one of the two markers from the name."
        )


class TestTheSplitKeepsTheSelectionItInherited:
    def test_every_shard_keeps_the_marker_filter(self):
        """The one environment-specific deselection: a real llama.cpp process, which a
        GPU-less runner does not have. The filter is on the shared step rather than per
        shard, so it is checked there."""
        run = _shared_pytest_step()["run"]
        assert (
            "not llama_cpp_load_progress_live" in run
        ), "the -k filter lost the live llama.cpp test"

    def test_mocked_gpu_tests_are_not_deselected(self):
        """These classes patch torch.cuda and the device probe, so they pass on CPU torch.
        They were deselected as "needs a real GPU" from May to October 2026 and ran nowhere;
        a deselection of them here would hide them again."""
        run = _shared_pytest_step()["run"]
        for name in (
            "TestGpuAutoSelection",
            "TestPreSpawnGpuResolution",
            "TestPerGpuFitGuardAllCounts",
            "TestTransformersIntrospection",
            "test_returns_cuda_when_cuda_available",
            "test_calls_cuda_cache_when_cuda",
        ):
            assert (
                f"not {name}" not in run
            ), f"{name} is mocked and must run here, not be deselected"

    def test_every_shard_keeps_the_timeouts(self):
        """A shard that loses these reports "cancelled" and names nothing, which is what
        #9515 / #9530 were about."""
        run = _shared_pytest_step()["run"]
        assert "--timeout=330" in run
        assert "timeout --signal=INT --kill-after=60" in run

    def test_the_serial_files_are_ignored_by_every_shard(self):
        """The twelve cannot share a worker, and sharding does not change that: three
        runners are still four workers each. They are held out by the shared --ignore list
        on the step, so this asks the question of the shard AND the step together."""
        run = _shared_pytest_step()["run"]
        shards = _shards()
        for path in _serial_paths():
            assert f"--ignore={path}" in run, (
                f"{path} is rerun by the serial step and no longer ignored by the parallel "
                f"one, so it runs twice and its timing assertions run under four workers"
            )
            claiming = _claiming_shards(path, shards)
            assert (
                claiming == []
            ), f"{path} is rerun by the serial step and shard(s) {claiming} collect it too"

    def test_the_serial_step_runs_in_one_process_in_one_shard(self):
        """Splitting it would put its relative timings back on two machines, and running it
        on every shard would run it three times."""
        for step in _job()["steps"]:
            if step.get("name") != _SERIAL_STEP:
                continue
            assert " -n " not in f" {step['run']} ", "the serial step is running under xdist"
            condition = str(step.get("if", ""))
            assert "matrix.shard ==" in condition, (
                f"the serial step is no longer pinned to one shard, so it runs once per "
                f"shard: {condition!r}"
            )
            pinned = [name for name in _shards() if f"matrix.shard == '{name}'" in condition]
            assert len(pinned) == 1, f"expected exactly one shard named in {condition!r}"
            return
        raise AssertionError(f"the {_SERIAL_STEP!r} step is gone")

    def test_the_serial_step_runs_the_thirteen_isolated_files(self):
        """The serial selection includes the timing-sensitive R1 parser regressions."""
        paths = _serial_paths()
        assert len(paths) == 13, f"the serial step runs {len(paths)} files, not 13: {paths}"
        assert len(set(paths)) == 13, f"the serial step names a file twice: {paths}"
        assert "tests/test_pr5624_regressions.py" in paths
        missing = [path for path in paths if not (_BACKEND_TESTS.parent / path).is_file()]
        assert not missing, f"the serial step names files that do not exist: {missing}"


class TestTheGuardIsNotVacuous:
    def test_a_gap_is_reported(self):
        """Shard 3 stops being a catch-all and starts being an allowlist."""
        broken = {
            "a-k": (["tests/"], [], ["tests/*/*", "tests/*_test.py", "tests/test_[!a-k]*.py"]),
            "l-r": (["tests/"], [], ["tests/*/*", "tests/*_test.py", "tests/test_[!l-r]*.py"]),
            _CATCH_ALL: (["tests/test_squash.py"], [], []),
        }
        assert _claiming_shards("tests/test_yam.py", broken) == []
        assert _claiming_shards("tests/multi_account/test_alice_bob_matrix.py", broken) == []

    def test_an_overlap_is_reported(self):
        """The mistake the ranges are written to avoid: shard 1 excluding `l-z` rather than
        excluding `not a-k`, which leaves every name outside a-z owned by both."""
        broken = {
            "a-k": (["tests/"], [], ["tests/*/*", "tests/test_[l-z]*.py"]),
            "l-r": (["tests/"], [], ["tests/*/*", "tests/test_[a-k]*.py", "tests/test_[s-z]*.py"]),
            _CATCH_ALL: (["tests/"], [], ["tests/test_[a-r]*.py"]),
        }
        # A name outside a-z falls into the catch-all as well, while a-z still partitions correctly.
        assert _claiming_shards("tests/test_Zebra.py", broken) == ["a-k", "l-r", _CATCH_ALL]
        assert _claiming_shards("tests/test_apple.py", broken) == ["a-k"]
        assert _claiming_shards("tests/test_monkey.py", broken) == ["l-r"]

    def test_the_glob_rule_matches_pytest(self):
        """`tests/test_[a-k]*.py` must not reach below a subdirectory, or shard 3 would lose
        multi_account to a shard that never names it. Measured against pytest directly when
        this landed; asserted here so a rewrite of _glob_hits cannot quietly change it."""
        assert _glob_hits("tests/test_[a-k]*.py", "tests/test_apple.py")
        assert not _glob_hits("tests/test_[a-k]*.py", "tests/multi_account/test_alice.py")
        assert _glob_hits("tests/*/*", "tests/multi_account/test_alice.py")
        assert not _glob_hits("tests/*/*", "tests/test_apple.py")
