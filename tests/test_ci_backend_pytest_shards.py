# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards that the sharded backend `pytest` job still runs every test it used to.

`(Python 3.13)` was one job discovering all of studio/backend/tests; it is now three shards
on three runners, and the whole risk is a file landing in no shard: the job goes green
faster having checked less, and nothing says so.

So the split is not three allowlists. studio/backend/tests is flat (923 files, plus 17 under
multi_account), so there are no directory roots to divide and the division is by the first
letter after `test_`. Shards 1 and 2 name their range and exclude everything else; shard 3 is
"tests/ except those two ranges", which makes exactly-one-shard a property of the shape
rather than of anyone remembering to edit the workflow. These tests hold that shape, and hold
that the inherited selection -- the -k filter, the two timeouts, and the twelve files the
serial step reruns -- came through intact.

All three shards root at `tests/` and differ only in ignores, because `--ignore=FILE` does
not filter a file passed as an explicit argument (checked against pytest): a shard rooted at
a shell-expanded glob would name the twelve serial files and run them anyway.

The per-test-id partition was checked directly when the split landed, via --junitxml diffs:
13,073 + 17,483 + 12,053 = 42,609 against the baseline's 42,609, no overlap, no gap, every id
in the same state bar three failure-to-passed flips, none caused by the split (two flip
between two runs of the UNSHARDED selection over the same tree; the third asserts four
concurrent sizings cannot all claim the same 8GB, which fails on a loaded box). That measured
one tree; these tests keep it true of the next one.
"""

import fnmatch
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

_BACKEND_CI = REPO_ROOT / ".github" / "workflows" / "studio-backend-ci.yml"
_BACKEND_TESTS = REPO_ROOT / "studio" / "backend" / "tests"

# Named once so the tests can say "and only there", and so a rename fails here rather than
# silently weakening them.
_CATCH_ALL = "rest"

# The twelve files the parallel run ignores and the serial step reruns. Read out of the
# workflow rather than listed here, so moving one between the two cannot make it look
# dropped, as tests/test_ci_repo_cpu_shards.py does for its isolated paths.
_SERIAL_STEP = "Backend tests that cannot share a worker"

# Files this job runs nowhere: a lost file would otherwise look like one of these. Only one,
# predating the split -- the thirteenth --ignore on the parallel step, which no step reruns.
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
    """{shard name: (roots, ignores, ignore globs)} for the command each shard really runs.

    The matrix carries only what the shards DIFFER by. The thirteen --ignore flags they
    all share sit on the step, so a model built from the matrix alone would have the
    catch-all shard collecting the twelve files the serial step reruns -- it does not, and
    reporting that it does would either fail honestly-written guards or push the flags into
    three copies that have to agree. Both halves of the command, then.

    The 3.11 floor-spot-check leg is not a shard of anything and carries no selection, so
    it is not in here.
    """
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
    """pytest's own --ignore-glob rule, asked of a path relative to studio/backend.

    pytest matches with `_pytest.pathlib.fnmatch_ex`, which fnmatches the ABSOLUTE path
    against the pattern with `*/` prepended when the pattern contains a separator and is
    relative. fnmatch's `*` crosses `/`, so the prepended prefix is free and matching the
    tail of the relative path asks the same question.
    """
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
                continue  # rerun by the serial step, or excluded from the job on purpose
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
        """Why shards 1 and 2 exclude their complement rather than name their range.

        Not every name starts with a lowercase letter: a subdirectory, an uppercase or digit
        first character, and pytest's other discovery pattern `*_test.py` are outside both
        ranges, and each would be claimed by BOTH ranged shards had they said "not l-z" and
        "not a-k". Only the catch-all may be open-ended, which is what makes exactly-one hold
        for names nobody anticipated.
        """
        claiming = _claiming_shards(path, _shards())
        assert claiming == [_CATCH_ALL], (
            f"{path} is not described by either range, so it must land in the "
            f"{_CATCH_ALL!r} shard and only there; got {claiming}"
        )

    def test_a_name_matching_both_discovery_patterns_falls_out_of_every_shard(self):
        """`test_api_test.py` matches `test_*.py` AND `*_test.py`, and lands nowhere.

        The ranged shards drop it for the suffix, the catch-all for the `test_[a-r]*` range;
        measured against real pytest, all three collect it zero times. Fixing it would need
        the suffix rule to mean "ends in _test.py unless it starts with test_", which fnmatch
        cannot say: diverging per prefix character over-consumes `te_test.py` into BOTH ranged
        shards (tried, and this guard caught it), and completing it needs nine globs per shard
        for a shape this repo does not use. Forbidden by name below instead.
        """
        assert _claiming_shards("tests/test_api_test.py", _shards()) == [], (
            "this is a known limitation; if it now lands in a shard the patterns have "
            "changed and the naming rule below can be relaxed"
        )

    def test_a_subdirectory_named_like_a_test_file_falls_out_of_every_shard(self):
        """The one shape the catch-all cannot absorb. Recorded, not hidden.

        fnmatch's `*` crosses `/` and fnmatch_ex matches the whole path, so the catch-all's
        `tests/test_[a-r]*.py` also excludes `tests/test_api/test_auth.py`, which the ranged
        shards already exclude via `tests/*/*`. Measured against real pytest: zero collections
        in all three. Both obvious repairs were measured and fail -- `[!/]` cannot stop `*`
        crossing a separator, and an extra root does not bypass --ignore-glob -- so the test
        below enforces the invariant by name instead.
        """
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
        """The environment-specific deselections: live GPU introspection and a real
        llama.cpp process, none of which exist on a GPU-less runner. The filter is on the
        shared step rather than per shard, so it is checked there."""
        run = _shared_pytest_step()["run"]
        for fragment in (
            "not llama_cpp_load_progress_live",
            "not TestGpuAutoSelection",
            "not TestPreSpawnGpuResolution",
            "not TestPerGpuFitGuardAllCounts",
            "not TestTransformersIntrospection",
            "not test_returns_cuda_when_cuda_available",
            "not test_calls_cuda_cache_when_cuda",
        ):
            assert fragment in run, f"the -k filter lost {fragment!r}"

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

    def test_the_serial_step_still_runs_the_same_twelve_files(self):
        """Not eleven and not thirteen. Whether any of them could go back into the parallel
        run was audited separately and the answer was no, so the set is carried across the
        split unchanged; a file leaving it here is a coverage change wearing a refactor's
        clothes."""
        paths = _serial_paths()
        assert len(paths) == 12, f"the serial step runs {len(paths)} files, not 12: {paths}"
        assert len(set(paths)) == 12, f"the serial step names a file twice: {paths}"
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
        # All three, in fact: a name outside a-z is outside every range, so the catch-all
        # takes it as well and the ranged shards no longer exclude it. The names in a-z are
        # still partitioned correctly, which is why this is the mistake that survives review.
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
