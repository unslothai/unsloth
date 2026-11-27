# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each tests/ directory is claimed by exactly one Repo tests (CPU) shard, the third a catch-all."""

import fnmatch
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

_BACKEND_CI = REPO_ROOT / ".github" / "workflows" / "studio-backend-ci.yml"
# Multi-rank shards are in a separate path-filtered workflow; checks below read both matrices.
_DISTRIBUTED_CI = REPO_ROOT / ".github" / "workflows" / "distributed-cpu-ci.yml"
_SHARD_JOBS = ((_BACKEND_CI, "repo-cpu-tests"), (_DISTRIBUTED_CI, "distributed-cpu-tests"))
_TESTS_DIR = REPO_ROOT / "tests"

# Directories under tests/ that no shard runs, and what runs them instead.
_NOT_IN_ANY_SHARD = {
    "tests/qlora": "GPU-bound: needs real weights",
    "tests/saving": "GPU-bound: needs real weights",
    "tests/utils": "a helpers folder, not a suite",
    "tests/sh": "the shell suite, run by the shell-installer-tests job",
    "tests/vllm_compat": "multi-version drift canary, run by version-compat-ci.yml",
    "tests/version_compat": "multi-version drift canary, run by version-compat-ci.yml",
}


def _backend_ci() -> dict:
    return yaml.safe_load(_BACKEND_CI.read_text(encoding = "utf-8"))


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding = "utf-8"))


def _shard_entries() -> list:
    return [
        entry
        for path, job in _SHARD_JOBS
        for entry in _workflow(path)["jobs"][job]["strategy"]["matrix"]["include"]
    ]


def _shards() -> dict:
    """{shard name: (roots, ignores)} parsed out of both jobs' matrices; a split shard's parts count as one."""
    shards, names = {}, set()
    for entry in _shard_entries():
        assert entry["shard"] not in names, f"shard {entry['shard']} is defined twice"
        names.add(entry["shard"])
        name = entry["shard"]
        if entry.get("split"):
            name = "split: " + " ".join(entry["selection"].split())
            if name in shards:
                continue
        roots, ignores = [], []
        tokens = entry["selection"].split()
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.startswith(("--ignore=", "--ignore-glob=")):
                ignores.append(token.split("=", 1)[1].rstrip("/"))
            elif token == "--deselect":
                index += 1  # a test id, not a path
            elif token.startswith("-"):
                pass
            else:
                roots.append(token.rstrip("/"))
            index += 1
        shards[name] = (roots, ignores)
    return shards


def _covers(prefix: str, path: str) -> bool:
    if "*" in prefix:  # a shell-expanded root or an --ignore-glob, matched like pytest does
        return fnmatch.fnmatch(path, prefix)
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _claiming_shards(path: str, shards: dict) -> list:
    """Which shards would collect `path`, by pytest's own root-and-ignore rules."""
    claiming = []
    for name, (roots, ignores) in shards.items():
        # --ignore only prunes while recursing into dirs, so a file root is collected anyway.
        if any(root.endswith(".py") and _covers(root, path) for root in roots):
            claiming.append(name)
            continue
        if any(_covers(ignore, path) for ignore in ignores):
            continue
        if any(_covers(root, path) for root in roots):
            claiming.append(name)
    return sorted(claiming)


def _isolated_paths() -> list:
    """Spoof files and load_freeze run in a separate step: they mutate module globals or assert timing."""
    paths = []
    steps = [step for path, job in _SHARD_JOBS for step in _workflow(path)["jobs"][job]["steps"]]
    for step in steps:
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
                continue
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
        """The catch-all lets a new directory under tests/ run without a workflow edit."""
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


class TestEveryGlobRootMatchesAFile:
    def test_no_shard_hands_pytest_an_unexpanded_pattern(self):
        """Unmatched glob reaches pytest as is and fails the shard; no split may empty a glob."""
        files = _test_files()
        empty = [
            (name, root)
            for name, (roots, _) in _shards().items()
            for root in roots
            if "*" in root and not any(fnmatch.fnmatch(path, root) for path in files)
        ]
        assert not empty, f"these shard globs match no test file: {empty}"

    def test_an_empty_glob_is_reported(self):
        assert not any(
            fnmatch.fnmatch(path, "tests/test_no_such_prefix_*.py") for path in _test_files()
        )


class TestTheSplitKeepsTheSelectionItInherited:
    def test_the_deselected_hub_tests_stay_deselected(self):
        """Both hit huggingface_hub live; the deselect acts only in the file's own shard."""
        owners = _claiming_shards("tests/test_model_registry.py", _shards())
        assert len(owners) == 1, f"tests/test_model_registry.py is collected by {owners}"
        selection = _selection_text(owners[0])
        for test_id in (
            "tests/test_model_registry.py::test_model_registration",
            "tests/test_model_registry.py::test_all_model_registration",
        ):
            assert f"--deselect {test_id}" in selection, (
                f"{test_id} reaches the network and must stay deselected in {owners[0]}, the "
                "shard that runs its file"
            )

    def test_every_shard_keeps_the_marker_filter(self):
        """`server` needs studio venv, `e2e` the network; the filter sits on the shared step."""
        step = _shard_pytest_step()
        assert "-m 'not server and not e2e'" in step["run"]

    def test_every_shard_keeps_the_timeouts(self):
        """A shard that loses these reports "cancelled" and names nothing, which is what
        #9515 / #9530 were about."""
        run = _shard_pytest_step()["run"]
        assert "--timeout=330" in run
        assert "timeout --signal=INT --kill-after=60" in run
        # SIGALRM cannot interrupt a test blocked in C code; only the thread method names it.
        assert "--timeout-method=thread" in run

    def test_the_split_out_shards_run_the_same_step_and_install(self):
        """Multi-rank shards moved workflows only: same pytest step, install and environment."""

        def named(path, job):
            return {
                step.get("name"): step
                for step in _workflow(path)["jobs"][job]["steps"]
                if step.get("name")
            }

        backend, distributed = named(*_SHARD_JOBS[0]), named(*_SHARD_JOBS[1])
        for name in (
            "Repo tests (CPU, auto-discovered)",
            "Install deps (shared shape with backend pytest job)",
            "Install the pinned ruff-unsloth (formatter fixed-point guard)",
        ):
            assert name in distributed, f"distributed-cpu-ci.yml lost the {name!r} step"
            for field in ("run", "env"):
                assert backend[name].get(field) == distributed[name].get(
                    field
                ), f"{name!r} differs between the two workflows in {field!r}"
        jobs = [_workflow(path)["jobs"][job] for path, job in _SHARD_JOBS]
        for field in ("runs-on", "timeout-minutes", "name"):
            assert jobs[0][field] == jobs[1][field], f"the two jobs differ in {field!r}"


class TestTheDistributedShardsForkTheirRanks:
    """Job-level env, so the shared pytest step stays byte-identical."""

    def _env(self, path, job):
        return _workflow(path)["jobs"][job].get("env") or {}

    def test_the_distributed_job_turns_the_zygote_on(self):
        env = self._env(*_SHARD_JOBS[1])
        assert env.get("UNSLOTH_TEST_RANK_ZYGOTE") == "${{ matrix.rank_zygote || '1' }}"
        entries = _workflow(_DISTRIBUTED_CI)["jobs"]["distributed-cpu-tests"]["strategy"]["matrix"][
            "include"
        ]
        assert all(entry.get("rank_zygote", "1") in ("0", "1") for entry in entries)
        assert any("rank_zygote" not in entry for entry in entries)
        assert env.get("UNSLOTH_TEST_RANK_ZYGOTE_STATS"), "the summary step reads this directory"
        steps = _workflow(_DISTRIBUTED_CI)["jobs"]["distributed-cpu-tests"]["steps"]
        summary = [s for s in steps if "rank_zygote.py --summary" in (s.get("run") or "")]
        assert summary and summary[0].get("if") == "always()"
        assert env["UNSLOTH_TEST_RANK_ZYGOTE_STATS"] in summary[0]["run"]

    def test_backend_ci_does_not(self):
        assert "UNSLOTH_TEST_RANK_ZYGOTE" not in self._env(*_SHARD_JOBS[0])
        assert "UNSLOTH_TEST_RANK_ZYGOTE" not in str(_shard_pytest_step().get("env"))

    def test_the_conftest_installs_it_only_under_the_flag(self):
        text = (_TESTS_DIR / "conftest.py").read_text(encoding = "utf-8")
        assert 'if os.environ.get("UNSLOTH_TEST_RANK_ZYGOTE") == "1":' in text
        assert "rank_zygote.install()" in text


class TestTheDistributedWorkflowSeesWhatItsTestsRead:
    """distributed-cpu-ci.yml runs only when its paths change, so its paths must cover every
    tests/ file its shards import, directly or through a sibling test module or helper."""

    def _paths(self, event: str) -> list:
        on = _workflow(_DISTRIBUTED_CI).get(True) or _workflow(_DISTRIBUTED_CI).get("on")
        return on[event]["paths"]

    def _shard_files(self) -> list:
        files = _test_files()
        roots = [
            root
            for entry in _workflow(_DISTRIBUTED_CI)["jobs"]["distributed-cpu-tests"]["strategy"][
                "matrix"
            ]["include"]
            for root in entry["selection"].split()
            if not root.startswith("-")
        ]
        return sorted({path for path in files for root in roots if fnmatch.fnmatch(path, root)})

    def _local_imports(self, start: list) -> set:
        import ast

        def resolve(module: str):
            parts = module.split(".")
            if parts[0] == "tests":
                parts = parts[1:]
            for base in (_TESTS_DIR, _TESTS_DIR / "_shared"):
                candidate = base.joinpath(*parts)
                for found in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                    if parts and found.is_file():
                        return str(found.relative_to(REPO_ROOT))
            return None

        seen, queue = set(), list(start)
        while queue:
            path = queue.pop()
            if path in seen:
                continue
            seen.add(path)
            try:
                tree = ast.parse((REPO_ROOT / path).read_text(encoding = "utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
                for name in names:
                    found = resolve(name)
                    if found and found not in seen:
                        queue.append(found)
        return seen

    def test_the_shards_select_files(self):
        assert len(self._shard_files()) > 100, "the distributed shards select almost nothing"

    def test_every_imported_test_file_triggers_the_workflow(self):
        start = self._shard_files() + ["tests/conftest.py"]
        paths = self._paths("pull_request")
        missed = sorted(
            path
            for path in self._local_imports(start)
            if not any(fnmatch.fnmatch(path, pattern.replace("**", "*")) for pattern in paths)
        )
        assert not missed, (
            "distributed-cpu-ci.yml runs these files (or imports them from its tests) but no "
            f"path in its pull_request filter matches them, so editing them skips it: {missed}"
        )

    def test_push_and_pull_request_paths_match(self):
        assert self._paths("push") == self._paths("pull_request")

    def test_unsloth_and_the_conftest_trigger_it(self):
        paths = self._paths("pull_request")
        for needed in ("unsloth/**", "tests/conftest.py", "tests/_shared/**", "pyproject.toml"):
            assert needed in paths, f"{needed} must trigger distributed-cpu-ci.yml"

    def test_a_nightly_run_catches_unsloth_zoo_drift(self):
        """The shards install unsloth_zoo from git main, which no path here can see."""
        on = _workflow(_DISTRIBUTED_CI).get(True) or _workflow(_DISTRIBUTED_CI).get("on")
        assert on.get("schedule"), "distributed-cpu-ci.yml needs its nightly schedule"


class TestTheGuardIsNotVacuous:
    def test_a_gap_is_reported(self):
        broken = {"one": (["tests/studio"], []), "two": (["tests/python"], [])}
        assert _claiming_shards("tests/test_orphan.py", broken) == []

    def test_an_overlap_is_reported(self):
        broken = {"one": (["tests/"], []), "two": (["tests/studio"], [])}
        assert _claiming_shards("tests/studio/test_two_owners.py", broken) == ["one", "two"]


def _selection_text(shard: str | None = None) -> str:
    return " ".join(
        " ".join(entry["selection"].split())
        for entry in _shard_entries()
        if shard is None
        or entry["shard"] == shard
        or (entry.get("split") and shard == "split: " + " ".join(entry["selection"].split()))
    )


def _shard_pytest_step() -> dict:
    for step in _backend_ci()["jobs"]["repo-cpu-tests"]["steps"]:
        if "${{ matrix.selection }}" in str(step.get("run", "")):
            return step
    raise AssertionError("the sharded pytest step is gone from repo-cpu-tests")


class TestASplitShardRunsEveryPartOnce:
    def _groups(self):
        groups = {}
        for entry in _workflow(_DISTRIBUTED_CI)["jobs"]["distributed-cpu-tests"]["strategy"][
            "matrix"
        ]["include"]:
            if entry.get("split"):
                groups.setdefault(" ".join(entry["selection"].split()), []).append(entry)
        return groups

    def test_every_part_of_each_split_shard_is_there_once(self):
        groups = self._groups()
        assert groups, "no shard is split any more; drop this class with the mechanism"
        for selection, entries in groups.items():
            specs = sorted(entry["split"] for entry in entries)
            n = len(entries)
            assert specs == sorted(f"{k}/{n}" for k in range(1, n + 1)), (selection, specs)
            assert len({entry.get("workers", 4) for entry in entries}) == 1, selection

    def test_the_job_passes_the_part_to_pytest(self):
        env = _workflow(_DISTRIBUTED_CI)["jobs"]["distributed-cpu-tests"].get("env") or {}
        assert env.get("UNSLOTH_TEST_SPLIT") == "${{ matrix.split || '' }}"
        backend = _workflow(_SHARD_JOBS[0][0])["jobs"][_SHARD_JOBS[0][1]]
        assert "UNSLOTH_TEST_SPLIT" not in str(
            backend.get("env")
        ), "Backend CI shards are never split"

    def test_the_conftest_applies_the_split(self):
        text = (_TESTS_DIR / "conftest.py").read_text(encoding = "utf-8")
        assert "from shard_split import select" in text and "select(config, items)" in text
