# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest

from unforgettable.eyes.basic import grade_run_action, inspect_tool_result
from unforgettable.rims.detect import detect_test_command, resolve_test_command
from unforgettable.store.records import insert_record
from unforgettable.tests.test_episode import FakeHost


def test_fakehost_run_action_false_is_fail(tmp_path: Path):
    host = FakeHost(tmp_path, [])
    result = asyncio.run(host.run_action("world", "terminal", {"command": "false"}))
    assert inspect_tool_result("terminal", result) is not None
    assert grade_run_action("terminal", result) is not None


def test_detect_pytest_ini(tmp_path: Path):
    tree = tmp_path / "proj"
    tree.mkdir()
    (tree / "pytest.ini").write_text("[pytest]\n")
    assert detect_test_command(tree) == "pytest"


def test_detect_pyproject_tool_pytest(tmp_path: Path):
    tree = tmp_path / "proj"
    tree.mkdir()
    (tree / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    assert detect_test_command(tree) == "pytest"


def test_detect_package_json_scripts_test(tmp_path: Path):
    tree = tmp_path / "proj"
    tree.mkdir()
    (tree / "package.json").write_text('{"scripts": {"test": "jest"}}\n')
    assert detect_test_command(tree) == "npm test"


def test_detect_go_mod(tmp_path: Path):
    tree = tmp_path / "proj"
    tree.mkdir()
    (tree / "go.mod").write_text("module example.com/x\n")
    assert detect_test_command(tree) == "go test ./..."


def test_detect_first_match_pytest_ini_beats_package_json(tmp_path: Path):
    tree = tmp_path / "proj"
    tree.mkdir()
    (tree / "pytest.ini").write_text("[pytest]\n")
    (tree / "package.json").write_text('{"scripts": {"test": "jest"}}\n')
    assert detect_test_command(tree) == "pytest"


def test_detect_missing_or_file_or_unreadable_returns_none(tmp_path: Path):
    assert detect_test_command(tmp_path / "missing") is None
    as_file = tmp_path / "not-a-dir"
    as_file.write_text("nope\n")
    assert detect_test_command(as_file) is None
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "pytest.ini").write_text("[pytest]\n")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK | os.X_OK):
            pytest.skip("process can still read chmod-0 dir")
        assert detect_test_command(locked) is None
    finally:
        locked.chmod(stat.S_IRWXU)


def test_resolve_requested_wins_over_procedure(tmp_path: Path, db_path: Path):
    insert_record(
        kind = "procedure",
        title = "test command",
        body = "pytest\n",
        provenance = "human",
        db_path = db_path,
    )
    tree = tmp_path / "proj"
    tree.mkdir()
    (tree / "go.mod").write_text("module example.com/x\n")
    assert resolve_test_command(requested = "npm test", db_path = db_path, tree = tree) == "npm test"


def test_resolve_procedure_title_wins_over_detector(tmp_path: Path, db_path: Path):
    insert_record(
        kind = "procedure",
        title = "Test Command",
        body = "custom-test\n",
        provenance = "human",
        db_path = db_path,
    )
    tree = tmp_path / "proj"
    tree.mkdir()
    (tree / "pytest.ini").write_text("[pytest]\n")
    assert resolve_test_command(requested = None, db_path = db_path, tree = tree) == "custom-test"
