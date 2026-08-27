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

from unforgettable.eyes.basic import (
    grade_run_action,
    inspect_tool_result,
    user_declares_failure,
)


def test_pytest_failures_banner_is_fail():
    blob = (
        "============================= test session starts ==============================\n"
        "=========================== FAILURES ===========================\n"
        "___________________________ test_foo ___________________________\n"
    )
    fail = inspect_tool_result("python", blob)
    assert fail is not None


def test_pytest_failed_node_is_fail():
    blob = "FAILED tests/test_foo.py::test_bar - assert 1 == 2\n"
    fail = inspect_tool_result("terminal", blob)
    assert fail is not None


def test_pytest_last_line_n_failed_is_fail():
    blob = "collected 3 items\n===== 1 failed, 2 passed in 0.12s =====\n"
    fail = inspect_tool_result("python", blob)
    assert fail is not None


def test_pytest_last_line_failed_eq_is_fail():
    blob = "running tests\nfailed=1\n"
    fail = inspect_tool_result("python", blob)
    assert fail is not None


def test_mid_blob_failed_import_with_clean_last_line_is_not_runner_fail():
    blob = "1 failed to import optional dep\nok\n"
    assert inspect_tool_result("python", blob) is None


def test_failed_import_without_fingerprint_is_not_runner_fail():
    assert inspect_tool_result("python", "failed to import optional dep") is None


def test_unittest_failed_paren_is_fail():
    fail = inspect_tool_result("python", "FAILED (failures=1)")
    assert fail is not None


def test_jest_fail_and_tests_failed():
    blob = "FAIL src/foo.test.js\nTests: 1 failed, 2 passed\n"
    fail = inspect_tool_result("terminal", blob)
    assert fail is not None


def test_go_fail_tab_is_fail_without_fail_space_token():
    fail = inspect_tool_result("terminal", "FAIL\tgithub.com/x/y\t0.01s\n")
    assert fail is not None
    assert inspect_tool_result("python", "build FAIL done\n") is None


def test_rims_enter_sim_source_is_tool():
    fail = inspect_tool_result("rims_enter_sim", "enter_sim requested", contact = "world")
    assert fail is not None
    assert fail.source == "tool"
    assert fail.summary == "enter_sim requested"
    dotted = inspect_tool_result("rims.enter_sim", "ok", contact = "world")
    assert dotted is not None
    assert dotted.source == "tool"


def test_user_declares_failure_phrases():
    assert user_declares_failure("That didn't work.")
    assert user_declares_failure("That didn’t work.")
    assert not user_declares_failure("please try again")


def test_world_studio_sentinels_are_recognized_failures():
    blobs = (
        "Execution timed out after 300 seconds.",
        "Execution cancelled.",
        "Blocked command(s) for safety: rm",
        "Execution error: [Errno 12] Cannot allocate memory",
        "No command provided.",
    )
    for blob in blobs:
        fail = inspect_tool_result("terminal", blob, contact = "world")
        assert fail is not None, repr(blob)


def test_grade_run_action_sentinels_are_fail():
    blobs = (
        "Execution timed out after 300 seconds.",
        "Execution cancelled.",
        "Blocked command(s) for safety: rm",
        "Execution error: [Errno 12] Cannot allocate memory",
        "Error: run_action supports python|terminal only, got 'web_search'",
        "",
    )
    for blob in blobs:
        fail = grade_run_action("terminal", blob)
        assert fail is not None, repr(blob)
