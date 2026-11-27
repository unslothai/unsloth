# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""tests/_shared/xdist_group_check.py: on a loadgroup worker, a grouped test that lost its group tag fails by name."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import xdist_group_check

REPO = Path(__file__).resolve().parents[1]


class _Item:
    def __init__(self, nodeid, grouped):
        self.nodeid = nodeid
        self._grouped = grouped

    def get_closest_marker(self, name):
        return object() if name == "xdist_group" and self._grouped else None


def _config(loadgroup = True, worker = True):
    config = SimpleNamespace(option = SimpleNamespace(loadgroup = loadgroup))
    if worker:
        config.workerinput = {}
    return config


def test_a_grouped_test_without_its_group_tag_fails_with_the_reason():
    lost, plain = _Item("t.py::test_a", True), _Item("t.py::test_b", False)
    assert xdist_group_check.ungrouped(_config(), [lost, plain]) == ["t.py::test_a"]
    xdist_group_check.check(_config(), [lost, plain])
    with pytest.raises(pytest.fail.Exception, match = "did not tag this xdist_group test"):
        lost.runtest()
    assert "runtest" not in vars(plain)


def test_tagged_and_ungrouped_tests_are_left_alone():
    items = [_Item("t.py::test_a@g", True), _Item("t.py::test_b", False)]
    xdist_group_check.check(_config(), items)
    assert all("runtest" not in vars(item) for item in items)


@pytest.mark.parametrize("config", [_config(loadgroup = False), _config(worker = False)])
def test_only_a_loadgroup_worker_is_checked(config):
    item = _Item("t.py::test_a", True)
    xdist_group_check.check(config, [item])
    assert "runtest" not in vars(item)


@pytest.mark.parametrize("conftest", ["tests/conftest.py", "studio/backend/tests/conftest.py"])
def test_both_suites_run_the_check_after_xdist_tags_groups(conftest):
    tree = ast.parse((REPO / conftest).read_text(encoding = "utf-8"))
    hook = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "pytest_collection_modifyitems"
        ),
        None,
    )
    assert hook is not None, f"{conftest} lost the loadgroup check"
    assert "trylast" in ast.unparse(hook.decorator_list[0])
    assert "check" in [ast.unparse(n.func) for n in ast.walk(hook) if isinstance(n, ast.Call)]


@pytest.mark.parametrize(
    "workflow, pin, unpinned",
    [
        (".github/workflows/studio-backend-ci.yml", "'pytest<9.2'", " pytest pytest-"),
        (".github/workflows/version-compat-ci.yml", "'pytest>=8,<9.2'", "'pytest>=8'"),
    ],
)
def test_ci_keeps_pytest_below_the_release_that_drops_xdist_groups(workflow, pin, unpinned):
    text = (REPO / workflow).read_text(encoding = "utf-8")
    assert pin in text
    assert unpinned not in text, f"{workflow} installs an unpinned pytest"
