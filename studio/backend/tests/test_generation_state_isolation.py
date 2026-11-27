# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generation fences must be reset between tests in conftest, since they persist on module state."""

from __future__ import annotations

import ast
from pathlib import Path

from state import active_generations


_CONFTEST = Path(__file__).resolve().parent / "conftest.py"
_FIXTURE = "_isolate_generation_state"


def test_reset_for_tests_clears_both_globals():
    """The primitive the fixture leans on. If this stops clearing, the fixture is decorative."""
    active_generations._FENCED.add("account-under-test")
    active_generations._ACTIVE["account-under-test"] = object()

    active_generations.reset_for_tests()

    assert active_generations._FENCED == set()
    assert active_generations._ACTIVE == {}


def test_this_test_did_not_inherit_a_fence():
    """Fence set starts empty for every test, regardless of what ran earlier in the worker."""
    assert active_generations._FENCED == set()


def test_conftest_isolates_the_generation_state_for_every_test():
    """Checks conftest source structurally, so deleting the autouse isolation fixture fails here."""
    tree = ast.parse(_CONFTEST.read_text(encoding = "utf-8"))
    fixtures = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == _FIXTURE
    ]
    assert fixtures, (
        f"conftest.py no longer defines {_FIXTURE}. Account fences are process-global; "
        f"without it a retirement test's fence decides an unrelated test's assertions."
    )

    fixture = fixtures[0]
    autouse = [
        keyword
        for decorator in fixture.decorator_list
        if isinstance(decorator, ast.Call)
        for keyword in decorator.keywords
        if keyword.arg == "autouse" and getattr(keyword.value, "value", False) is True
    ]
    assert autouse, f"{_FIXTURE} is no longer autouse, so it only isolates tests that ask"

    # Reset before and after: either alone leaks state across files or trusts other plugins.
    resets = [
        node
        for node in ast.walk(fixture)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reset_for_tests"
    ]
    assert len(resets) >= 2, (
        f"{_FIXTURE} calls reset_for_tests {len(resets)} time(s); it must reset both before "
        f"the test and after it"
    )
    assert any(
        isinstance(node, ast.Yield) for node in ast.walk(fixture)
    ), f"{_FIXTURE} no longer yields, so nothing runs between its two resets"
