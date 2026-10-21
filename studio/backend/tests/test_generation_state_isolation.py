# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account fences must not outlive the test that set them.

``state.active_generations`` keeps ``_ACTIVE`` and ``_FENCED`` on the module, so a fence set
by one test is still there for the next test in the same process. Retirement and deactivation
tests fence on purpose, and that is their correct end state, so the fix is isolation in
conftest rather than a cleanup obligation on each of them.

Without it, ``tests/multi_account/test_routing_invariance.py`` failed on
``assert active_generations._FENCED == set()`` carrying an account id it never created, but
only when an xdist worker happened to run a retirement test first. It passed alone, passed on
some pull requests and failed on others, which is the shape that gets a real test deleted for
being flaky.
"""

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
    """Whatever ran before this in the worker, the fence starts empty.

    Cheap, and it is the exact assertion that was failing in CI, so if the isolation regresses
    this fails in the same place rather than in an unrelated multi_account test.
    """
    assert active_generations._FENCED == set()


def test_conftest_isolates_the_generation_state_for_every_test():
    """Read from the source, so deleting the fixture or dropping autouse is caught here.

    A functional check cannot do this job: it would have to depend on test ORDER to observe a
    leak, and this suite runs under pytest-randomly, where order is not a thing a test may
    assume. So the guard is structural, and the two tests above cover the behaviour.
    """
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

    # Before AND after: clearing only on the way in leaves the last test of a worker holding a
    # fence for whatever the next file does, and clearing only on the way out trusts every
    # other conftest and plugin to have left it alone.
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
