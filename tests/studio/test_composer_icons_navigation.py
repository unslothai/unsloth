# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The composer icon check survives the dev page reloading itself mid-case.

On Windows runners the vite page sometimes navigated after its buttons rendered, and the next
evaluate died with "Execution context was destroyed" (main job 119495424607, PR 14592 job
119703801606). A reload also resets the display case, so re-running only the measurement would
record the default layout under the case's name: the case must be applied again with it.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "playwright_composer_icons.py"
DEFAULT = dict(font = 1, zoom = 1, dark = False, direction = "ltr")
CASE = dict(engine = "chromium", dpr = 2, font = 1.25, zoom = 0.8, dark = True, direction = "rtl")
LOST = "Page.evaluate: Execution context was destroyed, most likely because of a navigation"


@pytest.fixture
def mod(monkeypatch):
    if "playwright.sync_api" not in sys.modules:
        pkg = types.ModuleType("playwright")
        api = types.ModuleType("playwright.sync_api")
        api.sync_playwright = lambda: None
        pkg.sync_api = api
        monkeypatch.setitem(sys.modules, "playwright", pkg)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", api)
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("composer_icons_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePage:
    """Holds the applied display case; a reload resets it and kills the evaluate in flight."""

    def __init__(
        self,
        reloads_on_calls = (),
        error = LOST,
    ):
        self.state = dict(DEFAULT)
        self.calls = 0
        self.reloads_on_calls = set(reloads_on_calls)
        self.error = error
        self.waits = 0

    def evaluate(
        self,
        expression,
        arg = None,
    ):
        self.calls += 1
        if self.calls in self.reloads_on_calls:
            self.state = dict(DEFAULT)
            raise RuntimeError(self.error)
        if "root.dir = direction" in expression:
            self.state = {k: arg[k] for k in DEFAULT}
        if "getBBox" in expression:
            return [dict(self.state, name = f"icon{i}") for i in range(6)]
        return None

    def wait_for_load_state(self, *args, **kwargs):
        self.waits += 1

    def locator(self, selector):
        return types.SimpleNamespace(last = types.SimpleNamespace(wait_for = lambda: None))


def _case_state(measures):
    return [{k: m[k] for k in DEFAULT} for m in measures]


def test_a_case_is_applied_and_measured_in_one_evaluate(mod):
    page = FakePage()
    measures = mod.measure_case(page, CASE)
    assert page.calls == 1
    assert _case_state(measures) == [{k: CASE[k] for k in DEFAULT}] * 6


@pytest.mark.parametrize("reload_on", [1, 2])
def test_a_reload_mid_case_re_measures_the_same_case(mod, reload_on):
    page = FakePage(reloads_on_calls = range(1, reload_on + 1))
    measures = mod.measure_case(page, CASE)
    assert (
        _case_state(measures) == [{k: CASE[k] for k in DEFAULT}] * 6
    ), "a reload reset the display case and the default layout was recorded under its name"
    assert page.waits == reload_on


def test_a_page_that_keeps_navigating_still_fails(mod):
    page = FakePage(reloads_on_calls = range(1, 99))
    with pytest.raises(RuntimeError, match = "Execution context was destroyed"):
        mod.measure_case(page, CASE)
    assert page.calls == mod.CASE_ATTEMPTS


def test_any_other_evaluate_error_fails_at_once(mod):
    page = FakePage(reloads_on_calls = {1}, error = "ReferenceError: getBBox is not defined")
    with pytest.raises(RuntimeError, match = "ReferenceError"):
        mod.measure_case(page, CASE)
    assert page.calls == 1
