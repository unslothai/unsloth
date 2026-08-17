# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`hover_semantics_probe.py` gates CI, so its verdict has to be able to say no.

The probe drives a browser and then turns nine cases into an exit code in `failures_in()`. Every
way that function can return an empty list while something was wrong is a way the gate goes green
on a broken tree, and three separate review rounds found one: a case that raised carried no break
key, three cases had no break key at all, and a case that never ran scored the same as a case that
ran and passed.

None of that needs a browser to test. `failures_in()` is a pure function over the rows the cases
return, so it is exercised here against rows written by hand, including the shapes a real run
produces when the page will not cooperate.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STUDIO_TESTS = ROOT / "tests" / "studio"
PROBE = "hover_semantics_probe.py"

# What a case returns when it could not run at all, and when it ran but could not drive the page.
SKIPPED = {"skipped": "no user message in view"}
INCONCLUSIVE = {"inconclusive": True}


def load_probe(artifact_dir, env: dict | None = None):
    """Import the probe for real, under a controlled environment.

    It pulls in `playwright.sync_api` and creates its artifact directory at import time, so the
    browser binding is stubbed when absent and `PW_ART_DIR` is pointed at a temp dir. Every
    change is undone before this returns: a stub left in `sys.modules` would be handed to
    test_playwright_server_lifecycle.py, which runs in the same pytest process and imports the
    real one.
    """
    stubbed: list[str] = []
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        package = types.ModuleType("playwright")
        binding = types.ModuleType("playwright.sync_api")
        binding.sync_playwright = None
        package.sync_api = binding
        for name, stub in (("playwright", package), ("playwright.sync_api", binding)):
            if name not in sys.modules:
                sys.modules[name] = stub
                stubbed.append(name)
    overrides = {"PW_ART_DIR": str(artifact_dir), **(env or {})}
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        spec = importlib.util.spec_from_file_location("_probe_under_test", STUDIO_TESTS / PROBE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope = "module")
def probe(tmp_path_factory):
    return load_probe(tmp_path_factory.mktemp("hover_probe_artifacts"))


def gated_case(probe) -> tuple[str, str]:
    """A case that carries a break key, and that key. Read off the probe rather than hard-coded,
    so this file does not have to be edited every time a case is renamed."""
    name = next(iter(probe.BREAK_KEYS))
    return name, probe.BREAK_KEYS[name][0]


def test_every_case_is_gated_or_declared_observational(probe) -> None:
    # A case with no break key can only ever report success, whatever it saw. The probe asserts
    # this at import, so reaching here at all is most of the check; the rest states the intent.
    assert set(probe.CASES) == set(probe.BREAK_KEYS) | probe.OBSERVATIONAL
    assert not set(probe.BREAK_KEYS) & probe.OBSERVATIONAL


def test_a_break_is_reported(probe) -> None:
    name, key = gated_case(probe)
    assert probe.failures_in("chromium", name, [{key: True}])


def test_a_clean_run_is_not_reported(probe) -> None:
    # The control. Without it every assertion below would hold on a verdict that failed all rows.
    name, key = gated_case(probe)
    assert probe.failures_in("chromium", name, [{key: False}, {key: False}]) == []


def test_a_case_that_raised_is_reported(probe) -> None:
    # A raised case carries no break key, so before this it scored as a pass and the gate went
    # green precisely when the probe stopped working.
    name, _ = gated_case(probe)
    assert probe.failures_in("chromium", name, [{"failed": "TimeoutError"}])


def test_a_gated_case_that_never_ran_is_reported(probe) -> None:
    # The false green this is really about: a fixture or rendering regression that leaves no user
    # message in view makes every case return `skipped`, so nothing is asserted and the strict
    # gate still exits 0.
    name, _ = gated_case(probe)
    assert probe.failures_in("chromium", name, [SKIPPED, SKIPPED])
    assert probe.failures_in("chromium", name, [INCONCLUSIVE, INCONCLUSIVE])


def test_one_unusable_repetition_is_tolerated(probe) -> None:
    # Deliberately not "any skipped repetition fails". Several cases flag a page that would not
    # scroll or drag on that attempt, which is per-attempt noise by design, and at PROBE_REPS=2
    # the stricter reading turns one unlucky attempt into a red build.
    name, key = gated_case(probe)
    assert probe.failures_in("chromium", name, [SKIPPED, {key: False}]) == []
    assert probe.failures_in("chromium", name, [INCONCLUSIVE, {key: False}]) == []


def test_a_break_still_counts_when_another_repetition_was_skipped(probe) -> None:
    name, key = gated_case(probe)
    assert probe.failures_in("chromium", name, [SKIPPED, {key: True}])


def test_an_observational_case_that_never_ran_is_not_reported(probe) -> None:
    # It has nothing to assert, so it cannot fail to assert it.
    for name in probe.OBSERVATIONAL:
        assert probe.failures_in("chromium", name, [SKIPPED, SKIPPED]) == []


def test_an_unknown_case_filter_is_rejected(tmp_path) -> None:
    # A name that matches nothing selects nothing, and a run with no cases has nothing to report,
    # so one typo in PROBE_CASES exits 0 under PROBE_STRICT having asserted precisely nothing. A
    # targeted run is exactly where that reads as "the case I was chasing is fixed".
    with pytest.raises(SystemExit) as caught:
        load_probe(tmp_path / "typo", {"PROBE_CASES": "h3_contnuous"})
    assert "h3_contnuous" in str(caught.value)
    # The message has to be usable, so it names what IS available.
    assert "h3_continuous" in str(caught.value)


def test_a_known_case_filter_is_accepted(tmp_path) -> None:
    # The control: rejecting every filter would satisfy the test above just as well.
    module = load_probe(tmp_path / "ok", {"PROBE_CASES": "h3_continuous,h5_keyboard"})
    assert module.ONLY == ["h3_continuous", "h5_keyboard"]


def test_strict_is_what_turns_a_break_into_an_exit_code(probe) -> None:
    # The gate is only a gate under PROBE_STRICT; without it the file reports and exits 0.
    text = (STUDIO_TESTS / PROBE).read_text(encoding = "utf-8")
    assert "if STRICT:\n        return 1 if broken else 0" in text
