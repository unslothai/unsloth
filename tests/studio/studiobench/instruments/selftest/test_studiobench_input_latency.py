# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The keystroke instrument measures keystroke-to-paint, including the wait before the handler.

THE GATE THIS ENFORCES IS THE HARNESS'S OWN. `instruments/selfcheck.py` injects a 400 ms blocking
`keydown` listener and requires that it move keystroke p95 by at least 350 ms. `input.js` used to
start its clock inside the `input` handler, which is dispatched as the default action of that same
keydown -- so the stall had already finished, the wait was subtracted out of every sample, and the
measured p95 moved by -14.8 ms while the user waited 400 ms. The highest-weight metric in the
scoring table read clean exactly when typing was at its worst.

This drives a real engine because the defect only exists in a real input pipeline: a synthetic
`input` event dispatched from a script cannot show the queueing delay at all, which is the reason
the driver types with `page.keyboard` in the first place.

Skips cleanly with no engine installed, because the selftest job runs on a machine with none.

    python -m pytest tests/studio/studiobench/instruments/selftest/test_studiobench_input_latency.py
"""

from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.instruments.selfcheck import (  # noqa: E402
    INJECTED_INPUT_DELAY_MS,
    evaluate_input_delay_gate,
    input_delay_init_script,
)
from studiobench.runtime import resources  # noqa: E402

PAGE = (
    "<!doctype html><meta charset=utf-8><body>"
    "<textarea aria-label='Message input' rows='4'></textarea></body>"
)
URL = "data:text/html," + urllib.parse.quote(PAGE)
SELECTOR = "textarea[aria-label='Message input']"
CHARS = 12


def _engine(pw):
    """Chromium if it is downloaded, else whichever engine is. `None` means skip."""
    for name in ("chromium", "webkit", "firefox"):
        try:
            if Path(getattr(pw, name).executable_path).exists():
                return name
        except Exception:  # noqa: BLE001
            continue
    return None


def _typed(page, *, delay_armed: bool) -> dict:
    page.evaluate(
        "(on) => (on ? window.__sbInputDelay.arm() : window.__sbInputDelay.disarm())", delay_armed
    )
    page.fill(SELECTOR, "")
    page.click(SELECTOR)
    page.evaluate("(s) => window.__sb.input.arm(s)", SELECTOR)
    page.keyboard.type("a" * CHARS, delay = 60)
    page.wait_for_timeout(1000)
    got = page.evaluate("(n) => window.__sb.input.collect(n)", CHARS)
    got["injected_events"] = page.evaluate("() => window.__sbInputDelay.events")
    return got


def test_an_injected_keydown_stall_moves_keystroke_p95():
    playwright = pytest.importorskip("playwright.sync_api", reason = "playwright is not installed")
    # IMPORTORSKIP IS NOT ENOUGH IN THIS SUITE. tests/studio/test_heavy_thread_measurement_integrity.py
    # puts a stub `playwright.sync_api` into `sys.modules` at collection time and it stays there for
    # the rest of the session, so on the CPU job the import above SUCCEEDS against the stub and every
    # name raises RuntimeError when called. The stub is a bare ModuleType with no `__file__`, which the
    # real package always has.
    if getattr(playwright, "__file__", None) is None:
        pytest.skip("playwright.sync_api is the CPU-job stub, so there is no browser to drive")

    with playwright.sync_playwright() as pw:
        name = _engine(pw)
        if name is None:
            pytest.skip("no Playwright engine is downloaded on this machine")
        browser = getattr(pw, name).launch()
        try:
            context = browser.new_context()
            context.add_init_script(input_delay_init_script())
            context.add_init_script(resources.read_text("instruments/input.js"))
            page = context.new_page()
            # `goto`, never `set_content`: `document.open()` removes every listener registered on the window, so
            # a page built that way silently discards the injected stall and the test would pass against a
            # broken instrument.
            page.goto(URL)

            quiet = _typed(page, delay_armed = False)
            delayed = _typed(page, delay_armed = True)
        finally:
            browser.close()

    assert quiet["samples"] == CHARS, quiet
    assert delayed["samples"] == CHARS, delayed
    assert delayed["injected_events"] >= CHARS, delayed

    # THE GATE the harness would run on these two readings, applied to the readings the instrument
    # actually produces. This is the assertion that fails on an input-anchored clock.
    gate = evaluate_input_delay_gate(quiet["p95_ms"], delayed["p95_ms"])
    assert gate.passed, f"{gate.detail}: quiet={quiet}, delayed={delayed}"

    # Anchored on the key event, not on the input handler, or the wait cannot be seen at all.
    assert delayed.get("unanchored") == 0, delayed
    assert quiet.get("unanchored") == 0, quiet

    # The instrument's own account of the two halves.
    assert delayed.get("input_delay_p95_ms") >= INJECTED_INPUT_DELAY_MS * 0.8, delayed
    assert (quiet.get("input_delay_p95_ms") or 0) < 100.0, quiet


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
