# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keystroke timing starts before the handler, so a blocking keydown stall is counted, not subtracted."""

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
    # A session-wide stub playwright.sync_api lacks __file__; importorskip would not catch it.
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
            # goto, not set_content: document.open() removes window listeners, losing the injected stall.
            page.goto(URL)

            quiet = _typed(page, delay_armed = False)
            delayed = _typed(page, delay_armed = True)
        finally:
            browser.close()

    assert quiet["samples"] == CHARS, quiet
    assert delayed["samples"] == CHARS, delayed
    assert delayed["injected_events"] >= CHARS, delayed

    # This gate fails on an input-anchored clock.
    gate = evaluate_input_delay_gate(quiet["p95_ms"], delayed["p95_ms"])
    assert gate.passed, f"{gate.detail}: quiet={quiet}, delayed={delayed}"

    assert delayed.get("unanchored") == 0, delayed
    assert quiet.get("unanchored") == 0, quiet

    assert delayed.get("input_delay_p95_ms") >= INJECTED_INPUT_DELAY_MS * 0.8, delayed
    assert (quiet.get("input_delay_p95_ms") or 0) < 100.0, quiet


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
