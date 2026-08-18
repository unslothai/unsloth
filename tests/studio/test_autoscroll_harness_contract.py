# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two #8483 harnesses must read every guard they record.

Both files measure in a browser and then decide pass/fail in `main()`. A metric that is recorded
but never compared is how a harness goes false-green: it keeps reporting the number that would
have caught the regression while exiting 0. Three shipped that way already (an unasserted rAF
budget, an unasserted click count, and the two guards below), so the rule is pinned here rather
than left to review.
"""

import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDIO_TESTS = ROOT / "tests" / "studio"


def source(name: str) -> str:
    return (STUDIO_TESTS / name).read_text(encoding = "utf-8")


def verdict(name: str) -> str:
    """Everything from `def main()` on: the only place a metric turns into an exit code."""
    text = source(name)
    start = text.index("def main()")
    return text[start:]


def test_chat_autoscroll_asserts_the_detached_stream_actually_grew() -> None:
    # `stillDetached` says streaming did not re-pin the reader. It says nothing at all if the
    # streamed tokens added no height, which the harness measures as `grewWhileDetached`.
    main = verdict("playwright_chat_autoscroll.py")
    assert 'intent["stillDetached"]' in main
    assert 'intent["grewWhileDetached"]' in main


def test_research_freeze_asserts_the_dialog_took_the_modal_layer() -> None:
    # The stranded-`pointer-events: none` checks are the reported symptom (an unclickable
    # window). They only test anything if the dialog put the body on the modal layer first.
    main = verdict("playwright_research_freeze.py")
    assert 'modal["body_pointer_events_after_approve"]' in main
    assert 'modal["body_pointer_events_while_open"]' in main


def test_research_freeze_asserts_the_stream_had_something_to_follow() -> None:
    # An empty activity list runs no follow loop, so it clears the frame budgets by measuring
    # nothing. `seed()` leaves activities behind, so the count has to be compared against the
    # pre-stream baseline rather than against zero.
    main = verdict("playwright_research_freeze.py")
    assert 'stream["raf_per_second"]' in main
    assert 'stream["activities"]' in main
    assert 'stream["activities_before"]' in main


def test_research_freeze_asserts_the_report_stall_and_its_own_probe() -> None:
    # The stall budget is only a budget if a stall of zero fails too: no second sample means
    # the probe measured nothing and the comparison below it passes on any tree.
    main = verdict("playwright_research_freeze.py")
    assert 'results["report"]["main_thread_stall_ms"] > MAIN_THREAD_STALL_BUDGET_MS' in main
    assert 'results["report"]["main_thread_stall_ms"] <= 0' in main


def test_research_freeze_keeps_a_hit_tested_click_in_the_report_phase() -> None:
    # A synthetic element.click() skips hit testing and passes on a stranded
    # `pointer-events: none` tree, so the verdict must read actionability, not just the counter.
    source_text = source("playwright_research_freeze.py")
    assert "page.click('[data-smoke=\"click-probe\"]'" in source_text
    main = verdict("playwright_research_freeze.py")
    assert 'results["report"]["click_landed"]' in main
    assert 'results["report"]["clicks_registered"]' in main


def test_harnesses_report_why_the_page_failed() -> None:
    # A thrown entry module and a merely slow one both end as a timeout on a locator that
    # was never created. Run 31935573269 was that: 15s of nothing, no console, no page
    # error, no server output, on 7 of the 8 runs that reached this step.
    for name in (
        "playwright_chat_autoscroll.py",
        "playwright_research_freeze.py",
        "playwright_strip_ansi_smoke.py",
    ):
        assert "echo_browser_errors(page, info)" in source(
            name
        ), f"{name} discards pageerror and console.error, so a crashed page reads as a timeout"


def test_ansi_smoke_keeps_the_failed_page_and_the_server_output() -> None:
    # The live log dies with the runner; the screenshot, body excerpt and vite's own
    # transform errors are what remains.
    text = source("playwright_strip_ansi_smoke.py")
    assert "dump(page, vite)" in text, "the assertions do not run under the dump"
    assert "dump_diagnostics(page, ART" in text
    assert 'getattr(vite, "vite_tail"' in text, "vite's output is dropped on failure"


def test_the_ansi_dump_survives_a_vite_server_that_is_still_talking(tmp_path, monkeypatch) -> None:
    # A daemon thread appends to the tail deque for as long as vite lives, and the dump
    # runs before the server stops. Iterating it live while printing (stdout releases the
    # GIL) raises "deque mutated during iteration", losing the tail in the one case it was
    # added for: a reload or transform storm.
    import importlib.util
    import threading
    from collections import deque

    import pytest

    pytest.importorskip("playwright")
    monkeypatch.setenv("PW_ART_DIR", str(tmp_path / "art"))
    spec = importlib.util.spec_from_file_location(
        "_ansi_smoke_under_test", STUDIO_TESTS / "playwright_strip_ansi_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    tail: deque[str] = deque(maxlen = 20)
    for index in range(tail.maxlen or 20):
        tail.append(f"vite line {index}")
    vite = types.SimpleNamespace(vite_tail = tail)
    stop = threading.Event()

    def keep_talking() -> None:
        index = 0
        while not stop.is_set():
            tail.append(f"[vite] page reload {index}")
            index += 1

    talker = threading.Thread(target = keep_talking, daemon = True)
    talker.start()
    try:
        for _ in range(5):
            # `page` is unused by the tail print and dump_diagnostics is best-effort,
            # so a stub reaches the loop.
            smoke.dump(types.SimpleNamespace(), vite)
    finally:
        stop.set()
        talker.join(timeout = 5)


def test_harnesses_own_their_dev_server() -> None:
    # A server started beside the harness leaves the node child alive when the wrapper is
    # killed, stranding the port and the step's stdout. Each harness owns its own instead.
    for name in (
        "playwright_chat_autoscroll.py",
        "playwright_research_freeze.py",
        "playwright_strip_ansi_smoke.py",
    ):
        text = source(name)
        assert "start_vite" in text, f"{name} does not start its own server"
        assert "stop_process" in text, f"{name} never tears its server down"
        # Vite's SPA fallback answers 200 with index.html for a page that no longer exists.
        assert "wait_for_smoke_page" in text, f"{name} gates on status rather than on content"
