# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two #8483 harnesses must read every guard they record.

Both files measure in a browser and then decide pass/fail in `main()`. A metric that is recorded
but never compared is how a harness goes false-green: it keeps reporting the number that would
have caught the regression while exiting 0. Three shipped that way already (an unasserted rAF
budget, an unasserted click count, and the two guards below), so the rule is pinned here rather
than left to review.
"""

import ast
import types

import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDIO_TESTS = ROOT / "tests" / "studio"


def _require_playwright_page():
    """
    Skip unless `from playwright.sync_api import Page` would actually work.

    Two weaker guards were tried and both let this through. Checking the
    top-level package passes because "playwright" resolves as a namespace
    directory on the Repo tests (CPU) runner; checking "playwright.sync_api"
    passes too, because that resolves as a namespace package as well. Only the
    symbol the harnesses import is a real test of whether the import below can
    succeed, so that is what is checked, and it is checked the way the harness
    does it. The failure mode is a skip condition reported as

      ImportError: cannot import name 'Page' from 'playwright.sync_api'
      (unknown location)

    on every branch, which costs an investigation each time it is seen.
    """
    sync_api = pytest.importorskip("playwright.sync_api")
    if not hasattr(sync_api, "Page"):
        pytest.skip(
            "playwright.sync_api resolved from "
            f"{getattr(sync_api, '__file__', None) or list(getattr(sync_api, '__path__', []))} "
            "but has no Page; playwright is not usably installed here"
        )


def source(name: str) -> str:
    return (STUDIO_TESTS / name).read_text(encoding = "utf-8")


def verdict(name: str) -> str:
    """Everything from `def main()` on: the only place a metric turns into an exit code."""
    text = source(name)
    start = text.index("def main()")
    return text[start:]


def test_chat_autoscroll_asserts_the_detached_stream_actually_grew() -> None:
    # `stillDetached` says streaming did not re-pin the reader. It says nothing at all if the streamed tokens added
    # no height, which the harness measures as `grewWhileDetached`.
    main = verdict("playwright_chat_autoscroll.py")
    assert 'intent["stillDetached"]' in main
    assert 'intent["grewWhileDetached"]' in main


def test_research_freeze_asserts_the_dialog_took_the_modal_layer() -> None:
    # The stranded-`pointer-events: none` checks are the reported symptom (an unclickable window). They only test
    # anything if the dialog put the body on the modal layer first.
    main = verdict("playwright_research_freeze.py")
    assert 'modal["body_pointer_events_after_approve"]' in main
    assert 'modal["body_pointer_events_while_open"]' in main


def test_research_freeze_asserts_the_stream_had_something_to_follow() -> None:
    # An empty activity list runs no follow loop, so it clears the frame budgets by measuring nothing. `seed()`
    # leaves activities behind, so the count has to be compared against the pre-stream baseline rather than zero.
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
    # A synthetic element.click() skips hit testing and passes on a stranded `pointer-events: none` tree, so the verdict
    # must read actionability, not just the counter.
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
    # The live log dies with the runner; the screenshot, body excerpt and vite's own transform errors are what remains.
    text = source("playwright_strip_ansi_smoke.py")
    assert "dump(page, vite)" in text, "the assertions do not run under the dump"
    assert "dump_diagnostics(page, ART" in text
    assert 'getattr(vite, "vite_tail"' in text, "vite's output is dropped on failure"


def test_the_ansi_dump_survives_a_vite_server_that_is_still_talking(tmp_path, monkeypatch) -> None:
    # A daemon thread appends to the tail deque for as long as vite lives, and the dump runs before the server stops.
    import importlib.util
    import threading
    from collections import deque

    import pytest

    _require_playwright_page()
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
            # `page` is unused by the tail print and dump_diagnostics is best-effort, so a stub reaches the loop.
            smoke.dump(types.SimpleNamespace(), vite)
    finally:
        stop.set()
        talker.join(timeout = 5)


def test_stream_pacing_asserts_its_long_task_probe_measured_something() -> None:
    # longTaskMs is the metric the budgets turn on, and it is 0 both when the render is free and when the observer never
    # ran: `observe({type: "longtask"})` aborts silently on an engine without the entry type. Without these, a firefox
    # or webkit run, or a broken observer, scores a perfect zero and exits 0.
    main = verdict("playwright_stream_pacing.py")
    assert 'results.get("longTaskSupported")' in main
    assert 'results["longTasks"] <= 0' in main
    # Same for an unthrottled run: the renderer keeps up with any rate this can feed, so every budget passes on any
    # tree.
    assert 'results["cpu_throttle"] <= 1' in main


def test_stream_pacing_asserts_the_reply_was_actually_painted() -> None:
    # A page that rendered nothing scores a perfect zero on every budget, so the workload has to be asserted before the
    # numbers mean anything.
    main = verdict("playwright_stream_pacing.py")
    assert 'results["paintedChars"] < floor' in main
    assert 'results["arrivals"]' in main
    # paintedChars only climbs, so the length at settlement must be checked too, or an empty final DOM passes on the
    # peak it reached earlier.
    assert 'results["settledChars"] < floor' in main


def test_harnesses_own_their_dev_server() -> None:
    # A server started beside the harness leaves the node child alive when the wrapper is killed, stranding the port and
    # the step's stdout. Each harness owns its own instead.
    for name in (
        "playwright_chat_autoscroll.py",
        "playwright_research_freeze.py",
        "playwright_strip_ansi_smoke.py",
        "playwright_thread_weight.py",
        "playwright_stream_pacing.py",
    ):
        text = source(name)
        assert "start_vite" in text, f"{name} does not start its own server"
        assert "stop_process" in text, f"{name} never tears its server down"
        # Vite's SPA fallback answers 200 with index.html for a page that no longer exists.
        assert "wait_for_smoke_page" in text, f"{name} gates on status rather than on content"


# The #8977 thread-weight harness deliberately asserts no timing budget: it exists to produce the numbers a budget
# would later be set from. That removes the `main()` check the tests above rely on, so the same rule is enforced one
# step earlier, every metric it records has to reach the printed table. An unprinted metric there is the same failure
# as an unasserted one here: the number that would have shown the regression is collected and then thrown away.
THREAD_WEIGHT = "playwright_thread_weight.py"


def thread_weight_verdict() -> str:
    """Everything from `def harness_failures(` on. Unlike the harnesses above, this one turns
    metrics into an exit code there rather than in `main()`."""
    text = source(THREAD_WEIGHT)
    return text[text.index("def harness_failures(") :]


def _dict_keys(node: ast.AST, helpers: dict[str, set[str]]) -> set[str]:
    """String keys of a dict literal, following `**helper(...)` into the helper's return dict."""
    keys: set[str] = set()
    if not isinstance(node, ast.Dict):
        return keys
    for key, value in zip(node.keys, node.values):
        if key is None:  # **spread
            call = value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                keys |= helpers.get(call.func.id, set())
            continue
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def _thread_weight_recorded() -> dict[str, set[str]]:
    """{action: recorded metric names} for every `result["action"] = {...}` in the harness."""
    tree = ast.parse(source(THREAD_WEIGHT))
    helpers: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for statement in node.body:
                if isinstance(statement, ast.Return):
                    found = _dict_keys(statement.value, {})
                    if found:
                        helpers[node.name] = found
    recorded: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "result"
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
        ):
            keys = _dict_keys(node.value, helpers)
            if keys:
                recorded[target.slice.value] = keys
    return recorded


def test_thread_weight_records_the_four_actions() -> None:
    # Guards the parser above as much as the harness: an empty result set would make the coverage test below pass by
    # finding nothing to check.
    recorded = _thread_weight_recorded()
    assert {"keystroke", "scroll", "menu", "delete"} <= set(recorded)
    # Not `assert keys`: _thread_weight_recorded only stores an entry when it found keys, so
    # that could never fail. Each action records its own timings plus the five CDP counters and
    # the three long-task fields, so anything under ten means the parser lost a spread.
    for action, keys in recorded.items():
        assert len(keys) >= 10, f"{action} recorded only {len(keys)} metrics: {sorted(keys)}"


def test_thread_weight_row_shapes_stay_distinct() -> None:
    """TABLE_ROWS is (label, pick) and GROWTH_AXES is (label, pick, floored). They read almost
    identically, so a bulk edit to one lands in the other and print_table dies at the end of a
    forty-minute run, after every measurement is taken and before any of it is shown."""
    import importlib
    import sys

    sys.path.insert(0, str(STUDIO_TESTS))
    pytest = importlib.import_module("pytest")
    _require_playwright_page()
    module = importlib.import_module("playwright_thread_weight")
    assert all(len(row) == 2 for row in module.TABLE_ROWS)
    assert all(len(row) == 3 for row in module.GROWTH_AXES)


def test_thread_weight_prints_every_metric_it_records() -> None:
    text = source(THREAD_WEIGHT)
    table = text[text.index("TABLE_ROWS = (") : text.index("def print_table")]
    missing = [
        f'{action}["{metric}"]'
        for action, metrics in _thread_weight_recorded().items()
        for metric in sorted(metrics)
        if f'r["{action}"]["{metric}"]' not in table
    ]
    assert not missing, (
        "recorded but never printed, so the harness would report a curve with these missing: "
        + ", ".join(missing)
    )


def test_thread_weight_proves_it_discriminates_rather_than_gating_on_a_budget() -> None:
    # The one thing this harness must fail on. Without it a run where nothing was ever clicked
    # reports four identical columns and exits 0, which reads as "no regression".
    # Not `in source(...)`: the verdict is a substring of the source, so an `or` against the
    # whole file would make this unfailable.
    # Asserted against the verdict region, not the whole file: a check that only ever appears
    # in a comment or a table row would otherwise satisfy this.
    main = thread_weight_verdict()
    assert "GROWTH_AXES" in main
    assert "no measured axis rose with N" in main
    # The menu is opened non-modally now, so the body must NOT go onto the modal layer; what the verdict has to reject
    # is a run that mixes the two across N, whose columns then price different mechanisms.
    assert 'menu["body_pointer_events_while_open"]' in main
    # An empty popover satisfies "the menu opened" and costs nothing to render.
    assert 'menu["items_while_open"]' in main
    # A delete that deleted nothing is a fast delete.
    assert 'deleted["messages_after"]' in main
    # A keystroke that never reached the runtime still reports the double-rAF paint floor, so the floor has to be
    # compared rather than left implicit in the timings.
    assert 'row["paint_floor_ms"]' in main
    assert 'keystroke["runtime_text"] != keystroke["dom_text"]' in main
    # Requests reaching the network and console warnings both cost a CDP round trip per message, so both would grow with
    # N for reasons the app does not have.
    assert 'row["stray_api_requests"]' in main
    assert 'row["console_warnings"]' in main


def _thread_weight_row(deleted: bool = True, layer: str = "auto") -> dict:
    """One healthy column of the thread-weight table, optionally with a delete that did nothing."""
    return {
        "counts": {
            "messages": 10_000,
            "codeBlocks": 4,
            "katexNodes": 4,
            "actionBars": 3,
            "tooltipTriggers": 8,
        },
        "stray_api_requests": 0,
        "console_warnings": 0,
        "first_console_warning": None,
        "viewport": {"scrollHeight": 9000, "clientHeight": 800, "scrollTop": 0},
        "paint_floor_ms": 33,
        "keystroke": {"median_ms": 120, "runtime_text": "x" * 40, "dom_text": "x" * 40},
        "scroll": {"wall_ms": 100, "scrolled_px": 10_000},
        "menu": {
            "open_ms": 10,
            "close_ms": 10,
            "body_pointer_events_after_close": "auto",
            "body_pointer_events_while_open": layer,
            "items_while_open": 5,
            "triggers_while_hovered": 8,
        },
        "delete": {
            "ms": 5 if deleted else None,
            "messages_before": 10,
            "messages_after": 9 if deleted else 10,
        },
    }


def test_thread_weight_rejects_a_dead_delete_at_every_size() -> None:
    """The delete checks once sat under the whole-run modal-layer `if`, so a table whose delete
    never removed anything still passed while another axis grew. They belong to the per-size
    loop: every measured column has to prove its own delete, not just the last one."""
    import importlib
    import sys

    sys.path.insert(0, str(STUDIO_TESTS))
    pytest = importlib.import_module("pytest")
    _require_playwright_page()
    module = importlib.import_module("playwright_thread_weight")

    sizes = [10, 50]
    for dead in sizes:
        results = {
            "sizes": sizes,
            "by_size": {str(size): _thread_weight_row(deleted = size != dead) for size in sizes},
        }
        assert f"N={dead} never deleted a message" in module.harness_failures(results)
    healthy = {
        "sizes": sizes,
        "by_size": {str(size): _thread_weight_row() for size in sizes},
    }
    assert not [f for f in module.harness_failures(healthy) if "delete" in f]
