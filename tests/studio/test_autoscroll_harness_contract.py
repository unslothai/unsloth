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


def test_harnesses_own_their_dev_server() -> None:
    # A server started beside the harness leaves the node child alive when the wrapper is
    # killed, stranding the port and the step's stdout. Each harness owns its own instead.
    for name in (
        "playwright_chat_autoscroll.py",
        "playwright_research_freeze.py",
        "playwright_strip_ansi_smoke.py",
        "playwright_thread_weight.py",
    ):
        text = source(name)
        assert "start_vite" in text, f"{name} does not start its own server"
        assert "stop_process" in text, f"{name} never tears its server down"
        # Vite's SPA fallback answers 200 with index.html for a page that no longer exists.
        assert "wait_for_smoke_page" in text, f"{name} gates on status rather than on content"


# The #8977 thread-weight harness deliberately asserts no timing budget: it exists to produce the
# numbers a budget would later be set from. That removes the `main()` check the tests above rely
# on, so the same rule is enforced one step earlier -- every metric it records has to reach the
# printed table. An unprinted metric there is the same failure as an unasserted one here: the
# number that would have shown the regression is collected and then thrown away.

THREAD_WEIGHT = "playwright_thread_weight.py"


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
    # Guards the parser above as much as the harness: an empty result set would make the
    # coverage test below pass by finding nothing to check.
    recorded = _thread_weight_recorded()
    assert {"keystroke", "scroll", "menu", "delete"} <= set(recorded)
    for action, keys in recorded.items():
        assert keys, f"{action} recorded no metrics"


def test_thread_weight_prints_every_metric_it_records() -> None:
    text = source(THREAD_WEIGHT)
    table = text[text.index("TABLE_ROWS = ("):text.index("def print_table")]
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
    main = verdict(THREAD_WEIGHT)
    assert "GROWTH_AXES" in main or "GROWTH_AXES" in source(THREAD_WEIGHT)
    assert "no measured axis rose with N" in source(THREAD_WEIGHT)
    # The menu cost is only the reported one if the body really went onto the modal layer.
    assert 'menu["body_pointer_events_while_open"]' in source(THREAD_WEIGHT)
    # A delete that deleted nothing is a fast delete.
    assert 'deleted["messages_after"]' in source(THREAD_WEIGHT)
