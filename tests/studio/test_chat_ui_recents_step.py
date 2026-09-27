# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat UI driver's Recents step must wait for the thread, and must be able to fail.

The step used to click the first Recents row, sleep 500 ms and read the turns once. Opening a
chat runs a history loader that awaits four requests in a row before anything renders, and on
the Windows lane (two Studios and two browsers on one runner) that took longer than 500 ms in
about one run in four, so the read found `turns_text=''`. That never failed a job, because the
soft_fail sat inside the `try` whose `except Exception` was meant for click errors: in STRICT
mode the AssertionError was caught, logged as "recent-thread click 0 failed", and the step
passed. The claim was being logged, not checked.

The browser tests run the driver's helper, extracted by `ast` so nothing else in the driver
executes, against a page whose thread renders late. They skip when Chromium is unavailable;
the source checks below always run.
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
DRIVER_PATH = HERE / "playwright_chat_ui.py"
DRIVER = DRIVER_PATH.read_text(encoding = "utf-8")
TREE = ast.parse(DRIVER)
HELPER = "open_recent_thread_with_our_prompts"


def _helper_def() -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == HELPER:
            return node
    raise AssertionError(f"{HELPER} is no longer a top-level function of the driver")


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == name)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == name)
        )
    ]


def test_the_recents_step_goes_through_the_helper() -> None:
    step_calls = [
        c
        for c in _calls(TREE, "step")
        if c.args and isinstance(c.args[0], ast.Constant) and "Recents" in str(c.args[0].value)
    ]
    assert step_calls, "the Recents step is gone from the driver"
    assert _calls(TREE, HELPER), f"the driver no longer calls {HELPER}"


def test_the_helper_waits_for_the_turns_instead_of_sleeping() -> None:
    helper = _helper_def()
    assert not _calls(
        helper, "wait_for_timeout"
    ), "a fixed sleep before reading the thread is the race this helper replaced"
    assert _calls(helper, "wait_for_function"), "the helper must poll for the turns"


def test_no_soft_fail_in_the_helper_is_swallowed_by_a_broad_except() -> None:
    """A soft_fail raises in STRICT mode, so one inside `try: ... except Exception` is a no-op."""
    helper = _helper_def()
    for node in ast.walk(helper):
        if not isinstance(node, ast.Try):
            continue
        broad = any(
            h.type is None
            or (isinstance(h.type, ast.Name) and h.type.id in ("Exception", "BaseException"))
            for h in node.handlers
        )
        if not broad:
            continue
        for stmt in node.body:
            assert not _calls(stmt, "soft_fail") and not _calls(stmt, "fail"), (
                f"line {stmt.lineno}: a failure raised here is caught by the except at line "
                f"{node.handlers[0].lineno}, so the step cannot fail"
            )


# --- behaviour, in a real browser ------------------------------------------------------------

ORIGIN = "http://recents.test"

PAGE = """<!doctype html><html><body>
<nav id="recents"></nav><main id="thread"></main>
<script>
const threads = %(threads)s;
// Studio keeps one runtime while the next thread loads, so the previous turns stay on screen.
const keepStale = %(keep_stale)s;
const nav = document.getElementById("recents");
for (const t of threads) {
  const b = document.createElement("button");
  b.dataset.testid = "recent-thread";
  b.dataset.threadType = "single";
  b.dataset.threadId = t.id;
  b.textContent = t.title;
  b.onclick = () => {
    history.pushState(null, "", "/chat?thread=" + t.id);
    const main = document.getElementById("thread");
    if (!keepStale) main.innerHTML = "";
    // The history loader: nothing renders until its requests have all come back.
    setTimeout(() => {
      main.innerHTML = "";
      for (const [role, text] of t.turns) {
        const d = document.createElement("div");
        d.dataset.role = role;
        d.innerText = text;
        main.appendChild(d);
      }
    }, t.loadMs);
  };
  nav.appendChild(b);
}
</script></body></html>"""


def _load_helper(timeout_ms: int | None = None):
    sys.path.insert(0, str(HERE))
    from _playwright_robust import robust_evaluate

    keep = [
        n
        for n in TREE.body
        if (isinstance(n, ast.FunctionDef) and n.name == HELPER)
        or (
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "RECENTS_LOAD_TIMEOUT_MS" for t in n.targets)
        )
    ]
    assert len(keep) == 2, "the helper or its timeout constant moved"
    infos: list[str] = []

    def soft_fail(message: str) -> None:  # STRICT, as in CI
        raise AssertionError(f"[ui] FAIL: {message}")

    scope = {
        "time": time,
        "robust_evaluate": robust_evaluate,
        "info": infos.append,
        "soft_fail": soft_fail,
    }
    exec(compile(ast.Module(body = keep, type_ignores = []), str(DRIVER_PATH), "exec"), scope)
    if timeout_ms is not None:
        scope["RECENTS_LOAD_TIMEOUT_MS"] = timeout_ms
    return scope[HELPER], infos


@pytest.fixture(scope = "module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    # Without Playwright, test_heavy_thread_measurement_integrity.py leaves a stand-in
    # playwright.sync_api in sys.modules whose every name raises RuntimeError. Same answer.
    try:
        manager = sync_api.sync_playwright()
    except RuntimeError as exc:
        pytest.skip(f"playwright unavailable: {exc}")
    with manager as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # no browser build installed here
            pytest.skip(f"chromium unavailable: {exc}")
        yield b
        b.close()


def _open(
    browser,
    threads,
    keep_stale = False,
):
    import json

    ctx = browser.new_context()
    html = PAGE % {"threads": json.dumps(threads), "keep_stale": json.dumps(keep_stale)}
    ctx.route(f"{ORIGIN}/**", lambda route: route.fulfill(content_type = "text/html", body = html))
    page = ctx.new_page()
    page.goto(f"{ORIGIN}/chat")
    return ctx, page


OURS = {
    "id": "t-ours",
    "title": "Reply with exactly: rapid-first",
    "turns": [["user", "Reply with exactly: rapid-first"], ["assistant", "rapid-first"]],
}
SENT = ["Reply with exactly: rapid-first", "Reply with exactly: hello"]


def test_a_thread_that_renders_after_500ms_passes(browser) -> None:
    """The observed case: the old fixed 500 ms read reported this thread as empty."""
    helper, infos = _load_helper()
    ctx, page = _open(browser, [{**OURS, "loadMs": 1500}])
    try:
        helper(page, SENT, lambda name: None)
    finally:
        ctx.close()
    assert any("OK landed on a thread that includes our prompts" in m for m in infos), infos


def test_a_thread_that_never_shows_our_prompts_fails(browser) -> None:
    helper, _ = _load_helper(timeout_ms = 1500)
    ctx, page = _open(browser, [{**OURS, "loadMs": 50, "turns": [["user", "something else"]]}])
    try:
        with pytest.raises(AssertionError, match = "doesn't contain any of our sent prompts"):
            helper(page, SENT, lambda name: None)
    finally:
        ctx.close()


def test_the_entry_is_chosen_by_title_not_position(browser) -> None:
    """A newer, unrelated chat at the top of Recents must not be the one checked."""
    helper, infos = _load_helper(timeout_ms = 3000)
    other = {"id": "t-other", "title": "New Chat", "turns": [], "loadMs": 0}
    ctx, page = _open(browser, [other, {**OURS, "loadMs": 200}])
    try:
        helper(page, SENT, lambda name: None)
        assert "thread=t-ours" in page.url
    finally:
        ctx.close()
    assert any("Reply with exactly: rapid-first" in m for m in infos), infos


def test_auto_titled_chats_are_searched_past_an_unrelated_newer_one(browser) -> None:
    """No title matches when chats are auto-titled, so the newest row is tried first. One that
    loads someone else's turns is not ours, and the next row is tried instead of failing."""
    helper, infos = _load_helper(timeout_ms = 3000)
    other = {
        "id": "t-other",
        "title": "Weather chat",
        "turns": [["user", "What's the weather?"]],
        "loadMs": 50,
    }
    ctx, page = _open(browser, [other, {**OURS, "title": "Rapid replies", "loadMs": 300}])
    try:
        helper(page, SENT, lambda name: None)
        assert "thread=t-ours" in page.url
    finally:
        ctx.close()
    assert any("not this run's chat" in m for m in infos), infos


def test_auto_titled_chats_are_searched_past_an_empty_newer_one(browser) -> None:
    helper, _ = _load_helper(timeout_ms = 1000)
    empty = {"id": "t-empty", "title": "New Chat", "turns": [], "loadMs": 0}
    ctx, page = _open(browser, [empty, {**OURS, "title": "Rapid replies", "loadMs": 200}])
    try:
        helper(page, SENT, lambda name: None)
        assert "thread=t-ours" in page.url
    finally:
        ctx.close()


def test_a_chat_titled_with_our_prompt_must_show_our_turns(browser) -> None:
    """The title says it is ours, so turns that are not ours fail rather than move on."""
    helper, _ = _load_helper(timeout_ms = 1000)
    wrong = {**OURS, "id": "t-wrong", "loadMs": 50, "turns": [["user", "something else"]]}
    later = {**OURS, "id": "t-later", "title": "Rapid replies", "loadMs": 50}
    ctx, page = _open(browser, [wrong, later])
    try:
        with pytest.raises(AssertionError, match = "doesn't contain any of our sent prompts"):
            helper(page, SENT, lambda name: None)
        assert "thread=t-wrong" in page.url
    finally:
        ctx.close()


def test_the_previous_chats_turns_left_on_screen_are_not_read_as_the_next_one(browser) -> None:
    """The URL changes at once but the previous thread's turns stay until the next one loads, so
    a candidate must not be judged "someone else's" from what the last candidate left behind."""
    helper, _ = _load_helper(timeout_ms = 3000)
    other = {
        "id": "t-other",
        "title": "Weather chat",
        "turns": [["user", "What's the weather?"]],
        "loadMs": 50,
    }
    ctx, page = _open(
        browser, [other, {**OURS, "title": "Rapid replies", "loadMs": 800}], keep_stale = True
    )
    try:
        helper(page, SENT, lambda name: None)
        assert "thread=t-ours" in page.url
    finally:
        ctx.close()


def test_our_turns_left_on_screen_do_not_pass_a_different_chat(browser) -> None:
    """The mirror case: the page shows our thread, a different chat is clicked, and our turns
    are still on screen while it loads. That is not our chat, and the next row must be ours."""
    helper, _ = _load_helper(timeout_ms = 3000)
    other = {
        "id": "t-other",
        "title": "Weather chat",
        "turns": [["user", "What's the weather?"]],
        "loadMs": 800,
    }
    ctx, page = _open(
        browser, [other, {**OURS, "title": "Rapid replies", "loadMs": 50}], keep_stale = True
    )
    try:
        # Our thread is already on screen before the step runs.
        page.locator('[data-thread-id="t-ours"]').click()
        page.wait_for_selector('[data-role="user"]')
        helper(page, SENT, lambda name: None)
        assert "thread=t-ours" in page.url
    finally:
        ctx.close()


def test_this_runs_chat_is_opened_by_id_over_an_earlier_runs_same_titled_chat(browser) -> None:
    """The prompts are fixed, so a reused Studio home can hold an earlier run's chat with the
    same title and turns. Given this run's id, that one is not taken for ours."""
    helper, _ = _load_helper(timeout_ms = 3000)
    earlier = {**OURS, "id": "t-earlier", "loadMs": 50}
    ctx, page = _open(browser, [earlier, {**OURS, "title": "Rapid replies", "loadMs": 200}])
    try:
        helper(page, SENT, lambda name: None, our_thread_id = "t-ours")
        assert "thread=t-ours" in page.url
    finally:
        ctx.close()


def test_a_known_id_missing_from_recents_fails(browser) -> None:
    helper, _ = _load_helper(timeout_ms = 1000)
    ctx, page = _open(browser, [{**OURS, "loadMs": 50}])
    try:
        with pytest.raises(AssertionError, match = "is not in the sidebar"):
            helper(page, SENT, lambda name: None, our_thread_id = "t-gone")
    finally:
        ctx.close()


def _capture_js() -> str:
    found = [
        n.value
        for n in ast.walk(TREE)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and '[data-testid="recent-thread"][data-active="true"]' in n.value
    ]
    assert len(found) == 1, "the driver no longer captures this run's chat id"
    return found[0]


def test_the_driver_reads_this_runs_chat_id_from_the_url_or_the_active_row(browser) -> None:
    """The capture runs in the real driver only, so evaluate its exact source here."""
    js = _capture_js()
    rows = (
        '<button data-testid="recent-thread" data-active="false" data-thread-id="t-a"></button>'
        '<button data-testid="recent-thread" data-active="true" data-thread-id="t-b"></button>'
    )
    ctx = browser.new_context()
    ctx.route(f"{ORIGIN}/**", lambda route: route.fulfill(content_type = "text/html", body = rows))
    page = ctx.new_page()
    try:
        page.goto(f"{ORIGIN}/chat")
        assert page.evaluate(js) == "t-b"
        page.goto(f"{ORIGIN}/chat?thread=t-url")
        assert page.evaluate(js) == "t-url"
    finally:
        ctx.close()


def test_a_known_id_is_found_past_the_first_rows(browser) -> None:
    """Pinned and project chats share the row testid, so ours can sit well down the sidebar."""
    helper, _ = _load_helper(timeout_ms = 3000)
    pinned = [
        {"id": f"t-pin{k}", "title": f"Pinned {k}", "turns": [["user", f"pinned {k}"]], "loadMs": 0}
        for k in range(25)
    ]
    ctx, page = _open(browser, [*pinned, {**OURS, "title": "Rapid replies", "loadMs": 100}])
    try:
        helper(page, SENT, lambda name: None, our_thread_id = "t-ours")
        assert "thread=t-ours" in page.url
    finally:
        ctx.close()
