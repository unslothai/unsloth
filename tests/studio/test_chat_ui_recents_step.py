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
    main.innerHTML = "";
    // The history loader: nothing renders until its requests have all come back.
    setTimeout(() => {
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
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # no browser build installed here
            pytest.skip(f"chromium unavailable: {exc}")
        yield b
        b.close()


def _open(browser, threads):
    import json

    ctx = browser.new_context()
    html = PAGE % {"threads": json.dumps(threads)}
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
