# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The "persisted monitor: reset" step of playwright_chat_ui.py must not run script in
the page it is about to discard.

That page is CPU-throttled and its auth was revoked by the CLI rotation just before, so
the app can be busy retrying; page.evaluate waits for its event loop with no timeout of
its own. The step wedged for its whole watchdog budget on Windows and on the Kaggle T4
runner with nothing printed. The storage writes now go through the fresh page."""

import re
from pathlib import Path

DRIVER = Path(__file__).resolve().parent / "playwright_chat_ui.py"
CODE = DRIVER.read_text(encoding = "utf-8")
MARKER = 'step("persisted monitor: reset the browser session and open a fresh page")'


def _step_body() -> str:
    start = CODE.index(MARKER)
    end = CODE.index("\n    step(", start + len(MARKER))
    return CODE[start:end]


def test_the_stale_page_runs_no_script_before_it_is_replaced():
    body = _step_body()
    handoff = body.index("page = _fresh_page")
    before = body[:handoff]
    assert not re.search(
        r"\bpage\.evaluate\(|robust_evaluate\(\s*page\b", before
    ), "the reset step evaluates in the stale page, which can wait on a busy event loop forever"


def test_the_stale_page_is_closed_before_the_fresh_one_writes_storage():
    body = _step_body()
    assert body.index("close_page_without_waiting(ctx, page)") < body.index(
        "new_throttled_page(ctx)"
    ), (
        "close the stale page first, or its app can rewrite the shared localStorage "
        "after the fresh page has set it"
    )
    handoff = body.index("page = _fresh_page")
    writes = body.index("robust_evaluate(", handoff)
    assert "unsloth_monitor_overlay" in body[writes:]
    assert "unsloth_auth_refresh_token" in body[writes:]


def test_the_fresh_page_writes_storage_where_no_app_code_runs():
    body = _step_body()
    goto = re.search(r'page\.goto\(f"\{BASE\}(?P<path>[^"]+)"', body)
    assert goto and goto.group("path").startswith("/api/"), (
        "park the fresh page on an API endpoint; any SPA route boots the app, which reads "
        "the auth tokens this step is clearing"
    )
    assert "timeout" in body[goto.start() : body.index(")", goto.end())]


def test_every_blocking_call_announces_itself_first():
    body = _step_body()
    handoff = body.index("page = _fresh_page")
    calls = {
        "close_page_without_waiting(ctx, page)": body.index(
            "close_page_without_waiting(ctx, page)"
        ),
        "ctx.clear_cookies()": body.index("ctx.clear_cookies()"),
        "new_throttled_page(ctx)": body.index("new_throttled_page(ctx)"),
        "page.goto(": body.index("page.goto("),
        # The storage write is a timeout-free page.evaluate under the wrapper.
        "robust_evaluate(": body.index("robust_evaluate(", handoff),
    }
    for call, at in calls.items():
        # The statement directly before the call (a try: line in between is allowed),
        # not a nearby one: two calls in a row must each name themselves.
        preceding = [
            line.strip()
            for line in body[: body.rfind("\n", 0, at)].splitlines()
            if line.strip() and line.strip() != "try:"
        ]
        assert preceding and preceding[-1].startswith(
            "info("
        ), f"{call} has no breadcrumb directly before it, so a wedge there names the wrong call"


def test_the_stale_page_is_closed_without_waiting_on_its_renderer():
    """page.close() has no timeout and has sat out the step's whole budget on Windows.

    The step closes through close_page_without_waiting (browser-side Target.closeTarget on
    Chromium), and marks the page stale first, so a request it sends while the browser tears
    it down is not counted as the fresh page's /api/system poll.
    """
    body = _step_body()
    handoff = body.index("page = _fresh_page")
    assert "page.close()" not in body[:handoff], "the reset waits on the stale renderer again"
    close = body.index("close_page_without_waiting(ctx, page)")
    assert (
        body.index("stale_pages.add(page)") < close
    ), "mark the page stale before it can still send"
    helper = CODE[CODE.index("def close_page_without_waiting(") :]
    helper = helper[: helper.index("\ndef ")]
    assert '"Target.closeTarget"' in helper and "new_browser_cdp_session()" in helper
    assert (
        'PLAYWRIGHT_BROWSER == "chromium"' in helper
    ), "CDP is Chromium only; other engines fall back"


def test_the_system_request_count_skips_a_stale_page():
    counter = CODE[CODE.index("def _count_system_request(") :]
    counter = counter[: counter.index('ctx.on("request", _count_system_request)')]
    assert (
        "in stale_pages" in counter
    ), "requests from a page closed without waiting would read as the monitor polling"
