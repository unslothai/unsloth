# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Writes a sentinel before Control+C; if it survives, nothing was copied and no timing is reported."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.runtime.types import ActionContext, Cell  # noqa: E402
from studiobench.scene import actions as A  # noqa: E402

_DOM_JS = _STUDIO_TESTS / "studiobench" / "scene" / "dom.js"

# navigator.clipboard needs a secure context; about:blank is not one.
ORIGIN = "https://studiobench.localhost"

BODY = """<!doctype html><meta charset=utf-8><title>t</title>
<style>.aui-thread-viewport { height: 300px; overflow-y: auto; }</style>
<div class="aui-thread-root"><div class="aui-thread-viewport" id="vp"></div></div>
<script>
  const vp = document.getElementById("vp");
  for (let i = 1; i <= 6; i++) {
    const m = document.createElement("div");
    m.setAttribute("data-role", i % 2 ? "user" : "assistant");
    m.textContent = "message " + i + " " + "x".repeat(400);
    vp.appendChild(m);
  }
  window.__sbNextPaint = () => new Promise(
    (r) => requestAnimationFrame(() => requestAnimationFrame(r)));
</script>
"""


def _skip_reason() -> str | None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"playwright is not installed: {exc}"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason = _skip_reason() or "")


@pytest.fixture(scope = "module")
def context():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args = ["--no-sandbox"])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chromium could not be launched: {exc}")
        ctx = b.new_context(permissions = ["clipboard-read", "clipboard-write"])
        ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin = ORIGIN)
        yield ctx
        ctx.close()
        b.close()


@pytest.fixture()
def page(context):
    pg = context.new_page()
    pg.route(
        "**/*",
        lambda route: route.fulfill(status = 200, content_type = "text/html; charset=utf-8", body = BODY),
    )
    pg.goto(ORIGIN + "/chat")
    pg.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    yield pg
    pg.close()


def _ctx(page, log = None) -> ActionContext:
    return ActionContext(
        page = page,
        cdp = None,
        cell = Cell(cell_id = "r100K.base.rep0", rung = "100K", rung_tokens = 100_000),
        window = None,
        args = {"thread_id": "t1", "base_url": ORIGIN},
        budget_ms = 30_000,
        dom = None,
        log = log or (lambda _m: None),
    )


def _swallow_the_copy(page) -> None:
    """An engine that does not copy, reproduced faithfully: the keystroke is delivered, a `copy`
    event fires, and nothing reaches the clipboard."""
    page.evaluate("() => document.addEventListener('copy', (e) => { e.preventDefault(); }, true)")


def test_a_real_copy_is_still_measured(page):
    """THE CONTROL. Without it the guard could refuse everything and the tests below would pass on
    an action that never reports a timing at all."""
    got = A.select_all_copy(_ctx(page))
    assert got.ran is True, got.reason
    assert got.timings.get("copy_ms") is not None
    assert (got.expect or {}).get("clipboard_chars", 0) > 0


def test_an_engine_that_does_not_copy_is_NOT_RUN_rather_than_a_timing(page):
    """THE DEFECT. The keystroke is sent, 250ms passes, and before this the action returned that
    250ms as `copy_ms` -- stable, plausible, and about nothing."""
    _swallow_the_copy(page)
    got = A.select_all_copy(_ctx(page))
    assert got.ran is False, got
    assert "did not change the clipboard" in (got.reason or "")
    assert "sentinel" in (got.reason or "")
    assert not got.timings.get("copy_ms")


def test_the_refusal_says_the_number_would_have_been_the_harness_own_settle(page):
    """The reason has to name the failure, or the next person reads NOT RUN as a flaky run and adds
    the action to an allow-list."""
    _swallow_the_copy(page)
    said: list[str] = []
    got = A.select_all_copy(_ctx(page, said.append))
    assert "settle" in (got.reason or ""), got.reason


def _refuse_the_sentinel_write(page) -> None:
    """Makes clipboard writeText fail while readText still works, since the write can be refused alone."""
    page.evaluate(
        """() => {
        Object.defineProperty(navigator.clipboard, "writeText", {
          configurable: true,
          value: async () => { throw new DOMException("Write permission denied", "NotAllowedError"); },
        });
      }"""
    )


def test_a_failed_sentinel_write_does_not_re_admit_the_unchanged_clipboard(page):
    """A failed sentinel write must not let stale clipboard content pass as a fresh copy."""
    # Plausible stale content: an empty clipboard would already fail clipboard_chars > 0.
    page.evaluate("async () => await navigator.clipboard.writeText('x'.repeat(2400))")
    _refuse_the_sentinel_write(page)
    _swallow_the_copy(page)
    got = A.select_all_copy(_ctx(page))
    assert got.ran is False, got
    assert not got.timings.get("copy_ms")
    assert "did not change the clipboard" in (got.reason or ""), got.reason
    assert "snapshot" in (got.reason or ""), got.reason


def test_a_real_copy_is_still_measured_when_the_sentinel_could_not_be_written(page):
    """THE CONTROL FOR THE ABOVE. The fallback must not turn a missing write permission into a dead
    action: a copy that really happened is still a measurement, and the row says which pre-copy
    value it was confirmed against."""
    page.evaluate("async () => await navigator.clipboard.writeText('stale')")
    _refuse_the_sentinel_write(page)
    got = A.select_all_copy(_ctx(page))
    assert got.ran is True, got.reason
    assert got.timings.get("copy_ms") is not None
    assert (got.expect or {}).get("clipboard_chars", 0) > 0
    assert (got.expect or {}).get("copy_confirmed_against") == "snapshot"


def test_a_refusal_names_the_cause_and_not_just_the_exception_class(page):
    """`type(exc).__name__` is `Error` for every Playwright failure, and two complete 100K payloads
    refused this action with a reason that said exactly that and nothing else. The browser's own
    message is the diagnosis and has to travel with the row."""
    page.evaluate(
        """() => {
        for (const name of ["writeText", "readText"]) {
          Object.defineProperty(navigator.clipboard, name, {
            configurable: true,
            value: async () => { throw new DOMException("Document is not focused", "NotAllowedError"); },
          });
        }
      }"""
    )
    got = A.select_all_copy(_ctx(page))
    assert got.ran is False, got
    assert "Document is not focused" in (got.reason or ""), got.reason


def test_the_guard_is_on_the_clipboard_and_not_on_the_engine_name():
    """No engine list to maintain. An engine that starts performing the copy is admitted with no
    change here, and one that stops is refused with no change here."""
    import inspect

    src = inspect.getsource(A.select_all_copy)
    assert "sentinel" in src
    # Executable lines only: no code may branch on a specific engine's name.
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    for name in ("webkit", "WebKit", "firefox", "Firefox", "browser_name", "browser_type"):
        assert name not in code, f"the guard branches on {name!r} rather than on the clipboard"
