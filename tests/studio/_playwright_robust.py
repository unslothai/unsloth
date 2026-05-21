# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared robustness helpers for the Studio Playwright tests.

Both `playwright_chat_ui.py` and `playwright_extra_ui.py` re-implemented
the same set of CI-runner workarounds (Chromium launch flags, view-
transition CSS killer, change-password retry / page-recovery, post-
action response wait). When one diverged the other slowly rotted; the
mac/win/linux failure modes are mostly identical so the cure is the
same. This module is the single point of truth.

Importable directly by the standalone scripts via:

    sys.path.insert(0, str(Path(__file__).parent))
    from _playwright_robust import (...)

It does NOT depend on pytest -- both consumers run as plain Python.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────
# Chromium launch args.
# ─────────────────────────────────────────────────────────────────────
#
# Base set works on every CI runner. The four "throttling" flags fight
# Chromium's tendency to deprioritise CPU + timers when it thinks the
# window is backgrounded -- which CI runners routinely flag because
# the headless context has no real focus. Without these, gemma-3-270m
# inference on Mac slowed to a crawl mid-test (run 25586583024 had a
# turn budget that never released the Stop button) and the React
# render queue stalled long enough for `wait_for_function` waits to
# crowd their per-turn budget.
#
# `--disable-features=TranslateUI` strips the translate prompt that
# occasionally adds a popup which intercepts pointer events.
# `--disable-ipc-flooding-protection` lets us send rapid-fire clicks
# during the slider sweep without Chromium queuing them.
#
# `--single-process` is darwin-only. On Mac it is the documented free-
# runner fix for the pipeTransport.js JSON-RPC crash; on Win/Linux it
# strictly destabilises the renderer-isolation safety net so any
# crash takes the whole context down.
_BASE_CHROMIUM_ARGS = (
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-features=TranslateUI",
    "--disable-ipc-flooding-protection",
)


def chromium_launch_args(platform: str | None = None) -> list[str]:
    """Return the Chromium launch arg list appropriate for `platform`.

    Defaults to the running interpreter's `sys.platform`. Pass a
    string to test the darwin branch on Linux.
    """
    p = sys.platform if platform is None else platform
    args = list(_BASE_CHROMIUM_ARGS)
    if p == "darwin":
        args.append("--single-process")
    return args


# ─────────────────────────────────────────────────────────────────────
# Init scripts injected into every Playwright context.
# ─────────────────────────────────────────────────────────────────────
#
# CSS view-transitions are otherwise rendered as a full-window
# pseudo-element that intercepts pointer events for a beat after each
# theme/route swap. Even with `reduced_motion = "reduce"` set on the
# context, Studio's components run their own startViewTransition() in
# a few places (theme toggle, sidebar collapse) and Playwright's
# actionability check then reports `<html> intercepts pointer events`
# on the next click. Killing the pseudo-elements + monkey-patching
# document.startViewTransition into a synchronous shim removes both
# failure modes. Idempotent and safe to install on every page.
_VIEW_TRANSITION_KILLER_JS = """
(function () {
    try {
        const css = `
            ::view-transition,
            ::view-transition-group(*),
            ::view-transition-image-pair(*),
            ::view-transition-old(*),
            ::view-transition-new(*) {
                display: none !important;
                animation: none !important;
                opacity: 0 !important;
            }
            html, body { pointer-events: auto !important; }
        `;
        const style = document.createElement("style");
        style.id = "playwright-no-view-transition";
        style.textContent = css;
        (document.head || document.documentElement).appendChild(style);
        if (typeof document.startViewTransition === "function") {
            document.startViewTransition = function (cb) {
                try { if (cb) cb(); } catch (e) {}
                return {
                    ready: Promise.resolve(),
                    finished: Promise.resolve(),
                    updateCallbackDone: Promise.resolve(),
                    skipTransition: () => {},
                };
            };
        }
    } catch (e) { /* noop */ }
})();
"""


def install_view_transition_killer(ctx: Any) -> None:
    """Inject the CSS view-transition killer into every page in `ctx`."""
    ctx.add_init_script(_VIEW_TRANSITION_KILLER_JS)


# ─────────────────────────────────────────────────────────────────────
# Server health pre-flight.
# ─────────────────────────────────────────────────────────────────────
#
# Both workflows already wait for /api/health at the bash level before
# launching the Python script, but the macos-14 free runner has been
# observed to surface a brief window where /api/health responds 200
# yet /api/auth endpoints still 503 because the auth DB hasn't
# finished migrating. A second probe inside the script catches that
# narrow gap before we sink 60s into a change-password timeout.


def _http_get_status_and_body(url: str, timeout: float) -> tuple[int, dict | None]:
    try:
        with urllib.request.urlopen(url, timeout = timeout) as r:
            try:
                body = json.loads(r.read().decode("utf-8", errors = "replace"))
            except Exception:
                body = None
            return r.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        return -1, None


def wait_for_health(
    base_url: str,
    *,
    timeout: float = 30.0,
    info: Callable[[str], None] | None = None,
) -> bool:
    """Poll {base_url}/api/health until status==200 with healthy body.

    Returns True on success, False on timeout. Never raises -- the
    caller decides whether to fail. The test scripts use the boolean
    only for diagnostic logging, since the workflow's own /api/health
    wait is the authoritative gate.
    """
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    last_body: dict | None = None
    while time.monotonic() < deadline:
        status, body = _http_get_status_and_body(
            f"{base_url}/api/health",
            timeout = 3.0,
        )
        last_status, last_body = status, body
        # `chat_only` and `status` keys both exist; prefer status==healthy
        # but accept any 200 -- different Studio builds report differently.
        if status == 200:
            if info is not None:
                info(
                    f"health pre-flight OK: status=200, body keys={list((body or {}).keys())}"
                )
            return True
        time.sleep(0.5)
    if info is not None:
        info(
            f"health pre-flight TIMED OUT after {timeout}s; "
            f"last_status={last_status}, last_body={last_body!r}"
        )
    return False


# ─────────────────────────────────────────────────────────────────────
# Page recovery.
# ─────────────────────────────────────────────────────────────────────
#
# The single canonical "did the page die mid-test" recovery path. Used
# by every retry block in both scripts. If the page is closed, opens a
# fresh one in the same context (auth state in localStorage survives);
# otherwise leaves the page alone. Optionally re-navigates.


def recover_or_replace_page(
    page: Any,
    ctx: Any,
    *,
    default_timeout_ms: int = 60_000,
    goto_url: str | None = None,
    settle_networkidle: bool = True,
    info: Callable[[str], None] | None = None,
) -> Any:
    """Return a usable page. Replaces `page` if it is closed.

    If `goto_url` is provided, navigates the (possibly new) page there
    and best-effort waits for networkidle. Errors during recovery are
    logged through `info` (if provided) and swallowed -- the caller
    handles a still-broken page on the next retry iteration.
    """
    try:
        if page.is_closed():
            page = ctx.new_page()
            page.set_default_timeout(default_timeout_ms)
    except Exception as exc:
        if info is not None:
            info(f"recovery: page.is_closed() check failed: {exc!r}")
    if goto_url is not None:
        try:
            page.goto(
                goto_url, wait_until = "domcontentloaded", timeout = default_timeout_ms
            )
            if settle_networkidle:
                try:
                    page.wait_for_load_state("networkidle", timeout = 30_000)
                except Exception:
                    pass
        except Exception as exc:
            if info is not None:
                info(f"recovery: page.goto({goto_url!r}) failed: {exc!r}")
    return page


# ─────────────────────────────────────────────────────────────────────
# POST-and-wait: surface server errors immediately, fall back cleanly.
# ─────────────────────────────────────────────────────────────────────


def click_and_wait_for_response(
    page: Any,
    *,
    url_substr: str,
    method: str = "POST",
    do_click: Callable[[], None],
    timeout_ms: int = 30_000,
    info: Callable[[str], None] | None = None,
) -> tuple[int | None, Exception | None]:
    """Click + wait for the matching XHR/fetch response in one step.

    Returns (status, err). On success: (status, None). On failure to
    capture the response: (None, exception). Callers typically check
    `status >= 400` to surface a server-side rejection immediately
    rather than discovering it 60s later via a downstream wait_for.
    Falls back to a fire-and-forget click on any wait error so the
    outer retry loop still runs.
    """
    try:
        with page.expect_response(
            lambda r: url_substr in r.url and r.request.method == method,
            timeout = timeout_ms,
        ) as resp_info:
            do_click()
        resp = resp_info.value
        return resp.status, None
    except Exception as exc:
        if info is not None:
            info(
                f"click_and_wait_for_response({url_substr!r}, {method}) failed: "
                f"{type(exc).__name__}: {str(exc)[:150]}; falling back to fire-and-forget click"
            )
        try:
            do_click()
        except Exception:
            pass
        return None, exc


# ─────────────────────────────────────────────────────────────────────
# Console-error / page-error filtering.
# ─────────────────────────────────────────────────────────────────────
#
# Two categories:
#   - BENIGN_PAGE_ERROR_PATTERNS: thrown JS errors that fire as a side
#     effect of slow CI infra (server timeouts, request races) and have
#     no user-visible consequence. The page-error gate at the end of
#     each test should NOT count these.
#   - BENIGN_CONSOLE_ERROR_PATTERNS: console.error events that fire
#     for the same reason. Tests don't gate on console.error today
#     (they only count for diagnostics), but the same list is useful
#     for filtering noise out of the diagnostic dumps.

BENIGN_PAGE_ERROR_PATTERNS: tuple[str, ...] = (
    "Request failed (422)",
    "Failed to fetch",
    "NetworkError",
    "Load failed",
    "At least one non-system message is required",
    "An internal error occurred",
)

BENIGN_CONSOLE_ERROR_PATTERNS: tuple[str, ...] = (
    # macos-14 free runner buffer-exhaustion under --single-process
    # Chromium. The browser surfaces this on resource fetches but the
    # test catches the underlying request failure via expect_response
    # and retries; the console line itself is informational.
    "net::ERR_NO_BUFFER_SPACE",
    # Chromium emits a console.error every time a fetch is aborted,
    # even when the abort is intentional (component unmount, route
    # change). All four scripts trigger several of these per run.
    "AbortError",
    "The user aborted a request",
    # Same shape: lazy-loaded chunk that's no longer needed because
    # the user navigated away mid-load.
    "Loading chunk",
    # Filtered as a benign page-error too; included here for the
    # parallel diagnostic dump path.
    "Failed to fetch",
)


def is_benign_page_error(msg: str) -> bool:
    return any(p in msg for p in BENIGN_PAGE_ERROR_PATTERNS)


def is_benign_console_error(msg: str) -> bool:
    return any(p in msg for p in BENIGN_CONSOLE_ERROR_PATTERNS)


# ─────────────────────────────────────────────────────────────────────
# Diagnostic dump.
# ─────────────────────────────────────────────────────────────────────


def dump_diagnostics(
    page: Any,
    art_dir: Path | str,
    name: str,
    *,
    info: Callable[[str], None] | None = None,
    extra: dict | None = None,
) -> None:
    """Write a screenshot + URL/title + body excerpt + storage dump.

    Diagnostic only. Never raises. The screenshot path lives in
    `art_dir/{name}.png`; the JSON sidecar lives in `art_dir/{name}.json`.
    The screenshot is wrapped in try/except because Page.screenshot
    waits for webfonts to load and can crowd CI font load on macos-14
    even at 90s. The JSON sidecar is best-effort too.
    """
    art = Path(art_dir)
    try:
        art.mkdir(parents = True, exist_ok = True)
    except Exception:
        pass
    try:
        page.screenshot(
            path = str(art / f"{name}.png"),
            full_page = True,
            timeout = 90_000,
            animations = "disabled",
        )
    except Exception as exc:
        if info is not None:
            info(f"diagnostics: screenshot {name} failed: {exc}")
    payload: dict[str, Any] = {"name": name, "ts": time.time()}
    try:
        payload["url"] = page.url
    except Exception:
        payload["url"] = "<page closed>"
    try:
        payload["title"] = page.title()
    except Exception:
        pass
    try:
        payload["body_excerpt"] = page.evaluate(
            """() => (document.body && document.body.innerText || '').slice(0, 800)""",
        )
    except Exception:
        pass
    try:
        payload["local_storage_keys"] = page.evaluate(
            """() => Object.keys(localStorage)""",
        )
    except Exception:
        pass
    if extra:
        payload["extra"] = extra
    try:
        (art / f"{name}.json").write_text(
            json.dumps(payload, indent = 2, default = str),
            encoding = "utf-8",
        )
    except Exception as exc:
        if info is not None:
            info(f"diagnostics: json sidecar {name} failed: {exc}")


# ─────────────────────────────────────────────────────────────────────
# Bounded in-page fetch.
# ─────────────────────────────────────────────────────────────────────
#
# Playwright's `page.evaluate(...)` has no `timeout=` argument. If the
# JS body awaits a fetch that never resolves (the renderer's network
# thread wedges, the server accepts the connection but never replies,
# the macos-14 free runner under --single-process Chromium loses its
# IPC pipe), the entire Python script hangs until the runner-level
# timeout fires. Run 25696797934 / job 75446949358 on PR #5387 showed
# this exact failure: studio.log went idle after the chat surface
# mounted, no further requests reached the server, and Playwright
# burned 27+ minutes on a single page.evaluate(fetch /api/inference/
# load) before the 30-min runner cancel.
#
# `evaluate_fetch` wraps the fetch in an AbortController.signal so the
# JS side resolves either with a real response or with a synthetic
# `{status: 0, error: "AbortError..."}` after `timeout_ms` ms. Either
# way page.evaluate returns and the script proceeds (or fails) with
# a debuggable signal instead of a silent wedge.
def evaluate_fetch(
    page: Any,
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: Any = None,
    timeout_ms: int = 20_000,
) -> dict[str, Any]:
    """Run `fetch(url, opts)` inside the page with an AbortSignal deadline.

    Returns `{"status": int, "body": parsed_or_text, "error": str|None}`.
    On AbortSignal timeout returns `{"status": 0, "body": None, "error":
    "AbortError: ..."}`. Callers should treat `status == 0` (or any
    non-None `error`) as a transport failure rather than an HTTP
    response.

    `body` may be a `str` (sent verbatim) or a `dict`/`list` (JSON-
    encoded here). Pass headers explicitly when you need
    `Content-Type: application/json` or an `Authorization` bearer.
    """
    body_arg: str | None
    if body is None:
        body_arg = None
    elif isinstance(body, (str, bytes)):
        body_arg = body if isinstance(body, str) else body.decode("utf-8")
    else:
        body_arg = json.dumps(body)
    js = """
        async ({url, method, headers, body, timeoutMs}) => {
            const ctrl = new AbortController();
            const t = setTimeout(() => ctrl.abort(), timeoutMs);
            try {
                const opts = {method: method, headers: headers, signal: ctrl.signal};
                if (body !== null) opts.body = body;
                const r = await fetch(url, opts);
                clearTimeout(t);
                let parsed;
                try {
                    parsed = await r.json();
                } catch (_e) {
                    try {
                        parsed = await r.text();
                    } catch (_e2) {
                        parsed = null;
                    }
                }
                return {status: r.status, body: parsed, error: null};
            } catch (e) {
                clearTimeout(t);
                return {status: 0, body: null, error: String(e)};
            }
        }
    """
    return page.evaluate(
        js,
        {
            "url": url,
            "method": method,
            "headers": headers or {},
            "body": body_arg,
            "timeoutMs": int(timeout_ms),
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Wall-clock watchdog.
# ─────────────────────────────────────────────────────────────────────
#
# Even with every action and fetch bounded, a sufficiently strange
# wedge inside the browser (a CPU-pinned JS infinite loop, a renderer
# crash that doesn't propagate to Playwright, an asyncio deadlock in
# the sync wrapper) can still hang the script. The watchdog is a
# daemon Timer that calls `os._exit(2)` after `deadline_s` seconds,
# printing the wedge location to stderr so the CI log shows where the
# script was at force-kill time. The exit code matches "test failure
# by deadline" so the workflow's `set -e` propagates correctly.
#
# Pick `deadline_s` generously enough to cover the slowest healthy
# run -- macos-14 free runners with cold caches measure ~7-9 min for
# the comprehensive chat UI test. 12 minutes (720 s) leaves headroom
# without amplifying every real wedge to the 30-min runner-level cap.
def install_wall_clock_watchdog(
    deadline_s: float,
    *,
    label: str = "playwright",
    info: Callable[[str], None] | None = None,
) -> threading.Timer:
    """Start a daemon Timer that hard-exits the process at `deadline_s`.

    Returns the Timer so the caller can `.cancel()` it on clean exit.
    The Timer is daemonised; if the script exits normally before the
    deadline the Timer dies with the process even without an explicit
    cancel.
    """

    def _kaboom() -> None:
        msg = (
            f"[{label}] WATCHDOG: hit {deadline_s:.0f}s wall-clock "
            f"deadline; forcing exit(2). The script wedged somewhere "
            f"the per-action timeouts could not bound. Inspect the "
            f"most recent step printed above to localise."
        )
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(2)

    timer = threading.Timer(deadline_s, _kaboom)
    timer.daemon = True
    timer.start()
    if info is not None:
        info(f"watchdog armed: hard-exit at {deadline_s:.0f}s")
    return timer
