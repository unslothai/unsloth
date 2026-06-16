# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared robustness helpers for the Studio Playwright tests, the single
point of truth for the CI-runner workarounds (Chromium flags, view-transition
killer, page recovery, post-action response wait) that both
`playwright_chat_ui.py` and `playwright_extra_ui.py` need.

Importable directly by the standalone scripts:

    sys.path.insert(0, str(Path(__file__).parent))
    from _playwright_robust import (...)

Does NOT depend on pytest -- both consumers run as plain Python.
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

# Chromium launch args.
#
# The throttling flags stop Chromium deprioritising CPU/timers when it thinks
# the headless window is backgrounded (run 25586583024 stalled gemma-3-270m
# inference and the React render queue mid-test). TranslateUI strips a popup
# that intercepts pointer events; ipc-flooding-protection off lets rapid clicks
# through during the slider sweep.
#
# `--single-process` is darwin-only: the documented free-runner fix for the
# pipeTransport.js JSON-RPC crash; on Win/Linux it destabilises the renderer.
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
    """Return Chromium launch args for `platform` (defaults to `sys.platform`;
    pass a string to test the darwin branch on Linux)."""
    p = sys.platform if platform is None else platform
    args = list(_BASE_CHROMIUM_ARGS)
    if p == "darwin":
        args.append("--single-process")
    return args


# Init scripts injected into every Playwright context.
#
# CSS view-transitions render a full-window pseudo-element that intercepts
# pointer events for a beat after each theme/route swap, so Playwright reports
# `<html> intercepts pointer events` on the next click (even with
# reduced_motion, since Studio calls startViewTransition() directly). Killing
# the pseudo-elements + shimming startViewTransition synchronously fixes both.
# Idempotent and safe to install on every page.
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


# Server health pre-flight.
#
# The bash wait already gates on /api/health, but on the macos-14 free runner
# /api/health can return 200 while /api/auth still 503s (auth DB mid-migration).
# A second in-script probe catches that gap before a 60s change-password timeout.


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
    """Poll {base_url}/api/health until status==200. Returns True on success,
    False on timeout; never raises. Diagnostic only -- the workflow's own
    /api/health wait is the authoritative gate."""
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
                info(f"health pre-flight OK: status=200, body keys={list((body or {}).keys())}")
            return True
        time.sleep(0.5)
    if info is not None:
        info(
            f"health pre-flight TIMED OUT after {timeout}s; "
            f"last_status={last_status}, last_body={last_body!r}"
        )
    return False


# Page recovery.
#
# Canonical "did the page die mid-test" path used by every retry block. If the
# page is closed, opens a fresh one in the same context (localStorage auth
# survives); otherwise leaves it alone. Optionally re-navigates.


def recover_or_replace_page(
    page: Any,
    ctx: Any,
    *,
    default_timeout_ms: int = 60_000,
    goto_url: str | None = None,
    settle_networkidle: bool = True,
    info: Callable[[str], None] | None = None,
) -> Any:
    """Return a usable page, replacing `page` if it is closed. If `goto_url` is
    given, navigates there and best-effort waits for networkidle. Recovery
    errors are logged and swallowed; the caller retries a still-broken page."""
    try:
        if page.is_closed():
            page = ctx.new_page()
            page.set_default_timeout(default_timeout_ms)
    except Exception as exc:
        if info is not None:
            info(f"recovery: page.is_closed() check failed: {exc!r}")
    if goto_url is not None:
        try:
            page.goto(goto_url, wait_until = "domcontentloaded", timeout = default_timeout_ms)
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
    """Click + wait for the matching XHR/fetch response. Returns (status, None)
    on success or (None, exception) on capture failure. Callers check
    `status >= 400` to surface server rejections immediately. Falls back to a
    fire-and-forget click on any wait error so the outer retry loop runs."""
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


# Console-error / page-error filtering.
#
#   - BENIGN_PAGE_ERROR_PATTERNS: JS errors from slow CI infra (timeouts,
#     request races) with no user-visible effect; the page-error gate must not
#     count these.
#   - BENIGN_CONSOLE_ERROR_PATTERNS: same-cause console.error events, used only
#     to filter noise from diagnostic dumps (tests don't gate on console.error).

BENIGN_PAGE_ERROR_PATTERNS: tuple[str, ...] = (
    "Request failed (422)",
    "Failed to fetch",
    "NetworkError",
    "Load failed",
    "At least one non-system message is required",
    "An internal error occurred",
)

BENIGN_CONSOLE_ERROR_PATTERNS: tuple[str, ...] = (
    # macos-14 buffer-exhaustion under --single-process; the test catches the
    # underlying request failure via expect_response and retries.
    "net::ERR_NO_BUFFER_SPACE",
    # Intentional fetch aborts (unmount, route change) log a console.error.
    "AbortError",
    "The user aborted a request",
    # Lazy chunk no longer needed because the user navigated away mid-load.
    "Loading chunk",
    # Also a benign page-error; here for the diagnostic dump path.
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
    """Write a screenshot (`art_dir/{name}.png`) + a JSON sidecar
    (`art_dir/{name}.json`) with URL/title/body/storage. Diagnostic only, never
    raises; both are best-effort (screenshot can crowd CI font load on macos-14)."""
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


# Bounded in-page fetch.
#
# `page.evaluate(...)` has no `timeout=`, so a fetch that never resolves hangs
# the whole script until the runner timeout (run 25696797934 / PR #5387 burned
# 27+ min on one page.evaluate). `evaluate_fetch` wraps the fetch in an
# AbortController.signal so the JS side always resolves -- with a real response
# or a synthetic `{status: 0, error: "AbortError..."}` after `timeout_ms`.
def evaluate_fetch(
    page: Any,
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: Any = None,
    timeout_ms: int = 20_000,
    transport_retries: int = 2,
    transport_backoff_ms: int = 250,
) -> dict[str, Any]:
    """Run `fetch(url, opts)` in the page with an AbortSignal deadline. Returns
    `{"status", "body", "error"}`; on timeout `status==0` with an AbortError
    string. Treat `status == 0` or a non-None `error` as transport failure, not
    an HTTP response. `body` may be a str (verbatim) or dict/list (JSON-encoded
    here); pass headers explicitly for Content-Type / Authorization."""
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
    payload = {
        "url": url,
        "method": method,
        "headers": headers or {},
        "body": body_arg,
        "timeoutMs": int(timeout_ms),
    }
    # Bounded retry on transport failures only:
    #   status != 0 -> real HTTP response (incl. 4xx/5xx); propagate.
    #   AbortError  -> caller's deadline; propagate.
    #   else (==0)  -> stale-keepalive / "Failed to fetch" after auth rotation;
    #                  retry after backoff so the pool evicts the dead socket.
    last: dict[str, Any] | None = None
    attempts = max(1, int(transport_retries) + 1)
    for attempt in range(attempts):
        result = page.evaluate(js, payload)
        last = result
        try:
            status = int(result.get("status") or 0)
        except (TypeError, ValueError):
            status = 0
        if status != 0:
            return result
        err = str(result.get("error") or "")
        if "AbortError" in err:
            return result
        if attempt < attempts - 1:
            wait_ms = transport_backoff_ms * (2**attempt)
            try:
                sys.stderr.write(
                    f"[evaluate_fetch] {method} {url}: transport failure "
                    f"({attempt + 1}/{attempts}, err={err!r}); "
                    f"retrying in {wait_ms}ms\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            time.sleep(wait_ms / 1000.0)
    return last or {"status": 0, "body": None, "error": "no attempt made"}


# Wall-clock watchdog.
#
# Even with every action/fetch bounded, a strange browser wedge (CPU-pinned JS
# loop, renderer crash that doesn't propagate, asyncio deadlock) can hang the
# script. A daemon Timer calls `os._exit(2)` after `deadline_s`, printing the
# wedge location to stderr; exit code 2 lets the workflow's `set -e` propagate.
# Pick `deadline_s` above the slowest healthy run (macos-14 cold cache ~7-9 min;
# 720s leaves headroom without nearing the 30-min runner cap).
def install_wall_clock_watchdog(
    deadline_s: float,
    *,
    label: str = "playwright",
    info: Callable[[str], None] | None = None,
) -> threading.Timer:
    """Start a daemon Timer that hard-exits the process at `deadline_s`. Returns
    it so the caller can `.cancel()` on clean exit; being daemonised, it also
    dies with the process if the script exits first."""

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
