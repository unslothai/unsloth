# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared CI-runner workarounds for the Unsloth Playwright tests (Chromium flags,
view-transition killer, page recovery, post-action response wait). Imported
directly by the standalone scripts; does NOT depend on pytest.
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
# Throttling flags stop Chromium deprioritising CPU/timers when it thinks the
# headless window is backgrounded (run 25586583024 stalled inference + render).
# TranslateUI strips a pointer-intercepting popup; ipc-flooding-protection off
# lets rapid clicks through during the slider sweep.
# `--single-process` is darwin-only (fixes the pipeTransport.js JSON-RPC crash);
# on Win/Linux it destabilises the renderer.
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
    """Chromium launch args for `platform` (defaults to `sys.platform`; pass a
    string to test the darwin branch on Linux)."""
    p = sys.platform if platform is None else platform
    args = list(_BASE_CHROMIUM_ARGS)
    if p == "darwin":
        args.append("--single-process")
    return args


# Init script injected into every Playwright context.
# CSS view-transitions render a full-window pseudo-element that intercepts
# pointer events after each theme/route swap, so Playwright reports
# `<html> intercepts pointer events` on the next click. Killing the
# pseudo-elements + shimming startViewTransition synchronously fixes both.
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
# On the macos-14 free runner /api/health can return 200 while /api/auth still
# 503s (auth DB mid-migration); this in-script probe catches that gap before a
# 60s change-password timeout.


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
    """Poll {base_url}/api/health until status==200; True on success, False on
    timeout, never raises. Diagnostic only (the workflow's wait is authoritative)."""
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    last_body: dict | None = None
    while time.monotonic() < deadline:
        status, body = _http_get_status_and_body(
            f"{base_url}/api/health",
            timeout = 3.0,
        )
        last_status, last_body = status, body
        # Accept any 200 -- different Unsloth builds report status differently.
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


# Page recovery: if the page died mid-test, open a fresh one in the same context
# (localStorage auth survives); otherwise leave it alone. Optionally re-navigates.


def recover_or_replace_page(
    page: Any,
    ctx: Any,
    *,
    default_timeout_ms: int = 60_000,
    goto_url: str | None = None,
    settle_networkidle: bool = True,
    info: Callable[[str], None] | None = None,
) -> Any:
    """Return a usable page, replacing `page` if closed; optionally navigate to
    `goto_url`. Recovery errors are logged and swallowed for the caller to retry."""
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
    """Click + wait for the matching XHR/fetch response; (status, None) on success
    or (None, exception) on capture failure. Falls back to a fire-and-forget click
    so the outer retry loop runs. Callers check `status >= 400`."""
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
#   - BENIGN_PAGE_ERROR_PATTERNS: CI-infra JS errors with no user-visible effect;
#     the page-error gate must not count these.
#   - BENIGN_CONSOLE_ERROR_PATTERNS: same-cause console.error events, used only to
#     filter noise from diagnostic dumps (tests don't gate on console.error).

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
    """Write a screenshot + JSON sidecar (URL/title/body/storage) under art_dir.
    Diagnostic only, never raises; both best-effort."""
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


# Markers for the transient Playwright error raised when a navigation, reload, or
# auth refresh destroys the JS execution context while an evaluate is in flight.
# Stored lowercase and matched against a lowercased message: Playwright varies the
# casing across versions ("Frame was detached" vs "frame was detached"), so a
# case-sensitive check would miss the very races this is meant to catch.
_CONTEXT_LOST_MARKERS = (
    "execution context was destroyed",
    "context with specified id",
    "frame was detached",
    "target closed",
    "target page, context or browser has been closed",
    "execution context is not available",
)

# HTTP methods whose replay is side-effect-free, so an evaluate_fetch hit by a
# mid-call context loss may safely re-run. Mutating methods are excluded by
# default (see evaluate_fetch) to avoid double-applying an already-sent request.
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


# Robust page/locator.evaluate.
# A navigation mid-evaluate destroys the execution context and raises at the Python
# level (not a JS result), which would crash the script. Retry that transient class
# within a small budget, settling the page first; non-transient or persistent errors
# still propagate.
def robust_evaluate(
    target: Any,
    expression: str,
    arg: Any = None,
    *,
    retries: int = 2,
    backoff_ms: int = 250,
) -> Any:
    """`target.evaluate(expression, arg)` for a Page or Locator, retried when a
    concurrent navigation destroys the execution context. Re-raises on a
    non-transient error or after the final attempt."""
    page = target if hasattr(target, "wait_for_load_state") else getattr(target, "page", None)
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        try:
            return target.evaluate(expression, arg)
        except Exception as exc:
            exc_msg = str(exc).lower()
            transient = any(s in exc_msg for s in _CONTEXT_LOST_MARKERS)
            if not transient or attempt == attempts - 1:
                raise
            try:
                sys.stderr.write(
                    f"[robust_evaluate] execution context lost "
                    f"({attempt + 1}/{attempts}); settling + retrying\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            if page is not None:
                try:
                    page.wait_for_load_state("domcontentloaded", timeout = 10_000)
                except Exception:
                    pass
            time.sleep((backoff_ms * (2**attempt)) / 1000.0)


# Bounded in-page fetch.
# `page.evaluate(...)` has no `timeout=`, so a stuck fetch hangs the script until
# the runner timeout (run 25696797934 / PR #5387 burned 27+ min). evaluate_fetch
# wraps the fetch in an AbortController.signal so the JS side always resolves --
# real response, or synthetic `{status: 0, error: "AbortError..."}` after timeout_ms.
# It also retries the evaluate itself when a navigation destroys the execution
# context mid-call (a transient Playwright race, not a real fetch failure).
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
    retry_on_context_loss: bool | None = None,
) -> dict[str, Any]:
    """Run `fetch(url, opts)` in the page with an AbortSignal deadline; returns
    `{"status", "body", "error"}` (status==0 + AbortError on timeout). Treat
    status==0 or non-None error as transport failure. `body` may be str (verbatim)
    or dict/list (JSON-encoded); pass headers explicitly for Content-Type/Auth.

    `retry_on_context_loss` controls whether a navigation that destroys the JS
    context mid-call replays the in-page fetch. The request may have already
    reached the backend before the context died, so replaying a mutating call is
    unsafe: a spent single-use POST /api/auth/refresh comes back 401, and a
    duplicate POST /api/inference/load that lands while the first is still in
    `loading_models` is rejected (the backend returns False -> 500) even though
    the original load succeeds. Default (None) therefore retries only idempotent
    reads (GET/HEAD/OPTIONS) and never replays a mutating method; pass an explicit
    bool to override per call. Context loss on a non-retried call propagates."""
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
    # Retry transport failures only: status != 0 (real HTTP) and AbortError
    # (caller's deadline) propagate; status==0 (stale-keepalive / "Failed to
    # fetch" after auth rotation) retries after backoff to evict the dead socket.
    last: dict[str, Any] | None = None
    attempts = max(1, int(transport_retries) + 1)
    # Replay the in-page evaluate on a context loss only for idempotent reads;
    # mutating methods (POST/PUT/PATCH/DELETE) may have already hit the backend,
    # so retrying would re-send them (see docstring). Honor an explicit override.
    if retry_on_context_loss is None:
        retry_on_context_loss = method.upper() in _IDEMPOTENT_METHODS
    ctx_retries = 2 if retry_on_context_loss else 0
    for attempt in range(attempts):
        # robust_evaluate retries the evaluate when a navigation destroys the
        # execution context mid-call; the loop here retries transport failures.
        result = robust_evaluate(
            page, js, payload, retries = ctx_retries, backoff_ms = transport_backoff_ms
        )
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
# A browser wedge (CPU-pinned JS, silent renderer crash, asyncio deadlock) can
# still hang the script. A daemon Timer calls os._exit(2) after deadline_s; exit
# code 2 lets the workflow's `set -e` propagate. Pick deadline_s above the
# slowest healthy run (macos-14 cold cache ~7-9 min) but under the 30-min cap.
def install_wall_clock_watchdog(
    deadline_s: float,
    *,
    label: str = "playwright",
    info: Callable[[str], None] | None = None,
) -> threading.Timer:
    """Start a daemon Timer that hard-exits the process at `deadline_s`; returned
    so the caller can `.cancel()` on clean exit (daemonised, dies with process)."""

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
