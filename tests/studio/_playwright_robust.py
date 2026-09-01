# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared CI-runner workarounds for the Unsloth Playwright tests (Chromium flags,
view-transition killer, page recovery, post-action response wait). Imported
directly by the standalone scripts; does NOT depend on pytest.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Callable

FRONTEND = Path(__file__).resolve().parents[2] / "studio" / "frontend"
_LIVE_SERVERS: list[subprocess.Popen[str]] = []
_PREV_HANDLERS: dict[int, Any] = {}

# Chromium launch args.
# Throttling flags stop Chromium deprioritising CPU/timers when it thinks the headless window is backgrounded (run
# 25586583024 stalled inference + render).
# It was darwin-only, for a pipeTransport.js JSON-RPC crash, and it caps Chromium at exactly ONE BrowserContext: opening
# a second one kills the browser with SIGTRAP, and the next new_page() raises "Target page, context or browser has been
# closed".
# That is what made "Update banner layout regression" red on every macos-14 run (it needs a context per viewport), and
# it is why playwright_chat_ui.py had to keep every step inside one context.
# Measured with chromium-headless-shell 151: with the flag, a second context dies immediately;
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
    """Chromium launch args. Same on every platform; `platform` is accepted so
    callers that pass one keep working."""
    del platform
    return list(_BASE_CHROMIUM_ARGS)


# Init script injected into every Playwright context.
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


# The smoke pages are dev-server-only, so each harness owns its server.
# Hence the process group, stdout drain and SIGKILL escalation.
# A backgrounded `npm run dev &` puts the npm WRAPPER in $!, and killing that orphans the node child holding the port
# and stdout.


# Server health pre-flight.
# On the macos-14 free runner /api/health can return 200 while /api/auth still 503s (auth DB mid-migration);
# this in-script probe catches that gap before a 60s change-password timeout.
def drain_process_output(proc: subprocess.Popen[str], sink: deque[str] | None = None) -> None:
    """Consume vite's output so its pipe cannot fill and wedge; keep the tail for errors."""
    if proc.stdout is not None:
        for line in proc.stdout:
            if sink is not None:
                sink.append(line.rstrip())


def _port_is_taken(port: int, host: str) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((host, port)) == 0


def _stop_live_servers() -> None:
    while _LIVE_SERVERS:
        stop_process(_LIVE_SERVERS.pop())


def _handle_fatal_signal(signum, frame) -> None:
    _stop_live_servers()
    previous = _PREV_HANDLERS.get(signum, signal.SIG_DFL)
    if callable(previous):
        previous(signum, frame)
        return
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _arm_teardown_signals() -> None:
    """`finally` covers exceptions and SIGINT but not SIGTERM, and a CI cancel sends SIGTERM.
    Without this the server outlives the harness, which is the whole thing being fixed."""
    if _PREV_HANDLERS or os.name == "nt":
        return
    for signum in (signal.SIGTERM, signal.SIGHUP):
        try:
            _PREV_HANDLERS[signum] = signal.signal(signum, _handle_fatal_signal)
        except (ValueError, OSError):
            _PREV_HANDLERS.clear()
            return


def _require_frontend_toolchain() -> None:
    """Fail with the cause when the frontend dev dependencies are not installed.

    `npm run dev` on a tree with no `node_modules` exits 127 with `sh: 1: vite: not found`,
    and the readiness poll then reports "vite exited with code 127", which reads as a vite
    crash. It is not: the toolchain was never installed, and no amount of retrying or
    port-shuffling will help. A missing toolchain and a broken one are different failures
    and must not look the same.

    This is not hypothetical. A job that installs Unsloth from a warm frontend-dist cache
    never builds the frontend, so `studio/setup.sh` skips its `npm install` and there is no
    `node_modules` for this harness to use, while the same job on a cold cache builds and
    passes. That makes the failure look like flake instead of a missing setup step.
    """
    if not FRONTEND.is_dir():
        raise RuntimeError(f"no frontend at {FRONTEND}; this harness must run from the repo")
    binaries = FRONTEND / "node_modules" / ".bin"
    if any((binaries / name).exists() for name in ("vite", "vite.cmd", "vite.exe", "vite.bunx")):
        return
    raise RuntimeError(
        f"the frontend dev dependencies are not installed at {FRONTEND / 'node_modules'}, so "
        f"`npm run dev` would exit 127 with 'vite: not found'.\n"
        f"Run `npm ci` in {FRONTEND} first. In CI, a job that restores a prebuilt "
        f"studio/frontend/dist from cache never builds the frontend, so setup.sh skips its "
        f"npm install and this directory stays empty: such a job has to install the "
        f"dependencies itself before running a harness that serves its own vite."
    )


def start_vite(port: int, *, host: str = "127.0.0.1") -> subprocess.Popen[str]:
    """Start `vite dev` on `port` in its own process group, with stdout drained.

    Refuses an occupied port. --strictPort would make vite exit anyway, and then the
    readiness poll would be talking to whatever else is listening, not to us.
    """
    if _port_is_taken(port, host):
        raise RuntimeError(
            f"{host}:{port} is already serving. Stop it, or move this harness with SMOKE_PORT."
        )
    _require_frontend_toolchain()
    process_group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    # shutil.which honours PATHEXT, so this resolves npm.cmd on Windows;
    npm = shutil.which("npm") or "npm"
    proc = subprocess.Popen(
        [npm, "run", "dev", "--", "--host", host, "--port", str(port), "--strictPort"],
        cwd = FRONTEND,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        **process_group,
    )
    tail: deque[str] = deque(maxlen = 20)
    proc.vite_tail = tail  # type: ignore[attr-defined]
    threading.Thread(target = drain_process_output, args = (proc, tail), daemon = True).start()
    _LIVE_SERVERS.append(proc)
    _arm_teardown_signals()
    atexit.register(_stop_live_servers)
    return proc


def stop_process(proc: subprocess.Popen[str]) -> None:
    """SIGTERM the process group, escalating to SIGKILL if it does not go."""
    if proc in _LIVE_SERVERS:
        _LIVE_SERVERS.remove(proc)
    if proc.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T"],
            check = False,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    try:
        proc.wait(timeout = 10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check = False,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout = 10)
        except subprocess.TimeoutExpired:
            pass


def wait_for_smoke_page(
    url: str,
    entry: str,
    *,
    proc: subprocess.Popen[str] | None = None,
    timeout_s: float = 120.0,
    info: Callable[[str], None] | None = None,
) -> None:
    """Block until `url` serves a page that really references `entry`.

    Vite's SPA fallback answers 200 with index.html for any path it cannot resolve, so a
    deleted smoke page still looks healthy. Match the module specifier, not the status.
    """
    deadline = time.monotonic() + timeout_s
    last = "no response"
    while time.monotonic() < deadline:
        # Ours died (busy port, missing node_modules):
        if proc is not None and proc.poll() is not None:
            tail = "\n".join(getattr(proc, "vite_tail", []))
            raise RuntimeError(
                f"vite exited with code {proc.returncode} before serving {url}\n{tail}"
            )
        # Called from a `finally`: never raise over the failure that sent us here.
        try:
            with urllib.request.urlopen(url, timeout = 3.0) as r:
                body = r.read().decode("utf-8", errors = "replace")
                if r.status == 200 and entry in body:
                    if info is not None:
                        info(f"{url} ready (serves {entry})")
                    return
                last = (
                    f"status={r.status}, {entry} "
                    f"{'present' if entry in body else 'MISSING (SPA fallback?)'}"
                )
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    raise RuntimeError(f"vite did not serve {url} referencing {entry} within {timeout_s}s ({last})")


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


# Page recovery: if the page died mid-test, open a fresh one in the same context (localStorage auth survives);
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
# - BENIGN_PAGE_ERROR_PATTERNS: CI-infra JS errors with no user-visible effect;
# - BENIGN_CONSOLE_ERROR_PATTERNS: same-cause console.error events, used only to filter noise from diagnostic dumps
BENIGN_PAGE_ERROR_PATTERNS: tuple[str, ...] = (
    "Request failed (422)",
    "Failed to fetch",
    "NetworkError",
    "Load failed",
    "At least one non-system message is required",
    "An internal error occurred",
)

BENIGN_CONSOLE_ERROR_PATTERNS: tuple[str, ...] = (
    # macos-14 buffer exhaustion;
    # the test catches the underlying request failure via expect_response and retries.
    "net::ERR_NO_BUFFER_SPACE",
    # Intentional fetch aborts (unmount, route change) log a console.error.
    "AbortError",
    "The user aborted a request",
    # Lazy chunk no longer needed because the user navigated away mid-load.
    "Loading chunk",
    "Failed to fetch",
)


def wait_for_first(locator: Any, *, timeout_ms: int = 10_000) -> Any | None:
    """The first match once it exists, or None once the wait expires.

    `Locator.count()` does not wait. It answers about this instant, so every
    `if locator.count() > 0:` gate is a race with rendering that reads as "the
    feature is missing" the moment anything delays it -- and reports that as a
    product failure rather than as a timeout.

    #9251 is what this is written from. Its reload snapshot paints a cloned
    overlay over the app and takes it down on hydration (or after 5s), which
    opens a window where the composer is on screen but not yet in the
    accessibility tree. The Compare step read `count() == 0` **six milliseconds**
    after it began and reported "Compare nav not found", which is a true
    statement about that instant and a false one about the app.

    Playwright's auto-waiting covers actions and expectations, not `count()`, so
    the wait has to be asked for. Returning None rather than raising keeps the
    caller's existing "is this control present at all" branch, including the
    fallbacks that legitimately expect a miss.
    """
    # Imported here, not at module scope:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    try:
        locator.first.wait_for(state = "attached", timeout = timeout_ms)
    except PlaywrightTimeoutError:
        return None
    return locator.first


def is_benign_page_error(msg: str) -> bool:
    return any(p in msg for p in BENIGN_PAGE_ERROR_PATTERNS)


def is_benign_console_error(msg: str) -> bool:
    return any(p in msg for p in BENIGN_CONSOLE_ERROR_PATTERNS)


def echo_browser_errors(page: Any, info: Callable[[str], None]) -> None:
    """Print what the browser knows, live, as it happens.

    A harness that only asserts on the DOM cannot tell an entry module that threw
    from one that is merely slow: both end as an `expect(...)` timeout on a locator
    that was never created, under an empty CI log. The smokes each own a throwaway
    page, so printing straight through beats collecting for a caller to forward.
    """
    page.on("pageerror", lambda e: info(f"pageerror: {e}"))
    page.on(
        "console",
        lambda m: info(f"console.{m.type}: {m.text}") if m.type == "error" else None,
    )
    page.on("requestfailed", lambda r: info(f"requestfailed: {r.url} {r.failure}"))
    # Vite reloads the page after re-optimizing a late-discovered dep, unmounting the tree mid-assertion.
    page.on(
        "framenavigated",
        lambda f: info(f"navigated: {f.url}") if f is page.main_frame else None,
    )


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


# Markers for the transient Playwright error raised when a navigation, reload, or auth refresh destroys the JS
_CONTEXT_LOST_MARKERS = (
    "execution context was destroyed",
    "context with specified id",
    "frame was detached",
    "target closed",
    "target page, context or browser has been closed",
    "execution context is not available",
)

# HTTP methods whose replay is side-effect-free, so an evaluate_fetch hit by a mid-call context loss may safely re-run.
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


# Robust page/locator.evaluate.
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
# evaluate_fetch wraps the fetch in an AbortController.signal so the JS side always resolves
# real response, or synthetic `{status: 0, error: "AbortError..."}` after timeout_ms.
# `page.evaluate(...)` has no `timeout=`, so a stuck fetch hangs the script until the runner timeout (run 25696797934 /
# PR #5387 burned 27+ min).
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
    # Retry transport failures only:
    last: dict[str, Any] | None = None
    attempts = max(1, int(transport_retries) + 1)
    # Replay the in-page evaluate on a context loss only for idempotent reads;
    # mutating methods (POST/PUT/PATCH/DELETE) may have already hit the backend, so retrying would re-send them (see
    # docstring).
    if retry_on_context_loss is None:
        retry_on_context_loss = method.upper() in _IDEMPOTENT_METHODS
    ctx_retries = 2 if retry_on_context_loss else 0
    for attempt in range(attempts):
        # robust_evaluate retries the evaluate when a navigation destroys the execution context mid-call;
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
# A daemon Timer calls os._exit(2) after deadline_s;
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


def click_forced(
    locator: Any,
    *,
    timeout_ms: int = 5_000,
    **click_kwargs: Any,
) -> None:
    """Scroll into view, then click with actionability checks off.

    `click(force = True)` skips Playwright's actionability checks, which is what you
    want against a menu whose overlay would otherwise intercept the click. It also
    skips the part that scrolls the element into view, and Playwright will not click
    a point it cannot reach:

        playwright._impl._errors.Error: Locator.click: Element is outside of the viewport

    That is what took down `Compare tab: send to two panes` on macOS. The menu item
    existed, was found, and was off-screen, because a Mac runner's window is shorter
    than a Linux one and the item sits at the bottom of a long menu. On Linux the same
    code has always worked, which is why three forced clicks sat here unnoticed since
    the composer redesign.

    Scrolling first keeps the reason force was used -- the overlay is still ignored --
    and removes the assumption that the element happens to be on screen.

    The scroll is best-effort: an element that cannot be scrolled (fixed position, zero
    size) should still reach the click, and fail there with Playwright's own message
    rather than here with a scrolling one.
    """
    try:
        locator.scroll_into_view_if_needed(timeout = timeout_ms)
    except Exception:
        pass
    # Callers pass their own `timeout` through:
    locator.click(force = True, **click_kwargs)
