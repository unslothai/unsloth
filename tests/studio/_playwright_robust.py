# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared Playwright CI workarounds for standalone scripts; must not depend on pytest."""

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

# Throttling flags stop headless Chromium deprioritising timers; no `--single-process`: it caps
# Chromium at one BrowserContext and a second context kills the browser.
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


def usable_sync_api() -> Any | None:
    """Returns the real playwright.sync_api or None; a stub or namespace package has no file behind it."""
    try:
        import playwright.sync_api as sync_api
    except ImportError:
        return None
    if not getattr(sync_api, "__file__", None):
        return None
    return sync_api


def chromium_launch_args(platform: str | None = None) -> list[str]:
    """Chromium launch args. Same on every platform; `platform` is accepted so
    callers that pass one keep working."""
    del platform
    return list(_BASE_CHROMIUM_ARGS)


# CSS view-transition pseudo-elements intercept pointer events after theme/route swaps.
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


# Health pre-flight; /api/health can return 200 while /api/auth still 503s.


# `npm run dev &` puts the npm wrapper in $!, so kill the process group with SIGKILL escalation.


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
            _PREV_HANDLERS.clear()  # not the main thread; leave signals alone
            return


def _require_frontend_toolchain() -> None:
    """Fails with the cause when node_modules is missing, not as a vite crash exiting 127."""
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
    """Refuses an occupied port, so the readiness poll cannot be answered by another server."""
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
    # CreateProcess cannot run a bare "npm" on Windows; shutil.which resolves npm.cmd.
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
        # Called from a `finally`: never raise over the failure that sent us here.
        try:
            proc.wait(timeout = 10)
        except subprocess.TimeoutExpired:
            pass


def open_session_with_retry(
    factory: Callable[[], Any],
    *,
    retry_on: type[BaseException] | tuple[type[BaseException], ...],
    attempts: int = 3,
    pause: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> Any:
    """Retries only session creation, since safaridriver flakes on macOS runners; assertions run once."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(1, attempts + 1):
        try:
            return factory()
        except retry_on as exc:
            if attempt == attempts:
                raise
            log(f"browser session not created (attempt {attempt}/{attempts}), retrying: {exc}")
            sleep(pause)
    raise AssertionError("unreachable")


def wait_for_smoke_page(
    url: str,
    entry: str,
    *,
    proc: subprocess.Popen[str] | None = None,
    timeout_s: float = 120.0,
    info: Callable[[str], None] | None = None,
) -> None:
    """Waits for a page that references the entry module; SPA fallback answers 200 to unknown paths."""
    deadline = time.monotonic() + timeout_s
    last = "no response"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = "\n".join(getattr(proc, "vite_tail", []))
            raise RuntimeError(
                f"vite exited with code {proc.returncode} before serving {url}\n{tail}"
            )
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
        # Accept any 200: builds report the body status differently.
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


# Page-error patterns are CI noise the gate ignores; console patterns only filter diagnostic dumps.
BENIGN_PAGE_ERROR_PATTERNS: tuple[str, ...] = (
    "Request failed (422)",
    "Failed to fetch",
    "NetworkError",
    "Load failed",
    "At least one non-system message is required",
    "An internal error occurred",
)

BENIGN_CONSOLE_ERROR_PATTERNS: tuple[str, ...] = (
    "net::ERR_NO_BUFFER_SPACE",
    "AbortError",
    "The user aborted a request",
    "Loading chunk",
    "Failed to fetch",
)


# Busy runners can briefly exhaust socket buffers (net::ERR_NO_BUFFER_SPACE); retry only that.
SOCKET_BUFFER_BACKOFF_S: tuple[float, ...] = (5.0, 15.0)


def goto_with_socket_backoff(
    page: Any,
    url: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> Any:
    for backoff_s in (*SOCKET_BUFFER_BACKOFF_S, None):
        try:
            return page.goto(url, **kwargs)
        except Exception as exc:
            if backoff_s is None or "ERR_NO_BUFFER_SPACE" not in str(exc):
                raise
            print(f"[goto] {url}: ERR_NO_BUFFER_SPACE; retrying in {backoff_s:.0f}s", flush = True)
            sleep(backoff_s)
    raise AssertionError("unreachable")


def wait_for_first(locator: Any, *, timeout_ms: int = 10_000) -> Any | None:
    """Waits for a match: Locator.count() does not wait, so a render gap reads as a missing control."""
    # Lazy import: contract tests import this module on runners without playwright.
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


def echo_browser_errors(page: Any, info: Callable[[str], None]) -> list[str]:
    """Prints browser errors live and returns the uncaught ones, so a failure names its real cause."""
    thrown: list[str] = []

    def record(error: Any) -> None:
        thrown.append(str(error))
        info(f"pageerror: {error}")

    page.on("pageerror", record)
    page.on(
        "console",
        lambda m: info(f"console.{m.type}: {m.text}") if m.type == "error" else None,
    )
    page.on("requestfailed", lambda r: info(f"requestfailed: {r.url} {r.failure}"))
    page.on(
        "framenavigated",
        lambda f: info(f"navigated: {f.url}") if f is page.main_frame else None,
    )
    return thrown


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


# Lowercase: Playwright varies casing of these messages across versions.
_CONTEXT_LOST_MARKERS = (
    "execution context was destroyed",
    "context with specified id",
    "frame was detached",
    "target closed",
    "target page, context or browser has been closed",
    "execution context is not available",
)

_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


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


# page.evaluate has no timeout, so the fetch is bounded with an AbortController.
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
    """Default replays a context-lost fetch only for GET, HEAD and OPTIONS, never a mutating call"""
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
    # Retry only status==0 transport failures (stale keepalive); real HTTP and AbortError propagate.
    last: dict[str, Any] | None = None
    attempts = max(1, int(transport_retries) + 1)
    # Mutating methods may already have hit the backend, so only idempotent reads are replayed.
    if retry_on_context_loss is None:
        retry_on_context_loss = method.upper() in _IDEMPOTENT_METHODS
    ctx_retries = 2 if retry_on_context_loss else 0
    for attempt in range(attempts):
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


class _WallClockWatchdog:
    """A `deadline_s` budget that `kick()` restarts; never kicking is the absolute wall
    a `threading.Timer` gave. Timer has no reschedule, hence a thread over a live deadline.
    """

    def __init__(
        self,
        deadline_s: float,
        on_expiry: Callable[[], None],
        total_deadline_s: float | None = None,
    ) -> None:
        self._budget_s = float(deadline_s)
        self._on_expiry = on_expiry
        self._lock = threading.Lock()
        started = time.monotonic()
        # Kicks move the deadline; total_deadline_s is a ceiling no kick moves.
        self._ceiling = started + float(total_deadline_s) if total_deadline_s else None
        self.step_name: str | None = None
        self.step_budget_s: float | None = None
        self._step_started: float | None = None
        self._step_ceiling: float | None = None
        self._deadline = self._clamp(started + self._budget_s)
        self._cancelled = threading.Event()
        self._thread = threading.Thread(target = self._run, daemon = True)
        self.kicked = False

    def at_ceiling(self) -> bool:
        """Did the total cap, rather than the per-wait budget, decide the deadline?"""
        with self._lock:
            return self._ceiling is not None and self._deadline >= self._ceiling

    def at_step_ceiling(self) -> bool:
        """Did the running step's own budget, rather than inactivity, decide the deadline?"""
        with self._lock:
            return self._step_ceiling is not None and self._deadline >= self._step_ceiling

    def step_elapsed_s(self) -> float | None:
        """Seconds since the running step began, or None before the first begin_step()."""
        with self._lock:
            started = self._step_started
        return None if started is None else time.monotonic() - started

    def _clamp(self, deadline: float) -> float:
        for ceiling in (self._ceiling, self._step_ceiling):
            if ceiling is not None:
                deadline = min(deadline, ceiling)
        return deadline

    def start(self) -> "_WallClockWatchdog":
        self._thread.start()
        return self

    def kick(self) -> None:
        """Progress was made: restart the budget."""
        with self._lock:
            self.kicked = True
            self._deadline = self._clamp(time.monotonic() + self._budget_s)

    def begin_step(
        self,
        name: str,
        budget_s: float | None = None,
    ) -> None:
        """budget_s gives the step a ceiling kicks cannot extend; without it only inactivity applies."""
        with self._lock:
            now = time.monotonic()
            self.kicked = True
            self.step_name = name
            self._step_started = now
            self.step_budget_s = float(budget_s) if budget_s else None
            self._step_ceiling = now + self.step_budget_s if self.step_budget_s else None
            self._deadline = self._clamp(now + self._budget_s)

    def cancel(self) -> None:
        self._cancelled.set()

    def _run(self) -> None:
        while True:
            with self._lock:
                remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                if not self._cancelled.is_set():
                    self._on_expiry()
                return
            # Capped so a mid-sleep kick is seen.
            if self._cancelled.wait(min(remaining, 1.0)):
                return


# Exit 2 propagates through the workflow's `set -e`; keep deadline_s above the longest wait.
def install_wall_clock_watchdog(
    deadline_s: float,
    *,
    label: str = "playwright",
    info: Callable[[str], None] | None = None,
    total_deadline_s: float | None = None,
) -> _WallClockWatchdog:
    """Hard-exit `deadline_s` after the last `kick()`, or at `total_deadline_s` from arming
    if that comes first; returned so the caller can `.cancel()`."""

    def _kaboom() -> None:
        step = watchdog.step_name
        if total_deadline_s and watchdog.at_ceiling():
            spent = f"hit the {total_deadline_s:.0f}s total cap"
        elif step is not None and watchdog.at_step_ceiling():
            spent = f"step {step!r} used its whole {watchdog.step_budget_s:.0f}s budget"
        elif step is not None:
            spent = f"{deadline_s:.0f}s with no progress in step {step!r}"
        elif watchdog.kicked:
            spent = f"{deadline_s:.0f}s with no step reported"
        else:
            spent = f"hit {deadline_s:.0f}s wall-clock deadline"
        msg = (
            f"[{label}] WATCHDOG: {spent}; "
            f"forcing exit(2). The script wedged somewhere "
            f"the per-action timeouts could not bound. Inspect the "
            f"most recent step printed above to localise."
        )
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        try:
            import faulthandler
            faulthandler.dump_traceback(file = sys.stderr, all_threads = True)
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(2)

    # Bound before start(): at deadline_s <= 0 _kaboom can run during start().
    watchdog = _WallClockWatchdog(deadline_s, _kaboom, total_deadline_s)
    watchdog.start()
    if info is not None:
        info(f"watchdog armed: hard-exit {deadline_s:.0f}s after the last step")
    return watchdog


# Stretches every per-step budget at once for lanes slower than hosted Linux.
STEP_BUDGET_SCALE_ENV = "STUDIO_PW_STEP_BUDGET_SCALE"


def step_budget_s(seconds: float) -> float:
    """`seconds`, stretched by $STUDIO_PW_STEP_BUDGET_SCALE (default 1, never below 1)."""
    try:
        scale = float(os.environ.get(STEP_BUDGET_SCALE_ENV) or 1.0)
    except ValueError:
        scale = 1.0
    return float(seconds) * max(1.0, scale)


def report_failing_step(watchdog: _WallClockWatchdog, *, label: str = "playwright") -> None:
    """On an uncaught exception, prints one line naming the step it happened in, from begin_step()."""
    previous = sys.excepthook

    def _hook(exc_type, exc, tb) -> None:
        previous(exc_type, exc, tb)
        name = watchdog.step_name
        if name is None or issubclass(exc_type, KeyboardInterrupt):
            return
        elapsed = watchdog.step_elapsed_s() or 0.0
        first = (str(exc).strip().splitlines() or [""])[0][:300]
        try:
            sys.stderr.write(
                f"[{label}] FAIL in step {name!r} after {elapsed:.1f}s: "
                f"{exc_type.__name__}: {first}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass

    sys.excepthook = _hook


def wait_until(
    predicate: Callable[[], Any],
    *,
    timeout_s: float,
    what: str,
    interval_s: float = 0.1,
    page: Any = None,
) -> Any:
    """Pass page when the predicate reads event-filled state, so the pause is page.wait_for_timeout."""
    deadline = time.monotonic() + float(timeout_s)
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{what}: still not true after {timeout_s:g}s (last value {value!r})"
            )
        if page is not None:
            page.wait_for_timeout(interval_s * 1000.0)
        else:
            time.sleep(interval_s)


# "settled" = same box and no running animations for `frames` frames; "detached" = node replaced.
# The nonce keeps one call's count from carrying into the next.
_SETTLED_JS = """
([el, frames, nonce]) => {
    if (!el || !el.isConnected) return "detached";
    const r = el.getBoundingClientRect();
    const sig = [r.left, r.top, r.width, r.height].map((v) => Math.round(v * 10)).join(",");
    const running = typeof el.getAnimations === "function"
        && el.getAnimations({ subtree: true }).some((a) => a.playState === "running");
    if (!window.__pwSettled) window.__pwSettled = new WeakMap();
    const prev = window.__pwSettled.get(el);
    const count = !running && prev && prev.nonce === nonce && prev.sig === sig ? prev.count + 1 : 0;
    window.__pwSettled.set(el, { nonce, sig, count });
    return count >= frames ? "settled" : false;
}
"""
_settle_nonce = [0]


def wait_for_settled(
    locator: Any,
    *,
    frames: int = 3,
    timeout_ms: int = 10_000,
) -> None:
    """Waits until locator.first stops moving for several frames; a replaced node restarts the count."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    target = locator.first
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            raise PlaywrightTimeoutError(f"wait_for_settled: not settled within {timeout_ms}ms")
        _settle_nonce[0] += 1
        handle = target.element_handle(timeout = remaining_ms)
        try:
            outcome = target.page.wait_for_function(
                _SETTLED_JS,
                arg = [handle, int(frames), _settle_nonce[0]],
                polling = "raf",
                timeout = max(1, remaining_ms),
            ).json_value()
        finally:
            try:
                handle.dispose()
            except Exception:
                pass
        if outcome == "settled":
            return


def click_forced(
    locator: Any,
    *,
    timeout_ms: int = 5_000,
    **click_kwargs: Any,
) -> None:
    """Scrolls into view before a forced click, since force=True skips the scroll Playwright needs."""
    try:
        locator.scroll_into_view_if_needed(timeout = timeout_ms)
    except Exception:
        pass
    locator.click(force = True, **click_kwargs)
