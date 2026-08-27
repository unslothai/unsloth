# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser launch. Reuses `tests/studio/_playwright_robust.py` where it is available.

Playwright is imported LAZILY, inside the functions that need it, so `--help` and `--doctor` work
on a machine with nothing installed -- which is the first thing an external tester runs and the
worst possible moment for an ImportError.

`_playwright_robust` is imported the same way and its absence is survivable: the shipped zipapp
carries a copy of the pieces it needs, but a checkout run gets the real module and therefore the
real, maintained Chromium flags, the view-transition killer, the wall-clock watchdog and
`dump_diagnostics`.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


# The engine matching the tester's desktop webview family. Unsloth ships as a Tauri app, so the
# thing a user actually looks at is a system webview, not the browser they happen to have.
#   Windows -> WebView2, which is Chromium: `channel="msedge"` is the closest available.
#   macOS   -> WKWebView, which is WebKit.
#   Linux   -> WebKitGTK. Playwright's `webkit` is NOT WebKitGTK; it is a different embedding of
#              the same engine, so it is labelled A PROXY and never as the real thing.
def default_engine() -> tuple[str, dict, str]:
    system = platform.system()
    if system == "Windows":
        return "chromium", {"channel": "msedge"}, "Edge/WebView2, the Windows desktop webview"
    if system == "Darwin":
        return "webkit", {}, "WebKit, the engine behind macOS WKWebView"
    return (
        "webkit",
        {},
        (
            "WebKit, A PROXY FOR WebKitGTK: Playwright's webkit is a different "
            "embedding of the same engine, not the GTK one Unsloth runs in on Linux"
        ),
    )


def _robust() -> Any:
    """`tests.studio._playwright_robust`, or None when running from the zipapp."""
    for name in ("tests.studio._playwright_robust", "_playwright_robust"):
        try:
            return importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
    return None


_FALLBACK_CHROMIUM_ARGS = [
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-features=TranslateUI",
    "--disable-ipc-flooding-protection",
]


@dataclass
class BrowserBundle:
    playwright: Any
    browser: Any
    context: Any
    page: Any
    cdp: Any
    engine: str
    engine_note: str
    robust: Any

    def close(self) -> None:
        for closer in (
            lambda: self.context.close(),
            lambda: self.browser.close(),
            lambda: self.playwright.stop(),
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass


def launch(
    engine: Optional[str] = None,
    *,
    headless: bool = True,
    init_scripts: Optional[list[str]] = None,
    viewport: tuple[int, int] = (1440, 960),
    log: Callable[[str], None] = print,
) -> BrowserBundle:
    from playwright.sync_api import sync_playwright  # lazy, see the module docstring

    chosen, launch_kwargs, note = default_engine()
    if engine:
        chosen, launch_kwargs = (
            engine,
            ({} if engine != "chromium" or chosen != "chromium" else launch_kwargs),
        )
        note = f"{engine}, chosen explicitly"
    robust = _robust()
    args = robust.chromium_launch_args() if robust is not None else _FALLBACK_CHROMIUM_ARGS

    pw = sync_playwright().start()
    factory = getattr(pw, chosen)
    if chosen == "chromium":
        browser = factory.launch(headless = headless, args = args, **launch_kwargs)
    else:
        browser = factory.launch(headless = headless)
    context = browser.new_context(
        viewport = {"width": viewport[0], "height": viewport[1]},
        # Clipboard read is what proves the Copy action actually copied. Without it the action
        # still runs and the assertion reports that it could not be proved, which is the honest
        # outcome rather than a pass.
        permissions = ["clipboard-read", "clipboard-write"] if chosen == "chromium" else None,
    )
    if robust is not None:
        try:
            robust.install_view_transition_killer(context)
        except Exception:  # noqa: BLE001
            pass
    for script in init_scripts or []:
        context.add_init_script(script)
    page = context.new_page()
    # A GLOBAL CEILING on Playwright's actionability waits. The default is 30 seconds, which is
    # longer than most of this suite's slot budgets: one action waiting out the default on an
    # element that is present but not visible overran a 9-second slot by more than three times,
    # and the scheduler cannot intervene because the action still holds its own window. Actions
    # that legitimately need longer pass an explicit timeout.
    page.set_default_timeout(8000)
    cdp = None
    if chosen == "chromium":
        try:
            cdp = context.new_cdp_session(page)
            cdp.send("Performance.enable")
        except Exception:  # noqa: BLE001
            cdp = None
    log(f"  browser: {chosen} ({note}){'' if cdp else ', no CDP session'}")
    return BrowserBundle(
        playwright = pw,
        browser = browser,
        context = context,
        page = page,
        cdp = cdp,
        engine = chosen,
        engine_note = note,
        robust = robust,
    )


def cdp_metrics(cdp) -> dict:
    if cdp is None:
        return {}
    try:
        got = cdp.send("Performance.getMetrics")
    except Exception:  # noqa: BLE001
        return {}
    return {m["name"]: m["value"] for m in got["metrics"]}


def cdp_counters(before: dict, after: dict) -> dict:
    """CHROMIUM-ONLY, and every consumer prints `-` off Chromium rather than a zero.

    `LayoutDuration` is the direct read on M3: the autoscroll observer's callback synchronously
    reads scrollHeight, so forced layout per streamed character shows up here and nowhere else.
    An earlier harness read it as a FLAT FLOOR that barely moved with thread size, which is
    exactly what it would look like if the observer never ran -- which, on a fixture with a local
    adapter and no real DOM mutations, it did not.
    """
    if not before or not after:
        return {
            "layout_count": None,
            "recalc_style_count": None,
            "layout_ms": None,
            "recalc_style_ms": None,
            "task_ms": None,
            "cdp_attempted": False,
        }

    def d(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)

    return {
        "layout_count": round(d("LayoutCount"), 1),
        "recalc_style_count": round(d("RecalcStyleCount"), 1),
        "layout_ms": round(d("LayoutDuration") * 1000, 1),
        "recalc_style_ms": round(d("RecalcStyleDuration") * 1000, 1),
        "task_ms": round(d("TaskDuration") * 1000, 1),
        "script_ms": round(d("ScriptDuration") * 1000, 1),
        "cdp_attempted": True,
    }


def find_python_root_pid() -> int:
    return __import__("os").getpid()


def dump_diagnostics(
    page,
    out_dir: Path,
    label: str,
    log: Callable[[str], None] = print,
) -> None:
    robust = _robust()
    if robust is not None and hasattr(robust, "dump_diagnostics"):
        try:
            robust.dump_diagnostics(page, out_dir, label)
            return
        except Exception:  # noqa: BLE001
            pass
    try:
        out_dir.mkdir(parents = True, exist_ok = True)
        page.screenshot(path = str(out_dir / f"{label}.png"), full_page = False)
        (out_dir / f"{label}.html").write_text(page.content(), encoding = "utf-8")
    except Exception as exc:  # noqa: BLE001
        log(f"  diagnostics for {label} could not be written: {exc}")


def install_wall_clock_watchdog(
    deadline_s: float,
    label: str = "studiobench",
    log: Callable[[str], None] = print,
):
    robust = _robust()
    if robust is not None and hasattr(robust, "install_wall_clock_watchdog"):
        return robust.install_wall_clock_watchdog(deadline_s, label = label, info = log)
    import os
    import threading

    def _kaboom() -> None:
        sys.stderr.write(f"[{label}] WATCHDOG: hit {deadline_s:.0f}s wall clock; exit(2)\n")
        sys.stderr.flush()
        os._exit(2)

    timer = threading.Timer(deadline_s, _kaboom)
    timer.daemon = True
    timer.start()
    log(f"  watchdog armed: hard exit at {deadline_s:.0f}s")
    return timer
