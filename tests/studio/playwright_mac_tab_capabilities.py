# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth macOS tab-capability Playwright test.

Covers the two field failures from Unsloth Desktop 0.1.524-beta on Apple Silicon:

1. The Train and Video sidebar rows rendered blacked out for minutes after launch,
   then came back. The store seeded chat-only from a browser-UA guess, so on a Mac
   both rows greyed out on first paint and stayed that way until /api/health
   answered -- which, after the startup-speedup work moved the ML imports onto a
   background warm thread, can take a long time. A row whose capability is still
   unmeasured must spin, not grey out.

2. The desktop launcher's health watchdog killed the backend about a minute in
   ("Server stopped unexpectedly"). The warm thread holds the GIL through its
   C-extension imports, so probes time out while the process is perfectly alive.
   This polls the backend across the whole warm window and asserts it survives.

Runs against a live Studio; drives the real UI. Env contract matches the other
scripts here: BASE_URL, STUDIO_OLD_PW, PW_ART_DIR.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

# Run as a plain script (not via pytest), so prepend the dir to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    is_benign_page_error,
    wait_for_health,
)

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_mac_tabs"))
ART.mkdir(parents = True, exist_ok = True)

# The watchdog kills after 3 consecutive failures at a 15s interval, and the
# backend's startup grace is 300s. Outliving the grace is the whole point: a
# backend that dies at t+66s (the reported crash) fails here.
SURVIVAL_S = float(os.environ.get("STUDIO_MAC_SURVIVAL_S", "330"))
POLL_INTERVAL_S = float(os.environ.get("STUDIO_MAC_POLL_INTERVAL_S", "5"))
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "900"))
# Every tab the user reported interacting with, plus the ones that share the
# chat-only gate. (route, nav row id, human name).
TABS = [
    ("/chat", "projects", "Chat"),
    ("/hub", "hub", "Hub"),
    ("/images", "images", "Images"),
    ("/studio", "train", "Train"),
    ("/video", "video", "Video"),
    ("/export", "export", "Export"),
]

_failed: list[str] = []


def info(s: str) -> None:
    print(f"[mac-tabs] {s}", flush = True)


def step(s: str) -> None:
    print(f"[mac-tabs] STEP {s}", flush = True)


def fail(m: str) -> None:
    print(f"[mac-tabs] FAIL: {m}", flush = True)
    _failed.append(m)


def _get_json(path: str, timeout: float = 10.0) -> tuple[int, dict | None]:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout = timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(body)
            except ValueError:
                return resp.status, None
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        return 0, None


class BackendSurvivalPoller:
    """Poll /api/liveness and /api/health for the whole run, on a daemon thread.

    Records every non-200 and the worst latency seen. The UI drive happens
    concurrently, so this measures the backend under the same load the desktop
    launcher's watchdog would be probing it under.
    """

    def __init__(self) -> None:
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target = self._run, name = "survival-poll", daemon = True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            for path in ("/api/liveness", "/api/health"):
                began = time.monotonic()
                status, body = _get_json(path)
                self.samples.append({
                    "t": round(time.monotonic(), 1),
                    "path": path,
                    "status": status,
                    "ms": round((time.monotonic() - began) * 1000, 1),
                    "hardware_detecting": (body or {}).get("hardware_detecting"),
                })
            self.stop.wait(POLL_INTERVAL_S)

    def finish(self) -> None:
        self.stop.set()
        self.thread.join(timeout = 30)

    def report(self) -> None:
        (ART / "survival_samples.json").write_text(
            json.dumps(self.samples, indent = 1), encoding = "utf-8",
        )
        for path in ("/api/liveness", "/api/health"):
            got = [s for s in self.samples if s["path"] == path]
            if not got:
                fail(f"no samples collected for {path}")
                continue
            bad = [s for s in got if s["status"] != 200]
            worst = max(s["ms"] for s in got)
            info(f"{path}: {len(got)} samples, {len(bad)} non-200, worst {worst}ms")
            if bad:
                fail(
                    f"{path} returned non-200 {len(bad)} time(s) "
                    f"(first at t={bad[0]['t']}s status={bad[0]['status']}); "
                    "the backend did not stay up through the warm window"
                )


def assert_row_never_greyed_while_unmeasured(page) -> None:
    """A row whose verdict is unmeasured must spin, never grey out.

    Sampled from first paint, because the regression was visible on the very first
    frame: the UA seed greyed Train and Video out before any measurement existed.
    """
    step("sampling nav rows during the unmeasured window")
    deadline = time.monotonic() + 45
    seen_spinner = {"train": False, "video": False}
    violations: list[str] = []
    while time.monotonic() < deadline:
        try:
            state = page.evaluate(
                """() => {
                    const out = {};
                    for (const id of ["train", "video"]) {
                        const el = document.querySelector(`[data-testid="nav-row-${id}"]`);
                        if (!el) { out[id] = null; continue; }
                        out[id] = {
                            disabled: el.hasAttribute("disabled")
                                || el.getAttribute("aria-disabled") === "true",
                            spinner: el.getAttribute("data-spinner") === "true",
                        };
                    }
                    return out;
                }"""
            )
        except Exception:
            break
        measured = _get_json("/api/health")[1] or {}
        unmeasured = measured.get("hardware_detecting") is True
        for row_id, got in (state or {}).items():
            if not got:
                continue
            if got["spinner"]:
                seen_spinner[row_id] = True
            if unmeasured and got["disabled"]:
                violations.append(
                    f"{row_id} rendered disabled while /api/health still reported "
                    "hardware_detecting=true"
                )
        if not unmeasured:
            info("hardware detection settled; stopping the unmeasured sampling")
            break
        time.sleep(0.5)

    for v in sorted(set(violations)):
        fail(v)
    info(f"spinner observed during warm: {seen_spinner}")


def drive_tabs(page) -> None:
    for route, row_id, name in TABS:
        step(f"open {name} ({route})")
        try:
            page.goto(f"{BASE}{route}", wait_until = "domcontentloaded", timeout = 60000)
            page.wait_for_timeout(1500)
        except Exception as exc:
            fail(f"navigating to {route} raised {exc!r}")
            continue

        landed = page.url
        # The chat-only route guard may legitimately bounce Train/Video on a host
        # without the capability. What it must not do is bounce while the verdict
        # is still unknown, which is the race this run is here to catch.
        if route not in landed:
            detecting = (_get_json("/api/health")[1] or {}).get("hardware_detecting")
            if detecting is True:
                fail(f"{name}: redirected away from {route} while capabilities were still unmeasured")
            else:
                info(f"{name}: redirected to {landed} after a measured verdict (allowed)")

        # Clicking the row is the interaction the user reported; a greyed-out row
        # swallows the click, so this doubles as a check that it is reachable.
        try:
            row = page.locator(f'[data-testid="nav-row-{row_id}"]')
            if row.count() > 0 and row.first.is_enabled():
                row.first.click(timeout = 10000)
                page.wait_for_timeout(1000)
            elif row.count() > 0:
                info(f"{name}: nav row present but disabled (measured verdict)")
            else:
                info(f"{name}: nav row not pinned inline; reached by route instead")
        except Exception as exc:
            info(f"{name}: row click did not land ({exc!r})")

        page.screenshot(path = str(ART / f"tab_{row_id}.png"), full_page = False)


def main() -> int:
    step("waiting for the backend to answer")
    if not wait_for_health(BASE, timeout = 600):
        fail("backend never answered /api/health")
        return 1

    poller = BackendSurvivalPoller()
    poller.start()
    began = time.monotonic()
    watchdog = install_wall_clock_watchdog(WALL_TIMEOUT_S, label = "mac-tabs", info = info)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args = chromium_launch_args(sys.platform))
        ctx = browser.new_context(viewport = {"width": 1440, "height": 900})
        install_view_transition_killer(ctx)
        page = ctx.new_page()
        page.on("pageerror", lambda e: None if is_benign_page_error(str(e)) else fail(f"page error: {e}"))

        step("login")
        page.goto(BASE, wait_until = "domcontentloaded", timeout = 120000)
        try:
            pw_box = page.get_by_label("Password", exact = False)
            if pw_box.count() > 0:
                pw_box.first.fill(OLD)
                page.get_by_role("button", name = "Sign in").first.click()
                page.wait_for_timeout(3000)
        except Exception as exc:
            info(f"login form not presented ({exc!r}); assuming desktop auth")

        assert_row_never_greyed_while_unmeasured(page)
        drive_tabs(page)

        # Hold the session open until the survival window is covered. The reported
        # crash landed at t+66s, well inside this.
        remaining = SURVIVAL_S - (time.monotonic() - began)
        if remaining > 0:
            step(f"holding the session for {remaining:.0f}s more to outlive the watchdog grace")
            while remaining > 0:
                page.wait_for_timeout(min(15000, int(remaining * 1000)))
                # Keep the UI doing real work so the backend is genuinely serving.
                page.goto(f"{BASE}/chat", wait_until = "domcontentloaded", timeout = 60000)
                remaining = SURVIVAL_S - (time.monotonic() - began)

        page.screenshot(path = str(ART / "final.png"))
        ctx.close()
        browser.close()

    watchdog.cancel()
    poller.finish()
    poller.report()

    status, _ = _get_json("/api/liveness")
    if status != 200:
        fail(f"backend was not alive at the end of the run (/api/liveness -> {status})")

    if _failed:
        print(f"[mac-tabs] {len(_failed)} FAILURE(S)", flush = True)
        for m in _failed:
            print(f"[mac-tabs]   - {m}", flush = True)
        return 1
    print("[mac-tabs] PASS", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
