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

   That state is asserted on a window this script opens itself, by holding the
   browser's /api/health at hardware_detecting=true, rather than on the real warm.
   The real one is a race nobody wins: see sample_natural_warm_window.

2. The desktop launcher's health watchdog killed the backend about a minute in
   ("Server stopped unexpectedly"). The warm thread holds the GIL through its
   C-extension imports, so probes time out while the process is perfectly alive.
   This polls the backend across the whole warm window and asserts it survives.

Runs against a live Studio; drives the real UI. Env contract matches the other
scripts here: BASE_URL, STUDIO_OLD_PW, PW_ART_DIR.
"""

import json
import os
import re
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
# STUDIO_OLD_PW is what the repo's own macOS workflow exports; STUDIO_PW is what the
# staging harness exports. Accept either so the same script runs under both.
OLD = os.environ.get("STUDIO_OLD_PW") or os.environ["STUDIO_PW"]
# What to rotate the bootstrap password to, when the app forces a change on first
# login. Only used on that path: a harness that already rotated over the API (the
# staging one) never reaches it. Must differ from OLD, or the change is rejected.
NEW = os.environ.get("STUDIO_NEW_PW") or f"{OLD}-Rotated1!"
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_mac_tabs"))
ART.mkdir(parents = True, exist_ok = True)

# The watchdog kills after 3 consecutive failures at a 15s interval, and the
# backend's startup grace is 300s. Outliving the grace is the whole point: a
# backend that dies at t+66s (the reported crash) fails here.
SURVIVAL_S = float(os.environ.get("STUDIO_MAC_SURVIVAL_S", "330"))
POLL_INTERVAL_S = float(os.environ.get("STUDIO_MAC_POLL_INTERVAL_S", "5"))
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "900"))
# How long the forced-verdict check gives the row to settle into its pending state.
FORCED_PENDING_S = float(os.environ.get("STUDIO_MAC_FORCED_PENDING_S", "15"))
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

# Routes that mean "not signed in". Landing on one invalidates every later assertion,
# so they are matched explicitly rather than folded into the generic redirect check.
_SIGNED_OUT_PATHS = ("/login", "/onboarding", "/change-password")

# Rows the sidebar pins inline by default, per SIDEBAR_NAV_DEFAULT_PINNED in
# studio/frontend/src/features/settings/stores/appearance-custom-store.ts. Only these
# carry a data-testid: the overflow rows render as MoreMenuItem inside a dropdown that
# mounts nothing until it is opened and passes no test id even when it is, so
# `[data-testid="nav-row-video"]` returns null on every host, every time.
#
# That matters for what this file can claim. The field report named Train AND Video, and
# the first version of this script sampled both -- but the Video half could never observe
# anything, so half of its evidence was structurally empty. Video is not lost coverage:
# both rows carry the one `pending: capabilitiesUnknown` flag and both resolve it through
# resolveNavRowState (studio/frontend/src/components/nav-row-state.ts), so Train is the
# observable end of the same wire and the More flyout is covered by the frontend unit
# tests instead.
INLINE_ROW_IDS = ("hub", "projects", "images", "train")
# The row every pending-state assertion below is pinned to.
GATED_ROW_ID = "train"
# Intercept pattern for the browser's health reads. Matches whether api-base.ts builds a
# relative path or an absolute one.
_HEALTH_ROUTE = "**/api/health"

_failed: list[str] = []
# Every nav row located anywhere in the tab walk. A walk that never finds the rows the
# sidebar pins by default proves nothing about the gating, so it is a failure rather
# than a stream of info lines.
_rows_seen: set[str] = set()


def signed_out(url: str) -> bool:
    return any(url.rstrip("/").endswith(p) or f"{p}?" in url for p in _SIGNED_OUT_PATHS)


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
                self.samples.append(
                    {
                        "t": round(time.monotonic(), 1),
                        "path": path,
                        "status": status,
                        "ms": round((time.monotonic() - began) * 1000, 1),
                        "hardware_detecting": (body or {}).get("hardware_detecting"),
                        # Stage 0 of the warm only sets hardware_detecting; this one stays
                        # lit through the transformers and datasets imports after it, so the
                        # pair is what tells a reader of the artifacts how wide the real
                        # provisional window on this host was.
                        "torch_warm_in_progress": (body or {}).get("torch_warm_in_progress"),
                    }
                )
            self.stop.wait(POLL_INTERVAL_S)

    def finish(self) -> None:
        self.stop.set()
        self.thread.join(timeout = 30)

    def report(self) -> None:
        (ART / "survival_samples.json").write_text(
            json.dumps(self.samples, indent = 1),
            encoding = "utf-8",
        )
        for path in ("/api/liveness", "/api/health"):
            got = [s for s in self.samples if s["path"] == path]
            if not got:
                fail(f"no samples collected for {path}")
                continue
            bad = [s for s in got if s["status"] != 200]
            worst = max(s["ms"] for s in got)
            unmeasured = sum(1 for s in got if s["hardware_detecting"] is True)
            warming = sum(1 for s in got if s["torch_warm_in_progress"] is True)
            info(
                f"{path}: {len(got)} samples, {len(bad)} non-200, worst {worst}ms, "
                f"{unmeasured} with an unmeasured verdict, {warming} with the warm still running"
            )
            if bad:
                fail(
                    f"{path} returned non-200 {len(bad)} time(s) "
                    f"(first at t={bad[0]['t']}s status={bad[0]['status']}); "
                    "the backend did not stay up through the warm window"
                )


def rotate_password(page) -> None:
    """Complete the forced password change a bootstrap login lands on.

    Studio seeds a one-time bootstrap password and requires it to be replaced before
    the app proper is reachable. A harness that rotates it over the API first (the
    staging one does) never sees this screen; a harness that hands over the raw
    bootstrap password (this repo's macOS smoke does) always does. Handling it here
    means the script works under both instead of only the one it was written against.

    The current-password box is rendered only when the page did NOT receive the
    bootstrap password (auth-form.tsx:362), so fill it when present rather than
    requiring it.
    """
    step("completing the forced password change")
    try:
        page.locator("#new-password").wait_for(state = "visible", timeout = 60000)
        current = page.locator("#current-password")
        if current.count() > 0:
            current.fill(OLD)
        page.locator("#new-password").fill(NEW)
        confirm = page.locator("#confirm-password")
        if confirm.count() > 0:
            confirm.fill(NEW)
        page.get_by_role("button", name = re.compile(r"^change password$", re.I)).first.click()
        page.wait_for_url(lambda url: not signed_out(url), timeout = 60000)
        info("password rotated")
    except Exception as exc:
        info(f"forced password change did not complete: {exc!r}")
        page.screenshot(path = str(ART / "change_password_failed.png"))


def log_in(page) -> bool:
    """Sign in and prove it took. Returns False if the app is still signed out.

    Three things about this form make the obvious version of this helper silently
    do nothing, and all three cost a green run that checked an empty shell:

    * auth-form.tsx returns null while the auth-status request is in flight, so at
      domcontentloaded there is no form in the DOM at all. A `count()` reads 0 and
      does not wait, which is exactly how this fell through to "assuming desktop
      auth" on the runner. Wait for the field instead.
    * Login mode renders a password and nothing else -- there is no username box
      (auth-form.tsx:329-342). Requiring one made the fill a no-op.
    * The submit button is labelled "Login", not "Sign in".
    """
    page.goto(BASE, wait_until = "domcontentloaded", timeout = 120000)
    # A backend that still has its one-time bootstrap password injects it into the page
    # and signs itself in, landing on /change-password with no login form ever rendered.
    # Check that BEFORE waiting on #password, or the wait burns 60s and reports "no
    # password field" for a session that is actually authenticated.
    try:
        page.wait_for_url(
            lambda url: "/change-password" in url or "/login" in url,
            timeout = 30000,
        )
    except Exception:
        pass
    if "/change-password" in page.url:
        rotate_password(page)
    try:
        # #password is the login field; the change-password screen uses
        # #current-password / #new-password, which we never want to touch here.
        # Only look for a login form when still on a signed-out route. After the
        # bootstrap rotation above we are already authenticated, and waiting a full
        # minute for a form that is correctly absent burns CI time and logs a
        # misleading "no password field".
        pw_box = page.locator("#password") if signed_out(page.url) else None
        if pw_box is not None:
            try:
                pw_box.wait_for(state = "visible", timeout = 60000)
            except Exception:
                # No form after a full minute is either a genuinely password-less
                # desktop build or a frontend that never rendered. The redirect check
                # below tells the two apart, so do not conclude anything here.
                info("no password field appeared within 60s")
                pw_box = None
        if pw_box is not None:
            pw_box.fill(OLD)
            submit = page.get_by_role("button", name = re.compile(r"^(login|sign in)$", re.I))
            submit.first.click()
            # Settle on the post-auth route rather than sleeping a fixed 3s.
            try:
                page.wait_for_url(
                    lambda url: not signed_out(url),
                    timeout = 60000,
                )
            except Exception:
                info(f"still on {page.url} 60s after submitting the login form")
            # A first login with the bootstrap password lands on /change-password
            # (session.ts:93 getPostAuthRoute -> mustChangePassword), which is a
            # signed-out route here because it has no sidebar to assert against.
            # The session is real, so finish the rotation instead of giving up.
            if "/change-password" in page.url:
                rotate_password(page)
    except Exception as exc:
        info(f"login form interaction raised {exc!r}")

    # Prove it rather than assume it: land somewhere authed and check we stayed.
    try:
        page.goto(f"{BASE}/chat", wait_until = "domcontentloaded", timeout = 60000)
        page.wait_for_timeout(1500)
    except Exception as exc:
        info(f"post-login navigation raised {exc!r}")
    if signed_out(page.url):
        info(f"still signed out after the login attempt (at {page.url})")
        page.screenshot(path = str(ART / "login_failed.png"))
        return False
    info("signed in")
    return True


_ROW_STATE_JS = """(ids) => {
    const out = {};
    for (const id of ids) {
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


def row_states(page, ids = INLINE_ROW_IDS) -> dict:
    """DOM state of each nav row by test id; None for a row that is not rendered."""
    return page.evaluate(_ROW_STATE_JS, list(ids)) or {}


def sample_natural_warm_window(page) -> None:
    """Watch the real warm close, and fail on a real grey-out. Observes nothing on most runs.

    Deliberately asserts nothing about having reached the window, because reaching it is
    not something this script can arrange. `hardware_detecting` is stage 0 of the warm --
    on the macOS runner's `--no-torch` install that is a failed `import torch` plus one
    failed importlib.metadata lookup, so the verdict settles inside a second of the port
    binding. Getting an authenticated sidebar in front of that costs a Chromium launch,
    a login and two navigations, one of which spends the frontend's own 5s bounded wait
    on the verdict (HARDWARE_DETECT_WAIT_MS in studio/frontend/src/config/env.ts). The
    window is normally shut before the first sample, on a slow host as much as a fast
    one, so requiring it -- under an env flag or otherwise -- would buy a permanently red
    job rather than a test. The guarantee lives in assert_pending_state_on_forced_verdict
    below; this stays for the case the window IS open, where it is the only check that
    sees the real backend and the real UI disagree.
    """
    step("sampling nav rows during the unmeasured window")
    deadline = time.monotonic() + 45
    samples = 0
    unmeasured_samples = 0
    row_samples = 0
    spinner_samples = 0
    violations: list[str] = []
    while time.monotonic() < deadline:
        try:
            state = row_states(page, (GATED_ROW_ID,))
        except Exception as exc:
            # Only fatal before anything was read: a page that cannot be evaluated at all
            # is the signed-out/unrendered shape, and breaking out of it quietly is how
            # this function used to report success on zero observations.
            if samples == 0:
                fail(f"could not read the sidebar during the unmeasured window ({exc!r})")
            else:
                info(f"row sampling stopped early ({exc!r})")
            break
        samples += 1
        unmeasured = (_get_json("/api/health")[1] or {}).get("hardware_detecting") is True
        unmeasured_samples += int(unmeasured)
        got = state.get(GATED_ROW_ID)
        if got and unmeasured:
            row_samples += 1
            spinner_samples += int(bool(got["spinner"]))
            if got["disabled"]:
                violations.append(
                    f"{GATED_ROW_ID} rendered disabled while /api/health still reported "
                    "hardware_detecting=true"
                )
        if not unmeasured:
            info("hardware detection settled; stopping the unmeasured sampling")
            break
        time.sleep(0.5)

    for v in sorted(set(violations)):
        fail(v)
    info(
        f"real warm window: {samples} sample(s), {unmeasured_samples} with an unmeasured "
        f"verdict, {row_samples} of those with the {GATED_ROW_ID} row rendered, "
        f"{spinner_samples} of those spinning"
    )


def assert_pending_state_on_forced_verdict(page) -> None:
    """Hold the verdict unmeasured for the browser and require the Train row to spin.

    This is the check that cannot pass having observed nothing, and it is the reason the
    one above does not have to. Instead of racing a sub-second warm, it answers the
    browser's /api/health with a real reply that has the measurement taken back out, so
    the provisional window stays open for as long as the check needs, on every host.

    What the field report described is then exactly what this reads: with the verdict
    unmeasured, `pending` beats `disabled` in resolveNavRowState
    (studio/frontend/src/components/nav-row-state.ts), so nav-row-train must render
    enabled with data-spinner="true". The regression rendered it disabled with no
    spinner, which fails here on any host, fast or slow, warm window or none.

    A missing row is a failure, not a skip: nav-row-train is pinned inline by default, so
    if it is not in the DOM the sidebar did not render and there is nothing to assert on.
    A stub body that drifted out of shape also fails loudly for the same reason, rather
    than quietly passing -- there is no path through here that reports success without
    having read the row.
    """
    step("forcing an unmeasured verdict and re-checking the pinned Train row")
    status, live = _get_json("/api/health")
    if status != 200 or not isinstance(live, dict):
        fail(
            "/api/health gave no body to base the provisional reply on "
            f"(status {status}); the forced pending-state check could not run"
        )
        return
    # A real reply with the measurement removed, so the only thing the browser sees
    # differently is the field under test. device_type is what env.ts reads as "measured",
    # and chat_only stays the conservative pre-detection default -- the exact pair a Mac
    # got on first paint in the field report, where the row blacked out.
    provisional = {
        k: v for k, v in live.items() if k not in ("device_type", "hardware_detection_deferred")
    }
    provisional["hardware_detecting"] = True
    provisional["chat_only"] = True
    body = json.dumps(provisional)

    def serve_provisional(route) -> None:
        route.fulfill(status = 200, content_type = "application/json", body = body)

    page.route(_HEALTH_ROUTE, serve_provisional)
    try:
        try:
            page.goto(f"{BASE}/chat", wait_until = "domcontentloaded", timeout = 60000)
            page.wait_for_selector(f'[data-testid="nav-row-{GATED_ROW_ID}"]', timeout = 30000)
        except Exception as exc:
            page.screenshot(path = str(ART / "forced_pending_missing_row.png"))
            fail(
                f"the {GATED_ROW_ID} nav row never rendered under an unmeasured verdict "
                f"({exc!r}); it is pinned inline by default, so either the sidebar did not "
                "come up or the row is gated on the verdict it is supposed to spin on"
            )
            return
        # The row derives its state synchronously from the store, but the store is filled
        # by the root route's beforeLoad, so give it frames rather than one read.
        deadline = time.monotonic() + FORCED_PENDING_S
        got = None
        while True:
            try:
                got = row_states(page, (GATED_ROW_ID,)).get(GATED_ROW_ID)
            except Exception as exc:
                # Raising out of here would skip the survival report and the exit code
                # main() is built around, so it lands as a failure like any other.
                fail(f"could not read the {GATED_ROW_ID} row under a forced verdict ({exc!r})")
                return
            if got and got["spinner"] and not got["disabled"]:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.25)
        page.screenshot(path = str(ART / "forced_pending.png"))
        if not got:
            fail(f"the {GATED_ROW_ID} nav row vanished between the wait and the read")
        elif got["disabled"]:
            fail(
                f"{GATED_ROW_ID} rendered disabled while /api/health reported "
                "hardware_detecting=true; this is the blacked-out row from the field report"
            )
        elif not got["spinner"]:
            fail(
                f"{GATED_ROW_ID} rendered with no pending spinner while /api/health "
                "reported hardware_detecting=true; an unmeasured capability has to read "
                "as 'still checking', not as a settled verdict"
            )
        else:
            info(f"{GATED_ROW_ID} spun on a forced unmeasured verdict, as it must")
    finally:
        page.unroute(_HEALTH_ROUTE, serve_provisional)


def assert_row_never_greyed_while_unmeasured(page) -> None:
    """A row whose verdict is unmeasured must spin, never grey out.

    Two passes over the same contract. The first rides the real warm and is silent when
    it misses it; the second creates the window it needs. Only the second can hold this
    file to its promise, so it runs unconditionally and on every host.
    """
    sample_natural_warm_window(page)
    assert_pending_state_on_forced_verdict(page)


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
        if signed_out(landed):
            # Never legitimate here: the session was proven signed in before the walk
            # started. Calling this an allowed redirect is exactly how a run that
            # authenticated with nobody goes green having exercised nothing.
            fail(f"{name}: bounced to the login page at {landed}; the session was lost mid-walk")
            continue
        if route not in landed:
            detecting = (_get_json("/api/health")[1] or {}).get("hardware_detecting")
            if detecting is True:
                fail(
                    f"{name}: redirected away from {route} while capabilities were still unmeasured"
                )
            else:
                info(f"{name}: redirected to {landed} after a measured verdict (allowed)")

        # Read every inline row, not just this tab's: they render together, so one read
        # records that the sidebar came up at all and keeps the per-route detail in the log.
        try:
            _rows_seen.update(rid for rid, got in row_states(page).items() if got)
        except Exception as exc:
            info(f"{name}: could not read the sidebar rows ({exc!r})")

        # Clicking the row is the interaction the user reported; a greyed-out row
        # swallows the click, so this doubles as a check that it is reachable.
        try:
            row = page.locator(f'[data-testid="nav-row-{row_id}"]')
            if row.count() > 0 and row.first.is_enabled():
                row.first.click(timeout = 10000)
                page.wait_for_timeout(1000)
            elif row.count() > 0:
                info(f"{name}: nav row present but disabled (measured verdict)")
            elif row_id in INLINE_ROW_IDS:
                # Not an info line: this row is pinned inline by default, so its absence
                # means the sidebar did not render and this tab checked nothing.
                fail(f"{name}: nav row {row_id} is pinned inline by default but did not render")
            else:
                # Expected and permanent for Video and Export: they live under "More",
                # which renders no test id. Reached by route instead.
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
        page.on(
            "pageerror",
            lambda e: None if is_benign_page_error(str(e)) else fail(f"page error: {e}"),
        )

        step("login")
        if not log_in(page):
            # Every assertion past this point reads the authenticated shell. Signed out,
            # the sidebar does not render, so the tab checks would find no rows, report
            # nothing, and the run would go green having tested none of what it claims.
            fail("could not sign in; the tab assertions below would all be vacuous")
            poller.finish()
            poller.report()
            return 1

        assert_row_never_greyed_while_unmeasured(page)
        drive_tabs(page)
        missing = [rid for rid in INLINE_ROW_IDS if rid not in _rows_seen]
        if missing:
            fail(
                f"the sidebar rows pinned by default never rendered on any route ({', '.join(missing)}); "
                "the tab gating was never actually exercised, so a green run here would mean nothing"
            )

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
