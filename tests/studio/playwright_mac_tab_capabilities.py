# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mac nav rows must spin, not grey out, while capability is unmeasured; backend must survive warm-up."""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    is_benign_page_error,
    robust_evaluate,
    wait_for_health,
)

BASE = os.environ["BASE_URL"]
OLD = os.environ.get("STUDIO_OLD_PW") or os.environ["STUDIO_PW"]
# Must differ from OLD, or the change is rejected.
NEW = os.environ.get("STUDIO_NEW_PW") or f"{OLD}-Rotated1!"
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_mac_tabs"))
ART.mkdir(parents = True, exist_ok = True)

# How long to poll the backend; the window must reach well past the reported crash at t+66s.
SURVIVAL_S = float(os.environ.get("STUDIO_MAC_SURVIVAL_S", "330"))
POLL_INTERVAL_S = float(os.environ.get("STUDIO_MAC_POLL_INTERVAL_S", "5"))
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "900"))
FORCED_PENDING_S = float(os.environ.get("STUDIO_MAC_FORCED_PENDING_S", "15"))
# Matches HEALTH_PROBE_TIMEOUT in studio/src-tauri/src/commands.rs.
PROBE_TIMEOUT_S = 10.0
# Clears the longest observed stall (33.2s). Extends observation, not a retry: a dead backend
# answers none of these probes, so it cannot turn a real death into a pass.
RECOVERY_WINDOW_S = 90.0
# A reset returns in ~1ms, so pace probes to avoid hammering a struggling backend.
RECOVERY_PROBE_SPACING_S = 2.0
_READ_CHUNK_BYTES = 65536
_MAX_BODY_BYTES = 1 << 20

LIVENESS_PATH = "/api/liveness"
HEALTH_PATH = "/api/health"
# An answer from either proves serving, matching check_health_inner.
PROBE_PATHS = (LIVENESS_PATH, HEALTH_PATH)
TABS = [
    ("/chat", "projects", "Chat"),
    ("/hub", "hub", "Hub"),
    ("/library", "library", "Library"),
    ("/images", "images", "Images"),
    ("/studio", "train", "Train"),
    ("/video", "video", "Video"),
    ("/export", "export", "Export"),
]

_SIGNED_OUT_PATHS = ("/login", "/change-password")

# Only rows pinned by default (SIDEBAR_NAV_DEFAULT_PINNED, appearance-custom-store.ts) carry a
# data-testid. test_inline_row_ids_match_the_frontends_default_pinned_set keeps this in sync.
INLINE_ROW_IDS = ("hub", "projects", "library", "images", "train")
# The Projects row yields to the Projects section once projects load (app-sidebar.tsx).
ROW_STAND_INS = {"projects": '[data-sidebar-section="projects"]'}
GATED_ROW_ID = "train"
_HEALTH_ROUTE = "**/api/health"

_failed: list[str] = []
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


def _transport_kind(err: object) -> str:
    """Only ECONNREFUSED proves nothing is bound; any other failure is a stalled listener, not a
    dead one."""
    if isinstance(err, ConnectionRefusedError):
        return "refused"
    return "timeout"


def _read_within(resp, deadline: float) -> str:
    """Bounds the whole body read by one deadline, since urllib's timeout resets on every chunk."""
    reader = getattr(resp, "read1", None) or resp.read
    chunks: list[bytes] = []
    total = 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("response body did not finish inside the probe budget")
        chunk = reader(_READ_CHUNK_BYTES)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_BODY_BYTES:
            raise TimeoutError("response body exceeded the probe's size cap")
    return b"".join(chunks).decode("utf-8", "replace")


def _probe_once(path: str, timeout: float) -> tuple[int, dict | None, str]:
    """Call only through _get_json; only ECONNREFUSED means the backend died."""
    deadline = time.monotonic() + timeout
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout = timeout) as resp:
            kind = "ok" if resp.status == 200 else "http"
            body = _read_within(resp, deadline)
            try:
                return resp.status, json.loads(body), kind
            except ValueError:
                return resp.status, None, kind
    except urllib.error.HTTPError as exc:
        return exc.code, None, "http"
    except urllib.error.URLError as exc:
        # Connect-time failures arrive wrapped, so the reason decides.
        return 0, None, _transport_kind(exc.reason)
    except Exception as exc:
        return 0, None, _transport_kind(exc)


def _get_json(path: str, timeout: float = PROBE_TIMEOUT_S) -> tuple[int, dict | None, str]:
    """GET *path* under a WHOLE-REQUEST deadline, returning (status, body, kind).

    *timeout* bounds the entire probe: DNS, connect, response headers and body. It is
    not a per-socket-operation timeout, and it must not be turned back into one.

    That distinction is the whole reason this wrapper exists. urllib's own timeout
    applies to each socket operation separately, so any peer that keeps sending
    something, anything, more often than the timeout holds the call open forever. Each
    layer was bounded in turn and the hole simply moved: capping the body read left
    urlopen able to block indefinitely while response HEADERS trickled, because urlopen
    has not returned yet at that point and the body deadline never gets to run. Bounding
    the next layer down would only move it again, to the redirect chain or the TLS
    handshake. A deadline outside all of them cannot be outflanked by any of them.

    The probe therefore runs on a daemon thread and this joins it for at most *timeout*.
    A join that expires is a timeout, and the thread is abandoned rather than waited on:
    it is a daemon, so it cannot hold up interpreter exit, and _probe_once carries its
    own body deadline and size cap so an abandoned one still lets go of its socket
    instead of buffering forever. Those inner bounds are hygiene for the abandoned case;
    the join is what actually enforces the budget.

    Abandoning a thread per hung probe is affordable here because a backend that hangs
    probes is one this script is about to report on and exit.
    """
    outcome: list[tuple[int, dict | None, str]] = []

    def attempt() -> None:
        outcome.append(_probe_once(path, timeout))

    worker = threading.Thread(target = attempt, name = f"probe-{path}", daemon = True)
    worker.start()
    worker.join(timeout)
    if outcome:
        return outcome[0]
    return 0, None, "timeout"


def await_recovery(
    window_s: float = RECOVERY_WINDOW_S, spacing_s: float = RECOVERY_PROBE_SPACING_S
) -> tuple[str, int, float]:
    """Keeps probing after sampling stops; an answer of any status or a refused port ends the watch."""
    began = time.monotonic()
    probes: list[dict] = []
    status, kind = 0, "timeout"
    while True:
        remaining = window_s - (time.monotonic() - began)
        if probes and remaining <= 0:
            # No time left: starting a full-budget probe here would overrun the window.
            break
        probe_began = time.monotonic()
        status, _, kind = _get_json(LIVENESS_PATH, timeout = min(PROBE_TIMEOUT_S, remaining))
        # Same shape and clock as the poller's samples, so stalls after sampling stops are visible.
        probes.append(
            {
                "t": round(time.monotonic(), 1),
                "path": LIVENESS_PATH,
                "status": status,
                "kind": kind,
                "ms": round((time.monotonic() - probe_began) * 1000, 1),
                "inference_active": None,
                "hardware_detecting": None,
                "torch_warm_in_progress": None,
            }
        )
        if kind != "timeout":
            break
        elapsed = time.monotonic() - began
        if elapsed >= window_s:
            break
        # An instant failure was a reset, not silence; pace the next probe.
        idle = spacing_s - (time.monotonic() - probe_began)
        if idle > 0:
            time.sleep(min(idle, window_s - elapsed))
    return kind, status, round(time.monotonic() - began, 1), probes


def _stall_windows(samples: list[dict]) -> list[tuple[float, float, bool]]:
    """Gaps with no answer on either route; a gap closes at the issue time of the answering probe."""
    ordered = sorted(samples, key = lambda s: s["t"])
    spans: list[tuple[float, float, bool]] = []
    open_start = None
    for s in ordered:
        began = s["t"] - s["ms"] / 1000.0
        if s["kind"] == "ok":
            if open_start is not None:
                spans.append((open_start, began, False))
                open_start = None
        elif open_start is None:
            open_start = began
    if open_start is not None:
        spans.append((open_start, ordered[-1]["t"], True))
    return spans


class BackendSurvivalPoller:
    """Polls /api/liveness and /api/health on a daemon thread while the UI drive loads the backend."""

    def __init__(self) -> None:
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target = self._run, name = "survival-poll", daemon = True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            for path in PROBE_PATHS:
                began = time.monotonic()
                status, body, kind = _get_json(path)
                self.samples.append(
                    {
                        "t": round(time.monotonic(), 1),
                        "path": path,
                        "status": status,
                        "kind": kind,
                        "ms": round((time.monotonic() - began) * 1000, 1),
                        "inference_active": (body or {}).get("inference_active"),
                        "hardware_detecting": (body or {}).get("hardware_detecting"),
                        "torch_warm_in_progress": (body or {}).get("torch_warm_in_progress"),
                    }
                )
            self.stop.wait(POLL_INTERVAL_S)

    def finish(self) -> None:
        self.stop.set()
        self.thread.join(timeout = 30)

    def report(
        self,
        final_kind: str = "ok",
        final_status: int = 200,
        final_wait_s: float = 0.0,
        recovery_samples: "list[dict] | tuple" = (),
    ) -> None:
        """Includes the recovery probes, so a stall that starts after sampling stops is still reported."""
        # Written below with the recovery probes folded in, or the artifact hides a reported stall.
        for path in PROBE_PATHS:
            got = [s for s in self.samples if s["path"] == path]
            if not got:
                fail(f"no samples collected for {path}")
                continue
            bad = [s for s in got if s["kind"] != "ok"]
            worst = max(s["ms"] for s in got)
            unmeasured = sum(1 for s in got if s["hardware_detecting"] is True)
            warming = sum(1 for s in got if s["torch_warm_in_progress"] is True)
            info(
                f"{path}: {len(got)} samples, {len(bad)} miss(es), worst {worst}ms, "
                f"{unmeasured} with an unmeasured verdict, {warming} with the warm still running"
            )
            # Refused connections and HTTP errors are not stalls; they fail on sight.
            refused = [s for s in got if s["kind"] == "refused"]
            answered_badly = [s for s in got if s["kind"] == "http"]
            if refused:
                fail(
                    f"{path}: connection refused at t={refused[0]['t']}s; the port was gone, "
                    "so the backend did not stay up through the warm window"
                )
            elif answered_badly:
                fail(
                    f"{path}: answered {answered_badly[0]['status']} at "
                    f"t={answered_badly[0]['t']}s; the backend stayed up but reported itself "
                    "unhealthy through the warm window"
                )

        # Deliberately not a replay of the launcher watchdog, which is not running here and is covered by
        # the Rust tests in commands.rs. Only a backend that never answers again fails.
        observed = list(self.samples) + list(recovery_samples)
        (ART / "survival_samples.json").write_text(
            json.dumps(observed, indent = 1),
            encoding = "utf-8",
        )
        sampling_ended = max((s["t"] for s in self.samples), default = 0.0)
        spans = _stall_windows(observed)
        terminal = next((sp for sp in spans if sp[2]), None)
        longest = max(((end - start) for start, end, _ in spans), default = 0.0)
        widest = max(spans, key = lambda sp: sp[1] - sp[0], default = None)

        if final_kind == "refused":
            fail(
                f"{LIVENESS_PATH} was refused after the run ({final_wait_s}s of watching); "
                "the port is gone, so the backend did not survive the window"
            )
        elif final_kind == "http":
            fail(
                f"{LIVENESS_PATH} answered {final_status} after the run; the backend is up "
                "but reporting itself unhealthy"
            )
        elif final_kind == "timeout":
            if terminal is not None:
                # terminal already spans the recovery probes; do not count the watch twice.
                fail(
                    f"the backend stopped answering at t={round(terminal[0], 1)}s and never "
                    f"answered again: {round(terminal[1] - terminal[0], 1)}s of silence in "
                    f"total, of which the last {final_wait_s}s was the post-run watch. It "
                    "did not survive the window."
                )
            else:
                fail(
                    f"backend answered nothing for {final_wait_s}s after the run "
                    f"({LIVENESS_PATH} kept timing out), so it did not survive the window"
                )
        elif spans:
            # Recovered, so green, but a stall this long is still a defect worth reporting.
            worst_ms = max(s["ms"] for s in observed)
            if widest is None or widest[1] <= sampling_ended:
                cleared = "It answered again before the run ended."
            elif widest[0] >= sampling_ended:
                cleared = (
                    "That stall began after sampling ended and was seen only by the "
                    f"post-run watch, which ran for {final_wait_s}s before it cleared."
                )
            else:
                cleared = (
                    "Sampling ended during that stall; it cleared during the post-run "
                    f"watch, which ran for {final_wait_s}s. The length above spans both."
                )
            print(
                f"::warning::backend stalled: {len(spans)} window(s) with nothing answering, "
                f"longest {round(longest, 1)}s, worst single probe {worst_ms}ms against a "
                f"{PROBE_TIMEOUT_S}s budget. {cleared} Not a failure here. See "
                "logs/studio_tabs.log for which request was in flight.",
                flush = True,
            )


def rotate_password(page) -> None:
    """The current-password box only renders without the bootstrap password, so fill it when present."""
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
    """Auth form is absent until auth-status returns, so wait for the password field; a count() reads 0."""
    page.goto(BASE, wait_until = "domcontentloaded", timeout = 120000)
    # A backend with its bootstrap password signs itself in to /change-password; check before #password.
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
        # Only look for the login form on a signed-out route.
        pw_box = page.locator("#password") if signed_out(page.url) else None
        if pw_box is not None:
            try:
                pw_box.wait_for(state = "visible", timeout = 60000)
            except Exception:
                info("no password field appeared within 60s")
                pw_box = None
        if pw_box is not None:
            pw_box.fill(OLD)
            submit = page.get_by_role("button", name = re.compile(r"^(login|sign in)$", re.I))
            submit.first.click()
            try:
                page.wait_for_url(
                    lambda url: not signed_out(url),
                    timeout = 60000,
                )
            except Exception:
                info(f"still on {page.url} 60s after submitting the login form")
            # The bootstrap login lands on /change-password (getPostAuthRoute); finish the rotation.
            if "/change-password" in page.url:
                rotate_password(page)
    except Exception as exc:
        info(f"login form interaction raised {exc!r}")

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
    """Reads via robust_evaluate, since a password-rotation navigation can destroy the execution context."""
    return robust_evaluate(page, _ROW_STATE_JS, list(ids)) or {}


def sample_natural_warm_window(page) -> None:
    """Samples nav rows in the real warm window if it is still open; asserts nothing about reaching it."""
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
            # Fatal only before anything was read, or zero observations would report success.
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
    """Forces an unmeasured verdict; the Train row must spin, because pending beats disabled."""
    step("forcing an unmeasured verdict and re-checking the pinned Train row")
    status, live, _kind = _get_json("/api/health")
    if status != 200 or not isinstance(live, dict):
        fail(
            "/api/health gave no body to base the provisional reply on "
            f"(status {status}); the forced pending-state check could not run"
        )
        return
    # device_type is what env.ts reads as "measured"; chat_only stays the pre-detection default.
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
        # The store is filled by the root route's beforeLoad, so poll over frames.
        deadline = time.monotonic() + FORCED_PENDING_S
        got = None
        while True:
            try:
                got = row_states(page, (GATED_ROW_ID,)).get(GATED_ROW_ID)
            except Exception as exc:
                # Raising would skip the survival report and exit code main() is built around.
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
    """A row whose verdict is unmeasured must spin, never grey out; the forced pass runs on every host."""
    sample_natural_warm_window(page)
    assert_pending_state_on_forced_verdict(page)


def stand_in_shown(page, row_id: str) -> bool:
    """Checks visibility, not mounting: a collapsed rail keeps stand-ins in the DOM hidden by CSS."""
    selector = ROW_STAND_INS.get(row_id)
    if selector is None:
        return False
    stand_in = page.locator(selector)
    return stand_in.count() > 0 and stand_in.first.is_visible()


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
        # The route guard may bounce Train/Video on an incapable host, but never while the verdict is unknown.
        if signed_out(landed):
            # Never legitimate: the session was proven signed in before the walk.
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

        try:
            _rows_seen.update(rid for rid, got in row_states(page).items() if got)
            _rows_seen.update(rid for rid in ROW_STAND_INS if stand_in_shown(page, rid))
        except Exception as exc:
            info(f"{name}: could not read the sidebar rows ({exc!r})")

        # A greyed-out row swallows the click, so this also checks reachability.
        try:
            row = page.locator(f'[data-testid="nav-row-{row_id}"]')
            if row.count() > 0 and row.first.is_enabled():
                row.first.click(timeout = 10000)
                page.wait_for_timeout(1000)
            elif row.count() > 0:
                info(f"{name}: nav row present but disabled (measured verdict)")
            elif stand_in_shown(page, row_id):
                info(
                    f"{name}: nav row {row_id} stands down while its section shows; reached by route instead"
                )
            elif row_id in INLINE_ROW_IDS:
                fail(f"{name}: nav row {row_id} is pinned inline by default but did not render")
            else:
                # Video and Export live under "More", which renders no test id.
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

        remaining = SURVIVAL_S - (time.monotonic() - began)
        if remaining > 0:
            step(f"holding the session for {remaining:.0f}s more to outlive the watchdog grace")
            while remaining > 0:
                page.wait_for_timeout(min(15000, int(remaining * 1000)))
                page.goto(f"{BASE}/chat", wait_until = "domcontentloaded", timeout = 60000)
                remaining = SURVIVAL_S - (time.monotonic() - began)

        page.screenshot(path = str(ART / "final.png"))
        ctx.close()
        browser.close()

    poller.finish()

    step(f"watching up to {RECOVERY_WINDOW_S:.0f}s more for the backend to answer")
    kind, status, waited, recovery = await_recovery()
    info(f"post-run {LIVENESS_PATH}: {kind} after {waited}s of watching, {len(recovery)} probe(s)")
    poller.report(
        final_kind = kind,
        final_status = status,
        final_wait_s = waited,
        recovery_samples = recovery,
    )

    # Cancelled only after the recovery watch, so WALL_TIMEOUT_S still bounds it.
    watchdog.cancel()

    if _failed:
        print(f"[mac-tabs] {len(_failed)} FAILURE(S)", flush = True)
        for m in _failed:
            print(f"[mac-tabs]   - {m}", flush = True)
        return 1
    print("[mac-tabs] PASS", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
