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

   "Survives" here means the backend answered again, not that it answered every
   probe. A probe that times out while the process is alive is the symptom this
   file was written about, so failing on one states the opposite of the thing it
   is trying to prove. Run 32862298967 went red on one timed-out probe per route
   out of eighteen, on a commit that changed one frontend unit test. The stall
   behind it was real -- the server served no request at all for 10.0s and then
   33.2s, both windows ending on an /api/inference/status that had been in flight
   throughout -- and it is worth a warning, but the backend was serving again
   seconds later and was answering at the end of the run.

   The verdict deliberately does not reproduce the launcher's watchdog. See
   BackendSurvivalPoller.report: this phase runs no watchdog at all, and its 120s
   window is shorter than the rule needs to reach any verdict. What fails here is
   a backend that stops answering and never comes back, a non-200 answer, or a
   refused port.

   "Never comes back" is decided by await_recovery, which keeps watching after
   sampling stops rather than letting one probe settle it. Sampling ends at an
   arbitrary moment, so a stall straddling that boundary would otherwise fail a
   run where the same stall a minute earlier only warned, which is this file's
   own flake moved to the edge of the window instead of removed.

Runs against a live Unsloth; drives the real UI. Env contract matches the other
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
# STUDIO_OLD_PW is what the repo's own macOS workflow exports;
# STUDIO_PW is what the staging harness exports.
OLD = os.environ.get("STUDIO_OLD_PW") or os.environ["STUDIO_PW"]
# What to rotate the bootstrap password to, when the app forces a change on first
# login. Only used on that path: a harness that already rotated over the API (the
# staging one) never reaches it. Must differ from OLD, or the change is rejected.
NEW = os.environ.get("STUDIO_NEW_PW") or f"{OLD}-Rotated1!"
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_mac_tabs"))
ART.mkdir(parents = True, exist_ok = True)

# How long to keep polling the backend. The reported crash landed at t+66s, so the window has to reach well past that;
# the workflow narrows it to 120s because nothing in this phase runs the launcher's watchdog and the longer window buys
# no extra evidence.
SURVIVAL_S = float(os.environ.get("STUDIO_MAC_SURVIVAL_S", "330"))
POLL_INTERVAL_S = float(os.environ.get("STUDIO_MAC_POLL_INTERVAL_S", "5"))
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "900"))
# How long the forced-verdict check gives the row to settle into its pending state.
FORCED_PENDING_S = float(os.environ.get("STUDIO_MAC_FORCED_PENDING_S", "15"))
# Matches HEALTH_PROBE_TIMEOUT in studio/src-tauri/src/commands.rs, so a probe here waits as long as the launcher's does
# before calling it a miss. It is the only watchdog number this file needs;
# see BackendSurvivalPoller.report for why it does not mirror the rest.
PROBE_TIMEOUT_S = 10.0
# How long to keep watching after sampling stops before calling a stall terminal.
# Sized from the stalls actually observed on this job rather than from a round number. In run 32862298967 one macOS
# runner produced four of them, at 10.03s, 25.1s, 27.75s and 33.2s, so a window that
# decides "this one never ended" has to comfortably clear 33.2s.
# 90s is a little under three times that.
#
# This EXTENDS OBSERVATION; it is not a retry. A retry re-asks a question that already has an answer and hopes for a
# better one, which is how a flaky test hides a real failure. The question here has no answer yet: a probe that timed
# out says only that nothing came back within its budget, and the run ended at an arbitrary point that may fall
# mid-stall. A backend that is genuinely dead answers none of these probes either, so the window costs these seconds
# only on a run that was already going to fail, and it cannot turn a real death into a pass.
RECOVERY_WINDOW_S = 90.0
# Minimum spacing between recovery probes.
# _transport_kind classifies a connection reset as a stall rather than a death on purpose, because a reset means a
# listener accepted and then failed to finish. But a reset comes back in about a millisecond, so an unpaced loop turns
# this window into thousands of connections against a backend that is already in trouble, which can prolong the fault
# it is waiting to clear. Spacing costs nothing on the case that matters: a real stall spends the full probe budget
# before returning, so no pause is added at all there.
RECOVERY_PROBE_SPACING_S = 2.0
# Health replies are a few hundred bytes; these bound a body read that is going wrong.
_READ_CHUNK_BYTES = 65536
_MAX_BODY_BYTES = 1 << 20

LIVENESS_PATH = "/api/liveness"
HEALTH_PATH = "/api/health"
# Both are polled, and an answer from either is proof the backend was serving, matching check_health_inner, which probes
# /api/liveness and falls back to /api/health.
PROBE_PATHS = (LIVENESS_PATH, HEALTH_PATH)
# Every tab the user reported interacting with, plus the ones that share the chat-only gate.
# (route, nav row id, human name).
TABS = [
    ("/chat", "projects", "Chat"),
    ("/hub", "hub", "Hub"),
    ("/images", "images", "Images"),
    ("/studio", "train", "Train"),
    ("/video", "video", "Video"),
    ("/export", "export", "Export"),
]

# Routes that mean "not signed in". Landing on one invalidates every later assertion, so they are matched explicitly
# rather than folded into the generic redirect check.
_SIGNED_OUT_PATHS = ("/login", "/change-password")

# Rows the sidebar pins inline by default, per SIDEBAR_NAV_DEFAULT_PINNED in
# studio/frontend/src/features/settings/stores/appearance-custom-store.ts. Only these
# carry a data-testid: the overflow rows render as MoreMenuItem inside a dropdown that
# mounts nothing until it is opened and passes no test id even when it is, so
# `[data-testid="nav-row-video"]` returns null on every host, every time.
#
# That matters for what this file can claim. The field report named Train AND Video, and
# the first version of this script sampled both -- but while Video sat under "More" (layout
# v5, #7863) its half could never observe anything, so that evidence was structurally empty.
# #8932 pins Video under Images again as layout v7, so it renders a testid and is sampled
# once more. test_inline_row_ids_match_the_frontends_default_pinned_set holds this tuple to
# the store's pinned set, in both directions, so neither a pin nor an unpin can leave an
# assertion here silently observing nothing.
INLINE_ROW_IDS = ("hub", "projects", "images", "video", "train")
# The row every pending-state assertion below is pinned to.
GATED_ROW_ID = "train"
# Intercept pattern for the browser's health reads.
# Matches whether api-base.ts builds a relative path or an absolute one.
_HEALTH_ROUTE = "**/api/health"

_failed: list[str] = []
# Every nav row located anywhere in the tab walk. A walk that never finds the rows the sidebar pins by default proves
# nothing about the gating, so it is a failure rather than a stream of info lines.
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
    """Classify a failed probe as a dead port or a stalled one.

    ECONNREFUSED is the only error that proves nothing is bound: the kernel answers it
    itself, immediately, without a server involved. Everything else here -- a budget that
    ran out, a reset, a truncated response -- is a listener that accepted the connection
    and then failed to finish, which is the stall this poller is measuring. Treating those
    as death is what made a backend the launcher would have kept come out as a crash.
    """
    if isinstance(err, ConnectionRefusedError):
        return "refused"
    return "timeout"


def _read_within(resp, deadline: float) -> str:
    """Read a response body under one deadline for the whole read.

    urllib's ``timeout`` is per socket operation, not per request. A peer that dribbles
    bytes resets it on every chunk, so ``resp.read()`` can outlive any probe budget and,
    here, the script's own wall-clock watchdog. These probes exist to decide whether the
    backend is answering; a probe that never ends is the one outcome that must not
    happen, because a hung job reports nothing and burns the runner.

    read1() returns as soon as any data arrives rather than looping to fill the buffer,
    so the deadline is checked between arrivals and the whole read is bounded by the
    deadline plus at most one socket timeout.
    """
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
            # A health reply is a few hundred bytes. Anything this large is a fault, and reading it to the end would
            # be another way to sit here indefinitely.
            raise TimeoutError("response body exceeded the probe's size cap")
    return b"".join(chunks).decode("utf-8", "replace")


def _probe_once(path: str, timeout: float) -> tuple[int, dict | None, str]:
    """One GET attempt. Do not call directly; _get_json is what bounds it.

    ``kind`` keeps apart the outcomes the desktop watchdog keeps apart, because they
    are different failures and only one of them means the process is gone:

      "ok"      -- answered 200.
      "http"    -- answered, with a status that is not 200. The server is up and saying
                   something is wrong.
      "timeout" -- no answer inside the budget. The port is still there; the server is
                   stalled. This is the case the watchdog spends extra patience on.
      "refused" -- the connection was rejected. Nothing is listening, which is the only
                   one of these that means the backend died.

    Collapsing all three into "status != 200", which this used to do, reports a 10s
    stall in the same words as a crash.

    Only ECONNREFUSED earns "refused". A reset or a half-read response means there WAS
    a listener that failed to see the request through, which is a stall wearing a
    different errno, so it is counted as one rather than as a death.
    """
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
        # A connect-time failure arrives wrapped, so the reason decides, not the class.
        return 0, None, _transport_kind(exc.reason)
    except Exception as exc:
        # A read-time failure raises directly: TimeoutError, or an http.client error when the peer went away
        # mid-response.
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
    # Either still running, or it died without recording anything. Both are "no answer inside the budget", which is
    # exactly what a timeout means here.
    return 0, None, "timeout"


def await_recovery(
    window_s: float = RECOVERY_WINDOW_S, spacing_s: float = RECOVERY_PROBE_SPACING_S
) -> tuple[str, int, float]:
    """Watch the backend after sampling stops, until it answers or *window_s* elapses.

    Returns (kind, status, seconds spent watching, probes), where *probes* are
    sample-shaped records of every attempt, on the poller's clock.

    Only a stall is worth waiting on, so this returns the moment a probe brings back
    anything decisive: an answer of any status settles whether the process is alive, and
    a refused port is already death. Neither gets the window.

    Without this the verdict turned on one probe taken at whatever moment the UI drive
    happened to finish, which put the arbitrary end of the survival window in charge of
    the result. A 25s stall in the middle of the run warned and passed while the same
    stall straddling the boundary failed, which is the flake this file was changed to
    remove, moved to the edge rather than fixed.
    """
    began = time.monotonic()
    probes: list[dict] = []
    status, kind = 0, "timeout"
    while True:
        remaining = window_s - (time.monotonic() - began)
        if probes and remaining <= 0:
            # No time left to give a probe. Starting one anyway is how the watch ran past its own bound: the pacing
            # sleep is clamped to the window, so the loop could arrive here with nothing left and still begin a
            # full-budget request. That is the overrun the wall watchdog exists to catch.
            break
        # Never hand out more budget than the window has left either. A probe started near the end with the default
        # PROBE_TIMEOUT_S can outlive the window by most of that budget on its own.
        probe_began = time.monotonic()
        status, _, kind = _get_json(LIVENESS_PATH, timeout = min(PROBE_TIMEOUT_S, remaining))
        # Recorded in the same shape and on the same clock as the poller's samples, so
        # the caller can lay them end to end. Without this a stall that starts after
        # sampling stops is invisible: the verdict knows it waited, but nothing knows
        # for how long or that anything went wrong, and a recovered stall that goes
        # unreported is the one thing this window was added to avoid.
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
        # A probe that failed instantly did not spend its budget, so it was a reset
        # rather than silence. Pace the next one, and never past the end of the window.
        idle = spacing_s - (time.monotonic() - probe_began)
        if idle > 0:
            time.sleep(min(idle, window_s - elapsed))
    return kind, status, round(time.monotonic() - began, 1), probes


def _stall_windows(samples: list[dict]) -> list[tuple[float, float, bool]]:
    """Spans where no probe answered, merged across both routes.

    One answer from either route is proof the backend was serving at that instant, so an
    answer closes whatever span was open. A span is closed at the moment the successful
    probe was ISSUED rather than when it came back, which understates a stall that ended
    mid-probe; the number here only ever feeds a warning, so erring short is the right
    way to be wrong.

    The third element says the span was still open when sampling stopped, which is the
    only shape that can be a terminal stall.
    """
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
                        # Recorded for whoever reads the artifact after a stall, since "was it generating at the time"
                        # is the first question asked of one. No verdict below reads it.
                        "inference_active": (body or {}).get("inference_active"),
                        "hardware_detecting": (body or {}).get("hardware_detecting"),
                        # Stage 0 of the warm only sets hardware_detecting; this one stays lit through the transformers
                        # and datasets imports after it, so the pair is what tells a reader of the artifacts how wide
                        # the real provisional window on this host was.
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
        """Write the samples out and decide whether the backend survived.

        *final_kind* is what await_recovery saw once sampling stopped, and *final_wait_s*
        is how long it watched for. Together they separate a stall that happened to be in
        progress when the run ended from a backend that is genuinely gone, which is not a
        distinction the samples alone can make: they stop at an arbitrary moment.

        *recovery_samples* are that watch's own probes, on the same clock, and they are
        laid end to end with the poller's. A stall can begin after sampling stops, and
        one that then clears is exactly the case this file argues is worth reporting
        rather than failing; measuring spans from the poller's samples alone would let it
        pass in silence.
        """
        # Written below, once the recovery probes have been folded in.
        # Serialising self.samples alone published a healthy timeline next to a log reporting a long post-run stall,
        # which removes the evidence a reader needs to check the very thing the warning announces.
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
            # These two are the backend saying something, not failing to. Neither is a stall, so neither gets the
            # watchdog's patience: they fail on sight.
            refused = [s for s in got if s["kind"] == "refused"]
            answered_badly = [s for s in got if s["kind"] == "http"]
            if refused:
                # The port stopped accepting. Nothing transient does this to a backend that is meant to be up, so it
                # is fatal on the first occurrence.
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

        # What is left of the verdict, and deliberately so.
        # This used to replay the launcher's watchdog: a 15s grid, three consecutive misses, the widened budget while
        # inference_active is latched, the 30s last-chance probe.
        # This phase gets STUDIO_MAC_SURVIVAL_S = 120 (.github/workflows/studio-mac-ui-smoke.yml), while the busy path
        # alone is HEALTH_WATCHDOG_MAX_FAILURES_BUSY * 15s plus a 30s confirmation, about 210s, and
        # BACKEND_STARTUP_GRACE_PERIOD is 300s before a backend that has not yet answered healthy counts a failure at
        # all. The watchdog is not running here either: this phase boots `unsloth studio` directly, with no Tauri
        # shell, and the watchdog's own behaviour is covered by the Rust tests beside it in commands.rs.
        #
        # What is left is the part that needs no arithmetic and cannot false-positive: a backend that stops answering
        # and never comes back did not survive. Anything that answers again did, on any reading of any budget, so it
        # warns and passes. A threshold put back here has to be strictly longer than the launcher's most generous
        # path, and nothing that long fits in this window.
        #
        # One timeline: the poller's samples then the watch's probes, same clock. A span is therefore measured across
        # the join rather than truncated at it, and a stall that starts after sampling stops gets a span of its own
        # instead of none.
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
            # Nothing came back for the whole recovery window. A stall that was going to end had every one of those
            # seconds to end in.
            if terminal is not None:
                # terminal spans the recovery probes too, now that they are on the same timeline, so the total already
                # includes the watch. Naming the watch again as extra time on top of it would count it twice.
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
            # It came back, so the launcher would have kept it on any budget and this run
            # is green. Say so anyway: a stall this long is a real backend defect even
            # when it is not a fatal one, and it must not vanish into a pass.
            worst_ms = max(s["ms"] for s in observed)
            # Where the longest stall sits relative to the end of sampling decides what can honestly be said about it.
            # All three cases are recovered stalls; only the first one cleared while the run was still watching in the
            # normal way.
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
    """Complete the forced password change a bootstrap login lands on.

    Unsloth seeds a one-time bootstrap password and requires it to be replaced before
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
    # A backend that still has its one-time bootstrap password injects it into the page and signs itself in, landing on
    # /change-password with no login form ever rendered. Check that BEFORE waiting on #password, or the wait burns 60s
    # and reports "no password field" for a session that is actually authenticated.
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
            # A first login with the bootstrap password lands on /change-password (session.ts:93 getPostAuthRoute ->
            # mustChangePassword), which is a signed-out route here because it has no sidebar to assert against. The
            # session is real, so finish the rotation instead of giving up.
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
            # Only fatal before anything was read: a page that cannot be evaluated at all is the signed-out/unrendered
            # shape, and breaking out of it quietly is how this function used to report success on zero observations.
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
    status, live, _kind = _get_json("/api/health")
    if status != 200 or not isinstance(live, dict):
        fail(
            "/api/health gave no body to base the provisional reply on "
            f"(status {status}); the forced pending-state check could not run"
        )
        return
    # A real reply with the measurement removed, so the only thing the browser sees differently is the field under test.
    # device_type is what env.ts reads as "measured", and chat_only stays the conservative pre-detection default --
    # the exact pair a Mac got on first paint in the field report, where the row blacked out.
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
        # The row derives its state synchronously from the store, but the store is filled by the root route's
        # beforeLoad, so give it frames rather than one read.
        deadline = time.monotonic() + FORCED_PENDING_S
        got = None
        while True:
            try:
                got = row_states(page, (GATED_ROW_ID,)).get(GATED_ROW_ID)
            except Exception as exc:
                # Raising out of here would skip the survival report and the exit code main() is built around, so it
                # lands as a failure like any other.
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
            # Never legitimate here: the session was proven signed in before the walk started. Calling this an
            # allowed redirect is exactly how a run that authenticated with nobody goes green having exercised
            # nothing.
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

        # Read every inline row, not just this tab's: they render together, so one read records that the sidebar came
        # up at all and keeps the per-route detail in the log.
        try:
            _rows_seen.update(rid for rid, got in row_states(page).items() if got)
        except Exception as exc:
            info(f"{name}: could not read the sidebar rows ({exc!r})")

        # Clicking the row is the interaction the user reported; a greyed-out row swallows the click, so this doubles
        # as a check that it is reachable.
        try:
            row = page.locator(f'[data-testid="nav-row-{row_id}"]')
            if row.count() > 0 and row.first.is_enabled():
                row.first.click(timeout = 10000)
                page.wait_for_timeout(1000)
            elif row.count() > 0:
                info(f"{name}: nav row present but disabled (measured verdict)")
            elif row_id in INLINE_ROW_IDS:
                # Not an info line: this row is pinned inline by default, so its absence means the sidebar did not
                # render and this tab checked nothing.
                fail(f"{name}: nav row {row_id} is pinned inline by default but did not render")
            else:
                # Expected and permanent for Video and Export: they live under "More", which renders no test id.
                # Reached by route instead.
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
            # Every assertion past this point reads the authenticated shell.
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

        # Hold the session open until the survival window is covered.
        # The reported crash landed at t+66s, well inside this.
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

    poller.finish()

    # Watched after sampling stopped and handed to report(), which needs it to tell a
    # stall still in progress at the end of the run from a backend that never came back.
    step(f"watching up to {RECOVERY_WINDOW_S:.0f}s more for the backend to answer")
    kind, status, waited, recovery = await_recovery()
    info(f"post-run {LIVENESS_PATH}: {kind} after {waited}s of watching, {len(recovery)} probe(s)")
    poller.report(
        final_kind = kind,
        final_status = status,
        final_wait_s = waited,
        recovery_samples = recovery,
    )

    # Cancelled only now. The recovery watch adds up to RECOVERY_WINDOW_S after the UI drive, so disarming before it ran
    # left the longest-running part of the script with nothing enforcing WALL_TIMEOUT_S, and a probe that would not end
    # had the job's own cap as its only bound. A hung job reports nothing, which is the worst outcome here.
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
