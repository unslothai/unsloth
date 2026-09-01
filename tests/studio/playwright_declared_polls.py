# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every path the idle app polls must be declared in the log budget.

The budget suite in ``studio/backend/tests/log_budget`` checks two backend files against
each other: the suppression sets in ``loggers/handlers.py`` and the poll cadences in
``log_budget/session.py``. That catches quieting a path without declaring it, and
declaring a path without classifying it.

It cannot catch the case this exists for. A pull request that adds a polling call site in
the frontend touches neither file, so nothing fails, the new endpoint quietly lands in the
``normal`` class on a 300 ms window, and the access log grows by 12 lines a minute that
nobody chose. That is how the last several log-volume PRs became necessary.

Static analysis does not close it either: polling here is 53 ``setInterval`` call sites,
most of which are UI timers rather than fetches, plus 257 ``setTimeout`` of which an
unknown number are recursive poll loops, and the fetch is usually several helper layers
below the timer. So observe the browser instead. Whatever a path is polled BY, it shows up
here.

What this reaches, and what it does not. It sees any poll that runs unconditionally on a
screen the walk visits, which is where a newly added ``setInterval`` normally lands. It
does NOT see polls gated behind state a bare runner cannot reach: with no GPU and no model
loaded, the loaded-models indicator and the training views never start their timers, and
Train is not even clickable. Measured on such a runner the observable set is two paths. So
treat this as a net with a known mesh size rather than a proof that no new poll exists.

Run: BASE_URL, STUDIO_OLD_PW and STUDIO_NEW_PW as the other suites take them.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    install_wall_clock_watchdog,
    wait_for_health,
)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "studio" / "backend" / "tests"))
from log_budget.session import ALL_POLLS  # noqa: E402

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ["STUDIO_NEW_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-declared-polls"))
ART.mkdir(parents = True, exist_ok = True)

# How long to watch, once the app has stopped booting.
SETTLE_S = float(os.environ.get("PW_POLL_SETTLE_S", "20"))
WATCH_S = float(os.environ.get("PW_POLL_WATCH_S", "75"))
WALL_TIMEOUT_S = SETTLE_S + WATCH_S + 180

# A boot read fires once. A retry fires twice.
POLL_THRESHOLD = 3

# Sidebar sections to walk, so section-scoped timers get a chance to run.
SECTIONS = ("Model hub", "Projects", "Images", "Train", "New chat")

# Anything less means the listener never attached or the page never rendered.
# The floor that stops this suite passing on a run that observed nothing at all.
MIN_POLLED_PATHS = int(os.environ.get("PW_POLL_MIN_PATHS", "2"))


def info(message: str) -> None:
    print(message, flush = True)


def api(
    path: str,
    payload: dict,
    token: str | None = None,
) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data = json.dumps(payload).encode(),
        headers = {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(request, timeout = 30) as response:
        return json.loads(response.read().decode() or "{}")


def normalize(url: str) -> str:
    """Path only. The query string is part of the dedup identity in the middleware, but a
    poll declares a PATH, and a cursor or cache-buster would otherwise make every hit look
    unique and hide the poll entirely."""
    return urlsplit(url).path


def main() -> int:
    declared = set(ALL_POLLS)

    if not wait_for_health(BASE, info = info):
        info("FAIL backend never became healthy")
        return 1

    # Bootstrap as the sibling suites do: the first login forces a change.
    # Tolerate the rotation having already happened, because a second suite against the same instance would otherwise
    # die on a 401 traceback instead of just logging in.
    try:
        token = api("/api/auth/login", {"username": "unsloth", "password": OLD})["access_token"]
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise
        token = None
    if token is not None:
        try:
            api("/api/auth/change-password", {"current_password": OLD, "new_password": NEW}, token)
        except urllib.error.HTTPError:
            pass
    session = api("/api/auth/login", {"username": "unsloth", "password": NEW})

    seed_js = (
        "(() => {"
        f"  localStorage.setItem('unsloth_auth_token', {json.dumps(session['access_token'])});"
        f"  localStorage.setItem('unsloth_refresh_token', "
        f"{json.dumps(session.get('refresh_token', ''))});"
        "})();"
    )

    seen: Counter[str] = Counter()
    # Counts are attributed to the current dwell, never summed across the walk.
    # Visiting five sections fetches a per-view endpoint five times, which is indistinguishable from a 30s timer if you
    # only look at the total.
    # A timer repeats WITHIN one dwell;
    dwell_counts: Counter[str] = Counter()
    per_dwell_max: Counter[str] = Counter()
    counting = False

    def close_dwell() -> None:
        for path, n in dwell_counts.items():
            if n > per_dwell_max[path]:
                per_dwell_max[path] = n
        dwell_counts.clear()

    with sync_playwright() as p:
        install_wall_clock_watchdog(WALL_TIMEOUT_S, label = "declared-polls", info = info)
        browser = p.chromium.launch(headless = True, args = chromium_launch_args())
        context = browser.new_context(
            viewport = {"width": 1440, "height": 900}, reduced_motion = "reduce"
        )
        context.add_init_script(seed_js)
        page = context.new_page()

        def on_request(request) -> None:
            if not counting:
                return
            path = normalize(request.url)
            if path.startswith("/api/"):
                seen[path] += 1
                dwell_counts[path] += 1

        page.on("request", on_request)
        page.goto(BASE, wait_until = "domcontentloaded", timeout = 60_000)

        # Boot traffic is one-shot by nature and would otherwise be indistinguishable from a slow poll over a short
        page.wait_for_timeout(int(SETTLE_S * 1000))
        counting = True

        # Sitting on the default screen sees almost nothing:
        dwell = max(int(WATCH_S * 1000 / (len(SECTIONS) + 1)), 20_000)
        for label in SECTIONS:
            try:
                page.get_by_text(label, exact = True).first.click(timeout = 8_000)
            except Exception:
                info(f"note: could not reach {label!r}")
                continue
            page.wait_for_timeout(dwell)
            close_dwell()
        page.wait_for_timeout(dwell)
        close_dwell()
        counting = False

        current = normalize(page.url)
        context.close()
        browser.close()

    if current.startswith(("/login", "/change-password")):
        # Without this the suite passes by observing an app it never entered.
        # Every counted request would be the login screen's, which polls almost nothing.
        info(f"FAIL still on {current} after seeding a token; the run proved nothing")
        return 1

    polled = {path: n for path, n in per_dwell_max.items() if n >= POLL_THRESHOLD}
    undeclared = sorted(set(polled) - declared)
    (ART / "observed_polls.json").write_text(
        json.dumps(
            {
                "watch_seconds": WATCH_S,
                "threshold": POLL_THRESHOLD,
                "totals": dict(sorted(seen.items())),
                "per_dwell_max": dict(sorted(per_dwell_max.items())),
                "polled": dict(sorted(polled.items())),
                "undeclared": undeclared,
            },
            indent = 2,
        ),
        encoding = "utf-8",
    )

    info(
        f"watched {WATCH_S:.0f}s: {len(seen)} distinct API paths, "
        f"{len(polled)} repeated {POLL_THRESHOLD}x or more inside a single dwell"
    )

    if len(polled) < MIN_POLLED_PATHS:
        info(
            f"FAIL saw only {len(polled)} polled path(s), expected at least "
            f"{MIN_POLLED_PATHS}; this run cannot vouch for anything"
        )
        for path, n in sorted(polled.items()):
            info(f"  {path}  {n}x")
        return 1

    # never sees them.
    # Declared-but-unobserved is NOT a failure.
    unobserved = sorted(declared - set(polled))
    if unobserved:
        # Too thin to mean anything: the listener never attached, the walk never left one
        info(
            f"note: {len(unobserved)} declared polls not seen while idle "
            f"(expected for pane-scoped and busy-only polls)"
        )

    if undeclared:
        info("")
        info("FAIL these paths are polled by the app but are not in the log budget:")
        for path in undeclared:
            per_min = polled[path] * 60.0 / (WATCH_S / (len(SECTIONS) + 1))
            info(f"  {path}   {polled[path]}x within one dwell (~{per_min:.1f}/min)")
        info("")
        info("Each one needs two lines, in studio/backend/tests/log_budget/session.py:")
        info("  add the path to IDLE_POLLS (or BUSY_POLLS if it only runs during an")
        info("  operation) with the interval its call site declares, and give it a class")
        info("  in studio/backend/loggers/handlers.py. A path left unclassified logs")
        info("  every single hit.")
        return 1

    info("PASS every polled path is declared and classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
