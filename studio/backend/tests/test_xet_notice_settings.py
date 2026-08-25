# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Xet notice counter, which is the reason the notice ever stops.

It used to live in the browser, in localStorage, which is scoped to an origin. A
Studio origin is not stable: run.py defaults to port 8888 and falls back to the next
free port when it is taken, and 8888 is also Jupyter's default. Colab and the tunnel
are different origins again. Each of those handed the user a fresh set of three, so
"show this three times" meant "three times per port, forever".

These tests are about the two properties that gives the install: the count survives,
and concurrent callers cannot talk their way past the limit.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.xet_notice_settings as xet  # noqa: E402
from storage import studio_db  # noqa: E402


def test_three_are_granted_and_the_fourth_is_not():
    assert xet.get_xet_notice_count() == 0
    granted = [xet.reserve_xet_notice()["granted"] for _ in range(5)]
    assert granted == [True, True, True, False, False]
    assert xet.get_xet_notice_count() == xet.XET_NOTICE_LIMIT


def test_the_count_outlives_the_process_that_wrote_it():
    """The whole point. A restart, on any port, must not hand out three more."""
    for _ in range(2):
        assert xet.reserve_xet_notice()["granted"] is True

    # A fresh connection is what a restarted backend gets: nothing is cached in
    # module state, the value comes back off disk.
    assert xet.get_xet_notice_count() == 2
    assert xet.reserve_xet_notice() == {"granted": True, "shown": 3, "limit": 3}
    assert xet.reserve_xet_notice()["granted"] is False


def test_concurrent_reservations_never_exceed_the_limit():
    """Two tabs starting a download at the same moment.

    This is why the reserve is one transaction rather than get_app_setting followed
    by upsert_app_settings: those open a connection each, so both callers can read
    the same count and both be granted. The browser implementation needed Web Locks
    to paper over exactly this, and could not cover a second browser at all.
    """
    results: list[bool] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def worker():
        start.wait()
        granted = xet.reserve_xet_notice()["granted"]
        with lock:
            results.append(granted)

    threads = [threading.Thread(target = worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert sum(results) == xet.XET_NOTICE_LIMIT
    assert xet.get_xet_notice_count() == xet.XET_NOTICE_LIMIT


def test_a_legacy_hint_raises_the_count_but_never_lowers_it():
    """Migration from localStorage, in the safe direction only.

    Someone who already spent their three before this moved server-side must not get
    three more. Equally, the hint arrives from a client, so it must not be able to
    lower a stored count and win back sightings.
    """
    assert xet.reserve_xet_notice(seen_hint = 2)["shown"] == 3
    assert xet.reserve_xet_notice()["granted"] is False

    # A client claiming it has seen none does not reset anything.
    assert xet.reserve_xet_notice(seen_hint = 0)["granted"] is False
    assert xet.get_xet_notice_count() == 3


def test_a_hint_at_or_over_the_limit_grants_nothing():
    assert xet.reserve_xet_notice(seen_hint = 3)["granted"] is False
    assert xet.reserve_xet_notice(seen_hint = 99)["granted"] is False
    # The stored count is clamped by what was actually reserved, not by the claim.
    assert xet.get_xet_notice_count() >= xet.XET_NOTICE_LIMIT


def test_an_unreadable_stored_value_reads_as_a_fresh_install():
    """A hand-edited or half-written row must not wedge the notice off forever."""
    studio_db.upsert_app_settings({xet.XET_NOTICE_COUNT_KEY: "not a number"})
    assert xet.get_xet_notice_count() == 0
    assert xet.reserve_xet_notice()["granted"] is True

    studio_db.upsert_app_settings({xet.XET_NOTICE_COUNT_KEY: -5})
    assert xet.get_xet_notice_count() == 0

    # A bool is an int in Python; it is not a count.
    studio_db.upsert_app_settings({xet.XET_NOTICE_COUNT_KEY: True})
    assert xet.get_xet_notice_count() == 0


def test_the_stored_shape_is_a_plain_json_int():
    """Other readers of app_settings should find a number, not a stringified one."""
    xet.reserve_xet_notice()
    stored = studio_db.get_app_setting(xet.XET_NOTICE_COUNT_KEY, None)
    assert stored == 1
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            "SELECT value_json FROM app_settings WHERE key = ?",
            (xet.XET_NOTICE_COUNT_KEY,),
        ).fetchone()
    finally:
        conn.close()
    assert json.loads(row["value_json"]) == 1
