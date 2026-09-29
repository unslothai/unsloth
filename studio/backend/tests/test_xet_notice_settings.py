# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Xet notice counter: the count survives, and concurrency cannot beat the limit.

It lived in per-origin localStorage, and an Unsloth origin moves whenever port 8888 is
taken, so "three times" meant "three times per port, forever".
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
    """A restart, on any port, must not hand out three more."""
    for _ in range(2):
        assert xet.reserve_xet_notice()["granted"] is True

    # A fresh connection is what a restarted backend gets: nothing is cached in
    # module state, the value comes back off disk.
    assert xet.get_xet_notice_count() == 2
    assert xet.reserve_xet_notice() == {"granted": True, "shown": 3, "limit": 3}
    assert xet.reserve_xet_notice()["granted"] is False


def test_concurrent_reservations_never_exceed_the_limit():
    """Two tabs at once. Why the reserve is one transaction: a split read and write
    lets both callers read the same count and both be granted."""
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
    """Migration from localStorage, upwards only: it must not win back sightings."""
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
    """A junk row must not wedge the notice off forever."""
    studio_db.upsert_app_settings({xet.XET_NOTICE_COUNT_KEY: "not a number"})
    assert xet.get_xet_notice_count() == 0
    assert xet.reserve_xet_notice()["granted"] is True

    studio_db.upsert_app_settings({xet.XET_NOTICE_COUNT_KEY: -5})
    assert xet.get_xet_notice_count() == 0

    # A bool is an int in Python; it is not a count.
    studio_db.upsert_app_settings({xet.XET_NOTICE_COUNT_KEY: True})
    assert xet.get_xet_notice_count() == 0


def test_the_stored_shape_is_a_plain_json_int():
    """Other readers of app_settings should find a number, not a string."""
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
