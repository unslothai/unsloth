# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-request I/O never depends on how many accounts exist, and the owner's routing is fixed.

The counters below are read from a WARM request, because the interesting claim is about
steady state: a first request legitimately builds caches and opens the databases it will
reuse, and counting that would only ever measure start-up. Two unmeasured requests warm
it, and the third is the one read.

Two turned out not to be enough. Run 35530114495 read /api/chat/threads at 3 mkdir, 3
connections and 86 statements where the same request in the same test had just measured
2, 2 and 8, on a commit whose only shared-path change was error-message wording, and the
identical job passed on re-run. Warm-up cannot bound lazy work that is triggered by a
clock rather than by a request count: anything with a TTL can expire between the warm-up
and the measurement and be rebuilt inside it.

What that noise can do is add work, never remove it. So the measurement is repeated and
each counter is taken at its MINIMUM, which is the steady-state cost; a rebuild that
fires in one repeat is not in the others. A cost that is genuinely higher with more
accounts is higher in every repeat and still fails, which is what _steady_cost_keeps_a_
persistent_increase checks directly rather than by assertion.
"""

import os
import secrets
import sqlite3
from collections import Counter
from unittest.mock import patch

from auth import policy, storage
from core.inference.llama_admission import get_llama_admission_queue
from hub.services.models import account_access as access
from state import active_generations
from utils.account_context import OWNER, AccountContext, run_as

from .support import bearer

PATHS = ("/api/auth/status", "/account-probe", "/api/chat/threads")


def _create(names):
    for name in names:
        storage.create_initial_user(name, "account-password", secrets.token_urlsafe(32))
    policy.invalidate_account_cache()


# Three readings, not two: two cannot tell which of a disagreeing pair is the steady state.
_REPEATS = 3


def _steady(readings: list) -> dict:
    """The per-counter minimum across readings of the same request.

    Lazily rebuilt state adds work to whichever reading it lands in and takes none away,
    so the smallest reading of each counter is the cost with nothing rebuilding. A cost
    that really did grow grew in all of them, and the minimum grows with it.
    """
    keys = set().union(*readings) if readings else set()
    steady = {key: min(reading.get(key, 0) for reading in readings) for key in keys}
    # A counter only present in the noisy reading has a minimum of zero, and a zero is the
    # same statement as the key being absent: Counter never records one. Keeping it would
    # turn the noise back into a difference under the `==` below, which is the whole thing
    # this is trying to stop.
    return {key: value for key, value in steady.items() if value}


def _vector(client, headers, path) -> dict:
    """sqlite connections, statements, mkdirs and auth.db count queries for one warm request."""
    counters = Counter()
    connect, mkdir, count = sqlite3.connect, os.mkdir, storage.account_counts

    def open_connection(*args, **kwargs):
        counters["connections"] += 1
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(
            lambda _: counters.__setitem__("statements", counters["statements"] + 1)
        )
        return conn

    def create_directory(*args, **kwargs):
        counters["mkdir"] += 1
        return mkdir(*args, **kwargs)

    def counted(*args, **kwargs):
        counters["account_counts"] += 1
        return count(*args, **kwargs)

    for _ in range(2):
        assert client.get(path, headers = headers).status_code == 200
    readings = []
    for _ in range(_REPEATS):
        counters.clear()
        with (
            patch.object(sqlite3, "connect", open_connection),
            patch.object(os, "mkdir", create_directory),
            patch.object(storage, "account_counts", counted),
        ):
            assert client.get(path, headers = headers).status_code == 200
        readings.append(dict(counters))
    return _steady(readings)


def _vectors(client, username):
    headers = bearer(username)
    return {path: _vector(client, headers, path) for path in PATHS}


def test_owner_requests_cost_the_same_with_zero_and_many_accounts(isolated_auth, account_client):
    _create(["unsloth"])
    alone = _vectors(account_client, "unsloth")
    _create([f"user{index:02d}" for index in range(12)])
    crowded = _vectors(account_client, "unsloth")
    assert alone["/account-probe"]["connections"] >= 1, "the counters must observe the auth lookup"
    assert crowded == alone
    assert all(vector.get("account_counts", 0) == 0 for vector in alone.values())


def test_managed_requests_do_not_grow_with_account_count(isolated_auth, account_client):
    _create(["unsloth", "alice"])
    few = _vectors(account_client, "alice")
    _create([f"user{index:02d}" for index in range(12)])
    assert _vectors(account_client, "alice") == few
    assert all(vector.get("account_counts", 0) == 0 for vector in few.values())


def test_owner_routing_facts_and_shared_queue(isolated_auth, account_client):
    _create(["unsloth"])
    headers = bearer("unsloth")
    for path in PATHS:
        assert account_client.get(path, headers = headers).status_code == 200
    assert run_as(OWNER, access.account_scope) is None
    assert run_as(OWNER, access.resident_hidden, "chat") is False
    assert access._resident_sharers == {}
    assert active_generations._FENCED == set()

    _create(["alice"])
    alice = storage.get_account("alice")
    managed = AccountContext(alice.account_id, alice.username, alice.role)
    assert run_as(OWNER, access.account_scope) == "owner"
    assert run_as(managed, access.account_scope) == alice.account_id
    assert run_as(OWNER, access.resident_hidden, "chat") is False
    # Admission queues are keyed by the resident server, never by the calling account.
    key = "http://127.0.0.1:1"
    assert run_as(OWNER, get_llama_admission_queue, key) is run_as(
        managed, get_llama_admission_queue, key
    )


def test_steady_cost_keeps_a_persistent_increase_and_drops_a_one_off():
    """The minimum above must not be able to hide a cost that is really there.

    Read directly rather than through a request, because the two cases it has to tell
    apart are exactly the ones a live request cannot be made to produce on demand.
    """
    baseline = [{"connections": 2, "statements": 8}] * _REPEATS

    one_off = [{"connections": 2, "statements": 8}] * _REPEATS
    one_off[1] = {"connections": 3, "statements": 86}
    assert _steady(one_off) == _steady(baseline), (
        "a rebuild landing in one reading is not the steady-state cost, and taking the "
        "minimum is what makes that true"
    )

    persistent = [{"connections": 3, "statements": 9}] * _REPEATS
    assert _steady(persistent) != _steady(baseline), (
        "a cost that is higher on every reading is a real increase; if the minimum "
        "swallowed it this guard would pass a per-account regression"
    )

    # A counter that appears in only one reading is absent from the others, so its minimum
    # is zero, and a zero has to be dropped rather than reported. Counter never records a
    # zero, so a reading that never saw the mkdir has no `mkdir` key at all, and keeping
    # `{"mkdir": 0}` would fail the `==` against it just as surely as `{"mkdir": 1}` would.
    appears_once = [{"connections": 2}, {"connections": 2, "mkdir": 1}, {"connections": 2}]
    never_appears = [{"connections": 2}] * _REPEATS
    assert _steady(appears_once) == {"connections": 2}
    assert _steady(appears_once) == _steady(never_appears), (
        "a one-off counter has to leave the steady vector, not sit in it at zero: the "
        "comparisons above are exact, and a key the other side has never heard of "
        "differs whatever its value"
    )
