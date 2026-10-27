# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The distributed tests' MASTER_PORT comes from outside the ephemeral range, in a per-worker slice."""

import socket

from tests import _rendezvous_port as rp


def test_the_port_is_below_every_default_ephemeral_range_and_in_this_workers_slice(monkeypatch):
    for worker, base in (("gw0", 20000), ("gw2", 22000), ("gw11", 31000)):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", worker)
        for _ in range(20):
            port = rp.free_port()
            assert base <= port < base + rp.SLICE < 32768, (worker, port)


def test_no_xdist_worker_uses_the_first_slice(monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising = False)
    assert 20000 <= rp.free_port() < 21000


def test_a_taken_port_is_skipped(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw5")
    held = socket.socket()
    try:
        taken = next(p for p in range(25000, 26000) if _try_listen(held, p))
        other = 25000 + (taken - 25000 + 1) % rp.SLICE
        offsets = iter([taken - 25000, other - 25000])

        class _Rng:
            def randrange(self, n):
                return next(offsets)

        monkeypatch.setattr(rp.random, "Random", _Rng)
        assert rp.free_port() == other
    finally:
        held.close()


def _try_listen(sock, port):
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    sock.listen()
    return True
