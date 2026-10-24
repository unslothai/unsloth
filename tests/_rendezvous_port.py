# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A free port for a test's torch.distributed rendezvous (MASTER_PORT).

``bind(0)`` hands out a port from the kernel's ephemeral range and the test releases it before rank 0
listens on it. Gloo's own peer connections, in this test and in every other xdist worker, draw their
local ports from that same range, so one of them can take the port first and rank 0 fails with
EADDRINUSE (seen in Repo tests (CPU, fsdp2): "port: 33653 ... address already in use"). Ports here come
from below every OS's default ephemeral range, and each xdist worker draws from its own slice, so
neither an outgoing connection nor a test running alongside can pick the same one.
"""

import os
import random
import socket

# Linux allocates ephemeral ports from 32768 and macOS / Windows from 49152, so nothing below 32768 is
# handed out by the kernel on its own.
LOW, HIGH = 20000, 32000
SLICE = 1000


def _worker_index() -> int:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    return int(worker[2:]) if worker[2:].isdigit() else 0


def free_port() -> int:
    base = LOW + (_worker_index() * SLICE) % (HIGH - LOW)
    rng = random.Random()
    for _ in range(64):
        port = base + rng.randrange(SLICE)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:  # taken, or still in TIME_WAIT from an earlier test in this worker
                continue
            return port
    # Every candidate busy: the kernel's choice still works, just without the guarantee.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
