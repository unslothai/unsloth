# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Give every xdist worker its own torch.compile cache directory.

The suites here run under `pytest -n 4`. Inductor's on-disk caches default to one
directory per USER, not per process, so four workers on one runner share
`/tmp/torchinductor_<user>` and its `fxgraph`, `aotautograd` and Triton subtrees. The
upstream recipe is explicit that a common `TORCHINDUCTOR_CACHE_DIR` is what makes
processes share compiled artifacts, so the way to stop them sharing is to give each one
a different value.

Sharing a cache between four concurrent compilers is not obviously wrong -- the entries
are content-addressed -- but it is a write-write interaction between processes that
nothing in this repo controls, and a cache is exactly the sort of thing that turns a
deterministic suite into an intermittent one. Splitting it removes the interaction for
the price of some recompilation.

Measured before adding it, so nobody has to guess later what it bought: a full
unsloth_zoo run against an empty, dedicated cache directory finished in 578s and left
ZERO entries in it. That suite does not populate the on-disk cache at all, so for that
step this is neither a saving nor a cost. It is insurance for the suites that do
compile, and it is applied wherever tests run in parallel rather than only where a
problem has already been seen.

TRITON_CACHE_DIR is set explicitly rather than left to follow: it only derives from
TORCHINDUCTOR_CACHE_DIR when it is unset, and an environment that already exports it
would otherwise keep all four workers pointed at one Triton cache.

Imported for its side effect from the conftest files, and it must run BEFORE torch is
imported, which is why it is a module-level statement rather than a fixture.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

WORKER_ENV = "PYTEST_XDIST_WORKER"


def isolate_compile_caches() -> str | None:
    """Point this worker's inductor and Triton caches at a directory of its own.

    Returns the directory, or None when not running under xdist, where the process
    already has the default to itself and there is nothing to separate.
    """
    worker = os.environ.get(WORKER_ENV)
    if not worker:
        return None

    base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if base:
        # Respect an explicit choice of location, and split underneath it. Replacing it
        # would move the cache somewhere the caller did not ask for, which matters when
        # CI points it at a cached path on purpose.
        root = pathlib.Path(base)
    else:
        root = pathlib.Path(tempfile.gettempdir()) / f"torchinductor_{os.environ.get('USER', 'ci')}"

    mine = root / f"xdist_{worker}"
    mine.mkdir(parents = True, exist_ok = True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(mine)
    os.environ["TRITON_CACHE_DIR"] = str(mine / "triton")
    return str(mine)


isolate_compile_caches()
