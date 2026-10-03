# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep an in-process installer run from leaking ``os.environ`` into the session.

``os.environ`` is process-global. A test that drives production code which writes it
leaves that value set for the rest of the xdist worker's session, and every subprocess
the worker spawns afterwards inherits it. Same class as the shared venv root closed in
issue #9586's channel 1, different medium -- and no file sweep or ``git status`` check
can see this one.

The observed instance: ``_install_bnb_windows_rocm()`` in ``studio/install_python_stack.py``
sets ``BNB_ROCM_VERSION`` and ``UNSLOTH_BNB_ROCM_VERSION_SOURCE`` so the worker subprocess
inherits them, which is correct for a real install. A few lines above the write, the same
function READS both to decide whether to re-detect and re-persist. So a later test in the
same worker takes the other branch, ``_persist_detected_version`` stays false, and the
``sitecustomize`` write never happens -- the test passes for the wrong reason, and which
way it goes depends on worker ordering.

``monkeypatch`` cannot close this. It restores only the keys it was itself asked to set;
a direct ``os.environ[...] = ...`` inside production code is invisible to it.

Restoration mutates ``os.environ`` in place rather than rebinding it. The mapping proxies
``putenv``/``unsetenv`` to the real process environment, which is what subprocesses read,
and other modules hold references to the same object.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator


@contextlib.contextmanager
def contained_environ() -> Iterator[None]:
    """Restore ``os.environ`` to its entry state on exit.

    Restores keys the body added, changed, or deleted. Teardown is sound for this
    medium in a way it would not be for files: a hard-killed worker takes its own
    environment with it, so there is no residue for a later run to inherit.
    """
    before = dict(os.environ)
    try:
        yield
    finally:
        # Delete first, then re-set: a key the body renamed shows up in both loops,
        # and doing it in this order leaves the entry value rather than nothing.
        for key in [key for key in os.environ if key not in before]:
            del os.environ[key]
        for key, value in before.items():
            if os.environ.get(key) != value:
                os.environ[key] = value
