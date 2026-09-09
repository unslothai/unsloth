# SPDX-License-Identifier: AGPL-3.0-only
"""Own selected-runtime read grants; command workdir grants remain per-execution."""

import atexit
import hashlib
import json
import threading
import time

from .srt_windows_owner import _Broker

_lock = threading.Lock()
_owner = None
_stopping = False


def acquire(
    request,
    cancel_event = None,
    deadline = None,
):
    global _owner
    from . import srt_adapter, srt_probe

    deadline = deadline or time.monotonic() + 120
    with _lock:
        if _stopping or (cancel_event is not None and cancel_event.is_set()):
            raise srt_adapter.SrtError("Runtime read lease is unavailable or cancelled")
        identity = hashlib.sha256(
            json.dumps(
                [
                    request["readRoots"],
                    srt_adapter.installation_identity(),
                    srt_probe.runtime_inputs(),
                ],
                sort_keys = True,
                default = str,
            ).encode()
        ).hexdigest()
        if _owner is not None and (_owner.identity != identity or _owner.proc.poll() is not None):
            _owner.close()
            _owner = None
        if _owner is None:
            _owner = _Broker(
                identity,
                deadline,
                owner_script = "windows-read-owner.mjs",
                bootstrap = {"readRoots": request["readRoots"]},
            )
        if (cancel_event is not None and cancel_event.is_set()) or time.monotonic() >= deadline:
            raise srt_adapter.SrtError("Runtime read lease admission cancelled or timed out")
        return {"port": _owner.port, "token": _owner.token, "identity": identity}


def shutdown():
    global _owner, _stopping
    with _lock:
        _stopping = True
        if _owner is not None:
            _owner.close()
            _owner = None


atexit.register(shutdown)
