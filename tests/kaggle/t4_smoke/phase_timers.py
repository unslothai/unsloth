# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Split the load phase into FETCH and WEIGHT LOAD, so one dispatch can say
whether the Hub download is worth optimising at all.

Every leg today reports one `load` number covering `from_pretrained`, which is
a download, a disk read and a quantised materialisation in one figure. The
prefetch lane, the second-wave reorder and the whole `D` argument in the plan
rest on how that figure splits, and it has never been measured -- only bounded
by subtraction, which is not the same thing.

**Why this instruments the process rather than the cache.** The obvious
approach is to diff the hub cache around the call. It cannot work here: the
legs download into ONE shared cache CONCURRENTLY, so a whole-cache delta
credits another leg's bytes to this one and reports a rate the Hub never
delivered. Scoping the diff to one repo folder fixes that but reintroduces a
different error, because `load_in_4bit=True` redirects through Unsloth's
FLOAT_TO_INT_MAPPER and the repo that downloads is not the repo that was asked
for. Timing the download calls INSIDE this interpreter is immune to both: each
leg is its own process, so anything measured here was fetched by this leg.

**The failure mode this is built to avoid.** An instrument that silently
reports zero is worse than no instrument, because "no download happened" and
"the timer never attached" read identically in a report and only one of them is
a finding. So `seconds` is None until something is genuinely patched, and
`patched` names what was wrapped. A reader can tell the two apart.

Nothing here may fail the run. A payload that dies while collecting a
diagnostic reports nothing at all, which is the one outcome worse than a
missing number.
"""

from __future__ import annotations

import os
import threading
import time

# The download entry points transformers and unsloth actually reach. Both are
# patched where present: `huggingface_hub.hf_hub_download` is the public name,
# and `transformers.utils.hub` binds its own reference at import time, so
# patching only the former leaves the transformers path unmeasured.
_TARGETS = (
    ("huggingface_hub", "hf_hub_download"),
    ("huggingface_hub", "snapshot_download"),
    ("transformers.utils.hub", "hf_hub_download"),
)


class FetchTimer:
    """Accumulate wall time and bytes spent inside Hub download calls.

    Re-entrant by design: `snapshot_download` calls `hf_hub_download` per file,
    so a naive sum would count the inner calls twice and report more download
    seconds than the phase itself took. A depth counter means only the
    OUTERMOST call contributes to `seconds`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._seconds = 0.0
        self.calls = 0
        self.bytes = 0
        self.patched: list = []
        self._originals: list = []

    # -------------------------------------------------------------- patching

    def _wrap(self, fn):
        def wrapped(*args, **kwargs):
            with self._lock:
                outermost = self._depth == 0
                self._depth += 1
                self.calls += 1
            started = time.time()
            try:
                result = fn(*args, **kwargs)
            finally:
                with self._lock:
                    self._depth -= 1
                    if outermost:
                        self._seconds += time.time() - started
            # The returned path is the file or directory that now exists, so
            # its size is what this call put on disk. A warm cache returns the
            # same path having transferred nothing, which reads as its own size
            # rather than as zero -- accepted, and stated, because the leg runs
            # against a cache the prefetch may already have warmed and the
            # interesting number there is `seconds`, not `bytes`.
            try:
                self.bytes += _path_bytes(result)
            except Exception:  # noqa: BLE001
                pass
            return result

        return wrapped

    def install(self) -> "FetchTimer":
        import importlib
        for module_name, attr in _TARGETS:
            try:
                module = importlib.import_module(module_name)
            except Exception:  # noqa: BLE001
                continue
            original = getattr(module, attr, None)
            if original is None or not callable(original):
                continue
            try:
                setattr(module, attr, self._wrap(original))
            except Exception:  # noqa: BLE001
                continue
            self._originals.append((module, attr, original))
            self.patched.append(f"{module_name}.{attr}")
        return self

    def uninstall(self) -> None:
        for module, attr, original in self._originals:
            try:
                setattr(module, attr, original)
            except Exception:  # noqa: BLE001
                pass
        self._originals = []

    # --------------------------------------------------------------- reading

    @property
    def seconds(self):
        """None when nothing was patched, so a dead timer cannot read as 0.0."""
        if not self.patched:
            return None
        return round(self._seconds, 1)

    def record(self, total_seconds: float) -> dict:
        """The split, plus enough context to distrust it if it deserves that."""
        fetch = self.seconds
        out = {
            "patched": list(self.patched),
            "calls": self.calls,
            "fetch_seconds": fetch,
            "fetch_mb": round(self.bytes / 1024**2, 1) if self.patched else None,
            "total_seconds": round(total_seconds, 1),
        }
        if fetch is None:
            out["weight_load_seconds"] = None
            out["note"] = (
                "the fetch timer never attached, so this run says nothing about "
                "the split; do not read the absence as 'no download happened'"
            )
            return out
        # Clamped at zero: the two clocks are the same clock, but rounding and
        # a download finishing inside the final microseconds of the phase can
        # still produce a negative by a tenth, and a negative duration in a
        # report is read as a bug in the report rather than as rounding.
        out["weight_load_seconds"] = round(max(total_seconds - self._seconds, 0.0), 1)
        if self.bytes and self._seconds > 0:
            out["fetch_mb_s"] = round(self.bytes / 1024**2 / self._seconds, 1)
        return out

    def __enter__(self) -> "FetchTimer":
        return self.install()

    def __exit__(self, *_exc) -> None:
        self.uninstall()


def _path_bytes(path) -> int:
    if not path:
        return 0
    path = str(path)
    if os.path.isfile(path):
        return os.path.getsize(path)
    if not os.path.isdir(path):
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total
