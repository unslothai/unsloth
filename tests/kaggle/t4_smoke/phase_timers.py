# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Times Hub calls in-process, since a shared cache filled concurrently by legs can't be diffed."""

from __future__ import annotations

import os
import threading
import time

# transformers.utils.hub binds its own huggingface_hub names at import, so its aliases must be
# patched too; missing cached_files' snapshot_download hides the largest (sharded) fetch.
_TARGETS = (
    ("huggingface_hub", "hf_hub_download"),
    ("huggingface_hub", "snapshot_download"),
    ("transformers.utils.hub", "hf_hub_download"),
    ("transformers.utils.hub", "snapshot_download"),
)


class FetchTimer:
    """Re-entrant: snapshot_download nests hf_hub_download, so only the outermost call adds to seconds."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._seconds = 0.0
        self.calls = 0
        self.bytes = 0
        self.patched: list = []
        self._originals: list = []

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
            # A warm cache reports the existing path's size, not zero; `seconds` is the meaningful number.
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
        # Clamped at zero: rounding can produce a tiny negative.
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
