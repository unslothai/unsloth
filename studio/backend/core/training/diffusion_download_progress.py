# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Observe only this training child's active byte bars; never scan unrelated cached repositories."""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Iterable

from utils.account_context import account_thread


def _bytes(value: Any) -> int | None:
    try:
        return int(value) if math.isfinite(float(value)) and float(value) >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def download_snapshot(bars: Iterable[Any]) -> dict[str, Any] | None:
    entries = []
    for bar in bars:
        if getattr(bar, "disable", False) or getattr(bar, "unit", None) != "B":
            continue
        done, total = _bytes(getattr(bar, "n", None)), _bytes(getattr(bar, "total", None))
        if done is None or total is None or total <= 0:
            continue
        label = str(getattr(bar, "desc", "") or "Model files")[:200]
        entries.append((label, min(done, total), total))
    if not entries:
        return None
    # Network and reconstruction bars cover the SAME bytes: count only reconstruction.
    reconstructed = [entry for entry in entries if "reconstruct" in entry[0].lower()]
    transfers = [entry for entry in entries if entry[0].lower() == "downloading bytes"]
    selected = reconstructed or transfers or entries
    return {
        "bytes_done": sum(entry[1] for entry in selected),
        "bytes_total": sum(entry[2] for entry in selected),
        "label": selected[0][0] if len(selected) == 1 else f"{len(selected)} model files",
        "scope": "active_files",
    }


def _active_downloads() -> dict[str, Any] | None:
    from tqdm.std import tqdm
    with tqdm.get_lock():
        bars = list(tqdm._instances)
    return download_snapshot(bars)


class TrainingDownloadProgress:
    """Forward trainer events, adding byte progress only during its model-loading phase."""

    def __init__(
        self,
        emit: Callable[[dict], None],
        sample = _active_downloads,
        interval: float = 0.5,
    ):
        self._emit = emit
        self._sample = sample
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active = False
        self._generation = 0
        self._last = None
        self._thread = None

    def report(self, event: dict) -> None:
        with self._lock:
            kind = event.get("type")
            if kind == "model_load_started":
                self._active = True
                self._generation += 1
                self._last = None
            elif kind in ("model_load_completed", "preparing", "progress", "complete", "error"):
                self._active = False
                self._generation += 1
                self._last = None
            self._emit(event)

    def poll(self) -> None:
        with self._lock:
            if not self._active:
                return
            generation = self._generation
        progress = self._sample()
        with self._lock:
            if not self._active or generation != self._generation or progress == self._last:
                return
            self._last = progress
            self._emit({"type": "model_download_progress", "download_progress": progress})

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.poll()
            except Exception:  # noqa: BLE001 -- progress must never stop a training run
                pass

    def __enter__(self):
        self._thread = account_thread(target = self._run, daemon = True)
        self._thread.start()
        return self.report

    def __exit__(self, *_exc):
        with self._lock:
            self._active = False
            self._generation += 1
        self._stop.set()
        self._thread.join(timeout = 2)
