# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Laya on CUDA, ROCm or XPU, in a spawn child: a CUDA context is only returned when its process ends."""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Callable

_CTX = mp.get_context("spawn")
LOAD_TIMEOUT_S = 600.0
DECIDE_TIMEOUT_S = 300.0
CLOSE_TIMEOUT_S = 10.0


class LayaWorkerError(RuntimeError):
    pass


def run_laya_worker(conn, root: str, checkpoint, device: str) -> None:
    from core.systemone import laya_runtime

    try:
        agent, _ = laya_runtime._build_laya(Path(root), checkpoint, device)
    except BaseException as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", agent.device.type))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message[0] == "close":
            return
        _, state, questions = message
        try:
            # The device rides along: an out-of-memory decision moves the model to CPU here, not in the parent.
            answer = laya_runtime._predict(agent, state, questions)
            conn.send(("ok", (answer, agent.device.type)))
        except ValueError as exc:
            conn.send(("invalid", str(exc)))
        except BaseException as exc:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class LayaWorker:
    def __init__(self, root: Path, checkpoint, device: str, cancelled: Callable[[], bool]):
        self._broken: str | None = None
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )
        from utils.process_lifetime import (
            adopt_pid,
            is_process_shutting_down,
            spawn_on_lifetime_thread,
        )

        if is_process_shutting_down():
            raise LayaWorkerError("Studio is shutting down; the Laya worker was not started.")
        self._conn, child = _CTX.Pipe()
        with native_path_secret_removed_for_child_start():
            self._process = _CTX.Process(
                target = run_without_native_path_secret,
                args = (run_laya_worker, child, str(root), checkpoint, device),
                name = "systemone-laya",
                daemon = True,
            )
            # Linux PDEATHSIG follows the spawning thread, and this runs on a loader thread that ends after the load.
            spawn_on_lifetime_thread(self._process.start)
        adopt_pid(self._process.pid)
        child.close()
        try:
            deadline = time.monotonic() + LOAD_TIMEOUT_S
            while not self._conn.poll(1.0):
                if is_process_shutting_down():
                    raise LayaWorkerError("The Laya worker was stopped while it loaded.")
                if cancelled():
                    raise LayaWorkerError("A training run took the GPU while Laya loaded.")
                if time.monotonic() > deadline:
                    raise LayaWorkerError(
                        f"The Laya worker did not answer within {LOAD_TIMEOUT_S:.0f}s"
                    )
            kind, payload = self._receive(0)
            if kind != "ready":
                raise LayaWorkerError(payload)
            if cancelled():
                raise LayaWorkerError("A training run took the GPU while Laya loaded.")
        except BaseException:
            # A child still building cannot read "close", and training may already be claiming the GPU.
            self.close(grace = 0)
            raise
        self.device = payload

    def _receive(self, timeout: float):
        try:
            if not self._conn.poll(timeout):
                raise LayaWorkerError(f"The Laya worker did not answer within {timeout:.0f}s")
            return self._conn.recv()
        except (EOFError, OSError):
            raise LayaWorkerError(
                f"The Laya worker exited (code {self._process.exitcode})"
            ) from None

    def predict(self, state, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        # After a timeout the late answer is still in the pipe, so the worker is never asked again.
        if self._broken is not None:
            raise LayaWorkerError(self._broken)
        try:
            self._conn.send(("decide", state, questions))
            kind, payload = self._receive(DECIDE_TIMEOUT_S)
        except (OSError, ValueError, LayaWorkerError) as exc:
            self._broken = (
                str(exc)
                if isinstance(exc, LayaWorkerError)
                else f"The Laya worker exited (code {self._process.exitcode})"
            )
            raise LayaWorkerError(self._broken) from None
        if kind == "invalid":
            raise ValueError(payload)
        if kind != "ok":
            raise LayaWorkerError(payload)
        answer, self.device = payload
        return answer

    def close(self, grace: float | None = None) -> None:
        try:
            self._conn.send(("close",))
        except (OSError, ValueError):
            pass
        self._process.join(CLOSE_TIMEOUT_S if grace is None else grace)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(CLOSE_TIMEOUT_S)
        self._conn.close()
        if not self._process.is_alive():
            from utils.process_lifetime import forget_pid
            forget_pid(self._process.pid)
