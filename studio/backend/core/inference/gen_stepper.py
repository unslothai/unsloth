# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Advance a blocking generator off the event loop, one step per await, on its own thread."""

from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
import weakref

# A stepper whose consumer stops asking (slow client) frees its thread after this.
_IDLE_EXIT_S = 30.0


def _resolve(future: asyncio.Future, result, error) -> None:
    if future.cancelled():
        return
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)


def _work(ref, requests: queue.SimpleQueue) -> None:
    # Holds the stepper (so the generator) only while a step runs: a dropped stream's generator is
    # finalized as soon as its consumer lets go, as with a thread pool.
    while True:
        try:
            loop, future, context, default = requests.get(timeout = _IDLE_EXIT_S)
        except queue.Empty:
            stepper = ref()
            if stepper is None:
                return
            with stepper._lock:
                if requests.empty():
                    stepper._running = False
                    return
            del stepper
            continue
        stepper = ref()
        if stepper is None:
            return  # nobody can await this step any more
        try:
            result, error = context.run(next, stepper._gen, default), None
        except BaseException as exc:
            result, error = None, exc
        del stepper
        try:
            loop.call_soon_threadsafe(_resolve, future, result, error)
        except RuntimeError:
            pass  # loop closed: nobody is waiting
        del loop, future, context, default, result, error


class ThreadStepper:
    """``await stepper.next(default)`` behaves like ``await asyncio.to_thread(next, gen, default)``.

    Same context copy, exception and cancellation behaviour (a cancelled await abandons the step,
    which still finishes), but every step runs on one dedicated thread instead of a thread-pool
    round trip, and steps run one at a time in request order.
    """

    __slots__ = ("_gen", "_lock", "_requests", "_running", "__weakref__")

    def __init__(self, gen):
        self._gen = gen
        self._lock = threading.Lock()
        self._requests = queue.SimpleQueue()
        self._running = False

    async def next(self, default):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = (loop, future, contextvars.copy_context(), default)
        with self._lock:
            if not self._running:
                threading.Thread(
                    target = _work,
                    args = (weakref.ref(self), self._requests),
                    name = "studio-gen-step",
                    daemon = True,
                ).start()
                self._running = True
            self._requests.put(request)
        return await future
