# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stop one image generation without touching any other run of the same account.

A caller binds a ticket in its context; ``asyncio.to_thread`` copies the context into the engine
thread, where the engine hands the ticket its per-generation cancel event. Stopping the ticket
sets only that event, whether the run is still queued or already denoising.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Iterator, Optional


class GenerationTicket:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[threading.Event] = []
        self._stopped = False

    def bind(self, event: threading.Event) -> None:
        with self._lock:
            self._events.append(event)
            stopped = self._stopped
        if stopped:
            event.set()

    def stop(self) -> bool:
        with self._lock:
            self._stopped = True
            events = list(self._events)
        for event in events:
            event.set()
        return bool(events)


_ticket: contextvars.ContextVar[Optional[GenerationTicket]] = contextvars.ContextVar(
    "generation_ticket", default = None
)


def bind_generation_cancel(event: threading.Event) -> None:
    """Called by an engine with the cancel event of the generation it is about to run."""
    ticket = _ticket.get()
    if ticket is not None:
        ticket.bind(event)


@contextmanager
def generation_ticket(ticket: GenerationTicket) -> Iterator[GenerationTicket]:
    token = _ticket.set(ticket)
    try:
        yield ticket
    finally:
        _ticket.reset(token)
