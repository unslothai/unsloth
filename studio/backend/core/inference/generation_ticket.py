# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cancels one image run via a context ticket; asyncio.to_thread carries it, other runs untouched"""

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

    def cancelled(self) -> bool:
        with self._lock:
            return self._stopped or any(event.is_set() for event in self._events)

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            events = list(self._events)
        for event in events:
            event.set()


class TicketStopEvent(threading.Event):
    def __init__(self, ticket: GenerationTicket) -> None:
        super().__init__()
        self._ticket = ticket

    def set(self) -> None:
        super().set()
        self._ticket.stop()


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
