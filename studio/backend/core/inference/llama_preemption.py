# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pausing a generation instead of losing it.

One llama-server holds one KV cache. Studio launches it with ``--parallel N
--kv-unified -c N``, which is N cells TOTAL while every slot is told it has N.
The server admits any prompt that fits on its own, so chats that each fit
individually collide, and the overflow handler errors EVERY processing slot at
once rather than the one that overran. Four parallel chats died together that
way on 2026-09-01.

The fix is to pause a chat rather than let the pool overflow. Pausing means
aborting only the UPSTREAM llama-server stream: the Studio-side generator, the
tool-loop ledgers, the conversation and the client's SSE response all stay alive
in process, and the request is re-opened with the partial reply as a trailing
assistant turn plus ``continue_final_message``.

This module owns the mechanics of that pause. It deliberately does NOT decide
when to pause or who to pause: that policy arrives through
:class:`PreemptionPolicy`.

Two properties here are load-bearing and neither is obvious:

* The signal is **separate from the cancel event**. ``_TrackedCancel`` hands one
  ``threading.Event`` to both the cancel registry and ``active_generations``, so
  setting it means "the user pressed Stop" at six different terminal consumers,
  from writing ``status="cancelled"`` on the durable run to recording in-flight
  tools as cancelled. A pause that reused it would be indistinguishable from an
  abandonment.
* The signal can be **deferred**. Tool execution is not interruptible: between
  the point where calls materialise and the point where their results are
  appended, an abort would lose the results of tools that already ran, or run
  them twice on resume. :meth:`PreemptSignal.unsafe_window` holds a pending
  request invisible until the window closes.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

__all__ = [
    "LlamaStreamPreempted",
    "NullPreemptionPolicy",
    "PreemptSignal",
    "PreemptionPolicy",
    "StreamCheckpoint",
    "PREEMPT_ENV",
    "preemption_enabled",
]


PREEMPT_ENV = "UNSLOTH_LLAMA_ADMISSION_PREEMPT"

DEFAULT_PREEMPT_ENABLED = True

# A resume is cheap (the prefix cache usually still holds the prompt) but not
# free, so a pathological loop that pauses the same chat forever is bounded.
# Deliberately far above _MAX_LENGTH_CONTINUATIONS: that one caps how often a
# model may be asked to finish its own sentence, which is a quality judgement,
# while this caps churn under contention, which is a capacity one.
DEFAULT_MAX_PREEMPT_RESUMES = 32


class LlamaStreamPreempted(Exception):
    """The upstream stream was aborted to free KV, not abandoned.

    Distinct from ``_LlamaStreamCancelled`` on purpose. A cancel ends the turn;
    this one is expected to be caught and resumed, so anything that treats a
    dead stream as failure must not see it.
    """


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def preemption_enabled() -> bool:
    """Rollout switch, mirroring the other admission escape hatches.

    Read per call rather than cached at import: the tests and the settings route
    both flip it at runtime.
    """
    return _bool_env(PREEMPT_ENV, DEFAULT_PREEMPT_ENABLED)


class PreemptSignal:
    """A pause request that tool execution can hold off.

    ``request()`` asks the stream to stop. ``is_set()`` is what the stream
    plumbing polls, and it reports False while an unsafe window is open even
    though a request is pending, so a preempt lands at a safe point or not at
    all. Nothing is lost by deferring: the request stays pending and becomes
    visible the moment the window closes.
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.RLock()
        self._depth = 0
        self._pending = False
        self._reason: Optional[str] = None

    # ── asking ──────────────────────────────────────────────────

    def request(self, reason: str = "kv_pressure") -> None:
        with self._lock:
            self._pending = True
            self._reason = reason
            if self._depth == 0:
                self._event.set()

    def clear(self) -> None:
        """Forget a pending request, deferred or not. Called on resume."""
        with self._lock:
            self._pending = False
            self._reason = None
            self._event.clear()

    @property
    def reason(self) -> Optional[str]:
        with self._lock:
            return self._reason

    @property
    def pending(self) -> bool:
        """A request exists, whether or not it is currently visible."""
        with self._lock:
            return self._pending

    @property
    def deferred(self) -> bool:
        with self._lock:
            return self._depth > 0

    # ── what the stream plumbing sees ───────────────────────────

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout = timeout)

    # ── safe points ─────────────────────────────────────────────

    class _Window:
        __slots__ = ("_signal",)

        def __init__(self, signal: "PreemptSignal"):
            self._signal = signal

        def __enter__(self):
            self._signal._enter_unsafe()
            return self._signal

        def __exit__(self, *exc):
            self._signal._exit_unsafe()
            return False

    def unsafe_window(self) -> "PreemptSignal._Window":
        """Hold any pause request invisible for the duration.

        Nests: an inner window closing does not re-expose the request while an
        outer one is still open.
        """
        return PreemptSignal._Window(self)

    def _enter_unsafe(self) -> None:
        with self._lock:
            self._depth += 1
            # A request that arrived before the window opened is hidden too. It
            # was not acted on yet, and acting on it now would be exactly the
            # mid-execution abort the window exists to prevent.
            self._event.clear()

    def _exit_unsafe(self) -> None:
        with self._lock:
            if self._depth > 0:
                self._depth -= 1
            if self._depth == 0 and self._pending:
                self._event.set()


@dataclass
class StreamCheckpoint:
    """What a paused attempt had produced when it was cut.

    Written by the preemptor from the live stream accumulators. Deliberately not
    read back off the thread's trailing assistant row: an aborted run never
    writes one, so that row lags the stream by a whole attempt and resuming from
    it would replay text the user has already seen.
    """

    visible_text: str = ""
    reasoning_text: str = ""
    # Carried rather than re-derived. The archive is content-hash idempotent but
    # the `context_truncated` SSE event is not, so a truncation observed on the
    # aborted attempt has to be emitted exactly once, by whoever resumes.
    pending_truncations: list = field(default_factory = list)
    # Tokens the aborted attempt really did produce, to be charged once.
    charged_tokens: int = 0
    resumes: int = 0
    reason: Optional[str] = None

    def has_resume_point(self) -> bool:
        """Whether there is anything to continue from.

        Empty means the pause landed before the first token. There is nothing to
        extend, and ``continue_final_message`` refuses an empty assistant turn,
        so such a request is re-issued whole rather than continued.
        """
        return bool(self.visible_text.strip())


@runtime_checkable
class PreemptionPolicy(Protocol):
    """Supplied by the admission side; this module only calls it.

    ``await_resume`` returning False means "stop waiting and finish the turn",
    so a policy that dies or times out degrades to today's behaviour instead of
    hanging the chat.
    """

    def should_preempt(self) -> bool: ...

    def on_preempted(self, checkpoint: StreamCheckpoint) -> None: ...

    def await_resume(self, timeout: Optional[float] = None) -> bool: ...

    def on_resumed(self) -> None: ...


class NullPreemptionPolicy:
    """Never pauses. The default, so every existing call site is unchanged."""

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint: StreamCheckpoint) -> None:
        return None

    def await_resume(self, timeout: Optional[float] = None) -> bool:
        return True

    def on_resumed(self) -> None:
        return None
