# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who pauses, who keeps decoding, and when the room is actually back."""

from __future__ import annotations

import asyncio
import os
import math
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from core.inference.llama_admission import (
    LlamaAdmissionLease,
    _bool_env,
)


_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}

from loggers import get_logger

_log = get_logger(__name__)


PREEMPT_ENV = "UNSLOTH_LLAMA_ADMISSION_PREEMPT"
DEFAULT_PREEMPT_ENABLED = False


def _int_env(name: str, default: int) -> int:
    """Positive integer from the environment, else the default. Never raises."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


DEFAULT_PREEMPT_BUFFER_PER_SLOT = 192
DEFAULT_PREEMPT_BUFFER_MIN_TOKENS = 256

# Backstop for an announced prefill that never happens. The stream waits this long for a first
# token (`_DEFAULT_FIRST_TOKEN_TIMEOUT_S`), so a shorter figure drops a live reserve.
PENDING_PREFILL_TTL_S = 1200.0

# Bounds churn under contention: this many pauses in a row with no new output between them end the
# turn.
DEFAULT_MAX_STALLED_RESUMES = 4


def next_stall_count(stalled: int, new_tokens: int) -> int:
    """Consecutive pauses without output, after one that decoded ``new_tokens``."""
    return 0 if int(new_tokens or 0) > 0 else max(0, int(stalled or 0)) + 1


def resumes_stalled(stalled: int) -> bool:
    """Whether ``stalled`` pauses in a row without output should end the turn."""
    return int(stalled or 0) >= DEFAULT_MAX_STALLED_RESUMES


MAX_NESTED_RESUMES = 128


def resumes_exhausted(resumes: int) -> bool:
    """Whether a turn already resumed ``resumes`` times must not nest another."""
    return int(resumes or 0) + 1 >= MAX_NESTED_RESUMES


# A chat preempted this many times in a row outranks newest-first, so repeatedly losing does not
# become never finishing.
PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS = 3

# What a chat running ALONE must leave clear. Without it a chat past the shared ceiling can never
# be admitted OR resumed, and hangs until its client gives up.
SOLO_MARGIN_DIVISOR = 200

DEFAULT_RESUME_WAIT_TIMEOUT_S = 90.0

# Absolute bound on one resume wait, in stall timeouts, for the one case the stall clock cannot
# see: a cache that keeps moving but never has room for THIS chat.
MAX_RESUME_WAIT_MULTIPLE = 80


class LlamaStreamPreempted(Exception):
    """The upstream stream was aborted to free KV, not abandoned.

    Unlike ``_LlamaStreamCancelled`` it is caught and resumed, so anything that treats a
    dead stream as failure must not see it.
    """


class PreemptSignal:
    """A pause request that tool execution can hold off."""

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.RLock()
        self._depth = 0
        self._pending = False
        self._reason: Optional[str] = None
        # When the stream last saw its prefill advance, on `time.monotonic()`; 0 is never.
        self._prefill_progress_at = 0.0

    def note_prefill_progress(self) -> None:
        """The stream this signal pauses saw its prompt processing move forward."""
        self._prefill_progress_at = time.monotonic()

    @property
    def prefill_progress_at(self) -> float:
        return self._prefill_progress_at

    def request(self, reason: str = "kv_pressure") -> None:
        with self._lock:
            self._pending = True
            self._reason = reason
            if self._depth == 0:
                self._event.set()

    def set(self, reason: str = "kv_pressure") -> None:
        self.request(reason)

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

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout = timeout)

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
        """Hold any pause request invisible for the duration. Nests."""
        return PreemptSignal._Window(self)

    def _enter_unsafe(self) -> None:
        with self._lock:
            self._depth += 1
            self._event.clear()

    def _exit_unsafe(self) -> None:
        with self._lock:
            if self._depth > 0:
                self._depth -= 1
            if self._depth == 0 and self._pending:
                self._event.set()


@dataclass
class StreamCheckpoint:
    """What a paused attempt had produced when it was cut."""

    visible_text: str = ""
    reasoning_text: str = ""
    charged_tokens: int = 0
    resumes: int = 0
    reason: Optional[str] = None

    def has_resume_point(self) -> bool:
        """Whether there is anything to continue from."""
        return bool(self.visible_text.strip())

    def has_reasoning_resume_point(self) -> bool:
        """Whether there is a thought to continue, absent any visible prose."""
        return not self.visible_text.strip() and bool(self.reasoning_text.strip())

    def kept_chars(self) -> int:
        """Characters carried across the pause, prose or thought."""
        if self.has_resume_point():
            return len(self.visible_text)
        return len(self.reasoning_text) if self.has_reasoning_resume_point() else 0


@runtime_checkable
class PreemptionPolicy(Protocol):
    """Supplied by the admission side; this module only calls it."""

    def should_preempt(self) -> bool: ...

    def on_preempted(self, checkpoint: StreamCheckpoint) -> None: ...

    def await_resume(
        self,
        timeout: Optional[float] = None,
        *,
        cancel_event = None,
    ) -> bool: ...

    def on_resumed(self) -> None: ...

    def on_declined(self) -> None: ...


class DeferredPreemptionPolicy:
    """A policy handed to the stream before the real one can exist."""

    __slots__ = ("_inner",)

    def __init__(self, inner = None):
        self._inner = inner

    def bind(self, policy) -> None:
        self._inner = policy

    @property
    def bound(self) -> bool:
        return self._inner is not None

    def should_preempt(self) -> bool:
        return False if self._inner is None else bool(self._inner.should_preempt())

    def on_preempted(self, checkpoint: StreamCheckpoint) -> None:
        if self._inner is not None:
            self._inner.on_preempted(checkpoint)

    def await_resume(
        self,
        timeout: Optional[float] = None,
        *,
        cancel_event = None,
    ) -> bool:
        if self._inner is None:
            return False
        if cancel_event is None:
            return bool(self._inner.await_resume(timeout))
        try:
            return bool(self._inner.await_resume(timeout, cancel_event = cancel_event))
        except TypeError:
            return bool(self._inner.await_resume(timeout))

    def on_resumed(self) -> None:
        if self._inner is not None:
            self._inner.on_resumed()

    def on_declined(self) -> None:
        if self._inner is not None:
            self._inner.on_declined()

    def restart(self) -> None:
        if self._inner is not None:
            self._inner.restart()


class NullPreemptionPolicy:
    """Never pauses."""

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint: StreamCheckpoint) -> None:
        return None

    def await_resume(
        self,
        timeout: Optional[float] = None,
        *,
        cancel_event = None,
    ) -> bool:
        return True

    def on_resumed(self) -> None:
        return None

    def on_declined(self) -> None:
        return None


class ParticipantState:
    """Where a generation is, which decides whether it may be preempted."""

    QUEUED = "queued"
    DECODING = "decoding"
    PARKED_ON_TOOL = "parked_on_tool"
    TOOLS_RUNNING = "tools_running"
    PREEMPTING = "preempting"
    # Raw passthrough and the Responses surface: nothing to resume, so aborting is a CANCEL and it
    # is never a victim.
    STREAMING_RAW = "streaming_raw"
    PAUSED = "paused"
    RESUMING = "resuming"
    DONE = "done"


_HOLDS_KV = frozenset(
    {
        ParticipantState.DECODING,
        ParticipantState.PARKED_ON_TOOL,
        ParticipantState.TOOLS_RUNNING,
        ParticipantState.PREEMPTING,
        ParticipantState.STREAMING_RAW,
        ParticipantState.RESUMING,
    }
)

_PREEMPTABLE = frozenset(
    {
        ParticipantState.DECODING,
        ParticipantState.PARKED_ON_TOOL,
    }
)

_PARKED_ON_A_TOOL = frozenset(
    {
        ParticipantState.PARKED_ON_TOOL,
        ParticipantState.TOOLS_RUNNING,
    }
)

_DECODES_WHEN_TOKENS_ARRIVE = frozenset(
    {
        ParticipantState.TOOLS_RUNNING,
        ParticipantState.PARKED_ON_TOOL,
        ParticipantState.RESUMING,
    }
)


def preemption_enabled() -> bool:
    """Whether the operator opted in with ``UNSLOTH_LLAMA_ADMISSION_PREEMPT=1``."""
    return _bool_env(PREEMPT_ENV, DEFAULT_PREEMPT_ENABLED)


def preemption_buffer_tokens(
    budget: int,
    *,
    draft_tokens: int = 0,
    slots: int = 1,
    batch_tokens: int = 0,
    pending_prefill: int = 0,
) -> int:
    """Tokens held clear of ``budget``."""
    if budget <= 0:
        return 0
    per_slot = _int_env("UNSLOTH_LLAMA_PREEMPT_BUFFER_PER_SLOT", DEFAULT_PREEMPT_BUFFER_PER_SLOT)
    slot_count = max(1, int(slots or 1))
    reserve = max(DEFAULT_PREEMPT_BUFFER_MIN_TOKENS, per_slot * slot_count)
    # And room for the batch llama.cpp is processing: the cache fails when the next BATCH does not
    # fit, not when it is full.
    n_batch = max(0, int(batch_tokens or 0))
    reserve = max(reserve, min(n_batch, max(0, int(pending_prefill or 0))))
    reserve += max(0, int(draft_tokens or 0)) * slot_count
    return min(reserve, max(1, budget // 2))


@dataclass(**_SLOTS)
class Participant:
    """One in-flight generation, from the preemptor's point of view."""

    gen_id: str
    seq: int
    lease: Optional[LlamaAdmissionLease] = None
    tokens: int = 0
    # What admission charged; live growth is ADDED to it, never replacing it.
    base_tokens: int = 0
    prompt_tokens: int = 0
    measured: bool = False
    state: str = ParticipantState.DECODING
    consecutive_preemptions: int = 0
    # Set when this generation must stop. The caller aborts ONLY the upstream stream and keeps its
    # generator, ledgers, conversation and SSE response, which a cancel cannot say.
    preempt_event: PreemptSignal = field(default_factory = PreemptSignal)
    preempt_chosen_at: float = 0.0
    cells_reclaimed: bool = False
    # Erased but re-charged before the release ran, so it counts again: the next round boundary
    # still replays the whole prompt, and must announce it so.
    replay_pending: bool = False
    park_seq: int = 0
    park_charge_seq: Optional[int] = None
    # The ONLY thing that puts the batch term in the buffer, so it must be set before the request
    # carrying the prompt is sent.
    pending_prefill: int = 0
    pending_prefill_at: float = 0.0
    generated_seen: int = 0
    measured_at_seq: Optional[int] = None
    unconfirmed_growth: int = 0
    growth_at_seq: Optional[int] = None
    charged_tokens: int = 0
    charged_prompt_tokens: int = 0
    # The state this holder was in when the sweep chose it. Only read while PREEMPTING, so a value
    # left behind by an earlier round is never consulted.
    preempted_from: str = ""
    awaiting_victims: List[str] = field(default_factory = list)

    def in_reading_tokens(self) -> int:
        """The part of the charge a reading taken now can hold: none until measured, and not a
        round's growth until a reading started after the round lands."""
        if not self.measured:
            return 0
        return max(0, self.tokens - self.unconfirmed_growth)

    def on_top_tokens(self) -> int:
        """The part of the charge no reading holds yet, counted on top of the resident figure."""
        return max(0, self.tokens - self.in_reading_tokens())

    def replay_tokens(self) -> int:
        """What a resume sends back as prompt: the prompt, with every paused partial folded
        into it by `note_replayed`. The rest of the charge is output room, not replayed."""
        return self.prompt_tokens if self.prompt_tokens > 0 else self.tokens

    def resident_tokens(self, generated: int) -> int:
        """Cells this holder occupies now that its prompt is in the cache."""
        generated = max(0, int(generated or 0))
        if self.prompt_tokens <= 0:
            return self.base_tokens + generated
        return self.prompt_tokens + generated

    def prefill_pending(self, now: float) -> int:
        """Outstanding prefill worth reserving a batch for."""
        if self.pending_prefill <= 0 or not self.holds_kv:
            return 0
        since = max(
            self.pending_prefill_at,
            float(getattr(self.preempt_event, "prefill_progress_at", 0.0) or 0.0),
        )
        if now - since > PENDING_PREFILL_TTL_S:
            return 0
        return self.pending_prefill

    def announce_prefill(self, tokens: int) -> None:
        """A prompt chunk is about to be submitted for `tokens` tokens."""
        self.pending_prefill = max(0, int(tokens or 0))
        self.pending_prefill_at = time.monotonic()

    def prefill_done(self) -> None:
        """The prompt is in the cache (or will never be sent). Drop the reserve."""
        self.pending_prefill = 0
        self.pending_prefill_at = 0.0

    @property
    def promoted(self) -> bool:
        return self.consecutive_preemptions >= PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS

    @property
    def holds_kv(self) -> bool:
        return self.state in _HOLDS_KV and not self.cells_reclaimed

    @property
    def preemptable(self) -> bool:
        return self.state in _PREEMPTABLE

    @property
    def parked_on_a_tool(self) -> bool:
        """Whether an idle-slot erase takes THIS holder's cells."""
        if self.state in _PARKED_ON_A_TOOL:
            return True
        return (
            self.state == ParticipantState.PREEMPTING and self.preempted_from in _PARKED_ON_A_TOOL
        )


@dataclass(frozen = True, **_SLOTS)
class PreemptionSnapshot:
    key: str
    budget: int
    buffer: int
    committed: int
    decoding: int
    paused: int
    parked: int
    slots: int = 1
    tools_running: int = 0
    prefilling: int = 0
    holders: int = 0


# How long the last good `/slots` reading outlives a failed one.
_RESIDENT_HOLD_S = 5.0


class PreemptionController:
    """Victim choice and the epoch, for one llama-server backend."""

    __slots__ = (
        "key",
        "_lock",
        "_participants",
        "_seq",
        "_budget",
        "_kv_unified",
        "_draft_tokens",
        "_slots",
        "_batch_tokens",
        "_resident",
        "_resident_seq",
        "_last_sample_seq",
        "_last_sample_started",
        "_resident_at",
        "_clock",
        "_resume_tickets",
        "_replaced_by",
        "_reclaimable",
        "_residency_probe",
        "_drift_logged_at",
        "_progress_tokens",
    )

    def __init__(self, key: str):
        self.key = key
        self._lock = threading.Lock()
        self._participants: Dict[str, Participant] = {}
        self._seq = 0
        self._drift_logged_at = 0.0
        self._budget = 0
        # Preemption reclaims only where an idle slot's cells can be purged, which upstream gates
        # on kv_unified ALONE (try_clear_idle_slots, server-context.cpp:1656).
        self._kv_unified = False
        self._draft_tokens = 0
        self._batch_tokens = 0
        self._reclaimable = 0
        self._slots = 1
        self._resident: Optional[int] = None
        self._resident_seq = 0
        self._last_sample_seq = 0
        self._last_sample_started = 0
        self._resident_at = 0.0
        self._clock = time.monotonic
        self._resume_tickets: "OrderedDict[str, int]" = OrderedDict()
        self._replaced_by: Dict[str, Optional[List[str]]] = {}
        self._residency_probe: Optional[Callable[[], None]] = None
        self._progress_tokens = 0

    def configure(
        self,
        *,
        budget: Optional[int] = None,
        kv_unified: Optional[bool] = None,
        draft_tokens: Optional[int] = None,
        slots: Optional[int] = None,
        batch_tokens: Optional[int] = None,
    ) -> None:
        """Re-read the cache this backend actually allocated."""
        with self._lock:
            if budget is not None:
                self._budget = max(0, int(budget or 0))
            if kv_unified is not None:
                self._kv_unified = bool(kv_unified)
            if draft_tokens is not None:
                self._draft_tokens = max(0, int(draft_tokens or 0))
            if slots is not None:
                self._slots = max(1, int(slots or 1))
            if batch_tokens is not None:
                self._batch_tokens = max(0, int(batch_tokens or 0))

    @property
    def active(self) -> bool:
        """Whether preemption can do anything on this backend."""
        with self._lock:
            return bool(self._kv_unified) and self._budget > 0 and preemption_enabled()

    def register(
        self,
        gen_id: str,
        *,
        lease: Optional[LlamaAdmissionLease] = None,
        tokens: int = 0,
        state: str = ParticipantState.DECODING,
        signal: Optional[PreemptSignal] = None,
        prompt_tokens: Optional[int] = None,
    ) -> Participant:
        """``signal`` MUST be the object the stream polls."""
        with self._lock:
            existing = self._participants.get(gen_id)
            if existing is not None:
                return existing
            self._seq += 1
            participant = Participant(
                gen_id = gen_id,
                seq = self._seq,
                lease = lease,
                tokens = max(0, int(tokens or 0)),
                base_tokens = max(0, int(tokens or 0)),
                prompt_tokens = min(max(0, int(prompt_tokens or 0)), max(0, int(tokens or 0))),
                state = state,
                **({} if signal is None else {"preempt_event": signal}),
            )
            participant.charged_tokens = participant.tokens
            participant.charged_prompt_tokens = participant.prompt_tokens
            # Its whole prompt is about to be prefilled. Announced HERE rather than by the caller,
            # so a sweep firing in between plans against the raised buffer.
            participant.announce_prefill(participant.prompt_tokens or participant.tokens)
            self._participants[gen_id] = participant
            return participant

    def unregister(self, gen_id: str) -> None:
        """Drop a finished generation."""
        with self._lock:
            self._participants.pop(gen_id, None)
            self._resume_tickets.pop(gen_id, None)
            self._replaced_by.pop(gen_id, None)

    def begin_resume(self, gen_id: str, want: int) -> None:
        """Take a place in the resume line, before asking whether there is room."""
        with self._lock:
            self._resume_tickets.pop(gen_id, None)
            self._resume_tickets[gen_id] = max(0, int(want or 0))

    def end_resume(self, gen_id: str) -> None:
        """Leave the resume line: granted, given up or cancelled. Idempotent."""
        with self._lock:
            self._resume_tickets.pop(gen_id, None)

    def resume_queue_depth(self) -> int:
        """How many waits are in the resume line."""
        with self._lock:
            return len(self._resume_tickets)

    def _resume_reserved_locked(self, gen_id: str) -> int:
        """Tokens the waits AHEAD of ``gen_id`` are coming back for."""
        total = 0
        for queued, want in self._resume_tickets.items():
            if queued == gen_id:
                break
            total += want
        return total

    def _solo_ceiling_locked(self) -> int:
        """The cache less what a lone chat still needs clear to keep running."""
        margin = self._draft_tokens + 1 + max(64, self._budget // SOLO_MARGIN_DIVISOR)
        return max(1, self._budget - margin)

    def outgrew_the_shared_ceiling(self, want: int) -> bool:
        """Whether `want` can never fit beside anyone, however much is evicted."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return False
            pending = self._pending_prefill_locked() + max(0, int(want or 0))
            ceiling = max(0, self._budget - self._buffer_locked(pending = pending))
            return int(want or 0) > ceiling

    def cannot_ever_fit(self, want: int) -> bool:
        """Whether `want` exceeds the cache itself, so not even running alone helps."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return False
            return int(want or 0) > self._solo_ceiling_locked()

    def room_for(self, gen_id: str, want: int) -> bool:
        """Whether a paused generation may start again yet."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return True
            return self._room_for_locked(gen_id, want)

    def try_grant_resume(self, gen_id: str, want: int) -> bool:
        """Decide there is room AND take it, without letting go of the lock between."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return True
            if not self._room_for_locked(gen_id, want, self._resume_reserved_locked(gen_id)):
                return False
            participant = self._participants.get(gen_id)
            if participant is not None:
                need = max(0, int(want or 0))
                if need > 0:
                    participant.tokens = need
                    participant.base_tokens = need
                if participant.prompt_tokens > 0:
                    participant.prompt_tokens = max(participant.prompt_tokens, need)
                participant.measured = False
                participant.measured_at_seq = None
                participant.unconfirmed_growth = 0
                participant.growth_at_seq = None
                participant.state = ParticipantState.RESUMING
                participant.announce_prefill(need)
            return True

    def note_resume_failed(self, gen_id: str) -> None:
        """Give back a grant whose resume never happened."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None and participant.state in (
                ParticipantState.RESUMING,
                ParticipantState.DECODING,
            ):
                participant.state = ParticipantState.PAUSED
                participant.prefill_done()

    def _my_resident_locked(self, gen_id: str) -> int:
        """Of the residency reading, the part this generation can claim as its own."""
        participant = self._participants.get(gen_id)
        if participant is None or not participant.holds_kv:
            return 0
        return participant.in_reading_tokens()

    def _room_for_locked(
        self,
        gen_id: str,
        want: int,
        reserved: int = 0,
    ) -> bool:
        """The arithmetic behind `room_for`, callable by a holder of the lock."""
        # `want` REPLACES this generation's own announcement: saying yes here causes the prefill,
        # and a chat that already announced must not be charged twice.
        pending = self._pending_prefill_locked(exclude = gen_id) + max(0, int(want or 0))
        ceiling = max(0, self._budget - self._buffer_locked(pending = pending))
        holders = [p for gid, p in self._participants.items() if p.holds_kv and gid != gen_id]
        measured = sum(p.in_reading_tokens() for p in holders)
        unmeasured = sum(p.on_top_tokens() for p in holders)
        others = measured + unmeasured
        if self._resident is not None:
            occupied = self._resident - self._reclaimable
            others = max(measured, occupied - self._my_resident_locked(gen_id)) + unmeasured
        need = max(0, int(want or 0))
        others = max(0, others)
        reserved = max(0, int(reserved or 0))
        if others + need + reserved <= ceiling:
            return True
        # Outgrew the shared ceiling: it must run alone, or wait for room nothing can make.
        if need > ceiling and others == 0 and reserved == 0:
            return need <= self._solo_ceiling_locked()
        return False

    def observe(
        self,
        gen_id: str,
        generated: int,
        *,
        plan: bool = True,
    ) -> List["Participant"]:
        """Live growth during generation, and the eviction check that follows it."""
        if not preemption_enabled():
            return []
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None:
                reported = max(0, int(generated or 0))
                previous = participant.generated_seen
                participant.generated_seen = reported
                self._progress_tokens += (reported - previous) if reported >= previous else reported
                participant.tokens = participant.resident_tokens(reported)
                if reported > 0:
                    if not participant.measured and participant.measured_at_seq is None:
                        self._resident_seq += 1
                        participant.measured_at_seq = self._resident_seq
                    if participant.unconfirmed_growth > 0 and participant.growth_at_seq is None:
                        self._resident_seq += 1
                        participant.growth_at_seq = self._resident_seq
                    participant.cells_reclaimed = False
                    participant.replay_pending = False
                    participant.prefill_done()
                if participant.state in _DECODES_WHEN_TOKENS_ARRIVE:
                    participant.state = ParticipantState.DECODING
        if not plan:
            return []
        return self.plan_preemptions(needed = 0)

    def set_residency_probe(self, probe: Optional[Callable[[], None]]) -> None:
        """Register a way to re-read the cache on demand."""
        if probe is not None and not preemption_enabled():
            return
        self._residency_probe = probe

    def refresh_residency(self) -> None:
        """Re-read the cache now, if a probe was registered. Never raises."""
        probe = self._residency_probe
        if probe is None:
            return
        try:
            probe()
        except Exception:
            _log.debug("residency probe failed", exc_info = True)

    def residency_epoch(self) -> int:
        """The ordering clock, taken before a `/slots` probe is sent."""
        with self._lock:
            self._resident_seq += 1
            return self._resident_seq

    def note_resident(
        self,
        resident: Optional[int],
        reclaimable: int = 0,
        *,
        started_at_seq: Optional[int] = None,
    ) -> Optional[int]:
        """The cache as llama-server actually sees it."""
        with self._lock:
            if (
                started_at_seq is not None
                and self._last_sample_seq
                and started_at_seq <= self._last_sample_started
            ):
                return None
            if resident is None:
                if (
                    self._resident is not None
                    and self._clock() - self._resident_at <= _RESIDENT_HOLD_S
                ):
                    self._reclaimable = 0
                    return None
                self._resident = None
                self._reclaimable = 0
                return None
            ceiling = self._budget if self._budget > 0 else int(resident)
            self._resident = max(0, min(int(resident), ceiling))
            self._reclaimable = max(0, min(int(reclaimable or 0), self._resident))
            self._last_sample_started = (
                self._resident_seq if started_at_seq is None else int(started_at_seq)
            )
            self._resident_seq += 1
            self._last_sample_seq = self._resident_seq
            self._resident_at = self._clock()
            for participant in self._participants.values():
                if participant.growth_at_seq is not None and not (
                    started_at_seq is not None and participant.growth_at_seq > started_at_seq
                ):
                    participant.unconfirmed_growth = 0
                    participant.growth_at_seq = None
                if participant.measured_at_seq is None:
                    continue
                if started_at_seq is not None and participant.measured_at_seq > started_at_seq:
                    continue
                participant.measured = True
                participant.measured_at_seq = None
            return self._last_sample_seq

    def note_erased_since(
        self, recorded_seq: Optional[int], resident: int, reclaimable: int
    ) -> bool:
        """The reading `note_resident` recorded at ``recorded_seq``, less what an erase it
        planned has since freed.

        Not a new sample, so it neither orders against probes nor promotes anyone: that
        reading already promoted everyone it could. Dropped once a newer sample is in, since
        that one saw the erase or the cells refilling.
        """
        with self._lock:
            if recorded_seq is None or recorded_seq != self._last_sample_seq:
                return False
            if self._resident is None:
                return False
            self._resident = max(0, min(int(resident), self._resident))
            self._reclaimable = max(0, min(int(reclaimable or 0), self._resident))
            return True

    def note_tokens(
        self,
        gen_id: str,
        tokens: int,
        prompt_tokens: Optional[int] = None,
    ) -> None:
        """What a round boundary says this run now holds."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None:
                charge = max(0, int(tokens or 0))
                if prompt_tokens is not None:
                    participant.prompt_tokens = min(max(0, int(prompt_tokens or 0)), charge)
                previous = participant.tokens
                participant.tokens = (
                    participant.prompt_tokens if participant.prompt_tokens > 0 else charge
                )
                growth = participant.tokens - previous
                replayed = participant.cells_reclaimed or participant.replay_pending
                if replayed:
                    participant.announce_prefill(participant.tokens)
                elif growth > 0:
                    participant.announce_prefill(growth)
                elif participant.pending_prefill > participant.tokens:
                    participant.pending_prefill = participant.tokens
                if growth > 0:
                    # Same counter as `observe`: a waiter must be able to see a tool result land.
                    self._progress_tokens += growth
                participant.base_tokens = charge
                participant.generated_seen = 0
                if replayed:
                    participant.unconfirmed_growth = 0
                    participant.growth_at_seq = None
                elif growth > 0 and participant.measured:
                    participant.unconfirmed_growth += growth
                    participant.growth_at_seq = None
                if replayed:
                    participant.measured = False
                    participant.measured_at_seq = None
                participant.cells_reclaimed = False
                participant.replay_pending = False

    _LIVE_STATES = frozenset(
        {
            ParticipantState.DECODING,
            ParticipantState.PARKED_ON_TOOL,
            ParticipantState.TOOLS_RUNNING,
        }
    )

    def note_state(self, gen_id: str, state: str) -> bool:
        """Where a live tool-loop chat is: decoding, stopped on an approval, or in a tool."""
        if state not in self._LIVE_STATES:
            return False
        with self._lock:
            changed = self._note_state_locked(gen_id, state)
        if changed is None:
            replacements: List[str] = []
            try:
                replacements = [
                    v.gen_id for v in self.plan_preemptions(needed = 0) if not v.parked_on_a_tool
                ]
            finally:
                with self._lock:
                    self._replaced_by[gen_id] = replacements
            return True
        return changed

    def _note_state_locked(self, gen_id: str, state: str) -> Optional[bool]:
        """`note_state` under the lock: None when a pending choice was undone (and the state
        then recorded), else whether the state changed."""
        revoked = False
        participant = self._participants.get(gen_id)
        was_parked = participant is not None and participant.parked_on_a_tool
        if (
            participant is not None
            and participant.state == ParticipantState.PREEMPTING
            and state in (ParticipantState.PARKED_ON_TOOL, ParticipantState.TOOLS_RUNNING)
        ):
            participant.preempt_chosen_at = 0.0
            participant.consecutive_preemptions = max(0, participant.consecutive_preemptions - 1)
            participant.preempt_event.clear()
            participant.state = ParticipantState.DECODING
            self._replaced_by[gen_id] = None
            revoked = True
        if participant is None or participant.state not in self._LIVE_STATES:
            return False
        if participant.state == state:
            return None if revoked else False
        participant.state = state
        if state in _PARKED_ON_A_TOOL and not was_parked:
            participant.park_seq += 1
            participant.park_charge_seq = getattr(participant.lease, "charge_seq", None)
        if state == ParticipantState.DECODING:
            if participant.cells_reclaimed:
                participant.announce_prefill(participant.tokens)
            participant.cells_reclaimed = False
        return None if revoked else True

    def parked_holders(self) -> Dict[str, int]:
        """Who is parked on a tool right now, with each one's park count."""
        with self._lock:
            return {
                gen_id: participant.park_seq
                for gen_id, participant in self._participants.items()
                if participant.parked_on_a_tool
            }

    def note_cells_reclaimed(self, only: Optional[Dict[str, int]] = None) -> int:
        """An idle-slot reclaim just erased every idle slot."""
        released = []
        with self._lock:
            for participant in self._participants.values():
                if not participant.parked_on_a_tool:
                    continue
                if only is not None and only.get(participant.gen_id) != participant.park_seq:
                    continue
                if participant.cells_reclaimed:
                    continue
                participant.cells_reclaimed = True
                participant.prefill_done()
                released.append((participant, participant.park_seq, participant.park_charge_seq))
        for participant, park_seq, charge_seq in released:
            with self._lock:
                if not participant.cells_reclaimed or participant.park_seq != park_seq:
                    continue
            lease = participant.lease
            yield_parked = getattr(lease, "yield_parked_commitment", None)
            if callable(yield_parked):
                try:
                    if charge_seq is None:
                        yield_parked()
                    else:
                        yield_parked(charged_at = charge_seq)
                except Exception:  # pragma: no cover - bookkeeping must not fail a run
                    _log.debug("could not yield a parked commitment", exc_info = True)
            if charge_seq is not None and getattr(lease, "charge_seq", charge_seq) != charge_seq:
                # Re-charged before the scan, so the lease kept the new commitment: it is a prompt
                # being prefilled again, and must count, reserve included.
                with self._lock:
                    if participant.cells_reclaimed and participant.park_seq == park_seq:
                        participant.cells_reclaimed = False
                        participant.announce_prefill(participant.tokens)
                        participant.replay_pending = True
        return len(released)

    def note_cells_possibly_erased(self, only: Optional[Dict[str, int]] = None) -> int:
        """An erase may have taken a slot another request was handed after the reading."""
        marked = 0
        with self._lock:
            for participant in self._participants.values():
                if not participant.holds_kv or participant.cells_reclaimed:
                    continue
                if only is not None and only.get(participant.gen_id) == participant.park_seq:
                    continue
                participant.replay_pending = True
                marked += 1
        return marked

    def restart(self, gen_id: str) -> None:
        """The same request starts over from its original prompt: the next of `n` choices."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            participant.tokens = participant.charged_tokens
            participant.base_tokens = participant.charged_tokens
            participant.prompt_tokens = participant.charged_prompt_tokens
            participant.generated_seen = 0
            participant.measured = False
            participant.measured_at_seq = None
            participant.unconfirmed_growth = 0
            participant.growth_at_seq = None
            participant.cells_reclaimed = False
            participant.replay_pending = False
            participant.consecutive_preemptions = 0
            participant.state = ParticipantState.DECODING
            participant.preempt_event.clear()
            participant.announce_prefill(participant.prompt_tokens or participant.tokens)

    def note_replayed(self, gen_id: str, tokens: int) -> None:
        """Tokens a paused attempt decoded that the NEXT attempt sends back as prompt."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            replayed = max(0, int(tokens or 0))
            participant.base_tokens = participant.base_tokens + replayed
            if participant.prompt_tokens > 0:
                participant.prompt_tokens = participant.prompt_tokens + replayed
                participant.tokens = max(participant.tokens, participant.prompt_tokens)
            else:
                participant.tokens = max(participant.tokens, participant.base_tokens)

    def wait_released(
        self,
        gen_ids,
        timeout: float,
        *,
        poll_s: float = 0.02,
    ) -> bool:
        """Block until each of ``gen_ids`` has stopped being a pending victim, or ``timeout``.

        A chosen chat keeps decoding into its cells until its stream sees the signal and
        closes upstream, so the room a plan projected is not free yet. True when every one
        has paused, finished, or declined. Blocking: never call it on the event loop."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        pending = list(gen_ids or ())
        while True:
            with self._lock:
                seen = set(pending)
                still: List[str] = []
                work = list(pending)
                while work:
                    gid = work.pop()
                    replaced = self._replaced_by.get(gid, ())
                    if replaced is None:
                        still.append(gid)
                        continue
                    for new in replaced:
                        if new not in seen:
                            seen.add(new)
                            work.append(new)
                    p = self._participants.get(gid)
                    if p is not None and p.state == ParticipantState.PREEMPTING:
                        still.append(gid)
                pending = still
            if not pending:
                return True
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            time.sleep(min(poll_s, left))

    def take_awaited_victims(self, gen_id: str) -> List[str]:
        """The victims ``gen_id``'s plan chose and its route has not waited on yet."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return []
            taken, participant.awaiting_victims = participant.awaiting_victims, []
            return taken

    def set_state(self, gen_id: str, state: str) -> None:
        """Report a safe point."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            if state in _PARKED_ON_A_TOOL and not participant.parked_on_a_tool:
                participant.park_seq += 1
                participant.park_charge_seq = getattr(participant.lease, "charge_seq", None)
            participant.state = state
            if state not in _HOLDS_KV:
                participant.prefill_done()

    def note_measured(self, gen_id: str) -> None:
        """A holder that never reports tokens has prefilled: its charge stops being a
        reservation on top of the resident figure once a reading taken after this call is in.

        The raw passthroughs never call `observe` or `note_tokens`, so `_committed_locked`
        counted them twice once `/slots` saw them, pausing every Studio chat for a holder
        that is never a victim. Folded at once, a reading from before the prefill swallowed
        the whole charge instead, and the raw path has no probe of its own to refresh it.
        Idempotent; the state is left alone.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            if not participant.measured:
                # Bumped so the stamp is strictly past any epoch a probe already in flight captured.
                self._resident_seq += 1
                participant.measured_at_seq = self._resident_seq
            participant.cells_reclaimed = False
            participant.prefill_done()

    def note_resumed(self, gen_id: str) -> None:
        """A preempted generation is decoding again."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            participant.preempt_event.clear()
            participant.state = ParticipantState.DECODING

    def note_declined(self, gen_id: str) -> None:
        """A chosen victim will not pause after all, and goes on decoding."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None or participant.state != ParticipantState.PREEMPTING:
                return
            participant.state = ParticipantState.DECODING
            participant.preempt_chosen_at = 0.0
            participant.consecutive_preemptions = max(0, participant.consecutive_preemptions - 1)
            self._replaced_by[gen_id] = None
            participant.preempt_event.clear()
        replacements: List[str] = []
        try:
            replacements = [
                v.gen_id
                for v in self.plan_preemptions(needed = 0, exclude = gen_id)
                if not v.parked_on_a_tool
            ]
        finally:
            with self._lock:
                self._replaced_by[gen_id] = replacements

    def participant(self, gen_id: str) -> Optional[Participant]:
        """The registered participant, or None once it has finished."""
        with self._lock:
            return self._participants.get(gen_id)

    def is_idle(self) -> bool:
        """Nothing in flight, so this controller may be retired. Mirrors the queue's."""
        with self._lock:
            return not self._participants

    def committed_tokens(self) -> int:
        """What the watermark compares against the ceiling."""
        with self._lock:
            return self._committed_locked()

    def progress_signature(self) -> tuple:
        """What a waiter watches to tell "busy" from "stuck". A change in ANY term means the
        backend is working: `committed` moved, a holder left, a token was generated anywhere,
        or a tool call started or returned.

        The token term is the one that matters: `committed` is `max(resident, measured) +
        pending`, so while the resident figure is the larger one every token the ledger adds
        is invisible, and a waiter abandons its turn while three other chats decode.
        """
        with self._lock:
            return (
                self._committed_locked(),
                frozenset(p.gen_id for p in self._participants.values() if p.holds_kv),
                self._progress_tokens,
                sum(
                    1
                    for p in self._participants.values()
                    if p.state == ParticipantState.TOOLS_RUNNING
                ),
            )

    def _pending_prefill_locked(self, *, exclude: Optional[str] = None) -> int:
        """Prompt tokens announced but not yet in the cache, across every holder."""
        now = time.monotonic()
        return sum(
            p.prefill_pending(now)
            for gen_id, p in self._participants.items()
            if exclude is None or gen_id != exclude
        )

    def _buffer_locked(self, *, pending: Optional[int] = None) -> int:
        """Tokens held clear right now."""
        return preemption_buffer_tokens(
            self._budget,
            draft_tokens = self._draft_tokens,
            slots = self._slots,
            batch_tokens = self._batch_tokens,
            pending_prefill = (
                self._pending_prefill_locked() if pending is None else max(0, int(pending))
            ),
        )

    def _prune_locked(self) -> None:
        """Drop participants whose lease is finished with the cache."""
        dead = [
            gen_id
            for gen_id, p in self._participants.items()
            if p.lease is not None and getattr(p.lease, "is_released", False)
        ]
        for gen_id in dead:
            del self._participants[gen_id]
            self._resume_tickets.pop(gen_id, None)

    def _committed_locked(self) -> int:
        self._prune_locked()
        ledger = sum(p.tokens for p in self._participants.values() if p.holds_kv)
        # Whichever is larger, because both are real: the ledger knows what live generations were
        # admitted on, `resident` what the cache holds, finished requests included.
        if self._resident is None:
            return ledger
        holders = [p for p in self._participants.values() if p.holds_kv]
        measured = sum(p.in_reading_tokens() for p in holders)
        pending = sum(p.on_top_tokens() for p in holders)
        return max(self._resident, measured) + pending

    def _contended_locked(self) -> bool:
        """Whether anybody else could want the room this backend is holding."""
        holders = 0
        for participant in self._participants.values():
            if participant.state in (ParticipantState.QUEUED, ParticipantState.PAUSED):
                return True
            if participant.holds_kv:
                holders += 1
                if holders > 1:
                    return True
        return False

    def contended(self) -> bool:
        """Cheap enough for the token path: no HTTP, one lock, no arithmetic."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return False
            return self._contended_locked()

    def plan_preemptions(
        self,
        *,
        needed: int = 0,
        exclude: Optional[str] = None,
    ) -> List[Participant]:
        """Who must stop so ``needed`` more tokens fit."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return []
            if not self._contended_locked():
                return []
            buffer = self._buffer_locked()
            ceiling = max(0, self._budget - buffer)
            # Net of holders already chosen: they are on their way out, and counting them again
            # would stop a second chat for room the first is already giving back.
            total = self._committed_locked() - sum(
                p.tokens
                for p in self._participants.values()
                if p.holds_kv and p.state == ParticipantState.PREEMPTING and not p.parked_on_a_tool
            )
            want = max(0, int(needed or 0))
            ledger = sum(p.tokens for p in self._participants.values() if p.holds_kv)
            _now = time.monotonic()
            if (
                self._resident is not None
                and abs(self._resident - ledger) > 256
                and _now - self._drift_logged_at >= 5.0
            ):
                self._drift_logged_at = _now
                holders = [p for p in self._participants.values() if p.holds_kv]
                _log.info(
                    "llama preemption ledger-drift: committed=%s resident=%s measured=%s "
                    "pending=%s ledger=%s ceiling=%s buffer=%s prefilling=%s want=%s "
                    "holders=%s",
                    total,
                    self._resident,
                    sum(p.in_reading_tokens() for p in holders),
                    sum(p.on_top_tokens() for p in holders),
                    ledger,
                    ceiling,
                    buffer,
                    self._pending_prefill_locked(),
                    want,
                    len(holders),
                )
            if total + want <= ceiling:
                return []
            # Parked holders first: they hold KV and consume no compute, so their room is the
            # cheapest to take.
            victims = [
                p
                for p in self._participants.values()
                if p.preemptable and p.holds_kv and p.gen_id != exclude
            ]
            victims.sort(
                key = lambda p: (
                    p.state != ParticipantState.PARKED_ON_TOOL,
                    p.promoted,
                    -p.seq,
                )
            )
            standing = any(
                p.holds_kv
                and (not p.preemptable or p.gen_id == exclude)
                and p.state != ParticipantState.PREEMPTING
                for p in self._participants.values()
            )
            spare = len(victims) if standing else max(0, len(victims) - 1)
            chosen: List[Participant] = []
            for victim in victims[:spare]:
                if total + want <= ceiling:
                    break
                if victim.state not in _PARKED_ON_A_TOOL:
                    total -= victim.tokens
                victim.consecutive_preemptions += 1
                victim.preempted_from = victim.state
                victim.state = ParticipantState.PREEMPTING
                self._replaced_by.pop(victim.gen_id, None)
                victim.preempt_chosen_at = time.monotonic()
                victim.preempt_event.set()
                chosen.append(victim)
            if chosen:
                # A sweep that took others and left this one running broke its run of losses: the
                # count is "in a row", so it starts again.
                for passed_over in victims:
                    if passed_over not in chosen:
                        passed_over.consecutive_preemptions = 0
            return chosen

    def snapshot(self) -> PreemptionSnapshot:
        with self._lock:
            states = [p.state for p in self._participants.values()]
            return PreemptionSnapshot(
                key = self.key,
                budget = self._budget,
                buffer = self._buffer_locked(),
                committed = self._committed_locked(),
                decoding = states.count(ParticipantState.DECODING),
                paused = states.count(ParticipantState.PAUSED),
                parked = states.count(ParticipantState.PARKED_ON_TOOL),
                slots = max(1, self._slots or 1),
                tools_running = states.count(ParticipantState.TOOLS_RUNNING),
                prefilling = self._pending_prefill_locked(),
                holders = sum(1 for p in self._participants.values() if p.holds_kv),
            )


class ControllerPreemptionPolicy:
    """Binds one generation to the controller, satisfying ``PreemptionPolicy``."""

    __slots__ = ("_controller", "_gen_id", "_signal", "_resumes", "_stalled", "_loop")

    def __init__(
        self,
        controller: "PreemptionController",
        gen_id: str,
        signal: PreemptSignal,
        *,
        loop = None,
    ):
        self._controller = controller
        self._gen_id = gen_id
        self._signal = signal
        self._resumes = 0
        self._stalled = 0
        self._loop = loop

    def should_preempt(self) -> bool:
        return self._signal.is_set()

    def on_preempted(self, checkpoint: StreamCheckpoint) -> None:
        """The upstream response is closed, so the tokens may go back."""
        self._resumes = checkpoint.resumes
        self._stalled = next_stall_count(self._stalled, checkpoint.charged_tokens)
        _log.info(
            "llama preemption paused: gen_id=%s resumes=%s kept_chars=%s kept=%s charged=%s",
            self._gen_id,
            checkpoint.resumes,
            checkpoint.kept_chars(),
            "prose"
            if checkpoint.has_resume_point()
            else ("thought" if checkpoint.has_reasoning_resume_point() else "nothing"),
            checkpoint.charged_tokens,
        )
        participant = self._controller.participant(self._gen_id)
        if participant is None:
            return
        chosen_at = participant.preempt_chosen_at
        if chosen_at:
            _log.info(
                "llama preemption evict-latency: gen_id=%s ms=%.1f tokens=%s",
                self._gen_id,
                (time.monotonic() - chosen_at) * 1000.0,
                participant.tokens,
            )
            participant.preempt_chosen_at = 0.0
        if checkpoint.charged_tokens and (
            checkpoint.has_resume_point() or checkpoint.has_reasoning_resume_point()
        ):
            self._controller.note_replayed(self._gen_id, checkpoint.charged_tokens)
        self._controller.set_state(self._gen_id, ParticipantState.PAUSED)
        lease = participant.lease
        if lease is not None:
            try:
                lease.preempt()
            except Exception:
                # A handback that fails must not take the conversation with it.
                pass

    def await_resume(
        self,
        timeout: Optional[float] = None,
        *,
        cancel_event = None,
    ) -> bool:
        # None means "caller stated no preference", NOT "wait forever".
        if timeout is None:
            timeout = DEFAULT_RESUME_WAIT_TIMEOUT_S
        if resumes_stalled(self._stalled):
            _log.warning(
                "llama preemption resume-stalled: gen_id=%s resumes=%s stalled=%s",
                self._gen_id,
                self._resumes,
                self._stalled,
            )
            return False
        participant = self._controller.participant(self._gen_id)
        if participant is None:
            return False
        lease = participant.lease
        if lease is None:
            return True
        if self._loop is None:
            return False
        want = max(0, int(participant.replay_tokens() or 0))
        if self._controller.cannot_ever_fit(want):
            _log.info(
                "llama preemption too-large: gen_id=%s want=%s (exceeds the cache; "
                "finishing the turn)",
                self._gen_id,
                want,
            )
            return False
        if self._controller.outgrew_the_shared_ceiling(want):
            _log.info(
                "llama preemption needs-the-cache: gen_id=%s want=%s (past the shared "
                "ceiling; waiting for the cache to itself)",
                self._gen_id,
                want,
            )
        _log.info("llama preemption awaiting-room: gen_id=%s want=%s", self._gen_id, want)
        started = time.monotonic()
        deadline = started + timeout
        hard_deadline = started + timeout * MAX_RESUME_WAIT_MULTIPLE
        last = self._controller.progress_signature()
        self._controller.begin_resume(self._gen_id, want)
        try:
            self._controller.refresh_residency()
            # try_grant_resume, not room_for: the room must be BOOKED at the instant it is found,
            # or two waiters both find the same space and both take it.
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    # Stop pressed during the pause: there is nothing to resume, and the worker
                    # must not sit here waiting for room for a chat nobody is reading.
                    _log.info("llama preemption cancelled-while-paused: gen_id=%s", self._gen_id)
                    return False
                if self._controller.try_grant_resume(self._gen_id, want):
                    break
                self._controller.refresh_residency()
                now = time.monotonic()
                current = self._controller.progress_signature()
                # A holder parked on a tool moves nothing, so the signature freezes while a web
                # search runs.
                _snap = self._controller.snapshot()
                if _snap.parked > 0 or getattr(_snap, "tools_running", 0) > 0:
                    deadline = now + timeout
                if current != last:
                    deadline = now + timeout
                    last = current
                if now >= deadline:
                    _log.info(
                        "llama preemption gave-up: gen_id=%s want=%s (no progress for %ss)",
                        self._gen_id,
                        want,
                        timeout,
                    )
                    return False
                if now >= hard_deadline:
                    _log.info(
                        "llama preemption gave-up: gen_id=%s want=%s (still unserved after "
                        "%ss of a moving cache)",
                        self._gen_id,
                        want,
                        round(now - started, 1),
                    )
                    return False
                time.sleep(0.1)
        finally:
            self._controller.end_resume(self._gen_id)
        try:
            try:
                coro = lease.resume_async(
                    want,
                    timeout_s = timeout,
                    cancel_event = cancel_event,
                    progress = self._controller.progress_signature,
                    hard_deadline_at = hard_deadline,
                )
            except TypeError:
                try:
                    coro = lease.resume_async(
                        want,
                        timeout_s = timeout,
                        cancel_event = cancel_event,
                        progress = self._controller.progress_signature,
                    )
                except TypeError:
                    coro = lease.resume_async(want, timeout_s = timeout, cancel_event = cancel_event)
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            # A backstop for a loop that never runs the coroutine at all. resume_async resets its
            # own bound on progress up to the same hard deadline, so this waits as long, then cancels it.
            try:
                got = bool(future.result(timeout = max(0.0, hard_deadline - time.monotonic()) + 5.0))
            except BaseException:
                future.cancel()
                raise
            if not got:
                self._controller.note_resume_failed(self._gen_id)
            _log.info(
                "llama preemption %s: gen_id=%s want=%s",
                "resumed" if got else "gave-up",
                self._gen_id,
                want,
            )
            return got
        except Exception as exc:
            self._controller.note_resume_failed(self._gen_id)
            _log.warning(
                "llama preemption resume-failed: gen_id=%s want=%s error=%s",
                self._gen_id,
                want,
                exc,
            )
            return False

    def on_resumed(self) -> None:
        self._controller.note_resumed(self._gen_id)

    def restart(self) -> None:
        """The next of `n` choices: the same lease, a fresh count of resumes and the ledger
        back to what admission charged, since the last choice's partials are not replayed."""
        self._resumes = 0
        self._stalled = 0
        self._controller.restart(self._gen_id)

    def on_declined(self) -> None:
        """This generation was chosen to pause and is not going to."""
        _log.info("llama preemption declined: gen_id=%s (finishing instead)", self._gen_id)
        self._controller.note_declined(self._gen_id)


def _slot_decoded(slot: dict) -> int:
    """Tokens this slot has generated so far, from `/slots`."""
    raw = slot.get("next_token")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if not isinstance(raw, dict):
        return 0
    try:
        return max(0, int(raw.get("n_decoded") or 0))
    except (TypeError, ValueError):
        return 0


def read_slot_occupancy(fetch: Callable[[], Optional[list]]) -> Optional[dict]:
    """Tokens actually resident in the cache, INCLUDING slots that are idle."""
    slots = fetch()
    if not slots:
        return None
    resident = 0
    idle_tokens = 0
    idle = []
    idle_task = {}
    for slot in slots:
        tokens = 0
        counts_generated = False
        try:
            raw_total = int(slot.get("n_prompt_tokens") or 0)
        except (TypeError, ValueError):
            raw_total = 0
        try:
            raw_cache = int(slot.get("n_prompt_tokens_cache") or 0)
        except (TypeError, ValueError):
            raw_cache = 0
        if raw_total > 0:
            tokens = raw_total
            counts_generated = True
        elif raw_cache > 0:
            tokens = raw_cache
        if slot.get("is_processing") and not counts_generated:
            tokens += _slot_decoded(slot)
        resident += max(0, tokens)
        if not slot.get("is_processing") and tokens > 0:
            idle_tokens += max(0, tokens)
            idle.append((slot.get("id"), max(0, tokens)))
            idle_task[slot.get("id")] = slot.get("id_task")
    idle.sort(key = lambda pair: -pair[1])
    return {
        "resident": resident,
        "idle_tokens": idle_tokens,
        "idle": idle,
        "idle_task": idle_task,
        "slots": len(slots),
    }


def _slot_incarnation(slots: Optional[list], slot_id) -> Optional[tuple]:
    """``(is_processing, id_task)`` for one slot of a ``GET /slots`` reading, None if absent."""
    for slot in slots or ():
        if isinstance(slot, dict) and slot.get("id") == slot_id:
            return bool(slot.get("is_processing")), slot.get("id_task")
    return None


def reclaim_idle_slots(
    occupancy: Optional[dict],
    erase: Callable[[int], int],
    *,
    needed: int,
    recheck: Optional[Callable[[], Optional[list]]] = None,
    ambiguous: Optional[list] = None,
) -> int:
    """Free dead residue before asking a live chat to stop."""
    if not occupancy or needed <= 0:
        return 0
    seen_task = occupancy.get("idle_task") or {}
    freed = 0
    for slot_id, tokens in occupancy.get("idle") or ():
        if freed >= needed:
            break
        if slot_id is None:
            continue
        if recheck is not None:
            now = _slot_incarnation(recheck(), slot_id)
            if now is None or now[0] or now[1] != seen_task.get(slot_id, now[1]):
                continue
        try:
            freed += max(0, int(erase(slot_id) or 0))
        except Exception:
            continue
        if recheck is not None and ambiguous is not None:
            after = _slot_incarnation(recheck(), slot_id)
            if after is None or after[0] or after[1] != seen_task.get(slot_id, after[1]):
                ambiguous.append(slot_id)
    return freed


_CONTROLLERS_LOCK = threading.Lock()
_CONTROLLERS: Dict[str, PreemptionController] = {}


def get_preemption_controller(key: str) -> PreemptionController:
    with _CONTROLLERS_LOCK:
        controller = _CONTROLLERS.get(key)
        if controller is None:
            controller = PreemptionController(key)
            _CONTROLLERS[key] = controller
            stale = [k for k, c in _CONTROLLERS.items() if k != key and c.is_idle()]
            for k in stale:
                del _CONTROLLERS[k]
        return controller


def reset_preemption_controllers() -> None:
    with _CONTROLLERS_LOCK:
        _CONTROLLERS.clear()
