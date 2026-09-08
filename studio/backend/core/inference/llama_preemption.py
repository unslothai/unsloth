# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who pauses, who keeps decoding, and when the room is actually back.

``--parallel N --kv-unified -c N`` is N cells TOTAL while every slot is told it has N, and
llama-server polices only ``prompt_tokens < slot.n_ctx``, so chats that each fit on their
own are all admitted and then collide; ``server-context.cpp`` then calls ``send_error`` on
EVERY processing slot. Admission (``llama_admission``) decides who gets in; this module
decides who has to stop once they are in. Policy and state only: aborting the upstream
stream and resuming it belong to the caller, per ``LlamaAdmissionLease.preempt``.
"""

from __future__ import annotations

import asyncio
import os
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from core.inference.llama_admission import LlamaAdmissionLease, _bool_env


_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}

from loggers import get_logger

_log = get_logger(__name__)


# Off falls back to step 1's wire clamp alone.
PREEMPT_ENV = "UNSLOTH_LLAMA_ADMISSION_PREEMPT"
DEFAULT_PREEMPT_ENABLED = True


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        return default
    # A ratio outside (0, 0.5) is a typo: zero disables the margin, half is drastic.
    return value if 0.0 < value < 0.5 else default


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


# Kept only so an operator who set the old knob is not silently ignored.
DEFAULT_PREEMPT_BUFFER_RATIO = _float_env("UNSLOTH_LLAMA_PREEMPT_BUFFER_RATIO", 0.15)
# Reaction headroom for ONE decoding slot, between a sweep and the victim stopping.
DEFAULT_PREEMPT_BUFFER_PER_SLOT = 192
DEFAULT_PREEMPT_BUFFER_MIN_TOKENS = 256

# Escape hatch: reserve a whole --batch-size always, for a surface that forgets to announce.
STATIC_BATCH_ENV = "UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH"
DEFAULT_PREEMPT_STATIC_BATCH = False

# Drop the batch term where `_committed_locked` already charges the prefill on top of
# `resident`, so the same room is not reserved twice. Off: it assumes `resident` is fresh.
CHARGED_PREFILL_ENV = "UNSLOTH_LLAMA_PREEMPT_BATCH_ONLY_UNCHARGED"
DEFAULT_PREEMPT_BATCH_ONLY_UNCHARGED = False

# Backstop for an announced prefill that never happens. The stream waits this long for a
# first token (`_DEFAULT_FIRST_TOKEN_TIMEOUT_S`), so a shorter figure dropped the reserve
# under a prompt that was still going in.
PENDING_PREFILL_TTL_S = 1200.0

# Bounds churn under contention. _MAX_LENGTH_CONTINUATIONS is a quality cap, not this.
DEFAULT_MAX_PREEMPT_RESUMES = 32

# A chat preempted this many times in a row outranks longest-wins for the next epoch.
PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS = 3

# What a chat running ALONE must leave clear. Without it a chat past the shared ceiling
# can never be admitted OR resumed, and hangs until its client gives up.
SOLO_MARGIN_DIVISOR = 200

# The reclaim barrier gives up after this long: a server without --metrics never answers.
DEFAULT_RECLAIM_BARRIER_TIMEOUT_S = 10.0
DEFAULT_RECLAIM_BARRIER_POLL_S = 0.05

# An unbounded await_resume() turned three paused chats into a 33-minute hang. On expiry
# the turn finishes with what it has.
DEFAULT_RESUME_WAIT_TIMEOUT_S = 90.0

# Absolute bound on one resume wait, in stall timeouts, for the one case the stall clock
# cannot see: a cache that keeps moving but never has room for THIS chat.
MAX_RESUME_WAIT_MULTIPLE = 80


class LlamaStreamPreempted(Exception):
    """The upstream stream was aborted to free KV, not abandoned.

    Unlike ``_LlamaStreamCancelled`` it is caught and resumed, so anything that treats a
    dead stream as failure must not see it.
    """


class PreemptSignal:
    """A pause request that tool execution can hold off.

    ``is_set()`` reports False while an unsafe window is open even though a request is
    pending, so a preempt lands at a safe point or not at all.
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.RLock()
        self._depth = 0
        self._pending = False
        self._reason: Optional[str] = None

    def request(self, reason: str = "kv_pressure") -> None:
        with self._lock:
            self._pending = True
            self._reason = reason
            if self._depth == 0:
                self._event.set()

    # Spelled like threading.Event so a test double can pass a bare Event.
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
            # Also hides a request that predates the window: acting on it now is the
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

    From the live stream accumulators, not the thread's trailing assistant row: an aborted
    run never writes one, so resuming from it would replay text the user has seen.
    """

    visible_text: str = ""
    reasoning_text: str = ""
    # Carried rather than re-derived: the `context_truncated` SSE event is not idempotent.
    pending_truncations: list = field(default_factory = list)
    charged_tokens: int = 0
    resumes: int = 0
    reason: Optional[str] = None

    def has_resume_point(self) -> bool:
        """Whether there is anything to continue from. Empty means the pause landed
        before the first token, so the request is re-issued whole rather than continued.
        """
        return bool(self.visible_text.strip())

    def has_reasoning_resume_point(self) -> bool:
        """Whether there is a thought to continue, absent any visible prose. It resumes
        through ``reasoning_content``; as ``content`` it would render as the answer.
        """
        return not self.visible_text.strip() and bool(self.reasoning_text.strip())

    def kept_chars(self) -> int:
        """Characters carried across the pause, prose or thought. ``len(visible_text)``
        alone reads zero on every pause of a reasoning model.
        """
        if self.has_resume_point():
            return len(self.visible_text)
        return len(self.reasoning_text) if self.has_reasoning_resume_point() else 0


@runtime_checkable
class PreemptionPolicy(Protocol):
    """Supplied by the admission side; this module only calls it. ``await_resume``
    returning False means "stop waiting and finish the turn", so a policy that dies or
    times out degrades to today's behaviour instead of hanging the chat.
    """

    def should_preempt(self) -> bool: ...

    def on_preempted(self, checkpoint: StreamCheckpoint) -> None: ...

    def await_resume(self, timeout: Optional[float] = None) -> bool: ...

    def on_resumed(self) -> None: ...

    def on_declined(self) -> None: ...


class DeferredPreemptionPolicy:
    """A policy handed to the stream before the real one can exist.

    The tool-loop generator is BUILT before admission returns and only ITERATED after, so
    it cannot yet know its lease. Unbound it behaves as ``NullPreemptionPolicy``, which is
    correct rather than convenient: nothing can pause before the generator is iterated.
    """

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

    def await_resume(self, timeout: Optional[float] = None) -> bool:
        return False if self._inner is None else bool(self._inner.await_resume(timeout))

    def on_resumed(self) -> None:
        if self._inner is not None:
            self._inner.on_resumed()

    def on_declined(self) -> None:
        if self._inner is not None:
            self._inner.on_declined()


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

    def on_declined(self) -> None:
        return None


class ParticipantState:
    """Where a generation is, which decides whether it may be preempted.

    Strings rather than an enum so they cross the SSE boundary to the UI untranslated.
    """

    QUEUED = "queued"
    DECODING = "decoding"
    # Holds KV, consumes no compute: the cheapest room to reclaim, so taken first.
    PARKED_ON_TOOL = "parked_on_tool"
    # Never preempted: nothing is decoding, and it is inside the tool loop's unsafe window.
    TOOLS_RUNNING = "tools_running"
    # Asked to stop, but still holding KV until the stream reaches a safe point; counting
    # it free at the decision let four chats commit 16384 of a 16384 cache.
    PREEMPTING = "preempting"
    # Raw passthrough and the Responses surface: nothing to resume, so aborting is a CANCEL
    # and it is never a victim. Still registered, or the watermark fires late by its size.
    STREAMING_RAW = "streaming_raw"
    PAUSED = "paused"
    # Granted room and waiting for a slot to prefill into. Not a victim: a sweep would
    # count cells as freed that its prefill is about to fill.
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
# PREEMPTING is absent: asking twice would double-count the room its pause will free.

# States a generated token contradicts; `observe` moves such a holder to DECODING.
_DECODES_WHEN_TOKENS_ARRIVE = frozenset(
    {
        ParticipantState.TOOLS_RUNNING,
        ParticipantState.PARKED_ON_TOOL,
        ParticipantState.RESUMING,
    }
)


def preemption_enabled() -> bool:
    return _bool_env(PREEMPT_ENV, DEFAULT_PREEMPT_ENABLED)


def preemption_buffer_tokens(
    budget: int,
    *,
    draft_tokens: int = 0,
    slots: int = 1,
    batch_tokens: int = 0,
    pending_prefill: int = 0,
) -> int:
    """Tokens held clear of ``budget``. Zero for an unknown budget, which disables it.

    Capped at half the budget: a buffer that swallows it leaves a ceiling of zero, which
    preempts every participant forever. Speculative drafts are charged on top, since a
    drafter fills up to ``--spec-draft-n-max`` cells per slot before acceptance and
    admission never sees them; without that reserve llama-server halves n_batch looking for
    room and throws ``speculative batch index 4 is not inside the current sub-batch [0, 4)``
    (ggml-org/llama.cpp#24840). The batch term is charged only while a prefill is pending;
    ``UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH=1`` reserves one permanently.
    """
    if budget <= 0:
        return 0
    # PER SLOT, not per cache: reaction headroom scales with how many chats decode at once,
    # and an oversized buffer lowers the shared ceiling until the cache serialises.
    per_slot = _int_env("UNSLOTH_LLAMA_PREEMPT_BUFFER_PER_SLOT", DEFAULT_PREEMPT_BUFFER_PER_SLOT)
    slot_count = max(1, int(slots or 1))
    reserve = max(DEFAULT_PREEMPT_BUFFER_MIN_TOKENS, per_slot * slot_count)
    # And room for the batch llama.cpp is processing: the cache fails when the next BATCH
    # does not fit, not when it is full. Only while something prefills, since decoding
    # submits no chunk. Summed then capped, as llama-server does for itself
    # (`res + std::min(res_pmt, n_batch)`); max() because both buy space for the next step.
    n_batch = max(0, int(batch_tokens or 0))
    if _bool_env(STATIC_BATCH_ENV, DEFAULT_PREEMPT_STATIC_BATCH):
        batch_reserve = n_batch
    else:
        batch_reserve = min(n_batch, max(0, int(pending_prefill or 0)))
    reserve = max(reserve, batch_reserve)
    # Drafts are additional: cells put in BEFORE acceptance, on top of the batch.
    reserve += max(0, int(draft_tokens or 0)) * slot_count
    # Never the whole cache: a large draft window on a small -c degrades to a tight buffer.
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
    # True once this prompt is known to be IN the cache. Until then its charge is a
    # reservation the resident figure cannot see; after, adding it counts the cells twice.
    measured: bool = False
    state: str = ParticipantState.DECODING
    consecutive_preemptions: int = 0
    # Set when this generation must stop. The caller aborts ONLY the upstream stream and
    # keeps its generator, ledgers, conversation and SSE response, which a cancel cannot say.
    preempt_event: PreemptSignal = field(default_factory = PreemptSignal)
    # monotonic() when this participant was CHOSEN: the eviction latency the buffer covers.
    preempt_chosen_at: float = 0.0
    # True once an idle-slot reclaim erased this holder's cells: its charge then describes
    # cells that are gone. Cleared when it decodes again, which prefills the prompt back.
    cells_reclaimed: bool = False
    # Bumped every time this holder stops on a tool. A reclaim is planned from a `/slots`
    # reading and released after the erases, so the release has to tell the holder that
    # reading saw parked from one that parked (or parked again) while the erases were in
    # flight, whose cells no erase touched.
    park_seq: int = 0
    # The ONLY thing that puts the batch term in the buffer, so it must be set before the
    # request carrying the prompt is sent. `measured` asks whether the charge is resident.
    pending_prefill: int = 0
    pending_prefill_at: float = 0.0
    # Last count `observe` was given, so cumulative reports become a delta. Falls back to
    # zero rather than negative when a resumed attempt restarts llama-server's counter.
    generated_seen: int = 0

    def prefill_pending(self, now: float) -> int:
        """Outstanding prefill worth reserving a batch for. Zero when there is none, and
        for a holder whose cells are gone or which is not in the cache at all.
        """
        if self.pending_prefill <= 0 or not self.holds_kv:
            return 0
        if now - self.pending_prefill_at > PENDING_PREFILL_TTL_S:
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


@dataclass(frozen = True, **_SLOTS)
class PreemptionSnapshot:
    key: str
    budget: int
    buffer: int
    committed: int
    decoding: int
    paused: int
    parked: int
    winner: Optional[str]
    slots: int = 1
    tools_running: int = 0
    prefilling: int = 0
    # Holders from the ledger alone: ``committed`` folds in the last residency reading,
    # which can still be counting a chat that has just unregistered.
    holders: int = 0


class PreemptionController:
    """Victim choice and the epoch, for one llama-server backend.

    Not a scheduler: callers report where they are, ask whether there is room, and are told
    who must stop.
    """

    __slots__ = (
        "key",
        "_lock",
        "_participants",
        "_seq",
        "_epoch_winner",
        "_budget",
        "_kv_unified",
        "_draft_tokens",
        "_slots",
        "_batch_tokens",
        "_resident",
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
        self._epoch_winner: Optional[str] = None
        self._drift_logged_at = 0.0
        self._budget = 0
        # Preemption reclaims only where an idle slot's cells can be purged, which upstream
        # gates on kv_unified ALONE (try_clear_idle_slots, server-context.cpp:1656). NOT on
        # idle_slot_clearing_active: that tracks the PROACTIVE sweep, and a preempted task
        # has ENDED, so the reactive purge applies on every platform.
        self._kv_unified = False
        # Speculative drafts occupy cells no request is charged for.
        self._draft_tokens = 0
        # --batch-size llama-server was launched with, so the buffer can cover one chunk.
        self._batch_tokens = 0
        # Of `_resident`, how much is idle slots' cache: real, but erasable on demand.
        self._reclaimable = 0
        self._slots = 1
        # True cells resident from the last GET /slots, None when unreadable. Includes the
        # residue of FINISHED requests, which the ledger cannot see.
        self._resident: Optional[int] = None
        # Optional: everything works from the ledger alone, less precisely.
        self._residency_probe: Optional[Callable[[], None]] = None
        # Never decreasing: the one figure that moves whenever ANYBODY decodes, where
        # `committed` can sit still through thousands of generated tokens.
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
        """Re-read the cache this backend actually allocated. Called per request, since a
        reload can relaunch llama-server at a different ``-c``.
        """
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
    ) -> Participant:
        """``signal`` MUST be the object the stream polls.

        Without it a Participant makes its own, the caller hands a different one to the
        stream, and setting the participant's signal reaches nobody.
        """
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
                state = state,
                **({} if signal is None else {"preempt_event": signal}),
            )
            # Its whole prompt is about to be prefilled. Announced HERE rather than by the
            # caller, so a sweep firing in between plans against the raised buffer.
            participant.announce_prefill(participant.tokens)
            self._participants[gen_id] = participant
            return participant

    def unregister(self, gen_id: str) -> None:
        """Drop a finished generation and end its epoch if it held one."""
        with self._lock:
            self._participants.pop(gen_id, None)
            if self._epoch_winner == gen_id:
                self._epoch_winner = None

    def _solo_ceiling_locked(self) -> int:
        """The cache less what a lone chat still needs clear to keep running.

        Its own drafts and the estimate error, and ALSO one prefill batch: being alone
        removes the need for reaction headroom, not llama-server's need to fit the next
        batch. Without it a lone chat filled all but 87 of 16384 cells and its own next
        prefill failed as `Context size has been exceeded`.
        """
        margin = max(
            self._draft_tokens + max(64, self._budget // SOLO_MARGIN_DIVISOR),
            self._batch_tokens + self._draft_tokens,
        )
        return max(1, self._budget - margin)

    def outgrew_the_shared_ceiling(self, want: int) -> bool:
        """Whether `want` can never fit beside anyone, however much is evicted. Such a
        generation must not wait: no preemption admits it, so waiting is a deadlock rather
        than a delay, and it runs alone instead.
        """
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return False
            # Including this chat's own prefill: the ceiling has to be the one that exists
            # while it runs.
            pending = self._pending_prefill_locked() + max(0, int(want or 0))
            ceiling = max(0, self._budget - self._buffer_locked(pending = pending))
            return int(want or 0) > ceiling

    def cannot_ever_fit(self, want: int) -> bool:
        """Whether `want` exceeds the cache itself, so not even running alone helps.

        The turn ends with what it has, reported as `length`, which the continuation path
        resumes against a fresh window. Parking it forever instead is the hang.
        """
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return False
            return int(want or 0) > self._solo_ceiling_locked()

    def room_for(self, gen_id: str, want: int) -> bool:
        """Whether a paused generation may start again yet.

        Against the LIVE total: resuming on the admission queue's accounting let a chat back
        in over the watermark, so the next sweep evicted it again. The winner is counted
        like any other holder, so a resume waits for it rather than squeezing in beside it.
        """
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return True
            return self._room_for_locked(gen_id, want)

    def try_grant_resume(self, gen_id: str, want: int) -> bool:
        """Decide there is room AND take it, without letting go of the lock between.

        `room_for` only answers a question, and two paused chats asking at once both get
        yes: PAUSED is not in `_HOLDS_KV`, so neither appears in the other's arithmetic and
        nothing books the space before the prefill. Both then prefill together, and that
        overflow happens inside one prefill, where sampled residency never sees it.

        The grant charges `want` immediately, so the next caller sees the room as taken, and
        marks the participant unmeasured, since its prefill has not happened. RESUMING
        rather than DECODING: the lease's slot is reacquired after this, and a sweep in
        between could count a DECODING participant's reservation as freed and set a signal
        the admission wait never reads.

        Roll back with `note_resume_failed`, or the booking becomes room nobody is using.
        """
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return True
            if not self._room_for_locked(gen_id, want):
                return False
            participant = self._participants.get(gen_id)
            if participant is not None:
                need = max(0, int(want or 0))
                participant.tokens = max(participant.tokens, need)
                participant.base_tokens = max(participant.base_tokens, need)
                participant.measured = False
                participant.state = ParticipantState.RESUMING
                # Announced under the same lock that booked the room, so no participant
                # can see the booking without also seeing the batch it needs.
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

    def _room_for_locked(self, gen_id: str, want: int) -> bool:
        """The arithmetic behind `room_for`, callable by a holder of the lock."""
        # `want` REPLACES this generation's own announcement: saying yes here causes the
        # prefill, and a chat that already announced must not be charged twice.
        pending = self._pending_prefill_locked(exclude = gen_id) + max(0, int(want or 0))
        ceiling = max(0, self._budget - self._buffer_locked(pending = pending))
        ledger_others = sum(
            p.tokens for gid, p in self._participants.items() if p.holds_kv and gid != gen_id
        )
        # Resident cells count too, minus what this generation holds, or an idle slot's
        # leftovers are invisible here exactly as they were to the watermark.
        others = ledger_others
        if self._resident is not None:
            mine = self._participants.get(gen_id)
            # Minus the idle residue, erased for the waiter rather than waited out:
            # counting it refused every resume against a ceiling the slots already passed.
            occupied = self._resident - self._reclaimable
            others = max(others, occupied - (mine.tokens if mine else 0))
        need = max(0, int(want or 0))
        others = max(0, others)
        if others + need <= ceiling:
            return True
        # Outgrew the shared ceiling: it must run alone, or wait for room nothing can make.
        if need > ceiling and others == 0:
            return need <= self._solo_ceiling_locked()
        return False

    def observe(self, gen_id: str, generated: int) -> List["Participant"]:
        """Live growth during generation, and the eviction check that follows it.

        THE watermark sweep. Admission deliberately overcommits, so what prevents the cache
        filling is being told, often, how big each generation has become; a check only
        between rounds is too late, since one round can generate thousands of tokens.

        Returns whoever must stop, already signalled.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None:
                reported = max(0, int(generated or 0))
                # Counted at the one point every surface reports through. A fall means a
                # resumed attempt restarted its own count.
                previous = participant.generated_seen
                participant.generated_seen = reported
                self._progress_tokens += (reported - previous) if reported >= previous else reported
                participant.tokens = participant.base_tokens + reported
                participant.measured = True
                participant.cells_reclaimed = False
                if reported > 0:
                    # A generated token can only follow a finished prefill. Guarded on
                    # `generated`, or the round-boundary sweep's zero would drop the reserve
                    # `note_tokens` just announced.
                    participant.prefill_done()
                # And the chat is decoding, whatever it was last reported as: a round that
                # streams only tool-call deltas left it TOOLS_RUNNING, hence unpreemptable,
                # until the cache overran and llama-server ended both chats.
                if participant.state in _DECODES_WHEN_TOKENS_ARRIVE:
                    participant.state = ParticipantState.DECODING
        # Outside the lock: plan_preemptions takes it, and it is not reentrant.
        return self.plan_preemptions(needed = 0)

    def set_residency_probe(self, probe: Optional[Callable[[], None]]) -> None:
        """Register a way to re-read the cache on demand.

        The ledger adds up prompt ESTIMATES and the per-slot totals are exact. It matters
        most when granting a resume, where a chat comes back carrying its whole replayed
        partial and a reading a second old can be a thousand tokens stale.
        """
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

    def note_resident(
        self,
        resident: Optional[int],
        reclaimable: int = 0,
    ) -> None:
        """The cache as llama-server actually sees it. None means the read failed.

        ``reclaimable`` is the part held by IDLE slots: real occupancy, so it counts toward
        the watermark, but it is erased on demand rather than stood in a waiter's way. See
        ``_room_for_locked``.
        """
        with self._lock:
            if resident is None:
                self._resident = None
                self._reclaimable = 0
                return
            # Clamped: a per-slot sum is an upper bound (shared prefix cells, stale idle
            # entries), and unclamped it is unreachable, so every resume is refused.
            ceiling = self._budget if self._budget > 0 else int(resident)
            self._resident = max(0, min(int(resident), ceiling))
            self._reclaimable = max(0, min(int(reclaimable or 0), self._resident))

    def note_tokens(self, gen_id: str, tokens: int) -> None:
        """What a round boundary says this run now holds.

        Also the third and last place a prefill is announced: only the difference from the
        previous figure is submitted, which keeps the reserve honest on a chat whose 6000
        token history grew by 40.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None:
                previous = participant.tokens
                participant.tokens = max(0, int(tokens or 0))
                growth = participant.tokens - previous
                if participant.cells_reclaimed:
                    # A reclaim erased every cell: the whole prompt goes in again, not the
                    # round's growth. This runs before `note_state` sees DECODING.
                    participant.announce_prefill(participant.tokens)
                elif growth > 0:
                    participant.announce_prefill(growth)
                if growth > 0:
                    # Same counter as `observe`: a waiter must be able to see a tool
                    # result land.
                    self._progress_tokens += growth
                # Re-baselined: a round boundary restates the whole conversation.
                participant.base_tokens = participant.tokens
                participant.measured = True
                participant.cells_reclaimed = False

    # The three states a tool-loop chat moves between while it is alive; the rest are owned
    # by other transitions.
    _LIVE_STATES = frozenset(
        {
            ParticipantState.DECODING,
            ParticipantState.PARKED_ON_TOOL,
            ParticipantState.TOOLS_RUNNING,
        }
    )

    def note_state(self, gen_id: str, state: str) -> bool:
        """Where a live tool-loop chat is: decoding, stopped on an approval, or in a tool.

        Nobody set PARKED_ON_TOOL or TOOLS_RUNNING before, so a chat waiting on an approval
        stayed DECODING: eligible to be crowned a winner nobody benefits from, invisible to
        the resume wait's stall detector, and counted as holding cells a reclaim had erased.
        Only the live states move here; a chat asked to stop keeps its state until its own
        transition. True when the state changed.
        """
        if state not in self._LIVE_STATES:
            return False
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None or participant.state not in self._LIVE_STATES:
                return False
            if participant.state == state:
                return False
            participant.state = state
            if state in (ParticipantState.PARKED_ON_TOOL, ParticipantState.TOOLS_RUNNING):
                participant.park_seq += 1
            if state == ParticipantState.DECODING:
                # Back at the model: llama-server prefills the prompt in again, so the
                # cells are real once more.
                if participant.cells_reclaimed:
                    # The WHOLE prompt, not a round's growth: the reclaim erased every
                    # cell, so there is no prefix left to hit.
                    participant.announce_prefill(participant.tokens)
                participant.cells_reclaimed = False
            if self._epoch_winner == gen_id and state != ParticipantState.DECODING:
                self._epoch_winner = None
            return True

    def parked_holders(self) -> Dict[str, int]:
        """Who is parked on a tool right now, with each one's park count.

        Read BEFORE the `/slots` scrape a reclaim is planned from and handed back to
        `note_cells_reclaimed`, so the release covers the holders whose idle cells that
        reading counted and not one that parked while the erases were in flight.
        """
        with self._lock:
            return {
                gen_id: participant.park_seq
                for gen_id, participant in self._participants.items()
                if participant.state
                in (ParticipantState.PARKED_ON_TOOL, ParticipantState.TOOLS_RUNNING)
            }

    def note_cells_reclaimed(self, only: Optional[Dict[str, int]] = None) -> int:
        """An idle-slot reclaim just erased every idle slot. Tell the ledger.

        A holder parked on an approval or running its tools has an idle slot by definition,
        so the erase took its cells: its charge stops counting, and its lease hands the
        commitment back the way `recost_waiting` does. Returns how many holders it applied
        to.

        `only` is a `parked_holders()` reading taken before the scrape the erases were
        planned from. The erases are blocking HTTP calls that can take seconds, and a chat
        that parks inside that window has idle cells no erase touched and was never in the
        scrape's `idle_tokens`; releasing its commitment would let a waiter into KV that is
        still resident. Omitted, every parked holder is released, as before.
        """
        released = []
        with self._lock:
            for participant in self._participants.values():
                if participant.state not in (
                    ParticipantState.PARKED_ON_TOOL,
                    ParticipantState.TOOLS_RUNNING,
                ):
                    continue
                if only is not None and only.get(participant.gen_id) != participant.park_seq:
                    # Parked after the reading, or parked again since: its cells are the
                    # ones the erases did not take.
                    continue
                if participant.cells_reclaimed:
                    continue
                participant.cells_reclaimed = True
                # Whatever it was about to prefill went with the cells; `note_state`
                # re-announces when it decodes again.
                participant.prefill_done()
                released.append(
                    (
                        participant,
                        participant.park_seq,
                        getattr(participant.lease, "charge_seq", None),
                    )
                )
        for participant, park_seq, charge_seq in released:
            # The lease call runs outside the lock, after erases that took seconds: a holder
            # whose tool came back in between has restated its prompt and re-charged, and
            # that commitment is for cells now filling, not for the ones erased.
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
        return len(released)

    def note_replayed(self, gen_id: str, tokens: int) -> None:
        """Tokens a paused attempt decoded that the NEXT attempt sends back as prompt.

        They do not leave the cache, they change category, and the stream's own counter
        restarts at zero. Without this the sweep recomputes occupancy against the ORIGINAL
        prompt and undercounts by the whole replayed partial, by more on every pause.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            participant.base_tokens = participant.base_tokens + max(0, int(tokens or 0))
            participant.tokens = max(participant.tokens, participant.base_tokens)

    def set_state(self, gen_id: str, state: str) -> None:
        """Report a safe point. Ends the epoch when the winner stops decoding.

        The winner is fixed until it completes, blocks on a tool, or ends its turn, so two
        chats cannot trade places forever.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            # A park is counted once, however many times it is reported: a repeat would
            # make a holder that really was in the reading look like a newcomer to it.
            if state != participant.state and state in (
                ParticipantState.PARKED_ON_TOOL,
                ParticipantState.TOOLS_RUNNING,
            ):
                participant.park_seq += 1
            participant.state = state
            if state not in _HOLDS_KV:
                participant.prefill_done()
            if self._epoch_winner == gen_id and state != ParticipantState.DECODING:
                self._epoch_winner = None

    def note_measured(self, gen_id: str) -> None:
        """A holder that never reports tokens has prefilled: its charge stops being a
        reservation on top of the resident figure.

        The raw passthroughs never call `observe` or `note_tokens`, so `_committed_locked`
        counted them twice once `/slots` saw them, pausing every Studio chat for a holder
        that is never a victim. Idempotent; the state is left alone.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            participant.measured = True
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
        """A chosen victim will not pause after all, and goes on decoding.

        `plan_preemptions` had already moved it to PREEMPTING and set its signal, and nothing
        in the ordinary path takes either back, so it would decode on holding cells while
        outside `_PREEMPTABLE`, where no later sweep could ask it to stop.

        Only the DECISION is undone: nothing was released, since the lease is handed back in
        `on_preempted`, which this participant never reached. Only from PREEMPTING, or a
        participant that has since paused or finished would be resurrected as a holder.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None or participant.state != ParticipantState.PREEMPTING:
                return
            participant.state = ParticipantState.DECODING
            participant.preempt_chosen_at = 0.0
            participant.consecutive_preemptions = max(0, participant.consecutive_preemptions - 1)
            # Last, and inside the lock: a signal still set aborts the very stream this is
            # letting run, and clearing it first leaves a window for a re-arming sweep.
            participant.preempt_event.clear()

    def participant(self, gen_id: str) -> Optional[Participant]:
        """The registered participant, or None once it has finished. Returned by reference,
        so the caller reads live state written under this lock.
        """
        with self._lock:
            return self._participants.get(gen_id)

    def is_idle(self) -> bool:
        """Nothing in flight, so this controller may be retired. Mirrors the queue's."""
        with self._lock:
            return not self._participants

    def committed_tokens(self) -> int:
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
        """Prompt tokens announced but not yet in the cache, across every holder. `exclude`
        drops one participant so a caller asking about itself does not count it twice.
        """
        now = time.monotonic()
        # See CHARGED_PREFILL_ENV: an unmeasured holder's chunk comes out of cells
        # `_committed_locked` already adds on top of the resident figure.
        skip_charged = _bool_env(CHARGED_PREFILL_ENV, DEFAULT_PREEMPT_BATCH_ONLY_UNCHARGED)
        return sum(
            p.prefill_pending(now)
            for gen_id, p in self._participants.items()
            if (exclude is None or gen_id != exclude) and not (skip_charged and not p.measured)
        )

    def _buffer_locked(self, *, pending: Optional[int] = None) -> int:
        """Tokens held clear right now. `pending` overrides the live announcement sum.

        Callers deciding whether to LET somebody prefill pass their own `want`: answering at
        the idle buffer and raising it once the grant lands admits a chat into room that
        stops existing in the same breath.
        """
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
        """Drop participants whose lease is finished with the cache.

        Belt and braces beside ``unregister``: a generation ends on many branches, and one
        that forgets would count a dead conversation for the life of the model load.
        """
        dead = [
            gen_id
            for gen_id, p in self._participants.items()
            if p.lease is not None and getattr(p.lease, "is_released", False)
        ]
        for gen_id in dead:
            del self._participants[gen_id]
            if self._epoch_winner == gen_id:
                self._epoch_winner = None

    def _committed_locked(self) -> int:
        self._prune_locked()
        ledger = sum(p.tokens for p in self._participants.values() if p.holds_kv)
        # Whichever is larger, because both are real: the ledger knows what live generations
        # were admitted on, `resident` what the cache holds, finished requests included.
        if self._resident is None:
            return ledger
        # Not max(ledger, resident): that double counts every chat `resident` ALREADY
        # includes, and paused chats with the cache half empty. A measured chat is inside
        # `resident`, so take the larger of the two; an unmeasured one is added on top.
        holders = [p for p in self._participants.values() if p.holds_kv]
        measured = sum(p.tokens for p in holders if p.measured)
        pending = sum(p.tokens for p in holders if not p.measured)
        return max(self._resident, measured) + pending

    def _winner_locked(self) -> Optional[Participant]:
        """The one generation that keeps decoding, stable for an epoch. Promoted (starved)
        first, then longest-wins, then arrival order so the choice is deterministic.
        """
        held = self._participants.get(self._epoch_winner) if self._epoch_winner else None
        if held is not None and held.state == ParticipantState.DECODING:
            return held
        # A parked or tools-running holder is not a candidate: crowning it would pause
        # everyone for nobody's benefit.
        candidates = [
            p for p in self._participants.values() if p.state == ParticipantState.DECODING
        ]
        if not candidates:
            self._epoch_winner = None
            return None
        winner = min(candidates, key = lambda p: (not p.promoted, -p.tokens, p.seq))
        self._epoch_winner = winner.gen_id
        # Cured on crowning: resetting on resume would defeat the rule, since a resumed
        # victim can be preempted again at once.
        winner.consecutive_preemptions = 0
        return winner

    def plan_preemptions(self, *, needed: int = 0) -> List[Participant]:
        """Who must stop so ``needed`` more tokens fit. Empty when nothing must.

        Sets each victim's ``preempt_event`` and moves it to PREEMPTING, so the decision and
        the signal cannot drift apart. The caller aborts only the upstream stream, then calls
        ``lease.preempt()`` once that response is closed.
        """
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return []
            buffer = self._buffer_locked()
            ceiling = max(0, self._budget - buffer)
            total = self._committed_locked()
            want = max(0, int(needed or 0))
            # The two figures the sweep chooses between, recorded when they disagree:
            # `ledger` is prompt ESTIMATES plus counted output, `resident` is exact.
            ledger = sum(p.tokens for p in self._participants.values() if p.holds_kv)
            # Rate limited: drift over 256 is the normal case, and unthrottled this drowned
            # the events around it.
            _now = time.monotonic()
            if (
                self._resident is not None
                and abs(self._resident - ledger) > 256
                and _now - self._drift_logged_at >= 5.0
            ):
                self._drift_logged_at = _now
                # `ledger` decides NOTHING and is logged as a second opinion. `committed`
                # is what the watermark compares, and its split matters more: `measured` is
                # cells llama-server can see, `pending` is room held for prompts that are
                # not in the cache yet.
                holders = [p for p in self._participants.values() if p.holds_kv]
                _log.info(
                    "llama preemption ledger-drift: committed=%s resident=%s measured=%s "
                    "pending=%s ledger=%s ceiling=%s buffer=%s prefilling=%s want=%s "
                    "holders=%s",
                    total,
                    self._resident,
                    sum(p.tokens for p in holders if p.measured),
                    sum(p.tokens for p in holders if not p.measured),
                    ledger,
                    ceiling,
                    buffer,
                    # The batch term's input, so the ceiling's movement can be attributed.
                    self._pending_prefill_locked(),
                    want,
                    len(holders),
                )
            if total + want <= ceiling:
                return []
            # Parked holders first: they hold KV and consume no compute, so their room is
            # the cheapest to take. Then NEWEST first, as vLLM V1 does, so the work already
            # done is the work preserved; taking the LARGEST decoder ranked 5th of 7
            # simulated policies and worst of all on fairness.
            #
            # No generation is exempt, including one that arrived a moment ago: its charge is
            # a RESERVATION llama-server cannot see, so cancelling it evicts no cells, and
            # arming precedes the first upstream request, so no prefill is paid either. A
            # fixed epoch winner was exempt once and simply grew until it filled the window.
            #
            # `holds_kv` as well as `preemptable`: a participant whose cells an idle reclaim
            # erased keeps `preemptable` True while `holds_kv` has gone False, so the loop
            # below would subtract stale `tokens` from a figure that never included them.
            victims = [p for p in self._participants.values() if p.preemptable and p.holds_kv]
            # Promotion is a THRESHOLD, not a continuous term: ordering by the preemption
            # count would quietly become least-preempted-first, which was not benchmarked.
            victims.sort(
                key = lambda p: (
                    p.state != ParticipantState.PARKED_ON_TOOL,
                    p.promoted,
                    -p.seq,
                )
            )
            # Always leave one holder standing: pausing the last one decodes nothing and
            # makes the incumbent replay everything. One HOLDER, not one preemptable victim,
            # since a holder this sweep cannot choose is standing already. One already
            # PREEMPTING is on its way out and does not count, or a sweep between the
            # decision and the pause takes the last decoder too.
            standing = any(
                p.holds_kv and not p.preemptable and p.state != ParticipantState.PREEMPTING
                for p in self._participants.values()
            )
            spare = len(victims) if standing else max(0, len(victims) - 1)
            chosen: List[Participant] = []
            for victim in victims[:spare]:
                if total + want <= ceiling:
                    break
                # `total` is the PROJECTION deciding how many victims are needed; the
                # participant stays KV-holding until on_preempted says the stream stopped.
                total -= victim.tokens
                victim.consecutive_preemptions += 1
                victim.state = ParticipantState.PREEMPTING
                victim.preempt_chosen_at = time.monotonic()
                victim.preempt_event.set()
                chosen.append(victim)
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
                winner = self._epoch_winner,
                slots = max(1, self._slots or 1),
                tools_running = states.count(ParticipantState.TOOLS_RUNNING),
                prefilling = self._pending_prefill_locked(),
                holders = sum(1 for p in self._participants.values() if p.holds_kv),
            )


def wait_for_reclaim(
    scrape: Callable[[], Optional[dict]],
    *,
    target_processing: int,
    timeout_s: float = DEFAULT_RECLAIM_BARRIER_TIMEOUT_S,
    poll_s: float = DEFAULT_RECLAIM_BARRIER_POLL_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Block until llama-server reports at most ``target_processing`` requests in flight.

    Socket teardown is not evidence the cells are free. A BARRIER, never an attribution: the
    gauge cannot say which generation owns a slot. False means it could not be confirmed
    within ``timeout_s``, or ``/metrics`` is unavailable, and the caller proceeds anyway,
    since blocking forever on a gauge that may never answer is worse than the overrun.

    ``scrape`` is injected so this is testable without a server.
    """
    deadline = monotonic() + max(0.0, float(timeout_s))
    while True:
        metrics = scrape()
        if metrics is not None:
            processing = metrics.get("requests_processing")
            if processing is None:
                return False
            if int(processing) <= max(0, int(target_processing)):
                return True
        if monotonic() >= deadline:
            return False
        sleep(poll_s)


class ControllerPreemptionPolicy:
    """Binds one generation to the controller, satisfying ``PreemptionPolicy``.

    ``await_resume`` is where a policy bug would show up as a hung chat, so it is bounded
    twice over: by the caller's timeout, and by refusing to wait once the room is back.
    Returning False means "give up and finish the turn".

    ``loop`` is why this is a separate class rather than the controller implementing the
    protocol: the stream funnels are synchronous on a worker thread while ``resume_async``
    awaits the queue's slot acquire, so the two are joined with ``run_coroutine_threadsafe``.
    With no loop the turn simply finishes.
    """

    __slots__ = ("_controller", "_gen_id", "_signal", "_resumes", "_loop")

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
        self._loop = loop

    def should_preempt(self) -> bool:
        return self._signal.is_set()

    def on_preempted(self, checkpoint: StreamCheckpoint) -> None:
        """The upstream response is closed, so the tokens may go back. Order matters, as
        ``_release_admission`` states: hand the lease back only once nothing can still be
        decoding against it.
        """
        self._resumes = checkpoint.resumes
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
        # Decision to cells-released, in milliseconds. Not derivable from any other line:
        # `resident` is /metrics-polled, so it reports the poll interval instead.
        chosen_at = participant.preempt_chosen_at
        if chosen_at:
            _log.info(
                "llama preemption evict-latency: gen_id=%s ms=%.1f tokens=%s",
                self._gen_id,
                (time.monotonic() - chosen_at) * 1000.0,
                participant.tokens,
            )
            participant.preempt_chosen_at = 0.0
        # Before the state change, so a sweep in between sees the larger figure.
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
                # A handback that fails must not take the conversation with it; the tokens
                # come back when the lease is released either way.
                pass

    def await_resume(self, timeout: Optional[float] = None) -> bool:
        # None means "caller stated no preference", NOT "wait forever".
        if timeout is None:
            timeout = DEFAULT_RESUME_WAIT_TIMEOUT_S
        if self._resumes >= DEFAULT_MAX_PREEMPT_RESUMES:
            # Churning on one chat helps nobody; let it finish and take its room back.
            _log.warning(
                "llama preemption resume-cap: gen_id=%s resumes=%s", self._gen_id, self._resumes
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
        # Re-stated, not remembered: a resumed run carries the partial it generated, so it
        # needs more room than it was preempted holding.
        want = max(0, int(participant.tokens or 0))
        # `want` grows with every pause and can pass the ceiling the wait is measured
        # against, so without these two the wait is for room no eviction can produce.
        if self._controller.cannot_ever_fit(want):
            # Bigger than the cache. Ending here reports `length`, which the continuation
            # path resumes against a fresh window; waiting hangs until the client gives up.
            _log.info(
                "llama preemption too-large: gen_id=%s want=%s (exceeds the cache; "
                "finishing the turn)",
                self._gen_id,
                want,
            )
            return False
        if self._controller.outgrew_the_shared_ceiling(want):
            # Fits alone but beside nobody; `room_for` grants it the cache once the others
            # are out, so this only records why the wait may be long.
            _log.info(
                "llama preemption needs-the-cache: gen_id=%s want=%s (past the shared "
                "ceiling; waiting for the cache to itself)",
                self._gen_id,
                want,
            )
        _log.info("llama preemption awaiting-room: gen_id=%s want=%s", self._gen_id, want)
        # Wait for the cache to really have room before taking the lease back, or the queue
        # hands a resume out on its own optimistic accounting and the next sweep evicts the
        # same chat again. The clock measures STALL, not elapsed time: a flat deadline killed
        # two chats while the cache was steadily turning over. `hard_deadline` covers what a
        # stall detector cannot see: a cache that churns while THIS chat is unserved.
        started = time.monotonic()
        deadline = started + timeout
        hard_deadline = started + timeout * MAX_RESUME_WAIT_MULTIPLE
        last = self._controller.progress_signature()
        # Fresh reading before the first question: this grant lets a chat back in carrying
        # its whole replayed partial.
        self._controller.refresh_residency()
        # try_grant_resume, not room_for: the room must be BOOKED at the instant it is
        # found, or two waiters both find the same space and both take it.
        while not self._controller.try_grant_resume(self._gen_id, want):
            self._controller.refresh_residency()
            now = time.monotonic()
            current = self._controller.progress_signature()
            # A holder parked on a tool moves nothing, so the signature freezes while a web
            # search runs. A tool that never returns is caught by `hard_deadline` instead.
            _snap = self._controller.snapshot()
            if _snap.parked > 0 or getattr(_snap, "tools_running", 0) > 0:
                deadline = now + timeout
            if current != last:
                # ANY change resets it: the signature covers the whole backend, so "no
                # progress" claims NOTHING moved, which is what justifies abandoning a turn.
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
        try:
            try:
                coro = lease.resume_async(
                    want, timeout_s = timeout, progress = self._controller.progress_signature
                )
            except TypeError:
                # An older lease without the stall-aware wait.
                coro = lease.resume_async(want, timeout_s = timeout)
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            # Backstop for a loop that never runs the coroutine; resume_async is bounded.
            got = bool(future.result(timeout = timeout + 5.0))
            if not got:
                # The grant booked the room and nothing will use it, so hand it back.
                self._controller.note_resume_failed(self._gen_id)
            _log.info(
                "llama preemption %s: gen_id=%s want=%s",
                "resumed" if got else "gave-up",
                self._gen_id,
                want,
            )
            return got
        except Exception as exc:
            # Includes the future timing out. Either way the room did not come back.
            _log.warning(
                "llama preemption resume-failed: gen_id=%s want=%s error=%s",
                self._gen_id,
                want,
                exc,
            )
            return False

    def on_resumed(self) -> None:
        self._controller.note_resumed(self._gen_id)

    def on_declined(self) -> None:
        """This generation was chosen to pause and is not going to. Undo the decision.

        Called instead of the pause handshake, never beside it: `on_preempted` hands the
        lease back, and a stream that goes on decoding must keep it.
        """
        _log.info("llama preemption declined: gen_id=%s (finishing instead)", self._gen_id)
        self._controller.note_declined(self._gen_id)


def _slot_decoded(slot: dict) -> int:
    """Tokens this slot has generated so far, from `/slots`. ``next_token`` is a one-element
    LIST in some builds and a bare object in others; anything else answers zero, since an
    occupancy read that throws would take the whole watermark sweep down with it.
    """
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
    """Tokens actually resident in the cache, INCLUDING slots that are idle.

    llama.cpp keeps a slot's prompt cache after its request finishes, for prefix reuse, and
    that residue belongs to no live generation: neither the ledger nor /metrics can see it,
    and one idle slot was found holding an entire 16384-cell cache.

    ``fetch`` returns the parsed ``GET /slots`` array, or None when unavailable. None means
    "cannot say", never "empty".
    """
    slots = fetch()
    if not slots:
        return None
    resident = 0
    idle_tokens = 0
    idle = []
    for slot in slots:
        # WHICH field this came from decides whether the decoded tokens still have to be
        # added, so the two cannot collapse into one `or` chain: `n_prompt_tokens` is TOTAL
        # residency and already contains everything generated.
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
        # Plus what it has GENERATED: `/slots` reports the prompt, and on a long answer the
        # decoded tokens are most of the occupancy. Only while processing, since a finished
        # slot's cache already holds the sequence, and only when the figure above does not
        # already include them, or the total scores prompt + 2 x decoded.
        if slot.get("is_processing") and not counts_generated:
            tokens += _slot_decoded(slot)
        # Idle slots count too: `try_clear_idle_slots` runs FROM the KV-full retry, so cells
        # freed only by crashing first are occupied as far as this is concerned.
        resident += max(0, tokens)
        if not slot.get("is_processing") and tokens > 0:
            idle_tokens += max(0, tokens)
            idle.append((slot.get("id"), max(0, tokens)))
    # Largest first: the fewest erases free the most.
    idle.sort(key = lambda pair: -pair[1])
    return {
        "resident": resident,
        # Reported separately so the caller can free it BEFORE pausing anybody.
        "idle_tokens": idle_tokens,
        "idle": idle,
        "slots": len(slots),
    }


def reclaim_idle_slots(
    occupancy: Optional[dict], erase: Callable[[int], int], *, needed: int
) -> int:
    """Free dead residue before asking a live chat to stop. Returns tokens freed.

    Strictly better than preempting: an idle slot's cache belongs to a finished request, so
    erasing it costs a future prefix-cache hit and nothing else. llama.cpp does this itself
    only on the KV-full retry, once the decode has ALREADY failed.
    """
    if not occupancy or needed <= 0:
        return 0
    freed = 0
    for slot_id, tokens in occupancy.get("idle") or ():
        if freed >= needed:
            break
        if slot_id is None:
            continue
        try:
            freed += max(0, int(erase(slot_id) or 0))
        except Exception:
            # An erase that fails leaves the residue; the caller falls back to preempting.
            continue
    return freed


_CONTROLLERS_LOCK = threading.Lock()
_CONTROLLERS: Dict[str, PreemptionController] = {}


def get_preemption_controller(key: str) -> PreemptionController:
    with _CONTROLLERS_LOCK:
        controller = _CONTROLLERS.get(key)
        if controller is None:
            controller = PreemptionController(key)
            _CONTROLLERS[key] = controller
            # Each model load takes a fresh port, so it registers a new key; drop the idle
            # ones. is_idle takes each controller's own lock, so the list is built first.
            stale = [k for k, c in _CONTROLLERS.items() if k != key and c.is_idle()]
            for k in stale:
                del _CONTROLLERS[k]
        return controller


def reset_preemption_controllers() -> None:
    with _CONTROLLERS_LOCK:
        _CONTROLLERS.clear()
