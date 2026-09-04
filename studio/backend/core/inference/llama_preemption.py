# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who pauses, who keeps decoding, and when the room is actually back.

One llama-server holds one KV cache. Studio launches it with ``--parallel N --kv-unified
-c N``, which is N cells TOTAL while every slot is told it has N. llama-server polices
only ``prompt_tokens < slot.n_ctx``, never "is there room right now", so chats that each
fit on their own are all admitted and then collide; ``server-context.cpp`` then calls
``send_error`` on EVERY processing slot. Measured on 2026-09-01: four tool chats at
``-c 16384 --parallel 4``, all four lost together.

Admission (``llama_admission``) decides who gets in. This module decides who has to stop
once they are in, so an overrun becomes a pause instead of four dead conversations. It
owns policy and state only: aborting the upstream stream and resuming it belong to the
caller, and the contract for that is on ``LlamaAdmissionLease.preempt``.

This is request-granularity preemption with recompute, which is what vLLM does when its
KV cache is exhausted.
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


# Off falls back to step 1's wire clamp alone, which is the behaviour that predates any
# of this. Mirrors UNSLOTH_LLAMA_ADMISSION_KV_BUDGET, and is parsed by the same helper so
# it accepts the same spellings.
PREEMPT_ENV = "UNSLOTH_LLAMA_ADMISSION_PREEMPT"
DEFAULT_PREEMPT_ENABLED = True

# Room held clear of the budget. A commitment is an estimate, and the cost of being a
# little wrong is the crash this module exists to prevent, so the last few per cent are
# never handed out.
def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        return default
    # A ratio outside (0, 0.5) is a typo, not a policy: zero disables the margin that
    # keeps the cache off the retry path, and half is already drastic.
    return value if 0.0 < value < 0.5 else default


# Raised from 0.05 after measurement. At five per cent the watermark sat at 95% of the
# cache and llama-server still entered its shrinking-batch retry ten times in one run,
# which is the path that throws the speculative sub-batch error. The margin has to cover
# what this side cannot measure exactly: prompt figures are estimates rather than
# tokenisations, residency is sampled on a TTL rather than continuously, and up to 32
# tokens per slot are generated between reports. Tunable because the right number depends
# on the cache size and the traffic, and guessing it once is how it was wrong before.
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


# Kept only so an operator who set the old knob is not silently ignored; see
# `preemption_buffer_tokens` for why a fraction of the cache is the wrong shape.
DEFAULT_PREEMPT_BUFFER_RATIO = _float_env("UNSLOTH_LLAMA_PREEMPT_BUFFER_RATIO", 0.15)
# Reaction headroom for ONE decoding slot: what it can generate between a watermark sweep
# and the moment a chosen victim's stream actually stops holding its cells.
DEFAULT_PREEMPT_BUFFER_PER_SLOT = 192
DEFAULT_PREEMPT_BUFFER_MIN_TOKENS = 256

# A resume is cheap (the prefix cache usually still holds the prompt) but not free, so a
# pathological loop that pauses the same chat forever is bounded. Deliberately far above
# _MAX_LENGTH_CONTINUATIONS: that one caps how often a model may be asked to finish its own
# sentence, a quality judgement, while this caps churn under contention, a capacity one.
DEFAULT_MAX_PREEMPT_RESUMES = 32

# Approved anti-starvation rule: a chat preempted this many times in a row outranks
# longest-wins for the next epoch.
PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS = 3

# What a chat running ALONE must leave clear. The ratio buffer is reaction headroom: it
# exists so the sweep can act before several chats growing at once overrun the pool. A
# chat with the cache to itself has nothing to evict and no reaction to make, so the only
# cells that must stay clear are its own drafts plus a margin for the estimate.
#
# This is what makes a chat that outgrew the shared ceiling runnable at all. Without it
# such a chat can never be admitted OR resumed and waits until its client gives up, which
# is the live hang where one chat of four stayed open for a whole 2400s deadline while
# llama-server sat idle with every slot released. Simulated, adding it took makespan on
# the tool-heavy regime from 166799 steps to 924 and starvation from 1.6 chats to none.
SOLO_MARGIN_DIVISOR = 200

# The reclaim barrier gives up after this long and lets the replacement in anyway. A
# server without --metrics never answers, and blocking a conversation forever on a gauge
# that may not exist is worse than the overrun it guards.
DEFAULT_RECLAIM_BARRIER_TIMEOUT_S = 10.0
DEFAULT_RECLAIM_BARRIER_POLL_S = 0.05

# A pause must never be able to outlive the thing it was waiting for. The stream calls
# await_resume() with no argument, and an unbounded wait there turned three paused chats
# into a 33-minute hang with nothing decoding: strictly worse for a user than the crash
# this replaced, because a crash at least ends. On expiry the turn finishes with what it
# has, which is the behaviour that predates preemption, and the wire clamp still bounds
# what any request may occupy.
DEFAULT_RESUME_WAIT_TIMEOUT_S = 90.0

# The absolute bound on one resume wait, as a multiple of the stall timeout above. Only
# reached when the cache keeps moving but never has room for THIS chat, which the stall
# detector cannot distinguish from healthy queueing. 20 x 90s = 30 minutes, comfortably
# longer than the slowest answer measured here and still finite.
MAX_RESUME_WAIT_MULTIPLE = 20


class LlamaStreamPreempted(Exception):
    """The upstream stream was aborted to free KV, not abandoned.

    Distinct from ``_LlamaStreamCancelled`` on purpose. A cancel ends the turn;
    this one is expected to be caught and resumed, so anything that treats a
    dead stream as failure must not see it.
    """


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

    # Spelled like threading.Event so the policy half reads the same as any other
    # signal it sets, and so a Participant's field can be swapped for a bare Event in a
    # test double without the call sites changing.
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

    def has_reasoning_resume_point(self) -> bool:
        """Whether there is a thought to continue, absent any visible prose.

        Distinct from ``has_resume_point`` because the two resume through different
        fields: prose goes back as ``content`` and is extended, a thought goes back as
        ``reasoning_content`` and re-opens. Sending a thought as content would render it
        as the answer.
        """
        return not self.visible_text.strip() and bool(self.reasoning_text.strip())

    def kept_chars(self) -> int:
        """Characters carried across the pause, prose or thought.

        Reported rather than ``len(visible_text)`` alone: that read zero on every pause
        of a reasoning model and made a livelock look like an orderly pause.
        """
        if self.has_resume_point():
            return len(self.visible_text)
        return len(self.reasoning_text) if self.has_reasoning_resume_point() else 0


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


class DeferredPreemptionPolicy:
    """A policy handed to the stream before the real one can exist.

    The tool-loop generator is BUILT before admission returns and only ITERATED after,
    so the object passed in cannot yet know its lease. This forwards once bound and
    behaves as ``NullPreemptionPolicy`` until then, which is correct rather than merely
    convenient: nothing can pause before the generator is iterated, and an unbound
    ``await_resume`` answering False means "finish the turn", the behaviour that predates
    preemption.
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


class ParticipantState:
    """Where a generation is, which decides whether it may be preempted.

    Strings rather than an enum so they cross the SSE boundary to the paused/queued UI
    without a translation table.
    """

    QUEUED = "queued"
    DECODING = "decoding"
    # Stopped on a tool approval prompt: holds KV, consumes no compute. The cheapest
    # room to reclaim, so these are taken first.
    PARKED_ON_TOOL = "parked_on_tool"
    # Calls parsed and tools executing. Never preempted: nothing is decoding, so pausing
    # buys no compute, and it is inside the unsafe window the tool loop forbids.
    TOOLS_RUNNING = "tools_running"
    # Asked to stop, but still decoding until the stream reaches a safe point. Holds KV:
    # the room is not free because we decided it should be. Counting it as free the
    # instant a victim was chosen is what let four chats commit 16384 of a 16384 cache
    # while the controller believed 8192 were in use.
    PREEMPTING = "preempting"
    PAUSED = "paused"
    DONE = "done"


# Holds KV at llama-server, so it counts against the budget.
_HOLDS_KV = frozenset({
    ParticipantState.DECODING,
    ParticipantState.PARKED_ON_TOOL,
    ParticipantState.TOOLS_RUNNING,
    ParticipantState.PREEMPTING,
})

# May be asked to stop. QUEUED holds nothing yet, PAUSED already stopped, DONE is gone,
# and TOOLS_RUNNING is excluded for the reason on its constant.
_PREEMPTABLE = frozenset({
    ParticipantState.DECODING,
    ParticipantState.PARKED_ON_TOOL,
})
# PREEMPTING is deliberately absent: it has already been asked and asking twice would
# double-count the room its pause is going to free.


def preemption_enabled() -> bool:
    return _bool_env(PREEMPT_ENV, DEFAULT_PREEMPT_ENABLED)


def preemption_buffer_tokens(
    budget: int, *, draft_tokens: int = 0, slots: int = 1, batch_tokens: int = 0
) -> int:
    """Tokens held clear of ``budget``. Zero for an unknown budget, which disables it.

    Never the whole cache. The 256-token floor is larger than a very small ``-c``, and a
    buffer that swallows the budget leaves a ceiling of zero, which reads as "no room
    for anyone" and would preempt every participant on every call, forever. Capped at
    half so a tiny cache degrades to a smaller buffer rather than to a livelock.

    **Speculative decoding is charged on top.** A drafter puts up to
    ``--spec-draft-n-max`` tokens into the cache per slot BEFORE they are accepted or
    rejected, and admission never sees them: they are not part of any request's prompt or
    output. With every slot drafting at once that is ``draft_tokens * slots`` cells this
    module would otherwise believe were free. Observed 2026-09-01: the cache filled,
    llama-server halved n_batch 128 -> 4 looking for room, and at that width the
    speculative indices fell outside the sub-batch and it threw
    ``speculative batch index 4 is not inside the current sub-batch [0, 4)``
    (upstream ggml-org/llama.cpp#24840, where the retry path shifts ``slot.i_batch`` by
    the offset but never ``slot.spec_i_batch``). Reserving the drafts keeps the cache off
    the retry path that exposes it.
    """
    if budget <= 0:
        return 0
    # PER SLOT, not per cache. The buffer is reaction headroom: the cells that can be
    # generated between one watermark sweep and a chosen victim's stream actually
    # stopping. That quantity scales with how many chats are decoding at once, and not at
    # all with how big the cache is, so a fraction of the budget is the wrong shape.
    #
    # It was a fraction (15%) until 2026-09-03, chosen by guess. Simulated across cache
    # sizes, slot counts and eviction latencies, that was not merely wasteful but
    # actively harmful: on the default 16384 cache with four slots it held back 2458
    # tokens against the 768 this reserves, and makespan was 26890 steps against 239.
    # An oversized buffer lowers the shared ceiling, so more chats outgrow it, and each
    # of those then has to wait for the cache to itself. The cache serialises.
    #
    # 192 tokens per slot is the smallest value with zero overflow in every configuration
    # tried: 4096 to 65536 cells, 4 and 8 slots, and eviction latencies from one sweep
    # interval to sixteen. 128 per slot overflows; 256 costs a small cache dearly.
    per_slot = _int_env("UNSLOTH_LLAMA_PREEMPT_BUFFER_PER_SLOT", DEFAULT_PREEMPT_BUFFER_PER_SLOT)
    slot_count = max(1, int(slots or 1))
    reserve = max(DEFAULT_PREEMPT_BUFFER_MIN_TOKENS, per_slot * slot_count)
    # And room for the batch llama.cpp is actually processing, which is the term all of
    # the above was missing. The cache does not fail when it is full of tokens, it fails
    # when the next BATCH does not fit: llama-server prefills in chunks of --batch-size
    # (2048 by default), so a resumed chat replaying 5000 tokens asks for 2048 free cells
    # at once, not one at a time.
    #
    # Measured 2026-09-03, and it is why the watermark kept looking innocent: across 1329
    # samples peak residency was 13540 against a 15592 ceiling, never once over, while
    # llama-server halved its batch 19 times (2048, 1024, ... 4) and threw 4 speculative
    # sub-batch errors. 16384 - 13540 leaves 2844 free, which one 2048 chunk fits and two
    # concurrent ones do not. A 792 token buffer cannot cover a 2048 token chunk.
    #
    # max() rather than a sum: reaction headroom and batch headroom buy the same thing,
    # space for the next step, so the larger of the two covers both.
    reserve = max(reserve, max(0, int(batch_tokens or 0)))
    # Drafts are additional. They are cells the drafter puts in BEFORE acceptance, on top
    # of whatever the batch needs, and admission never sees them.
    reserve += max(0, int(draft_tokens or 0)) * slot_count
    # Still never the whole cache: a large draft window on a small -c must degrade to a
    # tight buffer, not to a ceiling of zero.
    return min(reserve, max(1, budget // 2))


@dataclass(**_SLOTS)
class Participant:
    """One in-flight generation, from the preemptor's point of view."""

    gen_id: str
    seq: int
    lease: Optional[LlamaAdmissionLease] = None
    tokens: int = 0
    # What admission charged. Live growth is added to THIS rather than replacing it, so a
    # report of "n tokens generated" cannot silently drop the prompt already resident.
    base_tokens: int = 0
    # True once this generation's prompt is known to be IN the cache, either because a
    # round boundary restated it or because it has decoded at least once. Until then its
    # charge is a reservation for a prefill that has not happened, which the resident
    # figure cannot see; afterwards its tokens are already inside that figure and adding
    # them to it would count the same cells twice.
    measured: bool = False
    state: str = ParticipantState.DECODING
    consecutive_preemptions: int = 0
    # Set when this generation must stop. The caller aborts ONLY the upstream
    # llama-server stream on it and keeps its own generator, ledgers, conversation and
    # SSE response alive; a shared cancel event could not express that, because six
    # separate consumers treat cancellation as terminal.
    preempt_event: PreemptSignal = field(default_factory = PreemptSignal)

    @property
    def promoted(self) -> bool:
        return self.consecutive_preemptions >= PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS

    @property
    def holds_kv(self) -> bool:
        return self.state in _HOLDS_KV

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


class PreemptionController:
    """Victim choice and the epoch, for one llama-server backend.

    Not a scheduler: nothing here runs generations or owns a thread. Callers report where
    they are, ask whether there is room, and are told who must stop.
    """

    __slots__ = (
        "key", "_lock", "_participants", "_seq", "_epoch_winner", "_budget",
        "_kv_unified", "_draft_tokens", "_slots", "_batch_tokens", "_resident",
        "_residency_probe",
    )

    def __init__(self, key: str):
        self.key = key
        self._lock = threading.Lock()
        self._participants: Dict[str, Participant] = {}
        self._seq = 0
        self._epoch_winner: Optional[str] = None
        self._budget = 0
        # Preemption reclaims only where an idle slot's cells can be purged, which
        # upstream gates on kv_unified ALONE (try_clear_idle_slots, server-context.cpp:1656,
        # reached from the KV-full retry at :3702). Deliberately NOT gated on
        # idle_slot_clearing_active: that flag tracks --cache-idle-slots, which governs the
        # PROACTIVE sweep over slots whose task is still alive. A preempted task has ENDED,
        # so its slot goes non-processing and the reactive purge above applies on every
        # platform, including Windows full-offload (--cache-ram 0, #5692) where the clamp
        # would otherwise be the only protection.
        self._kv_unified = False
        # Speculative drafts occupy cells no request is charged for; see
        # preemption_buffer_tokens.
        self._draft_tokens = 0
        # --batch-size llama-server was launched with, so the buffer can cover one chunk.
        self._batch_tokens = 0
        self._slots = 1
        # True cells resident in the cache from the last GET /slots, or None when it
        # could not be read. Includes the residue of FINISHED requests, which the ledger
        # cannot see and which is what kept the watermark firing too late.
        self._resident: Optional[int] = None
        # Set by the route to a callable that re-reads GET /slots and calls
        # note_resident. Optional: everything works from the ledger alone, less precisely.
        self._residency_probe: Optional[Callable[[], None]] = None

    def configure(
        self,
        *,
        budget: Optional[int] = None,
        kv_unified: Optional[bool] = None,
        draft_tokens: Optional[int] = None,
        slots: Optional[int] = None,
        batch_tokens: Optional[int] = None,
    ) -> None:
        """Re-read the cache this backend actually allocated.

        Called per request like ``LlamaAdmissionQueue.reserve`` re-reads ``_budget``: a
        reload can relaunch llama-server at a different ``-c``, and a stale budget would
        keep planning against a cache that no longer exists.
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
        stream, and setting the participant's signal reaches nobody. Observed live: four
        chats armed, two were selected as victims, neither ever paused, and the cache
        overran exactly as it did before any of this existed.
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
            self._participants[gen_id] = participant
            return participant

    def unregister(self, gen_id: str) -> None:
        """Drop a finished generation and end its epoch if it held one."""
        with self._lock:
            self._participants.pop(gen_id, None)
            if self._epoch_winner == gen_id:
                self._epoch_winner = None

    def _solo_ceiling_locked(self) -> int:
        """The cache less only what a lone chat's own drafts and estimate error need."""
        margin = self._draft_tokens + max(64, self._budget // SOLO_MARGIN_DIVISOR)
        return max(1, self._budget - margin)

    def outgrew_the_shared_ceiling(self, want: int) -> bool:
        """Whether `want` can never fit beside anyone, however much is evicted.

        Such a generation must not wait: no combination of preemptions will admit it, so
        waiting is a deadlock rather than a delay. It runs alone instead.
        """
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return False
            ceiling = max(0, self._budget - self._buffer_locked())
            return int(want or 0) > ceiling

    def cannot_ever_fit(self, want: int) -> bool:
        """Whether `want` exceeds the cache itself, so not even running alone helps.

        The turn ends with what it has, which llama-server reports as a `length` finish
        and the existing continuation path resumes against a fresh window. Parking it
        forever instead is the hang.
        """
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return False
            return int(want or 0) > self._solo_ceiling_locked()

    def room_for(self, gen_id: str, want: int) -> bool:
        """Whether a paused generation may start again yet.

        Against the LIVE total, which is the whole point. Resuming on the admission
        queue's accounting instead let a chat back in while the cache was still over its
        watermark, so the next sweep evicted it again immediately: 44 preemptions across
        four chats, one of them producing 611 characters in 374 seconds. A resume that is
        undone by the next sweep is worse than waiting, because it pays a prefill for
        nothing.

        The winner is excluded from nobody's arithmetic here: it is counted like any
        other holder, so a resume waits for it to finish rather than squeezing in beside
        it. That is the approved policy, "let the longest chat continue, then continue
        the rest once it frees its room".
        """
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return True
            return self._room_for_locked(gen_id, want)

    def try_grant_resume(self, gen_id: str, want: int) -> bool:
        """Decide there is room AND take it, without letting go of the lock between.

        `room_for` only answers a question, and two paused chats asking it at the same
        moment both get yes: PAUSED is not in `_HOLDS_KV`, so neither appears in the
        other's arithmetic, and nothing books the space between the answer and the
        prefill that follows it. Both then resume and prefill together.

        That is the run that produced 3 context-exhaustion errors and 4 speculative
        sub-batch errors (upstream #24840) on 2026-09-03 with three chats waiting at
        once, while the sampled residency never once passed the ceiling -- because the
        overflow happened inside a single prefill, between two samples, and the ledger
        had never been told the room was spoken for. The run before it, with fewer
        simultaneous waiters, was completely clean.

        So book it here. The grant marks the participant DECODING and charges it `want`
        immediately, which is what makes the next caller see the room as taken. It is
        marked unmeasured on purpose: its cells were freed when it paused and its prefill
        has not happened yet, so `want` must be ADDED to the resident figure rather than
        compared with it, exactly like any other chat that has not prefilled.

        Roll back with `note_resume_failed` if the resume does not go through, or the
        booking becomes room nobody is using.
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
                participant.state = ParticipantState.DECODING
            return True

    def note_resume_failed(self, gen_id: str) -> None:
        """Give back a grant whose resume never happened."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None and participant.state == ParticipantState.DECODING:
                participant.state = ParticipantState.PAUSED

    def _room_for_locked(self, gen_id: str, want: int) -> bool:
        """The arithmetic behind `room_for`, callable by a holder of the lock."""
        ceiling = max(0, self._budget - self._buffer_locked())
        ledger_others = sum(
            p.tokens
            for gid, p in self._participants.items()
            if p.holds_kv and gid != gen_id
        )
        # Resident cells count too, minus whatever this generation itself still holds, or
        # an idle slot's leftovers would be invisible here exactly as they were to the
        # watermark. Reading only the ledger said "yes, resume" against a cache an idle
        # slot had already filled.
        others = ledger_others
        if self._resident is not None:
            mine = self._participants.get(gen_id)
            others = max(others, self._resident - (mine.tokens if mine else 0))
        need = max(0, int(want or 0))
        others = max(0, others)
        if others + need <= ceiling:
            return True
        # Outgrew the shared ceiling: it can still run once it has the cache to itself,
        # and it must, or it waits for room that no eviction can ever make.
        if need > ceiling and others == 0:
            return need <= self._solo_ceiling_locked()
        return False

    def observe(self, gen_id: str, generated: int) -> List["Participant"]:
        """Live growth during generation, and the eviction check that follows it.

        THE watermark sweep. Admission deliberately overcommits now -- every chat is
        permitted the whole window -- so nothing about the arithmetic prevents the cache
        filling. What prevents it is being told, often, how big each generation has
        actually become, and evicting when the running total nears the ceiling. This is
        the `n1..np changing over time` the design asks for, and it is why the sweep
        cannot live only between rounds: one round can generate thousands of tokens, and
        a check that happens after it is a check that happens too late.

        Returns whoever must stop, already signalled.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None:
                participant.tokens = participant.base_tokens + max(0, int(generated or 0))
                # A token came back, so the prompt behind it is prefilled and resident.
                participant.measured = True
        # Outside the lock: plan_preemptions takes it, and it is not reentrant.
        return self.plan_preemptions(needed = 0)

    def set_residency_probe(self, probe: Optional[Callable[[], None]]) -> None:
        """Register a way to re-read the cache on demand.

        The ledger adds up prompt ESTIMATES; llama-server's per-slot totals are exact.
        Where the two disagree the exact one must win, and the moment that matters most
        is granting a resume: a chat comes back carrying its whole replayed partial, so
        a reading a second old can be a thousand tokens stale by the time it is used.
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
            # A failed read leaves the previous figure in place, which is what the
            # ledger-only path already assumes.
            _log.debug("residency probe failed", exc_info = True)

    def note_resident(self, resident: Optional[int]) -> None:
        """The cache as llama-server actually sees it. None means the read failed."""
        with self._lock:
            self._resident = None if resident is None else max(0, int(resident))

    def note_tokens(self, gen_id: str, tokens: int) -> None:
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None:
                participant.tokens = max(0, int(tokens or 0))
                # Re-baselined: a round boundary restates the whole conversation, so
                # later growth is measured from here rather than from admission.
                participant.base_tokens = participant.tokens
                participant.measured = True

    def note_replayed(self, gen_id: str, tokens: int) -> None:
        """Tokens a paused attempt decoded that the NEXT attempt sends back as prompt.

        They do not leave the cache, they change category. The resumed request replays
        the partial so the model can continue it, so what was `generated` last attempt
        is `prompt` this one, and the stream's own counter restarts at zero. Without
        this the sweep recomputes occupancy as `base_tokens + generated` against the
        ORIGINAL prompt and undercounts by the whole replayed partial, by more on every
        pause.

        Measured 2026-09-02, the run that first carried thoughts across a pause: one
        chat paused four times replaying 564, 59, 1079 and 507 tokens, the ledger saw
        none of them, and the run went from zero context-exhaustion errors to four with
        38 KV retries. Keeping the work was right; not charging for it was not.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            participant.base_tokens = participant.base_tokens + max(0, int(tokens or 0))
            participant.tokens = max(participant.tokens, participant.base_tokens)

    def set_state(self, gen_id: str, state: str) -> None:
        """Report a safe point. Ends the epoch when the winner stops decoding.

        The winner is fixed until it completes, blocks on a tool, or ends its turn, so
        two chats cannot trade places forever. Every one of those shows up here as a
        state that is not DECODING.
        """
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            participant.state = state
            if self._epoch_winner == gen_id and state != ParticipantState.DECODING:
                self._epoch_winner = None

    def note_resumed(self, gen_id: str) -> None:
        """A preempted generation is decoding again."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            participant.preempt_event.clear()
            participant.state = ParticipantState.DECODING

    def participant(self, gen_id: str) -> Optional[Participant]:
        """The registered participant, or None once it has finished.

        Returned by reference so the caller reads live state; every field the adapter
        touches is either immutable or written under this lock.
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
        """What a waiter watches to tell "busy" from "stuck".

        Only two things can end a wait for room: the cache gives some back, or a holder
        leaves. Both are visible here. Growth is deliberately NOT progress -- other chats
        decoding into the cache is the opposite of room appearing -- so a waiter that
        reset its patience on any change at all would never time out.
        """
        with self._lock:
            return (
                self._committed_locked(),
                frozenset(p.gen_id for p in self._participants.values() if p.holds_kv),
            )

    def _buffer_locked(self) -> int:
        return preemption_buffer_tokens(
            self._budget,
            draft_tokens = self._draft_tokens,
            slots = self._slots,
            batch_tokens = self._batch_tokens,
        )

    def _prune_locked(self) -> None:
        """Drop participants whose lease is finished with the cache.

        Belt and braces beside ``unregister``. A generation ends on many branches
        (normal finish, cancel, disconnect, admission timeout, HTTP error) and a single
        one that forgets to unregister would leave a dead conversation counted against
        the budget for the life of the model load, which would preempt everybody
        forever. Cheap: the registry holds at most ``capacity`` entries.
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
        # Whichever is larger, because they measure different things and both are real.
        # The ledger knows what live generations were admitted on; the resident figure
        # knows what the cache is actually holding, including finished requests whose
        # prompt cache llama.cpp keeps for prefix reuse. Trusting only the ledger is what
        # let four chats be scheduled against a cache an idle slot had already filled.
        if self._resident is None:
            return ledger
        # Not max(ledger, resident): that double counts every chat the resident figure
        # ALREADY includes. Measured 2026-09-03 over 1218 samples of a four-chat run, the
        # ledger overstated the cache in 1212 of them, by up to a whole 16384-cell window,
        # because each chat was carrying an equal-share reservation on top of cells
        # llama-server had already reported. Chats were paused with the cache half empty
        # and two gave up at the resume timeout.
        #
        # Split it instead. A measured chat is inside `resident`, so its ledger entry is
        # a second opinion about the same cells: take the larger of the two totals, which
        # keeps the guard against a reading that lags a prefill in progress. An unmeasured
        # chat has not been prefilled, so `resident` cannot see it and its charge is a
        # genuine reservation that has to be added on top.
        holders = [p for p in self._participants.values() if p.holds_kv]
        measured = sum(p.tokens for p in holders if p.measured)
        pending = sum(p.tokens for p in holders if not p.measured)
        return max(self._resident, measured) + pending

    def _winner_locked(self) -> Optional[Participant]:
        """The one generation that keeps decoding, stable for an epoch.

        Promoted (starved) first, then longest-wins, then arrival order so the choice is
        deterministic rather than dict-ordered.
        """
        held = self._participants.get(self._epoch_winner) if self._epoch_winner else None
        if held is not None and held.state == ParticipantState.DECODING:
            return held
        # The epoch is over (or never started): pick a new winner from those still
        # decoding. A parked or tools-running holder is not a candidate; it is not
        # decoding, so crowning it would pause everyone for nobody's benefit.
        candidates = [
            p for p in self._participants.values() if p.state == ParticipantState.DECODING
        ]
        if not candidates:
            self._epoch_winner = None
            return None
        winner = min(candidates, key = lambda p: (not p.promoted, -p.tokens, p.seq))
        self._epoch_winner = winner.gen_id
        # Its starvation is cured the moment it is crowned. Resetting on resume instead
        # would defeat the rule, since a resumed victim can be preempted again at once.
        winner.consecutive_preemptions = 0
        return winner

    def plan_preemptions(self, *, needed: int = 0) -> List[Participant]:
        """Who must stop so ``needed`` more tokens fit. Empty when nothing must.

        Sets each victim's ``preempt_event`` and marks it PAUSED, so the decision and the
        signal cannot drift apart. The caller aborts only the upstream stream, then calls
        ``lease.preempt()`` once that response is closed.
        """
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return []
            buffer = self._buffer_locked()
            ceiling = max(0, self._budget - buffer)
            total = self._committed_locked()
            want = max(0, int(needed or 0))
            # The two figures the sweep is choosing between, recorded whenever they
            # disagree. `ledger` is prompt ESTIMATES plus counted output;
            # `resident` is llama-server's exact per-slot totals. A run that overran the
            # cache with three slots holding 4237 + 5400 + 7390 = 17027 tokens could have
            # been either the ledger drifting low on code-heavy prompts or the sweep never
            # running during prefill, and nothing logged said which.
            ledger = sum(p.tokens for p in self._participants.values() if p.holds_kv)
            if self._resident is not None and abs(self._resident - ledger) > 256:
                _log.info(
                    "llama preemption ledger-drift: ledger=%s resident=%s ceiling=%s "
                    "want=%s holders=%s",
                    ledger, self._resident, ceiling, want,
                    sum(1 for p in self._participants.values() if p.holds_kv),
                )
            if total + want <= ceiling:
                return []
            # Parked holders first: they hold KV and consume no compute, so their room is
            # the cheapest to take, and this prefix is worth its keep -- dropping it cost
            # the chosen policy 2.89 mean rank against 4.28 across nine simulated load
            # regimes, most of it in completions.
            #
            # Then NEWEST first, which is what vLLM V1 does: it evicts the most recently
            # arrived running request, so the work already done is the work preserved.
            # This used to take the LARGEST decoder, on the reasoning that the fewest
            # victims free the most room. Simulated over nine regimes and 60 seeds each,
            # that ranked 5th of 7 policies (mean 4.25) and worst of all on fairness,
            # because the biggest chat is also the one carrying the most work to throw
            # away and the most tokens to replay when it resumes. Newest-first ranked
            # best overall at 2.89 and best of all on completions.
            #
            # No generation is exempt. A fixed epoch winner used to be, to stop two chats
            # trading places forever; it never measurably reduced thrash, it cost
            # completions in tool-heavy loads (6.57 against 6.77 of eight chats), and in a
            # live run the exempt chat simply grew until it filled the entire window and
            # its turn had to be truncated. Anti-starvation is handled by promotion after
            # repeated preemptions instead, which does not hand anyone the whole cache.
            victims = [p for p in self._participants.values() if p.preemptable]
            # A chat preempted this many times running is promoted above newest-first, so
            # repeatedly losing does not become never finishing. A THRESHOLD rather than a
            # continuous term: ordering by the count directly would let a single earlier
            # preemption outrank arrival order and quietly turn the policy into
            # least-preempted-first, which is not what was benchmarked.
            victims.sort(
                key = lambda p: (
                    p.state != ParticipantState.PARKED_ON_TOOL,
                    p.promoted,
                    -p.seq,
                )
            )
            # Always leave one holder standing. "Pause all but one" is the worst case, and
            # pausing the last one too is pure loss: nothing decodes, the room is handed to
            # a chat that has not started, and the incumbent has to replay everything it
            # had. The wait line already holds newcomers, so this costs them nothing they
            # were not already paying. Without it, the crowned-winner exemption's removal
            # would have let a sweep empty the cache entirely.
            spare = max(0, len(victims) - 1)
            chosen: List[Participant] = []
            for victim in victims[:spare]:
                if total + want <= ceiling:
                    break
                # `total` is the PROJECTION used to decide how many victims are needed.
                # The participant's own state stays KV-holding until on_preempted says
                # the stream really stopped, so the next caller plans against reality.
                total -= victim.tokens
                victim.consecutive_preemptions += 1
                victim.state = ParticipantState.PREEMPTING
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

    Socket teardown is not evidence the cells are free, so a preemptor waits here before
    the room it freed is handed to anyone else.

    A BARRIER, never an attribution. ``llama_stats`` warns the gauge cannot say which
    generation owns a slot, so this only ever answers "have that many finished". False
    means it could not be confirmed within ``timeout_s``, or ``/metrics`` is unavailable
    (no ``--metrics``, a build without the counter, a socket error). The caller proceeds
    anyway: blocking a conversation forever on a gauge that may never answer is worse
    than the overrun this guards, and step 1's wire clamp still bounds it.

    ``scrape`` is injected so this is testable without a server; production passes
    ``lambda: scrape_llama_metrics(base_url)``.
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

    The two halves of this module were designed against each other but neither could
    build this seam: the stream side knows when it is safe to stop and how to resume,
    the controller knows who should. This is the whole of the coupling between them.

    ``await_resume`` is where a policy bug would show up as a hung chat, so it is bounded
    twice over: by the caller's timeout, and by refusing to wait at all once the room is
    already back. Returning False means "give up and finish the turn", which degrades to
    the behaviour that predates preemption rather than parking the conversation.

    ``loop`` is the bridge, and the reason this is a separate class rather than the
    controller implementing the protocol directly. The stream funnels are synchronous and
    run on a worker thread, while ``resume_async`` has to await the queue's slot acquire,
    so the two are joined with ``run_coroutine_threadsafe`` as ``mcp_client`` already
    does. With no loop there is nothing to resume onto and the turn simply finishes.
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
        """The upstream response is closed, so the tokens may go back.

        Order matters and is the rule ``_release_admission`` already states: hand the
        lease back only once nothing can still be decoding against it. The stream side
        guarantees that by calling this from its except branch, after teardown.
        """
        self._resumes = checkpoint.resumes
        _log.info(
            "llama preemption paused: gen_id=%s resumes=%s kept_chars=%s kept=%s charged=%s",
            self._gen_id, checkpoint.resumes, checkpoint.kept_chars(),
            "prose" if checkpoint.has_resume_point()
            else ("thought" if checkpoint.has_reasoning_resume_point() else "nothing"),
            checkpoint.charged_tokens,
        )
        participant = self._controller.participant(self._gen_id)
        if participant is None:
            return
        # Before the state change, so a sweep that runs between the two sees the larger
        # figure rather than the stale one.
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
                # A handback that fails must not take the conversation with it: the
                # tokens are reclaimed when the lease is finally released either way.
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
        # Re-stated, not remembered: a resumed run carries the partial it already
        # generated, so it needs more room than it was preempted holding.
        want = max(0, int(participant.tokens or 0))
        # Two questions the wait could not previously answer, both of which made it wait
        # for room that no eviction could ever produce. A resumed run replays what it
        # generated as prompt, so `want` grows with every pause and can pass the ceiling
        # the wait is measured against.
        if self._controller.cannot_ever_fit(want):
            # Bigger than the cache. Ending the turn here reports `length`, which the
            # continuation path resumes against a fresh window; waiting reports nothing
            # and hangs until the client disconnects.
            _log.info(
                "llama preemption too-large: gen_id=%s want=%s (exceeds the cache; "
                "finishing the turn)", self._gen_id, want,
            )
            return False
        if self._controller.outgrew_the_shared_ceiling(want):
            # Fits alone but beside nobody. `room_for` grants it the cache once everyone
            # else is out, so this only has to say why the wait may be long.
            _log.info(
                "llama preemption needs-the-cache: gen_id=%s want=%s (past the shared "
                "ceiling; waiting for the cache to itself)", self._gen_id, want,
            )
        _log.info("llama preemption awaiting-room: gen_id=%s want=%s", self._gen_id, want)
        # Wait for the cache to actually have room before taking the lease back. Without
        # this the queue hands a resume out on its own optimistic accounting and the next
        # watermark sweep evicts the same chat again, which is thrash, not scheduling.
        # The clock measures STALL, not elapsed time. A flat wall-clock deadline cannot
        # tell a system that is working from one that is stuck, and both happen here: a
        # chat waiting behind a 10k-token answer waits minutes through healthy progress,
        # while a genuine deadlock shows nothing moving at all. Measured 2026-09-03,
        # 90 seconds of wall clock killed two chats outright while the cache was steadily
        # turning over.
        #
        # So the deadline resets whenever the cache gives room back or a holder leaves,
        # and expires only after `timeout` seconds in which neither happened. That still
        # ends the failure this bound was added for -- three paused chats and a 33 minute
        # hang with NOTHING decoding, which registers as a stall immediately -- while a
        # chat that is merely queued behind live work keeps its place.
        #
        # `hard_deadline` is the backstop for the case the stall detector cannot see: a
        # cache that keeps churning while this particular chat is never quite served.
        started = time.monotonic()
        deadline = started + timeout
        hard_deadline = started + timeout * MAX_RESUME_WAIT_MULTIPLE
        last = self._controller.progress_signature()
        # Fresh reading before the first question, not just the cached one: this is the
        # grant that lets a chat back in carrying its whole replayed partial.
        self._controller.refresh_residency()
        # try_grant_resume, not room_for: the room has to be BOOKED at the instant it is
        # found, or two chats waiting at once both find the same space and both take it.
        while not self._controller.try_grant_resume(self._gen_id, want):
            self._controller.refresh_residency()
            now = time.monotonic()
            current = self._controller.progress_signature()
            if current != last:
                committed, holders = current
                was_committed, was_holders = last
                # Room appeared, or somebody finished and is about to give theirs back.
                if committed < was_committed or not (was_holders <= holders):
                    deadline = now + timeout
                last = current
            if now >= deadline:
                _log.info(
                    "llama preemption gave-up: gen_id=%s want=%s (no progress for %ss)",
                    self._gen_id, want, timeout,
                )
                return False
            if now >= hard_deadline:
                _log.info(
                    "llama preemption gave-up: gen_id=%s want=%s (still unserved after "
                    "%ss of a moving cache)", self._gen_id, want, round(now - started, 1),
                )
                return False
            time.sleep(0.1)
        try:
            future = asyncio.run_coroutine_threadsafe(
                lease.resume_async(want, timeout_s = timeout), self._loop
            )
            # The future's own timeout is a backstop for a loop that never runs the
            # coroutine at all; resume_async is already bounded by timeout_s.
            got = bool(future.result(timeout = timeout + 5.0))
            if not got:
                # The grant above booked the room. Nothing is going to use it, so hand it
                # back rather than leave the ledger holding space for a chat that stopped.
                self._controller.note_resume_failed(self._gen_id)
            _log.info(
                "llama preemption %s: gen_id=%s want=%s",
                "resumed" if got else "gave-up", self._gen_id, want,
            )
            return got
        except Exception as exc:
            # Includes the future timing out. Whatever the cause, the honest answer is
            # that the room did not come back, and the caller finishes the turn.
            _log.warning(
                "llama preemption resume-failed: gen_id=%s want=%s error=%s",
                self._gen_id, want, exc,
            )
            return False

    def on_resumed(self) -> None:
        self._controller.note_resumed(self._gen_id)


def _slot_decoded(slot: dict) -> int:
    """Tokens this slot has generated so far, from `/slots`.

    llama-server nests it under ``next_token``, which is a one-element LIST in the builds
    seen here and a bare object in others. Both shapes are read, and anything else
    answers zero rather than raising: an occupancy read that throws would take the whole
    watermark sweep down with it.
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

    The term everything else was missing. llama.cpp keeps a slot's prompt cache after its
    request finishes, for prefix reuse, and that residue belongs to no live generation:
    the admission ledger cannot see it, and /metrics does not report it. Measured
    2026-09-01, the moment it mattered:

        purging slot 1 with 16383 tokens

    An idle slot holding the ENTIRE 16384-cell cache while four chats were being
    scheduled against a ledger that believed the cache was nearly empty. That is why the
    watermark kept firing too late and llama-server kept dropping into its
    shrinking-batch retry, which is where upstream #24840 throws.

    ``fetch`` returns the parsed ``GET /slots`` array, or None when unavailable (older
    build, endpoint disabled). None here means "cannot say", never "empty": guessing zero
    would restore exactly the blindness this exists to remove.
    """
    slots = fetch()
    if not slots:
        return None
    resident = 0
    idle_tokens = 0
    idle = []
    for slot in slots:
        try:
            tokens = int(
                slot.get("n_prompt_tokens_cache")
                or slot.get("n_prompt_tokens")
                or 0
            )
        except (TypeError, ValueError):
            tokens = 0
        # Plus what it has GENERATED, which is the term this was missing and the reason
        # every watermark diagnostic came back innocent while chats died. `/slots`
        # reports the prompt; the tokens decoded since occupy cells too, and on a chat
        # writing a long answer they are most of it. Sampled live 2026-09-03 at four
        # second intervals while one slot decoded:
        #
        #     reported= 6880 decoded= 571 true= 7451
        #     reported=12632 decoded=6323 true=18955
        #
        # 18955 cells in a 16384 cache, reported as 12632 against a ceiling of 14312. The
        # watermark could not fire because the figure it watches never moved past it, and
        # the buffer was being asked to cover a 6000 token undercount.
        #
        # Only while processing. A finished slot's prompt cache already holds the whole
        # sequence it produced, so `n_prompt_tokens_cache` covers it and adding a stale
        # `n_decoded` on top would count the generated half twice.
        if slot.get("is_processing"):
            tokens += _slot_decoded(slot)
        # Idle slots count too. Excluding them was tried on 2026-09-04, on the reasoning
        # that llama.cpp recycles an idle slot's cache by itself so charging for it evicts
        # live chats to reclaim room that was already free. The reasoning is wrong in the
        # one way that matters: `try_clear_idle_slots` is called FROM the KV-full retry,
        # so that recycling happens only after a decode has ALREADY failed, which is the
        # path #24840 throws on and the exact path this module exists to stay off. Cells
        # that are only freed by crashing first are occupied as far as we are concerned.
        #
        # Measured, same harness and config: counting them gave 3 clean runs of 4;
        # excluding them gave 0 clean of 2, with sub-batch errors in both.
        resident += max(0, tokens)
        if not slot.get("is_processing") and tokens > 0:
            idle_tokens += max(0, tokens)
            idle.append((slot.get("id"), max(0, tokens)))
    # Largest first: the fewest erases free the most.
    idle.sort(key = lambda pair: -pair[1])
    return {
        "resident": resident,
        # Reclaimable, not occupied: reported separately so the caller can free it
        # BEFORE pausing anybody, which is what reclaim_idle_slots is for.
        "idle_tokens": idle_tokens,
        "idle": idle,
        "slots": len(slots),
    }


def reclaim_idle_slots(
    occupancy: Optional[dict],
    erase: Callable[[int], int],
    *,
    needed: int,
) -> int:
    """Free dead residue before asking a live chat to stop.

    Strictly better than preempting: an idle slot's cache belongs to a finished request,
    so erasing it costs a future prefix-cache hit and nothing else, while preempting
    costs a running conversation its progress. llama.cpp does this itself on the KV-full
    retry (``try_clear_idle_slots``), but only once the decode has ALREADY failed, which
    is the path that trips the speculative sub-batch bug. Doing it earlier is the point.

    Returns tokens freed.
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
            # An erase that fails leaves the residue in place; the caller falls back to
            # preempting a live generation, which is the outcome without this at all.
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
            # base_url takes a fresh ephemeral port on every model load, so each load
            # registers a new key. Drop controllers with nothing in flight, exactly as
            # get_llama_admission_queue drops idle queues.
            # is_idle takes each controller's own lock, so the list is built first.
            stale = [k for k, c in _CONTROLLERS.items() if k != key and c.is_idle()]
            for k in stale:
                del _CONTROLLERS[k]
        return controller


def reset_preemption_controllers() -> None:
    with _CONTROLLERS_LOCK:
        _CONTROLLERS.clear()
