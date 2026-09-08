# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who pauses when the shared KV cache fills. llama-server admits on `prompt_tokens <
slot.n_ctx` alone, so chats that each fit collide and it errors EVERY processing slot."""

from __future__ import annotations

import asyncio
import os
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from core.inference.llama_admission import (
    LlamaAdmissionLease,
    _bool_env,
    llama_admission_config_from_env,
)
from core.inference.llama_exact import EXACT_STATE_OFF, EXACT_STATES


_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}

from loggers import get_logger

_log = get_logger(__name__)


PREEMPT_ENV = "UNSLOTH_LLAMA_ADMISSION_PREEMPT"
DEFAULT_PREEMPT_ENABLED = True

# `server` is llama-server's own --preempt-ram parking (unslothai/llama.cpp#184). Exclusive:
# with both on, the lower Studio watermark always fires first.
PREEMPT_MODE_ENV = "UNSLOTH_LLAMA_PREEMPT_MODE"
PREEMPT_MODE_AUTO = "auto"
PREEMPT_MODE_STUDIO = "studio"
PREEMPT_MODE_SERVER = "server"
_PREEMPT_MODES = (PREEMPT_MODE_AUTO, PREEMPT_MODE_STUDIO, PREEMPT_MODE_SERVER)


def preempt_mode_setting() -> str:
    raw = (os.environ.get(PREEMPT_MODE_ENV) or "").strip().lower()
    return raw if raw in _PREEMPT_MODES else PREEMPT_MODE_AUTO


def resolve_preempt_mode(server_preempts: bool) -> str:
    setting = preempt_mode_setting()
    if setting == PREEMPT_MODE_STUDIO:
        return PREEMPT_MODE_STUDIO
    return PREEMPT_MODE_SERVER if server_preempts else PREEMPT_MODE_STUDIO


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Reaction headroom for ONE decoding slot: what it generates between the sweep and the stop.
DEFAULT_PREEMPT_BUFFER_PER_SLOT = 192
DEFAULT_PREEMPT_BUFFER_MIN_TOKENS = 256

# Escape hatch restoring the permanent batch term, the dynamic one needing every prefill announced.
STATIC_BATCH_ENV = "UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH"
DEFAULT_PREEMPT_STATIC_BATCH = False

# Drop the batch term for a prefill `_committed_locked` already booked, the chunk being a
# SUBSET of those cells. A measured holder still needs it: other chats can mask its growth.
CHARGED_PREFILL_ENV = "UNSLOTH_LLAMA_PREEMPT_BATCH_ONLY_UNCHARGED"
DEFAULT_PREEMPT_BATCH_ONLY_UNCHARGED = False

# Backstop for an announcement that no first token, pause or departure ever clears. The
# stream's first-token wait (`_DEFAULT_FIRST_TOKEN_TIMEOUT_S`): shorter dropped the reserve
# under a prompt still going in.
PENDING_PREFILL_TTL_S = 1200.0

# Far above _MAX_LENGTH_CONTINUATIONS: that caps answer quality, this caps churn.
DEFAULT_MAX_PREEMPT_RESUMES = 32

# A chat preempted this many times in a row outranks longest-wins for the next epoch.
PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS = 3

# What a chat running ALONE must leave clear. Without it a chat that outgrew the shared ceiling
# can never be admitted OR resumed.
SOLO_MARGIN_DIVISOR = 200

# A server without --metrics never answers the barrier, so it gives up and admits anyway.
DEFAULT_RECLAIM_BARRIER_TIMEOUT_S = 10.0
DEFAULT_RECLAIM_BARRIER_POLL_S = 0.05

# An unbounded await_resume() turned three paused chats into a 33-minute hang with nothing
# decoding. On expiry the turn finishes with what it has.
DEFAULT_RESUME_WAIT_TIMEOUT_S = 90.0

# 80 x 90s = 2 hours, raised from 20: 30 minutes is shorter than one legitimate answer at the
# 2.3 tok/s measured on the 35B. Finite only to catch a cache the stall clock cannot see.
MAX_RESUME_WAIT_MULTIPLE = 80


class LlamaStreamPreempted(Exception):
    """Aborted to free KV, not abandoned: expected to be caught and resumed."""


class PreemptSignal:
    """A pause request tool execution can hold off: `is_set()` is False inside an unsafe window."""

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

    def set(self, reason: str = "kv_pressure") -> None:
        self.request(reason)

    def clear(self) -> None:
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
        return PreemptSignal._Window(self)

    def _enter_unsafe(self) -> None:
        with self._lock:
            self._depth += 1
            # A request that arrived before the window opened is hidden too: acting on it now
            # is the mid-execution abort the window exists to prevent.
            self._event.clear()

    def _exit_unsafe(self) -> None:
        with self._lock:
            if self._depth > 0:
                self._depth -= 1
            if self._depth == 0 and self._pending:
                self._event.set()


@dataclass
class StreamCheckpoint:
    """What a paused attempt had produced. Not read off the thread's trailing assistant row: an
    aborted run never writes one, so that row lags by a whole attempt."""

    visible_text: str = ""
    reasoning_text: str = ""
    # Carried rather than re-derived: the `context_truncated` SSE event is not idempotent.
    pending_truncations: list = field(default_factory = list)
    charged_tokens: int = 0
    resumes: int = 0
    reason: Optional[str] = None

    def has_resume_point(self) -> bool:
        """Empty means the pause landed before the first token, so the request is re-issued whole."""
        return bool(self.visible_text.strip())

    def has_reasoning_resume_point(self) -> bool:
        """A thought resumes as `reasoning_content`; sent as `content` it renders as the answer."""
        return not self.visible_text.strip() and bool(self.reasoning_text.strip())

    def kept_chars(self) -> int:
        """Prose or thought. `len(visible_text)` alone read zero on every pause of a reasoning
        model and made a livelock look like an orderly pause."""
        if self.has_resume_point():
            return len(self.visible_text)
        return len(self.reasoning_text) if self.has_reasoning_resume_point() else 0


@runtime_checkable
class PreemptionPolicy(Protocol):
    """``await_resume`` returning False means "stop waiting and finish the turn", so a policy
    that dies degrades instead of hanging the chat."""

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
    """A policy handed to the stream before the real one can exist: the tool-loop generator is
    BUILT before admission returns, and nothing can pause before it is iterated."""

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
        # Stop travels through, or every routed chat pauses without it: the caller's keyword
        # raised TypeError here and its fallback retried the wait with no Stop at all.
        if cancel_event is None:
            return bool(self._inner.await_resume(timeout))
        try:
            return bool(self._inner.await_resume(timeout, cancel_event = cancel_event))
        except TypeError:
            # An inner policy written against the older protocol.
            return bool(self._inner.await_resume(timeout))

    def on_resumed(self) -> None:
        if self._inner is not None:
            self._inner.on_resumed()

    def on_server_parked(self) -> None:
        hook = getattr(self._inner, "on_server_parked", None)
        if hook is not None:
            hook()

    def on_server_resumed(self) -> None:
        hook = getattr(self._inner, "on_server_resumed", None)
        if hook is not None:
            hook()

    def on_declined(self) -> None:
        if self._inner is not None:
            self._inner.on_declined()


class NullPreemptionPolicy:
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
    """Where a generation is. Strings, not an enum, so they cross the SSE boundary untranslated."""

    QUEUED = "queued"
    DECODING = "decoding"
    # Holds KV, consumes no compute: the cheapest room to reclaim, so these are taken first.
    PARKED_ON_TOOL = "parked_on_tool"
    # Never preempted: nothing is decoding, and it sits inside the tool loop's unsafe window.
    TOOLS_RUNNING = "tools_running"
    # Holds KV. Counted free the instant a victim was chosen, four chats committed 16384 of 16384.
    PREEMPTING = "preempting"
    # Raw passthrough and Responses: no conversation to resume from, so aborting one is a
    # CANCEL and it is never a victim. Registered all the same, or the watermark fires late.
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
# PREEMPTING is absent on purpose: asking twice double-counts the room its pause will free.

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


def preemption_eligible() -> bool:
    """The switches that have to be on before anything here may choose a victim.

    Every charge the controller plans against is an admission lease, so with admission control or
    the KV budget off the ledger is a column of zeroes and choosing on it pauses live streams for an
    accounting that is not running. Read here so ``active`` and ``plan_preemptions`` cannot drift.
    """
    if not preemption_enabled():
        return False
    config = llama_admission_config_from_env()
    return bool(config.enabled and config.kv_budget)


def preemption_buffer_tokens(
    budget: int,
    *,
    draft_tokens: int = 0,
    slots: int = 1,
    batch_tokens: int = 0,
    pending_prefill: int = 0,
) -> int:
    """Tokens held clear of ``budget``. Zero for an unknown budget, which disables it. Never the
    whole cache, a buffer that swallows the budget preempting everyone forever. Drafts are charged
    on top, unreserved ones putting the cache on the shrinking-batch retry where
    ggml-org/llama.cpp#24840 throws. The batch term is charged only while a prefill is pending."""
    if budget <= 0:
        return 0
    # PER SLOT, not per cache: reaction headroom scales with how many chats decode at once and
    # not with cache size. As 15% makespan was 26890 steps against 239 at this size.
    per_slot = _int_env("UNSLOTH_LLAMA_PREEMPT_BUFFER_PER_SLOT", DEFAULT_PREEMPT_BUFFER_PER_SLOT)
    slot_count = max(1, int(slots or 1))
    reserve = max(DEFAULT_PREEMPT_BUFFER_MIN_TOKENS, per_slot * slot_count)
    # Room for the batch llama.cpp is processing: the cache fails when the next BATCH does not fit,
    # not when it is full. ONLY WHILE PREFILLING; permanent, it held a quarter of an 8192 cache.
    n_batch = max(0, int(batch_tokens or 0))
    if _bool_env(STATIC_BATCH_ENV, DEFAULT_PREEMPT_STATIC_BATCH):
        batch_reserve = n_batch
    else:
        batch_reserve = min(n_batch, max(0, int(pending_prefill or 0)))
    reserve = max(reserve, batch_reserve)
    # Drafts are additional: cells the drafter puts in before acceptance, unseen by admission.
    reserve += max(0, int(draft_tokens or 0)) * slot_count
    # Never the whole cache: a large draft window on a small -c degrades to a tight buffer.
    return min(reserve, max(1, budget // 2))


@dataclass(**_SLOTS)
class Participant:
    gen_id: str
    seq: int
    lease: Optional[LlamaAdmissionLease] = None
    tokens: int = 0
    # What admission charged. Live growth is ADDED to this, so a report of "n tokens generated"
    # cannot silently drop the prompt already resident.
    base_tokens: int = 0
    # True once this generation's prompt is known to be IN the cache. Until then its charge is
    # a reservation the resident figure cannot see; afterwards adding it double-counts.
    measured: bool = False
    state: str = ParticipantState.DECODING
    consecutive_preemptions: int = 0
    # The caller aborts ONLY the upstream stream on this and keeps its generator, ledgers and
    # SSE response alive, which a shared cancel event could not express.
    preempt_event: PreemptSignal = field(default_factory = PreemptSignal)
    preempt_chosen_at: float = 0.0
    # True once an idle-slot reclaim erased this holder's cells while it was parked. Counting
    # them anyway kept two waiting chats out of an EMPTY cache for three minutes.
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
    # Last count `observe` was given, so cumulative reports become a DELTA. Falls back to zero
    # rather than going negative when a resumed attempt restarts llama-server's counter.
    generated_seen: int = 0

    def prefill_pending(self, now: float) -> int:
        """Zero for a holder whose cells are gone or which is not in the cache: a paused chat
        submits nothing until the resume grant, which is where it announces again."""
        if self.pending_prefill <= 0 or not self.holds_kv:
            return 0
        if now - self.pending_prefill_at > PENDING_PREFILL_TTL_S:
            return 0
        return self.pending_prefill

    def announce_prefill(self, tokens: int) -> None:
        self.pending_prefill = max(0, int(tokens or 0))
        self.pending_prefill_at = time.monotonic()

    def prefill_done(self) -> None:
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
    # The buffer carries a batch term exactly while this is non-zero, so a log line that reports
    # one without the other cannot be read back.
    prefilling: int = 0
    mode: str = PREEMPT_MODE_STUDIO
    # Reported, never acted on here. See core.inference.llama_exact.
    exact: str = EXACT_STATE_OFF
    # "Is anybody else in the cache" answered by the ledger alone: `committed` folds in the last
    # residency reading, which can still count a chat that has just unregistered.
    holders: int = 0


class PreemptionController:
    """Victim choice and the epoch, for one llama-server backend. Not a scheduler: callers ask
    whether there is room and are told who must stop."""

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
        "_server_mode",
        "_exact",
    )

    def __init__(self, key: str):
        self.key = key
        self._lock = threading.Lock()
        self._participants: Dict[str, Participant] = {}
        self._seq = 0
        self._epoch_winner: Optional[str] = None
        self._drift_logged_at = 0.0
        self._budget = 0
        # Preemption reclaims only where an idle slot's cells can be purged, which upstream gates on
        # kv_unified ALONE. NOT idle_slot_clearing_active: that sweep's tasks are still alive.
        self._kv_unified = False
        # Speculative drafts occupy cells no request is charged for.
        self._draft_tokens = 0
        # --batch-size llama-server was launched with, so the buffer can cover one chunk.
        self._batch_tokens = 0
        # Of `_resident`, how much is idle slots' cache: real, but erasable on demand.
        self._reclaimable = 0
        self._slots = 1
        # True cells resident from the last GET /slots, or None when it could not be read.
        # Includes the residue of FINISHED requests, which the ledger cannot see.
        self._resident: Optional[int] = None
        # Set by the route to a callable that re-reads GET /slots. Optional: everything works
        # from the ledger alone, less precisely.
        self._residency_probe: Optional[Callable[[], None]] = None
        # Every token this controller has been told about, never decreasing: the one figure that
        # moves whenever ANYBODY decodes, which `committed` does not, being a maximum.
        self._progress_tokens = 0
        # True when llama-server parks slots itself. The ledger keeps counting, but no sweep
        # here ever chooses a victim.
        self._server_mode = False
        # Reported, never acted on: see PreemptionSnapshot.exact.
        self._exact = EXACT_STATE_OFF

    def configure(
        self,
        *,
        budget: Optional[int] = None,
        kv_unified: Optional[bool] = None,
        draft_tokens: Optional[int] = None,
        slots: Optional[int] = None,
        batch_tokens: Optional[int] = None,
        server_mode: Optional[bool] = None,
        exact: Optional[str] = None,
    ) -> None:
        """Re-read the cache this backend allocated. A reload can relaunch at a different -c."""
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
            if server_mode is not None:
                self._server_mode = bool(server_mode)
            if exact is not None:
                self._exact = str(exact) if str(exact) in EXACT_STATES else EXACT_STATE_OFF

    @property
    def server_mode(self) -> bool:
        with self._lock:
            return self._server_mode

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._kv_unified) and self._budget > 0 and preemption_eligible()

    def register(
        self,
        gen_id: str,
        *,
        lease: Optional[LlamaAdmissionLease] = None,
        tokens: int = 0,
        state: str = ParticipantState.DECODING,
        signal: Optional[PreemptSignal] = None,
    ) -> Participant:
        """``signal`` MUST be the object the stream polls. Without it a Participant makes its
        own and setting it reaches nobody: four chats armed, two selected, neither paused."""
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
            # Announced HERE, so the sweep right afterwards already plans against the raised buffer.
            participant.announce_prefill(participant.tokens)
            self._participants[gen_id] = participant
            return participant

    def unregister(self, gen_id: str) -> None:
        with self._lock:
            self._participants.pop(gen_id, None)
            if self._epoch_winner == gen_id:
                self._epoch_winner = None

    def _solo_ceiling_locked(self) -> int:
        """The cache less what a lone chat still needs clear. Being alone removes the reaction
        headroom but not the need to fit the next batch: a lone chat took 16297 cells of 16384."""
        margin = max(
            self._draft_tokens + max(64, self._budget // SOLO_MARGIN_DIVISOR),
            self._batch_tokens + self._draft_tokens,
        )
        return max(1, self._budget - margin)

    def outgrew_the_shared_ceiling(self, want: int) -> bool:
        """Whether `want` can never fit beside anyone. It must not wait: no combination of
        preemptions will admit it, so waiting is a deadlock. It runs alone."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return False
            # Including this chat's own prefill: running means submitting its prompt.
            pending = self._pending_prefill_locked() + max(0, int(want or 0))
            ceiling = max(0, self._budget - self._buffer_locked(pending = pending))
            return int(want or 0) > ceiling

    def cannot_ever_fit(self, want: int) -> bool:
        """Whether `want` exceeds the cache itself. The turn ends with what it has and the
        continuation path resumes; parking it forever is the hang."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return False
            return int(want or 0) > self._solo_ceiling_locked()

    def room_for(self, gen_id: str, want: int) -> bool:
        """Whether a paused generation may start again yet, against the LIVE total. On the admission
        queue's accounting a chat was let back in over the watermark and evicted at once."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_enabled():
                return True
            return self._room_for_locked(gen_id, want)

    def try_grant_resume(self, gen_id: str, want: int) -> bool:
        """Decide there is room AND take it, without letting go of the lock between: `room_for` only
        answers, and two paused chats asking at once both get yes, PAUSED not being in `_HOLDS_KV`.
        Roll back with `note_resume_failed`. The grant marks the participant RESUMING, not DECODING:
        the lease's slot is reacquired after this, and a sweep in between would count it as freed."""
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
                # Announced under the same lock that booked the room.
                participant.announce_prefill(need)
            return True

    def note_resume_failed(self, gen_id: str) -> None:
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None and participant.state in (
                ParticipantState.RESUMING,
                ParticipantState.DECODING,
            ):
                participant.state = ParticipantState.PAUSED
                participant.prefill_done()

    def _room_for_locked(self, gen_id: str, want: int) -> bool:
        # `want` REPLACES this generation's own announcement: saying yes here is what causes
        # the prefill, and a chat that announced must not pay twice.
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
            # Minus the idle residue, erased for the waiter rather than waited out. Counting it
            # deadlocked scheduling: the ledger read 0 while the slots read 21304 of 14312.
            occupied = self._resident - self._reclaimable
            others = max(others, occupied - (mine.tokens if mine else 0))
        need = max(0, int(want or 0))
        others = max(0, others)
        if others + need <= ceiling:
            return True
        # Outgrew the shared ceiling: it must still run alone, or it waits for room no eviction
        # can ever make.
        if need > ceiling and others == 0:
            return need <= self._solo_ceiling_locked()
        return False

    def observe(self, gen_id: str, generated: int) -> List["Participant"]:
        """Live growth during generation, and THE watermark sweep. It cannot live only between
        rounds: one round can generate thousands of tokens. Returns whoever must stop, signalled."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None:
                reported = max(0, int(generated or 0))
                # The only progress signal a waiter can trust. A fall means a resumed attempt
                # restarted its count, not that tokens went back.
                previous = participant.generated_seen
                participant.generated_seen = reported
                self._progress_tokens += (reported - previous) if reported >= previous else reported
                participant.tokens = participant.base_tokens + reported
                participant.measured = True
                participant.cells_reclaimed = False
                if reported > 0:
                    # Guarded on `generated`: the round-boundary sweep calls this with zero
                    # right after `note_tokens` announced the growth.
                    participant.prefill_done()
                # A generated token is proof of decoding, whatever the route last reported: a round
                # streaming only tool-call deltas left a chat unpreemptable past the ceiling.
                if participant.state in _DECODES_WHEN_TOKENS_ARRIVE:
                    participant.state = ParticipantState.DECODING
        # Outside the lock: plan_preemptions takes it, and it is not reentrant.
        return self.plan_preemptions(needed = 0)

    def set_residency_probe(self, probe: Optional[Callable[[], None]]) -> None:
        """Register a way to re-read the cache on demand. The ledger adds up prompt ESTIMATES; a
        reading a second old can be a thousand tokens stale when a resume grant uses it."""
        self._residency_probe = probe

    def refresh_residency(self) -> None:
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
        """The cache as llama-server actually sees it, None when the read failed. ``reclaimable``
        is the part held by IDLE slots: it counts toward the watermark but is erased on demand."""
        with self._lock:
            if resident is None:
                self._resident = None
                self._reclaimable = 0
                return
            # Clamped to the cache. A per-slot sum is an upper bound, not a measurement: 21304
            # was reported for a 16384 cache. Unclamped, every resume is refused forever.
            ceiling = self._budget if self._budget > 0 else int(resident)
            self._resident = max(0, min(int(resident), ceiling))
            self._reclaimable = max(0, min(int(reclaimable or 0), self._resident))

    def note_tokens(self, gen_id: str, tokens: int) -> None:
        """What a round boundary says this run now holds, and the third place a prefill is
        announced. Only the DIFFERENCE from the previous figure is submitted, unless a reclaim
        erased the cells and the whole prompt goes in again."""
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
                    # Same counter as `observe`, for the same reason: a waiter must see it.
                    self._progress_tokens += growth
                # Re-baselined: a round boundary restates the whole conversation.
                participant.base_tokens = participant.tokens
                participant.measured = True
                participant.cells_reclaimed = False

    # PAUSED, PREEMPTING, QUEUED and the raw surfaces are owned by other transitions.
    _LIVE_STATES = frozenset(
        {
            ParticipantState.DECODING,
            ParticipantState.PARKED_ON_TOOL,
            ParticipantState.TOOLS_RUNNING,
        }
    )

    def note_state(self, gen_id: str, state: str) -> bool:
        """Where a live tool-loop chat is. PARKED_ON_TOOL and TOOLS_RUNNING were never set by
        anyone, so a chat on an approval prompt stayed DECODING and counted erased cells."""
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
                # Back at the model: the prompt is prefilled in again, so the cells are real.
                if participant.cells_reclaimed:
                    # And the WHOLE prompt: a reclaim erased every cell, so no prefix is left to hit.
                    participant.announce_prefill(participant.tokens)
                participant.cells_reclaimed = False
            if self._epoch_winner == gen_id and state != ParticipantState.DECODING:
                # A winner that stopped decoding is not winning anything.
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
                # Whatever it was about to prefill went with the cells. It re-announces in
                # `note_state` when it decodes again.
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
        """Tokens a paused attempt decoded that the NEXT attempt sends back as prompt: they do not
        leave the cache, they change category, and the stream's counter restarts at zero."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            participant.base_tokens = participant.base_tokens + max(0, int(tokens or 0))
            participant.tokens = max(participant.tokens, participant.base_tokens)

    def set_state(self, gen_id: str, state: str) -> None:
        """Report a safe point. Ends the epoch when the winner stops decoding, so two chats
        cannot trade places forever."""
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
                # Nothing is submitted until it asks again, and asking is where it re-announces.
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
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None:
                return
            participant.preempt_event.clear()
            participant.state = ParticipantState.DECODING

    def note_declined(self, gen_id: str) -> None:
        """A chosen victim will not pause after all, and goes on decoding: nothing else takes the
        decision back, `observe` leaving PREEMPTING alone. Only from PREEMPTING, or this invents a holder."""
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is None or participant.state != ParticipantState.PREEMPTING:
                return
            participant.state = ParticipantState.DECODING
            participant.preempt_chosen_at = 0.0
            participant.consecutive_preemptions = max(0, participant.consecutive_preemptions - 1)
            # Last, and inside the lock: a signal still set aborts the very stream this is
            # letting run, and clearing it first leaves a window for a sweep to re-arm it.
            participant.preempt_event.clear()

    def participant(self, gen_id: str) -> Optional[Participant]:
        """The registered participant, or None once it has finished. By reference."""
        with self._lock:
            return self._participants.get(gen_id)

    def is_idle(self) -> bool:
        with self._lock:
            return not self._participants

    def committed_tokens(self) -> int:
        with self._lock:
            return self._committed_locked()

    def progress_signature(self) -> tuple:
        """What a waiter watches to tell "busy" from "stuck": `committed`, the holders,
        `_progress_tokens`, and how many holders are inside a tool call. The token term is the one
        that was missing: `committed` is a maximum a decoding chat can leave untouched, and one turn
        was abandoned for "no progress" while three others decoded to completion."""
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
        """Prompt tokens announced but not yet in the cache, across every holder. `exclude` drops
        one participant so a caller asking "room for ME" can substitute its own figure."""
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
        """Tokens held clear right now. `pending` overrides the live announcement sum: answering
        at the idle buffer admits a chat into room that stops existing in the same breath."""
        if self._server_mode:
            # The server reserves its own drafts and an 8 cell margin per sequence
            # (preempt_kv_reserve) and parks a slot the moment the next batch does not fit.
            return 0
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
        """Drop participants whose lease is finished with the cache. Belt and braces beside
        ``unregister``: one branch that forgot would preempt everybody for the model load's life."""
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
        # Whichever is larger: the ledger knows what live generations were admitted on, the
        # resident figure knows what the cache holds, finished requests' prefix cache included.
        if self._resident is None:
            return ledger
        # Not max(ledger, resident): that double counts every chat the resident figure ALREADY
        # includes, the ledger having overstated the cache in 1212 of 1218 samples.
        holders = [p for p in self._participants.values() if p.holds_kv]
        measured = sum(p.tokens for p in holders if p.measured)
        pending = sum(p.tokens for p in holders if not p.measured)
        return max(self._resident, measured) + pending

    def _winner_locked(self) -> Optional[Participant]:
        """The one generation that keeps decoding, stable for an epoch. Promoted (starved) first,
        then longest-wins, then arrival order so the choice is deterministic."""
        held = self._participants.get(self._epoch_winner) if self._epoch_winner else None
        if held is not None and held.state == ParticipantState.DECODING:
            return held
        # A parked or tools-running holder is no candidate: crowning it would pause everyone for nobody.
        candidates = [
            p for p in self._participants.values() if p.state == ParticipantState.DECODING
        ]
        if not candidates:
            self._epoch_winner = None
            return None
        winner = min(candidates, key = lambda p: (not p.promoted, -p.tokens, p.seq))
        self._epoch_winner = winner.gen_id
        # Its starvation is cured the moment it is crowned. Resetting on resume defeats the
        # rule, a resumed victim being preemptable again at once.
        winner.consecutive_preemptions = 0
        return winner

    def plan_preemptions(self, *, needed: int = 0) -> List[Participant]:
        """Who must stop so ``needed`` more tokens fit. Sets each victim's ``preempt_event`` and
        marks it PAUSED, so decision and signal cannot drift."""
        with self._lock:
            if not self._kv_unified or self._budget <= 0 or not preemption_eligible():
                return []
            if self._server_mode:
                # llama-server parks and restores slots itself. Choosing a victim here would
                # abort a stream it was about to park in place.
                return []
            buffer = self._buffer_locked()
            ceiling = max(0, self._budget - buffer)
            total = self._committed_locked()
            want = max(0, int(needed or 0))
            # The two figures the sweep chooses between, recorded whenever they disagree: a run
            # that overran could have been the ledger drifting low or the sweep never running.
            ledger = sum(p.tokens for p in self._participants.values() if p.holds_kv)
            # Rate limited: a 569s four-chat run emitted 1947 of these, one every three tenths
            # of a second, drowning the events around it.
            _now = time.monotonic()
            if (
                self._resident is not None
                and abs(self._resident - ledger) > 256
                and _now - self._drift_logged_at >= 5.0
            ):
                self._drift_logged_at = _now
                # `ledger` is a raw sum and decides NOTHING. `committed` is what the watermark
                # compares against the ceiling, its `pending` half reserved for text not yet written.
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
            # Parked holders first: they hold KV and consume no compute (2.89 mean rank against 4.28
            # across nine simulated regimes). Then NEWEST first, as vLLM V1 does.
            #
            # No generation is exempt: a fixed epoch winner never measurably reduced thrash and grew
            # until it filled the window, and an arrival that generated nothing costs nothing.
            #
            # `holds_kv` as well as `preemptable`: a participant whose cells an idle reclaim
            # erased keeps `preemptable` True while `holds_kv` is False, so stale `tokens` get subtracted.
            victims = [p for p in self._participants.values() if p.preemptable and p.holds_kv]
            # Promoted above newest-first, so repeatedly losing does not become never finishing.
            # A THRESHOLD, not a continuous term, which is least-preempted-first by another name.
            victims.sort(
                key = lambda p: (
                    p.state != ParticipantState.PARKED_ON_TOOL,
                    p.promoted,
                    -p.seq,
                )
            )
            # Always leave one holder standing: pausing the last one is pure loss. One HOLDER, not
            # one preemptable victim; a holder already PREEMPTING does not count as standing.
            standing = any(
                p.holds_kv and not p.preemptable and p.state != ParticipantState.PREEMPTING
                for p in self._participants.values()
            )
            spare = len(victims) if standing else max(0, len(victims) - 1)
            chosen: List[Participant] = []
            for victim in victims[:spare]:
                if total + want <= ceiling:
                    break
                # `total` is the PROJECTION used to decide how many victims are needed. The
                # participant's own state stays KV-holding until on_preempted says so.
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
                mode = PREEMPT_MODE_SERVER if self._server_mode else PREEMPT_MODE_STUDIO,
                exact = self._exact,
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
    """Block until llama-server reports at most ``target_processing`` requests in flight, socket
    teardown not being evidence the cells are free. A BARRIER, never an attribution: the gauge cannot
    say which generation owns a slot. False means unconfirmed, and the caller proceeds anyway."""
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
    """Binds one generation to the controller, satisfying ``PreemptionPolicy``. ``await_resume`` is
    bounded twice, by the caller's timeout and by refusing to wait once the room is back. ``loop`` is
    why this is a separate class: the stream funnels are synchronous and run on a worker thread while
    ``resume_async`` awaits the queue's slot acquire, so the two are joined with run_coroutine_threadsafe.
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
        """The upstream response is closed, so the tokens may go back. Only once nothing can still be
        decoding against the lease, which the stream side guarantees by calling this after teardown."""
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
        # Decision to cells-released, in milliseconds. Not derivable: nothing logs the decision,
        # and `resident` is /metrics-polled, so it reports the poll interval.
        chosen_at = participant.preempt_chosen_at
        if chosen_at:
            _log.info(
                "llama preemption evict-latency: gen_id=%s ms=%.1f tokens=%s",
                self._gen_id,
                (time.monotonic() - chosen_at) * 1000.0,
                participant.tokens,
            )
            participant.preempt_chosen_at = 0.0
        # Before the state change, so a sweep between the two sees the larger figure.
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
                # A handback that fails must not take the conversation with it: the tokens are
                # reclaimed when the lease is finally released either way.
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
        if self._resumes >= DEFAULT_MAX_PREEMPT_RESUMES:
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
        # Re-stated, not remembered: a resumed run carries the partial it already generated.
        want = max(0, int(participant.tokens or 0))
        # Both of these made the wait sit on room no eviction could produce, `want` growing with
        # every pause until it passes the ceiling it is measured against.
        if self._controller.cannot_ever_fit(want):
            # Bigger than the cache. Ending the turn reports `length`, which the continuation
            # path resumes; waiting hangs until the client disconnects.
            _log.info(
                "llama preemption too-large: gen_id=%s want=%s (exceeds the cache; "
                "finishing the turn)",
                self._gen_id,
                want,
            )
            return False
        if self._controller.outgrew_the_shared_ceiling(want):
            # Fits alone but beside nobody. `room_for` grants it the cache once everyone else
            # is out, so this only has to say why the wait may be long.
            _log.info(
                "llama preemption needs-the-cache: gen_id=%s want=%s (past the shared "
                "ceiling; waiting for the cache to itself)",
                self._gen_id,
                want,
            )
        _log.info("llama preemption awaiting-room: gen_id=%s want=%s", self._gen_id, want)
        # Wait for the cache to actually have room before taking the lease back, or the queue hands
        # a resume out on optimistic accounting and the next sweep evicts it again. The clock
        # measures STALL, not elapsed time; `hard_deadline` covers a cache that churns forever.
        started = time.monotonic()
        deadline = started + timeout
        hard_deadline = started + timeout * MAX_RESUME_WAIT_MULTIPLE
        last = self._controller.progress_signature()
        # Fresh reading first: this grant lets a chat back in carrying its whole replayed partial.
        self._controller.refresh_residency()
        # try_grant_resume, not room_for: the room has to be BOOKED at the instant it is found,
        # or two chats waiting at once both find the same space and both take it. Stop is read
        # before every attempt: a grant that succeeds at once would otherwise skip it.
        while True:
            if cancel_event is not None and cancel_event.is_set():
                # Stop pressed during the pause: nothing to resume, and the worker must not
                # sit here until room appears for a chat nobody is reading.
                _log.info("llama preemption cancelled-while-paused: gen_id=%s", self._gen_id)
                return False
            if self._controller.try_grant_resume(self._gen_id, want):
                break
            self._controller.refresh_residency()
            now = time.monotonic()
            current = self._controller.progress_signature()
            # A holder parked on a tool moves nothing, so the signature freezes while a web
            # search runs. An outstanding external call is work in progress; a tool that never
            # returns is caught by `hard_deadline`.
            _snap = self._controller.snapshot()
            if _snap.parked > 0 or getattr(_snap, "tools_running", 0) > 0:
                deadline = now + timeout
            if current != last:
                # ANY change resets it, the signature covering the whole backend, so "no progress for
                # `timeout`" is a claim that NOTHING moved. Resetting only when room appeared reported
                # "no progress" about a server decoding at full rate; `(committed, holders)` was not enough.
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
                    want,
                    timeout_s = timeout,
                    cancel_event = cancel_event,
                    progress = self._controller.progress_signature,
                )
            except TypeError:
                # An older lease without the stall-aware wait.
                coro = lease.resume_async(want, timeout_s = timeout, cancel_event = cancel_event)
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            # A backstop for a loop that never runs the coroutine at all; resume_async is
            # already bounded by timeout_s.
            got = bool(future.result(timeout = timeout + 5.0))
            if not got:
                # The grant above booked the room. Nothing is going to use it, so hand it back.
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

    def on_server_parked(self) -> None:
        """llama-server parked this slot in host RAM. Nothing is handed back: the server holds
        the sequence and will restore it without asking."""
        participant = self._controller.participant(self._gen_id)
        _log.info(
            "llama preemption server-parked: gen_id=%s tokens=%s",
            self._gen_id,
            participant.tokens if participant is not None else None,
        )
        self._controller.set_state(self._gen_id, ParticipantState.PAUSED)

    def on_server_resumed(self) -> None:
        _log.info("llama preemption server-resumed: gen_id=%s", self._gen_id)
        self._controller.note_resumed(self._gen_id)

    def on_declined(self) -> None:
        """This generation was chosen to pause and is not going to. Called instead of the pause
        handshake, never beside it: `on_preempted` hands the lease back and this stream keeps it."""
        _log.info("llama preemption declined: gen_id=%s (finishing instead)", self._gen_id)
        self._controller.note_declined(self._gen_id)


def _slot_decoded(slot: dict) -> int:
    """Tokens this slot has generated so far, from `/slots`. ``next_token`` is a one-element LIST in
    some builds and a bare object in others; anything else answers zero rather than raising."""
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

    llama.cpp keeps a slot's prompt cache after its request finishes and neither the admission ledger
    nor /metrics sees that residue: ``purging slot 1 with 16383 tokens`` while four chats ran against
    a ledger that believed the cache empty. None from ``fetch`` means "cannot say", never "empty".
    """
    slots = fetch()
    if not slots:
        return None
    resident = 0
    idle_tokens = 0
    idle = []
    for slot in slots:
        # A parked slot (`is_preempted`) reports its logical sequence but holds no cells.
        # Counting it made a nearly empty pool read as over the ceiling.
        if slot.get("is_preempted"):
            continue
        # WHICH field this came from decides whether the decoded tokens still have to be added, so
        # the two cannot collapse into one `or` chain: `n_prompt_tokens_cache` was 0 in 128 samples.
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
        # Plus what it has GENERATED: `/slots` reports the prompt only, which is why every watermark
        # diagnostic came back innocent. Counted twice it scored 28238 cells in a 16384-cell cache.
        if slot.get("is_processing") and not counts_generated:
            tokens += _slot_decoded(slot)
        # Idle slots count too: `try_clear_idle_slots` recycles them only from the KV-full retry,
        # after a decode already failed. Counting them gave 3 clean runs of 4; excluding them 0 of 2.
        resident += max(0, tokens)
        if not slot.get("is_processing") and tokens > 0:
            idle_tokens += max(0, tokens)
            idle.append((slot.get("id"), max(0, tokens)))
    # Largest first: the fewest erases free the most.
    idle.sort(key = lambda pair: -pair[1])
    return {
        "resident": resident,
        # Reclaimable, not occupied: reported separately so the caller can free it BEFORE
        # pausing anybody.
        "idle_tokens": idle_tokens,
        "idle": idle,
        "slots": len(slots),
    }


def reclaim_idle_slots(
    occupancy: Optional[dict], erase: Callable[[int], int], *, needed: int
) -> int:
    """Free dead residue before asking a live chat to stop. llama.cpp does this itself in
    ``try_clear_idle_slots``, but only once the decode has ALREADY failed. Returns tokens freed."""
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
            # preempting a live generation.
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
            # base_url takes a fresh ephemeral port on every load. is_idle takes each
            # controller's own lock, so the list is built first.
            stale = [k for k, c in _CONTROLLERS.items() if k != key and c.is_idle()]
            for k in stale:
                del _CONTROLLERS[k]
        return controller


def reset_preemption_controllers() -> None:
    with _CONTROLLERS_LOCK:
        _CONTROLLERS.clear()
