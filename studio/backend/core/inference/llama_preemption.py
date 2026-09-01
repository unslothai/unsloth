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

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.inference.llama_admission import LlamaAdmissionLease, _bool_env


_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


# Off falls back to step 1's wire clamp alone, which is the behaviour that predates any
# of this. Mirrors UNSLOTH_LLAMA_ADMISSION_KV_BUDGET, and is parsed by the same helper so
# it accepts the same spellings.
PREEMPT_ENV = "UNSLOTH_LLAMA_ADMISSION_PREEMPT"
DEFAULT_PREEMPT_ENABLED = True

# Room held clear of the budget. A commitment is an estimate, and the cost of being a
# little wrong is the crash this module exists to prevent, so the last few per cent are
# never handed out.
DEFAULT_PREEMPT_BUFFER_RATIO = 0.05
DEFAULT_PREEMPT_BUFFER_MIN_TOKENS = 256

# Approved anti-starvation rule: a chat preempted this many times in a row outranks
# longest-wins for the next epoch.
PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS = 3

# The reclaim barrier gives up after this long and lets the replacement in anyway. A
# server without --metrics never answers, and blocking a conversation forever on a gauge
# that may not exist is worse than the overrun it guards.
DEFAULT_RECLAIM_BARRIER_TIMEOUT_S = 10.0
DEFAULT_RECLAIM_BARRIER_POLL_S = 0.05


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
    PAUSED = "paused"
    DONE = "done"


# Holds KV at llama-server, so it counts against the budget.
_HOLDS_KV = frozenset({
    ParticipantState.DECODING,
    ParticipantState.PARKED_ON_TOOL,
    ParticipantState.TOOLS_RUNNING,
})

# May be asked to stop. QUEUED holds nothing yet, PAUSED already stopped, DONE is gone,
# and TOOLS_RUNNING is excluded for the reason on its constant.
_PREEMPTABLE = frozenset({
    ParticipantState.DECODING,
    ParticipantState.PARKED_ON_TOOL,
})


def preemption_enabled() -> bool:
    return _bool_env(PREEMPT_ENV, DEFAULT_PREEMPT_ENABLED)


def preemption_buffer_tokens(budget: int) -> int:
    """Tokens held clear of ``budget``. Zero for an unknown budget, which disables it."""
    if budget <= 0:
        return 0
    scaled = int(math.ceil(budget * DEFAULT_PREEMPT_BUFFER_RATIO))
    return max(DEFAULT_PREEMPT_BUFFER_MIN_TOKENS, scaled)


@dataclass(**_SLOTS)
class Participant:
    """One in-flight generation, from the preemptor's point of view."""

    gen_id: str
    seq: int
    lease: Optional[LlamaAdmissionLease] = None
    tokens: int = 0
    state: str = ParticipantState.DECODING
    consecutive_preemptions: int = 0
    # Set when this generation must stop. The caller aborts ONLY the upstream
    # llama-server stream on it and keeps its own generator, ledgers, conversation and
    # SSE response alive; a shared cancel event could not express that, because six
    # separate consumers treat cancellation as terminal.
    preempt_event: threading.Event = field(default_factory = threading.Event)

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


class PreemptionController:
    """Victim choice and the epoch, for one llama-server backend.

    Not a scheduler: nothing here runs generations or owns a thread. Callers report where
    they are, ask whether there is room, and are told who must stop.
    """

    __slots__ = ("key", "_lock", "_participants", "_seq", "_epoch_winner", "_budget", "_kv_unified")

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

    def configure(self, *, budget: Optional[int] = None, kv_unified: Optional[bool] = None) -> None:
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
    ) -> Participant:
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
                state = state,
            )
            self._participants[gen_id] = participant
            return participant

    def unregister(self, gen_id: str) -> None:
        """Drop a finished generation and end its epoch if it held one."""
        with self._lock:
            self._participants.pop(gen_id, None)
            if self._epoch_winner == gen_id:
                self._epoch_winner = None

    def note_tokens(self, gen_id: str, tokens: int) -> None:
        with self._lock:
            participant = self._participants.get(gen_id)
            if participant is not None:
                participant.tokens = max(0, int(tokens or 0))

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

    def is_idle(self) -> bool:
        """Nothing in flight, so this controller may be retired. Mirrors the queue's."""
        with self._lock:
            return not self._participants

    def committed_tokens(self) -> int:
        with self._lock:
            return self._committed_locked()

    def _committed_locked(self) -> int:
        return sum(p.tokens for p in self._participants.values() if p.holds_kv)

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
            buffer = preemption_buffer_tokens(self._budget)
            ceiling = max(0, self._budget - buffer)
            total = self._committed_locked()
            want = max(0, int(needed or 0))
            if total + want <= ceiling:
                return []
            winner = self._winner_locked()
            # Parked holders first: they hold KV and consume no compute, so their room is
            # the cheapest to take. Then the largest decoders, so the fewest victims free
            # the most. The winner is never a victim.
            victims = [
                p
                for p in self._participants.values()
                if p.preemptable and (winner is None or p.gen_id != winner.gen_id)
            ]
            victims.sort(
                key = lambda p: (
                    p.state != ParticipantState.PARKED_ON_TOOL,
                    -p.tokens,
                    p.seq,
                )
            )
            chosen: List[Participant] = []
            for victim in victims:
                if total + want <= ceiling:
                    break
                total -= victim.tokens
                victim.consecutive_preemptions += 1
                victim.state = ParticipantState.PAUSED
                victim.preempt_event.set()
                chosen.append(victim)
            return chosen

    def snapshot(self) -> PreemptionSnapshot:
        with self._lock:
            states = [p.state for p in self._participants.values()]
            return PreemptionSnapshot(
                key = self.key,
                budget = self._budget,
                buffer = preemption_buffer_tokens(self._budget),
                committed = self._committed_locked(),
                decoding = states.count(ParticipantState.DECODING),
                paused = states.count(ParticipantState.PAUSED),
                parked = states.count(ParticipantState.PARKED_ON_TOOL),
                winner = self._epoch_winner,
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
