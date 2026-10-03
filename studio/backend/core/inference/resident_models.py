# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multiple GGUF models resident at once, each on its own llama-server.

Single-model Studio keeps one :class:`LlamaCppBackend` that every load swaps in
place: ``load_model`` kills the previous child before spawning the replacement.
The registry adds the multi-model shape on top without changing that default:

* ``active_backend()`` answers the backend the rest of Studio already calls
  ``get_llama_cpp_backend()`` for. With residency disabled (the default) it is
  always the original module-level backend, so every existing caller keeps
  today's behaviour and today's object identity.
* A slot is a second backend a load asked to keep beside the active one. Slots
  only exist while their backend holds a model; unloading drops the slot. No
  identity is mirrored here -- ``model_identifier``/``hf_variant`` stay on the
  backend and are read through it, so the registry can never disagree with the
  process that actually serves.
* Under residency the default backend is never loaded: a session's first load
  opens a slot like any other. That keeps "no active model" representable while
  residents remain (unload the active slot, keep the rest) -- the default then
  answers unloaded, exactly like a fresh single-model server.

There is no eviction policy here. Slots are opened by explicit ``keep_existing_loaded``
loads and closed by explicit unloads, the app shutdown, or a GPU-owner eviction.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from loggers import get_logger

if TYPE_CHECKING:  # pragma: no branch - import cycle is real, only types are needed
    from core.inference.llama_cpp import LlamaCppBackend

logger = get_logger(__name__)

# Read per call (not at import) so tests and embedded hosts can change it.
_MAX_SLOTS_ENV = "UNSLOTH_RESIDENT_MODEL_SLOTS"


class ResidentCapacityError(RuntimeError):
    """A keep-existing load would exceed the configured resident slot count."""


def max_resident_slots() -> int:
    """How many GGUF backends may be resident at once, including the active one.

    1 (the default) disables multi-residency entirely: no slot is ever opened,
    and ``active_backend()`` is permanently the module-level backend. Values
    below 1 are treated as 1 so a typo can never yield a server that refuses
    every load.
    """
    raw = os.environ.get(_MAX_SLOTS_ENV)
    if raw is None or not raw.strip():
        return 1
    try:
        return max(1, int(raw.strip()))
    except ValueError:
        return 1


def residency_enabled() -> bool:
    return max_resident_slots() > 1


# Module-level instance holder. The route module constructs the registry (its
# default backend is the module-level LlamaCppBackend) and registers it here at
# import, so core modules -- the GPU arbiter's chat eviction -- can reach it
# without importing the route package. None until routes has been imported.
_REGISTRY: Optional["ResidentLlamaRegistry"] = None


def set_registry(registry: "ResidentLlamaRegistry") -> None:
    global _REGISTRY
    _REGISTRY = registry


def get_registry() -> Optional["ResidentLlamaRegistry"]:
    return _REGISTRY


@dataclass(frozen = True)
class ResidentSlot:
    """One resident backend plus the registry's own bookkeeping."""

    id: int
    backend: "LlamaCppBackend"


class ResidentLlamaRegistry:
    """Owns the resident backends and the identity of the active one.

    Thread safety: one RLock guards the slot table and the active pointer. It is
    never held across a load or an unload -- those run on the route layer's
    lifecycle gate -- only across the table reads and writes themselves, so a
    poll can never observe a half-updated registry.
    """

    def __init__(
        self,
        default_backend: "LlamaCppBackend",
        *,
        backend_factory: Optional[Callable[[], "LlamaCppBackend"]] = None,
    ) -> None:
        self._default = default_backend
        # Injectable so tests can populate slots with doubles; production always
        # builds the real backend (whose constructor reaps orphans and registers
        # the atexit teardown, both wanted for a slot that owns a child).
        self._backend_factory = backend_factory
        self._lock = threading.RLock()
        self._slots: dict[int, ResidentSlot] = {}
        # A resident slot is registered before its backend can spawn or download.
        # Keep that in the registry rather than on the route: /unload must find
        # and stop a fresh slot while the successful-load activation is pending.
        self._loading_slot_ids: set[int] = set()
        self._next_slot_id = 1
        self._active_slot_id: Optional[int] = None

    # ── Active backend ────────────────────────────────────────────

    @property
    def default_backend(self) -> "LlamaCppBackend":
        return self._default

    def active_backend(self) -> "LlamaCppBackend":
        """The backend ``get_llama_cpp_backend()`` must answer.

        The active slot's backend when one is set; otherwise the default. The
        active slot may be mid-swap (its backend momentarily unloaded) and is
        still returned: callers read ``is_loaded`` themselves, and answering the
        default instead would hand them a backend that never loaded anything.
        """
        with self._lock:
            slot = (
                self._slots.get(self._active_slot_id) if self._active_slot_id is not None else None
            )
            return slot.backend if slot is not None else self._default

    def active_slot(self) -> Optional[ResidentSlot]:
        with self._lock:
            return (
                self._slots.get(self._active_slot_id) if self._active_slot_id is not None else None
            )

    def set_active(self, slot_id: int) -> None:
        with self._lock:
            if slot_id in self._slots:
                self._active_slot_id = slot_id

    # ── Slot table ────────────────────────────────────────────────

    def _busy_slot_count_locked(self) -> int:
        """Slots holding or acquiring a process.

        A slot whose backend lost its model (the idle auto-unload tears the
        active one down off the registry's books) still occupies the table but
        no longer holds VRAM, so it must not consume capacity. A newly opened
        slot is busy from ``start_loading`` onward, before its backend has
        published a process.
        """
        return sum(
            1
            for slot_id, slot in self._slots.items()
            if slot_id in self._loading_slot_ids or slot.backend.is_active
        )

    def at_capacity(self) -> bool:
        """Whether one more slot would exceed the configured maximum.

        A cheap pre-check so a refused load can reject before any teardown or
        generation cancellation runs. Safe against a racing load because loads
        serialize on the route layer's lifecycle gate; ``open_slot`` enforces
        the same bound for any caller without that gate.
        """
        with self._lock:
            return self._busy_slot_count_locked() >= max_resident_slots()

    def open_slot(self) -> ResidentSlot:
        """Register a fresh backend as a resident slot, at capacity or not.

        Raises :class:`ResidentCapacityError` when the table is full, before any
        backend is constructed, so a refused load never pays the orphan sweep
        the constructor runs.
        """
        from core.inference.llama_cpp import LlamaCppBackend
        with self._lock:
            if self._busy_slot_count_locked() >= max_resident_slots():
                raise ResidentCapacityError(
                    f"Cannot keep more than {max_resident_slots()} GGUF model(s) resident "
                    f"(set {_MAX_SLOTS_ENV} to raise the limit)."
                )
            factory = self._backend_factory or LlamaCppBackend
            slot = ResidentSlot(id = self._next_slot_id, backend = factory())
            self._slots[slot.id] = slot
            self._next_slot_id += 1
            return slot

    def start_loading(self, slot_id: int) -> None:
        """Mark a registered slot cancellable before its backend is active."""
        with self._lock:
            if slot_id in self._slots:
                self._loading_slot_ids.add(slot_id)

    def finish_loading(self, slot_id: int) -> None:
        """Clear the transient load marker after success or a completed abort."""
        with self._lock:
            self._loading_slot_ids.discard(slot_id)

    def slot_for_backend(self, backend: "LlamaCppBackend") -> Optional[ResidentSlot]:
        with self._lock:
            for slot in self._slots.values():
                if slot.backend is backend:
                    return slot
            return None

    def drop_slot(
        self,
        slot_id: int,
        *,
        unload: bool = True,
    ) -> Optional["LlamaCppBackend"]:
        """Remove a slot. ``unload`` defensively terminates a leftover child.

        Returns the dropped backend so callers can report what left; None when
        the slot was already gone (a double unload must not raise).
        """
        with self._lock:
            slot = self._slots.pop(slot_id, None)
            if slot is None:
                return None
            self._loading_slot_ids.discard(slot_id)
            if self._active_slot_id == slot_id:
                # No implicit promotion: another resident becomes active only by
                # an explicit load naming it.
                self._active_slot_id = None
        if unload:
            try:
                slot.backend.unload_model()
            except Exception:  # noqa: BLE001 - teardown best-effort, mirrors run.py sweeps
                logger.warning("resident_slot_drop_unload_failed", slot_id = slot_id, exc_info = True)
        return slot.backend

    # ── Queries ───────────────────────────────────────────────────

    def loaded_backends(self, *, active_first: bool = True) -> "list[LlamaCppBackend]":
        """Backends currently holding a model, active first.

        A slot whose load failed or whose slot was never used is not listed:
        it serves nothing and must not answer identity checks.
        """
        active = self.active_slot()
        with self._lock:
            slots = [s for s in self._slots.values() if s.backend.is_loaded]
        if active_first and active is not None:
            slots.sort(key = lambda s: s.id != active.id)
        return [s.backend for s in slots]

    def any_slot_busy(self) -> bool:
        """Whether any slot backend holds, or is on its way to holding, a process.

        ``is_active`` (a process exists, healthy or not), not ``is_loaded``: a
        starting model holds VRAM the GPU arbiter must still count before
        granting the device to a diffusion or video load.
        """
        with self._lock:
            return bool(self._loading_slot_ids) or any(
                s.backend.is_active for s in self._slots.values()
            )

    def any_slot_holding_vram(self) -> bool:
        """Whether any slot holds, or is on its way to holding, GPU VRAM.

        The arbiter's resource question, unlike :meth:`any_slot_busy`'s process
        question: a live server that is a settled, confirmed zero-VRAM launch
        holds none of what the arbiter allocates, so the CHAT claim may drop
        while it keeps serving. A slot whose load is still in flight only
        answers through its backend once healthy -- before the process exists
        or reports healthy, ``holds_no_vram`` still describes the previous
        launch and cannot be trusted, so those states count as holding.
        """
        with self._lock:
            for slot_id, slot in self._slots.items():
                backend = slot.backend
                if not backend.is_active:
                    # No process yet: only a load still in flight can reach the GPU.
                    if slot_id in self._loading_slot_ids:
                        return True
                    continue
                if not backend.is_loaded or not backend.holds_no_vram:
                    return True
        return False

    def loading_slots(self) -> "list[ResidentSlot]":
        """Slots with a GGUF load in progress, in deterministic slot order.

        Loads normally serialize on the route lifecycle gate, so there is one.
        Including an observed starting backend keeps the registry correct for
        callers that began before this marker existed.
        """
        with self._lock:
            return [
                slot
                for slot_id, slot in self._slots.items()
                if slot_id in self._loading_slot_ids
                or (slot.backend.is_active and not slot.backend.is_loaded)
            ]

    def slots_for_sweep(self) -> "list[ResidentSlot]":
        """A stable snapshot of the slot table for external sweeps (training
        VRAM frees). Callers judge each backend's GPU state themselves; this
        only guarantees the list cannot mutate while they act on it."""
        with self._lock:
            return list(self._slots.values())

    # ── Lifecycle ─────────────────────────────────────────────────

    def teardown_all(self) -> int:
        """Tear every slot down (app shutdown, GPU-owner eviction). Not the default.

        Returns how many backends were torn down. Errors are contained per slot
        so one stuck child cannot save another from its own teardown.
        """
        with self._lock:
            slots = list(self._slots.values())
            self._slots.clear()
            self._loading_slot_ids.clear()
            self._active_slot_id = None
        for slot in slots:
            try:
                slot.backend.unload_model()
            except Exception:  # noqa: BLE001 - see above
                logger.warning("resident_slot_teardown_failed", slot_id = slot.id, exc_info = True)
        return len(slots)

    def begin_lifecycle(self) -> None:
        """Reset shutdown flags on every backend for an embedded restart.

        Mirrors ``LlamaCppBackend._begin_server_lifecycle`` for the whole table:
        a host that calls ``run_server()`` twice must not leave the second
        session's slots believing they are shutting down.
        """
        with self._lock:
            backends = [s.backend for s in self._slots.values()]
        for backend in backends:
            backend._begin_server_lifecycle()
        self._default._begin_server_lifecycle()
