# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The process-wide GPU lease Flows steps take, and how many chat requests the loaded model serves at once."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Callable, Optional, Union


class NeedsExclusiveGpu(Exception):
    """The loaded model is not this shared step's: rerun it exclusively."""


def _llama_slots(backend: Any) -> int:
    return max(1, int(backend.effective_parallel_slots)) if backend.is_loaded else 1


def chat_slots(model: Optional[str] = None) -> int:
    """How many requests a loaded chat model serves at once: its llama-server's slots, else 1 (a Transformers model
    generates one request at a time). ``model`` names a model kept alongside the primary one; any other is the
    primary's."""
    try:
        slot = _kept_slot(model)
        if slot is not None:
            return _llama_slots(slot.llama)
        from routes.inference import get_llama_cpp_backend

        return _llama_slots(get_llama_cpp_backend())
    except Exception:
        return 1


def _kept_slot(model: Optional[str]) -> Any:
    """The slot keeping ``model`` loaded alongside the primary chat model, or None."""
    if not model:
        return None
    try:
        from core.inference import model_slots
        wanted = model.split(":", 1)[0].lower()
        for slot in model_slots.visible():
            names = (
                getattr(slot.llama, "model_identifier", None),
                getattr(slot.llama, "_openai_advertised_id", None),
            )
            if any(isinstance(n, str) and n.lower() == wanted for n in names):
                return slot
    except Exception:
        return None
    return None


def chat_model_key(params: dict) -> Optional[str]:
    """The lease key of a chat step: the repo it names when that model is kept alongside the primary one (it has
    its own slots), else None, the primary."""
    choice = params.get("model") if isinstance(params, dict) else None
    repo = choice.get("repo") if isinstance(choice, dict) else None
    if not (isinstance(repo, str) and repo) or _kept_slot(repo) is None:
        return None
    return repo.lower()


class GpuLease:
    """Exclusive holders run alone; ``shared()`` holders run up to the slot count, never while an exclusive
    holder runs or waits, so a model is never swapped under a generation."""

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._exclusive = False
        self._waiting = 0
        self._shared = 0
        self._by_model: dict[Optional[str], list[int]] = {}

    def locked(self) -> bool:
        return self._exclusive or self._shared > 0 or self._waiting > 0

    async def __aenter__(self) -> "GpuLease":
        async with self._cond:
            self._waiting += 1
            try:
                await self._cond.wait_for(lambda: not self._exclusive and self._shared == 0)
            finally:
                self._waiting -= 1
                # A cancelled exclusive waiter must not leave shared callers asleep behind it.
                self._cond.notify_all()
            self._exclusive = True
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        async with self._cond:
            self._exclusive = False
            self._cond.notify_all()

    @contextlib.asynccontextmanager
    async def shared(
        self,
        *,
        background: bool = False,
        slots: Union[int, Callable[[], int]] = 1,
        model: Union[Optional[str], Callable[[], Optional[str]]] = None,
    ):
        """``model`` keys the slot count: calls on different models kept loaded do not count against each other."""
        key: Optional[str] = None

        def admit() -> bool:
            nonlocal key
            key = model() if callable(model) else model
            n = max(1, int(slots() if callable(slots) else slots))
            cap = max(1, n - 1) if background else n
            held, held_bg = self._by_model.get(key, (0, 0))
            return (
                not self._exclusive
                and self._waiting == 0
                and held < n
                and (not background or held_bg < cap)
            )

        async with self._cond:
            await self._cond.wait_for(admit)
            self._shared += 1
            held, held_bg = self._by_model.get(key, (0, 0))
            self._by_model[key] = [held + 1, held_bg + background]
        try:
            yield
        finally:
            async with self._cond:
                self._shared -= 1
                counts = self._by_model[key]
                counts[0] -= 1
                counts[1] -= background
                if not counts[0]:
                    del self._by_model[key]
                self._cond.notify_all()
