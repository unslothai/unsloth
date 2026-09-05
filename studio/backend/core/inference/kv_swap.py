# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""KV-cache checkpoint swapping for local llama-server generation requests.

Admission control (``llama_admission``) decides how many chats may hold a slot.
This module decides what happens when the chats that already hold one no longer
fit together in the shared KV cache: instead of failing the newest with
"Context size has been exceeded", one chat keeps decoding and the others have
their KV written out with llama-server's slot state API, freeing the cells until
there is room again.

The mechanism, and why it is shaped this way, rests on four measured facts about
llama-server b10715 (see ``plans/impl_v2_slot_swap.md`` for the probes):

1. ``POST /slots/{id}?action=save`` writes the slot's sequence to
   ``--slot-save-path``; ``erase`` frees the cells; ``restore`` reads it back into
   ANY free slot, not only the one it came from.
2. After restoring an N-token save, a resume prompt hits the restored cells only
   if its first N tokens are the saved sequence AND it is at least N+1 tokens
   long. Exactly N tokens, or fewer, re-prefills from scratch: this build has no
   prefix-truncation reuse.
3. A request that ends cleanly leaves the caller holding exactly KV+1 tokens,
   because the last sampled token is not fed until the next decode. So a clean
   generation boundary is the only place where (2) is satisfied for free.
4. A stream aborted mid-flight leaves the KV 4-5 tokens AHEAD of the caller (the
   in-flight speculative batch), and draining does not close the gap. Preempting
   mid-generation therefore cannot resume without recompute, and falls back.

So generation is issued in bounded chunks and the chunk boundary is the
preemption point. Nothing here knows about FastAPI, SSE or the route shape; it
coordinates state and talks to llama-server over HTTP.
"""

from __future__ import annotations

import os
import struct
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# Mirrors llama_admission: dataclass(slots=True) is 3.10+, this package declares >=3.9.
_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


KV_SWAP_ENV = "UNSLOTH_LLAMA_KV_SWAP"
KV_SWAP_CHUNK_ENV = "UNSLOTH_LLAMA_KV_SWAP_CHUNK"
KV_SWAP_BUFFER_ENV = "UNSLOTH_LLAMA_KV_SWAP_BUFFER"
KV_SWAP_EVERY_ENV = "UNSLOTH_LLAMA_KV_SWAP_EVERY"

# 512 costs +0.86% on a 1536-token generation, 1024 +0.72%, 256 +3.41%. 512 keeps the
# pause granularity useful without paying for it; and chunking is armed only when more
# than one chat is active, so a solo chat runs unbounded and pays nothing at all.
DEFAULT_CHUNK_TOKENS = 512

# Per slot, the KV runs 4-5 tokens ahead of what the client has seen while a speculative
# batch is in flight. Three tokens of margin on top of the configured draft depth covers
# the observed lead; the total is multiplied by the slot count because every slot can be
# mid-batch at once. Raise only on an observed failure, and record the failure.
DEFAULT_DRAFT_MARGIN = 3

# llama_state_seq_save_file header, verified byte-for-byte against the token ids the
# server streamed back: magic, version, then n_token_count at offset 20 and the ids at 24.
SLOT_FILE_MAGIC = 0x67677371  # 'ggsq'
SLOT_FILE_VERSIONS = (3,)
_SLOT_FILE_COUNT_OFFSET = 20
_SLOT_FILE_TOKENS_OFFSET = 24

# States a swappable chat moves through.
RUNNING = "running"
PAUSED = "paused"
RESTORING = "restoring"
DONE = "done"

# A chat that has been swapped this many times in a row is protected while any other
# candidate exists, so one unlucky chat cannot be starved by a steady stream of rivals.
SWAP_STREAK_PROTECT = 3


class KvSwapError(RuntimeError):
    """A swap could not be completed; the caller falls back to re-prefill."""


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def kv_swap_enabled() -> bool:
    """Rollout switch. Default on; ``UNSLOTH_LLAMA_KV_SWAP=0`` restores plain admission."""
    return _bool_env(KV_SWAP_ENV, True)


def kv_swap_chunk_tokens() -> int:
    """Upstream ``max_tokens`` per generation chunk. 0 disables chunking entirely."""
    return _int_env(KV_SWAP_CHUNK_ENV, DEFAULT_CHUNK_TOKENS, minimum = 0)


def kv_swap_force_every() -> int:
    """Test knob: force a swap every N generated tokens. 0 (default) means never."""
    return _int_env(KV_SWAP_EVERY_ENV, 0, minimum = 0)


def default_buffer_tokens(n_parallel: int, draft_n_max: int) -> int:
    """Smallest buffer that covers the in-flight speculative lead on every slot.

    The KV runs ahead of the caller by the draft batch that is mid-flight. Every slot can
    be mid-batch simultaneously, so the reserve is per-slot and summed. An explicit
    ``UNSLOTH_LLAMA_KV_SWAP_BUFFER`` overrides it.
    """
    override = _int_env(KV_SWAP_BUFFER_ENV, -1, minimum = 0)
    if override >= 0:
        return override
    slots = max(1, int(n_parallel or 1))
    draft = max(0, int(draft_n_max or 0))
    return slots * (draft + DEFAULT_DRAFT_MARGIN)


def parse_slot_file_tokens(path) -> List[int]:
    """Return the exact token ids a saved slot file holds.

    The saved sequence, not what the client saw, is what the restored cells contain, and
    fact (2) above means the resume prompt has to line up with it exactly. Raises
    :class:`KvSwapError` on anything unexpected so the caller can fall back rather than
    resume onto a mismatched cache.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(_SLOT_FILE_TOKENS_OFFSET)
            if len(head) < _SLOT_FILE_TOKENS_OFFSET:
                raise KvSwapError(f"slot file {path} is truncated ({len(head)} bytes)")
            magic, version = struct.unpack_from("<II", head, 0)
            if magic != SLOT_FILE_MAGIC:
                raise KvSwapError(f"slot file {path} has magic 0x{magic:08x}, expected 0x{SLOT_FILE_MAGIC:08x}")
            if version not in SLOT_FILE_VERSIONS:
                raise KvSwapError(f"slot file {path} has version {version}, expected one of {SLOT_FILE_VERSIONS}")
            (count,) = struct.unpack_from("<I", head, _SLOT_FILE_COUNT_OFFSET)
            if count <= 0:
                raise KvSwapError(f"slot file {path} reports {count} tokens")
            raw = handle.read(4 * count)
            if len(raw) < 4 * count:
                raise KvSwapError(f"slot file {path} holds {len(raw) // 4} of {count} tokens")
            return list(struct.unpack(f"<{count}i", raw))
    except KvSwapError:
        raise
    except OSError as exc:
        raise KvSwapError(f"slot file {path} could not be read: {exc}") from exc


def resume_prompt_is_valid(saved: Sequence[int], resume: Sequence[int]) -> bool:
    """Whether ``resume`` will land on cells restored from ``saved`` instead of re-prefilling.

    Fact (2): the saved sequence must be a PROPER prefix of the resume prompt. Equal
    length re-prefills just as surely as a mismatch does, which is the trap this guards.
    """
    if not saved or len(resume) <= len(saved):
        return False
    return list(resume[:len(saved)]) == list(saved)


@dataclass(**_SLOTS)
class KvSwapChat:
    """One admitted chat's swap state."""

    chat_id: str
    slot: Optional[int] = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    state: str = RUNNING
    swap_streak: int = 0
    swaps_total: int = 0
    filename: Optional[str] = None
    saved_tokens: int = 0
    admitted_at: float = field(default_factory = time.monotonic)
    last_restore_ms: float = 0.0
    last_resume_prompt_n: Optional[int] = None

    @property
    def resident(self) -> int:
        """Cells this chat occupies: its prompt plus everything generated so far."""
        if self.state in (PAUSED, DONE):
            return 0
        return max(0, self.prompt_tokens) + max(0, self.generated_tokens)

    @property
    def size(self) -> int:
        """Cells it needs when running, whether or not it currently holds them."""
        return max(0, self.prompt_tokens) + max(0, self.generated_tokens)


@dataclass(**_SLOTS)
class KvSwapDecision:
    """What the controller wants done at one pressure check."""

    victims: List[str] = field(default_factory = list)
    keep: Optional[str] = None
    resident: int = 0
    budget: int = 0
    reason: str = ""


class KvSwapController:
    """Live KV accounting and the swap policy for one llama-server backend.

    Thread-safe: the tool loop drives generation from an executor thread while the route
    runs on the event loop, so every mutation takes ``_lock``. Nothing here blocks on I/O
    while holding it - the HTTP calls are made by the caller through the injected
    ``http_post``, outside the lock.
    """

    def __init__(
        self,
        key: str,
        *,
        n_ctx: int,
        n_parallel: int,
        draft_n_max: int = 0,
        save_dir: Optional[str] = None,
        http_post: Optional[Callable[..., object]] = None,
    ) -> None:
        self.key = key
        self._lock = threading.Lock()
        self._chats: Dict[str, KvSwapChat] = {}
        self.n_ctx = max(0, int(n_ctx or 0))
        self.n_parallel = max(1, int(n_parallel or 1))
        self.draft_n_max = max(0, int(draft_n_max or 0))
        self.save_dir = save_dir
        self._http_post = http_post
        self._token = uuid.uuid4().hex[:8]
        self.swaps_out = 0
        self.swaps_in = 0
        self.fallbacks = 0

    # ---------------------------------------------------------------- accounting

    @property
    def buffer_tokens(self) -> int:
        return default_buffer_tokens(self.n_parallel, self.draft_n_max)

    @property
    def budget(self) -> int:
        """Cells the chats may share: the context minus the smallest safe reserve."""
        return max(0, self.n_ctx - self.buffer_tokens)

    def admit(self, chat_id: str, *, prompt_tokens: int = 0, slot: Optional[int] = None) -> KvSwapChat:
        with self._lock:
            chat = self._chats.get(chat_id)
            if chat is None:
                chat = KvSwapChat(chat_id = chat_id)
                self._chats[chat_id] = chat
            chat.prompt_tokens = max(0, int(prompt_tokens or 0))
            chat.slot = slot
            chat.state = RUNNING
            return chat

    def update(self, chat_id: str, *, prompt_tokens: Optional[int] = None,
               generated_tokens: Optional[int] = None) -> None:
        with self._lock:
            chat = self._chats.get(chat_id)
            if chat is None:
                return
            if prompt_tokens is not None:
                chat.prompt_tokens = max(0, int(prompt_tokens))
            if generated_tokens is not None:
                chat.generated_tokens = max(0, int(generated_tokens))

    def finish(self, chat_id: str) -> None:
        with self._lock:
            chat = self._chats.pop(chat_id, None)
        if chat is not None:
            chat.state = DONE

    def get(self, chat_id: str) -> Optional[KvSwapChat]:
        with self._lock:
            return self._chats.get(chat_id)

    def resident_total(self) -> int:
        with self._lock:
            return sum(chat.resident for chat in self._chats.values())

    def active_count(self) -> int:
        """Chats holding cells right now. Chunking is armed only above one."""
        with self._lock:
            return sum(1 for chat in self._chats.values() if chat.state == RUNNING)

    def reconcile(self, slots_payload: Sequence[dict]) -> None:
        """Correct the running totals against llama-server's own ``/slots`` readout.

        The stream tells us what was emitted; the server knows what it actually holds, and
        the two differ by the in-flight speculative batch. Ground truth wins.
        """
        by_slot: Dict[int, int] = {}
        for entry in slots_payload or []:
            try:
                by_slot[int(entry.get("id"))] = max(0, int(entry.get("n_prompt_tokens") or 0))
            except (TypeError, ValueError):
                continue
        with self._lock:
            for chat in self._chats.values():
                if chat.state != RUNNING or chat.slot is None:
                    continue
                held = by_slot.get(chat.slot)
                if held is None or held <= 0:
                    continue
                # Keep the split between prompt and generated, but make the sum truthful.
                drift = held - chat.size
                if drift:
                    chat.generated_tokens = max(0, chat.generated_tokens + drift)

    # ------------------------------------------------------------------- policy

    def plan(self, *, incoming: int = 0, incoming_id: Optional[str] = None) -> KvSwapDecision:
        """Choose who keeps decoding and who is swapped out.

        Keep the chat with the most tokens: it is furthest from done and the costliest to
        move. Swap the others newest-first, never the last one standing, and skip a chat
        that has been swapped ``SWAP_STREAK_PROTECT`` times running while another
        candidate is available.
        """
        with self._lock:
            running = [c for c in self._chats.values() if c.state == RUNNING]
            resident = sum(c.resident for c in running)
            budget = self.budget
            need = resident + max(0, int(incoming or 0))
            decision = KvSwapDecision(resident = resident, budget = budget)
            if budget <= 0 or need <= budget or len(running) <= 1:
                decision.reason = "fits" if need <= budget else "single-chat"
                if running:
                    decision.keep = max(running, key = lambda c: c.size).chat_id
                return decision

            keeper = max(running, key = lambda c: (c.size, -c.admitted_at))
            decision.keep = keeper.chat_id
            # Newest first: the youngest chat has the least invested and the least to lose.
            candidates = [c for c in running if c.chat_id != keeper.chat_id]
            candidates.sort(key = lambda c: c.admitted_at, reverse = True)

            protected = [c for c in candidates if c.swap_streak >= SWAP_STREAK_PROTECT]
            preferred = [c for c in candidates if c.swap_streak < SWAP_STREAK_PROTECT]
            # Protected chats are only touched once nothing else is left to give.
            ordered = preferred + protected

            freed = 0
            for chat in ordered:
                if need - freed <= budget:
                    break
                if len(running) - len(decision.victims) <= 1:
                    break
                decision.victims.append(chat.chat_id)
                freed += chat.resident
            decision.reason = "pressure" if decision.victims else "no-candidate"
            return decision

    # -------------------------------------------------------------------- swaps

    def _post(self, slot: int, action: str, filename: Optional[str] = None):
        if self._http_post is None:
            raise KvSwapError("no HTTP transport configured for slot state calls")
        body = {"filename": filename} if filename else {}
        return self._http_post(slot = slot, action = action, body = body)

    def swap_out(self, chat_id: str) -> KvSwapChat:
        """Write the chat's KV out and free its cells.

        Save first, erase only once the save is known good: an erase after a failed save
        would throw the conversation away, and the fallback is meant to cost recompute,
        never content.
        """
        chat = self.get(chat_id)
        if chat is None:
            raise KvSwapError(f"unknown chat {chat_id}")
        if chat.slot is None:
            raise KvSwapError(f"chat {chat_id} is not pinned to a slot")
        if chat.state != RUNNING:
            raise KvSwapError(f"chat {chat_id} is {chat.state}, not {RUNNING}")

        filename = f"kvswap-{self._token}-{chat_id}-{uuid.uuid4().hex[:6]}.bin"
        saved = self._post(chat.slot, "save", filename)
        n_saved = 0
        if isinstance(saved, dict):
            try:
                n_saved = int(saved.get("n_saved") or 0)
            except (TypeError, ValueError):
                n_saved = 0
        if n_saved <= 0:
            self._discard(filename)
            raise KvSwapError(f"chat {chat_id} saved {n_saved} tokens")

        self._post(chat.slot, "erase")
        with self._lock:
            chat.state = PAUSED
            chat.filename = filename
            chat.saved_tokens = n_saved
            chat.swap_streak += 1
            chat.swaps_total += 1
            chat.slot = None
            self.swaps_out += 1
        return chat

    def swap_in(self, chat_id: str, slot: int) -> KvSwapChat:
        """Read the chat's KV back into ``slot``.

        ``restore`` accepts any free slot id, so a chat does not have to wait for the one
        it left. The caller must still re-issue with ``id_slot = slot`` and a resume prompt
        that satisfies :func:`resume_prompt_is_valid`.
        """
        chat = self.get(chat_id)
        if chat is None:
            raise KvSwapError(f"unknown chat {chat_id}")
        if chat.state != PAUSED or not chat.filename:
            raise KvSwapError(f"chat {chat_id} is {chat.state} with no checkpoint")
        with self._lock:
            chat.state = RESTORING
        started = time.monotonic()
        try:
            restored = self._post(slot, "restore", chat.filename)
        except Exception:
            with self._lock:
                chat.state = PAUSED
            raise
        n_restored = 0
        if isinstance(restored, dict):
            try:
                n_restored = int(restored.get("n_restored") or 0)
            except (TypeError, ValueError):
                n_restored = 0
        if n_restored != chat.saved_tokens:
            with self._lock:
                chat.state = PAUSED
            raise KvSwapError(
                f"chat {chat_id} restored {n_restored} of {chat.saved_tokens} tokens"
            )
        with self._lock:
            chat.state = RUNNING
            chat.slot = slot
            chat.last_restore_ms = (time.monotonic() - started) * 1000.0
            self.swaps_in += 1
        return chat

    def note_resume(self, chat_id: str, prompt_n: Optional[int]) -> None:
        """Record the prefill the resume actually cost. 1 means the checkpoint paid off."""
        with self._lock:
            chat = self._chats.get(chat_id)
            if chat is not None:
                chat.last_resume_prompt_n = prompt_n

    def note_progress(self, chat_id: str) -> None:
        """A chat that ran a full chunk without being swapped is no longer on a streak."""
        with self._lock:
            chat = self._chats.get(chat_id)
            if chat is not None:
                chat.swap_streak = 0

    def fall_back(self, chat_id: str) -> None:
        """Give up the checkpoint for this chat; resume will re-prefill instead."""
        with self._lock:
            chat = self._chats.get(chat_id)
            self.fallbacks += 1
            if chat is None:
                return
            filename = chat.filename
            chat.filename = None
            chat.saved_tokens = 0
            if chat.state in (PAUSED, RESTORING):
                chat.state = PAUSED
        if filename:
            self._discard(filename)

    def _discard(self, filename: str) -> None:
        if not self.save_dir or not filename:
            return
        try:
            Path(self.save_dir, filename).unlink()
        except OSError:
            pass

    def sweep(self) -> int:
        """Delete every checkpoint this controller still owns. Returns the count."""
        with self._lock:
            names = [c.filename for c in self._chats.values() if c.filename]
            for chat in self._chats.values():
                chat.filename = None
                chat.saved_tokens = 0
        for name in names:
            self._discard(name)
        return len(names)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "key": self.key,
                "n_ctx": self.n_ctx,
                "n_parallel": self.n_parallel,
                "buffer": self.buffer_tokens,
                "budget": self.budget,
                "resident": sum(c.resident for c in self._chats.values()),
                "running": sum(1 for c in self._chats.values() if c.state == RUNNING),
                "paused": sum(1 for c in self._chats.values() if c.state == PAUSED),
                "swaps_out": self.swaps_out,
                "swaps_in": self.swaps_in,
                "fallbacks": self.fallbacks,
            }


_CONTROLLERS: Dict[str, KvSwapController] = {}
_CONTROLLERS_LOCK = threading.Lock()


def get_kv_swap_controller(key: str, **kwargs) -> KvSwapController:
    """Process-wide registry, keyed like ``get_llama_admission_queue`` on the base URL.

    A model reload relaunches llama-server on a fresh ephemeral port, so a new key means a
    new controller and the old one is dropped along with its (already unlinked) files.
    """
    with _CONTROLLERS_LOCK:
        controller = _CONTROLLERS.get(key)
        if controller is None:
            controller = KvSwapController(key, **kwargs)
            _CONTROLLERS[key] = controller
        else:
            for name in ("n_ctx", "n_parallel", "draft_n_max", "save_dir"):
                if name in kwargs and kwargs[name] is not None:
                    setattr(controller, name, kwargs[name])
        return controller


def peek_kv_swap_controller(key: str) -> Optional[KvSwapController]:
    with _CONTROLLERS_LOCK:
        return _CONTROLLERS.get(key)


def reset_kv_swap_controllers() -> None:
    with _CONTROLLERS_LOCK:
        controllers = list(_CONTROLLERS.values())
        _CONTROLLERS.clear()
    for controller in controllers:
        controller.sweep()


def make_http_post(
    base_url: str,
    auth_headers: Optional[dict] = None,
    timeout: float = 30.0,
) -> Callable[..., object]:
    """Build the ``http_post`` a controller uses to drive ``/slots/{id}?action=...``.

    Kept out of :class:`KvSwapController` so the controller stays importable and testable
    without httpx, and so the tests can drive it with a fake server.
    """
    import httpx  # noqa: WPS433 - deferred exactly as the rest of this backend does

    def _post(*, slot: int, action: str, body: dict):
        response = httpx.post(
            f"{base_url}/slots/{int(slot)}",
            params = {"action": action},
            json = body or {},
            headers = auth_headers,
            timeout = timeout,
            trust_env = False,
        )
        if response.status_code != 200:
            raise KvSwapError(
                f"slot {slot} {action} returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise KvSwapError(f"slot {slot} {action} returned a non-JSON body") from exc

    return _post
