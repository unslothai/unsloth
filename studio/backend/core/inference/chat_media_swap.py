# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image generation from a chat turn when the chat model and the image model cannot both be loaded.

Opt-in (Settings key ``chat_image_swap``, or ``UNSLOTH_CHAT_IMAGE_SWAP=1``). Runs between two rounds
of one chat turn, after the model's ``generate_image`` call is complete and before the follow-up
generation starts, so nothing is streaming from llama-server while it is down:

1. save the chat KV (llama-server slots) to disk;
2. take the GPU for the image model with the arbiter's own chat evictor;
3. load the image model, generate;
4. unload the image model;
5. reload the chat model with the launch it had (the idle-unload replay path) and restore the KV;
6. hand the tool result back to the turn.

Every failure after step 2 still runs steps 4 and 5, so the turn always gets its chat model back.
The orchestration (``RoundTrip``) only sees two small adapters, so it is tested with fakes.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

SWAP_SETTING_KEY = "chat_image_swap"
SWAP_MODEL_SETTING_KEY = "chat_image_swap_model"
SWAP_ENV = "UNSLOTH_CHAT_IMAGE_SWAP"
SWAP_MODEL_ENV = "UNSLOTH_CHAT_IMAGE_SWAP_MODEL"

ERR_STOPPED = "Error: image generation was stopped."
ERR_NO_DEFAULT = (
    "Error: no image model is loaded and no default image model is set for chat. Pick one on the "
    "Images page once, or set one in Settings."
)

# Loading the image model and loading chat back, on top of the render's own limit.
SWAP_BUDGET_S = 900.0

# The image model the Images page or the API loaded last in this Studio session.
_last_image_model: Optional[str] = None


class SwapUnavailable(RuntimeError):
    """The swap cannot start; nothing was changed. The message is the tool result."""


class SwapStopped(RuntimeError):
    pass


class ChatNotRestored(RuntimeError):
    pass


def _env_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def swap_enabled() -> bool:
    env = _env_bool(SWAP_ENV)
    if env is not None:
        return env
    try:
        from utils.openai_auto_switch_settings import _cached_setting, _coerce_bool
        return bool(_coerce_bool(_cached_setting(SWAP_SETTING_KEY, None)))
    except Exception:  # noqa: BLE001 - an unreadable setting is the default, off
        return False


def remember_image_model(model_id: Optional[str]) -> None:
    global _last_image_model
    if isinstance(model_id, str) and model_id.strip():
        _last_image_model = model_id.strip()


def default_image_model() -> Optional[str]:
    """The setting, else the image model loaded last this session, else the only downloaded one."""
    env = os.environ.get(SWAP_MODEL_ENV)
    if env and env.strip():
        return env.strip()
    try:
        from utils.openai_auto_switch_settings import _cached_setting
        stored = _cached_setting(SWAP_MODEL_SETTING_KEY, None)
        if isinstance(stored, str) and stored.strip():
            return stored.strip()
    except Exception:  # noqa: BLE001
        pass
    if _last_image_model:
        return _last_image_model
    try:
        from core.inference.media_model_index import IMAGE_TASK, available_media_model_ids
        ids = list(available_media_model_ids(IMAGE_TASK) or ())
    except Exception:  # noqa: BLE001
        ids = []
    return ids[0] if len(ids) == 1 else None


@dataclass
class ChatSnapshot:
    """What brings the chat model back exactly as it was launched."""

    identity: Any
    launch: Any
    model: Optional[str] = None
    context_length: Optional[int] = None
    extra: dict = field(default_factory = dict)


@dataclass
class RoundTrip:
    """The swap, in order. ``chat`` and ``media`` are adapters (Studio's, or fakes in tests).

    chat:  snapshot() -> ChatSnapshot | None, save_kv(snap, should_abort) -> manifest | None,
           unload(snap), is_resident(snap) -> bool, reload(snap, manifest), discard_kv(manifest)
    media: load(name, should_abort), generate(should_abort) -> str, unload()
    All are coroutines."""

    chat: Any
    media: Any
    image_model: str
    cancel: threading.Event = field(default_factory = threading.Event)
    events: list = field(default_factory = list)
    timings: dict = field(default_factory = dict)
    on_cancel: Optional[Callable[[], None]] = None
    # When the render began: the tool's own timeout covers the render, not the swap around it.
    render_started: Optional[float] = None
    # Held from the KV save until the image model is gone, so a chat request that arrives meanwhile
    # (another tab, the next message after a Stop) waits for chat to come back instead of finding
    # no model. Released before the reload, which takes the same gate itself.
    hold_requests: Optional[Callable[[], Any]] = None
    # When they fit / Always: load the image model beside chat first and swap only if the arbiter
    # could not keep both (Off evicts chat up front).
    beside_first: bool = False

    def stop(self) -> None:
        self.cancel.set()
        if self.on_cancel is not None:
            try:
                self.on_cancel()
            except Exception:  # noqa: BLE001
                pass

    def _note(
        self,
        event: str,
        started: Optional[float] = None,
        **info,
    ) -> None:
        if started is not None:
            self.timings[event] = round(time.monotonic() - started, 3)
        self.events.append(event)
        logger.info(
            "chat image swap: %s%s%s",
            event,
            f" in {self.timings[event]:.2f}s" if started is not None else "",
            "".join(f" {k}={v}" for k, v in info.items()),
        )

    def _check(self) -> None:
        if self.cancel.is_set():
            raise SwapStopped()

    async def _step(self, name: str, coro: Awaitable, **info):
        started = time.monotonic()
        result = await coro
        self._note(name, started, **info)
        return result

    async def run(self) -> str:
        total = time.monotonic()
        snap = await self.chat.snapshot()
        if snap is None:
            raise SwapUnavailable(
                "Error: the chat model cannot be swapped out for the image model "
                "(only a GGUF chat model loaded on its own can be)."
            )
        if self.cancel.is_set():
            return ERR_STOPPED
        gate = contextlib.AsyncExitStack()
        if self.hold_requests is not None:
            try:
                await gate.enter_async_context(self.hold_requests())
            except asyncio.CancelledError:
                if self.cancel.is_set():
                    return ERR_STOPPED
                raise
        try:
            return await self._run_held(snap, gate, total)
        finally:
            await gate.aclose()

    async def _run_held(self, snap: ChatSnapshot, gate, total: float) -> str:
        try:
            kv = await self._step("chat_kv_saved", self.chat.save_kv(snap, self.cancel.is_set))
        except Exception as exc:  # noqa: BLE001 - without it the follow-up reprocesses the prompt
            logger.info("chat image swap: KV save failed: %s", exc)
            kv = None
        if self.cancel.is_set():
            await self.chat.discard_kv(kv)
            return ERR_STOPPED
        unloaded = False
        media_touched = False
        result: Optional[str] = None
        error: Optional[BaseException] = None
        beside = False
        try:
            if not self.beside_first:
                try:
                    await self._step("chat_unloaded", self.chat.unload(snap), model = snap.model)
                finally:
                    # A failed eviction may still have taken chat down: then it comes back too.
                    unloaded = not await self.chat.is_resident(snap)
                self._check()
            media_touched = True
            try:
                await self._step(
                    "image_loaded",
                    self.media.load(self.image_model, self.cancel.is_set),
                    model = self.image_model,
                )
            finally:
                if self.beside_first:
                    # The user's When they fit / Always policy placed it: beside chat when both fit,
                    # else the arbiter evicted chat for it (fit) or refused the load (always).
                    unloaded = not await self.chat.is_resident(snap)
                    beside = not unloaded
                    if unloaded:
                        self._note("chat_unloaded", model = snap.model, by = "arbiter")
            self._check()
            self.render_started = time.monotonic()
            result = await self._step("image_generated", self.media.generate(self.cancel.is_set))
        except SwapStopped:
            result = ERR_STOPPED
        except BaseException as exc:  # noqa: BLE001 - every failure still restores chat
            error = exc
        finally:
            if beside and error is None:
                # Both fit: the image model stays loaded as the policy keeps it; the next call is free.
                media_touched = False
                self._note("image_kept_beside_chat")
            if media_touched:
                try:
                    await self._step("image_unloaded", self.media.unload())
                except Exception as exc:  # noqa: BLE001 - chat reload evicts it anyway
                    logger.warning("chat image swap: image unload failed: %s", exc)
            await gate.aclose()
            if unloaded:
                await self._restore(snap, kv)
            else:
                await self.chat.discard_kv(kv)
        self.timings["total"] = round(time.monotonic() - total, 3)
        if error is not None:
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise error
            return f"Error: image generation failed: {error}"
        self._note("tool_result_returned")
        return result if result is not None else "Error: the image model returned no image."

    async def _restore(self, snap: ChatSnapshot, kv) -> None:
        """Reload chat as it was; one retry, then the turn is told it has no chat model."""
        last: Optional[BaseException] = None
        for attempt in range(2):
            started = time.monotonic()
            try:
                await self.chat.reload(snap, kv if attempt == 0 else None)
                self._note("chat_reloaded", started, model = snap.model, kv = bool(kv and attempt == 0))
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                logger.warning(
                    "chat image swap: chat reload attempt %d failed: %s", attempt + 1, exc
                )
        raise ChatNotRestored(f"the chat model could not be loaded back: {last}")


# ---------------------------------------------------------------- Studio adapters


class StudioChat:
    """The primary llama.cpp chat model. Never the route unload: that refuses (or cancels) this
    very turn's generation. The arbiter's evictor is the same one an Images page load uses."""

    def __init__(self, subject: str):
        self.subject = subject

    @staticmethod
    def _llama():
        from routes.inference import get_llama_cpp_backend
        return get_llama_cpp_backend()

    async def snapshot(self) -> Optional[ChatSnapshot]:
        from core.inference import model_slots
        from core.inference.llama_keepwarm import launch_to_replay, loaded_identity

        llama = self._llama()
        if not llama.is_loaded or getattr(llama, "holds_no_vram", False):
            return None
        if list(model_slots.slots):
            # Kept extra chat models would be evicted with it and not put back.
            raise SwapUnavailable(
                "Error: other chat models are kept loaded beside this one; unload them to let chat "
                "swap to the image model."
            )
        from core.inference import gpu_arbiter
        from utils.account_context import current_account_id

        owner = gpu_arbiter.owner_account(gpu_arbiter.CHAT)
        if owner is not None and owner != current_account_id():
            raise SwapUnavailable(
                "Error: the chat model was loaded by another account, so this chat cannot unload it "
                "for the image model."
            )
        identity = loaded_identity()
        launch = launch_to_replay(llama)
        if identity is None or launch is None:
            return None
        return ChatSnapshot(
            identity = identity,
            launch = launch,
            model = getattr(llama, "model_identifier", None),
            context_length = getattr(llama, "_effective_context_length", None),
        )

    async def save_kv(self, snap: ChatSnapshot, should_abort) -> Optional[dict]:
        llama = self._llama()
        try:
            return await asyncio.to_thread(llama.save_slots_for_resume, should_abort)
        except Exception as exc:  # noqa: BLE001 - a cold prompt reprocess still works
            logger.info("chat image swap: KV save skipped: %s", exc)
            return None

    async def unload(self, snap: ChatSnapshot) -> None:
        from core.inference import gpu_arbiter
        from utils.account_context import current_account_id

        # A claim with nothing registered: evicts every other resident (chat) under "off", whatever the
        # user's Keep multiple models policy, and raises if another account is generating.
        await asyncio.to_thread(
            gpu_arbiter.acquire_for,
            gpu_arbiter.DIFFUSION,
            None,
            policy = "off",
            account_id = current_account_id(),
        )
        if self._llama().is_active:
            raise RuntimeError("the chat model did not unload")

    async def is_resident(self, snap: ChatSnapshot) -> bool:
        from core.inference.llama_keepwarm import loaded_identity
        return self._llama().is_loaded and loaded_identity() == snap.identity

    async def reload(self, snap: ChatSnapshot, kv: Optional[dict]) -> None:
        from core.inference.llama_keepwarm import (
            end_in_process_request,
            loaded_identity,
            stash_for_reload,
            stash_unload_kv,
        )
        from routes.inference import (
            _RELOAD_ONLY_MODEL,
            _maybe_auto_switch_model,
            allow_model_switch_for_request,
        )

        stash_for_reload(snap.identity, snap.launch)
        if kv:
            # The next load of the same GGUF restores it (llama_keepwarm.note_model_loaded).
            stash_unload_kv(kv)
        request = _in_process_request()
        allow_model_switch_for_request(request.scope)
        answered = False
        try:
            await _maybe_auto_switch_model(_RELOAD_ONLY_MODEL, request, self.subject)
            answered = True
        finally:
            end_in_process_request(request.scope, answered = answered)
        if loaded_identity() != snap.identity:
            raise RuntimeError(f"reloaded {loaded_identity()!r}, expected {snap.identity!r}")

    async def discard_kv(self, kv: Optional[dict]) -> None:
        if kv:
            from core.inference.llama_keepwarm import _delete_resume_files
            _delete_resume_files(kv)


def _in_process_request():
    import sys

    from starlette.requests import Request

    from routes.inference import disable_auto_download_for_request

    app = getattr(sys.modules.get("main"), "app", None)
    path = "/api/inference/chat-image-swap"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
        "app": app,
        "state": {"generation_cancel_event": threading.Event(), "api_monitor_via_api_key": False},
    }
    disable_auto_download_for_request(scope)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


class StudioImage:
    """The image model, loaded and unloaded the way the media auto-switch and the unload route do."""

    def __init__(self, subject: str, render: Callable[[Callable[[], bool]], Awaitable[str]]):
        self.subject = subject
        self._render = render
        self.load_timeout_s = 600.0

    async def load(self, name: str, should_abort) -> None:
        from core.inference.gpu_arbiter import DIFFUSION
        from core.inference.media_auto_switch import (
            _resolve_pick,
            _retired_copy_as_base,
            _start_load,
        )
        from core.inference.media_locality import is_edit_only
        from core.inference.media_model_index import IMAGE_TASK, resident_is_pick
        from core.inference.media_switch_backends import backend_for

        pick = await asyncio.to_thread(_resolve_pick, DIFFUSION, name, IMAGE_TASK)
        if pick is None and os.path.isfile(os.path.join(name, "model_index.json")):
            # A diffusers folder named by path in the setting: loaded as the Images page loads it.
            from core.inference.media_model_index import MediaModelPick
            pick = MediaModelPick(name, name, None, "pipeline")
        if pick is None:
            raise RuntimeError(f"no downloaded image model matches '{name}'")
        pick = _retired_copy_as_base(pick)
        if await asyncio.to_thread(is_edit_only, pick):
            raise RuntimeError(f"'{pick.model_id}' only edits images")
        await _start_load(DIFFUSION, pick, self.subject)
        backend = backend_for(DIFFUSION)
        deadline = time.monotonic() + self.load_timeout_s
        while True:
            if should_abort():
                raise SwapStopped()
            progress = await asyncio.to_thread(backend.load_progress) or {}
            phase = progress.get("phase")
            if phase == "error":
                raise RuntimeError(progress.get("error") or "the image model failed to load")
            if phase in (None, "ready"):
                status = await asyncio.to_thread(backend.status)
                if resident_is_pick(status, name, pick):
                    remember_image_model(pick.model_id)
                    return
                raise RuntimeError(f"'{pick.model_id}' was replaced by another load")
            if time.monotonic() >= deadline:
                raise RuntimeError("the image model took too long to load")
            await asyncio.sleep(0.25)

    async def generate(self, should_abort) -> str:
        return await self._render(should_abort)

    async def unload(self) -> None:
        from core.inference import flux3_prompt_upsampler as pup
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        from core.inference.gpu_arbiter import DIFFUSION, release_if

        engine = get_active_diffusion_engine()
        await asyncio.to_thread(engine.unload)
        await asyncio.to_thread(pup.stop_for_handover)
        await asyncio.to_thread(
            release_if, DIFFUSION, lambda: not engine.loading_repo_ids() and not engine.is_loaded
        )


def chat_can_swap() -> bool:
    """Cheap check for offering the tool (runs on the event loop, so no model index scan): swap on
    and a GGUF chat model loaded. A missing default image model is the tool's own error."""
    if not swap_enabled():
        return False
    try:
        from routes.inference import get_llama_cpp_backend
        return bool(get_llama_cpp_backend().is_loaded)
    except Exception:  # noqa: BLE001
        return False


def run_in_tool(
    *,
    loop: asyncio.AbstractEventLoop,
    account,
    render: Callable[[Callable[[], bool]], Awaitable[str]],
    cancel_event,
    render_limit_s: float,
    stop_render: Callable[[], None],
    swap_budget_s: float = SWAP_BUDGET_S,
) -> str:
    """Runs the round trip on the server loop from the tool's thread; polls the turn's cancel.
    ``render_limit_s`` is the tool's timeout, counted from the render's start; the loads around it
    get ``swap_budget_s`` on top."""
    import concurrent.futures

    from state import active_generations
    from utils.account_context import arun_as

    name = default_image_model()
    if name is None:
        return ERR_NO_DEFAULT
    if active_generations.count() > 1:
        # Another chat is streaming from the same llama-server: unloading it would end that one.
        return "Error: another chat is generating on this model. Try again when it finishes."
    subject = getattr(account, "username", None) or "unsloth"
    from core.inference.gpu_residency import current_policy
    from core.inference.llama_keepwarm import inference_lifecycle_gate

    from core.inference.media_switch_backends import load_takes_the_gpu

    policy = current_policy()
    # A CPU image device (sd.cpp or diffusers on the CPU) never takes chat's GPU: load it beside.
    on_gpu = load_takes_the_gpu()
    trip = RoundTrip(
        chat = StudioChat(subject),
        media = StudioImage(subject, render),
        image_model = name,
        on_cancel = stop_render,
        hold_requests = inference_lifecycle_gate,
        beside_first = policy in ("fit", "always") or not on_gpu,
    )

    async def _go():
        try:
            return await trip.run()
        except SwapUnavailable as exc:
            return str(exc)
        except ChatNotRestored as exc:
            return f"Error: {exc}"

    future = asyncio.run_coroutine_threadsafe(arun_as(account, _go()), loop)
    started = time.monotonic()
    timed_out = False

    def _late() -> bool:
        now = time.monotonic()
        if trip.render_started is not None and now - trip.render_started >= render_limit_s:
            return True
        return now - started >= render_limit_s + swap_budget_s

    while not future.done():
        if not trip.cancel.is_set():
            if cancel_event is not None and cancel_event.is_set():
                trip.stop()
            elif _late():
                timed_out = True
                # Never cancel the future: the restore must run to the end.
                trip.stop()
        concurrent.futures.wait([future], timeout = 0.25)
    result = future.result()
    if timed_out and result == ERR_STOPPED:
        result = "Error: image generation took too long and was stopped."
    logger.info("chat image swap: timings %s", trip.timings)
    return result
