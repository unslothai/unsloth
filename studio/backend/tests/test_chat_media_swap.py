# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat image swap: order of operations, chat always restored, cancel, tool wiring."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.inference import chat_media_swap as swap
from core.inference import generate_image_tool as tool
from core.inference import media_chat_tools
from utils.account_context import OWNER

ORDER = [
    "chat_kv_saved",
    "chat_unloaded",
    "image_loaded",
    "image_generated",
    "image_unloaded",
    "chat_reloaded",
    "tool_result_returned",
]
KV = {"slots": [{"id": 0, "filename": "resume-x-slot0.bin", "n_saved": 42}]}
LAUNCH = {"kwargs": {"n_ctx": 4096, "n_gpu_layers": -1}, "cpu_fallback": False}


class Log(list):
    def add(self, *item):
        self.append(item)


class FakeChat:
    def __init__(
        self,
        log,
        *,
        fail = None,
        reload_failures = 0,
        kv = KV,
    ):
        self.log = log
        self.fail = fail or set()
        self.reload_failures = reload_failures
        self.kv = kv
        self.resident = True
        self.loaded_with = None
        self.discarded = []

    async def snapshot(self):
        if "snapshot_none" in self.fail:
            return None
        return swap.ChatSnapshot(
            identity = ("org/Qwen-GGUF", "Q4_K_M", "org/Qwen-GGUF"),
            launch = LAUNCH,
            model = "org/Qwen-GGUF",
            context_length = 4096,
        )

    async def save_kv(self, snap, should_abort):
        self.log.add("chat.save_kv", snap.identity)
        if "save_kv" in self.fail:
            raise RuntimeError("slot save failed")
        return self.kv

    async def unload(self, snap):
        self.log.add("chat.unload")
        if "unload_before" in self.fail:
            raise RuntimeError("another account is generating")
        self.resident = False
        if "unload_after" in self.fail:
            raise RuntimeError("evictor raised after the kill")

    async def is_resident(self, snap):
        return self.resident

    async def reload(self, snap, kv):
        self.log.add("chat.reload", snap.identity, snap.launch, kv)
        if self.reload_failures:
            self.reload_failures -= 1
            raise RuntimeError("llama-server did not start")
        self.resident = True
        self.loaded_with = (snap.identity, snap.launch, kv)

    async def discard_kv(self, kv):
        self.log.add("chat.discard_kv", kv)
        self.discarded.append(kv)


class FakeImage:
    def __init__(
        self,
        log,
        *,
        fail = None,
        chat = None,
        block = None,
    ):
        self.log = log
        self.fail = fail or set()
        self.chat = chat
        self.block = block
        self.loaded = False

    async def load(self, name, should_abort):
        assert self.chat is None or not self.chat.resident
        self.log.add("image.load", name)
        if self.block == "load":
            while not should_abort():
                await asyncio.sleep(0.01)
            raise swap.SwapStopped()
        if "load" in self.fail:
            raise RuntimeError("out of memory loading the image model")
        self.loaded = True

    async def generate(self, should_abort):
        self.log.add("image.generate")
        if self.block == "generate":
            while not should_abort():
                await asyncio.sleep(0.01)
            return swap.ERR_STOPPED
        if "generate" in self.fail:
            raise RuntimeError("render failed")
        return "Generated a 512x512 image (seed 1) and saved it as generated-image-1.png."

    async def unload(self):
        self.log.add("image.unload")
        if "unload" in self.fail:
            raise RuntimeError("unload failed")
        self.loaded = False


def _trip(chat, media, **kw):
    return swap.RoundTrip(chat = chat, media = media, image_model = "unsloth/sdxl-turbo", **kw)


def test_order_of_operations_and_same_settings():
    log = Log()
    chat = FakeChat(log)
    media = FakeImage(log, chat = chat)
    trip = _trip(chat, media)
    result = asyncio.run(trip.run())
    assert result.startswith("Generated a 512x512 image")
    assert trip.events == ORDER
    assert [entry[0] for entry in log] == [
        "chat.save_kv",
        "chat.unload",
        "image.load",
        "image.generate",
        "image.unload",
        "chat.reload",
    ]
    assert chat.loaded_with == (("org/Qwen-GGUF", "Q4_K_M", "org/Qwen-GGUF"), LAUNCH, KV)
    assert chat.resident and not media.loaded
    assert set(trip.timings) >= {
        "chat_kv_saved",
        "chat_unloaded",
        "image_loaded",
        "image_generated",
        "image_unloaded",
        "chat_reloaded",
        "total",
    }


@pytest.mark.parametrize("step", ["load", "generate"])
def test_media_failure_restores_chat(step):
    log = Log()
    chat = FakeChat(log)
    media = FakeImage(log, fail = {step}, chat = chat)
    trip = _trip(chat, media)
    result = asyncio.run(trip.run())
    assert result.startswith("Error: image generation failed:")
    assert chat.resident and chat.loaded_with[2] == KV
    assert log[-2][0] == "image.unload" and log[-1][0] == "chat.reload"


def test_image_unload_failure_still_restores_chat_and_returns_image():
    log = Log()
    chat = FakeChat(log)
    trip = _trip(chat, FakeImage(log, fail = {"unload"}, chat = chat))
    result = asyncio.run(trip.run())
    assert result.startswith("Generated")
    assert chat.resident


def test_kv_save_failure_still_swaps_without_kv():
    log = Log()
    chat = FakeChat(log, fail = {"save_kv"})
    trip = _trip(chat, FakeImage(log, chat = chat))
    assert asyncio.run(trip.run()).startswith("Generated")
    assert chat.loaded_with[2] is None


def test_refused_eviction_leaves_chat_alone():
    log = Log()
    chat = FakeChat(log, fail = {"unload_before"})
    media = FakeImage(log, chat = chat)
    trip = _trip(chat, media)
    result = asyncio.run(trip.run())
    assert result == "Error: image generation failed: another account is generating"
    assert [e[0] for e in log] == ["chat.save_kv", "chat.unload", "chat.discard_kv"]
    assert chat.discarded == [KV] and chat.resident


def test_eviction_that_raised_after_unloading_still_reloads():
    log = Log()
    chat = FakeChat(log, fail = {"unload_after"})
    trip = _trip(chat, FakeImage(log, chat = chat))
    result = asyncio.run(trip.run())
    assert result.startswith("Error: image generation failed")
    assert chat.resident and log[-1][0] == "chat.reload"
    assert not any(e[0].startswith("image.") for e in log)


def test_reload_retries_once_without_kv():
    log = Log()
    chat = FakeChat(log, reload_failures = 1)
    trip = _trip(chat, FakeImage(log, chat = chat))
    assert asyncio.run(trip.run()).startswith("Generated")
    reloads = [e for e in log if e[0] == "chat.reload"]
    assert [r[3] for r in reloads] == [KV, None]
    assert chat.resident


def test_reload_failing_twice_is_reported():
    log = Log()
    chat = FakeChat(log, reload_failures = 2)
    trip = _trip(chat, FakeImage(log, chat = chat))
    with pytest.raises(swap.ChatNotRestored):
        asyncio.run(trip.run())


def test_no_swappable_chat_changes_nothing():
    log = Log()
    chat = FakeChat(log, fail = {"snapshot_none"})
    with pytest.raises(swap.SwapUnavailable):
        asyncio.run(_trip(chat, FakeImage(log, chat = chat)).run())
    assert log == []


@pytest.mark.parametrize("where", ["load", "generate"])
def test_cancel_mid_swap_restores_chat(where):
    log = Log()
    chat = FakeChat(log)
    media = FakeImage(log, chat = chat, block = where)
    stopped = []
    trip = _trip(chat, media, on_cancel = lambda: stopped.append(True))

    async def go():
        task = asyncio.ensure_future(trip.run())
        while not any(e[0] == f"image.{where}" for e in log):
            await asyncio.sleep(0.01)
        trip.stop()
        return await task

    assert asyncio.run(go()) == swap.ERR_STOPPED
    assert stopped == [True]
    assert chat.resident and chat.loaded_with[2] == KV
    assert log[-2][0] == "image.unload" and log[-1][0] == "chat.reload"


def test_cancel_before_start_discards_kv():
    log = Log()
    chat = FakeChat(log)
    trip = _trip(chat, FakeImage(log, chat = chat))
    trip.cancel.set()
    assert asyncio.run(trip.run()) == swap.ERR_STOPPED
    assert [e[0] for e in log] == []


@pytest.fixture
def server_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target = loop.run_forever, daemon = True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def _patch_adapters(
    monkeypatch,
    log,
    chat_box,
    block = None,
):
    def make_chat(subject):
        chat_box["chat"] = FakeChat(log)
        return chat_box["chat"]

    def make_image(subject, render):
        return FakeImage(log, chat = chat_box["chat"], block = block)

    import core.inference.gpu_residency as residency
    import core.inference.media_switch_backends as backends

    monkeypatch.setattr(swap, "StudioChat", make_chat)
    monkeypatch.setattr(swap, "StudioImage", make_image)
    # Off on a GPU image device: the up-front swap, whatever this host or its settings say.
    monkeypatch.setattr(residency, "current_policy", lambda: "off")
    monkeypatch.setattr(backends, "load_takes_the_gpu", lambda: True)
    monkeypatch.setenv(swap.SWAP_MODEL_ENV, "unsloth/sdxl-turbo")
    from state import active_generations

    monkeypatch.setattr(active_generations, "count", lambda *a, **k: 1)


def test_run_in_tool_from_the_tool_thread(monkeypatch, server_loop):
    import time

    log, box = Log(), {}
    _patch_adapters(monkeypatch, log, box)
    result = swap.run_in_tool(
        loop = server_loop,
        account = OWNER,
        render = None,
        cancel_event = threading.Event(),
        render_limit_s = 30,
        stop_render = lambda: None,
    )
    assert result.startswith("Generated")
    assert box["chat"].resident


def test_run_in_tool_turn_cancel(monkeypatch, server_loop):
    import time

    log, box = Log(), {}
    _patch_adapters(monkeypatch, log, box, block = "generate")
    cancel = threading.Event()
    stopped = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    result = swap.run_in_tool(
        loop = server_loop,
        account = OWNER,
        render = None,
        cancel_event = cancel,
        render_limit_s = 30,
        stop_render = stopped.set,
    )
    assert result == swap.ERR_STOPPED
    assert stopped.is_set()
    assert box["chat"].resident and log[-1][0] == "chat.reload"


def test_run_in_tool_timeout(monkeypatch, server_loop):
    import time

    log, box = Log(), {}
    _patch_adapters(monkeypatch, log, box, block = "load")
    result = swap.run_in_tool(
        loop = server_loop,
        account = OWNER,
        render = None,
        cancel_event = None,
        render_limit_s = 0.2,
        stop_render = lambda: None,
        swap_budget_s = 0.2,
    )
    assert result == "Error: image generation took too long and was stopped."
    assert box["chat"].resident


def test_render_limit_counts_from_the_render(monkeypatch, server_loop):
    """A slow image load does not eat the render's timeout: only the render is held to it."""
    log, box = Log(), {}
    _patch_adapters(monkeypatch, log, box, block = "generate")
    real_load = FakeImage.load

    async def slow_load(self, name, should_abort):
        await asyncio.sleep(0.6)
        return await real_load(self, name, should_abort)

    monkeypatch.setattr(FakeImage, "load", slow_load)
    t0 = time.monotonic()
    result = swap.run_in_tool(
        loop = server_loop,
        account = OWNER,
        render = None,
        cancel_event = None,
        render_limit_s = 0.4,
        stop_render = lambda: None,
        swap_budget_s = 30,
    )
    assert result == "Error: image generation took too long and was stopped."
    # Load 0.6 s, then 0.4 s of render: stopped after both, not at 0.4 s.
    assert time.monotonic() - t0 >= 0.95
    assert [e[0] for e in log].index("image.generate") > 0 and box["chat"].resident


def test_run_in_tool_refuses_beside_another_chat(monkeypatch, server_loop):
    import time

    log, box = Log(), {}
    _patch_adapters(monkeypatch, log, box)
    from state import active_generations

    monkeypatch.setattr(active_generations, "count", lambda *a, **k: 2)
    result = swap.run_in_tool(
        loop = server_loop,
        account = OWNER,
        render = None,
        cancel_event = None,
        render_limit_s = 5,
        stop_render = lambda: None,
    )
    assert result.startswith("Error: another chat is generating")
    assert log == []


def test_run_in_tool_without_default_model(monkeypatch, server_loop):
    import time

    monkeypatch.delenv(swap.SWAP_MODEL_ENV, raising = False)
    monkeypatch.setattr(swap, "default_image_model", lambda: None)
    result = swap.run_in_tool(
        loop = server_loop,
        account = OWNER,
        render = None,
        cancel_event = None,
        render_limit_s = 5,
        stop_render = lambda: None,
    )
    assert result == swap.ERR_NO_DEFAULT


def test_tool_uses_swap_only_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(media_chat_tools, "server_loop", lambda: asyncio.new_event_loop())
    monkeypatch.setattr(media_chat_tools, "media_chat_status", lambda task: None)
    calls = []
    monkeypatch.setattr(swap, "run_in_tool", lambda **kw: calls.append(kw) or "swapped")
    monkeypatch.setenv(swap.SWAP_ENV, "0")
    assert (
        tool.run_generate_image({"prompt": "a cat"}, workdir = str(tmp_path)) == tool.ERR_NOT_LOADED
    )
    assert calls == []
    monkeypatch.setenv(swap.SWAP_ENV, "1")
    assert tool.run_generate_image({"prompt": "a cat"}, workdir = str(tmp_path)) == "swapped"
    assert len(calls) == 1 and callable(calls[0]["render"]) and callable(calls[0]["stop_render"])


def test_tool_offered_when_swap_can_load_default(monkeypatch):
    monkeypatch.setattr(media_chat_tools, "media_chat_status", lambda task: None)
    monkeypatch.setattr(swap, "chat_can_swap", lambda: False)
    assert media_chat_tools.offered_media_chat_tools({"generate_image"}) == []
    monkeypatch.setattr(swap, "chat_can_swap", lambda: True)
    offered = media_chat_tools.offered_media_chat_tools({"generate_image", "generate_video"})
    assert [t["function"]["name"] for t in offered] == ["generate_image"]


def test_default_image_model_order(monkeypatch):
    monkeypatch.delenv(swap.SWAP_MODEL_ENV, raising = False)
    monkeypatch.setattr(swap, "_last_image_model", None)
    import utils.openai_auto_switch_settings as settings

    monkeypatch.setattr(settings, "_cached_setting", lambda key, default, *a: None)
    import core.inference.media_model_index as index

    monkeypatch.setattr(index, "available_media_model_ids", lambda task: ["only/one"])
    assert swap.default_image_model() == "only/one"
    swap.remember_image_model("last/used")
    assert swap.default_image_model() == "last/used"
    monkeypatch.setattr(settings, "_cached_setting", lambda key, default, *a: "set/in-settings")
    assert swap.default_image_model() == "set/in-settings"
    monkeypatch.setenv(swap.SWAP_MODEL_ENV, "env/model")
    assert swap.default_image_model() == "env/model"


def test_request_gate_held_until_image_is_gone():
    """New chat requests wait while chat is away; the gate is freed before the reload."""
    import contextlib

    log = Log()
    held = {"now": False}

    @contextlib.asynccontextmanager
    async def gate():
        held["now"] = True
        log.add("gate.enter")
        try:
            yield
        finally:
            held["now"] = False
            log.add("gate.exit")

    chat = FakeChat(log)
    media = FakeImage(log, chat = chat)
    seen = {}
    for name in ("load", "generate", "unload"):
        real = getattr(media, name)

        def wrap(
            *a,
            _real = real,
            _name = name,
        ):
            seen[_name] = held["now"]
            return _real(*a)

        setattr(media, name, wrap)
    real_reload = chat.reload

    async def reload(snap, kv):
        seen["reload"] = held["now"]
        await real_reload(snap, kv)

    chat.reload = reload
    trip = _trip(chat, media, hold_requests = gate)
    assert asyncio.run(trip.run()).startswith("Generated")
    assert seen == {"load": True, "generate": True, "unload": True, "reload": False}
    kinds = [e[0] for e in log]
    assert kinds.index("gate.enter") < kinds.index("chat.save_kv")
    assert kinds.index("image.unload") < kinds.index("gate.exit") < kinds.index("chat.reload")
    assert kinds.count("gate.exit") == 1


class ArbiterImage(FakeImage):
    """fit / always: chat kept when both fit, else evicted (fit) or the load refused (always)."""

    def __init__(
        self,
        log,
        chat,
        *,
        fits,
        policy = "fit",
        **kw,
    ):
        super().__init__(log, **kw)
        self.arb_chat = chat
        self.fits = fits
        self.policy = policy

    async def load(self, name, should_abort):
        self.log.add("image.load", name)
        if not self.fits:
            if self.policy == "always":
                raise RuntimeError("Keep multiple models loaded is set to Always")
            self.arb_chat.resident = False
        self.loaded = True


def test_fit_policy_keeps_both_when_they_fit():
    log = Log()
    chat = FakeChat(log)
    media = ArbiterImage(log, chat, fits = True)
    trip = _trip(chat, media, beside_first = True)
    assert asyncio.run(trip.run()).startswith("Generated")
    kinds = [e[0] for e in log]
    assert "chat.unload" not in kinds and "chat.reload" not in kinds and "image.unload" not in kinds
    assert chat.discarded == [KV] and chat.resident and media.loaded
    assert "image_kept_beside_chat" in trip.events


def test_fit_policy_swaps_when_the_arbiter_evicts_chat():
    log = Log()
    chat = FakeChat(log)
    media = ArbiterImage(log, chat, fits = False)
    trip = _trip(chat, media, beside_first = True)
    assert asyncio.run(trip.run()).startswith("Generated")
    kinds = [e[0] for e in log]
    assert kinds == ["chat.save_kv", "image.load", "image.generate", "image.unload", "chat.reload"]
    assert chat.loaded_with[2] == KV and not media.loaded
    assert trip.events[:3] == ["chat_kv_saved", "image_loaded", "chat_unloaded"]


def test_always_policy_refusal_leaves_chat_alone():
    log = Log()
    chat = FakeChat(log)
    media = ArbiterImage(log, chat, fits = False, policy = "always")
    trip = _trip(chat, media, beside_first = True)
    result = asyncio.run(trip.run())
    assert result == "Error: image generation failed: Keep multiple models loaded is set to Always"
    assert chat.resident and chat.discarded == [KV]
    assert "chat.reload" not in [e[0] for e in log]


@pytest.mark.parametrize(
    "policy, on_gpu, beside",
    [("off", True, False), ("fit", True, True), ("always", True, True), ("off", False, True)],
)
def test_tier_choice(monkeypatch, server_loop, policy, on_gpu, beside):
    """Off swaps up front; When they fit / Always and a CPU image device try beside chat first."""
    import core.inference.gpu_residency as residency
    import core.inference.media_switch_backends as backends

    log, box = Log(), {}
    _patch_adapters(monkeypatch, log, box)
    monkeypatch.setattr(residency, "current_policy", lambda: policy)
    monkeypatch.setattr(backends, "load_takes_the_gpu", lambda: on_gpu)
    seen = {}
    real = swap.RoundTrip.run

    async def spy(self):
        seen["beside"] = self.beside_first
        return await real(self)

    monkeypatch.setattr(swap.RoundTrip, "run", spy)
    if beside:
        # The fake image load asserts chat is gone; beside chat it stays, so lift that check.
        monkeypatch.setattr(FakeImage, "load", ArbiterImage.load)
        monkeypatch.setattr(FakeImage, "fits", True, raising = False)
    swap.run_in_tool(
        loop = server_loop,
        account = OWNER,
        render = None,
        cancel_event = None,
        render_limit_s = 5,
        stop_render = lambda: None,
    )
    assert seen["beside"] is beside


@pytest.mark.parametrize(
    "load_policy, on_gpu, refused",
    [
        ("never", True, True),
        ("never", False, True),
        ("beside_only", True, True),
        ("beside_only", False, False),
        ("bogus", False, True),
        ("any", True, False),
    ],
)
def test_the_credential_load_policy_gates_the_swap(
    monkeypatch, server_loop, load_policy, on_gpu, refused
):
    """An inference-only key may neither load nor unload; the chat tools key only loads beside chat."""
    import dataclasses

    import core.inference.media_switch_backends as backends

    log, box = Log(), {}
    _patch_adapters(monkeypatch, log, box)
    monkeypatch.setattr(backends, "load_takes_the_gpu", lambda: on_gpu)
    if not on_gpu:
        monkeypatch.setattr(FakeImage, "load", ArbiterImage.load)
        monkeypatch.setattr(FakeImage, "fits", True, raising = False)
    result = swap.run_in_tool(
        loop = server_loop,
        account = dataclasses.replace(OWNER, load_policy = load_policy),
        render = None,
        cancel_event = None,
        render_limit_s = 5,
        stop_render = lambda: None,
    )
    if refused:
        assert result == swap.ERR_LOAD_POLICY and log == []
    else:
        assert result.startswith("Generated"), result


class _Engine:
    def __init__(self, loading = ()):
        self.unloads = 0
        self.loading = tuple(loading)
        self.is_loaded = False

    def unload(self):
        self.unloads += 1

    def loading_repo_ids(self):
        return self.loading


def _studio_image_unload(
    monkeypatch,
    *,
    admitted,
    status,
    loading = (),
):
    from core.inference import diffusion_engine_router as router
    from core.inference import flux3_prompt_upsampler as pup
    from core.inference import gpu_arbiter
    from core.inference import media_switch_backends as backends
    from core.inference.media_model_index import MediaModelPick

    engine = _Engine(loading)
    monkeypatch.setattr(router, "get_active_diffusion_engine", lambda: engine)
    monkeypatch.setattr(pup, "stop_for_handover", lambda: None)
    monkeypatch.setattr(gpu_arbiter, "release_if", lambda *a, **k: None)
    monkeypatch.setattr(
        backends,
        "backend_for",
        lambda owner: type("B", (), {"status": staticmethod(lambda: status)})(),
    )
    image = swap.StudioImage("unsloth", render = None)
    if admitted:
        image._admitted = (
            "unsloth/sdxl-turbo",
            MediaModelPick("unsloth/sdxl-turbo", "unsloth/sdxl-turbo"),
        )
    asyncio.run(image.unload())
    return engine.unloads


def test_cleanup_unloads_only_the_load_this_swap_admitted(monkeypatch):
    ours = {"loaded": True, "repo_id": "unsloth/sdxl-turbo"}
    other = {"loaded": True, "repo_id": "black-forest-labs/FLUX.2-klein-4B"}
    # The gated route refused the load (another account's model is generating): nothing of ours to unload.
    assert _studio_image_unload(monkeypatch, admitted = False, status = other) == 0
    # Admitted, but another model is resident or loading now: left alone.
    assert _studio_image_unload(monkeypatch, admitted = True, status = other) == 0
    assert (
        _studio_image_unload(monkeypatch, admitted = True, status = {}, loading = ("someone/else",)) == 0
    )
    # Our model resident, or our own load failed half way: unloaded.
    assert _studio_image_unload(monkeypatch, admitted = True, status = ours) == 1
    assert _studio_image_unload(monkeypatch, admitted = True, status = {"loaded": False}) == 1


def test_tool_description_unchanged_while_swap_is_off(monkeypatch):
    """With the swap off, the schema text matches the one before the swap existed."""
    from core.inference import generate_image_tool as tool

    monkeypatch.setattr(swap, "swap_enabled", lambda: False)
    off = tool.generate_image_schema({})["function"]["description"]
    assert off == (
        "Generate one image with the image model loaded in Unsloth Studio. Write the full prompt "
        "yourself from the user's request: subject, setting, style, lighting and composition. Steps "
        "and guidance follow the loaded model's own defaults; the size defaults to 1024x1024. It "
        "never loads or unloads a model."
    )
    monkeypatch.setattr(swap, "swap_enabled", lambda: True)
    on = tool.generate_image_schema({})["function"]["description"]
    assert "never loads or unloads" not in on
