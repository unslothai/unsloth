# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A flow with two chat models under Keep multiple models loaded: the second loads alongside the first and each
request names the model it means; with the setting off nothing changes."""

import asyncio
from types import SimpleNamespace

import pytest

from core.flows import engine as engine_mod
from core.flows.engine import GpuLease, NeedsExclusiveGpu
from core.flows.nodes import _common
from core.flows.registry import FlowNodeError
from studio_mcp import loading

BIG = "unsloth/Qwen3-14B-GGUF"
SMALL = "unsloth/Qwen3-4B-Instruct-2507-GGUF"


def _fakes(
    monkeypatch,
    mode,
    serving,
    kept_variant = None,
    refuse = None,
):
    loads = []
    state = {"serving": list(serving)}

    async def fake_load_llm(caller, ctx, **kwargs):
        if refuse:
            raise RuntimeError(refuse)
        loads.append(kwargs)
        if kwargs.get("alongside"):
            state["serving"].append(kwargs["model"])
        else:
            state["serving"] = [kwargs["model"]]

    async def fake_slots(caller):
        return {"llm": state["serving"]}

    async def fake_resident(caller):
        status = {"gguf_variant": None, "is_gguf": True, "serving": state["serving"]}
        if mode is not None:
            status["multi_model_mode"] = mode
        return {"llm": state["serving"]}, status

    async def fake_read_json(caller, path):
        assert path.startswith("/api/inference/status?model=")
        return {"gguf_variant": kept_variant}

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", fake_resident)
    monkeypatch.setattr(loading, "_read_json", fake_read_json)
    monkeypatch.setattr(_common, "_caller", lambda ctx: None)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(_common, "_LOADED_OPTIONS", {})
    monkeypatch.setattr(_common, "_ALONGSIDE_OPTIONS", {})
    return loads, state


class _Ctx:
    def __init__(self):
        self.resident, self.spec, self.gpu_shared = {}, {}, False

    def set_status(
        self,
        status,
        message = None,
    ):
        pass

    def on_cancel(self, fn):
        pass


async def _body(ctx):
    return await _common._chat_body(
        ctx,
        system = "s",
        prompt = "p",
        temperature = 0.5,
        max_tokens = 8,
        images = None,
        thinking = False,
        seed = None,
        stream = False,
    )


def test_off_replaces_the_primary_and_names_the_default_model(monkeypatch):
    loads, state = _fakes(monkeypatch, "off", [BIG])
    ctx = _Ctx()
    asyncio.run(_common.ensure_model(ctx, "chat", {"repo": SMALL}))
    assert "alongside" not in loads[0] and state["serving"] == [SMALL]
    assert asyncio.run(_body(ctx))["model"] == "default"
    loads, _ = _fakes(monkeypatch, None, [BIG])
    asyncio.run(_common.ensure_model(_Ctx(), "chat", {"repo": SMALL}))
    assert "alongside" not in loads[0]


@pytest.mark.parametrize("mode", ["fit", "always"])
def test_a_second_model_loads_alongside_and_requests_name_it(monkeypatch, mode):
    loads, state = _fakes(monkeypatch, mode, [BIG])
    ctx = _Ctx()
    asyncio.run(_common.ensure_model(ctx, "chat", {"repo": SMALL, "variant": "Q4_K_M"}))
    assert len(loads) == 1 and loads[0]["alongside"] is True
    assert state["serving"] == [BIG, SMALL]
    assert asyncio.run(_body(ctx))["model"] == f"{SMALL}:Q4_K_M"
    primary = _Ctx()
    asyncio.run(_common.ensure_model(primary, "chat", {"repo": BIG}))
    assert asyncio.run(_body(primary))["model"] == "default"
    again = _Ctx()
    asyncio.run(_common.ensure_model(again, "chat", {"repo": SMALL, "variant": "Q4_K_M"}))
    assert len(loads) == 1 and asyncio.run(_body(again))["model"] == f"{SMALL}:Q4_K_M"


def test_two_quantizations_of_one_model_are_refused(monkeypatch):
    loads, _ = _fakes(monkeypatch, "fit", [BIG, SMALL], kept_variant = "Q8_0")
    with pytest.raises(FlowNodeError, match = "already loaded alongside as Q8_0"):
        asyncio.run(_common.ensure_model(_Ctx(), "chat", {"repo": SMALL, "variant": "Q4_K_M"}))
    assert not loads


def test_always_surfaces_studios_refusal_and_never_replaces(monkeypatch):
    loads, state = _fakes(monkeypatch, "always", [BIG], refuse = "Not enough GPU memory to keep both")
    with pytest.raises(FlowNodeError, match = "alongside .*Not enough GPU memory"):
        asyncio.run(_common.ensure_model(_Ctx(), "chat", {"repo": SMALL}))
    assert state["serving"] == [BIG]


def test_a_shared_step_hands_the_alongside_load_to_an_exclusive_rerun(monkeypatch):
    loads, _ = _fakes(monkeypatch, "fit", [BIG])
    ctx = _Ctx()
    ctx.gpu_shared = True
    with pytest.raises(NeedsExclusiveGpu):
        asyncio.run(_common.ensure_model(ctx, "chat", {"repo": SMALL}))
    assert not loads


def test_each_kept_model_has_its_own_slots_in_the_lease():
    async def go():
        lease = GpuLease()
        inside, done = [], asyncio.Event()

        async def call(name, model):
            async with lease.shared(slots = 1, model = model):
                inside.append(name)
                await done.wait()

        tasks = [
            asyncio.create_task(call("big", None)),
            asyncio.create_task(call("small", SMALL.lower())),
            asyncio.create_task(call("big2", None)),
        ]
        await asyncio.sleep(0.02)
        assert sorted(inside) == ["big", "small"]
        done.set()
        await asyncio.gather(*tasks)
        assert sorted(inside) == ["big", "big2", "small"]
        assert not lease.locked()

    asyncio.run(go())


def test_the_lease_key_and_slot_count_follow_the_kept_model(monkeypatch):
    from core.inference import model_slots

    kept = SimpleNamespace(
        llama = SimpleNamespace(model_identifier = SMALL, is_loaded = True, effective_parallel_slots = 2)
    )
    monkeypatch.setattr(model_slots, "visible", lambda: [kept])
    assert engine_mod.chat_model_key({"model": {"repo": SMALL}}) == SMALL.lower()
    assert engine_mod.chat_slots(SMALL.lower()) == 2
    assert engine_mod.chat_model_key({"model": {"repo": BIG}}) is None
    assert engine_mod.chat_model_key({}) is None


def test_a_call_queued_behind_an_alongside_load_counts_against_the_new_model():
    async def go():
        lease = GpuLease()
        kept = {"small": False}
        loading, inside, done = asyncio.Event(), [], asyncio.Event()

        async def load():
            async with lease:
                await loading.wait()
                kept["small"] = True

        def key():
            return SMALL.lower() if kept["small"] else None

        async def call(name, model):
            async with lease.shared(slots = 1, model = model):
                inside.append(name)
                await done.wait()

        loader = asyncio.create_task(load())
        await asyncio.sleep(0.01)
        first = asyncio.create_task(call("first", key))
        await asyncio.sleep(0.01)
        loading.set()
        await asyncio.sleep(0.02)
        second = asyncio.create_task(call("second", SMALL.lower()))
        await asyncio.sleep(0.02)
        assert inside == ["first"]
        done.set()
        await asyncio.gather(loader, first, second)
        assert inside == ["first", "second"]

    asyncio.run(go())


def test_a_card_with_load_settings_still_loads_alongside(monkeypatch):
    loads, state = _fakes(monkeypatch, "fit", [BIG])
    ctx = _Ctx()
    options = {"max_seq_length": 8192, "cache_type_kv": "q8_0"}
    asyncio.run(_common.ensure_model(ctx, "chat", {"repo": SMALL}, options = options))
    assert len(loads) == 1 and loads[0]["alongside"] is True
    assert loads[0]["max_seq_length"] == 8192 and loads[0]["extra_body"] == {"cache_type_kv": "q8_0"}
    assert state["serving"] == [BIG, SMALL]


def test_a_kept_model_is_reused_only_with_the_settings_it_was_loaded_with(monkeypatch):
    loads, state = _fakes(monkeypatch, "fit", [BIG])
    options = {"max_seq_length": 8192, "cache_type_kv": "q8_0"}
    asyncio.run(_common.ensure_model(_Ctx(), "chat", {"repo": SMALL}, options = options))
    asyncio.run(_common.ensure_model(_Ctx(), "chat", {"repo": SMALL}, options = dict(options)))
    assert len(loads) == 1
    for other in ({"max_seq_length": 32768}, None):
        with pytest.raises(FlowNodeError, match = "other load settings"):
            asyncio.run(_common.ensure_model(_Ctx(), "chat", {"repo": SMALL}, options = other))
    assert len(loads) == 1
    # Kept by Chat rather than a flow: its settings are unknown, so only a card without its own may bind.
    state["serving"] = [BIG, "unsloth/gemma-3-4b-it-GGUF"]
    asyncio.run(_common.ensure_model(_Ctx(), "chat", {"repo": "unsloth/gemma-3-4b-it-GGUF"}))
    state["serving"] = [BIG, SMALL]
    asyncio.run(_common.ensure_model(_Ctx(), "chat", {"repo": SMALL}))
    with pytest.raises(FlowNodeError, match = "other load settings"):
        asyncio.run(_common.ensure_model(_Ctx(), "chat", {"repo": SMALL}, options = options))
    assert len(loads) == 1
