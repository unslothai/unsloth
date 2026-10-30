# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LLM nodes and Chat's saved per-model settings: which value wins, what reaches the request, and that a flow saved
before the setting existed runs exactly as it did."""

import asyncio

from core.flows import engine as engine_mod
from core.flows import model_settings
from core.flows.engine import param_values
from core.flows.nodes import _common
from core.flows.specs import NODE_SPECS
from core.inference import gpu_arbiter

GENERATE = next(s for s in NODE_SPECS if s["type"] == "llm.generate")
VISION = next(s for s in NODE_SPECS if s["type"] == "llm.vision")
REPO = "unsloth/Qwen3-4B-GGUF"
SAVED = {
    "temperature": 0.3,
    "topP": 0.7,
    "topK": 40,
    "minP": 0.05,
    "minPMode": "custom",
    "repetitionPenalty": 1.1,
    "presencePenalty": 0.5,
    "maxTokens": 2048,
    "systemPrompt": "Be brief.",
    "seed": 7,
}


def test_saved_settings_win_over_the_card_when_on():
    card = param_values(
        GENERATE, {"params": {"temperature": 1.5, "system": "card", "use_model_settings": True}}
    )
    out = model_settings.overlay(card, SAVED, GENERATE)
    assert out["temperature"] == 0.3 and out["system"] == "Be brief."
    assert (out["top_p"], out["top_k"], out["min_p"]) == (0.7, 40, 0.05)
    assert (out["repetition_penalty"], out["presence_penalty"], out["max_tokens"]) == (
        1.1,
        0.5,
        2048,
    )
    # The flow's seed stays the card's: Re-roll and the flow seed own it.
    assert out["seed"] is None


def test_unsaved_keys_keep_the_card_values():
    card = param_values(GENERATE, {"params": {"temperature": 1.5, "top_p": 0.9}})
    out = model_settings.overlay(card, {"temperature": 0.2}, GENERATE)
    assert out["temperature"] == 0.2 and out["top_p"] == 0.9 and out["max_tokens"] == 1024
    assert model_settings.overlay(card, None, GENERATE) is card


def test_server_min_p_is_not_copied_and_a_max_output_cap_takes_the_cards_ceiling():
    card = param_values(GENERATE, {"params": {"max_tokens": 900}})
    saved = {"minP": 0.2, "minPMode": "server-default", "maxTokens": 2**31 - 1}
    out = model_settings.overlay(card, saved, GENERATE)
    assert out["min_p"] is None and out["max_tokens"] == 32768
    vision = model_settings.overlay(param_values(VISION, {}), {"maxTokens": 2**31 - 1}, VISION)
    assert vision["max_tokens"] == 8192


def test_server_min_p_clears_the_cards_own():
    card = param_values(GENERATE, {"params": {"min_p": 0.3}})
    out = model_settings.overlay(card, {"minPMode": "server-default"}, GENERATE)
    assert out["min_p"] is None and "min_p" not in model_settings.request_params(out)
    assert (
        model_settings.overlay(card, {"minPMode": "server-default"}, GENERATE, {"min_p"})["min_p"]
        == 0.3
    )


def test_saved_values_are_clamped_to_the_node():
    out = model_settings.overlay(param_values(VISION, {"params": {}}), {"maxTokens": 20000}, VISION)
    assert out["max_tokens"] == 8192


def test_a_connected_input_beats_the_saved_value():
    card = param_values(GENERATE, {"params": {"temperature": 1.2}})
    out = model_settings.overlay(card, SAVED, GENERATE, {"temperature"})
    assert out["temperature"] == 1.2 and out["top_p"] == 0.7


def test_saved_entry_matches_the_model_id_case_insensitively():
    by_model = {"Unsloth/Qwen3-4B-GGUF": {"temperature": 0.1}, "other": {"temperature": 0.9}}
    assert model_settings.find_saved(by_model, REPO) == {"temperature": 0.1}
    assert model_settings.find_saved(by_model, "missing") is None
    assert model_settings.find_saved(None, REPO) is None


def test_an_old_flow_runs_unchanged():
    # Saved before these params existed: the toggle is off and no sampling key is sent.
    old = param_values(GENERATE, {"params": {"system": "s", "temperature": 0.2, "max_tokens": 300}})
    assert old["use_model_settings"] is False
    assert model_settings.request_params(old) == {}
    assert (old["system"], old["temperature"], old["max_tokens"]) == ("s", 0.2, 300)
    vision = param_values(VISION, {"params": {"max_tokens": 120}})
    assert (
        vision["temperature"] is None
        and vision["system"] == ""
        and not vision["use_model_settings"]
    )
    assert vision["thinking"] is False and vision["reasoning_effort"] is None


class _Ctx:
    chat_model = None
    resident: dict = {}

    def on_cancel(self, fn):
        pass


def _body(thinking, params):
    return asyncio.run(
        _common._chat_body(
            _Ctx(),
            system = "",
            prompt = "p",
            temperature = 0.5,
            max_tokens = 8,
            images = None,
            thinking = thinking,
            seed = None,
            stream = False,
            sampling = model_settings.request_params(params),
        )
    )


def test_request_params_reach_the_request_body():
    body = _body(False, {"top_p": 0.8, "top_k": 0, "min_p": None, "reasoning_effort": ""})
    assert body["top_p"] == 0.8 and body["top_k"] == 0
    assert "min_p" not in body and "reasoning_effort" not in body


def test_reasoning_effort_is_sent_only_with_thinking_on():
    assert _body(True, {"reasoning_effort": "high"})["reasoning_effort"] == "high"
    assert "reasoning_effort" not in _body(False, {"reasoning_effort": "high"})


def test_the_engine_reads_the_loaded_model_when_the_card_names_none(monkeypatch):
    seen = []

    async def fake_resident(ctx, kind):
        return REPO

    def fake_read(model_id):
        seen.append(model_id)
        return SAVED

    monkeypatch.setattr(_common, "resident_name", fake_resident)
    monkeypatch.setattr(model_settings, "read_saved", fake_read)
    params = param_values(GENERATE, {"params": {"use_model_settings": True}})
    asyncio.run(engine_mod._apply_saved_settings(_Ctx(), GENERATE, params, set()))
    assert seen == [REPO] and params["temperature"] == 0.3

    seen.clear()
    params = param_values(
        GENERATE,
        {"params": {"use_model_settings": True, "model": {"repo": "a/b", "variant": None}}},
    )
    asyncio.run(engine_mod._apply_saved_settings(_Ctx(), GENERATE, params, set()))
    assert seen == ["a/b"]


# Load settings: a card's own, or the model's saved launch override, reload only when they differ.

from types import SimpleNamespace

from core.flows.cache import cache_key
from core.flows.nodes import text as text_nodes
from studio_mcp import loading

GGUF = "unsloth/Qwen3-4B-GGUF"
CARD_LOAD = {"max_seq_length": 32768, "kv_cache_dtype": "q8_0"}


def test_card_load_settings_map_onto_the_load_request():
    kwargs = model_settings.load_kwargs(CARD_LOAD, is_gguf = True)
    assert kwargs == {"max_seq_length": 32768, "cache_type_kv": "q8_0"}
    assert model_settings.load_kwargs({}, is_gguf = True) is None
    assert model_settings.load_kwargs(None, is_gguf = True) is None
    # GPU placement is a GGUF-only knob, as in Chat's own load path.
    assert "gpu_layers" not in (model_settings.load_kwargs({"gpu_layers": 10}, is_gguf = False) or {})


def _load_fakes(monkeypatch, serving):
    loads = []
    state = {"serving": list(serving)}

    async def fake_load_llm(caller, ctx, **kwargs):
        loads.append(kwargs)
        state["serving"] = [kwargs["model"]]

    async def fake_slots(caller):
        return {"llm": state["serving"]}

    async def fake_resident(caller):
        return {"llm": state["serving"]}, {"is_gguf": True, "serving": state["serving"]}

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", fake_resident)
    monkeypatch.setattr(_common, "_caller", lambda ctx: None)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(_common, "_LOADED_OPTIONS", {})
    return loads


class _LoadCtx:
    def __init__(self, params):
        self.params, self.resident, self.spec, self.gpu_shared = params, {}, {}, False
        self.statuses = []

    def set_status(
        self,
        status,
        message = None,
    ):
        self.statuses.append(status)

    def on_cancel(self, fn):
        pass


async def _ensure(params, kind = "chat"):
    ctx = _LoadCtx(params)
    model, load = await text_nodes._model_and_load(ctx, kind)
    await _common.ensure_model(ctx, kind, model, load)
    return ctx


def test_a_card_reloads_when_its_load_settings_differ_and_not_when_they_match(monkeypatch):
    loads = _load_fakes(monkeypatch, [GGUF])
    card = {"model": {"repo": GGUF, "variant": None}, "load": CARD_LOAD}
    ctx = asyncio.run(_ensure(card))
    assert len(loads) == 1 and loads[0]["max_seq_length"] == 32768
    assert loads[0]["extra_body"] == {"cache_type_kv": "q8_0"}
    assert "loading_model" in ctx.statuses
    # The next card asks for the same: no reload.
    ctx = asyncio.run(_ensure(dict(card)))
    assert len(loads) == 1 and "loading_model" not in ctx.statuses
    # A different context does.
    asyncio.run(_ensure({**card, "load": {**CARD_LOAD, "max_seq_length": 8192}}))
    assert len(loads) == 2 and loads[1]["max_seq_length"] == 8192


def test_a_card_on_the_loaded_model_with_load_settings_reloads_that_model(monkeypatch):
    loads = _load_fakes(monkeypatch, [GGUF])
    asyncio.run(_ensure({"model": None, "load": {"max_seq_length": 4096}}))
    assert [l["model"] for l in loads] == [GGUF]


def test_an_old_card_without_load_settings_keeps_what_is_loaded(monkeypatch):
    loads = _load_fakes(monkeypatch, [GGUF])
    old = param_values(GENERATE, {"params": {"model": {"repo": GGUF, "variant": None}}})
    assert old["load"] is None
    asyncio.run(_ensure(old))
    asyncio.run(_ensure(param_values(GENERATE, {"params": {}})))
    assert loads == []


def test_saved_load_settings_apply_when_the_card_follows_them(monkeypatch):
    monkeypatch.setattr(model_settings, "read_saved", lambda model_id: None)
    monkeypatch.setattr(
        model_settings, "saved_load", lambda model_id, variant = None: {"max_seq_length": 65536}
    )
    params = param_values(
        GENERATE,
        {
            "params": {
                "use_model_settings": True,
                "model": {"repo": GGUF, "variant": None},
                "load": CARD_LOAD,
            }
        },
    )
    asyncio.run(engine_mod._apply_saved_settings(_Ctx(), GENERATE, params, set()))
    assert params["load"] == {"max_seq_length": 65536, "kv_cache_dtype": "q8_0"}


def test_the_cache_key_includes_the_load_settings():
    base = param_values(GENERATE, {"params": {"model": {"repo": GGUF, "variant": None}}})
    key = lambda p: cache_key(
        "llm.generate", 1, p, {}, model = {"repo": GGUF, "variant": None}, seed = 1
    )
    assert key(base) != key({**base, "load": CARD_LOAD})
    assert key({**base, "load": CARD_LOAD}) != key(
        {**base, "load": {**CARD_LOAD, "max_seq_length": 8192}}
    )


def _loads():
    return gpu_arbiter.load_count(gpu_arbiter.CHAT)


def _plan(nodes):
    return SimpleNamespace(
        specs = {nid: GENERATE for nid in nodes},
        nodes = {nid: {"params": params} for nid, params in nodes.items()},
    )


def test_ready_cards_sharing_a_load_run_back_to_back(monkeypatch):
    monkeypatch.setattr(_common, "_LOADED_OPTIONS", {})
    model = {"repo": GGUF, "variant": None}
    big = {"model": model, "load": {"max_seq_length": 32768}}
    small = {"model": model, "load": {"max_seq_length": 8192}}
    plan = _plan({"a": big, "b": small, "c": small})
    rs = SimpleNamespace(plan = plan, resident = {})
    engine = engine_mod.FlowEngine.__new__(engine_mod.FlowEngine)
    # Two ready cards share 8k: one of them goes first.
    assert engine._pick_gpu(rs, ["a", "b", "c"]) in ("b", "c")
    # With the model resident at 32k, the 32k card goes first.
    opts = model_settings.load_kwargs(big["load"], is_gguf = True)
    monkeypatch.setattr(
        _common, "_LOADED_OPTIONS", {"llm": (GGUF.lower(), _common.options_hash(opts), _loads())}
    )
    rs.resident = {"llm": GGUF}
    assert engine._pick_gpu(rs, ["b", "a", "c"]) == "a"


def test_a_card_on_the_loaded_model_waits_when_its_load_differs(monkeypatch):
    big = {"load": {"max_seq_length": 32768}}
    small = {"load": {"max_seq_length": 8192}}
    plan = _plan({"a": small, "b": big})
    opts = model_settings.load_kwargs(big["load"], is_gguf = True)
    monkeypatch.setattr(
        _common, "_LOADED_OPTIONS", {"llm": (GGUF.lower(), _common.options_hash(opts), _loads())}
    )
    rs = SimpleNamespace(plan = plan, resident = {"llm": GGUF})
    engine = engine_mod.FlowEngine.__new__(engine_mod.FlowEngine)
    assert engine._pick_gpu(rs, ["a", "b"]) == "b"
    plan = _plan({"a": small, "b": {}})
    rs.plan = plan
    assert engine._pick_gpu(rs, ["a", "b"]) == "b"


def test_gguf_only_settings_on_a_card_with_model_unset_are_compared(monkeypatch):
    layers = {"load": {"gpu_layers": 12}}
    other = {"load": {"gpu_layers": 40}}
    opts = model_settings.load_kwargs(other["load"], is_gguf = True)
    assert opts and model_settings.card_load_kwargs(layers, GGUF) != opts
    monkeypatch.setattr(
        _common, "_LOADED_OPTIONS", {"llm": (GGUF.lower(), _common.options_hash(opts), _loads())}
    )
    rs = SimpleNamespace(plan = _plan({"a": layers, "b": other}), resident = {"llm": GGUF})
    engine = engine_mod.FlowEngine.__new__(engine_mod.FlowEngine)
    assert engine._pick_gpu(rs, ["a", "b"]) == "b"


def test_an_explicit_zero_context_reaches_the_load(monkeypatch):
    # Manual GPU memory with Auto layers maps to max_seq_length 0: llama.cpp's fit sizes the context.
    loads = _load_fakes(monkeypatch, ["other/model"])
    load = {"gpu_memory_mode": "manual"}
    assert model_settings.load_kwargs(load, is_gguf = True)["max_seq_length"] == 0
    asyncio.run(_ensure({"model": {"repo": GGUF, "variant": None}, "load": load}))
    assert loads[0]["max_seq_length"] == 0
    assert _common._load_context(None) == _common.FLOW_CHAT_CONTEXT


def test_a_card_on_the_loaded_model_keeps_its_quant_through_a_reload(monkeypatch):
    loads = _load_fakes(monkeypatch, [GGUF])

    async def resident(caller):
        return {"llm": [GGUF]}, {"is_gguf": True, "serving": [GGUF], "gguf_variant": "UD-Q4_K_XL"}

    monkeypatch.setattr(loading, "_resident", resident)
    asyncio.run(_ensure({"model": None, "load": {"custom_context_length": 4096}}))
    assert loads[0]["variant"] == "UD-Q4_K_XL" and loads[0]["max_seq_length"] == 4096

    seen = []
    monkeypatch.setattr(model_settings, "read_saved", lambda model_id: None)
    monkeypatch.setattr(
        model_settings, "saved_load", lambda model_id, variant = None: seen.append(variant) or {}
    )

    async def fake_resident_name(ctx, kind):
        return GGUF

    monkeypatch.setattr(_common, "resident_name", fake_resident_name)
    params = param_values(GENERATE, {"params": {"use_model_settings": True}})
    asyncio.run(engine_mod._apply_saved_settings(_Ctx(), GENERATE, params, set()))
    assert seen == ["UD-Q4_K_XL"]


def test_the_context_pin_wins_under_manual_auto_layers():
    load = {"gpu_memory_mode": "manual", "custom_context_length": 12288}
    assert model_settings.load_kwargs(load, is_gguf = True)["max_seq_length"] == 12288


def test_a_stale_load_record_does_not_steer_the_scheduler(monkeypatch):
    big = {"load": {"max_seq_length": 32768}}
    small = {"load": {"max_seq_length": 8192}}
    opts = model_settings.load_kwargs(big["load"], is_gguf = True)
    # Recorded before some other load (Chat reloading the model) bumped the count.
    monkeypatch.setattr(
        _common,
        "_LOADED_OPTIONS",
        {"llm": (GGUF.lower(), _common.options_hash(opts), _loads() - 1)},
    )
    rs = SimpleNamespace(plan = _plan({"a": small, "b": big, "c": {}}), resident = {"llm": GGUF})
    engine = engine_mod.FlowEngine.__new__(engine_mod.FlowEngine)
    assert engine._pick_gpu(rs, ["a", "b", "c"]) == "c"


def test_a_model_unset_card_hashes_the_resident_quants_saved_load(monkeypatch):
    by_quant = {"Q8_0": {"max_seq_length": 32768}, None: {"max_seq_length": 8192}}
    monkeypatch.setattr(
        model_settings, "saved_load", lambda model_id, variant = None: dict(by_quant[variant])
    )
    opts = model_settings.load_kwargs(by_quant["Q8_0"], is_gguf = True)
    monkeypatch.setattr(
        _common, "_LOADED_OPTIONS", {"llm": (GGUF.lower(), _common.options_hash(opts), _loads())}
    )
    follows = {"use_model_settings": True}
    other = {"load": {"max_seq_length": 8192}}
    rs = SimpleNamespace(
        plan = _plan({"a": other, "b": follows}),
        resident = {"llm": GGUF},
        resident_variant = {"llm": "Q8_0"},
    )
    engine = engine_mod.FlowEngine.__new__(engine_mod.FlowEngine)
    assert engine._pick_gpu(rs, ["a", "b"]) == "b"
    rs.resident_variant = {}
    assert engine._pick_gpu(rs, ["a", "b"]) == "a"


def test_the_resident_quant_is_probed_only_for_a_model_unset_card_that_follows_saved(monkeypatch):
    from studio_mcp import loading

    calls = []

    async def fake_resident(caller):
        calls.append(caller)
        return {"llm": [GGUF]}, {"gguf_variant": "Q8_0"}

    monkeypatch.setattr(loading, "_resident", fake_resident)
    engine = engine_mod.FlowEngine.__new__(engine_mod.FlowEngine)
    rs = SimpleNamespace(
        plan = _plan({"a": {"load": {"max_seq_length": 8192}}, "b": {"use_model_settings": True}}),
        caller = None,
        resident_variant = {},
    )
    asyncio.run(engine._probe_resident_variant(rs, ["a"]))
    assert not calls
    asyncio.run(engine._probe_resident_variant(rs, ["a", "b"]))
    assert len(calls) == 1 and rs.resident_variant == {"llm": "Q8_0"}


def test_a_model_unset_card_runs_before_cards_that_switch_to_another_repo(monkeypatch):
    other = {"repo": "unsloth/gemma-3-4b-it-GGUF", "variant": None}
    opts = model_settings.load_kwargs({"max_seq_length": 32768}, is_gguf = True)
    monkeypatch.setattr(
        _common, "_LOADED_OPTIONS", {"llm": (GGUF.lower(), _common.options_hash(opts), _loads())}
    )
    unset = {"load": {"max_seq_length": 8192}}
    plan = _plan({"a": unset, "b": {"model": other}, "c": {"model": other}})
    rs = SimpleNamespace(plan = plan, resident = {"llm": GGUF})
    engine = engine_mod.FlowEngine.__new__(engine_mod.FlowEngine)
    # The two Gemma cards outnumber it, but loading Gemma first would change what the unset card runs on.
    assert engine._pick_gpu(rs, ["b", "c", "a"]) == "a"
    rs.plan = _plan({"b": {"model": other}, "c": {"model": other}})
    assert engine._pick_gpu(rs, ["b", "c"]) == "b"
