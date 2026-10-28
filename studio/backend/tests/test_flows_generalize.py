# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os

import httpx
import pytest

import core.inference.gpu_arbiter as arb
from core.flows import caller as flow_caller
from core.flows import specs as specs_mod
from core.flows.compiler import parse_document
from core.flows.engine import FlowEngine
from core.flows.nodes import _common, logic
from core.flows.registry import FlowCancelled, FlowNodeError
from core.flows.specs import P, inp, out, spec
from studio_mcp import loading

from ._flows_helpers import doc, edge, node, run_async


def _engine(handlers: dict) -> FlowEngine:
    eng = FlowEngine()
    eng.handlers = handlers
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    return eng


async def _run(eng: FlowEngine, d: dict, **options):
    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = options)
    await asyncio.wait_for(rs.task, timeout = 20)
    return rs


async def _result(ctx):
    return {"value": ctx.inputs.get("value")}


def _add_spec(monkeypatch, found: dict) -> None:
    monkeypatch.setitem(specs_mod.SPECS_BY_TYPE, found["type"], found)


@pytest.fixture
def no_lease(monkeypatch):
    monkeypatch.setattr(arb, "_lease", None)
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_residents", {})


class _PollCtx:
    def __init__(self, cancelled = False):
        self.progressed, self.hooks, self.cancelled = [], [], cancelled

    def progress(
        self,
        fraction = None,
        **kw,
    ):
        self.progressed.append((fraction, kw))

    def on_cancel(self, fn):
        self.hooks.append(fn)

    def check_cancelled(self):
        if self.cancelled:
            raise FlowCancelled()


def _states(*payloads):
    it = iter(payloads)

    async def status():
        return next(it)

    return status


def _parse(payload):
    return payload["state"], payload.get("progress"), payload.get("error")


def test_poll_job_reports_progress_and_returns_the_final_payload():
    ctx = _PollCtx()

    async def stop():
        return None

    final = asyncio.run(
        _common.poll_job(
            ctx,
            status = _states(
                {"state": "running", "progress": {"fraction": 0.5, "phase": "train"}},
                {"state": "completed", "n": 3},
            ),
            parse = _parse,
            cancel = stop,
            interval = 0,
        )
    )
    assert final == {"state": "completed", "n": 3}
    assert ctx.progressed == [(0.5, {"phase": "train"})]
    assert ctx.hooks == [stop]


def test_poll_job_raises_the_job_error_and_reports_a_stop_as_cancelled():
    with pytest.raises(FlowNodeError, match = "out of memory"):
        asyncio.run(
            _common.poll_job(
                _PollCtx(),
                status = _states({"state": "failed", "error": "out of memory"}),
                parse = _parse,
                interval = 0,
            )
        )
    with pytest.raises(FlowCancelled):
        asyncio.run(
            _common.poll_job(
                _PollCtx(cancelled = True),
                status = _states({"state": "cancelled"}),
                parse = _parse,
                interval = 0,
            )
        )


def _rows_spec():
    return spec(
        "test.rows",
        "Rows",
        "logic",
        "Publishes rows as it goes.",
        outputs = [out("rows", "list")],
        params = [P("fail", "bool", False)],
    )


@run_async
async def test_partial_rows_stream_but_never_feed_the_next_step(monkeypatch):
    _add_spec(monkeypatch, _rows_spec())
    seen = {}

    async def rows(ctx):
        ctx.partial("rows", [1], step = 1, total = 3)
        ctx.partial("rows", [1, 2], step = 2, total = 3)
        await asyncio.sleep(0.3)
        return {"rows": [1, 2, 3]}

    async def result(ctx):
        seen["value"] = ctx.inputs.get("value")
        return {"value": ctx.inputs.get("value")}

    eng = _engine({"test.rows": rows, "output.result": result})
    d = doc(
        [node("r", "test.rows"), node("o", "output.result", name = "r")],
        [edge("r", "rows", "o", "value")],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed"
    assert seen["value"] == [1, 2, 3]
    partials = [e for e in rs.events if e["type"] == "node.partial"]
    assert partials and partials[-1]["data"] == {
        "port": "rows",
        "value": [1, 2],
        "step": 2,
        "total": 3,
    }
    assert "partial" not in rs.nodes["r"].snapshot()
    assert rs.nodes["r"].outputs == {"rows": [1, 2, 3]}


@run_async
async def test_stored_partials_keep_their_position_but_not_the_value(monkeypatch):
    from core.flows import engine as engine_mod

    _add_spec(monkeypatch, _rows_spec())
    stored = []
    real = engine_mod.flows_db.write_batch

    def capture(run_id, events, *rest):
        stored.extend(event for _, event in events)
        return real(run_id, events, *rest)

    monkeypatch.setattr(engine_mod.flows_db, "write_batch", capture)

    async def rows(ctx):
        ctx.partial("rows", list(range(500)), step = 1, total = 2)
        await asyncio.sleep(0.3)
        return {"rows": [1]}

    rs = await _run(_engine({"test.rows": rows}), doc([node("r", "test.rows")]))
    live = [e for e in rs.events if e["type"] == "node.partial"]
    kept = [e for e in stored if e["type"] == "node.partial"]
    assert live and live[-1]["data"]["value"] == list(range(500))
    assert kept and all("value" not in e["data"] for e in kept)
    assert kept[-1]["data"]["step"] == 1


@run_async
async def test_a_failed_step_keeps_its_last_partial(monkeypatch):
    _add_spec(monkeypatch, _rows_spec())

    async def rows(ctx):
        ctx.partial("rows", ["a", "b"], step = 2, total = 5)
        raise FlowNodeError("the third row ran out of memory")

    eng = _engine({"test.rows": rows})
    rs = await _run(eng, doc([node("r", "test.rows")]))
    snap = rs.nodes["r"].snapshot()
    assert snap["status"] == "failed"
    assert snap["partial"]["rows"]["value"] == ["a", "b"]
    assert any(e["type"] == "node.partial" for e in rs.events)


def _llm_fakes(monkeypatch):
    loads, state = [], {"llm": [], "variant": None, "is_gguf": True}

    async def fake_load_llm(caller, ctx, **kwargs):
        loads.append(kwargs)
        state["llm"] = [kwargs["model"]]

    async def fake_slots(caller):
        return {"llm": state["llm"]}

    async def fake_resident(caller):
        return {"llm": state["llm"]}, {"gguf_variant": None, "is_gguf": state["is_gguf"]}

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", fake_resident)
    monkeypatch.setattr(_common, "_caller", lambda ctx: None)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(_common, "_LOADED_OPTIONS", {})
    return loads, state


class _ModelCtx:
    def __init__(self, spec = None):
        self.resident, self.spec = {}, spec or {}

    def set_status(
        self,
        status,
        message = None,
    ):
        pass


def test_load_settings_reload_only_when_they_change(monkeypatch):
    loads, _ = _llm_fakes(monkeypatch)
    ctx = _ModelCtx()
    choice = {"repo": "unsloth/Qwen3-0.6B-GGUF"}
    kv8 = {"cache_type_kv": "q8_0", "max_seq_length": 8192}

    async def go():
        await _common.ensure_model(ctx, "chat", choice)
        await _common.ensure_model(ctx, "chat", choice)
        await _common.ensure_model(ctx, "chat", choice, kv8)
        await _common.ensure_model(ctx, "chat", choice, dict(kv8))
        await _common.ensure_model(ctx, "chat", choice, {"cache_type_kv": "q4_0"})
        await _common.ensure_model(ctx, "chat", choice)

    asyncio.run(go())
    assert len(loads) == 4
    assert (
        loads[0]["extra_body"] is None and loads[0]["max_seq_length"] == _common.FLOW_CHAT_CONTEXT
    )
    assert (
        loads[1]["extra_body"] == {"cache_type_kv": "q8_0"} and loads[1]["max_seq_length"] == 8192
    )
    assert loads[2]["extra_body"] == {"cache_type_kv": "q4_0"}
    assert loads[3]["extra_body"] is None


def test_a_load_elsewhere_voids_the_remembered_settings(monkeypatch):
    from core.inference import gpu_arbiter

    loads, _ = _llm_fakes(monkeypatch)
    ctx = _ModelCtx()
    choice = {"repo": "unsloth/Qwen3-0.6B-GGUF"}
    kv8 = {"cache_type_kv": "q8_0"}
    asyncio.run(_common.ensure_model(ctx, "chat", choice, kv8))
    monkeypatch.setattr(gpu_arbiter, "_loads", gpu_arbiter._loads + 2)
    monkeypatch.setitem(
        gpu_arbiter._owner_loads,
        gpu_arbiter.CHAT,
        gpu_arbiter._owner_loads.get(gpu_arbiter.CHAT, 0) + 2,
    )
    asyncio.run(_common.ensure_model(ctx, "chat", choice, kv8))
    assert len(loads) == 2 and loads[1]["extra_body"] == kv8


def test_a_media_load_keeps_the_chat_models_settings(no_lease, monkeypatch):
    loads, _ = _llm_fakes(monkeypatch)
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: None)
    monkeypatch.setattr(arb, "_owner_loads", {})
    ctx = _ModelCtx()
    choice = {"repo": "unsloth/Qwen3-0.6B-GGUF"}
    kv8 = {"cache_type_kv": "q8_0"}
    asyncio.run(_common.ensure_model(ctx, "chat", choice, kv8))
    arb.acquire_for(arb.DIFFUSION, lambda: None)
    asyncio.run(_common.ensure_model(ctx, "chat", choice, kv8))
    assert len(loads) == 1


def test_without_settings_a_model_loaded_elsewhere_is_reused_as_before(monkeypatch):
    loads, state = _llm_fakes(monkeypatch)
    state["llm"] = ["unsloth/Qwen3-0.6B-GGUF"]
    asyncio.run(_common.ensure_model(_ModelCtx(), "chat", {"repo": "unsloth/Qwen3-0.6B-GGUF"}))
    assert loads == []


def test_a_requirement_that_cannot_be_checked_fails_closed(monkeypatch):
    _, state = _llm_fakes(monkeypatch)
    state["llm"] = ["unsloth/Qwen3-0.6B"]
    state["is_gguf"] = None
    ctx = _ModelCtx({"title": "Capture", "requires": ["transformers"]})
    with pytest.raises(FlowNodeError, match = "could not check"):
        asyncio.run(_common.ensure_model(ctx, "chat", {"repo": "unsloth/Qwen3-0.6B"}))


def test_a_node_that_needs_transformers_refuses_a_gguf(monkeypatch):
    _, state = _llm_fakes(monkeypatch)
    state["llm"] = ["unsloth/Qwen3-0.6B-GGUF"]
    ctx = _ModelCtx({"title": "Capture Activations", "requires": ["transformers"]})
    with pytest.raises(FlowNodeError, match = "needs a transformers model, not GGUF"):
        asyncio.run(_common.ensure_model(ctx, "chat", {"repo": "unsloth/Qwen3-0.6B-GGUF"}))
    state["is_gguf"] = False
    asyncio.run(_common.ensure_model(ctx, "chat", {"repo": "unsloth/Qwen3-0.6B-GGUF"}))


def test_settings_on_a_media_model_are_refused(monkeypatch):
    _llm_fakes(monkeypatch)
    with pytest.raises(FlowNodeError, match = "chat models only"):
        asyncio.run(
            _common.ensure_model(
                _ModelCtx(), "image", {"repo": "black-forest-labs/FLUX.2-klein"}, {"x": 1}
            )
        )


def test_a_local_model_key_changes_when_its_files_change(tmp_path):
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"a" * 10)
    eng = FlowEngine()
    found = {"model_kind": "chat"}
    params = {"model": {"repo": str(weights)}}

    first = asyncio.run(eng._model_key(None, found, params))
    weights.write_bytes(b"b" * 20)
    os.utime(weights, ns = (1, 2_000_000_000))
    second = asyncio.run(eng._model_key(None, found, params))
    assert first != second
    hub = asyncio.run(eng._model_key(None, found, {"model": {"repo": "unsloth/x"}}))
    assert hub == {"repo": "unsloth/x", "variant": None}


def _bindable_spec():
    return spec(
        "test.scale",
        "Scale",
        "logic",
        "Multiplies.",
        inputs = [inp("value", "number")],
        outputs = [out("value", "number")],
        params = [P("factor", "int", 2, "Factor", min = 1, max = 10, bindable = True)],
    )


def test_a_bindable_param_adds_an_optional_typed_input():
    found = _bindable_spec()
    port = next(p for p in found["inputs"] if p["id"] == "param:factor")
    assert port == {
        "id": "param:factor",
        "type": "number",
        "label": "Factor",
        "optional": True,
        "param": "factor",
    }


def test_existing_specs_publish_no_new_fields():
    for found in specs_mod.NODE_SPECS:
        assert "requires" not in found and "exclusive_gpu" not in found
        assert not any(p.get("param") for p in found["inputs"])


@run_async
async def test_a_connected_value_replaces_the_param_and_is_clamped(monkeypatch):
    _add_spec(monkeypatch, _bindable_spec())
    seen = []

    async def number(ctx):
        return {"value": ctx.params["value"]}

    async def scale(ctx):
        seen.append(ctx.params["factor"])
        assert "param:factor" not in ctx.inputs
        return {"value": ctx.inputs["value"] * ctx.params["factor"]}

    eng = _engine({"input.number": number, "test.scale": scale, "output.result": _result})
    d = doc(
        [
            node("n", "input.number", value = 3),
            node("f", "input.number", value = 50),
            node("s", "test.scale", factor = 2),
            node("o", "output.result", name = "r"),
        ],
        [
            edge("n", "value", "s", "value"),
            {**edge("f", "value", "s", "param:factor"), "id": "f-value-s-factor"},
            edge("s", "value", "o", "value"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.outputs == {"r": 30} and seen == [10]
    d["edges"] = [e for e in d["edges"] if e["target_port"] != "param:factor"]
    d["id"] = "f2"
    rs = await _run(eng, d)
    assert rs.outputs == {"r": 6}


def test_combinations_cover_every_pair_and_refuse_a_grid_too_big():
    class Ctx:
        inputs = {}
        params = {"axes": {"kv": ["f16", "q8_0"], "ctx": [4096, 8192, 16384]}, "max_items": 64}

    found = asyncio.run(logic.product(Ctx()))
    assert found["count"] == 6
    assert {"kv": "q8_0", "ctx": 16384} in found["items"]

    class Big:
        inputs = {"axes": {"a": list(range(20)), "b": list(range(20))}}
        params = {"max_items": 256}

    with pytest.raises(FlowNodeError, match = "400 combinations"):
        asyncio.run(logic.product(Big()))


def test_a_lease_refuses_other_loads_and_training_but_not_its_holder(no_lease, monkeypatch):
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: None)
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: None)
    assert arb.lease_refusal() is None
    arb.acquire_for(arb.CHAT, lambda: None)

    async def holder():
        with arb.operation_lease("bench", "run-1", 60, label = "Benchmark Model"):
            arb.acquire_for(arb.CHAT, lambda: None)
            await asyncio.to_thread(arb.acquire_for, arb.CHAT, lambda: None)
            other = asyncio.get_running_loop()
            refusal = await other.run_in_executor(None, arb.lease_refusal)
            return refusal

    refusal = asyncio.run(holder())
    assert isinstance(refusal, arb.GpuLeaseHeldError)
    assert "Benchmark Model is using the GPU" in str(refusal)
    assert arb.lease_refusal() is None


def test_a_refused_load_keeps_the_load_generation(no_lease, monkeypatch):
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: None)
    arb.acquire_for(arb.CHAT, lambda: None)
    before = arb.load_count()

    def refuse():
        raise RuntimeError("a load is already in progress")

    with pytest.raises(RuntimeError):
        arb.acquire_for(arb.CHAT, refuse)
    assert arb.load_count() == before


def test_while_leased_other_callers_get_409_and_plain_reasserts_pass(no_lease, monkeypatch):
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: None)
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: None)
    arb.acquire_for(arb.CHAT, lambda: None)
    lease = arb.operation_lease("bench", "run-1", 60, label = "Benchmark Model")
    lease.__enter__()
    arb._lease_holder.set(None)
    try:
        with pytest.raises(arb.GpuLeaseHeldError):
            arb.acquire_for(arb.DIFFUSION, lambda: None)
        with pytest.raises(arb.GpuLeaseHeldError):
            arb.acquire_for(arb.CHAT, lambda: None)
        arb.acquire_for(arb.CHAT)
        http = arb.lease_refusal().as_http_exception()
        assert http.status_code == 409 and http.detail["error"] == "gpu_leased"
        with pytest.raises(arb.GpuLeaseHeldError):
            with arb.operation_lease("eval", "run-2", 60):
                pass
    finally:
        arb._lease = None


def test_a_reassert_under_a_lease_never_evicts_the_leased_model(no_lease, monkeypatch):
    from core.inference import gpu_residency

    made_room, released = [], []
    monkeypatch.setattr(gpu_residency, "coreside_locked", lambda *a, **k: made_room.append(a))
    monkeypatch.setattr(arb, "_release_mask_segmenter", lambda *a, **k: released.append("mask"))
    monkeypatch.setattr(
        arb, "_release_idle_video_servers", lambda *a, **k: released.append("video")
    )
    monkeypatch.setattr(arb, "_owner", arb.CHAT)
    monkeypatch.setattr(arb, "_residents", {arb.DIFFUSION: None, arb.CHAT: None})
    arb.acquire_for(arb.CHAT, policy = "fit")
    assert len(made_room) == 1
    released.clear()
    lease = arb.operation_lease("bench", "run-1", 60, label = "Benchmark Image")
    lease.__enter__()
    arb._lease_holder.set(None)
    try:
        arb.acquire_for(arb.CHAT, policy = "fit")
        arb.acquire_for(arb.CHAT)
    finally:
        arb._lease = None
    assert len(made_room) == 1 and released == []
    assert arb.DIFFUSION in arb._residents


def test_an_exclusive_node_always_runs_on_the_gpu_lane():
    from core.flows.specs import spec
    assert spec("t.x", "X", "c", "d", exclusive_gpu = True)["lane"] == "gpu"
    assert spec("t.y", "Y", "c", "d")["lane"] == "cpu"


def test_an_abandoned_lease_expires(no_lease, monkeypatch):
    import time

    lease = arb.operation_lease("bench", "run-1", 1)
    lease.__enter__()
    arb._lease_holder.set(None)
    assert arb.lease_refusal() is not None
    arb._lease["expires"] = time.monotonic() - 1
    assert arb.lease_refusal() is None and not arb.lease_held()


def _measure_spec():
    return spec(
        "test.measure",
        "Measure",
        "logic",
        "Holds the GPU.",
        outputs = [out("value", "text")],
        lane = "gpu",
        cacheable = False,
        exclusive_gpu = True,
    )


@run_async
async def test_an_exclusive_node_holds_the_lease_and_its_cancel_hook_still_acts_as_holder(
    no_lease, monkeypatch
):
    _add_spec(monkeypatch, _measure_spec())
    seen = {}
    started = asyncio.Event()

    async def measure(ctx):
        seen["inside"] = arb.lease_refusal() is None and arb.lease_held()

        async def restore():
            seen["hook"] = arb.lease_refusal() is None

        ctx.on_cancel(restore)
        started.set()
        await asyncio.sleep(5)
        return {"value": "never"}

    eng = _engine({"test.measure": measure})
    parsed, _ = parse_document(doc([node("m", "test.measure")]))
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed)
    await asyncio.wait_for(started.wait(), 5)
    assert isinstance(arb.lease_refusal(), arb.GpuLeaseHeldError)
    await eng.cancel(rs.id)
    await asyncio.wait_for(rs.task, 20)
    assert seen == {"inside": True, "hook": True}
    assert rs.status == "cancelled"
    assert not arb.lease_held()


@run_async
async def test_an_exclusive_node_waits_for_another_lease(no_lease, monkeypatch):
    _add_spec(monkeypatch, _measure_spec())
    monkeypatch.setattr("core.flows.engine.LEASE_RETRY_S", 0.05)

    async def measure(ctx):
        return {"value": "ran"}

    other = arb.operation_lease("eval", "someone", 60, label = "Evaluate Model")
    other.__enter__()
    arb._lease_holder.set(None)
    eng = _engine({"test.measure": measure})
    parsed, _ = parse_document(doc([node("m", "test.measure")]))
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed)
    await asyncio.sleep(0.3)
    assert rs.nodes["m"].status == "waiting"
    assert "Evaluate Model is using the GPU" in (rs.nodes["m"].message or "")
    arb._lease = None
    await asyncio.wait_for(rs.task, 20)
    assert rs.nodes["m"].outputs == {"value": "ran"}


def test_training_start_is_refused_while_another_holder_has_the_lease(no_lease):
    from routes import training

    lease = arb.operation_lease("bench", "run-1", 60, label = "Benchmark Model")
    lease.__enter__()
    arb._lease_holder.set(None)
    try:
        refusal = arb.lease_refusal()
        assert refusal is not None
        source = open(training.__file__, encoding = "utf-8").read()
        # start_training and start_training_batch both run their guards through _prepare_training_request.
        start = source.index("async def _prepare_training_request(")
        assert "lease_refusal()" in source[start : start + 1500]
        start = source.index("async def start_diffusion_training(")
        assert "lease_refusal()" in source[start : start + 1500]
    finally:
        arb._lease = None


def test_unloads_and_training_teardown_respect_a_lease_taken_later(no_lease):
    import fastapi
    from routes import inference, training, video

    lease = arb.operation_lease("bench", "run-1", 60, label = "Benchmark Model")
    lease.__enter__()
    arb._lease_holder.set(None)
    try:
        with pytest.raises(fastapi.HTTPException) as err:
            inference._refuse_while_leased()
        assert err.value.status_code == 409
        with pytest.raises(arb.GpuLeaseHeldError):
            training._free_gpu_for_diffusion_training()
        source = open(training.__file__, encoding = "utf-8").read()
        hook = source.index("def _free_vram_for_training(")
        assert "raise_if_leased()" in source[hook : hook + 600]
        assert arb.GpuLeaseHeldError.blocks_training is True
        vsource = open(video.__file__, encoding = "utf-8").read()
        start = vsource.index("async def unload_video_model(")
        assert "lease_refusal()" in vsource[start : start + 1200]
        isource = open(inference.__file__, encoding = "utf-8").read()
        start = isource.index("async def unload_diffusion_model(")
        assert "_refuse_while_leased()" in isource[start : start + 900]
    finally:
        arb._lease = None
    with arb.operation_lease("bench", "run-1", 60):
        inference._refuse_while_leased()


def test_every_unload_path_shares_the_lease_guard(no_lease):
    import fastapi
    from models.inference import UnloadRequest
    from routes import inference

    lease = arb.operation_lease("bench", "run-1", 60, label = "Benchmark Model")
    lease.__enter__()
    arb._lease_holder.set(None)
    try:
        with pytest.raises(fastapi.HTTPException) as err:
            asyncio.run(inference._unload_model_impl(UnloadRequest(model_path = "x"), "owner"))
        assert err.value.status_code == 409
    finally:
        arb._lease = None


def test_a_lease_waits_for_a_residency_change_already_under_way(no_lease):
    with arb._lock:
        with pytest.raises(arb.GpuLeaseHeldError):
            with arb.operation_lease("bench", "run-1", 60):
                pass
    assert arb._lease is None
    with arb.operation_lease("bench", "run-1", 60):
        assert arb._lease is not None


def test_system_routes_are_reachable_for_reads_only():
    url = httpx.URL("http://studio/api/system")
    assert flow_caller.checked_path(url, "GET") == "/api/system"
    assert (
        flow_caller.checked_path(httpx.URL("http://studio/api/llama/backend"))
        == "/api/llama/backend"
    )
    with pytest.raises(RuntimeError, match = "do not forward"):
        flow_caller.checked_path(url, "POST")
    with pytest.raises(RuntimeError):
        flow_caller.checked_path(httpx.URL("http://studio/api/system/shutdown"), "GET")


def test_a_change_to_any_file_in_a_model_folder_changes_its_fingerprint(tmp_path):
    from core.flows.engine import _local_fingerprint

    for i in range(300):
        (tmp_path / f"shard-{i:04d}.bin").write_bytes(b"x")
    before = _local_fingerprint(str(tmp_path))
    (tmp_path / "shard-0299.bin").write_bytes(b"xy")
    assert _local_fingerprint(str(tmp_path)) != before


def test_loads_with_other_load_settings_are_never_followed(monkeypatch):
    from types import SimpleNamespace

    seen = []

    async def fake_start_load(caller, ctx, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(loading, "start_load", fake_start_load)
    for extra in ({"cache_type_kv": "q8_0"}, {"cache_type_kv": "q4_0"}, None):
        asyncio.run(
            loading.load_llm(
                SimpleNamespace(hf_token = None),
                None,
                model = "unsloth/Qwen3-0.6B-GGUF",
                variant = None,
                max_seq_length = 0,
                load_in_4bit = True,
                hf_token = None,
                extra_body = extra,
            )
        )
    assert len({repr(k["options"]) for k in seen}) == 3
    assert [k["attach"] for k in seen] == [False, False, True]
