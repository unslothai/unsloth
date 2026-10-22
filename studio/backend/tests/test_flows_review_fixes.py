# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

from core.flows import engine as engine_mod
from core.flows import package
from core.flows.compiler import CompileError, compile_flow, parse_document
from core.flows.engine import param_values
from core.flows.registry import FlowNodeError
from core.flows.nodes import training
from core.flows.specs import SPECS_BY_TYPE
from core.flows.story import node_value

from ._flows_helpers import doc, edge, node, run_async
from .test_flows_engine import BASE, _engine, _run, _ret, _status


def test_numeric_params_are_clamped_to_the_catalog():
    spec = SPECS_BY_TYPE["audio.concat"]
    assert param_values(spec, {"params": {"gap_ms": 10**9}})["gap_ms"] == 10_000
    assert param_values(spec, {"params": {"gap_ms": -5}})["gap_ms"] == 0


def test_latest_run_output_beats_the_baked_value():
    d = {"baked": {"g": {"story": "packaged"}}}
    assert node_value(d, {"g": {"story": "rerolled"}}, "g", "story") == "rerolled"
    assert node_value(d, {}, "g", "story") == "packaged"


def test_imported_media_urls_that_leave_studio_are_dropped():
    clean = package.scrub(
        {
            "a": {
                "kind": "image",
                "ref": "sha256:" + "0" * 64,
                "url": "https://evil.example/x.png",
            },
            "b": {"kind": "image", "ref": "image:1", "url": "/api/inference/images/gallery/1/file"},
            "c": {"kind": "image", "ref": "image:2", "url": "//evil.example/x"},
        }
    )
    assert "url" not in clean["a"] and "url" not in clean["c"]
    assert clean["b"]["url"] == "/api/inference/images/gallery/1/file"


@run_async
async def test_a_for_each_failure_in_an_early_iteration_fails_the_run():
    calls = []

    async def flaky(ctx):
        calls.append(ctx.inputs.get("values"))
        if len(calls) == 1:
            raise RuntimeError("boom")
        return {"text": "ok"}

    eng = _engine(
        {**BASE, "input.list": lambda ctx: _ret({"items": ["a", "b"]}), "text.template": flaky}
    )
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each"),
            node("t", "text.template", group = "fe"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("t", "text", "fe", "collect"),
        ],
    )
    rs = await _run(eng, d)
    assert len(calls) == 2
    assert rs.status == "failed"


@run_async
async def test_ignore_baked_recomputes_shipped_outputs():
    ran = []

    async def text(ctx):
        ran.append(ctx.node_id)
        return {"text": "fresh"}

    eng = _engine({**BASE, "input.text": text})
    d = doc([node("a", "input.text", value = "x")], baked = {"a": {"text": "shipped"}})
    rs = await _run(eng, d)
    assert ran == [] and _status(rs, "a") == "cached"
    rs = await _run(eng, d, ignore_baked = True)
    assert ran == ["a"] and rs.nodes["a"].outputs == {"text": "fresh"}


class _Ctx:
    def __init__(self, routes):
        self.routes, self.calls, self.hooks = routes, [], []
        self.inputs = {"dataset": {"kind": "dataset", "source": "recipe", "ref": "job1"}}
        self.params = {p["id"]: p["default"] for p in SPECS_BY_TYPE["train.finetune"]["params"]}
        self.seed, self.caller = 1, object()

    async def call(
        self,
        method,
        path,
        json = None,
        params = None,
        files = None,
        **_,
    ):
        self.calls.append((method, path, params, json, files))
        return self.routes(method, path, params, json)

    def on_cancel(self, fn):
        self.hooks.append(fn)


def test_finetune_trains_on_a_data_recipe_dataset(monkeypatch):
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(3)]

    def routes(method, path, params, body):
        if path.endswith("/jobs/job1/dataset"):
            off = params["offset"]
            return {"dataset": rows[off : off + 2] if off < 3 else [], "total": 3}
        if path == "/api/hub/datasets/upload":
            return {"stored_path": "/cache/recipe-job1.jsonl"}
        if path == "/api/train/start":
            raise RuntimeError(f"started with {body['local_datasets']}")
        raise AssertionError(path)

    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: "cache:recipe-job1.jsonl")
    ctx = _Ctx(routes)
    try:
        asyncio.run(training.finetune(ctx))
    except RuntimeError as exc:
        assert "cache:recipe-job1.jsonl" in str(exc)
    else:
        raise AssertionError("training never started")
    uploaded = [c for c in ctx.calls if c[1] == "/api/hub/datasets/upload"][0][4]["file"][1]
    assert uploaded.decode().count("\n") == 2


def test_edit_image_falls_back_when_the_family_has_no_instruction_editing():
    from core.flows.nodes import image
    from core.flows.registry import FlowNodeError

    class Ctx:
        params = {"model": None, "instruction": "make it smile"}
        inputs = {"image": {"kind": "image", "ref": "image:1"}}
        flow_params = {}
        seed = 3
        calls = []

        async def generate_image(self, **kw):
            self.calls.append(kw.get("workflow"))
            if kw.get("workflow") == "edit":
                raise FlowNodeError(
                    "Instruction editing is not supported for the 'flux.2-klein' model family."
                )
            return [{"kind": "image", "ref": "image:2"}]

        def log(self, *a, **k):
            pass

    ctx = Ctx()
    out = asyncio.run(image.edit(ctx))
    assert out["image"]["ref"] == "image:2" and ctx.calls == ["edit", "reference"]


@run_async
async def test_a_full_run_leaves_player_entrypoints_idle():
    async def reply(ctx):
        raise AssertionError("an entrypoint ran in a full run")

    eng = _engine({**BASE, "world.npc_reply": reply})
    d = doc([node("a", "input.text", value = "x"), node("r", "world.npc_reply")])
    rs = await _run(eng, d)
    assert rs.status == "completed", rs.error
    assert _status(rs, "r") == "idle"


def test_generate_image_with_only_references_uses_the_reference_workflow():
    from core.flows.nodes import image

    ref = {"kind": "image", "ref": "image:9"}

    class Ctx:
        params = {p["id"]: p["default"] for p in SPECS_BY_TYPE["image.generate"]["params"]}
        inputs = {"prompt": "the hero waves", "reference": [ref]}
        flow_params = {}
        seed = 5
        seen = {}

        async def generate_image(self, **kw):
            self.seen.update(kw)
            return [{"kind": "image", "ref": "image:10"}]

    ctx = Ctx()
    asyncio.run(image.generate(ctx))
    assert ctx.seen["workflow"] == "reference" and ctx.seen["init_image"] == ref
    assert ctx.seen["references"] is None and ctx.seen["strength"] is None


@run_async
async def test_cancel_hooks_can_still_reach_studio():
    started = asyncio.Event()
    forwarded = []

    async def fake_forward(caller, method, path, **kw):
        forwarded.append(path)
        raise RuntimeError("stop sent")

    async def slow(ctx):
        async def hook():
            await ctx.call("POST", "/api/export/cancel", json = {})

        ctx.on_cancel(hook)
        started.set()
        for _ in range(500):
            await asyncio.sleep(0.02)
            ctx.check_cancelled()
        return {"text": "late"}

    original = engine_mod.flow_caller.forward
    engine_mod.flow_caller.forward = fake_forward
    try:
        eng = _engine({**BASE, "llm.generate": slow})
        d = doc(
            [node("a", "input.text", value = "p"), node("g", "llm.generate")],
            [edge("a", "text", "g", "prompt")],
        )
        parsed, _ = parse_document(d)
        rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed, options = {})
        await asyncio.wait_for(started.wait(), 60)
        await eng.cancel(rs.id)
        await asyncio.wait_for(rs.task, 60)
    finally:
        engine_mod.flow_caller.forward = original
    assert rs.status == "cancelled"
    assert forwarded == ["/api/export/cancel"]


def test_transcribe_retries_without_timestamps_on_501(monkeypatch):
    from core.flows.nodes import audio
    from core.flows.registry import FlowNodeError

    async def fake_bytes(ctx, media):
        return b"RIFF", "audio/wav"

    monkeypatch.setattr(audio, "media_bytes", fake_bytes)

    class Ctx:
        params = {"translate": False, "speakers": False}
        inputs = {"audio": {"kind": "audio", "ref": "audio:1"}}
        formats = []

        async def call(
            self,
            method,
            path,
            files = None,
            data = None,
            **kw,
        ):
            self.formats.append(data["response_format"])
            if data["response_format"] != "json":
                raise FlowNodeError("timestamps need a language", status = 501)
            return {"text": "hello"}

    ctx = Ctx()
    assert asyncio.run(audio.transcribe(ctx))["text"] == "hello"
    assert ctx.formats == ["verbose_json", "json"]


def test_finishing_an_item_quest_takes_that_item():
    from core.flows.world import validate_npc_reply

    from .test_flows_world import _state

    out = validate_npc_reply(
        {"text": "thanks", "take_item": "pie", "complete_quest": "q"},
        _state(inventory = ["rod", "pie"]),
    )
    assert out["complete_quest"] == "q" and out["take_item"] == "rod"


def test_two_flow_outputs_cannot_share_a_name():
    d = doc([node("o1", "output.result", name = "x"), node("o2", "output.result", name = "x")])
    parsed, _ = parse_document(d)
    try:
        compile_flow(parsed)
    except CompileError as exc:
        assert any("already named" in i["message"] for i in exc.issues)
    else:
        raise AssertionError("duplicate output names compiled")


@run_async
async def test_use_the_loaded_model_is_never_cached():
    calls = []

    async def gen(ctx):
        calls.append(ctx.params.get("model"))
        return {"text": "hi"}

    eng = _engine({**BASE, "llm.generate": gen})

    def flow(model):
        return doc(
            [node("a", "input.text", value = "p"), node("g", "llm.generate", model = model)],
            [edge("a", "text", "g", "prompt")],
        )

    await _run(eng, flow(None))
    second = await _run(eng, flow(None))
    assert len(calls) == 2 and _status(second, "g") == "done"
    named = {"repo": "unsloth/x-GGUF", "variant": "Q4_K_M"}
    await _run(eng, flow(named))
    assert _status(await _run(eng, flow(named)), "g") == "cached"


@run_async
async def test_a_hook_registered_after_cancel_still_fires():
    started = asyncio.Event()
    release = asyncio.Event()
    fired = []

    async def starts_a_job(ctx):
        started.set()
        await release.wait()  # the job start request is still in flight when Stop is pressed

        async def hook():
            fired.append(ctx.node_id)

        ctx.on_cancel(hook)
        await asyncio.sleep(0.05)
        ctx.check_cancelled()
        return {"text": "x"}

    eng = _engine({**BASE, "llm.generate": starts_a_job})
    d = doc(
        [node("a", "input.text", value = "p"), node("g", "llm.generate")],
        [edge("a", "text", "g", "prompt")],
    )
    parsed, _ = parse_document(d)
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed, options = {})
    await asyncio.wait_for(started.wait(), 60)
    await eng.cancel(rs.id)
    release.set()
    await asyncio.wait_for(rs.task, 60)
    await asyncio.sleep(0.05)
    assert rs.status == "cancelled" and fired == ["g"]


@run_async
async def test_parallel_for_each_keeps_the_cpu_limit():
    live = 0
    peak = 0

    async def slow(ctx):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return {"text": "ok"}

    eng = _engine(
        {**BASE, "input.list": lambda ctx: _ret({"items": list(range(12))}), "text.template": slow}
    )
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each", parallel = True),
            node("t", "text.template", group = "fe"),
        ],
        [edge("l", "items", "fe", "items"), edge("fe", "item", "t", "values")],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed", rs.error
    assert 1 < peak <= engine_mod.CPU_CONCURRENCY


@run_async
async def test_a_request_aborted_by_stop_reads_as_cancelled():
    started = asyncio.Event()

    async def aborted(ctx):
        started.set()
        while not ctx.cancelled:
            await asyncio.sleep(0.01)
        raise FlowNodeError(
            "Client disconnected before an upstream llama-server generation slot was available"
        )

    eng = _engine({**BASE, "llm.generate": aborted})
    d = doc(
        [node("a", "input.text", value = "p"), node("g", "llm.generate")],
        [edge("a", "text", "g", "prompt")],
    )
    parsed, _ = parse_document(d)
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed, options = {})
    await asyncio.wait_for(started.wait(), 60)
    await eng.cancel(rs.id)
    await asyncio.wait_for(rs.task, 60)
    assert rs.status == "cancelled"
    assert _status(rs, "g") == "cancelled" and rs.nodes["g"].error is None
