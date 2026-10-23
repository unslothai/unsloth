# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

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


def test_finetune_waits_out_a_trainer_that_is_still_shutting_down(monkeypatch):
    monkeypatch.setattr(training, "POLL_S", 0)
    starts = []

    def routes(method, path, params, body):
        if path == "/api/train/start":
            starts.append(body)
            if len(starts) < 3:
                # The route's refusal names the run that is still active, not a new one.
                return {"job_id": "old", "status": "error", "error": "Training already active"}
            raise RuntimeError("started")
        raise AssertionError(path)

    ctx = _Ctx(routes)
    ctx.inputs = {"dataset": {"kind": "dataset", "source": "hf", "ref": "org/ds"}}
    ctx.progress = lambda *a, **k: None
    ctx.check_cancelled = lambda: None
    try:
        asyncio.run(training.finetune(ctx))
    except RuntimeError as exc:
        assert str(exc) == "started"
    assert len(starts) == 3


def test_finetune_reports_other_start_errors_instead_of_polling_another_job():
    from core.flows.registry import FlowNodeError

    def routes(method, path, params, body):
        if path == "/api/train/start":
            return {"job_id": "old", "status": "error", "message": "No GPU memory", "error": "oom"}
        raise AssertionError(path)

    ctx = _Ctx(routes)
    ctx.inputs = {"dataset": {"kind": "dataset", "source": "hf", "ref": "org/ds"}}
    with pytest.raises(FlowNodeError, match = "No GPU memory"):
        asyncio.run(training.finetune(ctx))


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


def test_transcribe_downloads_a_missing_speech_model_once(monkeypatch):
    from core.flows.nodes import audio
    from core.flows.registry import FlowNodeError
    from studio_mcp.tools import audio as mcp_audio

    async def fake_bytes(ctx, media):
        return b"RIFF", "audio/wav"

    downloads = []

    async def fake_download(caller, ctx, model):
        downloads.append(model)

    monkeypatch.setattr(audio, "media_bytes", fake_bytes)
    monkeypatch.setattr(mcp_audio, "_download_then_retry", fake_download)

    class Ctx:
        params = {"translate": False, "speakers": False}
        inputs = {"audio": {"kind": "audio", "ref": "audio:1"}}
        caller = object()
        calls = 0

        def progress(self, *a, **kw):
            pass

        async def call(
            self,
            method,
            path,
            files = None,
            data = None,
            **kw,
        ):
            self.calls += 1
            if not downloads:
                raise FlowNodeError("STT model 'x' is not downloaded yet.", status = 409)
            return {"text": "hello"}

    ctx = Ctx()
    assert asyncio.run(audio.transcribe(ctx))["text"] == "hello"
    assert downloads == [None] and ctx.calls == 2


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


@run_async
async def test_a_run_value_change_is_not_served_from_the_cache():
    calls = []

    async def template(ctx):
        calls.append(ctx.flow_params.get("name"))
        return {"text": f"hi {ctx.flow_params.get('name')}"}

    eng = _engine({**BASE, "text.template": template})
    d = doc([node("t", "text.template", template = "hi {{name}}")])
    first = await _run(eng, d, params = {"name": "Alice"})
    second = await _run(eng, d, params = {"name": "Bob"})
    again = await _run(eng, d, params = {"name": "Bob"})
    assert calls == ["Alice", "Bob"]
    assert _status(first, "t") == "done" and _status(second, "t") == "done"
    assert _status(again, "t") == "cached"


def test_zero_temperature_and_zero_items_are_kept():
    from core.flows.nodes import text, world as world_nodes

    from .flows_player_fakes import FakeCtx

    ctx = FakeCtx(
        "llm.generate", params = {"temperature": 0}, inputs = {"prompt": "hi"}, llm = lambda **kw: "ok"
    )
    asyncio.run(text.generate(ctx))
    assert ctx.llm_calls[0]["temperature"] == 0.0

    ctx = FakeCtx(
        "world.generator", params = {"premise": "a quiet pond", "items": 0}, llm = lambda **kw: {}
    )
    try:
        asyncio.run(world_nodes.world_generator(ctx))
    except Exception:
        pass
    assert "0 items" in ctx.llm_calls[0]["prompt"]
    assert ctx.llm_calls[0]["schema"]["properties"]["items"]["maxItems"] == 0


def test_a_truncated_recipe_is_reported():
    from core.flows.nodes import data

    class Ctx:
        logs = []

        def log(
            self,
            message,
            level = "info",
        ):
            self.logs.append((level, message))

        async def call(
            self,
            method,
            path,
            params = None,
            **kw,
        ):
            offset = params["offset"]
            return {"dataset": [{"i": offset + i} for i in range(500)], "total": 2000}

    ctx = Ctx()
    rows = asyncio.run(data.recipe_rows(ctx, "job1", limit = 1000))
    assert len(rows) == 1000
    assert ctx.logs and ctx.logs[0][0] == "warning" and "2000" in ctx.logs[0][1]


def test_a_maximum_size_grid_stays_bounded():
    import io

    from PIL import Image

    from core.flows.nodes.image import MAX_GRID_SIDE, compose_grid

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 10, 10)).save(buf, "PNG")
    out = compose_grid(
        [buf.getvalue()] * 64,
        [""] * 64,
        columns = 8,
        gap = 24,
        background = "#111",
        caption_style = "none",
        cell = 2048,
    )
    assert max(Image.open(io.BytesIO(out)).size) <= MAX_GRID_SIDE


def test_an_empty_segmentation_mask_never_leaves_an_invisible_character(monkeypatch):
    import base64
    import io

    from PIL import Image

    from core.flows.nodes import _common

    # A character on a plain background, and a segmenter that finds nothing.
    im = Image.new("RGB", (64, 96), (240, 240, 240))
    for x in range(20, 44):
        for y in range(20, 80):
            im.putpixel((x, y), (180, 60, 40))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    black = io.BytesIO()
    Image.new("L", (64, 96), 0).save(black, "PNG")

    async def fake_bytes(ctx, media):
        return buf.getvalue(), "image/png"

    monkeypatch.setattr(_common, "media_bytes", fake_bytes)

    class Ctx:
        saved = []
        logs = []

        async def call(
            self,
            method,
            path,
            json = None,
            **kw,
        ):
            return {"mask": "data:image/png;base64," + base64.b64encode(black.getvalue()).decode()}

        def log(
            self,
            message,
            level = "info",
        ):
            self.logs.append(level)

        async def save_asset(
            self,
            data,
            mime,
            kind = "image",
            extra = None,
        ):
            self.saved.append(data)
            return {"kind": "image", "ref": f"sha256:{len(self.saved)}"}

    ctx = Ctx()
    asyncio.run(
        _common.remove_background(ctx, {"kind": "image", "ref": "image:1"}, "the character")
    )
    assert _common._visible(ctx.saved[0])
    assert "warning" in ctx.logs


def test_a_maximum_size_sprite_sheet_stays_bounded():
    import io

    from PIL import Image

    from core.flows.nodes.image import MAX_GRID_SIDE, sprite_sheet

    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), (200, 10, 10, 255)).save(buf, "PNG")
    png, meta = sprite_sheet([buf.getvalue()] * 256, 1024, 16)
    assert max(Image.open(io.BytesIO(png)).size) <= MAX_GRID_SIDE
    assert meta["frame_width"] * meta["columns"] <= MAX_GRID_SIDE


@run_async
async def test_for_each_keeps_null_results_in_place():
    async def passthrough(ctx):
        return {"value": ctx.inputs.get("value")}

    eng = _engine(
        {
            **BASE,
            "input.list": lambda ctx: _ret({"items": ["a", None, "c"]}),
            "json.passthrough": passthrough,
        }
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

    async def maybe_null(ctx):
        values = ctx.inputs.get("values") or []
        return {"text": None if values == [None] else "x"}

    eng.handlers["text.template"] = maybe_null
    rs = await _run(eng, d)
    assert rs.status == "completed", rs.error
    assert len(rs.nodes["fe"].outputs["results"]) == 3


def test_captions_never_draw_missing_glyphs():
    from core.flows.nodes.image import _ASCII_PUNCT
    assert "first \u2014 even\u2026".translate(_ASCII_PUNCT) == "first - even..."


def test_cached_media_the_user_deleted_is_not_served(monkeypatch):
    from core.flows.nodes import _common
    from core.inference import audio_gallery, image_gallery, video_gallery

    present = {"keep"}
    for mod, fn in (
        (image_gallery, "owned_image_path"),
        (audio_gallery, "owned_audio_path"),
        (video_gallery, "owned_video_path"),
    ):
        monkeypatch.setattr(mod, fn, lambda media_id: "/x" if media_id in present else None)
    kept = {"images": [{"kind": "image", "ref": "image:keep"}], "text": "hi"}
    gone = {"clip": {"kind": "audio", "ref": "audio:gone"}}
    assert _common.gallery_media_present(kept)
    assert not _common.gallery_media_present(gone)
    # Embedded flow assets and plain values are not gallery entries.
    assert _common.gallery_media_present({"image": {"kind": "image", "ref": "sha256:abc"}, "n": 3})


class _UpscaleCtx:
    def __init__(self, png: bytes, error: str):
        self.inputs = {"image": {"kind": "image", "ref": "a" * 64}}
        self.params = {"factor": "2"}
        self.seed, self.error, self.png, self.logs, self.saved = 1, error, png, [], []

    async def generate_image(self, **kwargs):
        raise FlowNodeError(self.error)

    async def media_bytes(self, value):
        return self.png, "image/png"

    def log(
        self,
        message,
        level = "info",
    ):
        self.logs.append((level, message))

    async def save_asset(
        self,
        data,
        mime,
        kind = None,
    ):
        self.saved.append(data)
        return {"kind": "image", "ref": "b" * 64}


def _png_bytes(w, h):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), "red").save(buf, format = "PNG")
    return buf.getvalue()


def test_upscale_resizes_when_the_loaded_model_cannot_upscale():
    import io

    from PIL import Image

    from core.flows.nodes.image import upscale

    ctx = _UpscaleCtx(
        _png_bytes(64, 40), "upscale is not supported for the 'flux.2-klein' model family."
    )
    out = asyncio.run(upscale(ctx))
    assert out["image"]["kind"] == "image"
    assert Image.open(io.BytesIO(ctx.saved[0])).size == (128, 80)
    assert ctx.logs and ctx.logs[0][0] == "warning"


def test_upscale_still_raises_other_errors():
    from core.flows.nodes.image import upscale
    ctx = _UpscaleCtx(_png_bytes(8, 8), "CUDA out of memory")
    with pytest.raises(FlowNodeError):
        asyncio.run(upscale(ctx))


def test_speak_sends_the_reference_transcript_for_cloning(monkeypatch):
    from core.flows.nodes import _common

    sent = []

    async def fake_run(ctx, body):
        sent.append(body)
        return [{"kind": "audio", "ref": "audio:x"}]

    async def fake_ensure(ctx, kind, model):
        return None

    monkeypatch.setattr(_common, "audio_run", fake_run)
    monkeypatch.setattr(_common, "ensure_model", fake_ensure)
    clip = {"kind": "audio", "ref": "audio:abc"}
    asyncio.run(
        _common.speak(
            object(), model = None, text = "Hola", reference = clip, reference_text = " Hello there "
        )
    )
    asyncio.run(
        _common.speak(object(), model = None, text = "Hola", reference_text = "ignored without a clip")
    )
    assert sent[0]["workflow"] == "clone" and sent[0]["inputs"]["reference_text"] == "Hello there"
    assert sent[1]["workflow"] == "speak" and "inputs" not in sent[1]
    # The body must pass the route's own request model (extra fields are refused there).
    from models.inference import AudioRunRequest

    sent[0]["inputs"]["reference"] = {"clip_id": "abc"}
    AudioRunRequest.model_validate(sent[0])
    assert any(p["id"] == "reference_text" for p in SPECS_BY_TYPE["audio.speak"]["inputs"])


def test_streamed_text_hides_thinking_split_across_pieces():
    from core.flows.nodes._common import visible_so_far

    pieces = ["<th", "ink>plan the ", "reply</thi", "nk>Hello", " there <", "b>x"]
    shown, text = "", ""
    for piece in pieces:
        text += piece
        visible = visible_so_far(text)
        assert visible.startswith(shown), (shown, visible)
        shown = visible
    assert shown == "Hello there <b>x"
    assert "plan" not in shown and "<th" not in shown
    # A closing tag and the answer in one piece: the answer is shown.
    assert visible_so_far("<think>x</think>Answer") == "Answer"
