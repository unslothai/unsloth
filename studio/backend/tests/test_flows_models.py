# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest
from core.flows.nodes import _common
from studio_mcp import loading


def _with_status(slots):
    async def resident(caller):
        return await slots(caller), {}

    return resident


class _Ctx:
    def __init__(self):
        self.resident = {}
        self.statuses = []
        self.caller = object()

    def set_status(
        self,
        status,
        message = None,
    ):
        self.statuses.append(status)


def test_a_loaded_repo_is_reused_even_when_the_node_names_a_variant(monkeypatch):
    loads = []
    resident = {"llm": []}

    async def fake_load_llm(caller, ctx, **kwargs):
        loads.append(kwargs)
        resident["llm"] = [kwargs["model"]]
        resident["variant"] = kwargs["variant"]

    async def fake_slots(caller):
        return {"llm": resident["llm"]}

    async def fake_resident(caller):
        return {"llm": resident["llm"]}, {"gguf_variant": resident.get("variant")}

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", fake_resident)
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    ctx = _Ctx()
    choice = {"repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "variant": "Q4_K_M"}

    async def run():
        await _common.ensure_model(ctx, "chat", choice)
        await _common.ensure_model(ctx, "chat", choice)
        resident["llm"] = ["someone/else-GGUF"]
        await _common.ensure_model(ctx, "chat", choice)

    asyncio.run(run())
    assert len(loads) == 2
    assert loads[0]["max_seq_length"] == _common.FLOW_CHAT_CONTEXT


def test_a_different_variant_of_the_loaded_repo_is_loaded(monkeypatch):
    loads = []
    resident = {"llm": []}

    async def fake_load_llm(caller, ctx, **kwargs):
        loads.append(kwargs["variant"])
        resident["llm"] = [kwargs["model"]]
        resident["variant"] = kwargs["variant"]

    async def fake_slots(caller):
        return {"llm": resident["llm"]}

    async def fake_resident(caller):
        return {"llm": resident["llm"]}, {"gguf_variant": resident.get("variant")}

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", fake_resident)
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    ctx = _Ctx()
    repo = "unsloth/Qwen3-4B-Instruct-2507-GGUF"

    async def run():
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q4_K_M"})
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q8_0"})
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q8_0"})

    asyncio.run(run())
    assert loads == ["Q4_K_M", "Q8_0"]


def test_a_variant_loaded_outside_flows_is_seen(monkeypatch):
    repo = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    loads = []
    status = {"gguf_variant": "Q8_0"}

    async def fake_load_llm(caller, ctx, **kwargs):
        loads.append(kwargs["variant"])
        status["gguf_variant"] = kwargs["variant"]

    async def fake_slots(caller):
        return {"llm": [repo]}

    async def fake_resident(caller):
        return {"llm": [repo]}, dict(status)

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", fake_resident)
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {"llm": (repo.lower(), "q4_k_m")})
    ctx = _Ctx()

    async def run():
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q4_K_M"})
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q4_K_M"})

    asyncio.run(run())
    assert loads == ["Q4_K_M"]


def test_the_plain_json_retry_can_be_stopped(monkeypatch):
    from core.flows.registry import FlowNodeError

    async def no_load(ctx, kind, model):
        return None

    monkeypatch.setattr(_common, "ensure_model", no_load)

    class Ctx:
        hooks = []
        sent = []
        cancelled = []

        def on_cancel(self, fn):
            self.hooks.append(fn)

        def log(self, *a, **k):
            pass

        async def call(
            self,
            method,
            path,
            json = None,
            **kw,
        ):
            if path == "/api/inference/cancel":
                self.cancelled.append(json["cancel_id"])
                return {}
            self.sent.append(json["cancel_id"])
            if "response_format" in json:
                raise FlowNodeError("response_format is not supported", status = 400)
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    ctx = Ctx()

    async def run():
        out = await _common.llm_json(
            ctx, model = None, system = "", prompt = "p", schema = {"type": "object"}
        )
        for hook in ctx.hooks:
            await hook()
        return out

    assert asyncio.run(run()) == {"ok": True}
    assert len(set(ctx.sent)) == 2 and set(ctx.sent) <= set(ctx.cancelled)


def test_retry_bad_json_false_sends_one_request_for_an_unparsable_answer(monkeypatch):
    import pytest

    from core.flows.registry import FlowNodeError

    async def no_load(ctx, kind, model):
        return None

    monkeypatch.setattr(_common, "ensure_model", no_load)

    class Ctx:
        def __init__(self, reject_schema):
            self.hooks, self.posts, self.reject_schema = [], [], reject_schema

        def on_cancel(self, fn):
            self.hooks.append(fn)

        def log(self, *a, **k):
            pass

        async def call(
            self,
            method,
            path,
            json = None,
            **kw,
        ):
            if path == "/api/inference/cancel":
                return {}
            self.posts.append("response_format" in json)
            if self.reject_schema and "response_format" in json:
                raise FlowNodeError("response_format is not supported", status = 400)
            return {"choices": [{"message": {"content": "not json at all"}}]}

    def ask(ctx, **kw):
        return asyncio.run(
            _common.llm_json(
                ctx, model = None, system = "", prompt = "p", schema = {"type": "object"}, **kw
            )
        )

    once = Ctx(False)
    with pytest.raises(FlowNodeError):
        ask(once, retry_bad_json = False)
    assert once.posts == [True]
    retried = Ctx(False)
    with pytest.raises(FlowNodeError):
        ask(retried)
    assert retried.posts == [True, False], "default keeps the plain JSON retry"
    no_schema = Ctx(True)
    with pytest.raises(FlowNodeError):
        ask(no_schema, retry_bad_json = False)
    assert no_schema.posts == [
        True,
        False,
    ], "a server without structured output is still asked once in plain JSON"


def test_a_media_variant_reloaded_outside_flows_is_not_reused(monkeypatch):
    from core.inference import gpu_arbiter

    repo = "unsloth/FLUX.2-klein-4B-GGUF"
    loads = []
    counts = {"diffusion": 0}

    async def fake_load_media(caller, ctx, **kwargs):
        loads.append(kwargs["variant"])
        counts["diffusion"] += 1

    async def fake_slots(caller):
        return {"image": [repo]}

    monkeypatch.setattr(loading, "load_media", fake_load_media)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", _with_status(fake_slots))
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(gpu_arbiter, "load_count", lambda owner = None: counts.get(owner, 0))
    choice = {"repo": repo, "variant": "Q4_K_M"}

    async def run():
        await _common.ensure_model(_Ctx(), "image", choice)
        await _common.ensure_model(_Ctx(), "image", choice)
        assert loads == ["Q4_K_M"]
        # Studio reloads the same repo as Q8_0 outside Flows.
        counts["diffusion"] += 1
        await _common.ensure_model(_Ctx(), "image", choice)

    asyncio.run(run())
    assert loads == ["Q4_K_M", "Q4_K_M"]


class _ImageCtx(_Ctx):
    def __init__(self, status):
        super().__init__()
        self.status = status

    async def call(
        self,
        method,
        path,
        json = None,
    ):
        if path == _common.IMAGE_STATUS:
            return self.status
        if path == "/api/inference/images/generate":
            return {"images": [{"id": "out1"}]}
        return {}

    def on_cancel(self, fn):
        pass

    def progress(self, *args, **kwargs):
        pass

    def log(
        self,
        message,
        level = "info",
    ):
        pass


def test_flows_own_lora_bake_reload_keeps_the_variant_record(monkeypatch):
    from core.inference import gpu_arbiter

    repo = "unsloth/FLUX.2-klein-4B-GGUF"
    loads = []
    counts = {"diffusion": 0}

    async def fake_load_media(caller, ctx, **kwargs):
        loads.append(kwargs.get("knobs"))
        counts["diffusion"] += 1

    async def fake_slots(caller):
        return {"image": [repo]}

    monkeypatch.setattr(loading, "load_media", fake_load_media)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", _with_status(fake_slots))
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(_common, "_BAKED_LORAS", {})
    monkeypatch.setattr(gpu_arbiter, "load_count", lambda owner = None: counts.get(owner, 0))
    # An auto GGUF pick served as an int8 build: LoRAs are baked in by a second load.
    status = {
        "loaded": True,
        "repo_id": repo,
        "resolved": {"transformer_quant": {"requested": "int8", "value": "int8"}},
    }
    model = {"repo": repo, "variant": "Q4_K_M"}
    loras = [{"id": "someone/flux2-lora", "weight": 1.0}]
    per_node = []
    for _ in range(3):
        before = len(loads)
        asyncio.run(
            _common.generate_image(
                _ImageCtx(status), model = model, prompt = "a fox", options = {"loras": loras}
            )
        )
        per_node.append(len(loads) - before)
    assert per_node == [2, 0, 0], per_node


def test_streamed_reload_of_the_resident_model_keeps_its_gguf_variant(monkeypatch):
    repo = "unsloth/FLUX.2-klein-4B-GGUF"
    loads = []

    async def fake_load_media(caller, ctx, **kwargs):
        loads.append((kwargs.get("model"), kwargs.get("variant"), kwargs.get("knobs")))

    monkeypatch.setattr(loading, "load_media", fake_load_media)
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(_common, "_BAKED_LORAS", {})
    status = {"loaded": True, "repo_id": repo, "gguf_variant": "Q4_K_M"}
    assert asyncio.run(_common._reload_with_loras(_ImageCtx(status), None, []))
    assert loads == [(repo, "Q4_K_M", {})]


class _BakedPipe:
    def set_adapters(
        self,
        names,
        adapter_weights = None,
    ):
        self.active = (list(names), list(adapter_weights))


def _run_baked_nodes(monkeypatch, sets, placement):
    """Run real generate_image + _apply_loras on a stubbed int8 load; returns each node's media loads."""
    import threading
    from types import SimpleNamespace

    from core.flows.registry import FlowNodeError
    from core.inference import diffusion_lora, diffusion_memory, gpu_arbiter
    from core.inference.diffusion import DiffusionBackend

    fixed = getattr(diffusion_memory, "StreamedLoraWeightsFixedError", ())
    tag = getattr(diffusion_memory, "IMAGE_REFUSAL_STREAMED_LORA", None)
    repo = "unsloth/FLUX.2-klein-4B-GGUF"
    backend = DiffusionBackend()
    loads, counts, held = [], {gpu_arbiter.DIFFUSION: 0}, {}

    async def fake_load_media(caller, ctx, **kwargs):
        knobs = kwargs.get("knobs") or {}
        loads.append(knobs)
        counts[gpu_arbiter.DIFFUSION] += 1
        baked = knobs.get("loras") or []
        pipe = _BakedPipe()
        pipe._unsloth_loras_baked = bool(baked)
        pipe._unsloth_loras = tuple(
            (diffusion_lora.sanitize_alias(r["id"]), f"/{r['id']}.safetensors", r["weight"])
            for r in baked
        )
        held["state"] = SimpleNamespace(
            pipe = pipe,
            transformer_quant = "int8",
            kind = "gguf",
            hf_token = None,
            family = SimpleNamespace(name = "flux.2-klein"),
            cuda_graphs = (SimpleNamespace(placement = SimpleNamespace(mode = placement)),),
        )

    async def fake_slots(caller):
        return {"image": [repo] if held else []}

    class Ctx(_ImageCtx):
        async def call(
            self,
            method,
            path,
            json = None,
        ):
            if path == _common.IMAGE_STATUS:
                return {
                    "loaded": bool(held),
                    "repo_id": repo,
                    "resolved": {"transformer_quant": {"requested": "int8", "value": "int8"}},
                }
            if path == "/api/inference/images/generate":
                specs = [(r["id"], r["weight"]) for r in (json.get("loras") or [])]
                try:
                    backend._apply_loras(held["state"], specs or None, threading.Event())
                except ValueError as exc:
                    refusal = tag if isinstance(exc, fixed) else None
                    raise FlowNodeError(str(exc), status = 400, refusal = refusal) from None
                return {"images": [{"id": "out1"}]}
            return {}

    monkeypatch.setattr(loading, "load_media", fake_load_media)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", _with_status(fake_slots))
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(_common, "_BAKED_LORAS", {})
    monkeypatch.setattr(gpu_arbiter, "load_count", lambda owner = None: counts.get(owner, 0))
    monkeypatch.setattr(
        diffusion_lora,
        "resolve_specs",
        lambda specs, **kwargs: [
            diffusion_lora.ResolvedLora(
                i, diffusion_lora.sanitize_alias(i), f"/{i}.safetensors", "safetensors", w
            )
            for i, w in specs
            if w != 0
        ],
    )
    model = {"repo": repo, "variant": "Q4_K_M"}
    per_node = []
    for loras in sets:
        before = len(loads)
        asyncio.run(
            _common.generate_image(Ctx(None), model = model, prompt = "a fox", options = {"loras": loras})
        )
        per_node.append(loads[before:])
    return per_node


FOX = "someone/flux2-lora"
INK = "someone/flux2-ink"
BOTH = [{"id": INK, "weight": 1.0}, {"id": FOX, "weight": 1.0}]
ONE_CHANGED = [{"id": INK, "weight": 0.5}, {"id": FOX, "weight": 1.0}]
_CHANGES = {
    "weight": ([{"id": FOX, "weight": 1.0}], [{"id": FOX, "weight": 0.5}]),
    "removed": ([{"id": FOX, "weight": 1.0}], []),
    "one_of_two": (BOTH, ONE_CHANGED),
}


@pytest.mark.parametrize("change", sorted(_CHANGES))
def test_streamed_bake_change_reloads_once_with_the_new_set(monkeypatch, change):
    first, second = _CHANGES[change]
    nodes = _run_baked_nodes(monkeypatch, [first, second], "group")
    # The resident int8 build again, with the new set (none: a plain reload).
    build = {"transformer_quant": "int8"}
    assert nodes[1] == [{**build, "loras": second} if second else build], nodes


@pytest.mark.parametrize("change", sorted(_CHANGES))
def test_model_placement_bake_change_adjusts_in_place(monkeypatch, change):
    first, second = _CHANGES[change]
    nodes = _run_baked_nodes(monkeypatch, [first, second], "model")
    assert nodes[1] == [], nodes


def test_streamed_bake_subset_reloads_with_the_subset(monkeypatch):
    nodes = _run_baked_nodes(monkeypatch, [BOTH, [{"id": FOX, "weight": 1.0}]], "group")
    assert nodes[1] == [{"transformer_quant": "int8", "loras": [{"id": FOX, "weight": 1.0}]}], nodes


@pytest.mark.parametrize("placement", ["group", "model"])
def test_baked_unchanged_set_does_not_reload(monkeypatch, placement):
    fox = [{"id": FOX, "weight": 1.0}]
    nodes = _run_baked_nodes(monkeypatch, [fox, fox, fox], placement)
    assert [len(n) for n in nodes] == [2, 0, 0], nodes


def test_streamed_lora_refusal_is_tagged_through_the_flows_caller():
    import httpx
    from core.flows import caller as flow_caller
    from core.inference import diffusion, diffusion_memory
    from types import SimpleNamespace

    assert _common.STREAMED_LORA_REFUSAL == diffusion_memory.IMAGE_REFUSAL_STREAMED_LORA
    state = SimpleNamespace(cuda_graphs = (SimpleNamespace(placement = SimpleNamespace(mode = "group")),))
    with pytest.raises(diffusion_memory.StreamedLoraWeightsFixedError):
        diffusion._refuse_weight_change_on_streamed_graph(state)
    assert issubclass(diffusion_memory.StreamedLoraWeightsFixedError, ValueError)
    response = httpx.Response(
        400,
        json = {"detail": "fixed"},
        headers = {
            diffusion_memory.IMAGE_REFUSAL_HEADER: diffusion_memory.IMAGE_REFUSAL_STREAMED_LORA
        },
    )
    with pytest.raises(flow_caller.RouteError) as raised:
        flow_caller.parse_body(response)
    assert raised.value.refusal == _common.STREAMED_LORA_REFUSAL


@pytest.mark.parametrize("named", [None, {"repo": "someone/flux2-finetune-GGUF"}])
def test_a_streamed_reload_rebuilds_the_resident_build(monkeypatch, named):
    # The node names the resident repo without its quant, or names nothing: either way the reload is the
    # resident build (its exact file, precision, memory and speed settings), not the repo's defaults. An automatic
    # precision that engaged fp8 is asked automatically again: an explicit fp8 is refused under balanced offload.
    repo = "someone/flux2-finetune-GGUF"
    loads = []

    async def fake_load_media(caller, ctx, **kwargs):
        loads.append(kwargs)

    monkeypatch.setattr(loading, "load_media", fake_load_media)
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(_common, "_BAKED_LORAS", {})
    status = {
        "loaded": True,
        "repo_id": repo,
        "gguf_variant": "Q8_0",
        "model_kind": "gguf",
        "gguf_filename": "flux2-finetune-Q8_0.gguf",
        "base_repo": "black-forest-labs/FLUX.2-klein-4B",
        "transformer_quant": "auto",
        "resolved": {
            "transformer_quant": {"value": "fp8"},
            "text_encoder_quant": {"requested": "int8", "value": "int8"},
        },
        "text_encoder_quant": "int8",
        "memory_mode": "low_vram",
        "speed_mode": "max",
        "cpu_offload": False,
    }
    assert asyncio.run(_common._reload_with_loras(_ImageCtx(status), named, []))
    (load,) = loads
    assert load["model"] == repo and load["variant"] == "Q8_0"
    assert load["knobs"] == {
        "model_kind": "gguf",
        "gguf_filename": "flux2-finetune-Q8_0.gguf",
        "base_repo": "black-forest-labs/FLUX.2-klein-4B",
        "cpu_offload": False,
        "memory_mode": "low_vram",
        "speed_mode": "max",
        "text_encoder_quant": "int8",
    }


def test_a_single_file_resident_reloads_as_that_file():
    status = {
        "loaded": True,
        "repo_id": "someone/flux-single",
        "model_kind": "single_file",
        "gguf_filename": "flux-single.safetensors",
        "base_repo": "black-forest-labs/FLUX.1-dev",
        "memory_mode": "not-a-mode",
    }
    knobs = _common.resident_build_knobs(status)
    assert knobs["model_kind"] == "single_file"
    assert knobs["gguf_filename"] == "flux-single.safetensors"
    assert knobs["base_repo"] == "black-forest-labs/FLUX.1-dev"
    # A value no load request takes is left out, not sent to fail validation.
    assert "memory_mode" not in knobs
    # A pipeline is its own base.
    assert "base_repo" not in _common.resident_build_knobs(
        {"repo_id": "org/pipe", "model_kind": "pipeline", "base_repo": "org/pipe"}
    )


def test_a_reload_keeps_a_dense_text_encoder_dense():
    # "off" reports a null request and an engaged "off": unset, the reload would pick the family's auto int8.
    status = {
        "loaded": True,
        "repo_id": "Qwen/Qwen-Image",
        "model_kind": "pipeline",
        "text_encoder_quant": None,
        "resolved": {
            "text_encoder_quant": {"value": "off"},
            "transformer_quant": {"requested": "int8", "value": "int8"},
        },
    }
    knobs = _common.resident_build_knobs(status)
    assert knobs["text_encoder_quant"] == "off" and knobs["transformer_quant"] == "int8"
    # The cards the load named, physical ids as asked; an automatic pick stays automatic.
    assert _common.resident_build_knobs({**status, "gpu_ids": [1]})["gpu_ids"] == [1]
    assert "gpu_ids" not in _common.resident_build_knobs(status)
    # An automatic fp8 text encoder (T5 cast, CLIP-L kept dense) is asked automatically again: an explicit fp8
    # would cast CLIP-L too.
    status.update(text_encoder_quant = "fp8")
    status["resolved"]["text_encoder_quant"] = {"requested": None, "value": "fp8"}
    assert "text_encoder_quant" not in _common.resident_build_knobs(status)


def test_a_reload_keeps_the_supplied_text_encoder_and_vae_files():
    status = {
        "loaded": True,
        "repo_id": "/models/diffusion_models/flux1-dev-Q8_0.gguf",
        "model_kind": "gguf",
        "component_files": {"text_encoder_2": "t5xxl_fp8.safetensors", "vae": "ae.safetensors"},
        "component_specs": {
            "text_encoder": "/models/text_encoders/clip_l.safetensors",
            "text_encoder_2": "/models/text_encoders/t5xxl_fp8.safetensors",
            "vae": "/models/vae/ae.safetensors",
        },
    }
    knobs = _common.resident_build_knobs(status)
    assert knobs["text_encoder_file"] == [
        "/models/text_encoders/clip_l.safetensors",
        "/models/text_encoders/t5xxl_fp8.safetensors",
    ]
    assert knobs["vae_file"] == "/models/vae/ae.safetensors"
    one = _common.resident_build_knobs(
        {**status, "component_specs": {"text_encoder": "org/te/te.safetensors"}}
    )
    assert one["text_encoder_file"] == "org/te/te.safetensors" and "vae_file" not in one


@pytest.mark.parametrize("asked", [None, "IQ4_XS-3.97bpw"])
def test_a_bpw_resident_reloads_under_its_own_variant(monkeypatch, asked):
    # Status names the bare quant (IQ4_XS); the Hub lists the 3.53 and 3.97 bpw builds apart.
    repo = "someone/flux2-GGUF"
    loads = []

    async def fake_load_media(caller, ctx, **kwargs):
        loads.append(kwargs)

    monkeypatch.setattr(loading, "load_media", fake_load_media)
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    monkeypatch.setattr(_common, "_BAKED_LORAS", {})
    status = {
        "loaded": True,
        "repo_id": repo,
        "gguf_variant": "IQ4_XS",
        "model_kind": "gguf",
        "gguf_filename": "flux2-IQ4_XS-3.97bpw.gguf",
        "transformer_quant": "int8",
    }
    named = {"repo": repo, "variant": asked} if asked else None
    assert asyncio.run(_common._reload_with_loras(_ImageCtx(status), named, []))
    (load,) = loads
    assert load["variant"] == "IQ4_XS-3.97bpw"
    assert load["knobs"]["gguf_filename"] == "flux2-IQ4_XS-3.97bpw.gguf"
    loads.clear()
    asyncio.run(_common._baked_loras_ready(_ImageCtx(status), named, [{"id": FOX, "weight": 1.0}]))
    (load,) = loads
    assert load["variant"] == "IQ4_XS-3.97bpw"
    assert load["knobs"]["gguf_filename"] == "flux2-IQ4_XS-3.97bpw.gguf"


def test_a_reload_keeps_an_explicit_cache_off_and_the_built_family():
    # Status reports an explicit cache "off" only in its resolved record, and a family picked by hand only as the
    # family: unset, the reload would turn on the auto cache and re-detect a sibling family from the file.
    status = {
        "loaded": True,
        "repo_id": "/models/diffusion_models/qwen-edit.safetensors",
        "model_kind": "single_file",
        "family": "qwen-image-edit",
        "transformer_cache": None,
        "resolved": {
            "transformer_cache": {"requested": "off", "value": "off"},
            "family_override": {"requested": "qwen-image-edit", "value": "qwen-image-edit"},
        },
    }
    knobs = _common.resident_build_knobs(status)
    assert knobs["transformer_cache"] == "off"
    assert knobs["family_override"] == "qwen-image-edit"
    # Auto asked: left to the policy again, which tunes what it engages (an explicit "static" takes the defaults).
    status.update(transformer_cache = "static", attention_backend = "flash")
    status["resolved"]["transformer_cache"] = {"requested": None, "value": "static"}
    status["resolved"]["attention_backend"] = {"requested": None, "value": "flash"}
    knobs = _common.resident_build_knobs(status)
    assert "transformer_cache" not in knobs and "attention_backend" not in knobs
    status["resolved"]["transformer_cache"] = {"requested": "static", "value": "static"}
    assert _common.resident_build_knobs(status)["transformer_cache"] == "static"
