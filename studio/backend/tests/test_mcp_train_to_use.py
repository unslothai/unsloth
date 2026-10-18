# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_model(checkpoint=...) and load_model(export=...): the load goes out with a ref minted from the listed path, and the agent sees only the name it asked for."""

import asyncio
import json

import pytest

from hub.utils.host_paths import cache_reference, resolve_host_path_reference
from studio_mcp import checkpoints, trained_models

from .mcp_harness import OUT, RUN_CHECKPOINTS, bodies, make_caller, run_tool

EXPORTS = "/srv/unsloth/exports"
LORAS = {
    "outputs_dir": OUT,
    "loras": [
        {"display_name": "qwen-lora", "adapter_path": f"{OUT}/qwen-lora", "source": "training"},
        {
            "display_name": "qwen-lora / adapter",
            "adapter_path": f"{EXPORTS}/qwen-lora/adapter",
            "source": "exported",
            "export_type": "lora",
        },
        {
            "display_name": "qwen-lora / merged",
            "adapter_path": f"{EXPORTS}/qwen-lora/merged",
            "source": "exported",
            "export_type": "merged",
        },
        {
            "display_name": "qwen-GGUF",
            "adapter_path": f"{EXPORTS}/qwen-GGUF/qwen-Q8_0.gguf",
            "source": "exported",
            "export_type": "gguf",
        },
    ],
}


def _routes(evicted = ()):
    def load(request, body):
        sent = json.loads(body)["model_path"]
        return {"status": "loaded", "model": sent, "display_name": "leaf", "evicted": list(evicted)}

    return {
        ("GET", "/api/models/checkpoints"): RUN_CHECKPOINTS,
        ("GET", "/api/models/loras"): LORAS,
        ("POST", "/api/inference/load"): load,
        ("GET", "/api/inference/load-progress"): {},
        ("GET", "/api/models/download-progress"): {},
        ("GET", "/api/inference/status"): {"loaded": [], "serving": [], "serving_checkpoints": []},
    }


def _load(
    monkeypatch,
    args,
    evicted = (),
):
    result, studio = run_tool(monkeypatch, _routes(evicted), "load_model", args)
    assert result.get("isError") is not True, result
    text = json.dumps(result)
    assert "/srv" not in text and "ref:" not in text
    (body,) = bodies(studio, "/api/inference/load")
    return result["structuredContent"], body, studio


@pytest.mark.parametrize(
    "name,path",
    [
        ("qwen-lora", f"{OUT}/qwen-lora"),
        ("qwen-lora/checkpoint-30", f"{OUT}/qwen-lora/checkpoint-30"),
    ],
)
def test_a_checkpoint_loads_by_ref_and_reports_its_name(monkeypatch, name, path):
    output, body, _studio = _load(monkeypatch, {"checkpoint": name})
    assert body["model_path"] == cache_reference(path)
    assert resolve_host_path_reference(body["model_path"]) == path
    assert output["model"] == name and output["kind"] == "llm"


def test_an_evicted_checkpoint_is_named(monkeypatch):
    evicted = [cache_reference(f"{OUT}/qwen-lora"), "unsloth/Qwen3-0.6B"]
    output, _body, _studio = _load(
        monkeypatch, {"checkpoint": "qwen-lora-2/checkpoint-30"}, evicted
    )
    assert output["evicted"] == ["qwen-lora", "unsloth/Qwen3-0.6B"]


@pytest.mark.parametrize(
    "args,path,variant",
    [
        ({"export": "qwen-lora/adapter"}, f"{EXPORTS}/qwen-lora/adapter", None),
        ({"export": "QWEN-LORA/merged"}, f"{EXPORTS}/qwen-lora/merged", None),
        # The file name gives the quantization, so the folder loads with it as the variant.
        ({"export": "qwen-GGUF"}, f"{EXPORTS}/qwen-GGUF", "Q8_0"),
        ({"export": "qwen-GGUF", "variant": "Q4_K_M"}, f"{EXPORTS}/qwen-GGUF", "Q4_K_M"),
    ],
)
def test_an_export_loads_by_ref_and_reports_its_name(monkeypatch, args, path, variant):
    output, body, _studio = _load(monkeypatch, args)
    assert resolve_host_path_reference(body["model_path"]) == path
    assert body.get("gguf_variant") == variant
    assert output["model"] == args["export"]


def test_is_lora_follows_the_listing(monkeypatch):
    async def fake_route(method, path, **kwargs):
        return RUN_CHECKPOINTS if path == "/api/models/checkpoints" else LORAS

    monkeypatch.setattr(checkpoints, "route_json", fake_route)
    monkeypatch.setattr(trained_models, "route_json", fake_route)
    caller = make_caller()

    async def resolved():
        return [
            await trained_models.resolve_checkpoint(caller, "qwen-lora/checkpoint-30"),
            await trained_models.resolve_checkpoint(caller, "qwen-lora-2/checkpoint-30"),
            *[
                await trained_models.resolve_export(caller, name, "Q4_K_M")
                for name in ("qwen-lora/adapter", "qwen-lora/merged", "qwen-GGUF")
            ],
        ]

    found = asyncio.run(resolved())
    # qwen-lora-2 has no peft_type: a full finetune, not a LoRA.
    assert [t.is_lora for t in found] == [True, False, True, False, False]
    assert [t.gguf_variant for t in found] == [None, None, None, None, "Q4_K_M"]


@pytest.mark.parametrize(
    "args,message",
    [
        ({"checkpoint": "nope"}, "No checkpoint named nope. Available: qwen-lora, qwen-lora-2/"),
        (
            {"export": "nope"},
            'has two levels, such as "<run>/<name>". Available: qwen-GGUF, qwen-lora/',
        ),
        ({"export": "qwen-lora"}, "No export named qwen-lora."),
    ],
)
def test_an_unknown_name_lists_what_there_is(monkeypatch, args, message):
    result, studio = run_tool(monkeypatch, _routes(), "load_model", args)
    assert result["isError"] is True
    assert message in result["content"][0]["text"]
    assert "/srv" not in result["content"][0]["text"]
    assert bodies(studio, "/api/inference/load") == []


@pytest.mark.parametrize(
    "args,message",
    [
        ({}, "load_model needs exactly one of model, checkpoint or export."),
        ({"model": "m", "checkpoint": "qwen-lora"}, "needs exactly one of"),
        ({"checkpoint": "qwen-lora", "export": "qwen-GGUF"}, "needs exactly one of"),
        *[
            ({key: "qwen-lora", "kind": kind}, 'use kind "llm", "tts" or "audio".')
            for key in ("checkpoint", "export")
            for kind in ("image", "video", "stt")
        ],
    ],
)
def test_a_refused_load_reaches_no_route(monkeypatch, args, message):
    result, studio = run_tool(monkeypatch, {}, "load_model", args, fallback = {})
    assert result["isError"] is True
    assert message in result["content"][0]["text"]
    assert studio.state.calls == []


@pytest.mark.parametrize("kind", ["tts", "audio"])
def test_a_checkpoint_keeps_the_audio_kind(monkeypatch, kind):
    output, _body, _studio = _load(monkeypatch, {"checkpoint": "qwen-lora", "kind": kind})
    assert output["kind"] == kind and output["model"] == "qwen-lora"


def test_a_plain_model_load_lists_nothing(monkeypatch):
    output, body, studio = _load(monkeypatch, {"model": "unsloth/Qwen3-0.6B"})
    assert body["model_path"] == "unsloth/Qwen3-0.6B" and output["model"] == "unsloth/Qwen3-0.6B"
    assert not {"/api/models/checkpoints", "/api/models/loras"} & {c[1] for c in studio.state.calls}


def test_a_gguf_export_loads_on_the_cpu_as_gguf(monkeypatch):
    _output, body, _studio = _load(monkeypatch, {"export": "qwen-GGUF", "device": "cpu"})
    assert body["gguf_variant"] == "Q8_0"
    assert (body["gpu_layers"], body["speculative_type"]) == (0, "off")


def test_a_gguf_export_with_no_quant_in_its_name_needs_a_variant_to_place(monkeypatch):
    unnamed = {
        "display_name": "assistant",
        "adapter_path": f"{EXPORTS}/assistant/model.gguf",
        "source": "exported",
        "export_type": "gguf",
    }
    routes = {**_routes(), ("GET", "/api/models/loras"): {**LORAS, "loras": [unnamed]}}
    args = {"export": "assistant", "device": "cpu"}
    result, studio = run_tool(monkeypatch, routes, "load_model", args)
    assert result["isError"] is True
    assert "Pass variant" in result["content"][0]["text"]
    assert bodies(studio, "/api/inference/load") == []


def test_a_checkpoint_is_not_placed_on_the_cpu(monkeypatch):
    result, studio = run_tool(
        monkeypatch, _routes(), "load_model", {"checkpoint": "qwen-lora", "device": "cpu"}
    )
    assert result["isError"] is True
    assert "device='cpu' needs a GGUF model" in result["content"][0]["text"]
    assert bodies(studio, "/api/inference/load") == []


def test_a_load_that_outlasts_wait_s_is_named_in_its_job_result():
    from studio_mcp import loading
    from studio_mcp.outputs import LoadResult

    ref = cache_reference(f"{OUT}/qwen-lora")
    trained = trained_models.TrainedModel(
        name = "qwen-lora", ref = ref, is_lora = True, known = {ref: "qwen-lora"}
    )
    caller = make_caller()

    async def go():
        gate = asyncio.Event()

        async def run(_job):
            await gate.wait()
            return LoadResult(kind = "llm", model = ref, evicted = [ref])

        job = loading.LOADS.start(
            loading.LoadJob(job_id = "j-rename", account_id = caller.account_id), run
        )
        trained_models.rename_job(caller, "j-rename", trained)
        gate.set()
        await asyncio.wait({job.task})
        return job.result

    result = asyncio.run(go())
    assert (result.model, result.evicted) == ("qwen-lora", ["qwen-lora"])
