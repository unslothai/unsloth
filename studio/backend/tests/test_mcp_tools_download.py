# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""download_model's logic against a fake Unsloth Studio: start bodies, ids, status, cancel and the pre-load wait."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.responses import JSONResponse
from fastmcp.exceptions import ToolError

from studio_mcp import loading, model_downloads

from .mcp_harness import (
    WHISPER,
    bodies,
    fake_studio,
    make_caller,
    queries,
    run_tool,
    sequence,
    stt_state,
)

GGUF = "unsloth/Qwen3-0.6B-GGUF"
VARIANTS = {"default_variant": "Q4_K_M", "variants": [{"quant": "Q8_0"}, {"quant": "Q4_K_M"}]}
STARTED = {"state": "running", "accepted": True, "attached": False}
START, STATUS, CANCEL = "/api/hub/download", "/api/hub/download-status", "/api/hub/download/cancel"
NOT_YOURS = JSONResponse({"detail": "Download not found"}, status_code = 404)
FLUX, VAE = "unsloth/FLUX.1-dev-GGUF", "black-forest-labs/FLUX.1-dev"
PLAN = {
    "entries": [
        {"repo_id": FLUX, "files": ["flux-q4.gguf"], "checkpoint": True},
        {"repo_id": VAE, "files": ["vae/a.safetensors"]},
        {"repo_id": "empty/companion", "files": []},
    ],
}
FLUX_ID = f"{FLUX}::@diffusion,{VAE}::@diffusion"


class Ctx:
    def __init__(self):
        self.messages = []

    async def report_progress(
        self,
        progress,
        total = None,
        message = None,
    ):
        self.messages.append(message)


@pytest.fixture(autouse = True)
def fresh_memory(monkeypatch):
    monkeypatch.setattr(model_downloads, "_remembered", model_downloads.OrderedDict())
    monkeypatch.setattr(model_downloads, "WAIT_INTERVAL_S", 0.01)
    monkeypatch.setattr(loading, "POLL_INTERVAL_S", 0.01)


def run(studio, fn, *args, **kwargs):
    return asyncio.run(fn(make_caller(studio_app = studio), *args, **kwargs))


def start(studio, **kwargs):
    return run(studio, model_downloads.start, None, **{"variant": None, "hf_token": None, **kwargs})


def test_safetensors_sends_the_repo_auto_transport_and_the_hub_header():
    studio = fake_studio({("POST", START): STARTED})
    ref = start(studio, model = "unsloth/Qwen3-0.6B", kind = "llm", hf_token = "hf_x")
    assert bodies(studio, START) == [{"repo_id": "unsloth/Qwen3-0.6B", "transport_mode": "auto"}]
    assert studio.state.calls[0][2]["x-unsloth-hf-token"] == "hf_x"
    assert (ref.id, ref.status, ref.attached) == ("unsloth/Qwen3-0.6B", "running", False)


@pytest.mark.parametrize("kind", ["llm", "tts", "audio"])
def test_a_gguf_repo_downloads_only_one_variant(kind):
    studio = fake_studio({("GET", "/api/hub/gguf-variants"): VARIANTS, ("POST", START): STARTED})
    ref = start(studio, model = GGUF, kind = kind)
    assert bodies(studio, START) == [
        {"repo_id": GGUF, "transport_mode": "auto", "gguf_variant": "Q4_K_M"}
    ]
    assert (ref.id, ref.variant) == (f"{GGUF}::Q4_K_M", "Q4_K_M")
    assert start(studio, model = "org/repo", kind = kind, variant = "Q8_0").id == "org/repo::Q8_0"
    # No recommended variant: the first one listed.
    first = {"variants": [{"quant": "Q2_K"}]}
    studio = fake_studio({("GET", "/api/hub/gguf-variants"): first, ("POST", START): STARTED})
    assert start(studio, model = GGUF, kind = kind).variant == "Q2_K"


def test_a_gguf_repo_with_no_variant_is_refused_before_any_download():
    studio = fake_studio({("GET", "/api/hub/gguf-variants"): {"variants": []}}, fallback = STARTED)
    with pytest.raises(ToolError, match = "no GGUF variant"):
        start(studio, model = GGUF, kind = "llm")
    assert bodies(studio, START) == []


def test_a_busy_repo_names_the_other_download_and_an_attach_is_fine():
    busy = {"accepted": False, "attached": False, "state": "cancelling"}
    with pytest.raises(ToolError, match = "another download of it is cancelling"):
        start(fake_studio({("POST", START): busy}), model = "org/repo", kind = "llm")
    joined = {"accepted": False, "attached": True, "state": "running"}
    assert start(fake_studio({("POST", START): joined}), model = "org/repo", kind = "llm").attached


@pytest.mark.parametrize("kind, prefix", [("image", "images"), ("video", "video")])
def test_diffusion_stages_each_plan_entry_scoped(kind, prefix):
    listing = {
        "default_variant": "Q4_0",
        "variants": [{"quant": "Q4_0", "filename": "flux-q4.gguf"}],
    }
    studio = fake_studio(
        {
            ("GET", "/api/hub/gguf-variants"): listing,
            ("POST", f"/api/inference/{prefix}/download-plan"): PLAN,
            ("POST", START): STARTED,
        }
    )
    ref = start(studio, model = FLUX, kind = kind, hf_token = "hf_x")
    assert bodies(studio, f"/api/inference/{prefix}/download-plan") == [
        {
            "model_path": FLUX,
            "hf_token": "hf_x",
            "gguf_filename": "flux-q4.gguf",
            "model_kind": "gguf",
        }
    ]
    scoped = {"scope_id": "diffusion", "transport_mode": "auto"}
    assert bodies(studio, START) == [
        {"repo_id": FLUX, **scoped, "files": ["flux-q4.gguf"]},
        {"repo_id": VAE, **scoped, "files": ["vae/a.safetensors"]},
    ]
    assert ref.id == FLUX_ID
    assert model_downloads.parse_id(ref.id) == [(FLUX, "@diffusion"), (VAE, "@diffusion")]


@pytest.mark.parametrize(
    "plan, match",
    [
        ({"incompatible_reason": "FLUX.2 GGUF needs the matching base"}, "matching base"),
        ({**PLAN, "plan_failed": True}, "could not list the files"),
    ],
)
def test_a_refused_plan_never_downloads(plan, match):
    studio = fake_studio({("POST", "/api/inference/images/download-plan"): plan}, fallback = STARTED)
    with pytest.raises(ToolError, match = match):
        start(studio, model = "org/flux", kind = "image")
    assert bodies(studio, START) == []


def test_stt_downloads_through_its_own_route_and_has_no_id():
    done = {"downloading": False, "completed_download_ids": ["d1"]}
    status = sequence(
        {"transformers": stt_state()},
        {"transformers": stt_state(downloaded = [WHISPER], download = done)},
    )
    studio = fake_studio(
        {
            ("GET", "/api/inference/audio/stt/status"): status,
            ("POST", "/api/inference/audio/stt/download"): {"download_id": "d1"},
        }
    )
    ref = start(studio, model = WHISPER, kind = "stt")
    assert (ref.id, ref.status) == (None, "completed")
    assert bodies(studio, "/api/inference/audio/stt/download") == [
        {"model": WHISPER, "engine": "transformers"}
    ]
    cached = {"transformers": stt_state(downloaded = [WHISPER])}
    studio = fake_studio({("GET", "/api/inference/audio/stt/status"): cached})
    assert start(studio, model = WHISPER, kind = "stt").status == "completed"
    assert bodies(studio, "/api/inference/audio/stt/download") == []


def test_parse_id_limits():
    assert model_downloads.parse_id("a/b,a/b::Q4_K_M") == [("a/b", None), ("a/b", "Q4_K_M")]
    assert len(model_downloads.parse_id(",".join(["a/b"] * 16))) == 16
    for bad in (",".join(["a/b"] * 17), "", "::Q4", "a/b,", " , a/b"):
        with pytest.raises(ToolError):
            model_downloads.parse_id(bad)


@pytest.mark.parametrize(
    "states, expected",
    [
        (["complete", "error", "cancelling"], "failed"),
        (["running", "cancelling"], "cancelling"),
        (["complete", "running", "cancelled"], "running"),
        (["complete", "cancelled", "idle"], "cancelled"),
        (["complete", "idle"], "idle"),
        (["complete", "complete"], "completed"),
    ],
)
def test_status_aggregates_every_part(states, expected):
    answers = {
        f"org/r{i}": {"state": s, "error": s == "error" and "disk full" or None}
        for i, s in enumerate(states)
    }
    studio = fake_studio({("GET", STATUS): lambda r, b: answers[r.query_params["repo_id"]]})
    job = run(studio, model_downloads.status, ",".join(f"{r}::@diffusion" for r in answers))
    assert (job.kind, job.status, job.download.status) == ("download", expected, expected)
    assert job.error == ("disk full" if "error" in states else None)
    assert job.progress_percent is None
    assert {q["gguf_variant"] for q in queries(studio, STATUS)} == {"@diffusion"}


def test_a_single_safetensors_part_reports_percent_and_a_404_reads_idle():
    studio = fake_studio(
        {
            ("GET", STATUS): {"state": "running"},
            ("GET", "/api/hub/download-progress"): {"progress": 0.25},
        }
    )
    job = run(studio, model_downloads.status, "unsloth/Qwen3-0.6B")
    assert (job.status, job.progress_percent) == ("running", 25.0)
    assert queries(studio, STATUS) == [{"repo_id": "unsloth/Qwen3-0.6B", "gguf_variant": ""}]
    assert (
        run(fake_studio({("GET", STATUS): NOT_YOURS}), model_downloads.status, "o/r").status
        == "idle"
    )


def test_cancel_posts_every_part_and_only_cancelling_counts_as_stopped():
    studio = fake_studio({("POST", CANCEL): sequence({"state": "idle"}, {"state": "cancelling"})})
    assert run(studio, model_downloads.cancel_parts, FLUX_ID) == (True, "cancelling")
    assert bodies(studio, CANCEL) == [
        {"repo_id": FLUX, "gguf_variant": "@diffusion"},
        {"repo_id": VAE, "gguf_variant": "@diffusion"},
    ]
    studio = fake_studio({("POST", CANCEL): {"job_key": "k", "state": "idle"}})
    assert run(studio, model_downloads.cancel_parts, "a/x") == (False, "idle")
    assert bodies(studio, CANCEL) == [{"repo_id": "a/x"}]


def test_the_id_keeps_each_start_generation_and_cancel_sends_it():
    studio = fake_studio({("POST", START): {**STARTED, "generation": 7}})
    ref = start(studio, model = "org/repo", kind = "llm", variant = "Q8_0")
    assert ref.id == "org/repo::Q8_0#7"
    assert model_downloads.parse_id(ref.id) == [("org/repo", "Q8_0")]
    studio = fake_studio({("POST", CANCEL): {"state": "cancelling"}})
    run(studio, model_downloads.cancel_parts, f"{ref.id},{VAE}::@diffusion")
    assert bodies(studio, CANCEL) == [
        {"repo_id": "org/repo", "gguf_variant": "Q8_0", "generation": 7},
        {"repo_id": VAE, "gguf_variant": "@diffusion"},
    ]


def test_status_of_a_replaced_run_names_the_newer_one():
    studio = fake_studio({("GET", STATUS): {"state": "running", "generation": 8}})
    job = run(studio, model_downloads.status, "org/repo::Q8_0#7")
    assert job.status == "idle"
    assert 'id="org/repo::Q8_0#8"' in job.error
    assert run(studio, model_downloads.status, "org/repo::Q8_0#8").status == "running"
    # After a restart the numbers start again: a lower generation is another run too.
    studio = fake_studio({("GET", STATUS): {"state": "running", "generation": 1}})
    assert run(studio, model_downloads.status, "org/repo::Q8_0#7").status == "idle"
    # A registry that pruned the run reads generation 0: that is not another run.
    studio = fake_studio({("GET", STATUS): {"state": "idle", "generation": 0}})
    job = run(studio, model_downloads.status, "org/repo::Q8_0#7")
    assert (job.status, job.error) == ("idle", None)


def test_image_options_go_into_the_staged_plan():
    plan = "/api/inference/images/download-plan"
    studio = fake_studio({("POST", plan): PLAN, ("POST", START): STARTED})
    start(studio, model = "org/FLUX", kind = "image", knobs = {"text_encoder_quant": "fp8"})
    assert bodies(studio, plan)[0]["text_encoder_quant"] == "fp8"


def test_native_tts_stages_its_audio_plan_with_the_codec_repos():
    higgs, codec = "bosonai/higgs-audio-v2", "bosonai/higgs-audio-v2-tokenizer"
    plan = {
        "entries": [
            {"repo_id": higgs, "files": ["model.safetensors"], "checkpoint": True},
            {"repo_id": codec, "files": ["model.pth"]},
        ]
    }
    audio_plan = "/api/inference/audio/download-plan"
    studio = fake_studio({("POST", audio_plan): plan, ("POST", START): STARTED})
    ref = start(studio, model = higgs, kind = "tts")
    assert bodies(studio, audio_plan) == [{"model_path": higgs}]
    assert [(b["repo_id"], b["scope_id"]) for b in bodies(studio, START)] == [
        (higgs, "audio"),
        (codec, "audio"),
    ]
    assert ref.id == f"{higgs}::@audio,{codec}::@audio"
    staged = [(higgs, "@audio"), (codec, "@audio")]
    # tts and audio load the same native repos, so either kind waits on the prefetch.
    assert model_downloads.keys_for(make_caller(), "tts", higgs, None) == staged
    assert model_downloads.keys_for(make_caller(), "audio", higgs, None) == staged
    # Without one, the repo or the Audio page's staged download of it.
    assert model_downloads.keys_for(make_caller(), "tts", "org/other-tts", None) == [
        ("org/other-tts", None),
        ("org/other-tts", "@audio"),
    ]


def test_a_failed_part_stops_the_parts_this_call_started():
    refused = JSONResponse({"detail": "scope_file_mismatch"}, status_code = 409)
    studio = fake_studio(
        {
            ("POST", "/api/inference/images/download-plan"): PLAN,
            ("POST", START): sequence({**STARTED, "generation": 3}, refused),
            ("POST", CANCEL): {"state": "cancelling"},
        }
    )
    with pytest.raises(ToolError):
        start(studio, model = "org/FLUX", kind = "image")
    assert bodies(studio, CANCEL) == [
        {"repo_id": FLUX, "gguf_variant": "@diffusion", "generation": 3}
    ]
    assert model_downloads._remembered == {}


def test_cancel_stops_every_part_before_raising_a_refusal():
    owned = JSONResponse({"detail": "A load owns this download"}, status_code = 409)
    studio = fake_studio({("POST", CANCEL): sequence(owned, {"state": "cancelling"})})
    with pytest.raises(ToolError, match = "A load owns"):
        run(studio, model_downloads.cancel_parts, FLUX_ID)
    assert len(bodies(studio, CANCEL)) == 2


def test_repo_variant_shorthand_downloads_and_waits_on_the_variant():
    studio = fake_studio({("POST", START): STARTED})
    assert start(studio, model = f"{GGUF}:Q8_0", kind = "llm").id == f"{GGUF}::Q8_0"
    assert bodies(studio, START)[0]["repo_id"] == GGUF
    caller = make_caller()
    assert model_downloads.keys_for(caller, "llm", f"{GGUF}:Q8_0", None) == [(GGUF, "Q8_0")]
    # A plain load reads the default variant, not the one download_model staged last.
    assert model_downloads.keys_for(caller, "llm", GGUF, None) == [(GGUF, None)]


def wait(studio, **kwargs):
    ctx = Ctx()
    args = {"kind": "llm", "variant": None, "budget_s": 5.0, **kwargs}
    run(studio, model_downloads.wait_for_managed, ctx, **args)
    return ctx


def test_wait_returns_once_the_download_finishes():
    states = sequence({"state": "running"}, {"state": "running"}, {"state": "complete"})
    studio = fake_studio({("GET", STATUS): states})
    ctx = wait(studio, model = "org/repo", variant = "Q4_K_M")
    assert queries(studio, STATUS) == [{"repo_id": "org/repo", "gguf_variant": "Q4_K_M"}] * 3
    follow = 'get_job(kind="download", id="org/repo::Q4_K_M")'
    assert ctx.messages == [f"Waiting for Unsloth Studio's download; follow it with {follow}"] * 2


def test_wait_with_no_budget_left_tells_the_agent_to_follow_the_download():
    studio = fake_studio(
        {("GET", STATUS): {"state": "running"}, ("GET", "/api/hub/gguf-variants"): VARIANTS}
    )
    with pytest.raises(ToolError) as raised:
        wait(studio, model = GGUF, budget_s = 0)
    assert str(raised.value) == (
        f"Unsloth Studio is still downloading {GGUF}. "
        f'Follow it with get_job(kind="download", id="{GGUF}::Q4_K_M"), then call load_model again.'
    )


@pytest.mark.parametrize("answer", [NOT_YOURS, JSONResponse({"detail": "boom"}, status_code = 500)])
def test_wait_proceeds_when_the_download_is_not_readable(answer):
    wait(fake_studio({("GET", STATUS): answer}), model = "org/repo", budget_s = 0)


def test_wait_skips_stt_and_local_paths():
    studio = fake_studio(fallback = {"state": "running"})
    wait(studio, model = WHISPER, kind = "stt", budget_s = 0)
    wait(studio, model = "/srv/outputs/qwen-lora", budget_s = 0)
    assert studio.state.calls == []


def test_wait_reads_the_parts_download_model_staged_for_this_account():
    studio = fake_studio(
        {
            ("POST", "/api/inference/images/download-plan"): PLAN,
            ("POST", START): STARTED,
            ("GET", STATUS): {"state": "complete"},
        }
    )
    start(studio, model = "org/FLUX", kind = "image")
    staged = [(FLUX, "@diffusion"), (VAE, "@diffusion")]
    keys_for = model_downloads.keys_for
    assert keys_for(make_caller(), "image", "ORG/flux", None) == staged
    assert keys_for(make_caller(account_id = "b"), "image", "org/FLUX", None) == [
        ("org/FLUX", "@diffusion")
    ]
    wait(studio, model = "org/FLUX", kind = "image", budget_s = 0)
    assert [q["repo_id"] for q in queries(studio, STATUS)] == [FLUX, VAE]


# The tools over MCP: download_model, get_job(kind="download") and cancel(kind="model_download").
TOOL_PAYLOADS = {
    ("GET", "/api/hub/gguf-variants"): VARIANTS,
    ("POST", START): STARTED,
    ("GET", STATUS): {"state": "error", "error": "401 gated repo", "generation": 1, "attempt": 1},
    ("POST", CANCEL): {"job_key": "unsloth/qwen3-0.6b-gguf::q4_k_m", "state": "cancelling"},
}


def test_download_model_returns_the_id_get_job_and_cancel_take(monkeypatch):
    result, studio = run_tool(monkeypatch, TOOL_PAYLOADS, "download_model", {"model": GGUF})
    ref = result["structuredContent"]
    assert ref["id"] == f"{GGUF}::Q4_K_M" and ref["status"] == "running"
    assert bodies(studio, START) == [
        {"repo_id": GGUF, "transport_mode": "auto", "gguf_variant": "Q4_K_M"}
    ]
    job, studio = run_tool(
        monkeypatch, TOOL_PAYLOADS, "get_job", {"kind": "download", "id": ref["id"]}
    )
    assert job["structuredContent"]["status"] == "failed"
    assert job["structuredContent"]["error"] == "401 gated repo"
    assert queries(studio, STATUS) == [{"repo_id": GGUF, "gguf_variant": "Q4_K_M"}]
    stopped, studio = run_tool(
        monkeypatch, TOOL_PAYLOADS, "cancel", {"kind": "model_download", "id": ref["id"]}
    )
    assert stopped["structuredContent"]["cancelled"] is True
    assert bodies(studio, CANCEL) == [{"repo_id": GGUF, "gguf_variant": "Q4_K_M"}]


def test_get_job_download_needs_the_id(monkeypatch):
    result, studio = run_tool(monkeypatch, TOOL_PAYLOADS, "get_job", {"kind": "download"})
    assert result["isError"] is True and "download_model" in result["content"][0]["text"]
    assert studio.state.calls == []


def test_a_second_media_prefetch_keeps_the_first_ones_parts():
    plan = "/api/inference/images/download-plan"
    second = {"entries": [{"repo_id": "org/te-int8", "files": ["te.safetensors"]}]}
    studio = fake_studio({("POST", plan): sequence(PLAN, second), ("POST", START): STARTED})
    start(studio, model = "org/FLUX", kind = "image", knobs = {"text_encoder_quant": "fp8"})
    start(studio, model = "org/FLUX", kind = "image", knobs = {"text_encoder_quant": "int8"})
    assert model_downloads.keys_for(make_caller(), "image", "org/FLUX", None) == [
        (FLUX, "@diffusion"),
        (VAE, "@diffusion"),
        ("org/te-int8", "@diffusion"),
    ]


def test_two_ref_handles_are_two_models():
    assert not loading._same_id("ref:aaa", "ref:bbb")
    assert loading._same_id("ref:aaa", "ref:aaa")


def test_a_stalled_status_read_is_no_wait(monkeypatch):
    from .mcp_harness import slow

    monkeypatch.setattr(model_downloads, "READ_TIMEOUT_S", 0.05)
    studio = fake_studio({("GET", STATUS): slow({"state": "running"}, 2.0)})
    ctx = wait(studio, model = "org/repo", variant = "Q4_K_M", budget_s = 5.0)
    assert ctx.messages == []


def test_repo_variant_shorthand_for_speech_waits_on_the_variant():
    assert model_downloads.keys_for(make_caller(), "tts", "org/speech:Q4_K_M", None) == [
        ("org/speech", "Q4_K_M")
    ]


def test_a_gguf_repo_is_known_by_its_listing_not_its_name():
    studio = fake_studio({("GET", "/api/hub/gguf-variants"): VARIANTS, ("POST", START): STARTED})
    assert start(studio, model = "org/quants", kind = "llm").id == "org/quants::Q4_K_M"
