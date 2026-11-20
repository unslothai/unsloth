# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image prompt enhancer: registration, the reference templates' prompts and sampling, the JSON caption
extraction, per-mode defaults, the unwired aspect ratio, reference preparation, context sizing, fallbacks and the
routes. The llama-server is mocked; no GPU and no network."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading

import pytest

from core.inference import flux3_prompt_upsampler as pup
from core.inference import h3_image_prompt_enhancer as h3pe
from core.inference import h3_image_prompt_enhancer_prompts as prompts

H3 = "minimax-h3-image"


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    from core.inference import gpu_arbiter

    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    pup.clear_cache()
    for name in (h3pe.ENV_MODEL_OVERRIDE, h3pe.ENV_MMPROJ_OVERRIDE):
        monkeypatch.delenv(name, raising = False)
    yield
    pup.clear_cache()


# ── registration ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_registered_for_h3_image_only():
    assert pup.supports_family(H3) and pup.uses_h3_image_enhancer(H3)
    assert not pup.uses_h3_image_enhancer("flux.3-klein")
    assert not pup.supports_family("minimax-h3-video")
    assert pup.family_capabilities(H3) == {
        "supports_prompt_upsampler": True,
        "prompt_upsampler_default": True,
    }
    assert h3pe.ENHANCER_REPO == "unsloth/Qwen3.5-9B-GGUF"
    assert h3pe.ENHANCER_MODEL_FILE == "Qwen3.5-9B-UD-Q4_K_XL.gguf"
    assert h3pe.ENHANCER_MMPROJ_FILE == "mmproj-F16.gguf"
    assert h3pe.ENHANCER_MODEL_FILE_PREFERENCE[0] == h3pe.ENHANCER_MODEL_FILE


# ── prompts and sampling (the templates, verbatim) ──────────────────────────────────────────────────────────────────
# sha256 of the PrimitiveStringMultiline widgets in workflow_templates@79b02f4a79e2:
# image_minimax_h3_image_t2i.json subgraph nodes 17 (system) / 20 (task), image_minimax_h3_image_edit.json 36 / 39.
_SHA = {
    "T2I_SYSTEM_PROMPT": "93e41b4aa53a6e3dc1364b55525b27d58f7c7134bbb4968fbdc7d227e10dc525",
    "T2I_TASK_PROMPT": "1255e4a6cb02cc4e2e77665065fb8eebe191e19bba8de737c2eae118b94b5ead",
    "EDIT_SYSTEM_PROMPT": "93e41b4aa53a6e3dc1364b55525b27d58f7c7134bbb4968fbdc7d227e10dc525",
    "EDIT_TASK_PROMPT": "2338a3ae305ce9bedb92117a813e29faae6688f59679c1b7220df4793364be2b",
}


@pytest.mark.parametrize("name", sorted(_SHA))
def test_prompts_are_the_templates_verbatim(name):
    text = getattr(prompts, name)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == _SHA[name]


def test_task_prompts_take_the_input_once():
    assert prompts.T2I_TASK_PROMPT.count(prompts.INPUT_PLACEHOLDER) == 1
    assert prompts.EDIT_TASK_PROMPT.count(prompts.INPUT_PLACEHOLDER) == 1
    assert prompts.T2I_TASK_PROMPT.endswith("input_text: {input text}\noutput_text:\n")
    assert prompts.EDIT_TASK_PROMPT.endswith("User input: {input text}\noutput_text:\n")
    assert "caption-expansion and canvas-selection" in prompts.T2I_TASK_PROMPT


@pytest.mark.parametrize("edit, max_tokens", [(False, 8192), (True, 8196)])
def test_payload_matches_the_templates_text_generator(edit, max_tokens):
    refs = [b"\x89PNG one", b"\x89PNG two"] if edit else []
    body = h3pe.build_chat_payload("a red fox", refs, edit = edit)
    system, user = body["messages"]
    assert system == {
        "role": "system",
        "content": prompts.EDIT_SYSTEM_PROMPT if edit else prompts.T2I_SYSTEM_PROMPT,
    }
    task = (prompts.EDIT_TASK_PROMPT if edit else prompts.T2I_TASK_PROMPT).replace(
        "{input text}", "a red fox"
    )
    # Images first, then the task text, as the template's tokenizer places them.
    assert [part["type"] for part in user["content"]] == ["image_url"] * len(refs) + ["text"]
    assert user["content"][-1]["text"] == task
    if edit:
        assert (
            user["content"][0]["image_url"]["url"]
            == "data:image/png;base64," + base64.b64encode(refs[0]).decode()
        )
    # Values from the TextGenerateV2 widgets: temperature 1, top_k 20, top_p 0.95, min_p 0.
    assert body["max_tokens"] == max_tokens
    assert (body["temperature"], body["top_k"], body["top_p"], body["min_p"]) == (
        1.0,
        20,
        0.95,
        0.0,
    )
    assert (body["repeat_penalty"], body["presence_penalty"], body["seed"]) == (1.0, 0.0, 0)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["stream"] is False


def test_user_text_replaces_every_placeholder_like_the_string_replace_node():
    # StringReplace is str.replace: a "{input text}" the user typed is not re-expanded.
    text = h3pe.task_prompt("say {input text}", edit = False)
    assert text.count("say {input text}") == 1


# ── JSON caption extraction (JsonExtractString) ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"expanded_caption": "A fox.", "aspect_ratio": "16:9"}', "A fox."),
        ('```json\n{"expanded_caption": "A fox."}\n```', "A fox."),
        ('Sure: {"expanded_caption": "He said \\"hi\\"."} trailing', 'He said "hi".'),
        ('{"target_aspect_ratio": "3:2", "expanded_caption": "Edited."}', "Edited."),
        # The first object that decodes wins, even without the key.
        ('{"aspect_ratio": "1:1"} {"expanded_caption": "late"}', ""),
        # A broken brace is skipped and the next one tried.
        ('{oops {"expanded_caption": "B"}', "B"),
        ('{"expanded_caption": 42}', "42"),
        ('{"expanded_caption": null}', ""),
        ("no json at all", ""),
        ('{"expanded_caption": "unterminated', ""),
        ("", ""),
        (None, ""),
        # Reasoning ahead of the answer is dropped; an unclosed think block is a truncated reply.
        (
            '<think>\nplan {"expanded_caption": "draft"}\n</think>\n{"expanded_caption": "Final."}',
            "Final.",
        ),
        ('<think>\nstill thinking {"expanded_caption": "draft"}', ""),
    ],
)
def test_caption_from_reply(reply, expected):
    assert h3pe.caption_from_reply(reply) == expected


# ── per-mode defaults ───────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("edit", [False, True])
def test_default_is_on_in_both_modes(edit):
    # Edit template: PrimitiveBoolean "Enable PE?" [true]. T2I template: the subgraph instance's widgets_values
    # [prompt, true, 42, ...] set the promoted "enable_PE" (switch) input to true over the inner widget's [false].
    assert h3pe.default_enhance(edit) is True
    assert pup.resolve_enhance(H3, None) is True
    assert pup.resolve_enhance(H3, False) is False
    assert pup.resolve_enhance(H3, True) is True


# ── aspect ratio: unwired in the templates ──────────────────────────────────────────────────────────────────────────
def test_aspect_ratio_does_not_reach_the_canvas(monkeypatch):
    sidecar = _FakeSidecar('{"expanded_caption": "Wide fox.", "aspect_ratio": "21:9"}')
    _runnable(monkeypatch, sidecar)
    out = pup.enhance_for_generation(["fox"], family_name = H3, enhance_prompt = None)
    # Only the caption comes back: the size the user (or the default) picked stands.
    assert out.encoded("fox") == "Wide fox." and out.status == "applied"
    assert set(vars(out)) == {"status", "captions", "statuses", "load"}


# ── reference preparation (the edit template's two resizes) ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "size, expected",
    [
        # 4 MP is 4,194,304 px; a square lands on 2048x2048 exactly.
        ((1203, 1203), ((2048, 2048), (2048, 2048), (2048, 2048))),
        # Off the grid: covered to the floored multiple of 32, then centre-cropped.
        ((1475, 981), ((2511, 1670), (2511, 1670), (2496, 1664))),
        ((600, 400), ((2508, 1672), (2508, 1672), (2496, 1664))),
    ],
)
def test_reference_size(size, expected):
    scaled, covered, final = h3pe.reference_size(*size)
    assert (scaled, final) == (expected[0], expected[2])
    assert final[0] % 32 == 0 and final[1] % 32 == 0
    assert covered[0] >= final[0] and covered[1] >= final[1]


def test_prepare_reference_is_a_4mp_png_on_the_32_grid():
    from PIL import Image

    image = Image.new("RGBA", (1475, 981), (200, 30, 30, 128))
    out = Image.open(io.BytesIO(h3pe.prepare_reference(image)))
    assert out.format == "PNG" and out.mode == "RGB"
    assert out.size == (2496, 1664)
    # The alpha channel is dropped, not composited (the template's image loader keeps RGB).
    assert out.getpixel((10, 10)) == (200, 30, 30)


# ── context ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_context_fits_the_prompt_the_references_and_the_budget():
    assert h3pe.context_tokens_for("fox", 0) == pup._CONTEXT_TOKENS
    one = h3pe.context_tokens_for("fox", 1)
    assert one >= 7400 + 4096 + 8196 and one % 4096 == 0
    assert h3pe.context_tokens_for("fox", 3) > one


# ── running it ──────────────────────────────────────────────────────────────────────────────────────────────────────
class _FakeSidecar:
    def __init__(
        self,
        reply = '{"expanded_caption": "A long caption."}',
        error = None,
    ):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    def generate(
        self,
        payload,
        *,
        binary,
        model,
        mmproj,
        cancel_event = None,
        context_tokens = None,
    ):
        self.calls.append(
            {"payload": payload, "model": model, "mmproj": mmproj, "context": context_tokens}
        )
        if self.error is not None:
            raise self.error
        return self.reply(payload) if callable(self.reply) else self.reply

    def release_if_tight(self):
        return False


def _runnable(monkeypatch, sidecar):
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(h3pe, "model_path", lambda: "/m/qwen.gguf")
    monkeypatch.setattr(h3pe, "mmproj_path", lambda: "/m/mmproj.gguf")
    # The FLUX.3 files are never read for H3-Image.
    monkeypatch.setattr(pup, "model_path", lambda: pytest.fail("FLUX.3 model read for H3-Image"))
    monkeypatch.setattr(pup, "get_sidecar", lambda: sidecar)
    return sidecar


def _png_url(size = (64, 48)):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _echo(payload):
    text = payload["messages"][1]["content"][-1]["text"]
    typed = text.rsplit("input_text: ", 1)[-1].rsplit("User input: ", 1)[-1].split("\n", 1)[0]
    return json.dumps({"expanded_caption": "Caption for " + typed, "aspect_ratio": "1:1"})


def test_t2i_uses_qwen_on_the_shared_sidecar(monkeypatch):
    sidecar = _runnable(monkeypatch, _FakeSidecar(_echo))
    out = pup.enhance_for_generation(
        ["a cat", "a dog", "a cat"], family_name = H3, enhance_prompt = None
    )
    assert out.status == "applied" and len(sidecar.calls) == 2
    assert out.encoded("a cat") == "Caption for a cat"
    call = sidecar.calls[0]
    assert call["model"] == "/m/qwen.gguf" and call["mmproj"] is None
    assert call["context"] == pup._CONTEXT_TOKENS
    assert call["payload"]["max_tokens"] == 8192
    # Cached per prompt: a re-run asks nothing.
    pup.enhance_for_generation(["a cat"], family_name = H3, enhance_prompt = True)
    assert len(sidecar.calls) == 2


def test_edit_sends_the_references_with_the_projector(monkeypatch):
    sidecar = _runnable(monkeypatch, _FakeSidecar(_echo))
    out = pup.enhance_for_generation(
        ["swap the hat"],
        family_name = H3,
        enhance_prompt = None,
        reference_images = [_png_url(), _png_url((40, 40))],
    )
    assert out.encoded("swap the hat") == "Caption for swap the hat"
    call = sidecar.calls[-1]
    assert call["mmproj"] == "/m/mmproj.gguf"
    assert call["context"] == h3pe.context_tokens_for("swap the hat", 2)
    parts = call["payload"]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert call["payload"]["messages"][0]["content"] == prompts.EDIT_SYSTEM_PROMPT
    assert call["payload"]["max_tokens"] == 8196


def test_a_repeated_edit_skips_the_resizes_and_the_generation(monkeypatch):
    sidecar = _runnable(monkeypatch, _FakeSidecar(_echo))
    prepared = []
    real = h3pe.prepare_reference
    monkeypatch.setattr(h3pe, "prepare_reference", lambda image: prepared.append(1) or real(image))
    refs = [_png_url(), _png_url((40, 40))]
    first = h3pe.upsample("swap the hat", reference_images = refs)
    assert len(prepared) == 2 and len(sidecar.calls) == 1
    assert h3pe.upsample("swap the hat", reference_images = list(refs)) == first
    assert len(prepared) == 2 and len(sidecar.calls) == 1
    # Other pictures, or the same ones in another order, are another request.
    h3pe.upsample("swap the hat", reference_images = refs[::-1])
    assert len(sidecar.calls) == 2


def test_raw_reference_hash_is_order_and_content_sensitive():
    from PIL import Image

    a, b = _png_url(), _png_url((40, 40))
    assert h3pe.raw_references_hash([a, b]) == h3pe.raw_references_hash([a, b])
    assert h3pe.raw_references_hash([a, b]) != h3pe.raw_references_hash([b, a])
    red, blue = Image.new("RGB", (8, 8), (255, 0, 0)), Image.new("RGB", (8, 8), (0, 0, 255))
    assert h3pe.raw_references_hash([red]) != h3pe.raw_references_hash([blue])


def test_edit_without_projector_runs_text_only(monkeypatch):
    sidecar = _runnable(monkeypatch, _FakeSidecar(_echo))
    monkeypatch.setattr(h3pe, "mmproj_path", lambda: None)
    h3pe.upsample("swap the hat", reference_images = [_png_url()])
    call = sidecar.calls[-1]
    assert call["mmproj"] is None
    assert [p["type"] for p in call["payload"]["messages"][1]["content"]] == ["text"]
    assert "reference-conditioned" in call["payload"]["messages"][1]["content"][0]["text"]


# ── fallbacks ───────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        '{"aspect_ratio": "1:1"}',
        "<think>\nran out of tokens",
        "",
        '{"expanded_caption": "  "}',
    ],
)
def test_malformed_reply_runs_the_prompt_as_typed(monkeypatch, reply):
    _runnable(monkeypatch, _FakeSidecar(reply))
    out = pup.enhance_for_generation(["a cat"], family_name = H3, enhance_prompt = None)
    assert out.status == "failed" and out.encoded("a cat") == "a cat"
    assert out.caption_for("a cat") is None
    # Not cached: the next run asks again.
    assert pup._cache_get((h3pe.ENHANCER_ID, "a cat", False, "", "/m/qwen.gguf")) is None


def test_server_failure_runs_the_prompt_as_typed(monkeypatch):
    _runnable(monkeypatch, _FakeSidecar(error = RuntimeError("server died")))
    out = pup.enhance_for_generation(["a", "b"], family_name = H3, enhance_prompt = None)
    assert out.status == "failed" and out.encoded("a") == "a" and out.encoded("b") == "b"


def test_unavailable_without_weights_or_binary(monkeypatch):
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(h3pe, "model_path", lambda: None)
    out = pup.enhance_for_generation(["a"], family_name = H3, enhance_prompt = None)
    assert out.status == "unavailable" and out.encoded("a") == "a"
    with pytest.raises(pup.UpsamplerNotDownloadedError):
        h3pe.upsample("a")
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: None)
    monkeypatch.setattr(h3pe, "model_path", lambda: "/m/qwen.gguf")
    assert pup.enhance_for_generation(["a"], family_name = H3, enhance_prompt = None).status == (
        "unavailable"
    )


def test_off_never_calls(monkeypatch):
    sidecar = _runnable(monkeypatch, _FakeSidecar())
    out = pup.enhance_for_generation(["a"], family_name = H3, enhance_prompt = False)
    assert out.status == "off" and sidecar.calls == []


def test_cancel_surfaces(monkeypatch):
    event = threading.Event()

    def _reply(payload):
        event.set()
        raise pup.UpsamplerCancelledError("Cancelled.")

    _runnable(monkeypatch, _FakeSidecar(_reply))
    with pytest.raises(pup.UpsamplerCancelledError):
        pup.enhance_for_generation(["a"], family_name = H3, enhance_prompt = None, cancel_event = event)


# ── the sidecar's context ───────────────────────────────────────────────────────────────────────────────────────────
def test_sidecar_relaunches_for_a_longer_context_only(monkeypatch):
    sidecar = pup.PromptUpsamplerSidecar(keep_alive_seconds = 0)
    launched = []

    class _Proc:
        def poll(self):
            return None

    def _launch(
        binary,
        model,
        mmproj,
        context_tokens = None,
    ):
        launched.append(context_tokens)
        sidecar._process = _Proc()
        sidecar._key = (binary, model, mmproj)
        sidecar._context_tokens = context_tokens

    monkeypatch.setattr(sidecar, "_launch", _launch)
    monkeypatch.setattr(sidecar, "_release_locked", lambda: setattr(sidecar, "_process", None))
    monkeypatch.setattr(pup, "_post_chat", lambda *a, **k: "ok")
    run = lambda ctx: sidecar.generate(  # noqa: E731
        {}, binary = "b", model = "m", mmproj = "p", context_tokens = ctx
    )
    run(None)
    run(None)
    run(24576)
    run(None)
    run(20480)
    assert launched == [None, 24576]
    assert pup._overhead_mib(None) == pup._GPU_OVERHEAD_MIB
    assert pup._overhead_mib(24576) == pup._GPU_OVERHEAD_MIB + 8192 * 32 * 1024 / (1024 * 1024)


# ── routes ──────────────────────────────────────────────────────────────────────────────────────────────────────────
class _H3Backend:
    def __init__(self, family = H3):
        self.family = family
        self.calls: list[dict] = []

    def status(self):
        return {"loaded": True, "repo_id": "unsloth/h3", "family": self.family}

    def generate(
        self,
        *,
        seed = None,
        prompts = None,
        batch_size = 1,
        **kwargs,
    ):
        self.calls.append({"prompts": prompts, "seed": seed, **kwargs})
        count = len(prompts) if prompts else batch_size
        return {"images": [object()] * count, "seed": 5, "seeds": [5 + i for i in range(count)]}

    def generate_progress(self):
        return {"active": False, "step": 0, "total_steps": 0, "fraction": 0.0, "eta_seconds": None}

    def cancel_generate(self, expected_account = None):
        return False


@pytest.fixture
def routes(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import core.inference.diffusion_engine_router as engine_router
    import core.inference.image_gallery as gallery_module
    from auth.authentication import authenticated_via_api_key, get_current_subject
    from routes.inference import studio_router

    backend = _H3Backend()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: backend)
    saved: list[dict] = []

    def _save(image, meta):
        record = {**meta, "id": f"img{len(saved)}", "url": "/x"}
        saved.append(record)
        return record

    monkeypatch.setattr(gallery_module, "save", _save)
    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app), backend, saved


def test_generate_encodes_the_expanded_caption(routes, monkeypatch):
    client, backend, saved = routes
    _runnable(monkeypatch, _FakeSidecar(_echo))
    r = client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert r.status_code == 200, r.text
    assert backend.calls[-1]["prompt"] == "Caption for a cat"
    image = r.json()["images"][0]
    assert image["prompt"] == "a cat"
    assert image["enhanced_prompt"] == "Caption for a cat"
    assert image["prompt_enhancement"] == "applied"


def test_generate_falls_back_on_a_malformed_reply(routes, monkeypatch):
    client, backend, saved = routes
    _runnable(monkeypatch, _FakeSidecar("I cannot help with JSON."))
    r = client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert r.status_code == 200, r.text
    assert backend.calls[-1]["prompt"] == "a cat"
    assert r.json()["images"][0]["prompt_enhancement"] == "failed"


def test_status_download_and_preview_follow_the_family(routes, monkeypatch):
    client, backend, saved = routes
    sidecar = _runnable(monkeypatch, _FakeSidecar(_echo))
    monkeypatch.setattr(h3pe, "status", lambda: {"repo": h3pe.ENHANCER_REPO})
    monkeypatch.setattr(pup, "status", lambda: {"repo": pup.UPSAMPLER_REPO})
    # The loaded family decides; an explicit family wins.
    assert client.get("/api/inference/images/prompt-upsampler/status").json() == {
        "repo": "unsloth/Qwen3.5-9B-GGUF"
    }
    r = client.get(
        "/api/inference/images/prompt-upsampler/status", params = {"family": "flux.3-klein"}
    )
    assert r.json() == {"repo": pup.UPSAMPLER_REPO}
    backend.family = "flux.3-klein"
    r = client.get("/api/inference/images/prompt-upsampler/status", params = {"family": H3})
    assert r.json() == {"repo": "unsloth/Qwen3.5-9B-GGUF"}

    started = []
    monkeypatch.setattr(h3pe, "start_download", lambda token: started.append(token) or "dl-h3")
    monkeypatch.setattr(h3pe, "download_status", lambda: {"downloading": True})
    monkeypatch.setattr(pup, "start_download", lambda token: pytest.fail("FLUX.3 download started"))
    r = client.post(
        "/api/inference/images/prompt-upsampler/download",
        params = {"family": H3},
        json = {"hf_token": "hf_x"},
    )
    assert r.status_code == 200 and r.json()["download_id"] == "dl-h3" and started == ["hf_x"]
    monkeypatch.setattr(h3pe, "cancel_download", lambda: True)
    r = client.post("/api/inference/images/prompt-upsampler/cancel", params = {"family": H3})
    assert r.status_code == 200 and r.json()["cancelled"] is True

    r = client.post(
        "/api/inference/images/prompt-upsampler/preview", json = {"prompt": "a gymnast", "family": H3}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"enhanced_prompt": "Caption for a gymnast"}
    assert sidecar.calls[-1]["model"] == "/m/qwen.gguf"


def test_status_body_shape(monkeypatch):
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(h3pe, "model_path", lambda: None)
    monkeypatch.setattr(h3pe, "mmproj_path", lambda: None)
    body = h3pe.status()
    assert body["repo"] == "unsloth/Qwen3.5-9B-GGUF" and body["downloaded"] is False
    assert body["filename"] == "Qwen3.5-9B-UD-Q4_K_XL.gguf"
    assert body["current_missing_bytes"] == 5_966_095_584 + 918_166_080
    assert body["size_bytes"] == 5_966_095_584 and body["mmproj_size_bytes"] == 918_166_080


def test_another_render_seed_reuses_the_caption(routes, monkeypatch):
    client, backend, saved = routes
    sidecar = _runnable(monkeypatch, _FakeSidecar(_echo))
    for seed in (1, 2):
        r = client.post(
            "/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4, "seed": seed}
        )
        assert r.status_code == 200, r.text
        assert backend.calls[-1]["prompt"] == "Caption for a cat"
    assert len(sidecar.calls) == 1


def test_off_costs_nothing(routes, monkeypatch):
    client, backend, saved = routes
    sidecar = _runnable(monkeypatch, _FakeSidecar())
    # Enhance off probes no file, no binary and no server.
    monkeypatch.setattr(
        h3pe, "model_path", lambda: pytest.fail("enhancer files probed with Enhance off")
    )
    monkeypatch.setattr(
        pup, "find_llama_server_binary", lambda: pytest.fail("llama-server probed with Enhance off")
    )
    r = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a cat", "steps": 4, "enhance_prompt": False},
    )
    assert r.status_code == 200, r.text
    assert backend.calls[-1]["prompt"] == "a cat" and sidecar.calls == []
    assert r.json()["images"][0]["prompt_enhancement"] == "off"
