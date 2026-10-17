# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 prompt upsampling: request building, reference preparation, family defaults, fallbacks, the gallery
record and the routes. The llama-server is mocked; no GPU and no network."""

from __future__ import annotations

import base64
import hashlib
import io
import threading

import pytest

from core.inference import flux3_prompt_upsampler as pup


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    pup.clear_cache()
    monkeypatch.delenv(pup.ENV_MODEL_OVERRIDE, raising = False)
    monkeypatch.delenv(pup.ENV_MMPROJ_OVERRIDE, raising = False)
    yield
    pup.clear_cache()


# ── prompts and sampling ────────────────────────────────────────────────────────────────────────────────────────────
# sha256 of BFL's system_prompts.json entries (black-forest-labs/flux-3-image-klein-pup @ 8b069751), equal to
# diffusers SYSTEM_PROMPTS at d42ebf9c6e.
_T2I_SHA = "8b235a421bce2213d11ada01f5a1244443dd72303ffbb4cc9e5a0d8b0de5ab6f"
_I2I_SHA = "95cec384d5b6d341caa9fb3f2c7ca45ee80c2364b3a95d4044aebf976f6dd3bf"


def test_system_prompts_are_bfls_verbatim():
    assert hashlib.sha256(pup.SYSTEM_PROMPTS["t2i"].encode()).hexdigest() == _T2I_SHA
    assert hashlib.sha256(pup.SYSTEM_PROMPTS["i2i"].encode()).hexdigest() == _I2I_SHA


@pytest.mark.parametrize(
    "variant, repeat, presence",
    [("distilled", 1.05, 1.5), ("base", 1.0, 1.5)],
)
def test_t2i_payload(variant, repeat, presence):
    body = pup.build_chat_payload("a gymnast doing a backflip", variant)
    assert body["messages"] == [
        {"role": "system", "content": pup.SYSTEM_PROMPTS["t2i"]},
        {"role": "user", "content": [{"type": "text", "text": "a gymnast doing a backflip"}]},
    ]
    assert body["top_k"] == 1 and body["temperature"] == 1.0 and body["top_p"] == 0.95
    assert body["min_p"] == 0.0 and body["seed"] == 0
    assert body["repeat_penalty"] == repeat and body["presence_penalty"] == presence
    assert body["max_tokens"] == 3584
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("variant, repeat", [("distilled", 1.05), ("base", 1.0)])
def test_edit_payload_images_then_scaffold(variant, repeat):
    refs = [b"jpeg-one", b"jpeg-two"]
    body = pup.build_chat_payload("make it red", variant, refs)
    system, user = body["messages"]
    assert system == {"role": "system", "content": pup.SYSTEM_PROMPTS["i2i"]}
    parts = user["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(
        b"jpeg-one"
    ).decode()
    assert parts[2]["text"] == (
        "Reference images follow in order (the first image, the second image, ...).\n"
        "User request: make it red\n"
        "Output the caption."
    )
    assert body["repeat_penalty"] == repeat and body["presence_penalty"] == 0.0


def test_edit_without_images_uses_the_edit_prompt_text_only():
    body = pup.build_chat_payload("make it red", "distilled", [], edit = True)
    system, user = body["messages"]
    assert system["content"] == pup.SYSTEM_PROMPTS["i2i"]
    assert [p["type"] for p in user["content"]] == ["text"]
    assert "User request: make it red" in user["content"][0]["text"]


def test_unknown_variant_is_refused():
    with pytest.raises(ValueError):
        pup.build_chat_payload("x", "turbo")


def test_clean_caption():
    assert pup.clean_caption("<think>\n\n</think>\n\n A caption. ") == "A caption."
    assert pup.clean_caption('"quoted"') == "quoted"
    assert pup.clean_caption(None) == ""


# ── reference preparation ───────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "size, count, expected",
    [
        ((1024, 1024), 1, (1024, 1024)),
        ((4000, 3000), 1, (1152, 864)),
        ((4000, 3000), 2, (832, 608)),
        ((4000, 3000), 3, (576, 416)),
        ((4000, 3000), 10, (576, 416)),
        ((100, 50), 1, (96, 64)),
        ((1000, 31), 1, (992, 32)),
    ],
)
def test_reference_size(size, count, expected):
    w, h = pup.get_reference_size(*size, count)
    assert (w, h) == expected
    assert w % 32 == 0 and h % 32 == 0
    assert w * h <= pup.MAX_PIXELS_BY_REFS[count]


def test_reference_size_refuses_extreme_aspect():
    with pytest.raises(ValueError):
        pup.get_reference_size(10000, 10, 1)


def test_prepare_reference_rgba_over_white_and_jpeg():
    from PIL import Image

    image = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    out = Image.open(io.BytesIO(pup.prepare_reference(image, 1)))
    assert out.format == "JPEG" and out.mode == "RGB"
    assert out.size == (288, 192)
    r, g, b = out.getpixel((10, 10))
    assert min(r, g, b) > 245  # transparent became white, not black


def test_prepare_references_caps_at_ten_and_decodes_data_urls():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (800, 600), (10, 20, 30)).save(buf, format = "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    refs = pup.prepare_references([url] * 12)
    assert len(refs) == 10
    assert Image.open(io.BytesIO(refs[0])).size == pup.get_reference_size(800, 600, 10)


# ── family defaults ─────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "family, flag, expected",
    [
        ("flux.3-klein", None, True),
        ("flux.3-klein-base", None, True),
        ("flux.3-klein", False, False),
        ("flux.3-klein-base", True, True),
        ("flux.2-klein", None, False),
        ("flux.2-klein", True, False),
        ("z-image", True, False),
        (None, True, False),
    ],
)
def test_resolve_enhance(family, flag, expected):
    assert pup.resolve_enhance(family, flag) is expected


def test_variant_and_capabilities():
    assert pup.variant_for_family("flux.3-klein") == "distilled"
    assert pup.variant_for_family("flux.3-klein-base") == "base"
    assert pup.family_capabilities("flux.3-klein-base") == {
        "supports_prompt_upsampler": True,
        "prompt_upsampler_default": True,
    }
    assert pup.family_capabilities("qwen-image") == {
        "supports_prompt_upsampler": False,
        "prompt_upsampler_default": False,
    }


def test_status_annotation_carries_the_capabilities(monkeypatch):
    from core.inference import diffusion_engine_router as router
    from models.inference import DiffusionStatusResponse

    monkeypatch.setattr(router, "_supported_family_capabilities", lambda: ("flux.3-klein",))
    out = router.annotate_status({"loaded": True, "repo_id": "x", "family": "flux.3-klein"})
    assert out["supports_prompt_upsampler"] is True and out["prompt_upsampler_default"] is True
    assert DiffusionStatusResponse(**out).supports_prompt_upsampler is True
    out = router.annotate_status({"loaded": False, "repo_id": None, "family": None})
    assert out["supports_prompt_upsampler"] is False


# ── memory planning ─────────────────────────────────────────────────────────────────────────────────────────────────
GIB = 1024**3


def test_plan_gpu_layers():
    model = int(5.24 * GIB)
    assert pup.plan_gpu_layers(40_000, model, 900 * 1024 * 1024) == 99
    assert 4 <= pup.plan_gpu_layers(6_000, model) < 32
    assert pup.plan_gpu_layers(2_000, model) == 0
    assert pup.plan_gpu_layers(40_000, model, training = True) == 0
    assert pup.plan_gpu_layers(None, model, device = "cpu") == 0
    assert pup.plan_gpu_layers(None, model) == 99
    assert pup.plan_gpu_layers(1_000, model, device = "mps") == 99


def test_vram_is_tight():
    assert pup.vram_is_tight(4_000, 24_000)
    assert not pup.vram_is_tight(20_000, 24_000)
    assert pup.vram_is_tight(15_000, 80_000)  # under 20% of the card
    assert not pup.vram_is_tight(None, None)


# ── paths and fallbacks ─────────────────────────────────────────────────────────────────────────────────────────────
def test_env_override_paths(monkeypatch, tmp_path):
    model = tmp_path / "pup.gguf"
    model.write_bytes(b"x" * 10)
    monkeypatch.setenv(pup.ENV_MODEL_OVERRIDE, str(model))
    monkeypatch.setenv(pup.ENV_MMPROJ_OVERRIDE, str(tmp_path / "missing.gguf"))
    monkeypatch.setattr(pup, "_cached_at_any_revision", lambda *a, **k: "/cache/should-not-be-used")
    assert pup.model_path() == str(model)
    # A set but missing override is a typo, not a reason to read the cache.
    assert pup.mmproj_path() is None


class _FakeSidecar:
    def __init__(self, reply = "A long caption.", error = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    def generate(self, payload, *, binary, model, mmproj, cancel_event = None):
        self.calls.append({"payload": payload, "model": model, "mmproj": mmproj})
        if self.error is not None:
            raise self.error
        return self.reply(payload) if callable(self.reply) else self.reply

    def release_if_tight(self):
        return False


@pytest.fixture
def runnable(monkeypatch):
    sidecar = _FakeSidecar(
        reply = lambda payload: "Caption for " + payload["messages"][1]["content"][-1]["text"]
    )
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(pup, "model_path", lambda: "/m/pup.gguf")
    monkeypatch.setattr(pup, "mmproj_path", lambda: "/m/mmproj.gguf")
    monkeypatch.setattr(pup, "get_sidecar", lambda: sidecar)
    return sidecar


def test_one_upsample_per_distinct_prompt_and_cache(runnable):
    out = pup.enhance_for_generation(
        ["a cat", "a dog", "a cat"], family_name = "flux.3-klein", enhance_prompt = None
    )
    assert out.status == "applied"
    assert len(runnable.calls) == 2
    assert out.encoded("a cat") == "Caption for a cat"
    assert out.status_for("a dog") == "applied"
    # Same prompt + variant + refs: the cache answers, so a re-run encodes the same caption.
    again = pup.enhance_for_generation(["a cat"], family_name = "flux.3-klein", enhance_prompt = True)
    assert len(runnable.calls) == 2 and again.encoded("a cat") == "Caption for a cat"
    # A different variant is a different caption request.
    pup.enhance_for_generation(["a cat"], family_name = "flux.3-klein-base", enhance_prompt = True)
    assert len(runnable.calls) == 3
    assert runnable.calls[-1]["payload"]["repeat_penalty"] == 1.0


def test_off_and_other_families_never_call(runnable):
    assert pup.enhance_for_generation(["a"], family_name = "flux.3-klein", enhance_prompt = False).status == "off"
    other = pup.enhance_for_generation(["a"], family_name = "flux.2-klein", enhance_prompt = True)
    assert other.status is None and other.encoded("a") == "a"
    assert runnable.calls == []


def test_unavailable_without_weights_or_binary(monkeypatch):
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(pup, "model_path", lambda: None)
    out = pup.enhance_for_generation(["a"], family_name = "flux.3-klein", enhance_prompt = None)
    assert out.status == "unavailable" and out.encoded("a") == "a" and out.caption_for("a") is None
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: None)
    monkeypatch.setattr(pup, "model_path", lambda: "/m/pup.gguf")
    assert (
        pup.enhance_for_generation(["a"], family_name = "flux.3-klein", enhance_prompt = None).status
        == "unavailable"
    )


def test_server_failure_falls_back_to_the_typed_prompt(runnable):
    runnable.error = RuntimeError("boom")
    out = pup.enhance_for_generation(["a"], family_name = "flux.3-klein", enhance_prompt = None)
    assert out.status == "failed" and out.encoded("a") == "a" and out.caption_for("a") is None


def test_empty_caption_falls_back(runnable):
    runnable.reply = "  "
    assert pup.upsample("a", variant = "distilled") == "a"
    out = pup.enhance_for_generation(["a"], family_name = "flux.3-klein", enhance_prompt = None)
    assert out.status == "failed" and out.encoded("a") == "a"


def test_cancel_propagates(runnable):
    event = threading.Event()
    event.set()
    with pytest.raises(pup.UpsamplerCancelledError):
        pup.enhance_for_generation(
            ["a"], family_name = "flux.3-klein", enhance_prompt = None, cancel_event = event
        )
    assert pup.progress() is None


def _png_url(size = (640, 480)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_edit_sends_references_with_mmproj(runnable):
    caption = pup.upsample("make it blue", variant = "distilled", reference_images = [_png_url()])
    call = runnable.calls[-1]
    assert call["mmproj"] == "/m/mmproj.gguf"
    parts = call["payload"]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "text"]
    assert caption.startswith("Caption for Reference images follow in order")


def test_edit_without_mmproj_runs_text_only(runnable, monkeypatch):
    monkeypatch.setattr(pup, "mmproj_path", lambda: None)
    pup.upsample("make it blue", variant = "distilled", reference_images = [_png_url()])
    call = runnable.calls[-1]
    assert call["mmproj"] is None
    system, user = call["payload"]["messages"]
    assert system["content"] == pup.SYSTEM_PROMPTS["i2i"]
    assert [p["type"] for p in user["content"]] == ["text"]


def test_references_change_the_cache_key(runnable):
    pup.upsample("edit", reference_images = [_png_url((640, 480))])
    pup.upsample("edit", reference_images = [_png_url((640, 480))])
    pup.upsample("edit", reference_images = [_png_url((480, 640))])
    assert len(runnable.calls) == 2


def test_upsample_raises_when_not_downloaded(monkeypatch):
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(pup, "model_path", lambda: None)
    with pytest.raises(pup.UpsamplerNotDownloadedError):
        pup.upsample("a")


# ── the server request (mocked HTTP) ────────────────────────────────────────────────────────────────────────────────
def test_post_chat_parses_the_reply(monkeypatch):
    import json

    sent = {}

    class _Response:
        status = 200

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "The caption."}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Connection:
        def __init__(self, host, port, timeout = None):
            sent["target"] = (host, port)
            self.sock = None

        def request(self, method, path, body = None, headers = None):
            sent.update(method = method, path = path, body = json.loads(body), headers = headers)

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    monkeypatch.setattr(pup.http.client, "HTTPConnection", _Connection)
    payload = pup.build_chat_payload("a", "distilled")
    assert pup._post_chat(4321, payload, api_key = "k") == "The caption."
    assert sent["target"] == ("127.0.0.1", 4321)
    assert sent["path"] == "/v1/chat/completions" and sent["body"] == payload
    assert sent["headers"]["Authorization"] == "Bearer k"


@pytest.mark.parametrize(
    "status, props, expected",
    [
        (200, {"default_generation_settings": {"params": {"penalize_prompt": True}}}, True),
        (200, {"default_generation_settings": {"penalize_prompt": True}}, True),
        (200, {"default_generation_settings": {"params": {"presence_penalty": 0.0}}}, False),
        (200, {}, False),
        (404, {}, False),
        (200, None, False),
    ],
)
def test_penalize_prompt_probe(monkeypatch, status, props, expected):
    import json

    class _Response:
        def __init__(self):
            self.status = status

        def read(self):
            return b"not json" if props is None else json.dumps(props).encode()

    class _Connection:
        def __init__(self, host, port, timeout = None):
            pass

        def request(self, method, path, headers = None):
            assert (method, path) == ("GET", "/props")

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    monkeypatch.setattr(pup.http.client, "HTTPConnection", _Connection)
    assert pup._server_has_penalize_prompt(4321) is expected


@pytest.mark.parametrize("supported", [True, False])
def test_generated_only_penalties_follow_the_server(monkeypatch, supported):
    sent = {}
    sidecar = pup.PromptUpsamplerSidecar()

    def _fake_launch(binary, model, mmproj):
        sidecar._process = type("P", (), {"poll": lambda self: None})()
        sidecar._port, sidecar._key = 4321, (binary, model, mmproj)
        sidecar._penalize_prompt_supported = supported

    monkeypatch.setattr(sidecar, "_launch", _fake_launch)
    monkeypatch.setattr(sidecar, "_schedule_idle_locked", lambda: None)
    monkeypatch.setattr(pup, "_post_chat", lambda port, payload, **kw: sent.update(payload) or "caption")
    payload = pup.build_chat_payload("a", "base")
    assert sidecar.generate(payload, binary = "b", model = "m", mmproj = None) == "caption"
    if supported:
        assert sent["penalize_prompt"] is False and sent["repeat_last_n"] == 4096
    else:
        assert "penalize_prompt" not in sent and "repeat_last_n" not in sent
    assert sent["presence_penalty"] == 1.5 and "penalize_prompt" not in payload


# ── the download state reuses the dictation worker ──────────────────────────────────────────────────────────────────
def test_download_state_targets_the_unsloth_repo(monkeypatch):
    state = pup._get_download_state()
    assert state._spec(pup.UPSAMPLER_ID).repo == "unsloth/flux-3-image-klein-pup-GGUF"
    assert state._spec(pup.UPSAMPLER_ID).model_file == "flux-3-image-klein-pup-Q4_K_M.gguf"
    assert state._spec(pup.UPSAMPLER_ID).mmproj_file == "mmproj-F16.gguf"
    with pytest.raises(ValueError):
        state._resolve("black-forest-labs/flux-3-image-klein-pup")
    snapshot = state.status()
    assert {"downloading", "download_id", "error", "cancelled", "bytes_total", "bytes_done"} <= set(snapshot)


def test_status_shape(monkeypatch):
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: None)
    monkeypatch.setattr(pup, "model_path", lambda: None)
    monkeypatch.setattr(pup, "mmproj_path", lambda: None)
    out = pup.status()
    assert out["supported"] is False and out["downloaded"] is False and out["loaded"] is False
    assert out["mmproj_downloaded"] is False
    assert out["repo"] == pup.UPSAMPLER_REPO and out["filename"] == pup.UPSAMPLER_MODEL_FILE
    assert out["size_bytes"] == pup.UPSAMPLER_FILE_SIZES[pup.UPSAMPLER_MODEL_FILE]
    assert isinstance(out["download"], dict)


# ── gallery record ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_gallery_record_round_trip(monkeypatch, tmp_path):
    from PIL import Image

    from core.inference import image_gallery
    from models.inference import GalleryImage

    monkeypatch.setattr(image_gallery, "gallery_dir", lambda: tmp_path)
    meta = {
        "prompt": "a gymnast doing a backflip",
        "enhanced_prompt": "Full-color indoor photograph of a gymnast ...",
        "prompt_enhancement": "applied",
        "width": 64,
        "height": 64,
        "steps": 4,
        "guidance": 1.0,
        "seed": 7,
        "created_at": 1.0,
    }
    record = image_gallery.save(Image.new("RGB", (64, 64)), meta)
    listed = [r for r in image_gallery.list_images(10, 0) if r["id"] == record["id"]][0]
    image = GalleryImage(**listed)
    assert image.prompt == "a gymnast doing a backflip"
    assert image.enhanced_prompt == meta["enhanced_prompt"]
    assert image.prompt_enhancement == "applied"
    # Records written before this read as unset.
    old = {k: v for k, v in listed.items() if k not in ("enhanced_prompt", "prompt_enhancement")}
    image = GalleryImage(**old)
    assert image.enhanced_prompt is None and image.prompt_enhancement is None


# ── routes ──────────────────────────────────────────────────────────────────────────────────────────────────────────
class _FluxBackend:
    def __init__(self, family = "flux.3-klein"):
        self.family = family
        self.calls: list[dict] = []

    def status(self):
        return {"loaded": True, "repo_id": "unsloth/flux", "family": self.family}

    def generate(self, *, seed = None, prompts = None, batch_size = 1, **kwargs):
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

    backend = _FluxBackend()
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


def test_generate_encodes_the_caption_and_records_both(routes, runnable):
    client, backend, saved = routes
    r = client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert r.status_code == 200, r.text
    assert backend.calls[-1]["prompt"] == "Caption for a cat"
    image = r.json()["images"][0]
    assert image["prompt"] == "a cat"
    assert image["enhanced_prompt"] == "Caption for a cat"
    assert image["prompt_enhancement"] == "applied"


def test_generate_restore_with_enhance_off_is_untouched(routes, runnable):
    client, backend, saved = routes
    r = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "Caption for a cat", "enhance_prompt": False, "steps": 4},
    )
    assert r.status_code == 200, r.text
    assert backend.calls[-1]["prompt"] == "Caption for a cat"
    assert runnable.calls == []
    image = r.json()["images"][0]
    assert image["enhanced_prompt"] is None and image["prompt_enhancement"] == "off"


def test_generate_prompts_list_one_upsample_each(routes, runnable):
    client, backend, saved = routes
    r = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a", "prompts": ["a", "b", "a"], "steps": 4},
    )
    assert r.status_code == 200, r.text
    assert backend.calls[-1]["prompts"] == ["Caption for a", "Caption for b", "Caption for a"]
    assert len(runnable.calls) == 2
    assert [i["enhanced_prompt"] for i in r.json()["images"]] == [
        "Caption for a",
        "Caption for b",
        "Caption for a",
    ]


def test_generate_other_family_passes_through(routes, runnable):
    client, backend, saved = routes
    backend.family = "flux.2-klein"
    r = client.post(
        "/api/inference/images/generate", json = {"prompt": "a cat", "enhance_prompt": True, "steps": 4}
    )
    assert r.status_code == 200, r.text
    assert backend.calls[-1]["prompt"] == "a cat" and runnable.calls == []
    image = r.json()["images"][0]
    assert image["enhanced_prompt"] is None and image["prompt_enhancement"] is None


def test_generate_skips_the_upsampler_for_a_workflow_the_family_refuses(routes, runnable, monkeypatch):
    client, backend, saved = routes
    monkeypatch.setattr(
        backend, "status", lambda: {**_FluxBackend.status(backend), "workflows": ["txt2img", "reference"]}
    )
    client.post(
        "/api/inference/images/generate",
        json = {"prompt": "make it night time", "workflow": "edit", "init_image": "data:,", "steps": 4},
    )
    assert runnable.calls == []
    client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert len(runnable.calls) == 1


def test_unload_sidecar_never_creates_one(monkeypatch):
    monkeypatch.setattr(pup, "_sidecar", None)
    pup.unload_sidecar()
    assert pup._sidecar is None
    released = []
    sidecar = pup.PromptUpsamplerSidecar()
    monkeypatch.setattr(sidecar, "unload", lambda wait = True: released.append(wait))
    monkeypatch.setattr(pup, "_sidecar", sidecar)
    pup.unload_sidecar()
    assert released == [False]


def test_generate_upsampler_failure_never_fails_the_image(routes, runnable):
    client, backend, saved = routes
    runnable.error = RuntimeError("server died")
    r = client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert r.status_code == 200, r.text
    assert backend.calls[-1]["prompt"] == "a cat"
    assert r.json()["images"][0]["prompt_enhancement"] == "failed"


def test_generate_edit_sends_init_and_references(routes, runnable):
    client, backend, saved = routes
    r = client.post(
        "/api/inference/images/generate",
        json = {
            "prompt": "swap the hat",
            "init_image": _png_url(),
            "reference_images": [_png_url((320, 320))],
            "workflow": "edit",
            "steps": 4,
        },
    )
    assert r.status_code == 200, r.text
    parts = runnable.calls[-1]["payload"]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]


def test_upsampler_routes(routes, runnable, monkeypatch):
    client, backend, saved = routes
    monkeypatch.setattr(pup, "status", lambda: {"supported": True, "downloaded": True})
    assert client.get("/api/inference/images/prompt-upsampler/status").json() == {
        "supported": True,
        "downloaded": True,
    }
    r = client.post(
        "/api/inference/images/prompt-upsampler/preview", json = {"prompt": "a gymnast"}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"enhanced_prompt": "Caption for a gymnast"}
    backend.family = "flux.3-klein-base"
    client.post("/api/inference/images/prompt-upsampler/preview", json = {"prompt": "b"})
    assert runnable.calls[-1]["payload"]["repeat_penalty"] == 1.0
    client.post(
        "/api/inference/images/prompt-upsampler/preview",
        json = {"prompt": "c", "variant": "distilled"},
    )
    assert runnable.calls[-1]["payload"]["repeat_penalty"] == 1.05

    started = {}
    monkeypatch.setattr(pup, "start_download", lambda token: started.setdefault("token", token) and "dl-1")
    monkeypatch.setattr(pup, "download_status", lambda: {"downloading": True})
    r = client.post("/api/inference/images/prompt-upsampler/download", json = {"hf_token": "hf_x"})
    assert r.status_code == 200 and r.json()["download_id"] == "dl-1" and started["token"] == "hf_x"
    r = client.post("/api/inference/images/prompt-upsampler/download")
    assert r.status_code == 200
    monkeypatch.setattr(pup, "cancel_download", lambda: True)
    r = client.post("/api/inference/images/prompt-upsampler/cancel")
    assert r.status_code == 200 and r.json()["cancelled"] is True


def test_preview_errors(routes, monkeypatch):
    client, backend, saved = routes
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(pup, "model_path", lambda: None)
    r = client.post("/api/inference/images/prompt-upsampler/preview", json = {"prompt": "a"})
    assert r.status_code == 409
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: None)
    r = client.post("/api/inference/images/prompt-upsampler/preview", json = {"prompt": "a"})
    assert r.status_code == 501


def test_images_status_route_carries_the_capabilities(routes, monkeypatch):
    client, backend, saved = routes
    from core.inference import diffusion_engine_router as engine_router

    monkeypatch.setattr(engine_router, "_supported_family_capabilities", lambda: ("flux.3-klein",))
    body = client.get("/api/inference/images/status").json()
    assert body["supports_prompt_upsampler"] is True and body["prompt_upsampler_default"] is True
    backend.family = "z-image"
    body = client.get("/api/inference/images/status").json()
    assert body["supports_prompt_upsampler"] is False and body["prompt_upsampler_default"] is False


def test_download_status_keys_match_the_frontend():
    assert {"downloading", "bytes_total", "bytes_done", "error", "cancelled"} <= set(
        pup.download_status()
    )


def test_progress_reports_the_enhance_phase(routes, monkeypatch):
    client, backend, saved = routes
    monkeypatch.setattr(
        pup, "progress", lambda: {"done": 0, "total": 1, "message": "Enhancing prompt..."}
    )
    body = client.get("/api/inference/images/generate-progress").json()
    assert body["active"] is True and body["phase"] == "enhance"
    assert body["message"] == "Enhancing prompt..."
    monkeypatch.setattr(pup, "cancel_active", lambda: True)
    assert client.post("/api/inference/images/generate/cancel").json() == {"cancelled": True}
