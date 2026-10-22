# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 prompt upsampling: request building, reference preparation, family defaults, fallbacks, the gallery
record and the routes. The llama-server is mocked; no GPU and no network."""

from __future__ import annotations

import base64
import hashlib
import io
import threading
import time

import pytest

from core.inference import flux3_prompt_upsampler as pup


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    # The sidecar only starts while the image model holds the GPU, or nobody does yet. The arbiter is
    # process-wide, so pin that fresh-start owner here rather than inherit one an earlier test left behind.
    # Tests about a different owner set it themselves.
    from core.inference import gpu_arbiter

    monkeypatch.setattr(gpu_arbiter, "_owner", None)
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
_T2I_BBOX_SHA = "5f513d8ce5838732919f9e645507aa20ff373e88bf93132c768e063c9db3151f"
_I2I_BBOX_SHA = "9ad10f947e401337cf155cac0518cc428159ef16f2f029f48aae2aaab8380a5e"


def test_system_prompts_are_bfls_verbatim():
    assert hashlib.sha256(pup.SYSTEM_PROMPTS["t2i"].encode()).hexdigest() == _T2I_SHA
    assert hashlib.sha256(pup.SYSTEM_PROMPTS["i2i"].encode()).hexdigest() == _I2I_SHA
    assert hashlib.sha256(pup.SYSTEM_PROMPTS["t2i_bbox"].encode()).hexdigest() == _T2I_BBOX_SHA
    assert hashlib.sha256(pup.SYSTEM_PROMPTS["i2i_bbox"].encode()).hexdigest() == _I2I_BBOX_SHA


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
    assert (
        parts[0]["image_url"]["url"]
        == "data:image/jpeg;base64," + base64.b64encode(b"jpeg-one").decode()
    )
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
    def __init__(
        self,
        reply = "A long caption.",
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
    ):
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
    assert (
        pup.enhance_for_generation(["a"], family_name = "flux.3-klein", enhance_prompt = False).status
        == "off"
    )
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


def test_a_stopped_enhancement_still_releases_a_tight_server(runnable, monkeypatch):
    released = []
    monkeypatch.setattr(runnable, "release_if_tight", lambda: released.append(True) or True)
    runnable.error = pup.UpsamplerCancelledError("Cancelled.")
    with pytest.raises(pup.UpsamplerCancelledError):
        pup.enhance_for_generation(["a"], family_name = "flux.3-klein", enhance_prompt = None)
    assert released == [True]


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
        def __init__(
            self,
            host,
            port,
            timeout = None,
        ):
            sent["target"] = (host, port)
            self.sock = None

        def request(
            self,
            method,
            path,
            body = None,
            headers = None,
        ):
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
        def __init__(
            self,
            host,
            port,
            timeout = None,
        ):
            pass

        def request(
            self,
            method,
            path,
            headers = None,
        ):
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
    monkeypatch.setattr(
        pup, "_post_chat", lambda port, payload, **kw: sent.update(payload) or "caption"
    )
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
    assert state._spec(pup.UPSAMPLER_ID).model_file == "flux-3-image-klein-pup-UD-Q4_K_XL.gguf"
    assert state._spec(pup.UPSAMPLER_ID).mmproj_file == "mmproj-F16.gguf"
    with pytest.raises(ValueError):
        state._resolve("black-forest-labs/flux-3-image-klein-pup")
    snapshot = state.status()
    assert {"downloading", "download_id", "error", "cancelled", "bytes_total", "bytes_done"} <= set(
        snapshot
    )


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
        "/api/inference/images/generate",
        json = {"prompt": "a cat", "enhance_prompt": True, "steps": 4},
    )
    assert r.status_code == 200, r.text
    assert backend.calls[-1]["prompt"] == "a cat" and runnable.calls == []
    image = r.json()["images"][0]
    assert image["enhanced_prompt"] is None and image["prompt_enhancement"] is None


def test_generate_skips_the_upsampler_for_a_workflow_the_family_refuses(
    routes, runnable, monkeypatch
):
    client, backend, saved = routes
    monkeypatch.setattr(
        backend,
        "status",
        lambda: {**_FluxBackend.status(backend), "workflows": ["txt2img", "reference"]},
    )
    client.post(
        "/api/inference/images/generate",
        json = {
            "prompt": "make it night time",
            "workflow": "edit",
            "init_image": "data:,",
            "steps": 4,
        },
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
    r = client.post("/api/inference/images/prompt-upsampler/preview", json = {"prompt": "a gymnast"})
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
    monkeypatch.setattr(
        pup, "start_download", lambda token: started.setdefault("token", token) and "dl-1"
    )
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
    stopped = []
    monkeypatch.setattr(
        backend, "cancel_generate", lambda **_: stopped.append(True) or False, raising = False
    )
    assert client.post("/api/inference/images/generate/cancel").json() == {"cancelled": True}
    # Another tab's denoise is stopped too, not only the enhancement.
    assert stopped == [True]


def test_generate_retry_recomputes_the_caption_for_the_new_model(routes, runnable, monkeypatch):
    from core.inference.diffusion_families import DiffusionModelReplacedError, load_identity

    client, backend, saved = routes
    original = backend.generate

    def replaced_once(**kwargs):
        if not backend.calls:
            backend.calls.append({"replaced": True})
            backend.family = "flux.2-klein"
            raise DiffusionModelReplacedError(
                load_identity("unsloth/flux", None, "flux.3-klein"),
                load_identity("unsloth/flux", None, "flux.2-klein"),
            )
        return original(**kwargs)

    monkeypatch.setattr(backend, "generate", replaced_once)
    r = client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert r.status_code == 200, r.text
    # FLUX.2 has no upsampler: the retry encodes the typed prompt, not the FLUX.3 caption.
    assert backend.calls[-1]["prompt"] == "a cat"
    assert r.json()["images"][0]["prompt_enhancement"] is None


def test_single_user_generate_binds_the_caption_to_its_model(routes, runnable):
    from core.inference.diffusion_families import load_identity

    client, backend, saved = routes
    status = backend.status()
    r = client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert r.status_code == 200, r.text
    # Bound on single-user installs too: a model swapped in after the caption raises and retries.
    assert backend.calls[-1]["expected_load"] == load_identity(
        status.get("repo_id"), status.get("base_repo"), "flux.3-klein"
    )


def test_a_cached_caption_still_honors_a_stop(runnable):
    assert pup.upsample("a cat", variant = "distilled") == "Caption for a cat"
    event = threading.Event()
    event.set()
    with pytest.raises(pup.UpsamplerCancelledError):
        pup.upsample("a cat", variant = "distilled", cancel_event = event)


def test_launch_publishes_loading_before_the_training_probe(monkeypatch):
    sidecar = pup.PromptUpsamplerSidecar()
    seen = []
    monkeypatch.setattr(sidecar, "_spawn_server", lambda *a: seen.append(sidecar.is_loading()))
    sidecar._launch("b", "m", None)
    assert seen == [True] and not sidecar.is_loading()


def test_an_idle_callback_waiting_on_the_lock_sees_the_request(monkeypatch):
    sidecar = pup.PromptUpsamplerSidecar()
    reaped = []
    monkeypatch.setattr(pup, "_image_model_owns_gpu", lambda: True)
    monkeypatch.setattr(sidecar, "_release_locked", lambda: reaped.append(True))

    def launch(binary, model, mmproj):
        # The keep-alive timer fired just before this request: its callback runs while the server starts.
        sidecar._idle_unload(sidecar._generation)

    monkeypatch.setattr(sidecar, "_launch", launch)
    monkeypatch.setattr(pup, "_post_chat", lambda *a, **k: "caption")
    assert sidecar.generate({}, binary = "b", model = "m", mmproj = None) == "caption"
    assert reaped == [] and sidecar._active_requests == 0


def test_unload_route_stops_the_upsampler_blocking(routes, monkeypatch):
    client, backend, saved = routes
    waits, cancels = [], []
    monkeypatch.setattr(pup, "unload_sidecar", lambda wait = False: waits.append(wait))
    monkeypatch.setattr(pup, "cancel_active", lambda: cancels.append(True))
    monkeypatch.setattr(backend, "unload", lambda **_: {"loaded": False}, raising = False)
    r = client.post("/api/inference/images/unload", json = {})
    assert r.status_code == 200, r.text
    assert waits == [True] and cancels == [True]


def test_a_request_queued_before_a_blocking_unload_never_starts(monkeypatch):
    sidecar = pup.PromptUpsamplerSidecar()
    launched = []
    monkeypatch.setattr(pup, "_image_model_owns_gpu", lambda: True)
    monkeypatch.setattr(sidecar, "_launch", lambda *a: launched.append(a))
    monkeypatch.setattr(pup, "_post_chat", lambda *a, **k: "caption")
    sidecar._request_lock.acquire()
    errors = []

    def queued():
        try:
            sidecar.generate({}, binary = "b", model = "m", mmproj = None)
        except pup.UpsamplerUnavailableError as exc:
            errors.append(exc)

    worker = threading.Thread(target = queued)
    worker.start()
    time.sleep(0.2)
    # Training starts while that request waits on the lock: the drain finds nothing active and returns.
    sidecar.unload(wait = True)
    sidecar._request_lock.release()
    worker.join(5)
    assert errors and launched == []
    # A request after the handover is a new one.
    assert sidecar.generate({}, binary = "b", model = "m", mmproj = None) == "caption"


def test_an_unowned_gpu_starts_the_upsampler_on_the_cpu(monkeypatch):
    from core.inference import gpu_arbiter

    monkeypatch.setattr(gpu_arbiter, "current_owner", lambda: None)
    assert pup._gpu_unowned()
    monkeypatch.setattr(gpu_arbiter, "current_owner", lambda: gpu_arbiter.DIFFUSION)
    assert not pup._gpu_unowned()


def test_the_child_is_pinned_to_the_image_models_card(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,GPU-abc")
    assert pup._child_device_mask(2) == {"CUDA_VISIBLE_DEVICES": "GPU-abc"}
    assert pup._child_device_mask(1) == {"CUDA_VISIBLE_DEVICES": "5"}
    assert pup._child_device_mask(None) == {}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert pup._child_device_mask(1) == {"CUDA_VISIBLE_DEVICES": "1"}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert pup._child_device_mask(0) == {}


def test_rocm_pins_inside_hip_mask_when_cuda_mask_is_also_set(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.version, "hip", "6.2", raising = False)
    # HIP wins on ROCm: torch's cuda:1 is physical 3, which CUDA_VISIBLE_DEVICES=1 would not name.
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert pup._child_device_mask(1) == {"HIP_VISIBLE_DEVICES": "3"}
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    assert pup._child_device_mask(1) == {"CUDA_VISIBLE_DEVICES": "1"}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert pup._child_device_mask(1) == {"HIP_VISIBLE_DEVICES": "1"}


_VULKAN_ROWS = [
    {"index": 0, "name": "AMD Radeon Graphics (RADV RAPHAEL)"},
    {"index": 1, "name": "AMD Radeon RX 7600 (RADV NAVI33)"},
    {"index": 2, "name": "AMD Radeon RX 7900 XTX (RADV NAVI31)"},
]


@pytest.fixture
def spawn_argv(monkeypatch):
    """Run _spawn_server up to Popen on a ROCm torch with two cards; returns (argv, env) per call."""
    import torch

    from core.inference import llama_cpp, sd_cpp_backend, stt_mtmd_sidecar, stt_sidecar
    from utils import process_lifetime

    names = ["AMD Radeon RX 7900 XTX", "AMD Radeon RX 7600"]
    state = {"vulkan": True, "rows": _VULKAN_ROWS, "names": names, "ordinal": 1}
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "LLAMA_ARG_DEVICE",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i = 0: state["names"][i])
    monkeypatch.setattr(torch.version, "hip", "6.2", raising = False)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_is_vulkan_backend",
        staticmethod(lambda b = None: state["vulkan"]),
    )
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_run_vulkan_probe", staticmethod(lambda b = None: state["rows"])
    )
    monkeypatch.setattr(
        pup, "_VULKAN_ROWS_TTL_S", 0.0
    )  # rows change mid-test; the memo has its own test
    monkeypatch.setattr(
        sd_cpp_backend,
        "_physical_position_of",
        lambda hip: (state["names"][hip], state["names"][:hip].count(state["names"][hip])),
    )
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: state["ordinal"])
    monkeypatch.setattr(
        pup, "_accelerator_memory_mib", lambda o = None: (160 * 1024, 180 * 1024, "cuda")
    )
    monkeypatch.setattr(pup, "_gpu_unowned", lambda: False)
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)
    monkeypatch.setattr(stt_mtmd_sidecar, "_llama_server_child_env", lambda b: {})
    monkeypatch.setattr(llama_cpp, "_llama_server_api_key_enabled", lambda: False)
    monkeypatch.setattr(process_lifetime, "is_process_shutting_down", lambda: False)
    monkeypatch.setattr(process_lifetime, "spawn_on_lifetime_thread", lambda fn: fn())

    class _Stop(Exception):
        pass

    def _popen(
        cmd,
        env = None,
        **_kwargs,
    ):
        seen.append((cmd, env))
        raise _Stop

    monkeypatch.setattr(pup.subprocess, "Popen", _popen)
    seen = []

    def run():
        with pytest.raises(_Stop):
            pup.PromptUpsamplerSidecar()._spawn_server("/x/llama-server", __file__, None)
        return seen[-1]

    return state, run


def test_vulkan_build_pins_the_image_models_card_by_name(spawn_argv, warmable):
    state, run = spawn_argv
    sidecar, launched, _memory, model, _mmproj = warmable
    # The iGPU enumerates first on Vulkan, so torch's cuda:1 (the RX 7600) is Vulkan1, cuda:0 is Vulkan2.
    argv, env = run()
    assert argv[-2:] == ["--device", "Vulkan1"] and env == {"HIP_VISIBLE_DEVICES": "1"}
    state["ordinal"] = 0
    assert run()[0][-2:] == ["--device", "Vulkan2"]
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "started"
    sidecar._cancel_idle_locked()


def test_vulkan_build_leaves_an_unmatched_card_unpinned(spawn_argv, warmable, monkeypatch):
    state, run = spawn_argv
    sidecar, launched, _memory, model, _mmproj = warmable
    # Two cards of one model and no way to place the selection among them: no guess.
    state["names"] = ["AMD Radeon RX 7600", "AMD Radeon RX 7600"]
    state["rows"] = [_VULKAN_ROWS[1], {"index": 2, "name": "AMD Radeon RX 7600 (RADV NAVI33)"}]
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(sd_cpp_backend, "_physical_position_of", lambda hip: (None, None))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    assert "--device" not in run()[0]
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "placement"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    state["rows"] = []
    assert "--device" not in run()[0]
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "placement"
    assert launched == []


def test_non_vulkan_and_single_card_launches_keep_their_argv(spawn_argv, monkeypatch):
    import torch

    state, run = spawn_argv
    state["vulkan"] = False
    assert "--device" not in run()[0]
    state["vulkan"] = True
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "Vulkan0")  # the user's own pin wins
    assert "--device" not in run()[0]
    monkeypatch.delenv("LLAMA_ARG_DEVICE")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    state["rows"] = [
        _VULKAN_ROWS[2]
    ]  # one Vulkan device, the image model's card: nothing to choose
    argv, env = run()
    assert "--device" not in argv and env == {}


def test_vulkan_sole_device_must_be_the_image_models_card(spawn_argv, warmable, monkeypatch):
    import torch

    state, run = spawn_argv
    sidecar, launched, _memory, model, _mmproj = warmable
    # torch holds the image model on the 7900 XTX, but Vulkan lists only the iGPU and an RX 7600.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    state["names"] = ["AMD Radeon RX 7900 XTX"]
    state["ordinal"] = 0
    state["rows"] = _VULKAN_ROWS[:2]
    assert "--device" not in run()[0]
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "placement"


@pytest.mark.parametrize(
    "count, row, pinned",
    [
        (2, "AMD Radeon RX 7600 XT (RADV NAVI33)", False),  # containment only: another model
        (2, "AMD Radeon RX 7600 (RADV NAVI33)", True),
        (1, "AMD Radeon RX 7600 XT (RADV NAVI33)", True),  # one torch card: as before
    ],
)
def test_sole_vulkan_device_needs_the_exact_model_beside_other_cards(
    spawn_argv, monkeypatch, count, row, pinned
):
    import torch

    state, _run = spawn_argv
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    state["names"] = ["AMD Radeon RX 7600", "AMD Radeon RX 7900 XTX"][:count]
    state["ordinal"] = 0
    state["rows"] = [{"index": 0, "name": row}]
    assert pup._vulkan_device_args("b", 0) == ([] if pinned else None)


def test_users_vulkan_mask_wins(spawn_argv, monkeypatch):
    state, run = spawn_argv
    # GGML_VK_VISIBLE_DEVICES renumbers the probe's devices, so a position-based pick could name another card.
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1,2")
    assert "--device" not in run()[0]


def test_vulkan_pins_one_torch_card_beside_an_igpu(spawn_argv, monkeypatch):
    import torch

    state, run = spawn_argv
    # torch sees only the 7900 XTX, but Vulkan also lists the iGPU and a masked-off RX 7600.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    state["names"] = ["AMD Radeon RX 7900 XTX"]
    state["ordinal"] = 0
    argv, env = run()
    assert argv[-2:] == ["--device", "Vulkan2"] and env == {}


def test_empty_hip_mask_falls_back_to_the_cuda_mask(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.version, "hip", "6.2", raising = False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert pup._child_device_mask(1) == {"CUDA_VISIBLE_DEVICES": "5"}


@pytest.mark.parametrize("hip", [None, "6.2"])
def test_users_llama_arg_device_keeps_every_card_visible(monkeypatch, hip):
    # ROCm1 / CUDA1 would name a device a one-card mask no longer has.
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    # ROCm is also read from a "+rocm" __version__: pin it to the build this case names.
    monkeypatch.setattr(torch, "__version__", "2.11.0+rocm6.2" if hip else "2.11.0+cu130")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "ROCm1" if hip else "CUDA1")
    assert pup._child_device_mask(1) == {}
    monkeypatch.setenv("LLAMA_ARG_DEVICE", " ")
    assert pup._child_device_mask(1) == {("HIP" if hip else "CUDA") + "_VISIBLE_DEVICES": "3"}


def test_negative_ordinal_never_pins_the_last_mask_entry(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert pup._child_device_mask(-1) == {}


def test_rocm_wheel_without_version_hip_still_reads_the_hip_mask(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.9.0+rocm7.1.1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert pup._child_device_mask(1) == {"HIP_VISIBLE_DEVICES": "3"}


def test_warm_and_spawn_share_one_vulkan_probe(monkeypatch):
    from core.inference import llama_cpp

    calls = []
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_run_vulkan_probe",
        staticmethod(lambda b = None: calls.append(b) or list(_VULKAN_ROWS)),
    )
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    assert (
        pup._vulkan_rows("/x/llama-server") == pup._vulkan_rows("/x/llama-server") == _VULKAN_ROWS
    )
    assert calls == ["/x/llama-server"]


def test_references_follow_the_generate_decoders_side_limit():
    with pytest.raises(ValueError):
        pup.prepare_references([_png_url((4100, 64))])


def test_generate_skips_the_upsampler_while_another_account_runs(routes, runnable, monkeypatch):
    from hub.services.models import account_access

    client, backend, saved = routes

    def busy(path = None):
        # The combined guard: another account's image or video job, not only chat.
        raise account_access.GpuBusyForAnotherAccountError("other", 1)

    monkeypatch.setattr(account_access, "require_idle_other_accounts", busy)
    r = client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert r.status_code == 200, r.text
    assert runnable.calls == [] and backend.calls[-1]["prompt"] == "a cat"
    assert r.json()["images"][0]["prompt_enhancement"] == "unavailable"


def test_preview_is_refused_while_another_account_runs(routes, runnable, monkeypatch):
    from hub.services.models import account_access

    client, backend, saved = routes

    def busy(path = None):
        raise account_access.GpuBusyForAnotherAccountError("other", 1)

    monkeypatch.setattr(account_access, "require_idle_other_accounts", busy)
    r = client.post("/api/inference/images/prompt-upsampler/preview", json = {"prompt": "a cat"})
    assert r.status_code == 409
    assert runnable.calls == []


def test_references_are_decoded_one_at_a_time(monkeypatch):
    live = []
    peak = []
    real_decode, real_prepare = pup.decode_reference, pup.prepare_reference

    def decode(data):
        live.append(1)
        peak.append(len(live))
        return real_decode(data)

    def prepare(image, count):
        out = real_prepare(image, count)
        live.pop()
        return out

    monkeypatch.setattr(pup, "decode_reference", decode)
    monkeypatch.setattr(pup, "prepare_reference", prepare)
    refs = pup.prepare_references([_png_url((256, 256))] * 4)
    assert len(refs) == 4 and max(peak) == 1


def test_stop_during_server_start_cancels_the_start(monkeypatch):
    sidecar = pup.PromptUpsamplerSidecar()
    started, cancelled = threading.Event(), threading.Event()

    def launch(binary, model, mmproj):
        sidecar._loading = True
        started.set()
        try:
            assert sidecar._load_cancel.wait(5), "Stop never reached the starting server"
            cancelled.set()
            raise pup.UpsamplerCancelledError("The prompt upsampler load was cancelled.")
        finally:
            sidecar._loading = False

    monkeypatch.setattr(sidecar, "_launch", launch)
    event = threading.Event()
    threading.Thread(target = lambda: (started.wait(5), event.set()), daemon = True).start()
    with pytest.raises(pup.UpsamplerCancelledError):
        sidecar.generate({}, binary = "b", model = "m", mmproj = None, cancel_event = event)
    assert cancelled.is_set()


def test_training_stops_an_enhancement_that_has_not_reached_the_sidecar(monkeypatch):
    import sys

    from routes import training_vram

    stopped = []
    monkeypatch.setitem(sys.modules, "core.inference.flux3_prompt_upsampler", pup)
    monkeypatch.setattr(pup, "cancel_active", lambda: stopped.append(True) or True)
    monkeypatch.setattr(pup, "get_sidecar", lambda: pup.PromptUpsamplerSidecar())
    assert training_vram.free_prompt_upsampler_for_training("test") == []
    assert stopped == [True]


class _ResidentUpsampler:
    loaded = True

    def __init__(self):
        self.unloaded = []

    def holds_vram(self):
        return True

    def is_loading(self):
        return False

    def unload(self, wait = True):
        self.unloaded.append(wait)


def test_training_stops_the_upsampler_even_when_it_is_the_only_resident(monkeypatch):
    import sys

    from routes import training_vram

    resident = _ResidentUpsampler()
    monkeypatch.setattr(pup, "get_sidecar", lambda: resident)
    monkeypatch.setitem(sys.modules, "core.inference.flux3_prompt_upsampler", pup)
    idle = {"model": None, "device": None, "loading": False, "any": False}
    monkeypatch.setattr(training_vram, "summarize_resident_chat", lambda: dict(idle))
    monkeypatch.setattr(training_vram, "summarize_resident_stt", lambda: dict(idle))
    freed = training_vram.coordinate_models_for_training(lambda: (True, {}))
    assert freed == ["image:prompt-upsampler"] and resident.unloaded


def test_arbiter_diffusion_eviction_stops_the_upsampler(monkeypatch):
    import core.inference.diffusion_engine_router as engine_router
    from core.inference import gpu_arbiter

    class _Engine:
        def unload(self):
            pass

    sidecar = _ResidentUpsampler()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: _Engine())
    monkeypatch.setattr(pup, "_sidecar", sidecar)
    import sys

    monkeypatch.setitem(sys.modules, "core.inference.flux3_prompt_upsampler", pup)
    event = threading.Event()
    with pup._tracking(1, event):
        gpu_arbiter._evict_diffusion()
        # The run answering is stopped first, so the handover does not wait out its caption.
        assert event.is_set()
    assert sidecar.unloaded == [True]


def test_every_drop_of_the_diffusion_claim_stops_the_upsampler(monkeypatch):
    import sys

    from core.inference import gpu_arbiter

    stops = []
    monkeypatch.setitem(sys.modules, "core.inference.flux3_prompt_upsampler", pup)
    monkeypatch.setattr(pup, "stop_for_handover", lambda: stops.append(gpu_arbiter.current_owner()))
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    # A switch to a CPU-only image model releases the claim without an eviction.
    gpu_arbiter.release(gpu_arbiter.DIFFUSION)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    assert gpu_arbiter.release_if(gpu_arbiter.DIFFUSION, lambda: True)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    gpu_arbiter.release(gpu_arbiter.CHAT)
    # Stopped while DIFFUSION still held the claim, so no other owner can start beside it.
    assert stops == [gpu_arbiter.DIFFUSION, gpu_arbiter.DIFFUSION]


def test_a_failed_enhancement_still_binds_the_generation_to_its_model(routes, runnable):
    from core.inference.diffusion_families import load_identity

    client, backend, saved = routes
    runnable.error = RuntimeError("boom")
    status = backend.status()
    r = client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert r.status_code == 200, r.text
    assert r.json()["images"][0]["prompt_enhancement"] == "failed"
    assert backend.calls[-1]["expected_load"] == load_identity(
        status.get("repo_id"), status.get("base_repo"), "flux.3-klein"
    )


def test_a_queued_enhancement_keeps_the_active_one_tracked_and_stop_reaches_both():
    first, second = threading.Event(), threading.Event()
    with pup._tracking(1, first) as active:
        with pup._tracking(2, second):
            # The run holding llama-server is reported, not the one queued behind it.
            assert pup.progress()["total"] == 1
            pup._advance(active)
            assert pup.progress()["done"] == 1
        # The queued run leaving does not clear the active one.
        assert pup.progress()["total"] == 1
        with pup._tracking(2, second):
            assert pup.cancel_active() is True
            assert first.is_set() and second.is_set()
    assert pup.progress() is None and pup.cancel_active() is False


def test_model_and_projector_come_from_one_revision(monkeypatch):
    files = {
        ("rev_new", pup.UPSAMPLER_MODEL_FILE): "/c/new/model",
        ("rev_old", pup.UPSAMPLER_MODEL_FILE): "/c/old/model",
        ("rev_old", pup.UPSAMPLER_MMPROJ_FILE): "/c/old/mmproj",
    }
    monkeypatch.delenv(pup.ENV_MODEL_OVERRIDE, raising = False)
    monkeypatch.delenv(pup.ENV_MMPROJ_OVERRIDE, raising = False)
    monkeypatch.setattr(pup, "_candidate_revisions", lambda hub_cache = None: ["rev_new", "rev_old"])
    monkeypatch.setattr(
        pup, "_cached_file", lambda name, hub_cache = None, revision = None: files.get((revision, name))
    )
    # rev_new has no projector, so the pair is rev_old's, never rev_new's model with rev_old's projector.
    assert (pup.model_path(), pup.mmproj_path()) == ("/c/old/model", "/c/old/mmproj")
    del files[("rev_old", pup.UPSAMPLER_MMPROJ_FILE)]
    files[("rev_new", pup.UPSAMPLER_MMPROJ_FILE)] = None
    assert (pup.model_path(), pup.mmproj_path()) == ("/c/new/model", None)
    assert pup._cached_download_paths(pup.UPSAMPLER_ID) is None


def test_no_repo_download_beside_an_override(routes, monkeypatch):
    client, backend, saved = routes
    started = []
    monkeypatch.setattr(pup, "status", lambda: {"override": True})
    monkeypatch.setattr(pup, "start_download", lambda token: started.append(token) or "dl")
    r = client.post("/api/inference/images/prompt-upsampler/download")
    assert r.status_code == 409 and "UNSLOTH_FLUX3_UPSAMPLER_GGUF" in r.text and started == []


def test_a_systemic_failure_runs_the_rest_of_the_batch_as_typed(runnable):
    runnable.error = RuntimeError("llama-server did not become ready for the prompt upsampler.")
    out = pup.enhance_for_generation(
        ["a", "b", "c"], family_name = "flux.3-klein", enhance_prompt = None
    )
    # One start attempt, not one per prompt; every prompt falls back.
    assert len(runnable.calls) == 1
    assert out.statuses == {"a": "failed", "b": "failed", "c": "failed"} and out.captions == {}


def test_no_sidecar_start_after_the_gpu_moved_to_another_owner(monkeypatch):
    from core.inference import gpu_arbiter

    sidecar = pup.PromptUpsamplerSidecar()
    launched = []
    monkeypatch.setattr(sidecar, "_launch", lambda *a: launched.append(a))
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    with pytest.raises(pup.UpsamplerUnavailableError, match = "no longer holds the GPU"):
        sidecar.generate({}, binary = "b", model = "m", mmproj = None)
    assert launched == []


def test_the_preview_caps_the_aggregate_image_payload():
    from pydantic import ValidationError

    from models.inference import PromptUpsamplerPreviewRequest

    big = "x" * (30 * 1024 * 1024)
    PromptUpsamplerPreviewRequest(prompt = "p", reference_images = [big] * 4)
    with pytest.raises(ValidationError, match = "at most 128 MiB"):
        PromptUpsamplerPreviewRequest(prompt = "p", reference_images = [big] * 5)


def test_a_managed_account_is_authorized_before_the_upsampler_runs(routes, runnable, monkeypatch):
    from fastapi import HTTPException

    from hub.services.models import account_access

    client, backend, saved = routes
    monkeypatch.setattr(account_access, "managed_account", lambda: True)

    def deny(status, modality = None):
        raise HTTPException(status_code = 404, detail = "Model not found")

    monkeypatch.setattr(account_access, "require_media_generation_access", deny)
    r = client.post("/api/inference/images/generate", json = {"prompt": "a cat", "steps": 4})
    assert r.status_code == 404 and runnable.calls == []


# ── warm start ──────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def warmable(monkeypatch, tmp_path):
    """A sidecar that can start: the image model holds the GPU, no training, 160 of 180 GiB free."""
    from core.inference import gpu_arbiter
    from core.inference import stt_sidecar

    model = tmp_path / "pup.gguf"
    model.write_bytes(b"\0" * 1024)
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"\0" * 1024)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)
    slot = threading.Lock()

    def _take_slot():
        return (slot, "") if slot.acquire(blocking = False) else (None, "generating")

    monkeypatch.setattr(pup, "_take_render_slot", _take_slot)
    monkeypatch.setattr(pup, "_active_engine_loading", lambda: False)
    # One card: on a multi-GPU host an unpinned warm declines with "placement" (tests that need several patch it).
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: 1)
    memory = {"free": 160 * 1024, "total": 180 * 1024, "device": "cuda"}
    monkeypatch.setattr(
        pup,
        "_accelerator_memory_mib",
        lambda ordinal = None: (memory["free"], memory["total"], memory["device"]),
    )
    sidecar = pup.PromptUpsamplerSidecar()
    launched = []

    def _spawn(binary, model, mmproj):
        launched.append((binary, model, mmproj))
        sidecar._process = type("P", (), {"poll": lambda self: None})()
        sidecar._key = (binary, model, mmproj)

    monkeypatch.setattr(sidecar, "_spawn_server", _spawn)
    return sidecar, launched, memory, str(model), str(mmproj)


def test_warm_starts_the_server_and_arms_the_keep_alive(warmable):
    sidecar, launched, _memory, model, mmproj = warmable
    assert sidecar.warm(binary = "b", model = model, mmproj = mmproj) == "started"
    assert launched == [("b", model, mmproj)] and sidecar.loaded
    assert sidecar._active_requests == 0 and sidecar._idle_timer is not None
    # Already up for these files (or a text-only ask of a vision server): no second start, keep-alive refreshed.
    timer = sidecar._idle_timer
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "loaded"
    assert len(launched) == 1 and sidecar._idle_timer is not timer
    sidecar._cancel_idle_locked()


def test_warm_relaunches_for_other_files(warmable, monkeypatch):
    sidecar, launched, _memory, model, mmproj = warmable
    released = []
    monkeypatch.setattr(sidecar, "_release_locked", lambda: released.append(True))
    sidecar.warm(binary = "b", model = model, mmproj = None)
    # An edit needs the vision projector the text-only server lacks.
    assert sidecar.warm(binary = "b", model = model, mmproj = mmproj) == "started"
    assert released == [True] and launched[-1] == ("b", model, mmproj)
    sidecar._cancel_idle_locked()


@pytest.mark.parametrize(
    "free_mib, total_mib, device, expected",
    [
        (
            12 * 1024,
            24 * 1024,
            "cuda",
            "tight",
        ),  # 12 GiB free leaves ~3 GiB once it is up: under the 6 GiB floor
        (160 * 1024, 180 * 1024, "cuda", "started"),
        (None, None, "cuda", "memory"),
        (None, None, "cpu", "memory"),
    ],
)
def test_warm_declines_where_a_call_would_not_keep_it(
    warmable, monkeypatch, free_mib, total_mib, device, expected
):
    sidecar, launched, memory, model, _mmproj = warmable
    memory.update(free = free_mib, total = total_mib, device = device)
    # A 6.5 GB model: the size the shipped UD-Q4_K_XL file has.
    real = pup.os.path.getsize
    monkeypatch.setattr(
        pup.os.path, "getsize", lambda path: 6_500_000_000 if path == model else real(path)
    )
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == expected
    assert bool(launched) == (expected == "started")
    sidecar._cancel_idle_locked()


def test_warm_declines_without_the_gpu(warmable, monkeypatch):
    from core.inference import gpu_arbiter
    from core.inference import stt_sidecar

    sidecar, launched, _memory, model, _mmproj = warmable
    for owner in (None, gpu_arbiter.CHAT):
        monkeypatch.setattr(gpu_arbiter, "_owner", owner)
        assert sidecar.warm(binary = "b", model = model, mmproj = None) == "gpu"
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: True)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "gpu"
    assert launched == []


def test_warm_never_waits_on_a_request_or_starts_during_a_stop(warmable):
    sidecar, launched, _memory, model, _mmproj = warmable
    sidecar._request_lock.acquire()
    try:
        assert sidecar.warm(binary = "b", model = model, mmproj = None) == "busy"
    finally:
        sidecar._request_lock.release()
    sidecar._stopping = 1
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "stopped"
    sidecar._stopping = 0
    sidecar._update_in_progress = True
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "updating"
    assert launched == []


def test_warm_reports_a_cancelled_or_failed_start(warmable, monkeypatch):
    sidecar, _launched, _memory, model, _mmproj = warmable

    def _cancelled(*_a):
        raise pup.UpsamplerCancelledError("Cancelled.")

    monkeypatch.setattr(sidecar, "_spawn_server", _cancelled)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "cancelled"
    monkeypatch.setattr(sidecar, "_spawn_server", lambda *_a: 1 / 0)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "failed"
    assert sidecar._active_requests == 0 and not sidecar._request_lock.locked()
    assert not sidecar.is_loading()
    sidecar._cancel_idle_locked()


def test_warm_is_visible_to_training_before_its_probe(warmable, monkeypatch):
    """A training start between the eligibility probe and the spawn sees a load and cancels it (and its unload bumps
    the epoch), so exactly one of training and the warm start proceeds."""
    from core.inference import stt_sidecar

    sidecar, launched, _memory, model, _mmproj = warmable
    seen = []

    def _training_probe():
        seen.append(sidecar.is_loading())
        # Training admission runs now: free_prompt_upsampler_for_training finds the load and cancels it.
        sidecar.cancel_pending_load()
        return False

    monkeypatch.setattr(stt_sidecar, "_training_active", _training_probe)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "stopped"
    assert seen == [True] and launched == [] and not sidecar.is_loading()
    sidecar._cancel_idle_locked()


def test_warm_is_bound_to_the_epoch_it_was_scheduled_under(warmable):
    sidecar, launched, _memory, model, _mmproj = warmable
    epoch = sidecar.stop_epoch
    sidecar.unload(wait = True)  # a GPU handover between the route and the worker
    assert sidecar.warm(binary = "b", model = model, mmproj = None, expected_epoch = epoch) == "stopped"
    assert launched == []
    assert (
        sidecar.warm(binary = "b", model = model, mmproj = None, expected_epoch = sidecar.stop_epoch)
        == "started"
    )
    sidecar._cancel_idle_locked()


def test_a_declined_warm_keeps_the_running_server(warmable, monkeypatch):
    sidecar, launched, memory, model, mmproj = warmable
    released = []
    sidecar.warm(binary = "b", model = model, mmproj = None)
    monkeypatch.setattr(sidecar, "_release_locked", lambda: released.append(True))
    memory.update(free = None)
    assert sidecar.warm(binary = "b", model = model, mmproj = mmproj) == "memory"
    assert released == [] and len(launched) == 1
    sidecar._cancel_idle_locked()


def test_warm_for_family_runs_in_the_background(monkeypatch):
    calls = []

    class _Sidecar:
        loaded = False

        def is_loading(self):
            return False

        def serves(self, binary, model, mmproj):
            return False

        def warm(self, **kwargs):
            calls.append(kwargs)
            return "started"

    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(pup, "model_path", lambda: None)
    monkeypatch.setattr(pup, "mmproj_path", lambda: "/m/mmproj.gguf")
    monkeypatch.setattr(pup, "get_sidecar", lambda: _Sidecar())
    resident = {"value": ("unsloth/flux", "flux.3-klein-base")}
    monkeypatch.setattr(pup, "_resident_identity", lambda: resident["value"])
    assert pup.warm_for_family("flux.2-klein") == "unsupported"
    assert pup.warm_for_family("flux.3-klein") == "not_downloaded"
    monkeypatch.setattr(pup, "model_path", lambda: "/m/pup.gguf")
    assert pup.warm_for_family("flux.3-klein-base") == "scheduled"
    deadline = time.monotonic() + 5
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == [
        {
            "binary": "/bin/llama-server",
            "model": "/m/pup.gguf",
            "mmproj": "/m/mmproj.gguf",
            "expected_epoch": None,
        }
    ]
    # The model swapped before the worker ran: it never warms for the old one.
    resident["value"] = ("unsloth/other", "z-image")
    assert (
        pup.warm_for_family("flux.3-klein-base", resident = ("unsloth/flux", "flux.3-klein-base"))
        == "scheduled"
    )
    time.sleep(0.3)
    assert len(calls) == 1


def test_warm_route_warms_the_resident_family(routes, monkeypatch):
    client, backend, _saved = routes
    seen = []
    monkeypatch.setattr(
        pup, "warm_for_family", lambda family, **kw: seen.append((family, kw)) or "scheduled"
    )
    r = client.post("/api/inference/images/prompt-upsampler/warm")
    assert r.status_code == 200 and r.json() == {"warm": "scheduled"}
    assert seen == [
        (
            "flux.3-klein",
            {
                "expected_epoch": pup.get_sidecar().stop_epoch,
                "resident": ("unsloth/flux", "flux.3-klein"),
            },
        )
    ]
    monkeypatch.setattr(backend, "status", lambda: {"loaded": False})
    assert client.post("/api/inference/images/prompt-upsampler/warm").json() == {
        "warm": "not_loaded"
    }
    assert len(seen) == 1


def test_warm_route_skips_installations_with_managed_accounts(routes, monkeypatch):
    from auth import policy

    client, _backend, _saved = routes
    seen = []
    monkeypatch.setattr(
        pup, "warm_for_family", lambda family, **kw: seen.append(family) or "scheduled"
    )
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    assert client.post("/api/inference/images/prompt-upsampler/warm").json() == {"warm": "shared"}
    assert seen == []


def test_a_replacement_warm_credits_the_server_it_replaces(warmable, monkeypatch):
    """Free memory already excludes the text-only server a vision warm replaces, so its footprint counts back."""
    sidecar, launched, memory, model, mmproj = warmable
    real = pup.os.path.getsize
    monkeypatch.setattr(
        pup.os.path, "getsize", lambda path: 6_500_000_000 if path == model else real(path)
    )

    def _release():
        sidecar._process = sidecar._key = sidecar._gpu_layers = None

    monkeypatch.setattr(sidecar, "_release_locked", _release)
    sidecar.warm(binary = "b", model = model, mmproj = None)
    sidecar._gpu_layers = 99
    # 24 GiB card, 9 GiB free with the text-only server resident: ~17 GiB free once it goes, ~8.5 once replaced.
    memory.update(free = 9 * 1024, total = 24 * 1024)
    assert sidecar.warm(binary = "b", model = model, mmproj = mmproj) == "started"
    assert launched[-1] == ("b", model, mmproj)
    # Without a resident to reclaim, the same free memory is too tight.
    sidecar._release_locked()
    assert sidecar.warm(binary = "b", model = model, mmproj = mmproj) == "tight"
    sidecar._cancel_idle_locked()


def test_warm_declines_beside_a_render_and_holds_the_slot_while_it_starts(warmable, monkeypatch):
    sidecar, launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(pup, "_take_render_slot", lambda: (None, "generating"))
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "generating"
    assert launched == [] and not sidecar.is_loading()
    # While the server starts, the engine's generation slot is held: a render queues behind it.
    slot = threading.Lock()
    held = []
    monkeypatch.setattr(
        pup, "_take_render_slot", lambda: (slot.acquire(blocking = False) and slot, "")
    )
    real_spawn = sidecar._spawn_server
    monkeypatch.setattr(
        sidecar, "_spawn_server", lambda *a: held.append(slot.locked()) or real_spawn(*a)
    )
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "started"
    assert held == [True] and not slot.locked()
    sidecar._cancel_idle_locked()


def test_render_slot_is_never_taken_beside_a_render_or_a_load(monkeypatch):
    import core.inference.diffusion_engine_router as engine_router

    class _Engine:
        def __init__(self):
            self._generate_lock = threading.Lock()
            self._loading = None

    engine = _Engine()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: engine)
    slot, why = pup._take_render_slot()
    assert slot is engine._generate_lock and why == "" and slot.locked()
    # Held (a render, or a model transition): declined without waiting.
    assert pup._take_render_slot() == (None, "generating")
    slot.release()
    # A replacement load in flight: the old model still reads as loaded, so the load itself is what counts.
    engine._loading = type("L", (), {"error": None})()
    assert pup._take_render_slot() == (None, "loading") and not engine._generate_lock.locked()
    engine._loading = type("L", (), {"error": "boom"})()
    slot, _ = pup._take_render_slot()
    assert slot is not None
    slot.release()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: 1 / 0)
    assert pup._take_render_slot() == (None, "engine")


def test_a_replacement_warm_runs_on_the_background_thread(monkeypatch):
    """A server up for other files is replaced off the route: only a compatible one is refreshed inline."""
    calls, threads = [], []

    class _Sidecar:
        compatible = False

        def is_loading(self):
            return False

        def serves(self, binary, model, mmproj):
            return self.compatible

        def warm(self, **kwargs):
            calls.append(threading.current_thread().name)
            return "loaded" if self.compatible else "started"

    sidecar = _Sidecar()
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(pup, "model_path", lambda: "/m/pup.gguf")
    monkeypatch.setattr(pup, "mmproj_path", lambda: "/m/mmproj.gguf")
    monkeypatch.setattr(pup, "get_sidecar", lambda: sidecar)
    monkeypatch.setattr(pup, "_resident_identity", lambda: None)
    assert pup.warm_for_family("flux.3-klein") == "scheduled"
    deadline = time.monotonic() + 5
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == ["flux3-upsampler-warm"]
    sidecar.compatible = True
    assert pup.warm_for_family("flux.3-klein") == "loaded"
    assert calls[-1] == threading.current_thread().name


def test_a_partial_offload_credits_only_its_resident_share(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0" * (32 * 1024 * 1024))  # 32 MiB: 1 MiB per layer
    full = pup._footprint_mib(str(model), None)
    assert full == 32 + pup._GPU_OVERHEAD_MIB
    assert pup._footprint_mib(str(model), None, 99) == full
    assert pup._footprint_mib(str(model), None, 8) == 8 + pup._GPU_OVERHEAD_MIB


def test_warm_declines_where_it_cannot_be_pinned(warmable, monkeypatch):
    sidecar, launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: None)
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: 2)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "placement"
    # One visible card needs no pin.
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: 1)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "started"
    assert len(launched) == 1
    sidecar._cancel_idle_locked()


def test_a_stop_on_a_request_queued_behind_a_warm_start_cancels_it(warmable, monkeypatch):
    sidecar, _launched, _memory, model, _mmproj = warmable
    started, release = threading.Event(), threading.Event()

    def _slow_spawn(binary, model, mmproj):
        started.set()
        # A real start polls the cancel event while llama-server comes up.
        while not sidecar._load_cancel.is_set() and not release.is_set():
            time.sleep(0.01)
        if sidecar._load_cancel.is_set():
            raise pup.UpsamplerCancelledError("cancelled")

    monkeypatch.setattr(sidecar, "_spawn_server", _slow_spawn)
    outcomes = []
    warm = threading.Thread(
        target = lambda: outcomes.append(sidecar.warm(binary = "b", model = model, mmproj = None))
    )
    warm.start()
    assert started.wait(5) and sidecar._warming
    cancel = threading.Event()
    errors = []

    def _queued():
        try:
            sidecar.generate({}, binary = "b", model = model, mmproj = None, cancel_event = cancel)
        except pup.UpsamplerCancelledError as exc:
            errors.append(exc)

    queued = threading.Thread(target = _queued)
    queued.start()
    time.sleep(0.3)
    cancel.set()  # Stop pressed while the request waits on the warm start
    queued.join(5)
    warm.join(5)
    release.set()
    assert len(errors) == 1 and outcomes == ["cancelled"]
    assert not sidecar._warming and not sidecar._request_lock.locked()
    sidecar._cancel_idle_locked()


def test_cancel_warm_start_only_touches_a_warm_start(monkeypatch):
    sidecar = pup.PromptUpsamplerSidecar()
    monkeypatch.setattr(pup, "_sidecar", sidecar)
    assert pup.cancel_warm_start() is False
    # A request's own start is not a warm start: a teardown stops it through stop_for_handover as before.
    sidecar._loading = True
    assert pup.cancel_warm_start() is False and not sidecar._load_cancel.is_set()
    sidecar._warming = True
    assert pup.cancel_warm_start() is True and sidecar._load_cancel.is_set()
    monkeypatch.setattr(pup, "_sidecar", None)
    assert pup.cancel_warm_start() is False


def test_teardowns_cancel_a_warm_start_before_they_unload(routes, monkeypatch):
    import sys as _sys

    from core.inference import gpu_arbiter

    client, backend, _saved = routes
    order = []
    monkeypatch.setattr(pup, "cancel_warm_start", lambda: order.append("cancel") or True)
    monkeypatch.setattr(pup, "stop_for_handover", lambda: order.append("stop"))
    monkeypatch.setitem(_sys.modules, "core.inference.flux3_prompt_upsampler", pup)
    monkeypatch.setattr(
        backend, "unload", lambda **_: order.append("unload") or {"loaded": False}, raising = False
    )
    assert client.post("/api/inference/images/unload", json = {}).status_code == 200
    assert order[:2] == ["cancel", "unload"]
    order.clear()
    gpu_arbiter._evict_diffusion()
    assert order == ["cancel", "unload", "stop"]


def test_an_unenhanced_run_does_not_wait_out_a_warm_start(routes, runnable, monkeypatch):
    import sys as _sys

    client, backend, _saved = routes
    cancels = []
    monkeypatch.setattr(pup, "cancel_warm_start", lambda: cancels.append(True) or False)
    monkeypatch.setitem(_sys.modules, "core.inference.flux3_prompt_upsampler", pup)
    r = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a cat", "steps": 4, "enhance_prompt": False},
    )
    assert r.status_code == 200, r.text
    assert cancels == [True]
    # An enhanced run used the server the warm start brought up: nothing to cancel.
    r = client.post("/api/inference/images/generate", json = {"prompt": "a dog", "steps": 4})
    assert r.status_code == 200, r.text
    assert cancels == [True]


def test_a_load_published_while_the_server_came_up_drops_it(warmable, monkeypatch):
    """The window between a load's stop_for_handover and begin_load publishing it: the started server is released."""
    sidecar, launched, _memory, model, _mmproj = warmable
    released = []
    real_release = sidecar._release_locked

    def _release():
        released.append(True)
        sidecar._process = sidecar._key = None

    monkeypatch.setattr(sidecar, "_release_locked", _release)
    monkeypatch.setattr(pup, "_active_engine_loading", lambda: True)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "stale"
    assert launched and released == [True] and not sidecar.loaded
    monkeypatch.setattr(sidecar, "_release_locked", real_release)
    sidecar._cancel_idle_locked()


def test_release_route_cancels_a_warm_start_and_stops_an_idle_server(routes, monkeypatch):
    client, _backend, _saved = routes
    calls = []
    monkeypatch.setattr(pup, "cancel_warm_start", lambda: calls.append("cancel") or True)
    monkeypatch.setattr(pup, "unload_sidecar", lambda wait = False: calls.append(("unload", wait)))
    r = client.post("/api/inference/images/prompt-upsampler/release")
    assert r.status_code == 200 and r.json() == {"released": True, "cancelled_warm": True}
    # Non-blocking: a request still using the server keeps it.
    assert calls == ["cancel", ("unload", False)]


# ── layout captions (bounding boxes) ────────────────────────────────────────────────────────────────────────────────
# A finished caption: every box id is cited as <id> (BFL's generate.py sends it to the DiT as it is).
_FINISHED = (
    "A white canvas with a red circle <circle_1> upper left and a blue square <square_1> lower right. "
    '[{"id": "circle_1", "bbox": [100, 100, 400, 400], "desc": "solid red circle"}, '
    '{"id": "square_1", "bbox": [600, 600, 900, 900], "desc": "solid blue square"}]'
)
# A request with boxes (BFL examples/inputs/balloons_boxes.json on the 0-1000 canvas): no id is cited yet.
_REQUEST = (
    "hot-air balloons over a green valley at sunrise "
    '[{"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "a red and yellow striped hot-air balloon"}]'
)


@pytest.mark.parametrize(
    "prompt, finished, boxes",
    [
        (_FINISHED, True, True),
        (_REQUEST, False, True),
        ("A poster that reads [SALE]", False, False),
        ('A list ["a", "b"]', False, False),
        # 0-1 coordinates are not the 0-1000 canvas.
        ('x <o_1> [{"id": "o_1", "bbox": [0.1, 0.1, 0.4, 0.4], "desc": "d"}]', False, False),
    ],
)
def test_box_table_detection(prompt, finished, boxes):
    assert pup.is_finished_layout_caption(prompt) is finished
    assert (pup.detect_box_table(prompt) is not None) is boxes


def test_a_finished_layout_caption_skips_the_upsampler(runnable):
    assert pup.upsample(_FINISHED) == _FINISHED
    out = pup.enhance_for_generation([_FINISHED], family_name = "flux.3-klein", enhance_prompt = None)
    assert runnable.calls == []
    # Used as it is, which is what enhancement off records, not a failed rewrite.
    assert (
        out.status == "off"
        and out.encoded(_FINISHED) == _FINISHED
        and out.caption_for(_FINISHED) is None
    )


def test_a_box_request_uses_the_layout_prompt_and_keeps_the_boxes(runnable):
    def reply(payload):
        # The upsampler re-plans obj_1 under its own id and adds a row; reconcile restores the locked box and id.
        return (
            "Full-color photograph with a balloon <balloon_1> and the sun <sun_1>. "
            '[{"id": "balloon_1", "bbox": [82, 58, 478, 381], "desc": "A striped balloon."}, '
            '{"id": "sun_1", "bbox": [470, 455, 535, 525], "desc": "A low sun."}]'
        )

    runnable.reply = reply
    caption = pup.upsample(_REQUEST, variant = "base")
    payload = runnable.calls[0]["payload"]
    assert payload["messages"][0]["content"] == pup.SYSTEM_PROMPTS["t2i_bbox"]
    text = payload["messages"][1]["content"][0]["text"]
    assert text.startswith(
        "ORIGINAL PROMPT:\nhot-air balloons over a green valley at sunrise\n\nLOCKED"
    )
    assert '"bbox": [80, 60, 480, 380]' in text
    assert caption == (
        "Full-color photograph with a balloon <obj_1> and the sun <sun_1>. "
        '[{"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "A striped balloon."}, '
        '{"id": "sun_1", "bbox": [470, 455, 535, 525], "desc": "A low sun."}]'
    )


def test_a_box_edit_uses_the_layout_edit_prompt(runnable):
    pup.upsample(_REQUEST, reference_images = [_png_url()])
    content = runnable.calls[0]["payload"]["messages"][1]["content"]
    assert runnable.calls[0]["payload"]["messages"][0]["content"] == pup.SYSTEM_PROMPTS["i2i_bbox"]
    assert content[0]["type"] == "image_url"
    assert content[-1]["text"].startswith(
        "Reference images follow in attachment order: ref_image_0"
    )
    assert "User editing instruction: ORIGINAL PROMPT:" in content[-1]["text"]


def test_a_renamed_row_matches_the_first_valid_locked_box():
    # detect_box_table accepts a row with any valid box, so a malformed bbox beside a valid tgt_bbox is locked.
    locked = pup.detect_box_table(
        'Two balloons [{"id": "obj_1", "bbox": [100, 200], "from": "target", '
        '"tgt_bbox": [100, 100, 500, 400], "desc": "red balloon"}]'
    )[1]
    answer = 'A red balloon <obj_9>. [{"id": "obj_9", "bbox": [100, 100, 500, 400], "desc": "A red balloon."}]'
    caption, rows = pup.split_box_table(pup.reconcile_layout_caption(answer, locked))
    assert caption == "A red balloon <obj_1>."
    assert [row["id"] for row in rows] == ["obj_1"] and rows[0]["desc"] == "A red balloon."


def test_the_launch_reprobes_a_cached_empty_vulkan_answer(monkeypatch):
    # A transient empty probe in warm() must not leave the real launch unpinned for the TTL.
    from core.inference import llama_cpp

    answers = [[], list(_VULKAN_ROWS)]
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_run_vulkan_probe", staticmethod(lambda b = None: answers.pop(0))
    )
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    assert pup._vulkan_rows("/x/llama-server") == []
    assert pup._vulkan_rows("/x/llama-server") == []
    assert pup._vulkan_rows("/x/llama-server", retry_empty = True) == _VULKAN_ROWS


@pytest.mark.parametrize(
    "var, value", [("LLAMA_ARG_DEVICE", "CUDA1"), ("LLAMA_ARG_DEVICE", "ROCm1")]
)
@pytest.mark.parametrize("count, expected", [(2, "placement"), (1, "started")])
def test_a_user_device_pick_declines_the_warm_start_beside_other_cards(
    warmable, monkeypatch, var, value, count, expected
):
    # The memory probe reads the image model's card, not the one the user picked.
    sidecar, _launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: 0)
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: count)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "started"
    sidecar._cancel_idle_locked()
    sidecar._process = sidecar._key = None
    monkeypatch.setenv(var, value)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == expected
    sidecar._cancel_idle_locked()


@pytest.mark.parametrize("count, expected", [(2, None), (1, [])])
def test_an_external_backend_path_is_unknown_placement_beside_other_cards(
    monkeypatch, count, expected
):
    import torch
    from core.inference import llama_cpp

    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False)
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setenv("GGML_BACKEND_PATH", "/opt/ggml/libggml-vulkan.so")
    assert pup._vulkan_device_args("b", 0) == expected
    monkeypatch.delenv("GGML_BACKEND_PATH")
    assert pup._vulkan_device_args("b", 0) == []


def test_the_vulkan_probe_runs_before_the_render_slot(warmable, monkeypatch):
    # A slow probe under the slot would hold renders and handovers until its timeout.
    from core.inference import llama_cpp

    import torch

    sidecar, _launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    order = []
    take = pup._take_render_slot
    monkeypatch.setattr(pup, "_take_render_slot", lambda: order.append("slot") or take())
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: True)
    )
    monkeypatch.setattr(
        pup, "_vulkan_rows", lambda b, retry_empty = False, **_kw: order.append("probe") or []
    )
    sidecar.warm(binary = "b", model = model, mmproj = None)
    # An empty answer is retried once, still before the slot.
    assert order[: order.index("slot")] == ["probe", "probe"]
    sidecar._cancel_idle_locked()


def test_repeated_hip_entries_do_not_shift_the_pin(monkeypatch):
    # clr keeps the first of a repeated HIP entry, so HIP=0,0,1 exposes GPUs 0 and 1.
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.version, "hip", "6.2", raising = False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,0,1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert pup._child_device_mask(1) == {"HIP_VISIBLE_DEVICES": "1"}
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert pup._child_device_mask(1) == {"CUDA_VISIBLE_DEVICES": "5"}


@pytest.mark.parametrize("vulkan, expected", [(False, "started"), (True, "placement")])
def test_a_vulkan_mask_declines_the_warm_start_only_for_a_vulkan_build(
    warmable, monkeypatch, vulkan, expected
):
    # CUDA and HIP builds ignore GGML_VK_VISIBLE_DEVICES; their launch is pinned by the visibility mask.
    from core.inference import llama_cpp

    sidecar, _launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: 0)
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: 2)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: vulkan)
    )
    monkeypatch.setattr(pup, "_vulkan_rows", lambda b, retry_empty = False: [])
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1")
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == expected
    sidecar._cancel_idle_locked()


@pytest.mark.parametrize("hip, probed", [(None, False), ("6.2", True)])
def test_a_hip_mask_on_a_cuda_torch_never_steers_the_vulkan_pin(
    spawn_argv, monkeypatch, hip, probed
):
    # Only ROCm applies HIP_VISIBLE_DEVICES; on a CUDA torch the card lookup would name another card of the pair.
    import torch

    state, _run = spawn_argv
    calls = []
    monkeypatch.setattr(pup, "_vulkan_rows", lambda b, retry_empty = False: calls.append(b) or [])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    # ROCm is also read from a "+rocm" __version__: pin it to the build this case names.
    monkeypatch.setattr(torch, "__version__", "2.11.0+rocm6.2" if hip else "2.11.0+cu130")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
    assert pup._vulkan_device_args("b", 0) is None
    assert bool(calls) is probed


def test_a_negative_ordinal_is_never_pinned():
    assert pup._child_device_mask(-1) == {}
    assert pup._vulkan_device_args("b", -1) == []


@pytest.mark.parametrize(
    "row, pinned",
    [("AMD Radeon RX 7600 XT (RADV NAVI33)", False), ("AMD Radeon RX 7600 (RADV NAVI33)", True)],
)
def test_a_loose_name_match_beside_other_vulkan_devices_is_unknown(
    spawn_argv, monkeypatch, row, pinned
):
    # An iGPU beside the XT must not make the XT stand in for the RX 7600 the memory probe read.
    import torch

    state, _run = spawn_argv
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    state["names"] = ["AMD Radeon RX 7600"]
    state["ordinal"] = 0
    state["rows"] = [_VULKAN_ROWS[0], {"index": 1, "name": row}]
    assert pup._vulkan_device_args("b", 0) == (["--device", "Vulkan1"] if pinned else None)


@pytest.mark.parametrize("vulkan", [True, False])
def test_an_external_backend_path_is_unknown_even_for_a_vulkan_bundle(monkeypatch, vulkan):
    # The probe would list the bundled Vulkan library, not the backends GGML_BACKEND_PATH loads.
    import torch
    from core.inference import llama_cpp

    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: vulkan)
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    calls = []
    monkeypatch.setattr(
        pup, "_vulkan_rows", lambda b, retry_empty = False: calls.append(b) or list(_VULKAN_ROWS)
    )
    monkeypatch.setenv("GGML_BACKEND_PATH", "/opt/ggml/backends")
    assert pup._vulkan_device_args("b", 0) is None
    assert calls == []


def test_a_user_main_gpu_is_a_device_pick(monkeypatch):
    # A one-device --device list would leave LLAMA_ARG_MAIN_GPU=1 out of range under split mode none.
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)
    monkeypatch.setenv("LLAMA_ARG_MAIN_GPU", "1")
    assert pup._user_device_pick() is True
    assert pup._vulkan_device_args("b", 0) == []


from core.inference import sd_cpp_backend as _sd_cpp_backend  # noqa: E402

_REAL_PHYSICAL_POSITION_OF = _sd_cpp_backend._physical_position_of


def _real_card_lookup(monkeypatch, amd_rows, hip_by_row):
    """Undo the fixture's position stub: the real lookup over a mocked amd-smi inventory and HIP id map."""
    from core.inference import sd_cpp_backend
    from utils.hardware import amd

    monkeypatch.setattr(sd_cpp_backend, "_physical_position_of", _REAL_PHYSICAL_POSITION_OF)
    inventory = {
        "devices": [{"index": i, "vendor": "amd", "name": name} for i, name in enumerate(amd_rows)]
    }
    monkeypatch.setattr(sd_cpp_backend, "_card_lookup_inventory", lambda: inventory)
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: dict(hip_by_row))


_MIXED_ROWS = [
    {"index": 0, "name": "AMD Radeon Graphics (RADV RAPHAEL)"},
    {"index": 1, "name": "AMD Radeon RX 7600 (RADV NAVI33)"},
    {"index": 2, "name": "NVIDIA GeForce RTX 4090"},
    {"index": 3, "name": "NVIDIA GeForce RTX 4090"},
]


@pytest.mark.parametrize("ordinal, device", [(0, "Vulkan2"), (1, "Vulkan3")])
def test_a_cuda_torch_never_takes_an_amd_cards_position(spawn_argv, monkeypatch, ordinal, device):
    # Two RTX 4090s on a CUDA torch beside AMD cards amd-smi reads: HIP device 1 (the RX 7600) is no torch cuda:1.
    import torch

    state, run = spawn_argv
    _real_card_lookup(monkeypatch, ["AMD Radeon Graphics", "AMD Radeon RX 7600"], {0: 0, 1: 1})
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    state["names"] = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"]
    state["rows"] = _MIXED_ROWS
    state["ordinal"] = ordinal
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    argv, env = run()
    assert argv[-2:] == ["--device", device] and env == {"CUDA_VISIBLE_DEVICES": str(ordinal)}


@pytest.mark.parametrize("order", [None, "FASTEST_FIRST"])
def test_identical_cuda_cards_are_not_told_apart_outside_pci_order(
    spawn_argv, warmable, monkeypatch, order
):
    # CUDA's default FASTEST_FIRST may not match Vulkan's (PCI) order, so torch's position among twins is no pin.
    import torch

    state, run = spawn_argv
    sidecar, launched, _memory, model, _mmproj = warmable
    _real_card_lookup(monkeypatch, ["AMD Radeon Graphics", "AMD Radeon RX 7600"], {0: 0, 1: 1})
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: 2)
    state["names"] = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"]
    state["rows"] = _MIXED_ROWS
    state["ordinal"] = 1
    if order is None:
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    else:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    assert pup._vulkan_device_args("b", 1) is None
    assert "--device" not in run()[0]
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "placement"
    # A card with no twin needs no order to be found.
    state["names"] = ["NVIDIA GeForce RTX 4090", "AMD Radeon RX 7600"]
    assert pup._vulkan_device_args("b", 0) == ["--device", "Vulkan2"]


@pytest.mark.parametrize(
    "var, value", [("CUDA_VISIBLE_DEVICES", "1,0"), ("ROCR_VISIBLE_DEVICES", "1,0")]
)
def test_a_cuda_torch_mask_never_steers_the_vulkan_pin_onto_a_guess(
    spawn_argv, warmable, monkeypatch, var, value
):
    # CVD reorders the identical pair (the order Vulkan cannot see); ROCR is not read by a CUDA torch at all.
    import torch

    state, run = spawn_argv
    sidecar, launched, _memory, model, _mmproj = warmable
    _real_card_lookup(monkeypatch, ["AMD Radeon Graphics", "AMD Radeon RX 7600"], {0: 0, 1: 1})
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: 2)
    state["names"] = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"]
    state["rows"] = _MIXED_ROWS
    state["ordinal"] = 1
    monkeypatch.setenv(var, value)
    expected = None if var == "CUDA_VISIBLE_DEVICES" else ["--device", "Vulkan3"]
    assert pup._vulkan_device_args("b", 1) == expected
    if expected is None:
        assert "--device" not in run()[0]
        assert sidecar.warm(binary = "b", model = model, mmproj = None) == "placement"


@pytest.mark.parametrize("vulkan", [False, True])
def test_one_torch_card_and_one_vulkan_device_launch_as_before(
    spawn_argv, warmable, monkeypatch, vulkan
):
    # The pre-pinning launch: no --device, no mask, a start. A non-Vulkan build never probes; one Vulkan row (even a
    # renamed `Radeon(TM)` card) is nothing to choose between, so no name match can decline it.
    import torch

    from core.inference import llama_cpp

    state, run = spawn_argv
    sidecar, launched, _memory, model, _mmproj = warmable
    calls = []
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_run_vulkan_probe",
        staticmethod(
            lambda b = None: calls.append(b)
            or [{"index": 0, "name": "AMD Radeon Graphics (RADV RAPHAEL)"}]
        ),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    state["vulkan"] = vulkan
    state["names"] = ["AMD Radeon(TM) Graphics"]
    state["ordinal"] = 0
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: 0)
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "started"
    sidecar._cancel_idle_locked()
    argv, env = run()
    assert "--device" not in argv and env == {}
    assert bool(calls) is vulkan


@pytest.mark.parametrize("cards", [1, 2])
def test_a_warm_start_retries_an_empty_vulkan_answer_once(spawn_argv, warmable, monkeypatch, cards):
    # The warm start retries an empty answer once, as the launch it stands in for did; its real spawn adds none.
    import torch

    from core.inference import llama_cpp

    state, _run = spawn_argv
    sidecar, launched, _memory, model, _mmproj = warmable
    calls = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: cards)
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: cards)
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: 0)
    monkeypatch.setattr(pup, "_VULKAN_ROWS_TTL_S", 120.0)
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_run_vulkan_probe",
        staticmethod(lambda b = None: calls.append(b) or []),
    )
    monkeypatch.setattr(
        sidecar, "_spawn_server", pup.PromptUpsamplerSidecar._spawn_server.__get__(sidecar)
    )
    # Several cards: declined, not started unpinned; the first enhanced run re-probes (and pins, or launches as
    # before). One card goes on to the spawn unpinned, as before pinning.
    out = sidecar.warm(binary = "/x/llama-server", model = model, mmproj = None)
    assert (out == "placement") is (cards > 1)
    assert len(calls) == 2


def test_maintenance_waits_for_every_abandoned_vulkan_probe(monkeypatch):
    # Two warm starts cancelled within one probe's timeout leave two probes holding the llama.cpp libraries.
    import torch

    from core.inference import llama_cpp

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: True)
    )
    for var in (
        "HIP_VISIBLE_DEVICES",
        "GGML_BACKEND_PATH",
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_MAIN_GPU",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    monkeypatch.setattr(pup, "_vulkan_probe_workers", [], raising = False)
    releases = [threading.Event(), threading.Event()]
    entered = []

    def _probe(b = None):
        release = releases[len(entered)]
        entered.append(b)
        release.wait(10)
        return []

    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "_run_vulkan_probe", staticmethod(_probe))
    try:
        for _ in releases:
            cancel = threading.Event()
            threading.Timer(0.2, cancel.set).start()
            with pytest.raises(pup.UpsamplerCancelledError):
                pup._prefetch_vulkan_rows("b", cancel)
        releases[1].set()
        waiter = threading.Thread(target = pup._wait_for_vulkan_probe, args = (10,))
        waiter.start()
        waiter.join(0.5)
        assert waiter.is_alive()
        releases[0].set()
        waiter.join(5)
        assert not waiter.is_alive()
    finally:
        for release in releases:
            release.set()


@pytest.mark.parametrize(
    "name, row",
    [
        ("AMD Radeon(TM) Graphics", "AMD Radeon Graphics (RADV RAPHAEL)"),
        ("AMD Radeon(TM) 8060S Graphics", "AMD Radeon 8060S Graphics (RADV GFX1151)"),
    ],
)
def test_trademark_marks_do_not_hide_the_card(spawn_argv, warmable, monkeypatch, name, row):
    import torch

    state, run = spawn_argv
    sidecar, launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: 2)
    state["names"] = [name, "AMD Radeon RX 7900 XTX"]
    state["rows"] = [{"index": 0, "name": row}, _VULKAN_ROWS[2]]
    state["ordinal"] = 0
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: 0)
    assert run()[0][-2:] == ["--device", "Vulkan0"]
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "started"
    sidecar._cancel_idle_locked()


def test_a_cancelled_warm_start_does_not_wait_out_the_vulkan_probe(warmable, monkeypatch):
    # A blocking unload (handover, training start) must not sit behind a probe that can take its 15 s timeout.
    from core.inference import llama_cpp

    sidecar, launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: 2)
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: True)
    )
    for var in (
        "HIP_VISIBLE_DEVICES",
        "GGML_BACKEND_PATH",
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_MAIN_GPU",
    ):
        monkeypatch.delenv(var, raising = False)
    entered, release = threading.Event(), threading.Event()

    def _probe(b = None):
        entered.set()
        release.wait(10)
        return list(_VULKAN_ROWS)

    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "_run_vulkan_probe", staticmethod(_probe))
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    out = []
    warm = threading.Thread(
        target = lambda: out.append(sidecar.warm(binary = "b", model = model, mmproj = None))
    )
    warm.start()
    try:
        assert entered.wait(5)
        assert sidecar.cancel_warm_start()
        unload = threading.Thread(target = sidecar.unload)
        unload.start()
        unload.join(3)
        assert not unload.is_alive()
        warm.join(3)
        assert out == ["cancelled"] and launched == []
    finally:
        release.set()
        warm.join(5)


def test_maintenance_waits_for_a_probe_a_cancelled_warm_start_left_running(warmable, monkeypatch):
    # The probe loads the llama.cpp tree's libraries; replacing that tree under it fails on Windows.
    import torch

    from core.inference import llama_cpp

    sidecar, launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(pup, "_cuda_device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: True)
    )
    for var in (
        "HIP_VISIBLE_DEVICES",
        "GGML_BACKEND_PATH",
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_MAIN_GPU",
    ):
        monkeypatch.delenv(var, raising = False)
    entered, release = threading.Event(), threading.Event()

    def _probe(b = None):
        entered.set()
        release.wait(10)
        return list(_VULKAN_ROWS)

    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "_run_vulkan_probe", staticmethod(_probe))
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    out, phase = [], []
    warm = threading.Thread(
        target = lambda: out.append(sidecar.warm(binary = "b", model = model, mmproj = None))
    )
    warm.start()

    def _maintain():
        with sidecar.update_maintenance():
            phase.append("replacing")

    try:
        assert entered.wait(5)
        maint = threading.Thread(target = _maintain)
        maint.start()
        warm.join(3)
        assert out == ["cancelled"]
        maint.join(0.5)
        assert phase == [] and maint.is_alive()
        release.set()
        maint.join(5)
        assert phase == ["replacing"] and launched == []
    finally:
        release.set()
        warm.join(5)


def _prefetch_host(monkeypatch, cards, rows):
    import torch

    from core.inference import llama_cpp

    calls, looked_up = [], []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: cards)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: True)
    )
    for var in (
        "HIP_VISIBLE_DEVICES",
        "GGML_BACKEND_PATH",
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_MAIN_GPU",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_run_vulkan_probe",
        staticmethod(lambda b = None: calls.append(b) or list(rows)),
    )
    monkeypatch.setattr(pup, "_VULKAN_ROWS_TTL_S", 120.0)
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: 0)
    monkeypatch.setattr(pup, "_vulkan_card_name", lambda o, t: looked_up.append(o) or ("x", None))
    monkeypatch.setattr(pup, "_vulkan_probe_workers", [])
    return calls, looked_up


def test_a_warm_start_prefetch_retries_a_cached_empty_probe(monkeypatch):
    # The launch a warm start stands in for re-probed a (maybe transient) empty answer; the prefetch does too.
    calls, looked_up = _prefetch_host(monkeypatch, 1, _VULKAN_ROWS)
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {("b", None): ([], time.monotonic())})
    pup._prefetch_vulkan_rows("b")
    assert calls == ["b"] and looked_up == [0]
    # A cached non-empty answer is reused.
    pup._prefetch_vulkan_rows("b")
    assert calls == ["b"]


def test_one_card_and_one_vulkan_row_skips_the_card_lookup(monkeypatch):
    # That launch stays unpinned, so the prefetch reads no inventory for it.
    calls, looked_up = _prefetch_host(monkeypatch, 1, _VULKAN_ROWS[:1])
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    pup._prefetch_vulkan_rows("b")
    assert calls == ["b"] and looked_up == []


def test_a_start_cancelled_before_its_probe_never_starts_one(monkeypatch):
    calls, _looked_up = _prefetch_host(monkeypatch, 2, _VULKAN_ROWS)
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(pup.UpsamplerCancelledError):
        pup._prefetch_vulkan_rows("b", cancel)
    assert calls == [] and pup._vulkan_probe_workers == []


def test_a_lone_vulkan_device_of_another_vendor_declines_the_warm_start(
    spawn_argv, warmable, monkeypatch
):
    # A CUDA torch's image model on an RTX 4090 while Vulkan reaches only an AMD iGPU: not the same card.
    import torch

    state, run = spawn_argv
    sidecar, launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    state["names"] = ["NVIDIA GeForce RTX 4090"]
    state["ordinal"] = 0
    state["rows"] = [{"index": 0, "name": "AMD Radeon Graphics (RADV RAPHAEL)"}]
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "placement"
    # The explicit launch is unchanged: unpinned, as before Vulkan pinning.
    argv, env = run()
    assert "--device" not in argv
    # Its own NVIDIA row warm-starts as before.
    state["rows"] = [{"index": 0, "name": "NVIDIA GeForce RTX 4090"}]
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "started"
    sidecar._cancel_idle_locked()


@pytest.mark.parametrize(
    "name", ["Tesla T4", "Quadro RTX 8000", "GeForce GTX 1080 Ti", "NVIDIA GeForce RTX 4090"]
)
def test_an_nvidia_vulkan_row_without_the_vendor_name_warm_starts(
    spawn_argv, warmable, monkeypatch, name
):
    # Older NVIDIA drivers name the Vulkan device "Tesla T4": that is the image model's card, not another vendor's.
    import torch

    state, _run = spawn_argv
    sidecar, _launched, _memory, model, _mmproj = warmable
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    state["names"] = [name]
    state["ordinal"] = 0
    state["rows"] = [{"index": 0, "name": name}]
    assert sidecar.warm(binary = "b", model = model, mmproj = None) == "started"
    sidecar._cancel_idle_locked()


@pytest.mark.parametrize(
    "rocm, card, row, other",
    [
        (False, "Tesla T4", "AMD Radeon Graphics (RADV RAPHAEL)", True),
        (False, "NVIDIA GeForce RTX 4090", "Intel(R) Arc(tm) A770 Graphics (DG2)", True),
        (False, "NVIDIA GeForce RTX 4090", "llvmpipe (LLVM 17.0.6, 256 bits)", True),
        (True, "AMD Radeon RX 7900 XTX", "NVIDIA GeForce RTX 3060", True),
        (True, "AMD Radeon RX 7900 XTX", "AMD Radeon RX 7900 XTX (RADV NAVI31)", False),
        (False, "Tesla T4", "Tesla T4", False),
        # A generic card name never outweighs a row that names another vendor.
        (False, "NVIDIA Graphics Device", "Intel(R) Graphics", True),
        (False, "GeForce GTX 1080 Ti", "NVIDIA GeForce GTX 1080 Ti", False),
        (False, "NVIDIA A100-SXM4-40GB", "Virtual GPU 0", False),
        (True, "AMD Radeon RX 7900 XTX", "Virtual GPU 0", False),
    ],
)
def test_another_vendor_needs_positive_evidence(monkeypatch, rocm, card, row, other):
    # Only a row naming a different vendor is another card; an unknown name is not evidence either way.
    from core.inference import llama_cpp
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda t = None: rocm)
    )
    assert pup._vulkan_row_is_another_vendor(row, object()) is other


def test_repeated_prefetches_retry_an_empty_probe_once_per_window(monkeypatch):
    # The retry stands for the window: two prefetches cost one probe and one retry, not two each.
    calls, _looked_up = _prefetch_host(monkeypatch, 2, [])
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    pup._prefetch_vulkan_rows("b")
    pup._prefetch_vulkan_rows("b")
    assert calls == ["b", "b"]
    # A new window probes (and may retry) again.
    monkeypatch.setattr(pup, "_VULKAN_ROWS_TTL_S", 0.0)
    pup._prefetch_vulkan_rows("b")
    assert len(calls) == 4


def test_a_real_launch_reprobes_after_throttled_prefetches(monkeypatch):
    # Prefetches stop retrying for the window, a launch does not: a device that came up since is found.
    calls, _looked_up = _prefetch_host(monkeypatch, 2, [])
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    pup._prefetch_vulkan_rows("b")
    pup._prefetch_vulkan_rows("b")
    assert calls == ["b", "b"]
    pup._vulkan_rows("b", retry_empty = True)
    assert len(calls) == 3


def test_concurrent_prefetches_share_one_probe(monkeypatch):
    calls, _looked_up = _prefetch_host(monkeypatch, 2, [])
    monkeypatch.setattr(pup, "_vulkan_rows_memo", {})
    threads = [threading.Thread(target = pup._prefetch_vulkan_rows, args = ("b",)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert calls == ["b", "b"]


def test_maintenance_never_joins_an_unstarted_probe(monkeypatch):
    # A worker is registered only once started, so a maintenance pass cannot hit "cannot join thread".
    import inspect
    source = inspect.getsource(pup._prefetch_vulkan_rows)
    assert source.index("worker.start()") < source.index("_vulkan_probe_workers[:]")
