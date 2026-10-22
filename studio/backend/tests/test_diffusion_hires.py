# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hires fix and model upscalers."""

from __future__ import annotations

import inspect
import re
import types

import pytest

from core.inference import diffusion_hires as hires
from core.inference import diffusion_upscalers as ups
from core.inference.diffusion_families import _FAMILIES


def _fam(name):
    return next(f for f in _FAMILIES if f.name == name)


def test_request_defaults_follow_the_comfyui_two_pass_example():
    from models.inference import DiffusionGenerateRequest, HiresSpec

    spec = HiresSpec()
    assert (spec.scale, spec.denoise, spec.steps, spec.upscaler, spec.guidance) == (
        1.5,
        0.5,
        None,
        None,
        None,
    )
    req = DiffusionGenerateRequest(prompt = "p", hires = {"scale": 2})
    assert req.hires.scale == 2 and req.hires.denoise == 0.5
    assert DiffusionGenerateRequest(prompt = "p").hires is None


@pytest.mark.parametrize(
    "bad",
    [
        {"scale": 0.9},
        {"scale": 4.5},
        {"denoise": -0.1},
        {"denoise": 1.1},
        {"steps": 0},
        {"steps": 101},
    ],
)
def test_request_bounds(bad):
    from pydantic import ValidationError

    from models.inference import DiffusionGenerateRequest
    with pytest.raises(ValidationError):
        DiffusionGenerateRequest(prompt = "p", hires = bad)


def test_route_refuses_hires_on_an_engine_without_it():
    from models.inference import DiffusionGenerateRequest
    from routes.inference import _hires_kwargs

    def no_hires(*, prompt):
        return None

    def with_hires(
        *,
        prompt,
        upscale_model = None,
        hires = None,
    ):
        return None

    req = DiffusionGenerateRequest(prompt = "p", hires = {"scale": 1.5, "upscaler": "lanczos"})
    with pytest.raises(ValueError, match = "Hires fix is not available"):
        _hires_kwargs(no_hires, req)
    out = _hires_kwargs(with_hires, req)
    assert out["hires"] == hires.HiresParams(scale = 1.5, denoise = 0.5, upscaler = "lanczos")
    assert _hires_kwargs(no_hires, DiffusionGenerateRequest(prompt = "p")) == {}
    with pytest.raises(ValueError, match = "Model upscalers are not available"):
        _hires_kwargs(
            no_hires, DiffusionGenerateRequest(prompt = "p", upscale_model = "realesrgan_x4plus")
        )


def test_engine_signature_advertises_hires_but_not_the_private_pass():
    from core.inference.diffusion import DiffusionBackend
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    params = inspect.signature(DiffusionBackend.generate).parameters
    assert "hires" in params and "upscale_model" in params and "_hires_pass" not in params
    assert "hires" in inspect.signature(SdCppDiffusionBackend.generate).parameters


def test_hires_refuses_conditioned_requests():
    for kwargs in (
        {"init_image": "x"},
        {"mask_image": "x"},
        {"reference_images": ["x"]},
        {"workflow": "edit"},
        {"upscale": 2.0},
        {"upscale_model": "realesrgan_x4plus"},
        {"controlnet": ("id", "img", "canny", 1.0, 0.0, 1.0)},
    ):
        with pytest.raises(ValueError):
            hires._check_request({"prompt": "p", **kwargs})
    hires._check_request({"prompt": "p", "controlnet": ("id", "img", "canny", 0.0, 0.0, 1.0)})


@pytest.mark.parametrize("rule", [hires.RULE_FLOOR, hires.RULE_FLOW])
@pytest.mark.parametrize("steps", [1, 4, 9, 14, 20, 28, 50])
@pytest.mark.parametrize("denoise", [0.05, 0.2, 0.3, 0.35, 0.5, 0.55, 0.7, 0.9, 1.0])
def test_refine_runs_exactly_the_requested_steps(rule, steps, denoise):
    n = hires.refine_inference_steps(steps, denoise, rule)
    assert hires.steps_run(n, denoise, rule) == steps


def test_comfyui_example_numbers():
    # hiresfix_latent_workflow.png: 768 -> 1152, KSampler 14 steps at denoise 0.5 (28-step schedule, last 14).
    assert hires.hires_target_size(768, 768, 1.5) == (1152, 1152)
    assert hires.refine_inference_steps(14, 0.5, hires.RULE_FLOOR) == 28
    assert hires.refine_inference_steps(14, 0.5, hires.RULE_FLOW) == 28


def test_target_size_rounds_to_16_and_caps_the_longest_side():
    assert hires.hires_target_size(832, 1216, 1.5) == (1248, 1824)
    assert hires.hires_target_size(832, 1216, 2.0) == (1408, 2048)
    assert hires.hires_target_size(1024, 1024, 4.0) == (2048, 2048)
    w, h = hires.hires_target_size(1024, 1024, 4.0, max_pixels = 2048 * 1536)
    assert w * h <= 2048 * 1536 and w % 16 == 0 and h % 16 == 0


def test_timestep_rule_reads_each_pipelines_rounding():
    diffusers = pytest.importorskip("diffusers")
    assert (
        hires.timestep_rule(
            diffusers.StableDiffusionXLImg2ImgPipeline.__new__(
                diffusers.StableDiffusionXLImg2ImgPipeline
            )
        )
        == hires.RULE_FLOOR
    )
    for name in ("FluxImg2ImgPipeline", "QwenImageImg2ImgPipeline", "ZImageImg2ImgPipeline"):
        cls = getattr(diffusers, name)
        assert hires.timestep_rule(cls.__new__(cls)) == hires.RULE_FLOW, name


def test_capability_only_where_the_family_has_img2img():
    sdxl = hires.capability(_fam("sdxl"))
    assert sdxl["default_upscaler"] == "latent" and sdxl["defaults"] == {
        "scale": 1.5,
        "denoise": 0.5,
    }
    assert [u["id"] for u in sdxl["upscalers"]][:2] == ["latent", "lanczos"]
    for name in ("flux.1", "qwen-image", "z-image"):
        cap = hires.capability(_fam(name))
        assert cap["default_upscaler"] == "lanczos", name
        assert "latent" not in [u["id"] for u in cap["upscalers"]], name
    for name in ("krea-2", "lumina-2", "flux.2-dev", "hidream-i1"):
        assert hires.capability(_fam(name)) is None, name
        assert "image-to-image" in hires.unavailable_reason(_fam(name))
    assert hires.capability(_fam("qwen-image-edit")) is None
    assert hires.unavailable_reason(_fam("qwen-image-edit")) is None


def test_model_rows_join_the_upscaler_list():
    cap = hires.capability(
        _fam("flux.1"), model_rows = [{"id": "realesrgan_x4plus", "label": "R-ESRGAN 4x+"}]
    )
    assert cap["upscalers"][-1] == {
        "id": "realesrgan_x4plus",
        "label": "R-ESRGAN 4x+",
        "kind": "model",
    }


def test_resolve_params_refuses_latent_off_sdxl_and_unknown_ids():
    assert hires.resolve_params(hires.HiresParams(), "sdxl").upscaler == "latent"
    assert hires.resolve_params(hires.HiresParams(), "flux.1").upscaler == "lanczos"
    with pytest.raises(ValueError, match = "SDXL only"):
        hires.resolve_params(hires.HiresParams(upscaler = "latent"), "flux.1")
    with pytest.raises(ValueError, match = "needs a refine pass"):
        hires.resolve_params(hires.HiresParams(upscaler = "latent", denoise = 0.0), "sdxl")
    with pytest.raises(ValueError, match = "Unknown upscaler"):
        hires.resolve_params(hires.HiresParams(upscaler = "../../etc/passwd"), "sdxl")


@pytest.mark.parametrize(
    "bad", ["local:../x", "local:a/b", "local:", "/abs/path.pth", "local:..", "nope"]
)
def test_upscaler_ids_never_name_a_path(bad):
    with pytest.raises(ValueError):
        ups.check_upscaler_id(bad)


def test_local_drop_in_resolves_only_inside_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ups, "models_dir", lambda: tmp_path)
    (tmp_path / "4x_Remacri.pth").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "RealESRGAN_x4plus.pth").write_bytes(
        b"x"
    )  # a catalog file shows under its catalog id
    assert set(ups._local_files()) == {"4x_Remacri"}
    assert ups.resolve_model_path("local:4x_Remacri") == tmp_path / "4x_Remacri.pth"
    with pytest.raises(ValueError, match = "not in the upscale_models folder"):
        ups.resolve_model_path("local:missing")


def test_catalog_download_is_verified(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(ups, "models_dir", lambda: tmp_path)
    entry = ups.CATALOG[1]
    payload = b"not the weights"

    class _Resp:
        def __init__(self):
            self.chunks = [payload, b""]

        def read(self, n):
            return self.chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ups.urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match = "integrity check"):
        ups.resolve_model_path(entry.id)
    assert not any(tmp_path.iterdir())  # no partial file is left behind
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ValueError, match = "offline"):
        ups.resolve_model_path(entry.id)


def test_catalog_entries_are_pinned():
    for entry in ups.CATALOG:
        assert entry.license == "BSD-3-Clause"
        assert entry.url.startswith("https://github.com/xinntao/Real-ESRGAN/releases/download/")
        assert re.fullmatch(r"[0-9a-f]{64}", entry.sha256) and entry.size > 1_000_000


def _a1111_fields(line):
    # modules/infotext_utils.py re_param
    re_param = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
    return {k: v.strip() for k, v in re_param.findall(line)}


def test_infotext_writes_a1111_hires_keys_and_the_base_size():
    from core.inference import image_gallery

    meta = {
        "prompt": "a castle",
        "steps": 28,
        "guidance": 7.0,
        "seed": 1,
        "width": 1248,
        "height": 1824,
        "model": "m",
        "workflow": "txt2img",
        **hires.record_fields(
            {
                "scale": 1.5,
                "denoise": 0.5,
                "steps": 14,
                "upscaler": "latent",
                "guidance": None,
                "base_width": 832,
                "base_height": 1216,
            }
        ),
    }
    text = image_gallery._params_text(meta)
    fields = _a1111_fields(text.splitlines()[-1])
    assert fields["Size"] == "832x1216"
    assert fields["Denoising strength"] == "0.5"
    assert fields["Hires upscale"] == "1.5"
    assert fields["Hires steps"] == "14"
    assert fields["Hires upscaler"] == "Latent (nearest-exact)"
    assert "Hires CFG Scale" not in fields
    assert meta["width"] == 1248


def test_infotext_omits_hires_steps_when_same_and_names_models_like_a1111():
    meta = {
        "hires_scale": 2.0,
        "hires_denoise": 0.3,
        "hires_steps": None,
        "hires_upscaler": "realesrgan_x4plus",
        "hires_guidance": 4.0,
    }
    text = hires.infotext_fields(meta)
    assert "Hires steps" not in text
    assert "Hires upscaler: R-ESRGAN 4x+" in text
    assert "Hires CFG Scale: 4.0" in text
    assert hires.infotext_fields({"hires_scale": None}) == ""
    assert ups.model_label("local:4x_foolhardy_Remacri") == "4x_foolhardy_Remacri"


def test_gallery_round_trips_hires_fields(tmp_path, monkeypatch):
    from PIL import Image

    from core.inference import image_gallery
    from models.inference import GalleryImage

    monkeypatch.setattr(image_gallery, "gallery_dir", lambda: tmp_path)
    fields = hires.record_fields(
        {
            "scale": 1.5,
            "denoise": 0.5,
            "steps": 14,
            "upscaler": "lanczos",
            "guidance": None,
            "base_width": 256,
            "base_height": 256,
        }
    )
    rec = image_gallery.save(
        Image.new("RGB", (384, 384)),
        {
            "prompt": "p",
            "width": 384,
            "height": 384,
            "steps": 4,
            "guidance": 0,
            "seed": 1,
            "created_at": 1.0,
            "workflow": "txt2img",
            **fields,
        },
    )
    img = GalleryImage(**rec)
    assert (img.hires_scale, img.hires_steps, img.hires_upscaler, img.hires_base_width) == (
        1.5,
        14,
        "lanczos",
        256,
    )
    png = Image.open(tmp_path / f"{rec['id']}.png")
    assert (
        "Hires upscale: 1.5" in png.info["parameters"] and "Size: 256x256" in png.info["parameters"]
    )


@pytest.mark.parametrize(
    "length,tile,overlap",
    [(100, 512, 32), (512, 512, 32), (513, 512, 32), (1500, 512, 32), (2048, 256, 16), (70, 64, 8)],
)
def test_tile_starts_cover_with_the_overlap(length, tile, overlap):
    starts = ups.tile_starts(length, tile, overlap)
    win = min(tile, length)
    assert starts[0] == 0 and starts[-1] + win == length
    covered = set()
    for s in starts:
        covered.update(range(s, s + win))
    assert covered == set(range(length))
    for a, b in zip(starts, starts[1:]):
        assert a + win - b >= overlap  # neighbours share at least the overlap


def test_tiled_apply_matches_untiled_for_a_local_operator():
    torch = pytest.importorskip("torch")

    def up2(x):
        return torch.nn.functional.interpolate(x, scale_factor = 2, mode = "nearest")

    x = torch.rand(1, 3, 150, 97)
    full = up2(x)
    tiled = ups.tiled_apply(x, up2, 2, tile = 64, overlap = 8)
    assert tiled.shape == full.shape
    assert torch.allclose(tiled, full, atol = 1e-6)
    assert len(ups.tile_boxes(97, 150, 64, 8)) > 1


def test_tiled_esrgan_matches_untiled_on_a_tiny_random_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("spandrel")
    from PIL import Image
    from spandrel.architectures.ESRGAN import ESRGAN

    torch.manual_seed(0)
    net = ESRGAN(in_nc = 3, out_nc = 3, num_filters = 8, num_blocks = 1, scale = 4)
    path = tmp_path / "tiny_esrgan.pth"
    torch.save(net.state_dict(), path)
    monkeypatch.setattr(ups, "models_dir", lambda: tmp_path)
    img = Image.fromarray((torch.rand(96, 80, 3) * 255).to(torch.uint8).numpy())
    untiled = ups.upscale_with_model(img, "local:tiny_esrgan", (320, 384), device = "cpu", tile = 512)
    tiled = ups.upscale_with_model(
        img, "local:tiny_esrgan", (320, 384), device = "cpu", tile = 48, overlap = 16
    )
    assert untiled.tiles == 1 and tiled.tiles > 1
    assert untiled.image.size == tiled.image.size == (320, 384)
    import numpy as np

    a = np.asarray(untiled.image, dtype = np.int16)
    b = np.asarray(tiled.image, dtype = np.int16)
    # The receptive field crosses the seams, so feathered blending is close, not exact.
    assert np.abs(a - b).mean() < 2.0
    assert untiled.released and not ups._resident  # freed after use when not asked to stay


def test_native_request_carries_the_hires_block():
    from core.inference.sd_cpp_args import build_img_gen_request

    block = hires.sd_cpp_payload(
        hires.HiresParams(scale = 1.5, denoise = 0.5, steps = None, upscaler = "latent"),
        20,
        (1248, 1824),
        hires.sd_cpp_upscaler_name("latent"),
    )
    req = build_img_gen_request(prompt = "p", width = 832, height = 1216, steps = 20, hires = block)
    assert req["hires"] == {
        "enabled": True,
        "upscaler": "Latent (nearest-exact)",
        "scale": 1.5,
        "target_width": 1248,
        "target_height": 1824,
        "steps": 20,
        "denoising_strength": 0.5,
    }
    assert "hires" not in build_img_gen_request(prompt = "p")
    assert hires.sd_cpp_upscaler_name("lanczos") == "Lanczos"
    assert hires.sd_cpp_upscaler_name("realesrgan_x4plus", "hires_x") == "hires_x"


def test_native_capability_needs_the_server_and_a_build_with_hires(monkeypatch):
    monkeypatch.setattr(hires, "sd_cpp_supports_hires", lambda binary: binary == "new")
    cap, reason = hires.sd_cpp_capability(_fam("sdxl"), mode = "server", binary = "new")
    assert cap is not None and reason is None
    assert hires.sd_cpp_capability(_fam("sdxl"), mode = "oneshot", binary = "new")[0] is None
    cap, reason = hires.sd_cpp_capability(_fam("sdxl"), mode = "server", binary = "old")
    assert cap is None and "no Hires fix support" in reason
    assert hires.sd_cpp_capability(_fam("krea-2"), mode = "server", binary = "new")[0] is None
    with pytest.raises(ValueError, match = "denoise above 0"):
        hires.plan_sd_cpp(
            _fam("sdxl"),
            hires.HiresParams(denoise = 0.0, upscaler = "lanczos"),
            mode = "server",
            binary = "new",
            width = 1024,
            height = 1024,
            steps = 20,
        )
    with pytest.raises(ValueError, match = "Hires CFG"):
        hires.plan_sd_cpp(
            _fam("sdxl"),
            hires.HiresParams(guidance = 3.0),
            mode = "server",
            binary = "new",
            width = 1024,
            height = 1024,
            steps = 20,
        )
    plan = hires.plan_sd_cpp(
        _fam("flux.1"),
        hires.HiresParams(steps = 6),
        mode = "server",
        binary = "new",
        width = 832,
        height = 1216,
        steps = 20,
    )
    assert (
        plan.target == (1248, 1824) and plan.refine_steps == 6 and plan.params.upscaler == "lanczos"
    )


def test_help_probe_fails_closed(monkeypatch, tmp_path):
    import core.inference.sd_cpp_backend as scb

    binary = tmp_path / "sd-server"
    binary.write_text("")
    monkeypatch.setattr(
        scb, "_sd_cpp_probe_output", lambda b, *a: "  --hires-denoising-strength  x"
    )
    hires._sd_cpp_help_cache.clear()
    assert hires.sd_cpp_supports_hires(str(binary)) is True
    hires._sd_cpp_help_cache.clear()
    monkeypatch.setattr(scb, "_sd_cpp_probe_output", lambda b, *a: None)
    assert hires.sd_cpp_supports_hires(str(binary)) is False
    assert hires.sd_cpp_supports_hires(None) is False


@pytest.fixture(scope = "module")
def tiny_sdxl():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny pipeline not reachable: {exc}")
    return diffusers.StableDiffusionXLPipeline.from_pretrained(path, torch_dtype = torch.float32)


def _backend(pipe):
    from core.inference.diffusion import DiffusionBackend, _LoadState

    b = DiffusionBackend()
    b._state = _LoadState(
        pipe = pipe,
        family = _fam("sdxl"),
        repo_id = "tiny",
        base_repo = "tiny",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )
    return b


def test_tiny_sdxl_hires_sequencing(tiny_sdxl, monkeypatch):
    import numpy as np
    import diffusers

    calls = []
    orig = diffusers.StableDiffusionXLImg2ImgPipeline.__call__

    import functools

    @functools.wraps(orig)  # keeps the signature the engine gates its kwargs on
    def spy(self, *a, **k):
        img = k.get("image")
        calls.append(
            (
                k.get("num_inference_steps"),
                k.get("strength"),
                tuple(getattr(img, "shape", ())) or img.size,
            )
        )
        steps_seen = []
        cb = k.get("callback_on_step_end")

        def counting(pipe, i, t, kw):
            steps_seen.append(i)
            return cb(pipe, i, t, kw) if cb else kw

        k["callback_on_step_end"] = counting
        out = orig(self, *a, **k)
        calls[-1] = calls[-1] + (len(steps_seen),)
        return out

    monkeypatch.setattr(diffusers.StableDiffusionXLImg2ImgPipeline, "__call__", spy)
    b = _backend(tiny_sdxl)
    kw = dict(
        prompt = "a cat", width = 256, height = 256, steps = 4, guidance = 5.0, seed = 7, live_preview = False
    )
    plain = b.generate(**kw)
    again = b.generate(**kw, hires = None)
    # hires=None is the unchanged path: same seed, same pixels, no img2img call.
    assert np.array_equal(np.asarray(plain["images"][0]), np.asarray(again["images"][0]))
    assert calls == [] and "hires" not in plain

    out = b.generate(
        **kw, hires = hires.HiresParams(scale = 1.5, denoise = 0.5, steps = 3, upscaler = "latent")
    )
    assert out["images"][0].size == (384, 384)
    assert out["workflow"] == "txt2img"
    assert out["hires"]["refine_steps"] == 3 and out["hires"]["base_width"] == 256
    # One refine pass on 4-channel latents: 3 steps of a 6-step schedule at strength 0.5.
    f = tiny_sdxl.vae_scale_factor
    assert calls == [(6, 0.5, (1, 4, 384 // f, 384 // f), 3)]

    calls.clear()
    out = b.generate(
        **kw, batch_size = 2, hires = hires.HiresParams(scale = 1.5, denoise = 0.35, upscaler = "lanczos")
    )
    assert [im.size for im in out["images"]] == [(384, 384)] * 2
    assert len(calls) == 2 and all(c[1] == 0.35 and c[2] == (384, 384) and c[3] == 4 for c in calls)
    assert out["seeds"] == plain["seeds"][:1] + [plain["seeds"][0] + 1]

    calls.clear()
    out = b.generate(**kw, hires = hires.HiresParams(scale = 2.0, denoise = 0.0, upscaler = "lanczos"))
    assert out["images"][0].size == (512, 512) and calls == [] and out["hires"]["refine_steps"] == 0
    from PIL import Image

    expected = plain["images"][0].resize((512, 512), Image.LANCZOS)
    assert np.array_equal(np.asarray(out["images"][0]), np.asarray(expected))
    assert b._gen is None and b._active_generate_cancel is None


def test_tiny_sdxl_hires_refusals(tiny_sdxl):
    b = _backend(tiny_sdxl)
    with pytest.raises(ValueError, match = "text to image only"):
        b.generate(
            prompt = "p", width = 256, height = 256, steps = 2, init_image = "x", hires = hires.HiresParams()
        )
    with pytest.raises(ValueError, match = "Unknown upscaler"):
        b.generate(
            prompt = "p", width = 256, height = 256, steps = 2, hires = hires.HiresParams(upscaler = "x")
        )
    with pytest.raises(ValueError, match = "SDXL only"):
        hires.resolve_params(hires.HiresParams(upscaler = "latent"), "z-image")


def test_preflight_refuses_before_the_base_pass(tiny_sdxl, monkeypatch):
    import core.inference.diffusion_memory as dm

    b = _backend(tiny_sdxl)
    real = dm.raise_on_image_activation_shortfall
    seen = []

    def guard(**kw):
        seen.append((kw["width"], kw["height"]))
        if kw["width"] > 256:
            raise dm.ImageActivationShortfallError("needs more memory")
        return real(**kw)

    monkeypatch.setattr(dm, "raise_on_image_activation_shortfall", guard)
    monkeypatch.setattr("core.inference.diffusion.raise_on_image_activation_shortfall", guard)
    ran = []
    orig = type(tiny_sdxl).__call__
    monkeypatch.setattr(
        type(tiny_sdxl), "__call__", lambda self, *a, **k: ran.append(1) or orig(self, *a, **k)
    )
    with pytest.raises(
        dm.ImageActivationShortfallError, match = "refine pass at 384x384 does not fit"
    ):
        b.generate(
            prompt = "p",
            width = 256,
            height = 256,
            steps = 2,
            seed = 1,
            live_preview = False,
            hires = hires.HiresParams(upscaler = "lanczos"),
        )
    assert ran == []  # refused before the base pass denoised
    assert (384, 384) in seen


def test_run_hires_pass_kwargs_without_a_model(monkeypatch):
    """Sequencing on a stand-in engine: per-image seeds, no variation seed, cancel re-armed, progress cleared."""
    import contextlib
    import threading

    from PIL import Image

    from core.inference.diffusion_sd_controls import SdControls

    calls = []

    class _Backend:
        def __init__(self):
            self._state = types.SimpleNamespace(
                family = _fam("z-image"), offload_policy = "none", device = "cpu"
            )
            self._lock = threading.Lock()
            self._generation_cancel_lock = threading.Lock()
            self._gen = None
            self._active_generate_cancel = None
            self._active_generate_account = None

        @contextlib.contextmanager
        def _generation_slot(self, cancel):
            yield

    def fake_generate(backend, **kw):
        ctx = kw.pop("_hires_pass")
        calls.append(
            (ctx.refine, ctx.defer_bg_kick, backend._active_generate_cancel is ctx.cancel, kw)
        )
        if not ctx.refine:
            return {
                "images": [Image.new("RGB", (256, 256))] * 2,
                "seed": 5,
                "seeds": [5, 6],
                "workflow": "txt2img",
            }
        assert ctx.source.size == (384, 384)
        return {"images": [ctx.source], "seed": kw["seed"], "seeds": [kw["seed"]]}

    b = _Backend()
    out = hires.run_hires(
        b,
        fake_generate,
        hires.HiresParams(denoise = 0.4, steps = 6, guidance = 2.5),
        {
            "prompt": "p",
            "prompts": ["a", "b"],
            "width": 256,
            "height": 256,
            "steps": 8,
            "guidance": 0.0,
            "seed": 5,
            "batch_size": 1,
            "sd_controls": SdControls(variation_seed = 9, variation_strength = 0.3, clip_skip = 2),
        },
    )
    assert [c[0] for c in calls] == [False, True, True]
    assert calls[0][1] is True  # the base pass defers a pending background compile
    assert calls[1][2] and calls[2][2]  # Stop is registered again for each refine pass
    refine = [c[3] for c in calls[1:]]
    assert [k["prompt"] for k in refine] == ["a", "b"] and [k["seed"] for k in refine] == [5, 6]
    assert all(k["steps"] == 6 and k["strength"] == 0.4 and k["guidance"] == 2.5 for k in refine)
    assert all("prompts" not in k and "seeds" not in k and k["batch_size"] == 1 for k in refine)
    assert all(
        k["sd_controls"].variation_strength is None and k["sd_controls"].clip_skip == 2
        for k in refine
    )
    assert calls[0][3]["sd_controls"].variation_strength == 0.3  # the base pass keeps it
    assert out["hires"]["refine_steps"] == 6 and out["hires"]["upscaler"] == "lanczos"
    assert [im.size for im in out["images"]] == [(384, 384)] * 2
    assert b._gen is None
