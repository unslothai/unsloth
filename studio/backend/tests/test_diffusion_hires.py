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


def test_denoise_below_the_a1111_comfyui_step_is_refused():
    # A1111 "Denoising strength" and ComfyUI KSampler denoise step by 0.01; below it steps / denoise is unbounded.
    for bad in (0.00001, 1e-300, 0.009):
        with pytest.raises(ValueError, match = "at least 0.01"):
            hires.resolve_params(hires.HiresParams(denoise = bad, upscaler = "lanczos"), "flux.1")
    assert hires.resolve_params(hires.HiresParams(denoise = 0.0, upscaler = "lanczos"), "flux.1")
    for rule in (hires.RULE_FLOOR, hires.RULE_FLOW):
        n = hires.refine_inference_steps(100, hires.MIN_DENOISE, rule)
        assert n <= 10004 and hires.steps_run(n, hires.MIN_DENOISE, rule) == 100


def test_upscaler_stop_raises_the_canonical_cancellation():
    import threading

    import torch

    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError) as exc:
        ups.tiled_apply(
            torch.zeros(1, 3, 64, 64), lambda x: x, 1, tile = 32, overlap = 8, cancel = cancel
        )
    assert str(exc.value) == DIFFUSION_CANCELLED_MSG


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
    import threading

    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    stop = threading.Event()
    stop.set()
    with pytest.raises(RuntimeError) as exc:
        ups.resolve_model_path(entry.id, cancel = stop)
    assert str(exc.value) == DIFFUSION_CANCELLED_MSG
    assert not any(tmp_path.iterdir())
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
    assert "Hires resize" not in fields
    assert meta["width"] == 1248


def test_infotext_writes_a_capped_target_as_a1111_hires_resize():
    from core.inference import image_gallery

    # 1536x1024 at 2x is capped to 2048 px: the factor alone would describe 3072x2048.
    target = hires.hires_target_size(1536, 1024, 2.0)
    assert target == (2048, 1360)
    meta = {
        "prompt": "p",
        "steps": 20,
        "seed": 1,
        "width": target[0],
        "height": target[1],
        "model": "m",
        "workflow": "txt2img",
        **hires.record_fields(
            {
                "scale": 2.0,
                "denoise": 0.5,
                "upscaler": "lanczos",
                "base_width": 1536,
                "base_height": 1024,
            }
        ),
    }
    fields = _a1111_fields(image_gallery._params_text(meta).splitlines()[-1])
    assert fields["Size"] == "1536x1024"
    assert fields["Hires resize"] == "2048x1360"
    assert "Hires upscale" not in fields


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
    assert ups.tiled_apply(x, up2, 2, tile = 64, overlap = 8, output_device = "cpu").device.type == "cpu"


@pytest.mark.parametrize("tile", [512, 64])
def test_stop_during_the_last_upscaler_tile_is_honoured(tile):
    import threading

    torch = pytest.importorskip("torch")
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    cancel = threading.Event()
    calls = []

    def up_then_stop(x):
        calls.append(1)
        if len(calls) == len(ups.tile_boxes(97, 150, tile, 8)):
            cancel.set()  # Stop lands while the final (or only) tile runs
        return torch.nn.functional.interpolate(x, scale_factor = 2, mode = "nearest")

    with pytest.raises(RuntimeError) as exc:
        ups.tiled_apply(
            torch.rand(1, 3, 150, 97), up_then_stop, 2, tile = tile, overlap = 8, cancel = cancel
        )
    assert str(exc.value) == DIFFUSION_CANCELLED_MSG


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
    seen_devices = []
    real_tiled_apply = ups.tiled_apply

    def spy(*a, **k):
        seen_devices.append(k.get("output_device"))
        return real_tiled_apply(*a, **k)

    monkeypatch.setattr(ups, "tiled_apply", spy)
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
    assert seen_devices == ["cpu", "cpu"]  # the full-scale result accumulates in host memory


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
    # The control hides Hires CFG and upscale-only, which plan_sd_cpp refuses.
    assert cap["supports_guidance"] is False and cap["min_denoise"] == hires.MIN_DENOISE
    diffusers_cap = hires.capability(_fam("sdxl"))
    assert diffusers_cap["supports_guidance"] is True and diffusers_cap["min_denoise"] == 0.0
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
            totals.append(getattr(holder[0]._gen, "total_steps", None))
            return cb(pipe, i, t, kw) if cb else kw

        k["callback_on_step_end"] = counting
        out = orig(self, *a, **k)
        calls[-1] = calls[-1] + (len(steps_seen),)
        return out

    monkeypatch.setattr(diffusers.StableDiffusionXLImg2ImgPipeline, "__call__", spy)
    b = _backend(tiny_sdxl)
    holder, totals = [b], []
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
    # Progress counts the 3 steps the refine runs, not the 6-step schedule it enters halfway.
    assert totals == [3, 3, 3]

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
                "negative_prompt": None,  # base guidance 0 leaves the negative off
            }
        assert ctx.source.size == (384, 384)
        return {
            "images": [ctx.source],
            "seed": kw["seed"],
            "seeds": [kw["seed"]],
            "negative_prompt": kw.get("negative_prompt"),
        }

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
            "negative_prompt": "blurry",
            "sd_controls": SdControls(variation_seed = 9, variation_strength = 0.3, clip_skip = 2),
        },
    )
    assert [c[0] for c in calls] == [False, True, True]
    assert calls[0][1] is True  # the base pass defers a pending background compile
    # ...and so does every refine but the last, or image two waits on the compile image one kicked.
    assert [c[1] for c in calls[1:]] == [True, False]
    assert out["negative_prompt"] == "blurry"  # Hires CFG engaged it, so the recipe records it
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


def test_run_hires_plans_against_the_model_loaded_when_the_slot_is_held():
    """A load that wins the slot first swaps the family; the plan must follow it (FLUX -> SDXL: latent default)."""
    import contextlib
    import threading

    from PIL import Image

    seen = []

    class _Backend:
        def __init__(self):
            self._state = types.SimpleNamespace(
                family = _fam("flux.1"), offload_policy = "none", device = "cpu"
            )
            self._lock = threading.Lock()
            self._generation_cancel_lock = threading.Lock()
            self._gen = None
            self._active_generate_cancel = None
            self._active_generate_account = None

        @contextlib.contextmanager
        def _generation_slot(self, cancel):
            self._state = types.SimpleNamespace(
                family = _fam("sdxl"), offload_policy = "none", device = "cpu"
            )
            yield

    def fake_generate(backend, **kw):
        ctx = kw.pop("_hires_pass")
        seen.append(ctx)
        if not ctx.refine:
            return {"images": [Image.new("RGB", (256, 256))], "seed": 1, "seeds": [1]}
        return {"images": [Image.new("RGB", ctx.source.size)], "seed": 1, "seeds": [1]}

    out = hires.run_hires(
        _Backend(),
        fake_generate,
        hires.HiresParams(),
        {"prompt": "p", "width": 256, "height": 256, "steps": 8, "seed": 1, "batch_size": 1},
    )
    assert out["hires"]["upscaler"] == "latent"
    # The latent upscaler asks the base pass for latents; a refine never decodes for it.
    assert seen[0].latent_output is True and seen[1].latent_output is False
    assert seen[1].latent_from is not None


def test_native_cpu_restart_restages_the_model_upscaler(tmp_path, monkeypatch):
    import dataclasses
    import io
    import threading

    from PIL import Image

    import core.inference.sd_cpp_backend as scb

    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format = "PNG")
    png = buf.getvalue()
    payloads = []

    class _Server:
        def __init__(self, lora_dir, fail):
            self.lora_dir, self.fail = str(lora_dir), fail

        def img_gen(self, payload, **kw):
            payloads.append((self.lora_dir, dict(payload["hires"])))
            if self.fail:
                raise RuntimeError("ggml unsupported op abort")
            return [png]

    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    staged_paths = []

    def fake_stage(
        upscaler_id,
        scratch_dir,
        *,
        cancel = None,
    ):
        dest = tmp_path / scratch_dir / "hires_x.pth"
        dest.write_bytes(b"w")
        staged_paths.append(dest)
        return "hires_x", dest

    monkeypatch.setattr(hires, "stage_sd_cpp_upscaler", fake_stage)

    @dataclasses.dataclass(frozen = True)
    class _State:
        family: object
        server: object
        sampling_method: object = None
        flow_shift: object = None
        sample_sigmas: object = None

    b = scb.SdCppDiffusionBackend()
    new_server = _Server(new_dir, fail = False)
    monkeypatch.setattr(b, "_restart_server_on_cpu_backend", lambda state, text, cancel: new_server)
    plan = hires.SdCppHiresPlan(
        params = hires.HiresParams(upscaler = "realesrgan_x4plus"),
        base_width = 64,
        base_height = 64,
        target = (96, 96),
        refine_steps = 4,
    )
    state = _State(family = _fam("sdxl"), server = _Server(old_dir, fail = True))
    # The replacement is published only while the backend still holds the server it replaces.
    b._state = state
    images, _ = b._generate_server(
        state,
        prompt = "p",
        negative_prompt = None,
        width = 64,
        height = 64,
        steps = 4,
        seed = 1,
        batch_size = 1,
        cfg_scale = None,
        flux_guidance = None,
        lora_resolved = [],
        cancel = threading.Event(),
        hires_plan = plan,
    )
    assert len(images) == 1
    assert [d for d, _ in payloads] == [str(old_dir), str(new_dir)]
    assert all(h["upscaler"] == "hires_x" for _, h in payloads)
    assert [p.parent for p in staged_paths] == [
        old_dir,
        new_dir,
    ]  # staged again in the new server's dir
    assert not any(p.exists() for p in staged_paths)  # both removed after
    assert b._state.server is new_server


def test_diffusers_status_names_why_hires_is_hidden():
    from core.inference.diffusion import _hires_status

    assert "image-to-image" in _hires_status(_fam("krea-2"))["hires_unavailable_reason"]
    shown = _hires_status(_fam("sdxl"))
    assert shown["hires"] is not None and shown["hires_unavailable_reason"] is None


def test_native_hires_progress_follows_each_sampling_pass():
    import core.inference.sd_cpp_backend as scb

    # sd.cpp make_hires_sigma_schedule: int(14 / 0.5) = 28 -> 14; int(20 / 0.35) = 57 -> int(19.95) = 19.
    assert hires.sd_cpp_hires_sampled_steps(14, 0.5) == 14
    assert hires.sd_cpp_hires_sampled_steps(20, 0.35) == 19
    assert hires.sd_cpp_hires_sampled_steps(5, 1.0) == 4

    b = scb.SdCppDiffusionBackend()
    for base, refine in ((20, 14), (20, 20)):
        b._gen = scb._SdGen(total_steps = base + refine, passes = (base, refine))
        seen = []
        for line in (
            [f"|====| {i}/{base} - 1.0s/it" for i in range(1, base + 1)]
            + [f"|====| {base}/{base} - 1.0s/it"]  # a repeated last line is not the next pass
            + [f"|====| {i}/{refine} - 1.0s/it" for i in range(1, refine + 1)]
        ):
            b._on_log(line)
            seen.append(b._gen.step)
        assert seen[base - 1] == base and seen[base] == base
        assert seen[base + 1] == base + 1 and seen[-1] == base + refine
        assert b.generate_progress()["fraction"] == 1.0
        b._on_log(f"|=| 1/{base} - 1.0s/it")  # a batch's next image starts its base pass again
        assert b._gen.step == 1
    b._gen = scb._SdGen(total_steps = 8)
    b._on_log("|=| 3/8 - 1.0s/it")
    assert b._gen.step == 3  # a plain render is unchanged


def test_mlx_engine_takes_the_hires_keyword_and_refuses_it():
    from core.inference.mlx_diffusion_backend import MlxDiffusionBackend
    with pytest.raises(ValueError, match = "not available on the MLX engine"):
        MlxDiffusionBackend.generate(types.SimpleNamespace(), prompt = "p", hires = hires.HiresParams())


def test_native_hires_sends_the_steps_that_sample_the_requested_count(monkeypatch):
    # sd.cpp samples int(int(n / s) * s) (capped at n / s - 1): 5 at 1.0 needs 6 sent; 20 at 0.35 has no exact
    # count (20 -> 19, 21 -> 21), so the request is sent and the sampled 19 is what the plan reports.
    assert hires.sd_cpp_hires_request_steps(5, 1.0) == 6
    assert hires.sd_cpp_hires_request_steps(14, 0.5) == 14
    assert hires.sd_cpp_hires_request_steps(20, 0.35) == 20
    for steps, denoise in ((5, 1.0), (14, 0.5), (8, 0.3), (30, 0.7), (1, 0.5)):
        sent = hires.sd_cpp_hires_request_steps(steps, denoise)
        assert hires.sd_cpp_hires_sampled_steps(sent, denoise) == steps, (steps, denoise)
    monkeypatch.setattr(hires, "sd_cpp_supports_hires", lambda binary: True)
    plan = hires.plan_sd_cpp(
        _fam("sdxl"),
        hires.HiresParams(denoise = 1.0, steps = 5, upscaler = "lanczos"),
        mode = "server",
        binary = "new",
        width = 512,
        height = 512,
        steps = 20,
    )
    assert (plan.request_steps, plan.refine_steps) == (6, 5)
    block = hires.sd_cpp_payload(
        plan.params, 20, plan.target, "Lanczos", request_steps = plan.request_steps
    )
    assert block["steps"] == 6
    assert hires.result_block(plan)["refine_steps"] == 5 and hires.result_block(plan)["steps"] == 5


@pytest.mark.parametrize("env", ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"])
def test_catalog_download_honours_every_offline_switch(tmp_path, monkeypatch, env):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv(env, "1")
    monkeypatch.setattr(ups, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(
        ups.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used while offline")
    )
    with pytest.raises(ValueError, match = "offline"):
        ups.resolve_model_path(ups.CATALOG[0].id)


def test_upscale_only_with_a_model_upscaler_defers_the_compile_until_it_ran(monkeypatch):
    import contextlib
    import threading

    from PIL import Image

    order = []

    class _Bg:
        def kick(self):
            order.append("kick")
            return True

    class _Backend:
        def __init__(self):
            self._state = types.SimpleNamespace(
                family = _fam("flux.1"), offload_policy = "none", device = "cpu", bg_compile = _Bg()
            )
            self._lock = threading.Lock()
            self._generation_cancel_lock = threading.Lock()
            self._gen = None
            self._active_generate_cancel = None
            self._active_generate_account = None

        @contextlib.contextmanager
        def _generation_slot(self, cancel):
            yield

    def fake_upscale(image, upscaler_id, target, **kw):
        order.append("upscale")
        return ups.UpscaleOutcome(Image.new("RGB", target), 4, 1, "float32", 0.0, True)

    monkeypatch.setattr(ups, "spandrel_available", lambda: True)
    monkeypatch.setattr(ups, "upscale_with_model", fake_upscale)
    seen = []

    def fake_generate(backend, **kw):
        ctx = kw.pop("_hires_pass")
        seen.append(ctx.defer_bg_kick)
        return {"images": [Image.new("RGB", (256, 256))] * 2, "seed": 1, "seeds": [1, 2]}

    hires.run_hires(
        _Backend(),
        fake_generate,
        hires.HiresParams(denoise = 0.0, upscaler = "realesrgan_x4plus"),
        {"prompt": "p", "width": 256, "height": 256, "steps": 4, "seed": 1, "batch_size": 2},
    )
    assert seen == [True]  # the base pass leaves the compile pending
    assert order == ["upscale", "upscale", "kick"]  # started only once the GPU upscales finished
    order.clear()
    hires.run_hires(
        _Backend(),
        fake_generate,
        hires.HiresParams(denoise = 0.0, upscaler = "lanczos"),
        {"prompt": "p", "width": 256, "height": 256, "steps": 4, "seed": 1, "batch_size": 1},
    )
    assert (
        seen[-1] is False and order == []
    )  # a CPU Lanczos upscale lets the base pass kick as before


def test_stop_during_an_upscale_only_resize_returns_no_image(monkeypatch):
    import contextlib
    import threading

    from PIL import Image

    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    class _Backend:
        def __init__(self):
            self._state = types.SimpleNamespace(
                family = _fam("flux.1"), offload_policy = "none", device = "cpu"
            )
            self._lock = threading.Lock()
            self._generation_cancel_lock = threading.Lock()
            self._gen = None
            self._active_generate_cancel = None
            self._active_generate_account = None

        @contextlib.contextmanager
        def _generation_slot(self, cancel):
            yield

    def stopped_upscale(image, upscaler_id, target, **kw):
        kw["cancel"].set()  # Stop accepted while the model's output is resized
        return ups.UpscaleOutcome(Image.new("RGB", target), 4, 1, "float32", 0.0, True)

    monkeypatch.setattr(ups, "spandrel_available", lambda: True)
    monkeypatch.setattr(ups, "upscale_with_model", stopped_upscale)

    def fake_generate(backend, **kw):
        kw.pop("_hires_pass")
        return {"images": [Image.new("RGB", (256, 256))], "seed": 1, "seeds": [1]}

    with pytest.raises(RuntimeError) as exc:
        hires.run_hires(
            _Backend(),
            fake_generate,
            hires.HiresParams(denoise = 0.0, upscaler = "realesrgan_x4plus"),
            {"prompt": "p", "width": 256, "height": 256, "steps": 4, "seed": 1, "batch_size": 1},
        )
    assert str(exc.value) == DIFFUSION_CANCELLED_MSG


def test_a_drop_in_runs_through_the_spandrel_descriptor(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("spandrel")
    from PIL import Image
    from spandrel import ModelLoader
    from spandrel.architectures.MMRealSR import MMRealSR

    torch.manual_seed(0)
    net = MMRealSR(
        num_in_ch = 3,
        num_out_ch = 3,
        scale = 4,
        num_feat = 8,
        num_block = 1,
        num_grow_ch = 4,
        num_feats = [8, 8, 8, 8],
        num_blocks = [1, 1, 1, 1],
    )
    path = tmp_path / "tiny_mmrealsr.pth"
    torch.save(net.state_dict(), path)
    descriptor = ModelLoader(device = "cpu").load_from_file(str(path))
    assert type(descriptor.architecture).__name__ == "MMRealSRArch"
    probe = torch.rand(1, 3, 32, 32)
    with torch.inference_mode():
        assert isinstance(descriptor.model(probe), tuple)
        want = descriptor(probe.clone())
    monkeypatch.setattr(ups, "models_dir", lambda: tmp_path)
    img = Image.fromarray((torch.rand(48, 40, 3) * 255).to(torch.uint8).numpy())
    untiled = ups.upscale_with_model(img, "local:tiny_mmrealsr", (160, 192), device = "cpu", tile = 512)
    tiled = ups.upscale_with_model(
        img, "local:tiny_mmrealsr", (160, 192), device = "cpu", tile = 32, overlap = 8
    )
    assert untiled.model_scale == 4 and untiled.tiles == 1 and tiled.tiles > 1
    assert untiled.image.size == tiled.image.size == (160, 192)
    assert want.shape == (1, 3, 128, 128)


def test_a_resident_drop_in_reloads_when_its_file_is_replaced(tmp_path, monkeypatch):
    import os

    torch = pytest.importorskip("torch")
    pytest.importorskip("spandrel")
    import numpy as np
    from PIL import Image
    from spandrel.architectures.ESRGAN import ESRGAN

    monkeypatch.setattr(ups, "models_dir", lambda: tmp_path)
    path = tmp_path / "tiny_esrgan.pth"
    img = Image.fromarray((torch.rand(32, 32, 3) * 255).to(torch.uint8).numpy())

    def write(seed):
        torch.manual_seed(seed)
        torch.save(
            ESRGAN(in_nc = 3, out_nc = 3, num_filters = 8, num_blocks = 1, scale = 4).state_dict(), path
        )

    def run(keep):
        out = ups.upscale_with_model(
            img, "local:tiny_esrgan", (128, 128), device = "cpu", keep_resident = keep
        )
        return np.asarray(out.image)

    try:
        write(0)
        first = run(True)
        st = path.stat()
        write(1)
        os.utime(
            path, ns = (st.st_atime_ns, st.st_mtime_ns + 1_000_000_000)
        )  # a distinct mtime on any fs
        replaced = run(True)
        assert not np.array_equal(first, replaced)
        assert np.array_equal(replaced, run(False))  # the resident copy is the new file's weights
    finally:
        ups.release_models()


def _fake_model_upscaler(tmp_path, monkeypatch, fn):
    path = tmp_path / "fake.pth"
    path.write_bytes(b"w")
    import torch

    monkeypatch.setattr(ups, "spandrel_available", lambda: True)
    monkeypatch.setattr(ups, "resolve_model_path", lambda upscaler_id, cancel = None: path)

    class _Descriptor:
        scale = 2

        def __call__(self, image):
            return fn(image)

    monkeypatch.setattr(ups, "_load", lambda p, dev: (_Descriptor(), torch.float32))


def test_model_upscaler_halves_the_tile_on_oom_like_comfyui(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from PIL import Image

    sizes = []

    def needs_small_tiles(piece):
        sizes.append(piece.shape[-1])
        if piece.shape[-1] > 128:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return torch.nn.functional.interpolate(piece, scale_factor = 2, mode = "nearest")

    _fake_model_upscaler(tmp_path, monkeypatch, needs_small_tiles)
    img = Image.new("RGB", (300, 300), (10, 20, 30))
    out = ups.upscale_with_model(img, "local:fake", (600, 600), device = "cpu")
    # 512 (one 300 px tile) and 256 run out of memory; 128 px tiles fit.
    assert sizes[:2] == [300, 256] and set(sizes[2:]) == {128}
    assert out.tiles == len(ups.tile_boxes(300, 300, 128, ups.DEFAULT_OVERLAP))
    assert out.image.size == (600, 600) and out.image.getpixel((5, 5)) == (10, 20, 30)

    sizes.clear()

    def always_oom(piece):
        sizes.append(piece.shape[-1])
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    _fake_model_upscaler(tmp_path, monkeypatch, always_oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        ups.upscale_with_model(img, "local:fake", (600, 600), device = "cpu")
    assert sizes == [300, 256, 128]  # gives up below ComfyUI's 128 px floor

    calls = []

    def broken(piece):
        calls.append(1)
        raise ValueError("bad weights")

    _fake_model_upscaler(tmp_path, monkeypatch, broken)
    with pytest.raises(ValueError):
        ups.upscale_with_model(img, "local:fake", (600, 600), device = "cpu")
    assert calls == [1]  # only an OOM retries


def test_download_rechecks_stop_between_short_reads(tmp_path, monkeypatch):
    import threading

    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(ups, "models_dir", lambda: tmp_path)
    stop = threading.Event()
    reads = []

    class _Trickle:
        def read(self, n):
            raise AssertionError("read(n) blocks until n bytes arrive")

        def read1(self, n):
            reads.append(n)
            stop.set()  # Stop pressed while the server trickles
            return b"x"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ups.urllib.request, "urlopen", lambda *a, **k: _Trickle())
    with pytest.raises(RuntimeError) as exc:
        ups.resolve_model_path(ups.CATALOG[0].id, cancel = stop)
    assert str(exc.value) == DIFFUSION_CANCELLED_MSG
    assert len(reads) == 1 and reads[0] <= 1 << 16
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("fail_in", ["refine", "upscale"])
def test_a_failed_pass_after_the_base_still_kicks_the_deferred_compile(monkeypatch, fail_in):
    import contextlib
    import threading

    from PIL import Image

    kicks = []

    class _Backend:
        def __init__(self):
            self._state = types.SimpleNamespace(
                family = _fam("flux.1"),
                offload_policy = "none",
                device = "cpu",
                bg_compile = types.SimpleNamespace(kick = lambda: kicks.append(1)),
            )
            self._lock = threading.Lock()
            self._generation_cancel_lock = threading.Lock()
            self._gen = None
            self._active_generate_cancel = None
            self._active_generate_account = None

        @contextlib.contextmanager
        def _generation_slot(self, cancel):
            yield

    def broken_upscale(*a, **k):
        raise RuntimeError("upscale failed")

    monkeypatch.setattr(ups, "spandrel_available", lambda: True)
    monkeypatch.setattr(ups, "upscale_with_model", broken_upscale)

    def fake_generate(backend, **kw):
        ctx = kw.pop("_hires_pass")
        if ctx.refine:
            raise RuntimeError("refine failed")
        return {"images": [Image.new("RGB", (256, 256))], "seed": 1, "seeds": [1]}

    params = (
        hires.HiresParams(denoise = 0.5, upscaler = "lanczos")
        if fail_in == "refine"
        else hires.HiresParams(denoise = 0.0, upscaler = "realesrgan_x4plus")
    )
    kw = {"prompt": "p", "width": 256, "height": 256, "steps": 4, "seed": 1, "batch_size": 1}
    with pytest.raises(RuntimeError, match = "failed"):
        hires.run_hires(_Backend(), fake_generate, params, kw)
    assert kicks == [1]

    kicks.clear()

    def base_fails(backend, **kw):
        raise RuntimeError("base failed")

    with pytest.raises(RuntimeError, match = "base failed"):
        hires.run_hires(_Backend(), base_fails, params, kw)
    assert kicks == []  # nothing was recorded, as a failed plain render


@pytest.fixture(scope = "module")
def tiny_sdxl_path():
    """Resolved at module scope, before the per-test empty hub cache and the outbound-network block apply."""
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    import sys
    from pathlib import Path

    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from conftest import allow_outbound
    from huggingface_hub import snapshot_download

    repo = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
    try:
        return snapshot_download(repo, local_files_only = True)
    except Exception:  # noqa: BLE001 - not cached yet
        try:
            with allow_outbound():
                return snapshot_download(repo)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"tiny pipeline not reachable: {exc}")


def _fresh_tiny_sdxl(path):
    import diffusers
    import torch

    # A fresh copy: offload hooks stay on the pipe, so the shared module fixture must not see them.
    return diffusers.StableDiffusionXLPipeline.from_pretrained(path, torch_dtype = torch.float32)


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch


def _latent_hires_backend(path, device, offload):
    from core.inference.diffusion import DiffusionBackend, _LoadState

    pipe = _fresh_tiny_sdxl(path)
    if offload == "model":
        pipe.enable_model_cpu_offload(device = device)
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload(device = device)
    else:
        pipe.to(device)
    b = DiffusionBackend()
    b._state = _LoadState(
        pipe = pipe,
        family = _fam("sdxl"),
        repo_id = "tiny",
        base_repo = "tiny",
        device = device,
        dtype = "float32",
        cpu_offload = offload != "none",
        offload_policy = offload,
    )
    return b, pipe


@pytest.mark.parametrize(
    "device, offload",
    [("cpu", "none"), ("cuda", "none"), ("cuda", "model"), ("cuda", "sequential")],
)
def test_latent_upscaler_refines_the_sampler_latents_like_comfyui(
    tiny_sdxl_path, monkeypatch, device, offload
):
    """ComfyUI LatentUpscale: the refine pass gets exactly ``F.interpolate(base_latents, nearest-exact)``, with no base
    decode and no re-encode (fail-before: the base pass decodes to 8-bit pixels and the refine VAE-encodes them)."""
    import functools

    import diffusers
    import numpy as np
    import torch
    import torch.nn.functional as F

    if device == "cuda":
        _cuda_or_skip()
    b, pipe = _latent_hires_backend(tiny_sdxl_path, device, offload)
    vae_calls = {"encode": 0, "decode": 0}
    for name in vae_calls:
        bound = getattr(pipe.vae, name)

        def counted(
            *a,
            _name = name,
            _bound = bound,
            **k,
        ):
            vae_calls[_name] += 1
            return _bound(*a, **k)

        monkeypatch.setattr(pipe.vae, name, counted)
    base_out, refine_in = [], []
    orig_base = diffusers.StableDiffusionXLPipeline.__call__
    orig_refine = diffusers.StableDiffusionXLImg2ImgPipeline.__call__

    @functools.wraps(orig_base)
    def base_spy(self, *a, **k):
        out = orig_base(self, *a, **k)
        base_out.append((k.get("output_type"), out.images))
        return out

    @functools.wraps(orig_refine)
    def refine_spy(self, *a, **k):
        refine_in.append(k["image"].detach().clone())
        return orig_refine(self, *a, **k)

    monkeypatch.setattr(diffusers.StableDiffusionXLPipeline, "__call__", base_spy)
    monkeypatch.setattr(diffusers.StableDiffusionXLImg2ImgPipeline, "__call__", refine_spy)
    kw = dict(
        prompt = "a cat", width = 256, height = 256, steps = 2, guidance = 5.0, seed = 7, live_preview = False
    )
    plain = b.generate(**kw, batch_size = 2)
    assert vae_calls == {"encode": 0, "decode": 1}
    base_out.clear()
    vae_calls.update(encode = 0, decode = 0)

    out = b.generate(
        **kw,
        batch_size = 2,
        hires = hires.HiresParams(scale = 1.5, denoise = 0.5, steps = 2, upscaler = "latent"),
    )
    assert [im.size for im in out["images"]] == [(384, 384)] * 2
    assert out["seeds"] == plain["seeds"]
    # One decode per refined image, none for the base pass, no encode at all.
    assert vae_calls == {"encode": 0, "decode": 2}
    ((output_type, latents),) = base_out
    f = pipe.vae_scale_factor
    assert output_type == "latent" and tuple(latents.shape) == (2, 4, 256 // f, 256 // f)
    assert len(refine_in) == 2
    for i, got in enumerate(refine_in):
        want = F.interpolate(latents[i : i + 1], size = (384 // f, 384 // f), mode = "nearest-exact")
        assert got.device == want.device and torch.equal(got, want)
    # The base pass is unchanged: its latents decode to the pixels a plain render returns.
    with torch.inference_mode():
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict = False)[0]
        pixels = pipe.image_processor.postprocess(decoded, output_type = "pil")
    for got, want in zip(pixels, plain["images"]):
        diff = np.abs(np.asarray(got, dtype = np.int16) - np.asarray(want, dtype = np.int16))
        assert diff.max() <= 1
    assert b._gen is None and b._active_generate_cancel is None


def test_latent_upscale_is_comfyui_nearest_exact_and_needs_latents():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    lat = torch.randn(1, 4, 8, 12)
    pipe = types.SimpleNamespace(vae_scale_factor = 8)
    got = hires.upscale_latents(pipe, lat, (144, 96))
    assert torch.equal(got, F.interpolate(lat, size = (12, 18), mode = "nearest-exact"))
    from PIL import Image

    with pytest.raises(TypeError, match = "latents"):
        hires.upscale_latents(pipe, Image.new("RGB", (64, 64)), (96, 96))


def test_generation_log_line_takes_a_latent_base_pass():
    """The latent base pass returns tensors, whose ``size`` is a method, not a pixel size."""
    torch = pytest.importorskip("torch")
    from core.inference.diffusion_auto_policy import format_generation_for_log

    line = format_generation_for_log(
        {"images": [torch.zeros(1, 4, 8, 8)], "workflow": "txt2img"}, engine = "diffusers"
    )
    assert "images=1" in line and "size=" not in line
