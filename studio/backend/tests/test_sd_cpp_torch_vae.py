# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only FLUX.2 sd.cpp renders: the server's final latent decodes in torch the way diffusers' FLUX.2 pipelines decode
it, only where that is allowed, and a decode failure falls back to sd.cpp's own decode."""

from __future__ import annotations

import base64
import json
import threading
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("safetensors")
from diffusers import AutoencoderKLFlux2  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.inference import sd_cpp_backend as bk  # noqa: E402
from core.inference import sd_cpp_torch_vae as tv  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402
from core.inference.sd_cpp_args import SdCppModelFiles  # noqa: E402
from core.inference.sd_cpp_server import SdCppServer  # noqa: E402

_TINY = dict(
    block_out_channels = (32, 32),
    down_block_types = ("DownEncoderBlock2D",) * 2,
    up_block_types = ("UpDecoderBlock2D",) * 2,
    layers_per_block = 1,
    latent_channels = 8,
    norm_num_groups = 8,
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(tv.ENV, raising = False)
    monkeypatch.setattr(tv.platform, "machine", lambda: "x86_64")
    tv._CACHE.clear()
    tv.STATS.update(decoded = 0, fallback = 0)
    yield
    tv._CACHE.clear()


def _tiny_vae_file(tmp_path, with_config = True):
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(**_TINY).eval()
    # Non-trivial batch-norm statistics, so the denormalize step is exercised.
    vae.bn.running_mean.copy_(torch.randn_like(vae.bn.running_mean) * 0.1)
    vae.bn.running_var.copy_(torch.rand_like(vae.bn.running_var) + 0.5)
    path = tmp_path / "vae" / "diffusion_pytorch_model.safetensors"
    path.parent.mkdir(parents = True)
    save_file({k: v.contiguous() for k, v in vae.state_dict().items()}, str(path))
    if with_config:
        (path.parent / "config.json").write_text(
            json.dumps({"_class_name": "AutoencoderKLFlux2", "force_upcast": True, **_TINY})
        )
    return vae, str(path)


def _reference(vae, latent):
    """Flux2KleinPipeline's post-denoise decode, written against diffusers' own helpers."""
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    z = Flux2KleinPipeline._unpatchify_latents(latent * std + mean)
    with torch.inference_mode():
        image = vae.decode(z, return_dict = False)[0]
    return VaeImageProcessor().postprocess(image, output_type = "pil")


def _as_wire(latent):
    return (tuple(latent.shape), latent.contiguous().numpy().tobytes())


def test_decode_matches_the_diffusers_pipeline_decode(tmp_path):
    vae, path = _tiny_vae_file(tmp_path)
    torch.manual_seed(1)
    latent = torch.randn(1, _TINY["latent_channels"] * 4, 6, 5)
    got = tv.decode([_as_wire(latent)], path, threads = 2)
    want = _reference(vae, latent)
    assert len(got) == 1 and got[0].size == want[0].size == (20, 24)
    import numpy as np

    diff = np.abs(np.asarray(got[0], dtype = np.int16) - np.asarray(want[0], dtype = np.int16))
    # channels_last against contiguous fp32: at most one rounding step on a few pixels.
    assert diff.max() <= 1 and (diff > 0).mean() < 0.01
    assert tv.STATS["decoded"] == 1


def test_decode_without_a_config_uses_the_flux2_defaults(tmp_path, monkeypatch):
    _vae, path = _tiny_vae_file(tmp_path, with_config = False)
    # No config next to the file: AutoencoderKLFlux2() defaults, which the tiny weights do not fit.
    with pytest.raises(RuntimeError):
        tv.decode([_as_wire(torch.zeros(1, 32, 2, 2))], path)


def test_decode_keeps_the_thread_count(tmp_path):
    _vae, path = _tiny_vae_file(tmp_path)
    before = torch.get_num_threads()
    tv.decode([_as_wire(torch.zeros(1, 32, 2, 2))], path, threads = 1)
    assert torch.get_num_threads() == before


def test_decode_rejects_a_short_or_misshapen_latent(tmp_path):
    _vae, path = _tiny_vae_file(tmp_path)
    with pytest.raises(ValueError):
        tv.decode([((1, 32, 2, 2), b"\0" * 8)], path)
    with pytest.raises(ValueError):
        tv.decode([((1, 30, 2, 2), b"\0" * 480)], path)


def _state(
    tmp_path,
    family = "flux.2-klein",
    device = "cpu",
    vae = True,
):
    vae_path = tmp_path / ("ae.safetensors" if vae else "missing.safetensors")
    if vae:
        vae_path.write_bytes(b"x")
    return types.SimpleNamespace(
        family = detect_family(family),
        device = device,
        server = object(),
        files = SdCppModelFiles(diffusion_model = "/m/d.gguf", vae = str(vae_path)),
    )


def test_applies_only_to_cpu_flux2_renders(tmp_path, monkeypatch):
    assert tv.applies(_state(tmp_path))
    assert tv.applies(_state(tmp_path, family = "flux.2-dev"))
    assert not tv.applies(_state(tmp_path), layers = 3)
    assert not tv.applies(_state(tmp_path, device = "vulkan"))
    assert not tv.applies(_state(tmp_path, family = "z-image"))
    assert not tv.applies(_state(tmp_path, vae = False))
    monkeypatch.setenv(tv.ENV, "0")
    assert not tv.applies(_state(tmp_path))


def test_other_cpu_architectures_need_the_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(tv.platform, "machine", lambda: "aarch64")
    assert not tv.applies(_state(tmp_path))
    monkeypatch.setenv(tv.ENV, "1")
    assert tv.applies(_state(tmp_path))


def test_server_client_parses_and_checks_latents():
    data = torch.arange(8, dtype = torch.float32).numpy().tobytes()
    job = {
        "result": {
            "images": [
                {
                    "index": 1,
                    "latent": {
                        "b64": base64.b64encode(data).decode(),
                        "dtype": "float32",
                        "shape": [1, 8, 1, 1],
                    },
                },
                {
                    "index": 0,
                    "latent": {
                        "b64": base64.b64encode(data).decode(),
                        "dtype": "float32",
                        "shape": [1, 2, 2, 2],
                    },
                },
            ]
        }
    }
    out = SdCppServer._decode_latents(job)
    assert [shape for shape, _ in out] == [(1, 2, 2, 2), (1, 8, 1, 1)]
    job["result"]["images"][0]["latent"]["shape"] = [1, 8, 2, 1]
    with pytest.raises(RuntimeError):
        SdCppServer._decode_latents(job)
    with pytest.raises(RuntimeError):
        SdCppServer._decode_latents({"result": {"images": [{"index": 0, "b64_json": "AA=="}]}})


def test_img_gen_accepts_reads_the_img_gen_features():
    srv = SdCppServer.__new__(SdCppServer)
    srv._vid_gen_features = {}
    srv._img_gen_features = {"output_latent": True}
    srv._vid_gen_output_formats = ()
    assert srv.img_gen_accepts(("output_latent",))
    srv._img_gen_features = {}
    assert not srv.img_gen_accepts(("output_latent",))


class _Server:
    lora_dir = None

    def __init__(
        self,
        latents,
        accepts = True,
    ):
        self.latents = latents
        self.accepts = accepts
        self.calls = []

    def is_alive(self):
        return True

    def img_gen_accepts(self, features):
        return self.accepts

    def img_gen_latents(self, payload, **_kw):
        self.calls.append("latent")
        return self.latents

    def img_gen(self, payload, **_kw):
        import io

        self.calls.append("png")
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format = "PNG")
        return [buf.getvalue()]


def _backend(
    tmp_path,
    server,
    family = "flux.2-klein",
    device = "cpu",
):
    _vae, path = _tiny_vae_file(tmp_path)
    fam = detect_family(family)
    b = bk.SdCppDiffusionBackend.__new__(bk.SdCppDiffusionBackend)
    b._on_log = lambda *_a, **_k: None
    state = bk._SdState(
        repo_id = "unsloth/FLUX.2-klein-4B-GGUF",
        base_repo = fam.base_repo,
        family = fam,
        device = device,
        files = SdCppModelFiles(diffusion_model = "/m/k.gguf", vae = path),
        mode = "server",
        server = server,
        threads = 1,
    )
    return b, state


def _kwargs():
    return dict(
        prompt = "a fox",
        negative_prompt = None,
        width = 16,
        height = 16,
        steps = 1,
        seed = 1,
        batch_size = 1,
        cfg_scale = None,
        flux_guidance = None,
        lora_resolved = [],
        cancel = threading.Event(),
    )


def test_a_cpu_flux2_render_decodes_the_latent_in_torch(tmp_path):
    server = _Server([_as_wire(torch.zeros(1, 32, 2, 2))])
    b, state = _backend(tmp_path, server)
    images, seeds = b._generate_server(state, **_kwargs())
    assert server.calls == ["latent"]
    assert len(images) == 1 and images[0].size == (8, 8) and images[0].mode == "RGB"
    assert tv.STATS == {"decoded": 1, "fallback": 0}


def test_a_failed_torch_decode_falls_back_to_the_server_decode(tmp_path):
    server = _Server([((1, 32, 2, 2), b"short")])
    b, state = _backend(tmp_path, server)
    images, _seeds = b._generate_server(state, **_kwargs())
    assert server.calls == ["latent", "png"]
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert tv.STATS["fallback"] == 1


@pytest.mark.parametrize(
    "accepts, device, env", [(False, "cpu", None), (True, "vulkan", None), (True, "cpu", "0")]
)
def test_the_server_decode_stays_everywhere_else(tmp_path, monkeypatch, accepts, device, env):
    if env is not None:
        monkeypatch.setenv(tv.ENV, env)
    server = _Server([_as_wire(torch.zeros(1, 32, 2, 2))], accepts = accepts)
    b, state = _backend(tmp_path, server, device = device)
    b._generate_server(state, **_kwargs())
    assert server.calls == ["png"]
    assert tv.STATS["decoded"] == 0


def test_an_unloadable_decoder_skips_the_latent_request(tmp_path):
    # A decoder that cannot be built must not cost a render: the server decode runs once, no latent is asked for.
    server = _Server([_as_wire(torch.zeros(1, 32, 2, 2))])
    b, state = _backend(tmp_path, server)
    Path(state.files.vae).write_bytes(b"not a safetensors file")
    tv._CACHE.clear()
    b._generate_server(state, **_kwargs())
    assert server.calls == ["png"]
    assert tv.STATS == {"decoded": 0, "fallback": 1}


def test_a_host_without_room_for_the_decode_keeps_the_server_decode(tmp_path, monkeypatch):
    from core.inference import diffusion_memory

    need = tv.decode_bytes(16, 16) + tv._DECODE_RESERVE_BYTES
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: (need >> 20) - 1)
    server = _Server([_as_wire(torch.zeros(1, 32, 2, 2))])
    b, state = _backend(tmp_path, server)
    b._generate_server(state, **_kwargs())
    assert server.calls == ["png"] and tv.STATS["decoded"] == 0
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: (need >> 20) + 1)
    assert tv.fits_host_ram(16, 16)
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: None)
    assert tv.fits_host_ram(1 << 20, 1 << 20), "unknown host RAM is not a refusal"


def test_a_batched_latent_decodes_one_image_at_a_time(tmp_path, monkeypatch):
    _vae, path = _tiny_vae_file(tmp_path)
    torch.manual_seed(2)
    latent = torch.randn(3, _TINY["latent_channels"] * 4, 6, 5)
    vae = tv._load_decoder(path)
    batches = []
    real = vae.decode
    monkeypatch.setattr(
        vae, "decode", lambda z, **kw: (batches.append(z.shape[0]), real(z, **kw))[1]
    )
    together = tv.decode([_as_wire(latent)], path)
    one_by_one = [tv.decode([_as_wire(latent[i : i + 1])], path)[0] for i in range(3)]
    assert batches == [1] * 6
    assert [im.tobytes() for im in together] == [im.tobytes() for im in one_by_one]
