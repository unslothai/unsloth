# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An fp32 run of a force_upcast 16-bit VAE keeps its stored (pinned) tensors and matches stock fp32 output."""

from __future__ import annotations

import contextlib
import copy
import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_vae_fp16 as vf  # noqa: E402

CUDA = types.SimpleNamespace(device = "cuda")


def _vu():
    # Imported per test, so the behavioural tests below also run (and fail) against a tree without the helper.
    from core.inference import diffusion_vae_upcast
    return diffusion_vae_upcast


def _tiny_vae(dtype = torch.float16):
    torch.manual_seed(0)
    return (
        diffusers.AutoencoderKL(
            in_channels = 3,
            out_channels = 3,
            down_block_types = ("DownEncoderBlock2D",) * 3,
            up_block_types = ("UpDecoderBlock2D",) * 3,
            block_out_channels = (16, 32, 32),
            layers_per_block = 1,
            latent_channels = 4,
            norm_num_groups = 8,
            force_upcast = True,
        )
        .eval()
        .to(dtype)
    )


def _storage(module):
    return {name: (p.data_ptr(), p.dtype) for name, p in module.named_parameters()}


def _stock_fp32(vae):
    ref = copy.deepcopy(vae)
    ref.to(torch.float32)
    return ref


def test_round_trip_puts_back_the_same_tensors():
    vu = _vu()
    vae = _tiny_vae()
    before = _storage(vae)
    with vu.fp32_keeping_storage(vae):
        assert all(p.dtype == torch.float32 for p in vae.parameters())
    assert _storage(vae) == before


def test_parts_only_upcast_the_named_submodules():
    vu = _vu()
    vae = _tiny_vae()
    before = _storage(vae)
    with vu.fp32_keeping_storage(vae, [vae.encoder, None]):
        assert next(vae.encoder.parameters()).dtype == torch.float32
        assert next(vae.decoder.parameters()).dtype == torch.float16
    assert _storage(vae) == before


def test_a_tensor_replaced_meanwhile_is_cast_back_not_overwritten():
    vu = _vu()
    vae = _tiny_vae()
    token = vu.upcast_keeping_storage(vae)
    fresh = torch.zeros_like(vae.decoder.conv_out.weight.data)
    vae.decoder.conv_out.weight.data = fresh
    vu.restore_storage(token)
    w = vae.decoder.conv_out.weight
    assert w.dtype == torch.float16 and bool((w == 0).all())


def test_pipeline_round_trip_is_bit_identical_and_keeps_storage():
    vu = _vu()
    # diffusers' SDXL decode: vae.to(dtype=fp32), decode, vae.to(dtype=fp16).
    vae = _tiny_vae()
    ref = _stock_fp32(vae)
    z = torch.randn(1, 4, 8, 8)
    before = _storage(vae)
    pipe = types.SimpleNamespace(vae = vae)
    with vu.pipeline_upcasts_keep_storage(pipe):
        assert getattr(vae.__dict__["to"], vu.DTYPE_ONLY_TO_ATTR)
        assert vae.to(dtype = torch.float32) is vae
        assert vae.dtype == torch.float32
        out = vae.decode(z).sample
        vae.to(dtype = torch.float16)
        assert vae.to("cpu") is vae
    assert "to" not in vae.__dict__
    assert _storage(vae) == before
    assert torch.equal(out, ref.decode(z).sample)


def test_pipeline_raise_mid_upcast_leaves_the_load_dtype():
    vu = _vu()
    vae = _tiny_vae()
    before = _storage(vae)
    with contextlib.suppress(RuntimeError):
        with vu.pipeline_upcasts_keep_storage(types.SimpleNamespace(vae = vae)):
            vae.to(torch.float32)
            raise RuntimeError("decode failed")
    assert _storage(vae) == before


@pytest.mark.parametrize("dtype,force_upcast", [(torch.float32, True), (torch.float16, False)])
def test_pipeline_wrapper_is_inert_off_the_upcast_case(dtype, force_upcast):
    vu = _vu()
    vae = _tiny_vae(dtype)
    vae.register_to_config(force_upcast = force_upcast)
    with vu.pipeline_upcasts_keep_storage(types.SimpleNamespace(vae = vae)):
        assert "to" not in vae.__dict__


def test_fp16_decode_fp32_encode_keeps_encoder_storage():
    vae = _tiny_vae()
    ref = _stock_fp32(vae)
    assert vf.enable_fp16_vae_decode(
        types.SimpleNamespace(vae = vae, unet = type("UNet2DConditionModel", (), {})()), CUDA
    )
    x = torch.randn(1, 3, 32, 32, dtype = torch.float16)
    before = _storage(vae)
    got = vae.encode(x).latent_dist.mean
    assert _storage(vae) == before
    assert torch.equal(got, ref.encode(x.float()).latent_dist.mean)


@pytest.fixture(scope = "module")
def tiny_sdxl_path():
    """Resolved at module scope, before the per-test empty hub cache and the outbound-network block apply."""
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


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    pytest.importorskip("accelerate")


def test_sdxl_model_offload_vae_stays_pinned_cuda(monkeypatch, tiny_sdxl_path):
    """Stock: the decode's upcast lands before the VAE's first onload, so it is never pinned and every onload copies
    every tensor pageable. Here it is pinned after the first render and stays so; output bit-identical."""
    from core.inference import diffusion_memory as dm

    _cuda_or_skip()
    path = tiny_sdxl_path
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    outs = {}
    for arm in ("stock", "kept"):
        pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(path, torch_dtype = torch.float16)
        pipe.vae.register_to_config(force_upcast = True)
        pipe.enable_model_cpu_offload(device = "cuda:0")
        dm.keep_cpu_weights_on_offload(pipe)
        wrap = (
            _vu().pipeline_upcasts_keep_storage
            if arm == "kept"
            else lambda p: contextlib.nullcontext()
        )
        renders = []
        for _ in range(2):
            with wrap(pipe):
                renders.append(
                    pipe(
                        "a cat",
                        num_inference_steps = 2,
                        height = 64,
                        width = 64,
                        generator = torch.Generator("cpu").manual_seed(0),
                        output_type = "pt",
                    ).images
                )
        params = list(pipe.vae.parameters())
        assert all(p.dtype == torch.float16 and p.device.type == "cpu" for p in params)
        pinned = sum(p.is_pinned() for p in params)
        if arm == "kept":
            assert pinned == len(params)
            assert pipe.vae.__dict__[dm._KEEP_ATTR].get("pin_layout")
        outs[arm] = renders
    assert torch.equal(outs["stock"][0], outs["kept"][0])
    assert torch.equal(outs["kept"][0], outs["kept"][1])


def test_nested_load_dtype_casts_still_restore_the_original_tensors():
    # diffusers < 0.37 upcast_vae(): whole VAE to fp32, then these submodules back to the load dtype.
    vu = _vu()
    vae = _tiny_vae()
    before = _storage(vae)
    with vu.pipeline_upcasts_keep_storage(types.SimpleNamespace(vae = vae)):
        vae.to(dtype = torch.float32)
        for module in (vae.post_quant_conv, vae.decoder.conv_in, vae.decoder.mid_block):
            module.to(torch.float16)
        vae.to(dtype = torch.float16)
    assert _storage(vae) == before


def test_a_changed_same_dtype_replacement_is_left_alone():
    vu = _vu()
    vae = _tiny_vae()
    token = vu.upcast_keeping_storage(vae)
    vae.decoder.conv_out.to(torch.float16)
    with torch.no_grad():
        vae.decoder.conv_out.weight.zero_()
    vu.restore_storage(token)
    assert bool((vae.decoder.conv_out.weight == 0).all())


@pytest.mark.parametrize("route", ["helper", "pipeline"])
def test_sequential_offload_replacement_parameters_return_to_the_load_dtype(route):
    # accelerate's leaf hooks register new Parameter objects on every onload / offload.
    pytest.importorskip("accelerate")
    from accelerate import cpu_offload

    vu = _vu()
    vae = _tiny_vae()
    ref = _stock_fp32(vae)
    cpu_offload(vae, execution_device = torch.device("cpu"))
    assert next(vae.parameters()).device.type == "meta"
    x = torch.randn(1, 3, 32, 32, dtype = torch.float16)
    if route == "helper":
        with vu.fp32_keeping_storage(vae):
            got = vae.encode(x.float()).latent_dist.mean
    else:
        with vu.pipeline_upcasts_keep_storage(types.SimpleNamespace(vae = vae)):
            vae.to(dtype = torch.float32)
            got = vae.encode(x.float()).latent_dist.mean
            vae.to(dtype = torch.float16)
    assert {p.dtype for p in vae.parameters()} == {torch.float16}
    assert torch.equal(got, ref.encode(x.float()).latent_dist.mean)
