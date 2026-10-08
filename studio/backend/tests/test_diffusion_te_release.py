# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text encoders on unified memory (#11792): the int8 ConvRot encoder on Apple Silicon, the load-time refusal it
lifts, and releasing the encoders between prompts. The release tests run on whatever accelerator the runner has
(MPS on Apple Silicon runners, CUDA, else CPU), never skipped for the device."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_te_prequant as te_prequant
from core.inference import diffusion_te_release as te_release
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.diffusion_memory import (
    DeviceMemory,
    plan_diffusion_memory,
    unified_memory_shortfall_message,
)

GIB = 1024

# Qwen-Image-2.1 on a 48 GB M5 Max, from the report: macOS said 27 GB free.
Q21_DIT_Q4_K_M_MIB = 4_199_565_024 >> 20
Q21_TE_BF16_MIB = 17_534_339_488 >> 20
Q21_TE_INT8_CONVROT_MIB = 9_349_769_248 >> 20
Q21_VAE_MIB = 650


def _target(device: str, dtype = None) -> DiffusionDeviceTarget:
    return DiffusionDeviceTarget(
        device = device,
        dtype = dtype if dtype is not None else torch.bfloat16,
        backend = device,
        vendor = None,
        supports_model_cpu_offload = device == "cuda",
        supports_default_torch_compile = device == "cuda",
        supports_pinned_transfer = device == "cuda",
    )


def _q21_family():
    from core.inference.diffusion_families import detect_family

    fam = detect_family("Qwen/Qwen-Image-2.1")
    assert fam is not None and fam.name == "qwen-image-2.1"
    return fam


def _plan(memory: DeviceMemory, weights: int, encoders: int, headroom: int = 8 * GIB):
    return plan_diffusion_memory(
        target = _target("mps" if memory.memory_kind == "unified_memory" else "cuda"),
        device_memory = memory,
        model_dense_mib = weights,
        runtime_headroom_mib = headroom,
        companion_dense_mib = encoders + Q21_VAE_MIB,
        text_encoder_dense_mib = encoders,
    )


M5_MAX = DeviceMemory("mps", "mps", "unified_memory", free_mib = 27 * GIB, total_mib = 48 * GIB)


# -- the int8 ConvRot encoder on Apple Silicon --------------------------------------------------------------------


def test_mps_takes_the_int8_convrot_encoder_without_the_fp8_fallback():
    sources = te_prequant.te_prequant_sources(
        _q21_family(), te_quant_mode = "int8", target = _target("mps")
    )
    source = sources["text_encoder"]
    assert source.filename == "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors"
    # The fp8 names it falls back to on CUDA cannot run on MPS.
    assert source.fallback_filenames == ()


def test_mps_fp8_request_still_takes_no_precast_encoder():
    assert (
        te_prequant.te_prequant_sources(_q21_family(), te_quant_mode = "fp8", target = _target("mps"))
        == {}
    )


def test_cpu_takes_no_precast_encoder():
    assert (
        te_prequant.te_prequant_sources(
            _q21_family(), te_quant_mode = "int8", target = _target("cpu", torch.float32)
        )
        == {}
    )


def test_cuda_keeps_the_fp8_fallback(monkeypatch):
    from core.inference import diffusion_precision

    monkeypatch.setattr(diffusion_precision, "te_quant_supported", lambda target, mode: True)
    source = te_prequant.te_prequant_sources(
        _q21_family(), te_quant_mode = "int8", target = _target("cuda")
    )["text_encoder"]
    assert source.fallback_filenames, "the CUDA path must keep its fp8 fallback names"
    assert (
        te_prequant.te_prequant_budget_scale(
            _q21_family(), te_quant_mode = "int8", target = _target("cuda"), base = "Qwen/Qwen-Image-2.1"
        )
        == te_prequant.TE_PREQUANT_BUDGET_SCALE
    )


def test_mps_budgets_the_int8_encoder_at_its_own_size():
    scale = te_prequant.te_prequant_budget_scale(
        _q21_family(), te_quant_mode = "int8", target = _target("mps"), base = "Qwen/Qwen-Image-2.1"
    )
    assert scale == te_prequant.TE_INT8_CONVROT_BUDGET_SCALE
    assert Q21_TE_BF16_MIB * scale >= Q21_TE_INT8_CONVROT_MIB


def test_int8_convrot_linear_runs_on_this_accelerator():
    """The int8 ConvRot projection the Apple Silicon path now loads: plain tensors, so it runs on MPS too."""
    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class

    device = _device()
    torch.manual_seed(0)
    dense = torch.nn.Linear(512, 256, bias = True)
    codes, scale = te_prequant.quantize_int8_convrot_weight(dense.weight.detach(), group_size = 256)
    layer = _int8_convrot_linear_class()(512, 256, bias = True, group_size = 256)
    layer.load_state_dict(
        {"weight": codes, "weight_scale": scale, "bias": dense.bias.detach().to(torch.bfloat16)},
        assign = True,
    )
    layer = layer.to(device)
    x = torch.randn(4, 512)
    with torch.no_grad():
        got = layer(x.to(device)).float().cpu()
        want = dense(x)
    rel = (got - want).norm() / want.norm()
    assert got.shape == (4, 256)
    assert rel < 0.02, float(rel)


# -- the load-time refusal ----------------------------------------------------------------------------------------


def test_m5_max_refuses_the_bf16_encoder():
    weights = Q21_DIT_Q4_K_M_MIB + Q21_TE_BF16_MIB + Q21_VAE_MIB
    assert unified_memory_shortfall_message(_plan(M5_MAX, weights, Q21_TE_BF16_MIB)) is not None


def test_m5_max_accepts_the_int8_convrot_encoder():
    encoders = int(Q21_TE_BF16_MIB * te_prequant.TE_INT8_CONVROT_BUDGET_SCALE)
    weights = Q21_DIT_Q4_K_M_MIB + encoders + Q21_VAE_MIB
    assert unified_memory_shortfall_message(_plan(M5_MAX, weights, encoders)) is None


def test_a_genuinely_oversized_load_is_still_refused():
    # The dense bf16 Qwen-Image-2.1 transformer with the int8 encoder does not fit 27 GB free either.
    encoders = int(Q21_TE_BF16_MIB * te_prequant.TE_INT8_CONVROT_BUDGET_SCALE)
    weights = (14_230_284_408 >> 20) + encoders + Q21_VAE_MIB
    assert unified_memory_shortfall_message(_plan(M5_MAX, weights, encoders)) is not None


def test_discrete_vram_never_refuses_here():
    memory = DeviceMemory("cuda", "cuda", "discrete_vram", free_mib = 8 * GIB, total_mib = 24 * GIB)
    weights = Q21_DIT_Q4_K_M_MIB + Q21_TE_BF16_MIB + Q21_VAE_MIB
    assert unified_memory_shortfall_message(_plan(memory, weights, Q21_TE_BF16_MIB)) is None


# -- when the release engages -------------------------------------------------------------------------------------


def test_release_engages_on_unified_memory_when_the_resident_set_does_not_fit(monkeypatch):
    monkeypatch.delenv(te_release.RELEASE_ENV, raising = False)
    encoders = int(Q21_TE_BF16_MIB * te_prequant.TE_INT8_CONVROT_BUDGET_SCALE)
    weights = Q21_DIT_Q4_K_M_MIB + encoders + Q21_VAE_MIB
    wanted, reason = te_release.release_wanted(_plan(M5_MAX, weights, encoders))
    assert wanted, reason


def test_release_stays_off_when_everything_fits(monkeypatch):
    monkeypatch.delenv(te_release.RELEASE_ENV, raising = False)
    roomy = DeviceMemory("mps", "mps", "unified_memory", free_mib = 100 * GIB, total_mib = 128 * GIB)
    weights = Q21_DIT_Q4_K_M_MIB + Q21_TE_BF16_MIB + Q21_VAE_MIB
    assert te_release.release_wanted(_plan(roomy, weights, Q21_TE_BF16_MIB))[0] is False


def test_release_stays_off_on_discrete_vram(monkeypatch):
    monkeypatch.delenv(te_release.RELEASE_ENV, raising = False)
    memory = DeviceMemory("cuda", "cuda", "discrete_vram", free_mib = 8 * GIB, total_mib = 24 * GIB)
    weights = Q21_DIT_Q4_K_M_MIB + Q21_TE_BF16_MIB + Q21_VAE_MIB
    assert te_release.release_wanted(_plan(memory, weights, Q21_TE_BF16_MIB))[0] is False


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False)])
def test_release_env_forces_either_way(monkeypatch, value, expected):
    monkeypatch.setenv(te_release.RELEASE_ENV, value)
    memory = DeviceMemory("cuda", "cuda", "discrete_vram", free_mib = 8 * GIB, total_mib = 24 * GIB)
    assert te_release.release_wanted(_plan(memory, 1, 1))[0] is expected


# -- release and reload, on this runner's accelerator ------------------------------------------------------------


def _device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _Encoder(torch.nn.Module):
    """A stand-in encoder: a tied embedding / head pair, int8 buffers like the ConvRot encoder, a small norm."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(512, 1024)
        self.proj = torch.nn.Linear(1024, 1024)
        self.register_buffer("codes", torch.randint(-127, 127, (1024, 1024), dtype = torch.int8))
        self.register_buffer("rotary", torch.arange(16, dtype = torch.float32), persistent = False)
        self.norm = torch.nn.LayerNorm(1024)
        self.head = torch.nn.Linear(1024, 512, bias = False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        h = self.norm(self.proj(self.embed(ids)) + self.codes[: ids.shape[-1]].to(self.proj.weight.dtype).sum(-1, keepdim = True))
        return self.head(h) + self.rotary.sum()


@pytest.fixture
def snapshot_root(tmp_path, monkeypatch):
    monkeypatch.setattr(te_release, "_snapshot_root", lambda: tmp_path / "snap")
    return tmp_path / "snap"


def _pipe(device: str):
    torch.manual_seed(0)
    encoder = _Encoder().to(device)
    return types.SimpleNamespace(text_encoder = encoder, transformer = torch.nn.Linear(4, 4).to(device))


def test_release_frees_and_reload_restores_bit_for_bit(snapshot_root):
    device = _device()
    pipe = _pipe(device)
    ids = torch.arange(8, device = device)
    with torch.no_grad():
        before = pipe.text_encoder(ids).cpu()
    releaser = te_release.TextEncoderReleaser([("text_encoder", pipe.text_encoder)])
    total = releaser.releasable_bytes
    assert total > 2 << 20
    assert releaser.release() == total
    assert releaser.released
    assert pipe.text_encoder.proj.weight.numel() == 0
    assert pipe.text_encoder.codes.numel() == 0
    # Device and dtype survive, so pipelines that read them before encoding still work.
    assert pipe.text_encoder.proj.weight.device.type == device
    assert pipe.text_encoder.proj.weight.dtype == torch.float32
    # Small and non-persistent tensors stay.
    assert pipe.text_encoder.rotary.numel() == 16
    # The forward pre-hook reloads before the encoder runs.
    with torch.no_grad():
        after = pipe.text_encoder(ids).cpu()
    assert not releaser.released and releaser.reloads == 1
    assert torch.equal(before, after)
    # The tie survives: one tensor, released and restored once.
    assert pipe.text_encoder.head.weight is pipe.text_encoder.embed.weight
    # A second release reuses the snapshot; a submodule called directly also reloads.
    shards = list(snapshot_root.rglob("*.safetensors"))
    assert releaser.release() == total
    assert sorted(snapshot_root.rglob("*.safetensors")) == sorted(shards)
    with torch.no_grad():
        pipe.text_encoder.proj(torch.zeros(1, 1024, device = device))
    assert releaser.reloads == 2
    releaser.close()
    assert not any(snapshot_root.rglob("*.safetensors"))


def test_release_refuses_offload_hooked_encoders(snapshot_root):
    pipe = _pipe(_device())
    pipe.text_encoder.proj._hf_hook = object()
    releaser = te_release.TextEncoderReleaser([("text_encoder", pipe.text_encoder)])
    assert releaser.release() == 0
    assert pipe.text_encoder.proj.weight.numel() > 0


def test_release_refuses_when_the_disk_is_full(snapshot_root, monkeypatch):
    pipe = _pipe(_device())
    usage = types.SimpleNamespace(total = 1, used = 1, free = 1)
    monkeypatch.setattr(te_release.shutil, "disk_usage", lambda path: usage)
    releaser = te_release.TextEncoderReleaser([("text_encoder", pipe.text_encoder)])
    assert releaser.release() == 0
    assert pipe.text_encoder.proj.weight.numel() > 0


def test_maybe_install_only_when_wanted(snapshot_root, monkeypatch):
    pipe = _pipe(_device())
    monkeypatch.setenv(te_release.RELEASE_ENV, "0")
    assert te_release.maybe_install(pipe, None) is None
    monkeypatch.setenv(te_release.RELEASE_ENV, "1")
    releaser = te_release.maybe_install(pipe, None)
    assert releaser is not None
    releaser.close()


@pytest.mark.allow_network
def test_tiny_qwen_image_21_pipeline_renders_the_same_with_release(snapshot_root, monkeypatch):
    """The real QwenImage21Pipeline, tiny: release at denoise entry, reload for a new prompt, same images."""
    diffusers = pytest.importorskip("diffusers")
    pipeline_cls = getattr(diffusers, "QwenImage21Pipeline", None)
    if pipeline_cls is None:
        pytest.skip("this diffusers has no QwenImage21Pipeline")
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download("hf-internal-testing/tiny-qwenimage21-pipe")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny pipeline not reachable: {exc}")
    from core.inference.media_decode_phase import denoise_phase

    device = _device()
    dtype = torch.float32

    def render(pipe, prompt, releaser = None):
        def on_denoise():
            if releaser is not None:
                releaser.release()

        with denoise_phase(pipe, on_denoise):
            out = pipe(
                prompt = prompt,
                height = 64,
                width = 64,
                num_inference_steps = 2,
                generator = torch.Generator("cpu").manual_seed(0),
                output_type = "pt",
            )
        return out.images.cpu()

    pipe = pipeline_cls.from_pretrained(path, torch_dtype = dtype).to(device)
    pipe.set_progress_bar_config(disable = True)
    reference = [render(pipe, "a red cube"), render(pipe, "a blue sphere")]

    releaser = te_release.TextEncoderReleaser([("text_encoder", pipe.text_encoder)])
    first = render(pipe, "a red cube", releaser)
    assert releaser.released and releaser.releases == 1
    second = render(pipe, "a blue sphere", releaser)
    assert releaser.reloads == 1 and releaser.releases == 2
    assert torch.equal(first, reference[0])
    assert torch.equal(second, reference[1])
    releaser.close()
