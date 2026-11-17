# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm: the fused SDPA kernels take a head dim of 512 (the VAE's single-head attention) and compute it wrong,
so such calls must go to the exact math path, in inference and in the LoRA trainer's latent encode."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_vae_attn_chunked as C  # noqa: E402


def _cuda_q(head_dim):
    return types.SimpleNamespace(
        device = types.SimpleNamespace(type = "cuda"),
        dtype = torch.float32,
        shape = (1, 1, 4096, head_dim),
        requires_grad = False,
    )


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.14.0", raising = False)


@pytest.mark.parametrize("head_dim", [384, 512])
def test_windows_rocm_wide_head_is_not_a_fused_call(monkeypatch, rocm, head_dim):
    monkeypatch.setattr(sys, "platform", "win32")
    q = _cuda_q(head_dim)
    assert C.fused_sdpa_available(q, q, q) is False


@pytest.mark.parametrize("platform, head_dim", [("win32", 128), ("win32", 256), ("linux", 512)])
def test_other_heads_and_platforms_keep_the_kernel_query(monkeypatch, rocm, platform, head_dim):
    # A fake query makes torch's own SDPAParams query raise, which counts as "a fused kernel takes it".
    monkeypatch.setattr(sys, "platform", platform)
    q = _cuda_q(head_dim)
    assert C.fused_sdpa_available(q, q, q) is True


def test_windows_cuda_is_unaffected(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    monkeypatch.setattr(sys, "platform", "win32")
    q = _cuda_q(512)
    assert C.fused_sdpa_available(q, q, q) is True


def test_linux_rocm_wide_head_uses_fp32_math_when_fused_dispatch_is_unavailable(monkeypatch, rocm):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(C, "fused_sdpa_available", lambda q, k, v: False)
    monkeypatch.setattr(C, "_rocm_untrusted_head", lambda q, torch: True)
    monkeypatch.setattr(C, "_half_math_call", lambda q: True)
    calls = []
    real = C.fp32_math_sdpa
    monkeypatch.setattr(C, "fp32_math_sdpa", lambda *a, **k: calls.append(1) or real(*a, **k))

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 64, 512).to(torch.bfloat16) for _ in range(3))
    with C.chunked_attention_mode(budget = 1 << 30):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert calls == [1]
    assert out.dtype == torch.bfloat16


def test_dit_trainer_routes_its_vae_attention_on_rocm(rocm):
    diffusers = pytest.importorskip("diffusers")
    from core.training import diffusion_dit_trainer as dit

    torch.manual_seed(0)
    vae = diffusers.AutoencoderKL(
        in_channels = 3,
        out_channels = 3,
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, 64),
        layers_per_block = 1,
        latent_channels = 4,
        norm_num_groups = 8,
    ).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        ref = vae.encode(x).latent_dist.mode()
    assert dit._guard_vae_attention(vae) > 0
    patched = [m for m in vae.modules() if getattr(m, "_unsloth_vae_attn_chunked", False)]
    assert patched
    with torch.no_grad():
        # Exact: the routed path is the same math on CPU.
        assert torch.equal(vae.encode(x).latent_dist.mode(), ref)


class _UNet2DConditionModel:
    pass


@pytest.mark.parametrize("platform, allowed", [("win32", False), ("linux", True)])
def test_vae_decode_compile_stays_eager_on_windows_rocm(monkeypatch, rocm, platform, allowed):
    # A compiled decode bypasses the eager routing; a U-Net pipe otherwise always compiles its VAE decode.
    from core.inference import diffusion_speed as S

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(S, "_fused_vae_planned", lambda pipe: False)
    monkeypatch.setattr(S, "_UNET_WHOLE_COMPILE", frozenset({"_UNet2DConditionModel"}))
    pipe = types.SimpleNamespace(unet = _UNet2DConditionModel(), vae = object())
    assert S._vae_decode_compile_allowed(pipe, "default") is allowed


@pytest.mark.parametrize("budget", [2**30, 2**16])
def test_windows_rocm_fp32_wide_head_never_reaches_the_stock_kernel(monkeypatch, rocm, budget):
    # fp32 (every trainer VAE, and fp32 inference VAEs) is no half-math call, and Windows ROCm's stock dispatch still
    # picks the fused kernels that miscompute head 512 in fp32 too: the routed call must compute the math itself.
    import torch.nn.functional as F
    from torch.overrides import TorchFunctionMode

    monkeypatch.setattr(sys, "platform", "win32")
    # CPU tensors answer "fused" before the device check: stand in for the cuda branch's head guard.
    monkeypatch.setattr(
        C, "fused_sdpa_available", lambda q, k, v: not C._windows_rocm_untrusted_head(q, torch)
    )
    stock = []

    class _Spy(TorchFunctionMode):
        def __torch_function__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            if func is F.scaled_dot_product_attention:
                stock.append(1)
            return func(*args, **(kwargs or {}))

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 256, 512) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)
    with _Spy():
        with C.chunked_attention_mode(budget):
            out = F.scaled_dot_product_attention(q, k, v)
    assert stock == []
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("half_math_env", ["", "0"])
def test_a_rejected_fp32_head_runs_explicit_math_not_the_default_dispatch(
    monkeypatch, half_math_env
):
    # The trainer's VAE is fp32, which the half-precision math route skips: a rejected Windows ROCm head must still
    # leave the default dispatch (which picks the broken fused kernels), also with half math turned off.
    monkeypatch.setenv(C.VAE_ATTN_HALF_MATH_ENV, half_math_env)
    monkeypatch.setattr(C, "fused_sdpa_available", lambda q, k, v: False)
    monkeypatch.setattr(C, "_windows_rocm_untrusted_head", lambda q, torch: True)
    calls = []
    real = C.half_math_sdpa
    monkeypatch.setattr(C, "half_math_sdpa", lambda *a, **k: calls.append(1) or real(*a, **k))
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 64, 512) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    with C.chunked_attention_mode(budget = 1 << 30):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert calls and torch.allclose(out, ref, atol = 1e-5)
    calls.clear()
    with C.chunked_attention_mode(budget = 64 * 1024):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert calls and torch.allclose(out, ref, atol = 1e-5)


@pytest.mark.parametrize("warm", [False, True])
def test_dit_trainer_guards_the_preview_vae_a_warm_cache_loads(monkeypatch, tmp_path, warm):
    # A warm conditioning cache skips the encode but still loads the VAE to decode previews: that decode goes
    # through the same head-512 attention, so it needs the same routing as the cold path's encode.
    import contextlib
    import random

    from core.training import diffusion_dit_trainer as dit
    from core.training.diffusion_train_common import DiffusionLoraConfig

    for name, ns in (
        ("diffusers", types.SimpleNamespace(FlowMatchEulerDiscreteScheduler = object)),
        ("diffusers.optimization", types.SimpleNamespace(get_scheduler = object)),
        ("diffusers.training_utils", types.SimpleNamespace(cast_training_params = object)),
        ("peft", types.SimpleNamespace(LoraConfig = object)),
        ("peft.utils", types.SimpleNamespace(get_peft_model_state_dict = object)),
    ):
        monkeypatch.setitem(sys.modules, name, ns)
    monkeypatch.setattr("loggers.config.quiet_third_party_progress_bars", lambda: None)
    monkeypatch.setattr(dit, "plan_image_canvases", lambda *_a: (None, False))
    monkeypatch.setattr(dit, "_plan_cache_variants", lambda *_a, **_k: object())
    monkeypatch.setattr(dit, "_free_text_encoders", lambda *_a: None)
    monkeypatch.setattr(
        dit, "_encode_prompts_cached", lambda _s, _p, texts, *_a: [object() for _ in texts]
    )
    monkeypatch.setattr(dit, "_resolve_base_precision", lambda *_a: "bf16")
    monkeypatch.setattr(dit, "isolated_sampling", lambda *_a: contextlib.nullcontext())
    monkeypatch.setattr(dit, "_build_latent_cache", lambda *_a, **_k: [object()])
    monkeypatch.setattr(dit, "plan_samples", lambda *_a: types.SimpleNamespace(encode_texts = ["p"]))
    monkeypatch.setattr("core.training.diffusion_train_extras.source_revision", lambda *_a: "local")
    monkeypatch.setattr(dit, "PersistentConditioningCache", lambda *_a: object())
    monkeypatch.setattr(
        dit,
        "_load_warm_conditioning",
        lambda *_a: ({"caption": object()}, [object()]) if warm else (None, None),
    )
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_LATENT_CACHE", raising = False)
    guarded = []
    monkeypatch.setattr(dit, "_guard_vae_attention", lambda v: guarded.append(v) or 1)

    class _Stop(Exception):
        pass

    def stop(*_a):
        raise _Stop

    vae = types.SimpleNamespace(to = lambda *_a: None)
    spec = types.SimpleNamespace(
        family = "flux.1",
        lora_targets = ("to_q",),
        decode_latents = True,
        load_conditioners = lambda *_a: (types.SimpleNamespace(), vae),
        load_transformer = stop,
    )
    cfg = DiffusionLoraConfig(
        base_model = "unsloth/FLUX.1-dev",
        data_dir = str(tmp_path),
        output_dir = str(tmp_path / "out"),
        cache_latents = True,
        cond_cache_dir = str(tmp_path / "cache"),
    )
    with pytest.raises(_Stop):
        dit._train_dit(
            cfg,
            spec,
            [(str(tmp_path / "image.png"), "caption")],
            random.Random(0),
            "cpu",
            None,
            lambda *_a: None,
            lambda: False,
            lambda: True,
            None,
        )
    assert guarded and all(v is vae for v in guarded)


@pytest.mark.parametrize("half_math_env, fp32_math", [("", 0), ("0", 1)])
def test_a_rejected_half_head_honours_the_half_math_switch(monkeypatch, half_math_env, fp32_math):
    # UNSLOTH_VAE_ATTN_HALF_MATH=0 promises the fp32 upcast; a rejected Windows ROCm head must keep that promise
    # instead of falling back to the default dispatch (broken there) or to half-precision math.
    monkeypatch.setenv(C.VAE_ATTN_HALF_MATH_ENV, half_math_env)
    monkeypatch.setattr(C, "fused_sdpa_available", lambda q, k, v: False)
    monkeypatch.setattr(C, "_windows_rocm_untrusted_head", lambda q, torch: True)
    # CPU tensors are never a half-math call: stand in for the cuda + ROCm part of the check.
    monkeypatch.setattr(
        C, "_half_math_call", lambda q: q.dtype == torch.bfloat16 and C.half_math_enabled()
    )
    calls = []
    real = C.fp32_math_sdpa
    monkeypatch.setattr(C, "fp32_math_sdpa", lambda *a, **k: calls.append(1) or real(*a, **k))
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 64, 512).to(torch.bfloat16) for _ in range(3))
    with C.chunked_attention_mode(budget = 1 << 30):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert len(calls) == fp32_math
    assert out.dtype == torch.bfloat16
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    torch.testing.assert_close(out.float(), ref, atol = 2e-2, rtol = 2e-2)
