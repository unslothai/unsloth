# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastWan2.2-TI2V-5B-FullAttn: the dense-attention DMD student loads as the Wan2.2 5B family, samples its
three trained timesteps without CFG through the stock Wan pipeline loop, and the sparse-attention FastWan
releases stay refused. The sampler is checked end to end on a tiny random Wan pipeline against an fp64 oracle."""

from __future__ import annotations

import pytest

from core.inference import video_families as vf

_FULLATTN = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"


@pytest.mark.parametrize(
    "identifier",
    [
        _FULLATTN,
        "/models/FastWan2.2_TI2V_5B_FullAttn",
        "fastwan2.2-ti2v-5b-fullattn-diffusers",
    ],
)
def test_fullattn_student_loads_as_wan22_5b(identifier):
    assert vf.unsupported_video_checkpoint_reason(identifier) is None
    fam = vf.detect_video_family(identifier)
    assert fam is not None and fam.name == "wan2.2-ti2v-5b"
    assert vf.default_video_generation_params(
        identifier, fallback = (fam.default_steps, fam.default_guidance), family = fam.name
    ) == (3, 1.0)


@pytest.mark.parametrize(
    "identifier, family",
    [
        ("FastVideo/FastWan2.1-T2V-1.3B-Diffusers", "wan2.1-t2v-1.3b"),
        ("/models/FastWan2.1-T2V-1.3B-Diffusers/", "wan2.1-t2v-1.3b"),
    ],
)
def test_dense_13b_student_loads_as_wan21_13b(identifier, family):
    from core.inference import video_dmd

    assert vf.unsupported_video_checkpoint_reason(identifier) is None
    fam = vf.detect_video_family(identifier)
    assert fam is not None and fam.name == family
    assert video_dmd.dmd_timesteps(identifier) == (1000, 757, 522)
    assert vf.default_video_generation_params(
        identifier, fallback = (fam.default_steps, fam.default_guidance), family = fam.name
    ) == (3, 1.0)


@pytest.mark.parametrize(
    "identifier",
    [
        "FastVideo/FastWan2.1-T2V-14B-Diffusers",
        "FastWan2.1-T2V-1.3B-Q8_0.gguf",
        "FastWan2.1-T2V-1.3B_bf16.safetensors",
        "FastWan2.2-TI2V-5B-FullAttn-Q4_K_M.gguf",
        "someone/FastWan2.2-TI2V-5B-FullAttn-GGUF",
        "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        "FastVideo/FastWan-QAD-1.3B",
        "FastVideo/FastWan2.2-TI2V-5B-FullAttention-VSA",
    ],
)
def test_sparse_attention_fastwan_stays_refused(identifier):
    message = vf.unsupported_video_checkpoint_reason(identifier)
    assert message is not None and "FastVideo FastWan checkpoint" in message
    assert vf.detect_video_family(identifier) is None


def test_base_wan_defaults_unchanged():
    fam = vf.detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert vf.default_video_generation_params(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        fallback = (fam.default_steps, fam.default_guidance),
        family = fam.name,
    ) == (20, 5.0)
    # An owner or folder named fastwan is not a FastWan checkpoint.
    assert vf.default_video_generation_params(
        "/home/fastwan/models/Wan2.2-TI2V-5B-Diffusers",
        fallback = (fam.default_steps, fam.default_guidance),
        family = fam.name,
    ) == (20, 5.0)


def test_student_never_takes_the_base_hosted_denoiser():
    from core.inference.diffusion_families import family_prequant_repo

    fam = vf.detect_video_family(_FULLATTN)
    for scheme in ("int8", "fp8", "nvfp4"):
        assert family_prequant_repo(fam, scheme, base_repo = _FULLATTN) is None
    assert family_prequant_repo(fam, "fp8", base_repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers")


def test_dmd_recipe_and_sigmas():
    from core.inference import video_dmd

    assert video_dmd.dmd_timesteps("Wan-AI/Wan2.2-TI2V-5B-Diffusers") is None
    assert video_dmd.dmd_timesteps(None, _FULLATTN) == (1000, 757, 522)
    # Training table: s = k / 1000, sigma = 8 s / (1 + 7 s); nearest entries to t / 1000 are s = 0.28 and 0.12.
    got = video_dmd.dmd_sigmas((1000, 757, 522))
    assert got == pytest.approx([1.0, 28 / 37, 12 / 23], abs = 1e-12)


def _tiny_wan_pipe(expand_timesteps: bool):
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from diffusers import (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )

    torch.manual_seed(0)
    transformer = WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 12,
        in_channels = 16,
        out_channels = 16,
        text_dim = 32,
        freq_dim = 256,
        ffn_dim = 32,
        num_layers = 2,
        cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads",
        rope_max_seq_len = 32,
    ).eval()
    vae = AutoencoderKLWan(
        base_dim = 3,
        z_dim = 16,
        dim_mult = [1, 1, 1, 1],
        num_res_blocks = 1,
        temperal_downsample = [False, True, True],
    ).eval()
    scheduler = UniPCMultistepScheduler(
        prediction_type = "flow_prediction", use_flow_sigmas = True, flow_shift = 5.0
    )
    pipe = WanPipeline(
        tokenizer = None,
        text_encoder = None,
        transformer = transformer,
        vae = vae,
        scheduler = scheduler,
        expand_timesteps = expand_timesteps,
    )
    del diffusers
    return pipe


@pytest.mark.parametrize("expand_timesteps", [False, True])
def test_dmd_sampling_through_the_stock_wan_loop_matches_fp64_oracle(expand_timesteps):
    torch = pytest.importorskip("torch")
    from core.inference import video_dmd

    pipe = _tiny_wan_pipe(expand_timesteps)
    original = pipe.scheduler
    prompt_embeds = torch.randn(1, 8, 32, generator = torch.Generator().manual_seed(1))
    latents = torch.randn(1, 16, 2, 8, 8, generator = torch.Generator().manual_seed(2))
    timesteps, seed = (1000, 757, 522), 1234
    video_dmd.engaged_counts(reset = True)
    calls = []
    real_forward = pipe.transformer.forward

    def spy(*args, **kwargs):
        calls.append(kwargs["timestep"].detach().clone())
        return real_forward(*args, **kwargs)

    pipe.transformer.forward = spy
    with torch.no_grad(), video_dmd.dmd_sampling(pipe, timesteps, seed):
        out = pipe(
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = None,
            guidance_scale = 1.0,
            num_inference_steps = 3,
            height = 64,
            width = 64,
            num_frames = 5,
            latents = latents.clone(),
            output_type = "latent",
        ).frames
    pipe.transformer.forward = real_forward
    assert pipe.scheduler is original
    assert video_dmd.engaged_counts() == {"renders": 1, "steps": 3}
    # One forward per trained timestep (no CFG branch), each at the raw trained timestep.
    assert len(calls) == 3
    assert [float(c.flatten()[0]) for c in calls] == [1000.0, 757.0, 522.0]

    # Oracle: fp64 DMD update with hand-derived sigmas and the same CPU noise stream.
    sigmas = [1.0, 28 / 37, 12 / 23]
    gen = torch.Generator().manual_seed(video_dmd.renoise_seed(seed))
    x = latents.clone()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            ts = torch.full((1,), float(t))
            if expand_timesteps:
                ts = ts.expand(1, (x.shape[2] * x.shape[3] * x.shape[4]) // 4)
            v = real_forward(
                hidden_states = x.to(torch.float32),
                timestep = ts,
                encoder_hidden_states = prompt_embeds,
                return_dict = False,
            )[0]
            x0 = x.double() - sigmas[i] * v.double()
            if i + 1 < len(timesteps):
                eps = torch.randn(tuple(x.shape), generator = gen, dtype = torch.float32).double()
                x = ((1 - sigmas[i + 1]) * x0 + sigmas[i + 1] * eps).float()
            else:
                x = x0.float()
    torch.testing.assert_close(out, x, rtol = 1e-5, atol = 1e-5)


def test_dmd_scheduler_reseeds_per_render():
    torch = pytest.importorskip("torch")
    from core.inference import video_dmd

    sched = video_dmd.make_dmd_scheduler((1000, 757, 522), 7)
    sample = torch.zeros(1, 4, 2, 2)
    pred = torch.ones(1, 4, 2, 2)

    def render():
        sched.set_timesteps(3)
        x = sample
        for t in sched.timesteps:
            x = sched.step(pred, t, x, return_dict = False)[0]
        return x

    torch.testing.assert_close(render(), render(), rtol = 0, atol = 0)
    assert sched.sigmas.shape[0] == 4 and float(sched.sigmas[-1]) == 0.0
    sched.set_shift(12.0)  # a request flow shift must not move the trained levels
    assert video_dmd.dmd_sigmas((1000, 757, 522))[1] == pytest.approx(28 / 37)


@pytest.mark.parametrize("seed", [0, 7, 42, 2**31 - 1])
def test_renoise_stream_is_independent_of_the_initial_latent_noise(seed):
    """On CPU / MPS the pipeline draws its initial latent from a CPU generator seeded with the request seed; the
    first re-noise draw must not repeat it (same shape, same dtype)."""
    torch = pytest.importorskip("torch")
    from core.inference import video_dmd

    shape = (1, 16, 2, 8, 8)
    initial = torch.randn(shape, generator = torch.Generator().manual_seed(seed), dtype = torch.float32)
    sched = video_dmd.make_dmd_scheduler((1000, 757, 522), seed)
    zeros = torch.zeros(shape)
    out = sched.step(zeros, sched.timesteps[0], zeros, return_dict = False)[0]
    noise = out / sched._dmd_sigmas[1]
    assert not torch.allclose(noise, initial)
    assert abs(float(torch.corrcoef(torch.stack([noise.flatten(), initial.flatten()]))[0, 1])) < 0.2
    # Still deterministic per request seed (an OOM retry redraws the same noise).
    assert video_dmd.renoise_seed(seed) == video_dmd.renoise_seed(seed) != seed
