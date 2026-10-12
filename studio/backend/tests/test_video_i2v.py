# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image-to-video: the family table, the request rules, the detector refusals, and the class swap
that serves a first frame through the loaded pipeline (video_i2v)."""

from __future__ import annotations

import pytest

from core.inference.video_families import (
    _FAMILIES,
    detect_video_family,
    first_frame_canvas,
    validate_video_keyframe_conditioning,
    video_checkpoint_refusal,
    video_family_accepts_first_frame,
    video_family_accepts_last_frame,
    video_family_uses_i2v_pipeline,
    video_i2v_call_kwargs,
)


def _fam(name):
    fam = detect_video_family("", override = name)
    assert fam is not None, name
    return fam


# ── family table ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,i2v_class",
    [
        ("wan2.2-ti2v-5b", "WanImageToVideoPipeline"),
        ("ltx-2", "LTX2ImageToVideoPipeline"),
    ],
)
def test_declared_families_take_a_first_frame_through_their_i2v_pipeline(name, i2v_class):
    fam = _fam(name)
    assert fam.i2v_pipeline_class == i2v_class
    assert video_family_accepts_first_frame(fam)
    assert video_family_uses_i2v_pipeline(fam)
    assert not video_family_accepts_last_frame(fam)


def test_h3_keeps_native_keyframes_and_needs_no_i2v_pipeline():
    fam = _fam("minimax-h3")
    assert video_family_accepts_first_frame(fam) and video_family_accepts_last_frame(fam)
    assert not video_family_uses_i2v_pipeline(fam)


@pytest.mark.parametrize("name", ["wan2.2-t2v-a14b", "hunyuanvideo-1.5", "hunyuanvideo-1.5-720p"])
def test_text_only_families_take_no_frame(name):
    fam = _fam(name)
    assert fam.i2v_pipeline_class is None
    assert not video_family_accepts_first_frame(fam)
    assert not video_family_accepts_last_frame(fam)


def test_i2v_class_is_never_the_t2v_class():
    # A family naming its own t2v class as the i2v one would "swap" to the same class and pass the image to a
    # __call__ that ignores it.
    for fam in _FAMILIES:
        if fam.i2v_pipeline_class:
            assert fam.i2v_pipeline_class != fam.pipeline_class
            assert fam.i2v_pipeline_class.endswith("Pipeline")


# ── request validation ──────────────────────────────────────────────────────────────────────────


def test_first_frame_accepted_only_where_declared():
    validate_video_keyframe_conditioning(_fam("wan2.2-ti2v-5b"), None, has_keyframes = True)
    validate_video_keyframe_conditioning(_fam("ltx-2"), None, has_keyframes = True)
    for name in ("wan2.2-t2v-a14b", "hunyuanvideo-1.5"):
        with pytest.raises(ValueError, match = "no first or last frame"):
            validate_video_keyframe_conditioning(_fam(name), None, has_keyframes = True)


def test_last_frame_refused_by_first_frame_only_families():
    for name in ("wan2.2-ti2v-5b", "ltx-2"):
        with pytest.raises(ValueError, match = "first frame only"):
            validate_video_keyframe_conditioning(
                _fam(name), None, has_keyframes = True, has_last_frame = True
            )
    validate_video_keyframe_conditioning(
        _fam("minimax-h3"), "fl2va", has_keyframes = True, has_last_frame = True
    )


def test_text_to_video_requests_are_never_judged():
    for fam in _FAMILIES:
        validate_video_keyframe_conditioning(fam, None, has_keyframes = False)


# ── i2v defaults ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,aspect,expected",
    [
        ("wan2.2-ti2v-5b", (1920, 1080), (1280, 704)),
        ("wan2.2-ti2v-5b", (832, 1104), (704, 1280)),
        # Square is equally far from both TI2V presets; the family default (first) wins the tie.
        ("wan2.2-ti2v-5b", (512, 512), (1280, 704)),
        ("ltx-2", (1920, 1080), (1216, 704)),
        ("ltx-2", (3, 2), (768, 512)),
        ("ltx-2", (2, 3), (512, 768)),
        ("ltx-2", (9, 16), (704, 1216)),
    ],
)
def test_first_frame_canvas_is_the_nearest_aspect_preset(name, aspect, expected):
    assert first_frame_canvas(_fam(name), *aspect) == expected


def test_first_frame_canvas_rejects_a_degenerate_image():
    with pytest.raises(ValueError):
        first_frame_canvas(_fam("ltx-2"), 0, 10)


def test_ltx23_takes_comfyui_image_crf_and_ltx20_keeps_the_pipeline_default():
    fam = _fam("ltx-2")
    assert video_i2v_call_kwargs(fam, "ltx-2.3-22b-distilled", None) == {"image_crf": 18}
    assert video_i2v_call_kwargs(fam, None, "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf") == {
        "image_crf": 18
    }
    assert video_i2v_call_kwargs(fam, "Lightricks/LTX-2", "ltx-2-19b-dev.safetensors") == {}
    assert video_i2v_call_kwargs(_fam("wan2.2-ti2v-5b"), "Wan-AI/Wan2.2-TI2V-5B-Diffusers") == {}


# ── detector refusals ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ids",
    [
        ("Lightricks/LTX-Video",),
        ("Lightricks/LTX-Video-0.9.7-dev",),
        ("Lightricks/LTX-Video-0.9.8-13B-distilled",),
        ("a-r-r-o-w/LTX-Video-0.9.1-diffusers",),
        ("city96/LTX-Video-gguf", "ltx-video-2b-v0.9-Q4_K_M.gguf"),
        ("/models", "ltxv-13b-0.9.7-dev-Q4_K_M.gguf"),
    ],
)
def test_ltx_video_09_is_refused_not_built_as_ltx2(ids):
    fam = detect_video_family("/".join(ids))
    assert fam is not None and fam.name == "ltx-2"  # what the detector claims it as
    message = video_checkpoint_refusal(fam, *ids)
    assert message and "LTX-Video 0.9" in message


@pytest.mark.parametrize(
    "ids",
    [
        ("Lightricks/LTX-2",),
        ("Lightricks/LTX-2.3",),
        ("unsloth/LTX-2.3-GGUF", "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf"),
        ("unsloth/LTX-2-FP8",),
        ("Lightricks/LTX-Video-2",),
        # A folder named after the old model holding an LTX-2.3 file: the filename names LTX-2.
        ("/models/ltx-video", "ltx-2.3-22b-dev-Q4_K_M.gguf"),
    ],
)
def test_ltx2_checkpoints_are_not_refused(ids):
    assert video_checkpoint_refusal(_fam("ltx-2"), *ids) is None


@pytest.mark.parametrize(
    "repo,family",
    [
        ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v", "hunyuanvideo-1.5"),
        ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v", "hunyuanvideo-1.5-720p"),
        (
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
            "hunyuanvideo-1.5",
        ),
        ("someone/hunyuanvideo1.5_i2v_480p-GGUF", "hunyuanvideo-1.5"),
    ],
)
def test_hv15_i2v_weights_are_refused_by_the_t2v_family(repo, family):
    fam = detect_video_family(repo)
    assert fam is not None and fam.name == family
    message = video_checkpoint_refusal(fam, repo)
    assert message and "image-to-video" in message


@pytest.mark.parametrize(
    "repo",
    [
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "unsloth/Wan2.2-TI2V-5B-GGUF",
        "MiniMaxAI/MiniMax-H3",
    ],
)
def test_supported_checkpoints_are_not_refused(repo):
    fam = detect_video_family(repo)
    assert fam is not None
    assert video_checkpoint_refusal(fam, repo) is None


def test_load_validation_refuses_before_any_load():
    from core.inference.video import VideoBackend

    backend = VideoBackend()
    with pytest.raises(ValueError, match = "LTX-Video 0.9"):
        backend.validate_load_request("Lightricks/LTX-Video", model_kind = "pipeline")
    with pytest.raises(ValueError, match = "image-to-video checkpoint"):
        backend.validate_load_request(
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v", model_kind = "pipeline"
        )


# ── the class swap over a real (tiny) diffusers Wan pipeline ────────────────────────────────────


def _tiny_wan_pipe():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "WanImageToVideoPipeline"):
        pytest.skip("diffusers without WanImageToVideoPipeline")
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLWan(
        base_dim = 3,
        z_dim = 16,
        dim_mult = [1, 1, 1, 1],
        num_res_blocks = 1,
        temperal_downsample = [False, True, True],
    )
    torch.manual_seed(0)
    transformer = diffusers.WanTransformer3DModel(
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
    )
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift = 7.0)
    pipe = diffusers.WanPipeline(
        tokenizer = None,
        text_encoder = None,
        vae = vae,
        scheduler = scheduler,
        transformer = transformer,
        expand_timesteps = True,  # TI2V-5B's config
    )
    return torch, pipe


def _render(torch, pipe, **extra):
    gen = torch.Generator().manual_seed(7)
    embeds = torch.Generator().manual_seed(3)
    prompt_embeds = torch.randn(1, 8, 32, generator = embeds)
    return pipe(
        prompt_embeds = prompt_embeds,
        height = 32,
        width = 32,
        num_frames = 5,
        num_inference_steps = 2,
        guidance_scale = 1.0,
        generator = gen,
        output_type = "np",
        **extra,
    ).frames[0]


def test_class_swap_runs_i2v_over_the_same_modules_and_restores_t2v():
    from PIL import Image

    from core.inference.video_i2v import first_frame_pipeline

    torch, pipe = _tiny_wan_pipe()
    t2v_cls = type(pipe)
    before_dict = dict(pipe.__dict__)
    params_before = {
        n: (p.data_ptr(), p.dtype, p.device) for n, p in pipe.transformer.named_parameters()
    }
    # Studio state set on the pipeline instance must be visible to the i2v call and survive it.
    pipe._unsloth_marker = "kept"

    t2v_a = _render(torch, pipe)

    image = Image.new("RGB", (32, 32), (200, 40, 40))
    with first_frame_pipeline(pipe, "WanImageToVideoPipeline") as i2v:
        assert i2v is pipe  # the same object: nothing was copied
        assert type(pipe).__name__ == "WanImageToVideoPipeline"
        assert pipe._unsloth_marker == "kept"
        assert pipe.config.expand_timesteps is True
        i2v_out = _render(torch, pipe, image = image)
        pipe._unsloth_written_during_i2v = 1

    assert type(pipe) is t2v_cls
    # The overlay (config, processors, image-encoder slots) is gone; the render's own writes stay.
    assert "image_encoder" not in pipe.__dict__ and "image_processor" not in pipe.__dict__
    assert pipe.__dict__["_internal_dict"] is before_dict["_internal_dict"]
    assert pipe.__dict__["video_processor"] is before_dict["video_processor"]
    assert pipe._unsloth_written_during_i2v == 1
    for key in ("transformer", "vae", "scheduler"):
        assert pipe.__dict__[key] is before_dict[key]
    assert {
        n: (p.data_ptr(), p.dtype, p.device) for n, p in pipe.transformer.named_parameters()
    } == params_before

    # Text-to-video after an image-to-video render is bit-identical to before it.
    t2v_b = _render(torch, pipe)
    assert (t2v_a == t2v_b).all()
    assert not (i2v_out == t2v_a).all()


def test_class_swap_restores_on_failure():
    from core.inference.video_i2v import first_frame_pipeline

    _torch, pipe = _tiny_wan_pipe()
    t2v_cls = type(pipe)
    keys = set(pipe.__dict__)
    with pytest.raises(RuntimeError, match = "boom"):
        with first_frame_pipeline(pipe, "WanImageToVideoPipeline"):
            raise RuntimeError("boom")
    assert type(pipe) is t2v_cls
    assert set(pipe.__dict__) == keys


def test_class_swap_is_a_noop_without_a_class():
    from core.inference.video_i2v import first_frame_pipeline
    sentinel = object()
    with first_frame_pipeline(sentinel, None) as got:
        assert got is sentinel


def test_from_pipe_would_upcast_which_is_why_it_is_not_used():
    """Pins the reason video_i2v does not call DiffusionPipeline.from_pipe: without torch_dtype it
    casts every shared module to float32, an extra copy of the weights and a different numeric
    path. If diffusers changes that, the swap is still correct; this test only documents it."""
    torch, pipe = _tiny_wan_pipe()
    import diffusers

    pipe.transformer.to(torch.bfloat16)
    diffusers.WanImageToVideoPipeline.from_pipe(pipe)
    if next(pipe.transformer.parameters()).dtype == torch.bfloat16:
        pytest.skip("this diffusers from_pipe no longer upcasts")
    assert next(pipe.transformer.parameters()).dtype == torch.float32


def test_status_schema_carries_the_per_frame_support_and_the_canvas_rule():
    from models.inference import VideoGenerationDefaults, VideoStatusResponse

    fields = VideoStatusResponse.model_fields
    assert "supports_first_frame" in fields and "supports_last_frame" in fields
    assert VideoGenerationDefaults.model_fields["canvas_from_presets"].default is False


def test_request_last_frame_route_check_names_first_frame_only():
    fam = _fam("wan2.2-ti2v-5b")
    with pytest.raises(ValueError, match = "first frame only"):
        validate_video_keyframe_conditioning(fam, None, has_keyframes = True, has_last_frame = True)


def test_an_unrecorded_render_never_seeds_the_background_compile():
    from core.inference import diffusion_bg_compile as bgc

    bg = bgc.BackgroundCompile(object())
    with bgc.force_eager(), bgc.no_record():
        assert bg._record(None, (), {}) is None
    assert bg.samples == [] and bg.state == "recording"


def test_a_component_assigned_mid_swap_stays_registered_after_it():
    import diffusers

    from core.inference.video_i2v import first_frame_pipeline

    _torch, pipe = _tiny_wan_pipe()
    with first_frame_pipeline(pipe, "WanImageToVideoPipeline"):
        pipe.scheduler = diffusers.UniPCMultistepScheduler()
    assert type(pipe).__name__ == "WanPipeline"
    assert tuple(pipe.config["scheduler"])[1] == "UniPCMultistepScheduler"
    assert isinstance(pipe.scheduler, diffusers.UniPCMultistepScheduler)
    assert "image_encoder" not in pipe.config


def test_availability_probe_never_raises():
    from core.inference.video_i2v import i2v_pipeline_available
    assert i2v_pipeline_available(None) is False
    assert i2v_pipeline_available("NoSuchPipelineAnywhere") is False
