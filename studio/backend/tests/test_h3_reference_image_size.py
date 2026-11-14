# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 ref2va reference images reach the VAE and the vision tower at the size Studio fitted.

The Diffusers half runs the real modular blocks (no weights): the Ref2VA setup step, the vision
features the text-encoder step builds, and the reference VAE encode step with its encoder stubbed
to record the pixels it is handed.
"""

from __future__ import annotations

import functools
import math
import types

import pytest

from core.inference import sd_cpp_args
from core.inference.video_minimax_h3 import (
    H3_REF_SIZE_MATCH,
    H3_REF_SIZE_MAX,
    fit_h3_reference_image,
    keep_h3_reference_image_sizes,
)

Image = pytest.importorskip("PIL.Image")


def _spec_size(w: int, h: int, gen_w: int, gen_h: int, policy: str) -> tuple[int, int]:
    """The reference sizing rule, written out independently of Studio's implementation."""
    if policy == H3_REF_SIZE_MATCH:
        scale = min(1.0, math.sqrt((gen_w * gen_h) / (w * h)))
    else:
        scale = min(1.0, 2048 / min(w, h))
    return (max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32))


def _sd_cpp_size(w: int, h: int, gen_w: int, gen_h: int, policy: str) -> tuple[int, int]:
    """sd.cpp's minimax_h3_reference_size (src/pipeline/video.cpp), line for line: std::round rounds
    halves away from zero, unlike Python's round."""
    if policy == H3_REF_SIZE_MAX:
        scale = min(1.0, 2048.0 / min(w, h))
    else:
        scale = min(1.0, math.sqrt((float(gen_w) * gen_h) / (float(w) * h)))
    align32 = lambda v: max(32, int(math.floor(v / 32.0 + 0.5)) * 32)  # noqa: E731
    return (align32(w * scale), align32(h * scale))


@functools.lru_cache(maxsize = None)
def _image(
    w: int,
    h: int,
    seed: int = 0,
) -> "Image.Image":
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype = np.uint8), "RGB")


SOURCES = [
    (6000, 4000),  # 3:2
    (4000, 5000),  # 4:5
    (3000, 3000),  # 1:1
    (4096, 2304),  # 16:9
    (2304, 4096),  # 9:16
    (800, 600),  # smaller than the generation area: never upscaled
    (640, 360),
    (1500, 1000),  # short edge under 2048: "max" keeps it
    (2048, 2048),
    (5120, 1280),  # 4:1
]
GENERATIONS = [(960, 544), (544, 960), (1344, 768), (832, 480)]


@pytest.mark.parametrize("policy", [H3_REF_SIZE_MATCH, H3_REF_SIZE_MAX])
@pytest.mark.parametrize("gen", GENERATIONS)
@pytest.mark.parametrize("source", SOURCES)
def test_fit_follows_the_sizing_rule_and_never_upscales(source, gen, policy):
    calls = []

    class Source:  # only the size matters here; test_fit_uses_lanczos covers the pixels
        size = source

        def resize(
            self,
            size,
            resample = None,
            **kw,
        ):
            calls.append(resample)
            return types.SimpleNamespace(size = size)

    fitted = fit_h3_reference_image(Source(), width = gen[0], height = gen[1], policy = policy)
    assert calls == [Image.LANCZOS]
    assert fitted.size == _spec_size(*source, *gen, policy)
    assert fitted.size[0] % 32 == 0 and fitted.size[1] % 32 == 0
    # The snap may round up by at most 16 px, never beyond the source.
    assert fitted.size[0] <= max(32, round(source[0] / 32) * 32)
    assert fitted.size[1] <= max(32, round(source[1] / 32) * 32)
    if policy == H3_REF_SIZE_MAX:
        assert min(fitted.size) <= 2048


def test_fit_uses_lanczos():
    source = _image(1200, 800, seed = 3)
    fitted = fit_h3_reference_image(source, width = 960, height = 544, policy = H3_REF_SIZE_MATCH)
    expected = source.resize(fitted.size, Image.LANCZOS)
    assert fitted.tobytes() == expected.tobytes()


# ---- Diffusers: the live Ref2VA blocks ---------------------------------------------------------

blocks_mod = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")


def _pipeline():
    return blocks_mod.MiniMaxH3Blocks().init_pipeline()


def _find(pipe, cls_name):
    stack = [pipe._blocks]
    while stack:
        block = stack.pop()
        if type(block).__name__ == cls_name or any(
            c.__name__ == cls_name for c in type(block).__mro__
        ):
            return block
        stack.extend((getattr(block, "sub_blocks", None) or {}).values())
    raise AssertionError(f"{cls_name} not in the pipeline")


def _run_setup(pipe, images, gen):
    from diffusers.modular_pipelines import PipelineState

    state = PipelineState()
    refs = [blocks_mod.MiniMaxH3ImageReference(image = image) for image in images]
    for key, value in (
        ("references", refs),
        ("width", gen[0]),
        ("height", gen[1]),
        ("num_frames", 124),
    ):
        state.set(key, value)
    setup = _find(pipe, "MiniMaxH3Ref2VASetupStep")
    _, state = setup(pipe, state)
    return state.get("normalized_references")


def _vision_token_counts(pipe, normalized):
    """Token counts the text-encoder step derives, through the hosted checkpoint's processor config."""
    transformers = pytest.importorskip("transformers")
    processor = types.SimpleNamespace(
        image_processor = transformers.Qwen2VLImageProcessor(
            size = {"longest_edge": 16777216, "shortest_edge": 65536},
            patch_size = 16,
            temporal_patch_size = 2,
            merge_size = 2,
            image_mean = [0.5, 0.5, 0.5],
            image_std = [0.5, 0.5, 0.5],
        )
    )
    step = _find(pipe, "MiniMaxH3Ref2VATextEncoderStep")
    out = step._gather_vision_features(processor, normalized, 24.0)
    return list(out[1]), [tuple(int(v) for v in g) for g in out[0]["image_grid_thw"]]


def _vae_input_sizes(pipe, normalized, monkeypatch):
    """(width, height) of every image the reference encode step hands the video VAE."""
    torch = pytest.importorskip("torch")
    from diffusers.modular_pipelines import PipelineState
    from diffusers.modular_pipelines.minimax_h3 import encoders

    seen = []

    def record(vae, pixels, *a, **k):
        seen.append((int(pixels.shape[-1]), int(pixels.shape[-2])))
        return torch.zeros(1)

    monkeypatch.setattr(encoders, "encode_vae_condition", record)
    components = types.SimpleNamespace(
        _execution_device = "cpu",
        audio_vae = types.SimpleNamespace(
            config = types.SimpleNamespace(latents_mean = [0.0] * 32, latents_std = [1.0] * 32)
        ),
        vae = None,
        pixel_mean = pipe.pixel_mean,
        pixel_std = pipe.pixel_std,
        keyframe_encode_seed = 0,
        audio_latent_channels = 32,
    )
    state = PipelineState()
    state.set("normalized_references", normalized)
    _find(pipe, "MiniMaxH3Ref2VAReferenceEncoderStep")(components, state)
    return seen


MULTI = [(6000, 4000), (4000, 5000), (3000, 3000), (800, 600)]


@pytest.mark.parametrize("policy", [H3_REF_SIZE_MATCH, H3_REF_SIZE_MAX])
def test_vae_and_vision_tower_see_the_fitted_images(policy, monkeypatch):
    gen = (960, 544)
    pipe = _pipeline()
    assert keep_h3_reference_image_sizes(pipe) == 1
    fitted = [
        fit_h3_reference_image(_image(*s, seed = i), width = gen[0], height = gen[1], policy = policy)
        for i, s in enumerate(MULTI)
    ]
    sizes = [image.size for image in fitted]
    stock_calls = []
    stock = type(pipe.image_processor).resize
    monkeypatch.setattr(
        type(pipe.image_processor),
        "resize",
        lambda self, image, *a, **k: stock_calls.append(image) or stock(self, image, *a, **k),
    )
    normalized = _run_setup(pipe, fitted, gen)
    assert stock_calls == []

    assert [ref.kind for ref in normalized] == ["image"] * len(fitted)
    # The very images Studio sized, pixel for pixel: no second resample.
    assert all(ref.image is image for ref, image in zip(normalized, fitted))
    assert _vae_input_sizes(pipe, normalized, monkeypatch) == sizes
    tokens, grids = _vision_token_counts(pipe, normalized)
    assert grids == [(1, h // 16, w // 16) for w, h in sizes]
    assert tokens == [(w // 32) * (h // 32) for w, h in sizes]
    # The processor's resize is only shadowed while the setup step runs.
    assert "resize" not in vars(pipe.image_processor)


def test_stock_diffusers_rescales_every_image_to_2048(monkeypatch):
    """What the switch prevents: a "match" reference comes back at the 2048 short edge."""
    gen = (960, 544)
    pipe = _pipeline()
    fitted = [
        fit_h3_reference_image(_image(*s, seed = i), width = gen[0], height = gen[1], policy = "match")
        for i, s in enumerate(MULTI)
    ]
    normalized = _run_setup(pipe, fitted, gen)
    assert [min(size) for size in _vae_input_sizes(pipe, normalized, monkeypatch)] == [2048] * 4


def test_max_on_large_sources_equals_diffusers_alone():
    """For sources with a short edge of at least 2048, "max" is the released 2048 encode exactly."""
    gen = (960, 544)
    sources = [_image(6000, 4000, seed = 1), _image(2304, 4096, seed = 2), _image(2048, 2048, seed = 4)]
    stock = _run_setup(_pipeline(), sources, gen)
    pipe = _pipeline()
    keep_h3_reference_image_sizes(pipe)
    fitted = [fit_h3_reference_image(s, width = gen[0], height = gen[1], policy = "max") for s in sources]
    ours = _run_setup(pipe, fitted, gen)
    for a, b in zip(stock, ours):
        assert a.image.size == b.image.size
        assert a.image.tobytes() == b.image.tobytes()


def test_switch_is_idempotent_and_scoped_to_the_pipeline():
    pipe = _pipeline()
    assert keep_h3_reference_image_sizes(pipe) == 1
    switched = type(_find(pipe, "MiniMaxH3Ref2VASetupStep"))
    assert keep_h3_reference_image_sizes(pipe) == 1
    assert type(_find(pipe, "MiniMaxH3Ref2VASetupStep")) is switched
    # Another pipeline built from the same blocks class keeps the stock step.
    other = _pipeline()
    assert type(_find(other, "MiniMaxH3Ref2VASetupStep")).__name__ == "MiniMaxH3Ref2VASetupStep"


def test_switch_ignores_pipelines_without_a_ref2va_step():
    assert keep_h3_reference_image_sizes(types.SimpleNamespace()) == 0
    assert keep_h3_reference_image_sizes(types.SimpleNamespace(_blocks = None)) == 0


def test_array_references_keep_diffusers_sizing():
    np = pytest.importorskip("numpy")
    pipe = _pipeline()
    keep_h3_reference_image_sizes(pipe)
    array = np.asarray(_image(640, 480))
    (normalized,) = _run_setup(pipe, [array], (960, 544))
    assert min(normalized.image.size) == 2048


def test_mixed_references_keep_order_and_only_images_skip_the_rescale():
    np = pytest.importorskip("numpy")
    from diffusers.modular_pipelines import PipelineState

    pipe = _pipeline()
    keep_h3_reference_image_sizes(pipe)
    first = fit_h3_reference_image(_image(6000, 4000), width = 960, height = 544, policy = "match")
    last = fit_h3_reference_image(_image(800, 600), width = 960, height = 544, policy = "match")
    frames = [Image.fromarray(np.zeros((360, 640, 3), np.uint8))] * 48
    fitted = [Image.fromarray(np.zeros((352, 640, 3), np.uint8))] * 48
    refs = [
        blocks_mod.MiniMaxH3ImageReference(image = first),
        blocks_mod.MiniMaxH3VideoReference(frames = frames, fps = 24.0),
        blocks_mod.MiniMaxH3ImageReference(image = last),
        blocks_mod.MiniMaxH3VideoReference(frames = fitted, fps = 24.0),
    ]
    state = PipelineState()
    for key, value in (("references", refs), ("width", 960), ("height", 544), ("num_frames", 124)):
        state.set(key, value)
    _, state = _find(pipe, "MiniMaxH3Ref2VASetupStep")(pipe, state)
    out = state.get("normalized_references")
    assert [ref.kind for ref in out] == ["image", "video", "image", "video"]
    assert out[0].image is first and out[2].image is last
    # Frames off the 32 px grid keep Diffusers' own canvas rule; Studio-fitted frames keep their size.
    assert out[1].frames.shape[1:3] == (768, 1344)
    assert out[3].frames.shape == (48, 352, 640, 3)


def test_resize_is_restored_when_the_step_raises():
    pipe = _pipeline()
    keep_h3_reference_image_sizes(pipe)
    with pytest.raises(ValueError):
        _run_setup(pipe, [_image(640, 128)], (960, 544))  # 5:1 is refused by the step
    assert "resize" not in vars(pipe.image_processor)


# ---- sd.cpp -------------------------------------------------------------------------------------


def test_sd_cpp_max_is_a_no_op_on_fitted_images_and_match_is_not():
    """Why sd.cpp gets its "max" mode: re-applying its "match" rule to a fitted image resizes it again."""
    resized_again = 0
    for gen in GENERATIONS:
        for w in range(256, 6001, 37):
            for h in range(max(256, w // 4), min(6001, 4 * w + 1), 41):
                for policy in (H3_REF_SIZE_MATCH, H3_REF_SIZE_MAX):
                    fitted = _spec_size(w, h, *gen, policy)
                    assert _sd_cpp_size(*fitted, *gen, H3_REF_SIZE_MAX) == fitted
                    if policy == H3_REF_SIZE_MATCH:
                        resized_again += _sd_cpp_size(*fitted, *gen, H3_REF_SIZE_MATCH) != fitted
    assert resized_again > 0


def _video_params(**kw):
    return sd_cpp_args.SdCppVideoGenParams(prompt = "p", width = 960, height = 544, num_frames = 124, **kw)


def _video_cmd(params):
    files = sd_cpp_args.SdCppModelFiles(
        diffusion_model = "/m/dit.gguf", vae = "/m/vae.gguf", llm = "/m/llm.gguf"
    )
    return sd_cpp_args.build_sd_cpp_video_command("sd-cli", files, params, output_path = "/o/out.avi")


def test_sd_cpp_cli_takes_reference_images_as_sized():
    cmd = _video_cmd(_video_params(ref_images = ("/r/a.png", "/r/b.png")))
    assert cmd[cmd.index("--ref-image-args") + 1] == "ref_image_size=max"
    assert cmd.count("--ref-image-args") == 1
    assert "--ref-image-args" not in _video_cmd(_video_params())
    assert "--ref-image-args" not in _video_cmd(
        _video_params(ref_audios = ("/r/a.wav",), ref_videos = ("/r/v",))
    )


def test_sd_cpp_server_takes_reference_images_as_sized():
    req = sd_cpp_args.build_vid_gen_request(
        _video_params(ref_images = ("/r/a.png",)), ref_images_b64 = ["AAA"]
    )
    assert req["ref_image_args"] == "ref_image_size=max"
    assert "ref_image_args" not in sd_cpp_args.build_vid_gen_request(_video_params())
