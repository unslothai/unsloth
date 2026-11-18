# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 on Diffusers follows the reference implementation's media handling, noise, schedule and sampler.

Every ``_ref_*`` helper below is the reference implementation's code, written out literally (with its own float32
arithmetic), and Studio's result is compared with it bit for bit. The Diffusers half runs the real modular blocks
without weights.
"""

from __future__ import annotations

import math
import os
import types

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
av = pytest.importorskip("av")

from core.inference import video_minimax_h3 as VM  # noqa: E402
from core.inference import video_minimax_h3_sampling as VS  # noqa: E402


# ---- the reference implementation, literally -------------------------------------------------------------------


def _ref_adapt_canvas(width, height):
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = 768 * ratio, 768
    else:
        nom_w, nom_h = 768, 768 / ratio
    if nom_w * nom_h > 768 * 1344:
        s = math.sqrt(768 * 1344 / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(32, round(nom_w / 32) * 32), max(32, round(nom_h / 32) * 32))


def _ref_resample(
    waveform,
    orig_freq,
    new_freq,
    lowpass_filter_width = 6,
    rolloff = 0.99,
):
    divisor = math.gcd(int(orig_freq), int(new_freq))
    orig_freq = int(orig_freq) // divisor
    new_freq = int(new_freq) // divisor
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype = waveform.dtype)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype = waveform.dtype)[:, None, None] / new_freq + idx
    t = (t * base_freq).clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, 1.0, t.sin() / t)
    kernel *= window * (base_freq / orig_freq)
    shape = waveform.shape
    length = shape[-1]
    waveform = waveform.reshape(-1, length)
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    output = torch.nn.functional.conv1d(waveform[:, None], kernel, stride = orig_freq)
    output = output.transpose(1, 2).reshape(waveform.shape[0], -1)
    target_length = math.ceil(np.float32(new_freq * length / orig_freq))
    output = output[..., :target_length]
    return output.reshape(*shape[:-1], output.shape[-1])


def _ref_audio_vae_input(waveform_cl, sample_rate):
    """(channels, samples) -> what the audio VAE encodes: resampled, centre-cropped to 800, two channels."""
    waveform = (
        _ref_resample(waveform_cl[None], sample_rate, 32000)
        if sample_rate != 32000
        else waveform_cl[None]
    )
    pixels = waveform[:1].movedim(1, -1)  # [1, L, C]
    length = pixels.shape[1]
    x = (length // 800) * 800
    if x != length:
        pixels = pixels.narrow(1, (length % 800) // 2, x)
    if pixels.shape[-1] > 2:
        pixels = pixels[..., :2]
    elif pixels.shape[-1] < 2:
        pixels = torch.nn.functional.pad(pixels, (0, 2 - pixels.shape[-1]), mode = "replicate")
    return pixels[0].transpose(0, 1)


def _ref_simple_sigmas(shift, steps):
    t = (torch.arange(1, 1001, 1) / 1000) * 1000
    table = shift * (t / 1000) / (1 + (shift - 1) * (t / 1000))
    ss = len(table) / steps
    return torch.FloatTensor([float(table[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])


def _ref_time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def _ref_res_multistep(model, x, sigmas):
    """k-diffusion res_multistep, eta 0 (no noise), as the reference samples."""
    sigma_fn = lambda t: t.neg().exp()  # noqa: E731
    t_fn = lambda sigma: sigma.log().neg()  # noqa: E731
    phi1_fn = lambda t: torch.expm1(t) / t  # noqa: E731
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t  # noqa: E731
    old_sigma_down = old_denoised = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i])
        sigma_down = sigmas[i + 1]
        if sigma_down == 0 or old_denoised is None:
            d = (x - denoised) / sigmas[i]
            x = x + d * (sigma_down - sigmas[i])
        else:
            t, t_old, t_next, t_prev = (
                t_fn(sigmas[i]),
                t_fn(old_sigma_down),
                t_fn(sigma_down),
                t_fn(sigmas[i - 1]),
            )
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan = 0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan = 0.0)
            x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised = denoised
        old_sigma_down = sigma_down
    return x


def _ref_patchify(latent):
    b, c, t, h, w = latent.shape
    x = latent.reshape(b, c, t, 1, h // 2, 2, w // 2, 2)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * (h // 2) * (w // 2), c * 4)


def _ref_video_block(frames):
    """[2, H, W, 3] float frames -> flattened patches, (grid_h, grid_w)."""
    imgs = frames.permute(0, 3, 1, 2)
    height, width = frames.shape[1], frames.shape[2]
    h_bar, w_bar = round(height / 32) * 32, round(width / 32) * 32
    if h_bar * w_bar > 12845056:
        beta = math.sqrt((height * width) / 12845056)
        h_bar = max(32, math.floor(height / beta / 32) * 32)
        w_bar = max(32, math.floor(width / beta / 32) * 32)
    elif h_bar * w_bar < 3136:
        beta = math.sqrt(3136 / (height * width))
        h_bar = math.ceil(height * beta / 32) * 32
        w_bar = math.ceil(width * beta / 32) * 32
    imgs = torch.nn.functional.interpolate(
        imgs, size = (h_bar, w_bar), mode = "bilinear", align_corners = False
    )
    imgs = (imgs - 0.5) / 0.5
    gh, gw = h_bar // 16, w_bar // 16
    patches = imgs.reshape(1, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16).permute(
        0, 3, 6, 4, 7, 2, 1, 5, 8
    )
    return patches.reshape(gh * gw, 3 * 2 * 16 * 16), (gh, gw)


# ---- media ---------------------------------------------------------------------------------------------------------


def _write_video(
    path,
    w,
    h,
    n,
    *,
    rotation = 0,
    audio_layout = None,
    audio_rate = 48000,
    audio_seconds = None,
):
    rng = np.random.default_rng(w + 3 * h)
    base = rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype = np.uint8)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate = 24)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        if rotation:
            stream.set_display_rotation(rotation)
        audio = (
            out.add_stream("aac", rate = audio_rate, layout = audio_layout) if audio_layout else None
        )
        for i in range(n):
            img = np.kron(np.roll(base, i, axis = 1), np.ones((8, 8, 1), np.uint8))[:h, :w]
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format = "rgb24")
            ):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        if audio is not None:
            channels = len(av.AudioLayout(audio_layout).channels)
            total = int((audio_seconds or n / 24) * audio_rate)
            t = np.arange(total) / audio_rate
            wave = np.stack(
                [0.3 * np.sin(2 * np.pi * (200 + 90 * c) * t) for c in range(channels)]
            ).astype(np.float32)
            for start in range(0, total, 1024):
                frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(wave[:, start : start + 1024]),
                    format = "fltp",
                    layout = audio_layout,
                )
                frame.sample_rate, frame.pts = audio_rate, start
                for packet in audio.encode(frame):
                    out.mux(packet)
            for packet in audio.encode():
                out.mux(packet)
    return path.read_bytes()


def _float_decode(blob):
    """Every frame as float RGB, the reference decode for yuv420p, plus each frame's display rotation."""
    import io
    with av.open(io.BytesIO(blob)) as container:
        return [(f.to_ndarray(format = "gbrpf32le"), f.rotation) for f in container.decode(video = 0)]


def test_rotated_video_is_decoded_upright(tmp_path):
    blob = _write_video(tmp_path / "rot.mp4", 640, 352, 48, rotation = 90)
    frames, _, _ = VM.decode_h3_reference_video(blob, decode_audio = False)
    raw = _float_decode(blob)
    assert raw[0][1] == 90
    # portrait: 352 wide, 640 tall, and a quarter turn counter-clockwise of the stored pixels
    assert frames[0].size == (352, 640)
    for k in (0, 23, 47):
        expect = np.rot90(VM.h3_float_rgb_to_uint8(raw[k][0]), k = 1, axes = (0, 1))
        assert np.array_equal(np.asarray(frames[k]), expect)


@pytest.mark.parametrize("rotation, turns", [(0, 0), (90, 1), (180, 2), (270, 3), (-90, 3)])
def test_rotation_quarter_turns(rotation, turns):
    assert VM.h3_rotation_quarter_turns(rotation) == turns
    pixels = np.arange(2 * 3 * 3, dtype = np.uint8).reshape(2, 3, 3)
    out = VM.rotate_h3_video_frame(Image.fromarray(pixels), rotation)
    assert np.array_equal(np.asarray(out), np.rot90(pixels, k = turns, axes = (0, 1)))


@pytest.mark.parametrize(
    "size", [(640, 352), (720, 400)]
)  # 720 is off the 32 px grid: padded for the conversion
def test_yuv_frames_go_through_float_rgb(tmp_path, size):
    blob = _write_video(tmp_path / "v.mp4", *size, 50)
    frames, _, _ = VM.decode_h3_reference_video(blob, decode_audio = False)
    raw = _float_decode(blob)
    expect = VM.h3_float_rgb_to_uint8(raw[5][0])
    got = np.asarray(frames[5])
    if size[0] % 32 == 0 and size[1] % 32 == 0:
        assert np.array_equal(got, expect)
    else:
        # fitted to the 32 px grid, so compare before the resize through the frame converter itself
        with av.open(__import__("io").BytesIO(blob)) as container:
            frame = list(container.decode(video = 0))[5]
            converted = np.asarray(VM.h3_video_frame_rgb(frame))
        assert converted.shape == (size[1], size[0], 3)
        assert np.abs(converted.astype(int) - expect.astype(int)).max() <= 1
    assert VM.h3_float_rgb_to_uint8(
        np.array([0.5 / 255, 1.5 / 255, 254.6 / 255], np.float32)
    ).tolist() == [0, 2, 255]


ASPECTS = [
    (1920, 1080),
    (1080, 1920),
    (1900, 1000),
    (1000, 1000),
    (640, 360),
    (2560, 1080),
    (800, 3000),
    (4000, 1001),
]


@pytest.mark.parametrize("source", ASPECTS)
def test_reference_video_canvas_rounds_each_axis(source):
    assert VM.h3_reference_video_canvas(*source) == _ref_adapt_canvas(*source)


def test_reference_video_canvas_example_and_no_upscale():
    # 1.9:1: each axis to its nearest 32 (1408x736), where a cap-first rule gave 1376x736.
    assert VM.h3_reference_frame_size(1900, 1000) == (1408, 736)
    # smaller than the canvas: kept, snapped to the grid, never upscaled
    assert VM.h3_reference_frame_size(640, 360) == (640, 352)


# ---- audio ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate, channels, seconds", [(48000, 2, 2.37), (44100, 6, 3.1), (22050, 1, 1.9), (32000, 8, 2.0)]
)
def test_reference_audio_is_conformed(rate, channels, seconds):
    rng = np.random.default_rng(rate + channels)
    wave = (0.2 * rng.standard_normal((int(rate * seconds), channels))).astype(np.float32)
    out, out_rate = VM.conform_h3_reference_audio(wave, rate)
    expect = _ref_audio_vae_input(torch.from_numpy(np.ascontiguousarray(wave.T)), rate)
    assert out_rate == 32000
    assert out.shape == (expect.shape[1], 2) and out.shape[0] % 800 == 0
    assert np.array_equal(out, expect.transpose(0, 1).numpy())
    if channels == 1:
        assert np.array_equal(out[:, 0], out[:, 1])


def test_resampler_is_torchaudios_default_kernel():
    x = torch.randn(3, 4410)
    assert torch.equal(VM.resample_h3_audio(x, 44100, 32000), _ref_resample(x, 44100, 32000))
    torchaudio = pytest.importorskip("torchaudio")
    expect = torchaudio.functional.resample(x, 44100, 32000)
    assert torch.allclose(VM.resample_h3_audio(x, 44100, 32000), expect, atol = 1e-6)


def test_output_audio_is_limited_and_trimmed():
    quiet = 0.05 * torch.randn(2, 170000)
    out = VM.h3_output_audio(quiet, 32000, 24, 124)
    assert torch.equal(out, quiet[:, :165334])
    loud = 0.9 * torch.randn(2, 170000)
    std = (
        torch.std(loud, dim = [0, 1], keepdim = True) * 5.0
    )  # the reference's divisor, over the whole decode
    assert torch.equal(VM.h3_output_audio(loud, 32000, 24, 124), (loud / std)[:, :165334])


def test_output_audio_ends_with_the_last_frame():
    assert VM.h3_output_audio_samples(32000, 24, 124) == math.ceil(32000 / 24 * 124) == 165334
    assert VM.h3_output_audio_samples(32000, 24, 120) == 160000


# ---- Diffusers setup step: small videos stay small, soundtracks stay whole --------------------------------------

blocks_mod = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")


def _pipeline():
    pipe = blocks_mod.MiniMaxH3Blocks().init_pipeline()
    VM.keep_h3_reference_image_sizes(pipe)
    VS.install_h3_reference_recipe(pipe)
    return pipe


def _find(pipe, cls_name):
    stack = [pipe._blocks]
    while stack:
        block = stack.pop()
        if any(c.__name__ == cls_name for c in type(block).__mro__):
            return block
        stack.extend((getattr(block, "sub_blocks", None) or {}).values())
    raise AssertionError(cls_name)


def _setup(
    pipe,
    references,
    frames = 124,
):
    from diffusers.modular_pipelines import PipelineState

    state = PipelineState()
    for key, value in (
        ("references", references),
        ("width", 960),
        ("height", 544),
        ("num_frames", frames),
    ):
        state.set(key, value)
    _, state = _find(pipe, "MiniMaxH3Ref2VASetupStep")(pipe, state)
    return state.get("normalized_references")


def _frames(
    w,
    h,
    n,
    seed = 0,
):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype = np.uint8)) for _ in range(n)]


def test_small_reference_video_is_not_upscaled_and_soundtrack_is_whole():
    pipe = _pipeline()
    frames = _frames(640, 352, 60)
    rng = np.random.default_rng(1)
    soundtrack = (0.1 * rng.standard_normal((32000 * 15, 2))).astype(np.float32)
    refs = VM.h3_diffusers_references(VM.MiniMaxH3References(videos = ((frames, soundtrack, 32000),)))
    (video,) = _setup(pipe, refs)
    assert video.frames.shape == (60, 352, 640, 3)
    assert np.array_equal(video.frames[7], np.asarray(frames[7]))
    # 15 s of sound for a 124-frame (5.2 s) render: encoded whole
    assert tuple(video.audio.shape) == (2, 32000 * 15)


def test_stock_setup_step_upscales_and_cuts():
    """What the switch prevents."""
    pipe = blocks_mod.MiniMaxH3Blocks().init_pipeline()
    frames = _frames(640, 352, 60)
    soundtrack = np.zeros((32000 * 15, 2), np.float32)
    refs = VM.h3_diffusers_references(VM.MiniMaxH3References(videos = ((frames, soundtrack, 32000),)))
    (video,) = _setup(pipe, refs)
    assert video.frames.shape[1:3] != (352, 640)
    assert video.audio.shape[-1] < 32000 * 15


# ---- vision input ----------------------------------------------------------------------------------------------


def _gather(pipe, references):
    proc = types.SimpleNamespace(
        image_processor = types.SimpleNamespace(patch_size = 16, merge_size = 2),
        video_processor = types.SimpleNamespace(temporal_patch_size = 2),
    )
    return _find(pipe, "MiniMaxH3Ref2VATextEncoderStep")._gather_vision_features(
        proc, references, 24.0
    )


def test_vision_frames_are_picked_after_the_snap_and_blocks_sized_per_frame():
    pipe = _pipeline()
    frames = _frames(640, 352, 100, seed = 4)
    (video,) = _setup(
        pipe, VM.h3_diffusers_references(VM.MiniMaxH3References(videos = ((frames, None, None),)))
    )
    vision, _, tokens, timestamps = _gather(pipe, [video])
    # 100 frames snap to 90; 2 fps reads 0, 12, ..., 84 (8 frames) -> 4 blocks, not the 5 the unsnapped clip gives
    assert vision["video_grid_thw"].tolist() == [[4, 22, 40]]
    assert tokens == [22 * 40 // 4]
    assert timestamps == [[0.25, 1.25, 2.25, 3.25]]
    expect = []
    pixels = torch.from_numpy(video.frames).float() / 255.0
    for start in range(0, 8, 2):
        idx = [12 * (start + j) for j in range(2)]
        expect.append(_ref_video_block(pixels[idx])[0])
    assert torch.equal(vision["pixel_values_videos"], torch.cat(expect))


def test_long_reference_keeps_full_size_blocks(monkeypatch):
    pipe = _pipeline()
    one = np.random.default_rng(5).integers(0, 256, (768, 1344, 3), dtype = np.uint8)
    frames = np.broadcast_to(one, (345, 768, 1344, 3))  # 345 frames without 1 GB of copies
    video = blocks_mod.MiniMaxH3VideoReference(frames = frames, fps = 24.0)
    vision, _, tokens, timestamps = _gather(pipe, [video])
    # 345 frames, 2 fps: 29 frames, 15 blocks of 1344x768, each 1008 tokens (a whole-clip budget shrinks them)
    assert vision["video_grid_thw"].tolist() == [[15, 48, 84]]
    assert tokens == [1008]
    monkeypatch.setenv(VS.H3_VISION_BUDGET_ENV, "clip")
    assert VS.h3_vision_per_block() is False


def test_image_references_use_the_reference_pixel_limits():
    pipe = _pipeline()
    small = Image.fromarray(np.random.default_rng(6).integers(0, 256, (64, 96, 3), dtype = np.uint8))
    vision, image_tokens, _, _ = _gather(pipe, [blocks_mod.MiniMaxH3ImageReference(image = small)])
    # 96x64 is above the 3136-pixel floor: kept, where the stock 65,536 floor upscaled it
    assert vision["image_grid_thw"].tolist() == [[1, 4, 6]]
    assert image_tokens == [6]


# ---- noise ---------------------------------------------------------------------------------------------------------


def test_noise_is_video_then_audio_from_one_seeded_cpu_generator():
    from diffusers.modular_pipelines import PipelineState

    pipe = _pipeline()
    step = _find(pipe, "MiniMaxH3PrepareLatentsStep")
    state = PipelineState()
    for key, value in (("num_latent_frames", 37), ("latent_height", 34), ("latent_width", 60),
                       ("num_audio_latents", 207), ("generator", torch.Generator().manual_seed(77))):  # fmt: skip
        state.set(key, value)
    components = types.SimpleNamespace(_execution_device = "cpu", vae_latent_channels = 24, audio_latent_channels = 32,
                                       audio_channels = 2, patch_size = (1, 2, 2))  # fmt: skip
    step(components, state)
    gen = torch.manual_seed(77)
    video = torch.randn([1, 24, 37, 34, 60], dtype = torch.float32, generator = gen, device = "cpu")
    audio = torch.randn([1, 32, 2, 207], dtype = torch.float32, generator = gen, device = "cpu")
    assert torch.equal(state.get("latents"), _ref_patchify(video))
    assert torch.equal(state.get("audio_latents"), audio[0].permute(1, 2, 0).reshape(2 * 207, 32))


def test_condition_noise_restarts_per_condition_in_row_shape():
    from diffusers.modular_pipelines import PipelineState

    pipe = _pipeline()
    step = _find(pipe, "MiniMaxH3PrepareConditionLatentsStep")
    conds = [torch.randn(1, 24, 1, 34, 60), torch.randn(1, 24, 7, 22, 40)]
    expect = []
    for z in conds:
        r = _ref_patchify(z.to(torch.float32))
        noise = torch.randn(
            r.shape, generator = torch.Generator("cpu").manual_seed(91), dtype = torch.float32
        )
        expect.append(0.999 * r + (1.0 - 0.999) * noise)
    expect = torch.cat(expect)
    state = PipelineState()
    for key, value in (("condition_latents", conds), ("generator", torch.Generator().manual_seed(91)),
                       ("num_condition_video_rows", int(expect.shape[0]))):  # fmt: skip
        state.set(key, value)
    step(
        types.SimpleNamespace(
            _execution_device = "cpu", patch_size = (1, 2, 2), keyframe_noise_aug = 0.999
        ),
        state,
    )
    assert torch.equal(state.get("condition_rows"), expect)


# ---- schedule and sampler -------------------------------------------------------------------------------------------


def _schedulers(
    sampler,
    shift_v = 12.0,
    shift_a = 3.0,
):
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    video, audio = MiniMaxH3Scheduler(shift = shift_v), MiniMaxH3Scheduler(shift = shift_a)
    assert VS.install_h3_schedulers(
        types.SimpleNamespace(scheduler = video, audio_scheduler = audio), sampler
    )
    return video, audio


@pytest.mark.parametrize("steps", [7, 8, 20, 30, 33])
def test_sigma_grid_and_timesteps(steps):
    video, audio = _schedulers(VS.H3_SAMPLER_RES_MULTISTEP)
    video.set_timesteps(steps + 1)
    audio.set_timesteps(steps + 1)
    sigmas = _ref_simple_sigmas(12.0, steps)
    assert torch.equal(video.sigmas, sigmas)
    model_sigma = ((sigmas[:-1] * 1000) / 1000.0).float().clamp(min = 1e-6)
    assert torch.equal(video.timesteps, 1.0 - model_sigma)
    assert torch.equal(audio.timesteps, 1.0 - _ref_time_shift_sigma(model_sigma, 12.0, 3.0))
    assert video.num_inference_steps == audio.num_inference_steps == steps


def test_sigma_grid_floors_when_steps_do_not_divide_1000():
    sigmas = VS.h3_simple_sigmas(1.0, 30)
    # without the shift, entry k is 1 - floor(k * 1000 / 30) / 1000
    assert sigmas[1].item() == pytest.approx(1 - 33 / 1000, abs = 1e-7)
    assert sigmas[-2].item() == pytest.approx(1 - 966 / 1000, abs = 1e-7)


def test_res_multistep_trajectory_equals_the_reference_for_video_and_carried_audio():
    torch.manual_seed(0)
    steps, scale = 20, 12.0 / 3.0
    sigmas = _ref_simple_sigmas(12.0, steps)
    xv0, xa0 = torch.randn(48, 96), torch.randn(30, 32)
    out_v = [torch.randn(48, 96) for _ in range(steps)]
    out_a = [torch.randn(30, 32) for _ in range(steps)]
    calls = []

    def model(x, sigma):
        # the reference model: velocity noise-ward, audio carried as (sigma_v / sigma_a) * x_a
        i = len(calls)
        y = x[:, 48 * 96 :].reshape(30, 32)
        sig_v = (sigma * 1000 / 1000.0).float().clamp(min = 1e-6)
        sig_a = _ref_time_shift_sigma(sig_v, 12.0, 3.0)
        x_a = y * (sig_a / sig_v)
        calls.append(x_a)
        dy = (1.0 - scale) * x_a + (1.0 + (scale - 1.0) * sig_a) * out_a[i]
        out = torch.cat([out_v[i].reshape(1, -1), dy.reshape(1, -1)], dim = 1)
        return x - out * sigma

    final = _ref_res_multistep(
        model, torch.cat([xv0.reshape(1, -1), xa0.reshape(1, -1)], dim = 1), sigmas
    )
    video, audio = _schedulers(VS.H3_SAMPLER_RES_MULTISTEP)
    video.set_timesteps(steps + 1)
    audio.set_timesteps(steps + 1)
    xv, xa = xv0.clone(), xa0.clone()
    for i in range(steps):
        assert torch.equal(xa, calls[i])
        # Diffusers' velocity is data-ward: the negative of the reference's
        xv = video.step(out_v[i].neg(), video.timesteps[i], xv, return_dict = False)[0]
        xa = audio.step(out_a[i].neg(), audio.timesteps[i], xa, return_dict = False)[0]
    assert torch.equal(xv, final[:, : 48 * 96].reshape(48, 96))
    assert torch.equal(xa, final[:, 48 * 96 :].reshape(30, 32) * (1.0 / scale))


def test_euler_mode_keeps_the_stock_step():
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    video, _ = _schedulers(VS.H3_SAMPLER_EULER)
    stock = MiniMaxH3Scheduler(shift = 12.0)
    video.set_timesteps(9)
    stock.set_timesteps(9, sigmas = video.sigmas)
    stock.timesteps = video.timesteps.clone()
    x, v = torch.randn(10, 8), torch.randn(10, 8)
    for i in range(3):
        a = video.step(v, video.timesteps[i], x, return_dict = False)[0]
        b = stock.step(v, stock.timesteps[i], x, return_dict = False)[0]
        assert torch.equal(a, b)
        x = a


def test_sampler_choice(monkeypatch):
    monkeypatch.delenv(VS.H3_SAMPLER_ENV, raising = False)
    assert VS.h3_sampler(pdd = False) == VS.H3_SAMPLER_RES_MULTISTEP
    assert VS.h3_sampler(pdd = True) == VS.H3_SAMPLER_EULER
    monkeypatch.setenv(VS.H3_SAMPLER_ENV, "euler")
    assert VS.h3_sampler(pdd = False) == VS.H3_SAMPLER_EULER


def test_install_is_idempotent_and_scoped():
    pipe = _pipeline()
    report = VS.install_h3_reference_recipe(pipe)
    assert report["MiniMaxH3PrepareLatentsStep"] >= 1
    assert report["MiniMaxH3Ref2VATextEncoderStep"] >= 1
    assert VS.install_h3_reference_recipe(types.SimpleNamespace(_blocks = None)) == {
        "MiniMaxH3PrepareLatentsStep": 0,
        "MiniMaxH3PrepareConditionLatentsStep": 0,
        "MiniMaxH3Ref2VATextEncoderStep": 0,
        "MiniMaxH3FL2VATextEncoderStep": 0,
        "MiniMaxH3KeyframeVaeEncoderStep": 0,
        "MiniMaxH3Ref2VAReferenceEncoderStep": 0,
        "sampler": None,
    }
    other = blocks_mod.MiniMaxH3Blocks().init_pipeline()
    assert (
        type(_find(other, "MiniMaxH3PrepareLatentsStep")).__name__ == "MiniMaxH3PrepareLatentsStep"
    )


# ---- keyframes -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", [(1200, 900), (1000, 1500), (1920, 1080), (1344, 768)])
@pytest.mark.parametrize("target", [(1344, 768), (960, 544), (768, 1344)])
def test_last_keyframe_centre_crop(source, target):
    pil = Image.fromarray(
        np.random.default_rng(source[0]).integers(0, 256, (source[1], source[0], 3), dtype = np.uint8)
    )
    got = VM.fit_h3_keyframe(pil, *target, anchor = VM.H3_ANCHOR_LAST)
    ow, oh = source
    old_aspect, new_aspect = ow / oh, target[0] / target[1]
    x = y = 0
    if old_aspect > new_aspect:
        x = round((ow - ow * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((oh - oh * (old_aspect / new_aspect)) / 2)
    expect = pil.crop((x, y, ow - x, oh - y)).resize(target, Image.LANCZOS)
    assert np.array_equal(np.asarray(got), np.asarray(expect))


def test_condition_encode_holds_cudnn_benchmark_off(monkeypatch):
    """A timed cuDNN search picks the conv algorithm per process; the condition encode must not depend on it."""
    from diffusers.modular_pipelines import PipelineState
    from diffusers.modular_pipelines.minimax_h3 import encoders

    pipe = _pipeline()
    seen = []
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(
        encoders,
        "encode_vae_condition",
        lambda *a, **k: seen.append(torch.backends.cudnn.benchmark) or torch.zeros(1),
    )
    step = _find(pipe, "MiniMaxH3KeyframeVaeEncoderStep")
    state = PipelineState()
    state.set("keyframes", [Image.fromarray(np.zeros((64, 64, 3), np.uint8))])
    step(types.SimpleNamespace(_execution_device = "cpu", vae = None, pixel_mean = (0, 0, 0), pixel_std = (1, 1, 1),
                               keyframe_encode_seed = 42), state)  # fmt: skip
    assert seen == [False]
    assert torch.backends.cudnn.benchmark is True
