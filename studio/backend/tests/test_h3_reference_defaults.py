# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 video defaults follow the reference implementation's defaults.

Each ``_ref_*`` helper is the reference implementation's code written out literally, and Studio's default is compared
with it bit for bit: the output 8-bit quantisation, the dense time-embedding dtype, the AdaLN curve, the keyframe
canvas when no size is given, the VAE decoder and the page defaults of the reference workflow templates.
"""

from __future__ import annotations

import base64
import io
import math
import types

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from core.inference import video_minimax_h3_adaln as VA  # noqa: E402
from core.inference import video_minimax_h3_sampling as VS  # noqa: E402
from core.inference.video_families import detect_video_family  # noqa: E402


# ---- the reference implementation, literally -------------------------------------------------------------------


def _ref_save_bytes(frames):
    # the reference video save: (frame * 255).clamp(0, 255).byte()
    return (frames * 255).clamp(0, 255).byte()


class _RefTimeEmbedder(torch.nn.Module):
    def __init__(self, freq_dim, hidden, out):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = torch.nn.Linear(freq_dim, hidden, bias = True)
        self.proj_out = torch.nn.Linear(hidden, out, bias = True)

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype = torch.float32) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim = -1)
        return self.proj_out(torch.nn.functional.silu(self.proj_in(emb)))


def _ref_adaln(linear, t_emb, expand, modalities, hidden, apply_silu):
    x = linear(torch.nn.functional.silu(t_emb) if apply_silu else t_emb)
    x = x.view(x.shape[0] * modalities, expand * hidden)
    return x.chunk(expand, dim = -1)


def _ref_curve_rows(table, unique_t):
    t_vals = torch.tensor(unique_t, dtype = torch.float32)
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max = table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


# ---- output frames ---------------------------------------------------------------------------------------------


def _processor_pipe():
    from diffusers.video_processor import VideoProcessor
    return types.SimpleNamespace(
        video_processor = VideoProcessor(vae_scale_factor = 16, do_normalize = False)
    )


def _decoded(seed = 0):
    g = torch.Generator().manual_seed(seed)
    video = torch.rand(1, 3, 5, 24, 40, generator = g)
    video[0, :, 0, 0, :5] = torch.tensor([0.0, 1.0, 0.5 / 255, 1.5 / 255, 254.6 / 255])
    return video


def test_output_frames_are_truncated_like_the_reference_save():
    pytest.importorskip("diffusers")
    pipe = _processor_pipe()
    assert VS.install_h3_output_truncation(pipe)
    assert VS.install_h3_output_truncation(pipe)  # idempotent: one wrapper
    assert not hasattr(pipe.video_processor.postprocess.__wrapped__, "__wrapped__")
    video = _decoded()
    expect = _ref_save_bytes(video[0].permute(1, 2, 3, 0))
    proc = pipe.video_processor
    np_frames = proc.postprocess_video(video, output_type = "np")[0]
    assert torch.equal(torch.from_numpy((np_frames * 255).round().astype(np.uint8)), expect)
    pil = proc.postprocess_video(video, output_type = "pil")[0]
    assert torch.equal(torch.from_numpy(np.stack([np.asarray(f) for f in pil])), expect)
    pt = proc.postprocess_video(video, output_type = "pt")[0]
    assert torch.equal((pt.permute(0, 2, 3, 1) * 255).round().to(torch.uint8), expect)


def test_every_truncated_level_survives_a_later_rounding():
    levels = torch.arange(256, dtype = torch.float32) / 255
    for nudge in (0.0, 0.4999 / 255, 0.9999 / 255):
        x = (levels + nudge).clamp(0, 1)
        expect = _ref_save_bytes(x)
        got = (VS.h3_truncate_frames(x.clone()) * 255).round().to(torch.uint8)
        assert torch.equal(got, expect)


def test_truncation_is_in_place_and_idempotent():
    x = torch.rand(2, 3, 8, 8)
    expect = _ref_save_bytes(x)
    out = VS.h3_truncate_frames(x)
    assert out.data_ptr() == x.data_ptr()  # no copy of the clip
    again = VS.h3_truncate_frames(out.clone())
    assert torch.equal(again, out) and torch.equal((out * 255).round().to(torch.uint8), expect)
    half = torch.rand(4, 4).half()
    assert VS.h3_truncate_frames(half).dtype is torch.float32


def test_the_on_device_uint8_export_keeps_the_truncation():
    pytest.importorskip("diffusers")
    from core.inference.video_frames import uint8_video_frames

    pipe = _processor_pipe()
    VS.install_h3_output_truncation(pipe)
    video = _decoded(3)
    with uint8_video_frames(pipe):
        out = pipe.video_processor.postprocess_video(video, output_type = "np")
    assert torch.equal(
        torch.as_tensor(np.asarray(out))[0], _ref_save_bytes(video[0].permute(1, 2, 3, 0))
    )


def test_a_pipeline_without_a_processor_is_left_alone():
    assert not VS.install_h3_output_truncation(types.SimpleNamespace())


# ---- dense time embedding --------------------------------------------------------------------------------------


def _dense_pair(
    freq = 32,
    hidden = 24,
    out = 16,
    width = 8,
):
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AdaLayerNormModulation,
    )

    torch.manual_seed(5)
    ref = _RefTimeEmbedder(freq, hidden, out)
    proj = Timesteps(num_channels = freq, flip_sin_to_cos = True, downscale_freq_shift = 0)
    emb = TimestepEmbedding(in_channels = freq, time_embed_dim = hidden, out_dim = out)
    with torch.no_grad():
        emb.linear_1.weight.copy_(ref.proj_in.weight)
        emb.linear_1.bias.copy_(ref.proj_in.bias)
        emb.linear_2.weight.copy_(ref.proj_out.weight)
        emb.linear_2.bias.copy_(ref.proj_out.bias)
    ada = MiniMaxH3AdaLayerNormModulation(out, width).to(torch.bfloat16)
    holder = types.SimpleNamespace(
        time_embedder = emb, transformer_blocks = [types.SimpleNamespace(adaln_proj = ada)]
    )
    return ref, proj, emb, ada, holder


def test_dense_time_embedding_is_rounded_before_the_silu():
    pytest.importorskip("diffusers")
    ref, proj, emb, ada, holder = _dense_pair()
    t = torch.tensor([0.0, 0.25, 0.6125, 0.999, 1.0], dtype = torch.float32)
    with torch.no_grad():
        expect = _ref_adaln(ada.linear, ref(t).to(torch.bfloat16), 6, 3, ada.hidden_size, True)
        stock = ada(emb(proj(t)))
        assert not all(torch.equal(a, b) for a, b in zip(stock, expect))  # the check can fail
        assert VA.install_h3_reference_temb(holder)
        assert VA.install_h3_reference_temb(holder)
        got = ada(emb(proj(t)))
    assert emb(proj(t)).dtype is torch.bfloat16
    for a, b in zip(got, expect):
        assert torch.equal(a, b)


def test_curve_and_float32_stacks_keep_their_time_embedding():
    curve = types.SimpleNamespace(time_embedder = VA._build_curve_time_embedder(5, 4))
    assert not VA.install_h3_reference_temb(curve)
    pytest.importorskip("diffusers")
    _, _, emb, ada, holder = _dense_pair()
    ada.float()
    assert not VA.install_h3_reference_temb(holder)
    assert not VA.install_h3_reference_temb(types.SimpleNamespace())


def test_curve_adaln_interpolates_and_projects_like_the_reference():
    torch.manual_seed(2)
    grid, dim, width = 9, 6, 4
    embedder = VA._build_curve_time_embedder(grid, dim)
    table = torch.randn(grid, dim)
    with torch.no_grad():
        embedder.table.copy_(table)
    unique_t = [0.0, 0.0371, 0.5, 0.8125, 0.999, 1.0]
    rows = _ref_curve_rows(table, unique_t)
    got_rows = embedder(torch.tensor(unique_t, dtype = torch.float32))
    assert torch.equal(got_rows, rows)
    linear = torch.nn.Linear(dim, 6 * width * 3).float()
    proj = types.SimpleNamespace(
        linear = linear, hidden_size = width, _unsloth_adaln_out_dtype = torch.bfloat16
    )
    expect = [m.to(torch.bfloat16) for m in _ref_adaln(linear, rows, 6, 3, width, False)]
    for a, b in zip(VA._curve_modulation_forward(proj, got_rows), expect):
        assert torch.equal(a, b)


# ---- keyframe canvas, decoder and defaults ---------------------------------------------------------------------


def _b64(size, seed = 0):
    img = Image.fromarray(
        np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype = np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode(), img


@pytest.mark.parametrize("size", [(1200, 900), (900, 1600), (2400, 400), (1344, 768)])
def test_keyframes_without_a_size_keep_the_reference_canvas(size):
    from core.inference.video import VideoBackend
    from core.inference.video_minimax_h3 import H3_ANCHOR_FIRST, H3_ANCHOR_LAST, fit_h3_keyframe

    h3 = detect_video_family("", override = "minimax-h3")
    data, img = _b64(size)
    first, last, width, height, mode = VideoBackend._resolve_keyframes(
        h3, None, data, data, None, None
    )
    assert (width, height) == (1344, 768) == tuple(h3.resolution_presets[0])
    assert np.array_equal(
        np.asarray(first), np.asarray(fit_h3_keyframe(img, 1344, 768, anchor = H3_ANCHOR_FIRST))
    )
    assert np.array_equal(
        np.asarray(last), np.asarray(fit_h3_keyframe(img, 1344, 768, anchor = H3_ANCHOR_LAST))
    )
    # an explicit size still wins
    *_, w2, h2, _ = VideoBackend._resolve_keyframes(h3, None, data, None, 960, 544)
    assert (w2, h2) == (960, 544)


def test_the_int8_decoder_stays_the_default_like_the_reference_template(monkeypatch):
    # The reference templates decode with an int8 ConvRot VAE; Studio's int8 decoder lands as close to that decode
    # as its float16 one and is faster and smaller, so it stays the default.
    from core.inference import video_minimax_h3_vae as H

    monkeypatch.delenv(H.H3_VAE_INT8_ENV, raising = False)
    for tier in ("default", "max", None):
        assert H.LEVER_INT8_DECODER in H.plan_h3_vae_levers(tier, workflow = "t2va")
    monkeypatch.setenv(H.H3_VAE_INT8_ENV, "0")
    assert H.LEVER_INT8_DECODER not in H.plan_h3_vae_levers("default", workflow = "t2va")


def test_h3_defaults_equal_the_reference_templates():
    h3 = detect_video_family("", override = "minimax-h3")
    # template: res_multistep + simple, 20 steps, BasicGuider, 5 s at 24 fps, no SigmaShift node
    assert h3.default_steps == 20 and h3.default_guidance == 1.0
    assert h3.default_num_frames == 124 and h3.default_fps == 24
    assert (h3.default_flow_shift, h3.default_audio_flow_shift) == (12.0, 3.0)
    assert VS.h3_sampler(False) == VS.H3_SAMPLER_RES_MULTISTEP
    # the reference node's 1344x768 for a request without a size; the templates' 0.4 MP 16:9 for the page
    assert tuple(h3.resolution_presets[0]) == (1344, 768)
    assert h3.ui_default_resolution == (864, 480) and h3.keyframes_keep_canvas
    scale = math.sqrt(0.4 * 1024 * 1024 / (16 * 9))
    assert (round(16 * scale / 32) * 32, round(9 * scale / 32) * 32) == h3.ui_default_resolution


def test_status_reports_the_page_default_only_when_it_is_a_preset(monkeypatch):
    from core.inference import video as V

    h3 = detect_video_family("", override = "minimax-h3")
    assert V._ui_default_resolution(h3) == [864, 480]
    assert V._ui_default_resolution(types.SimpleNamespace(ui_default_resolution = None)) is None
    no_480p = types.SimpleNamespace(
        ui_default_resolution = (864, 480), resolution_presets = ((1344, 768),)
    )
    assert V._ui_default_resolution(no_480p) is None
