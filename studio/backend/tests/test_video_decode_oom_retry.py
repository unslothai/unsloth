# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Kaggle T4 Wan crash: streamed untiled decode OOM, non-finite clip, fp32 rerun on non-finite latents."""

from __future__ import annotations

import ast
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("av")
pytest.importorskip("diffusers")

try:
    from diffusers import AutoencoderKLWan  # noqa: E402
    from diffusers.video_processor import VideoProcessor  # noqa: E402
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"diffusers is not importable here: {type(exc).__name__}", allow_module_level = True)

from core.inference import diffusion_speed as S  # noqa: E402
from core.inference import diffusion_vae_oom as O  # noqa: E402
from core.inference import video_frames as VF  # noqa: E402
from core.inference import video_stream_encode as vse  # noqa: E402
from core.inference import video_vae_untiled as U  # noqa: E402

FAMILY = "wan2.1-t2v-1.3b"


@pytest.fixture(autouse = True)
def _few_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(min(4, threads))
    yield
    torch.set_num_threads(threads)


def _vae():
    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim = 16,
        z_dim = 4,
        dim_mult = [1, 1, 1, 1],
        num_res_blocks = 1,
        temperal_downsample = [False, True, True],
        latents_mean = [0.0] * 4,
        latents_std = [1.0] * 4,
    ).eval()
    vae.enable_tiling()
    return vae


def _stack(monkeypatch):
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    vae = _vae()
    pipe = types.SimpleNamespace(vae = vae, video_processor = VideoProcessor(vae_scale_factor = 8))
    assert U.install_untiled_decode(pipe, FAMILY)
    assert O.install(vae, None, pipe = pipe)
    fired = []

    def oom_once(_module, _args, _kwargs):
        if not vae.use_tiling and not fired:
            fired.append(True)
            raise torch.OutOfMemoryError("CUDA out of memory. (injected)")

    vae.decoder.register_forward_pre_hook(oom_once, with_kwargs = True)
    return pipe, vae, fired


def _gate_stats(vae):
    fn = vae.decode
    for _ in range(16):
        if getattr(fn, "_unsloth_untiled_decode", False):
            return fn._unsloth_untiled_stats
        fn = O.decode_layers.next_layer(fn)
    raise AssertionError("untiled gate not in the decode chain")


def _render(pipe, z):
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 16)
    try:
        with torch.inference_mode(), session.hooks():
            return pipe.vae.decode(z, return_dict = False)[0]
    finally:
        session.close()


Z = torch.randn(1, 4, 3, 40, 40, generator = torch.Generator().manual_seed(1))


def test_streamed_render_untiled_oom_is_handled_by_the_gate(monkeypatch):
    """Fail-before: the wrapper hid the gate, so every later render OOMed again."""
    pipe, vae, fired = _stack(monkeypatch)
    out = _render(pipe, Z)
    assert fired == [True] and bool(torch.isfinite(out).all())
    stats, ladder = _gate_stats(vae), vae._unsloth_vae_oom_stats
    assert stats["oom_fallback"] == 1
    assert ladder["tiled_retry"] == 0
    fired.clear()
    fired.append(True)
    _render(pipe, Z)
    assert stats["tiled"] == 1 and stats["untiled"] == 0


def test_streamed_render_leaves_the_oom_scope_installed(monkeypatch):
    """Fail-before: the scope reinstalled itself above the per-render wrapper, which hooks() then dropped."""
    pipe, vae, fired = _stack(monkeypatch)
    fired.append(True)
    _render(pipe, Z)
    assert O._scope_intact(vae)


def test_stream_chain_kill_switch(monkeypatch):
    monkeypatch.setenv(vse.STREAM_CHAIN_ENV, "0")
    pipe, vae, fired = _stack(monkeypatch)
    _render(pipe, Z)
    assert _gate_stats(vae)["oom_fallback"] == 0
    assert vae._unsloth_vae_oom_stats["tiled_retry"] == 1


def test_streamed_render_without_oom_is_unchanged(monkeypatch):
    pipe, vae, fired = _stack(monkeypatch)
    fired.append(True)
    got = _render(pipe, Z)
    ref = _vae()
    ref.disable_tiling()
    with torch.inference_mode():
        want = ref.decode(Z, return_dict = False)[0]
    assert torch.equal(got, want)


def _half_pipe(monkeypatch):
    monkeypatch.setattr(S, "_fp16_compile_capable", lambda target: True)
    seen = []

    class Vae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Conv3d(1, 1, 1)

        def decode(
            self,
            z,
            return_dict = False,
        ):
            seen.append((z.dtype, next(self.decoder.parameters()).dtype))
            return (z * 1,)

    vae = Vae()
    pipe = types.SimpleNamespace(vae = vae)
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", ordinal = None)
    family = types.SimpleNamespace(vae_force_fp32 = True)
    assert S._video_vae_half_decode(pipe, target, family, None)
    return vae, seen


def test_nonfinite_latents_do_not_rerun_the_vae_in_fp32(monkeypatch):
    vae, seen = _half_pipe(monkeypatch)
    z = torch.zeros(1, 1, 2, 4, 4)
    z[0, 0, 1, 2, 2] = float("nan")
    out = vae.decode(z)[0]
    assert seen == [(torch.float16, torch.float16)]
    assert out.dtype == torch.float16 and not bool(torch.isfinite(out).all())
    vae.decode(torch.zeros(1, 1, 2, 4, 4))
    assert seen[-1] == (torch.float16, torch.float16)


def test_nonfinite_fp16_output_from_finite_latents_still_reruns_fp32(monkeypatch):
    vae, seen = _half_pipe(monkeypatch)
    z = torch.full((1, 1, 2, 4, 4), 70000.0)  # finite in fp32, inf in fp16
    out = vae.decode(z)[0]
    assert seen == [(torch.float16, torch.float16), (torch.float32, torch.float32)]
    assert bool(torch.isfinite(out).all())


def test_nonfinite_latents_kill_switch(monkeypatch):
    monkeypatch.setenv(S.VIDEO_VAE_NONFINITE_INPUT_ENV, "0")
    vae, seen = _half_pipe(monkeypatch)
    z = torch.full((1, 1, 2, 4, 4), float("nan"))
    vae.decode(z)
    assert seen[-1] == (torch.float32, torch.float32)


def test_require_finite_frames():
    VF.require_finite_frames(np.zeros((2, 4, 4, 3), np.uint8))
    VF.require_finite_frames(np.full((2, 4, 4, 3), 0.5, np.float32))
    VF.require_finite_frames(torch.zeros(2, 4, 4, 3, dtype = torch.uint8))
    bad = np.full((2, 4, 4, 3), 0.5, np.float32)
    bad[1, 2, 2, 0] = np.nan
    with pytest.raises(VF.NonFiniteFramesError):
        VF.require_finite_frames(bad)
    with pytest.raises(VF.NonFiniteFramesError):
        VF.require_finite_frames(torch.from_numpy(bad))


def test_encode_mp4_refuses_a_nonfinite_clip_with_a_clear_error(monkeypatch):
    from core.inference import video as V

    monkeypatch.setattr(V, "nvenc_gpu", lambda logger = None: None)
    frames = np.full((2, 16, 16, 3), 0.5, np.float32)
    frames[0, 0, 0, 0] = np.nan
    with pytest.raises(VF.NonFiniteFramesError) as info:
        V.VideoBackend._encode_mp4(frames, 16, None, None)
    detail = V.video_failure_detail(info.value)
    assert detail.startswith(V.VIDEO_GENERATION_FAILED_MSG) and "NaN" in detail


def test_video_speed_layer_gets_the_logger():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "run_speed_layer"
    ]
    assert calls and all(any(k.arg == "logger" for k in c.keywords) for c in calls)
