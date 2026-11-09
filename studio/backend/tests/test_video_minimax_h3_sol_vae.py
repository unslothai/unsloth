# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in Sol-H3-VAE decoder: frame lattice, streamed decode against the unstreamed reference, checkpoint layout,
the switch, and the fallbacks. CPU only: tiny random weights in the vendor checkpoint layout."""

from __future__ import annotations

import ast
import logging
import pathlib
import types

import pytest
import torch

from core.inference import decode_layers
from core.inference import video_minimax_h3_sol_vae as S
from core.inference import video_minimax_h3_vae as HV
from core.inference.video_families import detect_video_family

HEADS, DH, LAYERS, FFN, REG, WINDOW = 4, 16, 3, 96, 4, 4
DIM = HEADS * DH


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(S.SOL_VAE_ENV, raising = False)


def _vendor_state_dict(seed: int = 0, layers: int = LAYERS) -> dict:
    """Random weights with the released checkpoint's keys and layouts (fused per-head qkv, SwiGLU [gate | up])."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale = 0.2):
        return torch.randn(*shape, generator = g) * scale

    sd = {
        "post_quant_conv.weight": r(24, 24, 1, 1, 1),
        "post_quant_conv.bias": r(24),
        "x_embedder.weight": r(DIM, 24),
        "x_embedder.bias": r(DIM),
        "register_tokens": r(1, REG, DIM, scale = 1.0),
        "norm_out.weight": 1 + r(DIM),
        "norm_out.bias": r(DIM),
        "proj_out.weight": r(3 * 4 * 16 * 16, DIM),
        "proj_out.bias": r(3 * 4 * 16 * 16),
    }
    for i in range(layers):
        p = f"blocks.{i}."
        sd.update(
            {
                p + "norm1.weight": 1 + r(DIM),
                p + "norm2.weight": 1 + r(DIM),
                p + "scale1": r(DIM, scale = 0.5),
                p + "scale2": r(DIM, scale = 0.5),
                p + "attn.to_qkv.weight": r(3 * DIM, DIM),
                p + "attn.to_qkv.bias": r(3 * DIM),
                p + "attn.to_out.weight": r(DIM, DIM),
                p + "attn.to_out.bias": r(DIM),
                p + "ff.w1.weight": r(2 * FFN, DIM),
                p + "ff.w1.bias": r(2 * FFN),
                p + "ff.w2.weight": r(DIM, FFN),
                p + "ff.w2.bias": r(DIM),
            }
        )
    return sd


_CONFIG = {"decoder_num_attention_heads": HEADS, "decoder_rope_dim_ratio": 0.75, "decoder_rope_theta": 100.0,
           "decoder_norm_eps": 1e-5}  # fmt: skip


def _tiny(seed: int = 0, window: int = WINDOW):
    return S.build_sol_decoder(
        _vendor_state_dict(seed), vae_config = _CONFIG, dtype = torch.float32, window = window
    )


# ------------------------------------------------------------------------------------------------------ frame math
def test_sol_frames_are_the_h3_lattice_up_to_345():
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert fam is not None and fam.name == "minimax-h3"
    assert (fam.frame_step, fam.frame_offset, fam.max_num_frames) == (17, 5, 345)
    lattice = list(range(fam.frame_offset, fam.max_num_frames + 1, fam.frame_step))
    assert lattice[0] == 5 and lattice[-1] == 345 and 124 in lattice
    for frames in lattice:
        t = S.sol_latents_for_frames(frames)
        k = (frames - 5) // 17
        assert t == 5 * k + 2
        assert S.sol_num_frames(t) == frames
    assert S.sol_latents_for_frames(124) == 37 and S.sol_latents_for_frames(345) == 102


def test_sol_frame_count_round_trips_and_rejects_off_lattice_counts():
    for t in range(1, 103):
        f = S.sol_num_frames(t)
        assert f == 4 * t - 3 * -(-t // 5)
        assert S.sol_latents_for_frames(f) == t
    assert [S.sol_latents_for_frames(f) for f in (0, 2, 3, 4, 6, 16)] == [None] * 6
    with pytest.raises(ValueError):
        S.sol_num_frames(0)


# ------------------------------------------------------------------------------------------------------ the decoder
@pytest.mark.parametrize(
    "T,H,W",
    [(7, 6, 9), (12, 5, 7), (3, 9, 6), (5, 4, 4), (11, 3, 17), (2, 8, 8)],
)
def test_streamed_decode_matches_the_unstreamed_reference(T, H, W):
    """Edge windows on both shifts (H, W not multiples of the window), T not a multiple of 5, partial last blocks."""
    model = _tiny()
    z = torch.randn(1, 24, T, H, W, generator = torch.Generator().manual_seed(T * 100 + H))
    got = model.decode_raw(z)
    want = model.reference_decode_raw(z)
    assert got.shape == want.shape == (1, 3, S.sol_num_frames(T), 16 * H, 16 * W)
    assert got.dtype is torch.float32
    assert torch.isfinite(got).all()
    assert (got - want).abs().max() <= 1e-4 * want.abs().max()


def test_the_streamed_decode_holds_at_most_two_block_slots():
    model = _tiny()
    held: list = []
    orig = model._slot

    def slot(blk, x, b, s, st):
        held.append(b)
        return orig(blk, x, b, s, st)

    model._slot = slot
    peak = {"n": 0}
    orig_window = model._window_attn

    def window_attn(*args, **kwargs):
        frame = __import__("inspect").currentframe().f_back
        peak["n"] = max(peak["n"], len(frame.f_locals["slots"]))
        return orig_window(*args, **kwargs)

    model._window_attn = window_attn
    model.decode_raw(torch.randn(1, 24, 17, 3, 3))
    assert peak["n"] == 2
    # every block is projected once per layer
    assert len(held) == LAYERS * 4


def test_a_still_decodes_as_a_block_of_copies_keeping_the_first_frame():
    model = _tiny()
    z = torch.randn(1, 24, 1, 5, 6)
    got = model.decode_raw(z)
    assert got.shape == (1, 3, 1, 80, 96)
    block = model.decode_raw(z.expand(1, 24, 5, 5, 6))
    assert torch.equal(got[:, :, 0], block[:, :, 0])
    assert (got - model.reference_decode_raw(z)).abs().max() <= 1e-4 * got.abs().max()


def test_a_batch_decodes_each_sample_alone():
    model = _tiny()
    z = torch.randn(2, 24, 7, 4, 5)
    got = model.decode_raw(z)
    for i in range(2):
        assert torch.allclose(got[i : i + 1], model.decode_raw(z[i : i + 1]), atol = 1e-6)


def test_the_checkpoint_conversion_keeps_the_vendor_math():
    """Per-head [q | k | v] rows split into to_q / to_k / to_v; SwiGLU silu(gate) * up kept after the row swap."""
    sd = _vendor_state_dict()
    model = _tiny()
    x = torch.randn(10, DIM)
    fused = torch.nn.functional.linear(
        x, sd["blocks.0.attn.to_qkv.weight"], sd["blocks.0.attn.to_qkv.bias"]
    )
    q, k, v = fused.view(10, HEADS, 3 * DH).chunk(3, -1)
    blk = model.blocks[0]
    assert torch.allclose(blk.to_q(x), q.reshape(10, DIM), atol = 1e-5)
    assert torch.allclose(blk.to_k(x), k.reshape(10, DIM), atol = 1e-5)
    assert torch.allclose(blk.to_v(x), v.reshape(10, DIM), atol = 1e-5)
    gate, up = torch.nn.functional.linear(
        x, sd["blocks.0.ff.w1.weight"], sd["blocks.0.ff.w1.bias"]
    ).chunk(2, -1)
    want = torch.nn.functional.silu(gate) * up
    assert torch.allclose(HV._swiglu(blk.w1(x)), want, atol = 1e-5)


def test_gemm_weights_take_the_pipeline_dtype_and_the_rest_stays_float32():
    model = S.build_sol_decoder(
        _vendor_state_dict(), vae_config = _CONFIG, dtype = torch.bfloat16, window = WINDOW
    )
    blk = model.blocks[0]
    assert {p.dtype for p in (blk.to_q.weight, blk.to_out.bias, blk.w1.weight, blk.w2.weight)} == {
        torch.bfloat16
    }
    assert {p.dtype for p in (blk.norm1.weight, blk.scale1, model.x_embedder.weight, model.proj_out.weight,
                              model.register_tokens, model.norm_out.weight)} == {torch.float32}  # fmt: skip
    # bf16 GEMMs stay close to the float32 decoder (a smoke check of the mixed-dtype path, not a quality gate)
    z = torch.randn(1, 24, 7, 4, 5)
    ref = _tiny().decode_raw(z)
    got = model.decode_raw(z)
    assert got.dtype is torch.float32
    assert (got - ref).norm() / ref.norm() < 5e-2


def test_the_activation_estimate_grows_with_the_clip():
    small = S.sol_decode_activation_gb(960, 544, 124)
    big = S.sol_decode_activation_gb(1344, 768, 124)
    longest = S.sol_decode_activation_gb(1344, 768, 345)
    assert 0 < small < big < longest
    # the clip-wide float32 residual + pixels dominate: 1344x768x345 is ~3.4 + 4.3 GB of them
    assert 7.6 < longest < 16


# ------------------------------------------------------------------------------------------------------ the switch
class AutoencoderKLMiniMaxH3:  # the name install_h3_sol_vae checks; a stand-in for the Diffusers VAE
    def __init__(self):
        self.config = dict(_CONFIG)
        self.stock_calls = 0

    def _decode(self, z):
        self.stock_calls += 1
        return torch.zeros(
            z.shape[0], 3, S.sol_num_frames(z.shape[2]), 16 * z.shape[3], 16 * z.shape[4]
        )


@pytest.fixture
def weights(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "sol_h3_vae.safetensors"
    save_file({k: v.contiguous() for k, v in _vendor_state_dict().items()}, str(path))
    return str(path)


def _install(vae, path, caplog, **kw):
    log = logging.getLogger("sol_test")
    with caplog.at_level(logging.INFO, logger = "sol_test"):
        return S.install_h3_sol_vae(
            vae, device = "cpu", dtype = torch.float32, logger = log, weights_path = path, **kw
        )


def test_switch_off_leaves_the_stock_decode_and_the_planner_alone(weights, caplog, monkeypatch):
    from core.inference.video_minimax_h3 import estimate_h3_diffusers_vram_gb
    from core.inference.video_minimax_h3_residency import h3_phase_need_gb

    vae = AutoencoderKLMiniMaxH3()
    monkeypatch.setattr(
        S, "fetch_sol_weights", lambda **k: pytest.fail("fetched with the switch off")
    )
    assert S.install_h3_sol_vae(vae, device = "cpu", dtype = torch.float32) is False
    assert "_decode" not in vae.__dict__ and decode_layers.find(vae, "_decode", S.SOL_LAYER) is None
    assert S.sol_decode_need_gb(1344, 768, 345) == 0.0
    from core.inference import video_minimax_h3 as M
    from core.inference import video_minimax_h3_residency as R

    off = (
        estimate_h3_diffusers_vram_gb(1344, 768, 345),
        h3_phase_need_gb(1344, 768, 345, te_streamed_gb = 3.0),
    )
    vol = 1344 * 768 * 345 / 1e6
    assert off[0] == pytest.approx(
        M.h3_diffusers_vram_base_gb() + M.H3_DIFFUSERS_VRAM_GB_PER_MPIXEL_FRAME * vol
    )
    denoise = R.H3_STREAM_WINDOW_GB + R.H3_ACTIVATION_GB_PER_MPIXEL_FRAME * vol * (
        1 + R.H3_FRAGMENTATION_FRACTION
    )
    assert off[1] == pytest.approx(max(3.0, denoise, R.H3_VAE_DECODE_GB) + R.H3_PHASE_OVERHEAD_GB)
    monkeypatch.setenv(S.SOL_VAE_ENV, "0")
    assert S.install_h3_sol_vae(vae, device = "cpu", dtype = torch.float32) is False
    monkeypatch.setenv(S.SOL_VAE_ENV, "1")
    assert S.sol_decode_need_gb(1344, 768, 345) > S.SOL_WEIGHTS_GB
    assert h3_phase_need_gb(1344, 768, 345, te_streamed_gb = 3.0) >= off[1]


def test_switch_on_routes_the_video_decode_through_sol(weights, caplog, monkeypatch):
    monkeypatch.setenv(S.SOL_VAE_ENV, "1")
    vae = AutoencoderKLMiniMaxH3()
    assert _install(vae, weights, caplog) is True
    assert "video.h3_sol_vae: engaged" in caplog.text
    z = torch.randn(1, 24, 7, 4, 5)
    with caplog.at_level(logging.INFO, logger = "sol_test"):
        out = vae._decode(z)
    assert vae.stock_calls == 0 and vae.__dict__["_unsloth_h3_sol_vae_stats"]["calls"] == 1
    want = S.build_sol_decoder(
        _vendor_state_dict(), vae_config = _CONFIG, dtype = torch.float32
    ).decode_raw(z)
    assert torch.allclose(out, want, atol = 1e-6)
    assert "video.h3_sol_vae: decoded 7 latent frames -> 22 frames" in caplog.text
    # idempotent: a second install does not stack a second layer
    assert _install(vae, weights, caplog) is True
    assert len([l for l in vae.__dict__[decode_layers.LAYERS_ATTR] if l.name == S.SOL_LAYER]) == 1


def test_a_failed_download_or_load_keeps_the_stock_decoder(weights, caplog, monkeypatch, tmp_path):
    monkeypatch.setenv(S.SOL_VAE_ENV, "1")

    def offline(**kw):
        raise OSError("offline and not cached")

    monkeypatch.setattr(S, "fetch_sol_weights", offline)
    vae = AutoencoderKLMiniMaxH3()
    assert _install(vae, None, caplog) is False
    assert (
        "video.h3_sol_vae: not engaged, keeping the stock decoder: offline and not cached"
        in caplog.text
    )
    assert "_decode" not in vae.__dict__
    bad = tmp_path / "broken.safetensors"
    bad.write_bytes(b"not a safetensors file")
    assert _install(vae, str(bad), caplog) is False
    assert "_decode" not in vae.__dict__
    # not the H3 video VAE (the audio VAE, say): untouched
    other = types.SimpleNamespace(_decode = lambda z: z, config = {})
    assert _install(other, weights, caplog) is False


def test_a_failed_decode_falls_back_to_stock_and_settles_the_status(weights, caplog, monkeypatch):
    monkeypatch.setenv(S.SOL_VAE_ENV, "1")
    vae = AutoencoderKLMiniMaxH3()
    assert _install(vae, weights, caplog) is True
    model = vae.__dict__["_unsloth_h3_sol_vae"]

    def boom(z):
        raise RuntimeError("kernel exploded")

    model.decode_raw = boom
    z = torch.randn(1, 24, 7, 4, 5)
    with caplog.at_level(logging.INFO, logger = "sol_test"):
        out = vae._decode(z)
    assert vae.stock_calls == 1 and out.shape == (1, 3, 22, 64, 80)
    assert (
        "video.h3_sol_vae: decode failed (RuntimeError: kernel exploded); using the stock decoder"
        in caplog.text
    )
    assert S.h3_sol_vae_failed(vae)
    vae._decode(z)  # disabled after a non-OOM failure: stock, no second attempt
    assert vae.stock_calls == 2 and vae.__dict__["_unsloth_h3_sol_vae_stats"]["fallbacks"] == 1
    state = types.SimpleNamespace(speed_optims = ("h3_vae_fused_decoder", "h3_sol_vae"))
    HV.settle_h3_vae_fallback(state, types.SimpleNamespace(vae = vae))
    assert state.speed_optims == ("h3_vae_fused_decoder",)


def test_an_oom_falls_back_for_that_decode_only(weights, caplog, monkeypatch):
    monkeypatch.setenv(S.SOL_VAE_ENV, "1")
    vae = AutoencoderKLMiniMaxH3()
    assert _install(vae, weights, caplog) is True
    model = vae.__dict__["_unsloth_h3_sol_vae"]
    real = model.decode_raw
    calls = {"n": 0}

    def once(z):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(z)

    model.decode_raw = once
    z = torch.randn(1, 24, 7, 4, 5)
    vae._decode(z)
    assert vae.stock_calls == 1 and not S.h3_sol_vae_failed(vae)
    vae._decode(z)
    assert vae.stock_calls == 1 and vae.__dict__["_unsloth_h3_sol_vae_stats"]["calls"] == 1


@pytest.mark.parametrize(
    "error", ["kernel exploded", "CUDA out of memory. Tried to allocate 2.00 GiB"]
)
def test_a_resident_decoder_leaves_the_card_before_the_stock_fallback(
    weights, caplog, monkeypatch, error
):
    monkeypatch.setenv(S.SOL_VAE_ENV, "1")
    real_load = S.load_sol_decoder
    monkeypatch.setattr(S, "_free_device_gb", lambda device: None)
    monkeypatch.setattr(
        S, "load_sol_decoder", lambda path, **k: real_load(path, **{**k, "device": "cpu"})
    )
    vae = AutoencoderKLMiniMaxH3()
    log = logging.getLogger("sol_test")
    assert S.install_h3_sol_vae(
        vae, device = "cuda:0", dtype = torch.float32, logger = log, weights_path = weights
    )
    model = vae.__dict__["_unsloth_h3_sol_vae"]
    where = {"on_host": False}
    moves = []

    def fake_to(target):
        moves.append(str(target))
        where["on_host"] = str(target) == "cpu"
        return model

    monkeypatch.setattr(S, "_on_host", lambda m: where["on_host"])
    model.to = fake_to

    def boom(z):
        raise RuntimeError(error)

    model.decode_raw = boom
    vae._decode(torch.randn(1, 24, 7, 4, 5))
    assert vae.stock_calls == 1 and moves == ["cpu"]


def test_the_h3_load_installs_sol_after_the_stock_levers_and_under_the_oom_scope():
    source = (pathlib.Path(S.__file__).parent / "video.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    load = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_h3_modular_pipeline"
    )

    def calls(node, name):
        return [c for c in ast.walk(node) if isinstance(c, ast.Call)
                and (getattr(c.func, "id", None) == name or getattr(c.func, "attr", None) == name)]  # fmt: skip

    (install,) = calls(tree, "install_h3_sol_vae")
    assert any(c is install for c in ast.walk(load)), "only the H3 load installs Sol"
    (levers,) = calls(load, "apply_h3_vae_speedups")
    outermost = calls(load, "ensure_outermost")
    assert levers.lineno < install.lineno < min(c.lineno for c in outermost)
    guard = next(
        n
        for n in ast.walk(load)
        if isinstance(n, ast.If) and any(c is install for c in ast.walk(n.test))
    )
    assert "h3_sol_vae_requested" in ast.unparse(guard.test)
    assert {k.arg for k in install.keywords} >= {
        "device",
        "dtype",
        "hf_token",
        "local_files_only",
        "logger",
    }
    # the audio VAE is never handed to it
    assert "audio_vae" not in ast.unparse(install)


# ------------------------------------------------------------------------------------------------ host reads
def _count_host_reads(monkeypatch) -> dict:
    n = {"reads": 0}
    orig_bool, orig_item = torch.Tensor.__bool__, torch.Tensor.item

    def counted_bool(self):
        n["reads"] += 1
        return orig_bool(self)

    def counted_item(self):
        n["reads"] += 1
        return orig_item(self)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)
    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    return n


@pytest.mark.parametrize("T,H,W", [(7, 6, 9), (10, 8, 8), (1, 8, 8)])
def test_the_decode_reads_nothing_back_and_matches_the_old_path(monkeypatch, T, H, W):
    """Window masks are decided from the window geometry (host ints), not by reading the mask back per window; the
    kill switch restores the per-window read and must give the same bits."""
    model = _tiny()
    z = torch.randn(1, 24, T, H, W, generator = torch.Generator().manual_seed(T + H))
    monkeypatch.setenv(S.SOL_LEAN_WINDOWS_ENV, "0")
    old = model.decode_raw(z)
    monkeypatch.delenv(S.SOL_LEAN_WINDOWS_ENV)
    n = _count_host_reads(monkeypatch)
    got = model.decode_raw(z)
    assert n["reads"] == 0
    assert torch.equal(got, old)


def test_the_rope_tables_are_built_once_per_window_geometry():
    model = _tiny()
    calls: list = []
    orig = model._rope

    def rope(t0, n, H, W, s, st, device):
        calls.append((n, s, st))
        return orig(t0, n, H, W, s, st, device)

    model._rope = rope
    model.decode_raw(torch.randn(1, 24, 12, 8, 8, generator = torch.Generator().manual_seed(3)))
    assert len(calls) == len(
        set(calls)
    )  # 3 layers, 3 blocks each: one table per (rows, shift) only
    assert model._rope_tables is None  # released with the decode
