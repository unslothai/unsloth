# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Composited-canvas tiled VAE decode (diffusion_vae_canvas): the overlap-add oracle, stock equality outside the
four-tile corners, kill switches, fallbacks and the graph race bookkeeping. CPU only."""

from __future__ import annotations

import copy

import pytest
import torch

from core.inference import diffusion_vae_canvas as C


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (C.CANVAS_ENV, C.TILE_BATCH_ENV, C.TILE_GRAPHS_ENV, C.HALF_ENV):
        monkeypatch.delenv(name, raising = False)


def _image_vae_cls():
    from core.inference.diffusion_minimax_h3_image import vae_class
    return vae_class()


def _tiny_image_vae(seed: int = 0):
    torch.manual_seed(seed)
    vae = _image_vae_cls()(
        latent_channels = 8,
        block_out_channels = (16, 32),
        layers_per_block = 1,
        spatial_downsample_factors = (2, 2),
        temporal_downsample_factors = (1, 1),
        norm_num_groups = 8,
        decoder_num_layers = 2,
        decoder_num_attention_heads = 2,
        decoder_attention_head_dim = 16,
        decoder_num_register_tokens = 2,
        decoder_ffn_mult = 2,
        latents_mean = (0.0,) * 8,
        latents_std = (1.0,) * 8,
    ).eval()
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("scale1") or name.endswith("scale2"):
                p.normal_(0, 0.5)
            elif name.endswith("bias"):
                p.normal_(0, 0.1)
    # 4x: 16 px tiles, >= 4 px overlaps (latent-aligned), so a 6x9-latent canvas is a 2x3 grid.
    vae.tile_sample_min_height = vae.tile_sample_min_width = 16
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 4
    return vae


def _overlap_add(
    tiles,
    y,
    x,
    dtype = torch.float64,
):
    """Independent oracle: sum of w * tile over sum of w, w = product of the stock ramps, in float64."""
    ys, yl, yov = y
    xs, xl, xov = x
    H, W = ys[-1] + yl[-1], xs[-1] + xl[-1]
    lead = tiles[0][0].shape[:-2]
    acc = torch.zeros((*lead, H, W), dtype = dtype)
    wsum = torch.zeros((H, W), dtype = dtype)

    def ramp(lengths, overlaps, k):
        w = torch.ones(lengths[k], dtype = dtype)
        if k > 0:
            e = overlaps[k - 1]
            w[:e] = torch.arange(e, dtype = dtype) / e
        if k < len(lengths) - 1:
            e = overlaps[k]
            w[lengths[k] - e :] = 1 - torch.arange(e, dtype = dtype) / e
        return w

    for i, row in enumerate(tiles):
        for j, t in enumerate(row):
            w = ramp(yl, yov, i)[:, None] * ramp(xl, xov, j)[None, :]
            acc[..., ys[i] : ys[i] + yl[i], xs[j] : xs[j] + xl[j]] += t.to(dtype) * w
            wsum[ys[i] : ys[i] + yl[i], xs[j] : xs[j] + xl[j]] += w
    return acc / wsum.clamp_min(1e-12)


def _corner_mask(y, x):
    """Pixels inside both a vertical and a horizontal overlap (where four tiles meet)."""
    ys, yl, yov = y
    xs, xl, xov = x
    H, W = ys[-1] + yl[-1], xs[-1] + xl[-1]
    rows = torch.zeros(H, dtype = torch.bool)
    cols = torch.zeros(W, dtype = torch.bool)
    for k, e in enumerate(yov):
        rows[ys[k + 1] : ys[k + 1] + e] = True
    for k, e in enumerate(xov):
        cols[xs[k + 1] : xs[k + 1] + e] = True
    return rows[:, None] & cols[None, :]


def _raw_tiles(vae, z, plan):
    (ys, yl, _), (xs, xl, _), r = plan
    with torch.no_grad():
        return [
            [
                vae.decoder(
                    vae.post_quant_conv(z[..., a // r : (a + la) // r, b // r : (b + lb) // r])
                )
                for b, lb in zip(xs, xl)
            ]
            for a, la in zip(ys, yl)
        ]


def test_constant_grid_matches_the_overlap_add_and_stock_does_not():
    """V1: 4 constant tiles 0/1/2/3 with overlap 2 (scripts/consolidate_nov12/seam_check.py)."""
    vals = [[0.0, 1.0], [2.0, 3.0]]
    tiles = [[torch.full((1, 1, 4, 4), v) for v in r] for r in vals]
    y = x = ([0, 2], [4, 4], [2])
    got = C.oracle_stitch(tiles, y, x)[0, 0]
    want = _overlap_add(tiles, y, x)[0, 0].float()
    assert torch.equal(got, want)
    assert got[2, 2] == 0.0 and got[3, 3] == 1.5
    stock = _image_vae_cls()._stitch_tiles(_tiny_image_vae(), tiles, [2], [2])[0, 0]
    assert stock[2, 2] == 2.0 and stock[3, 3] == 2.0
    corner = _corner_mask(y, x)
    assert torch.equal(stock[~corner], got[~corner])


@pytest.mark.parametrize("shape", [(6, 9), (10, 5), (8, 8), (3, 12), (4, 4)])
def test_canvas_equals_stock_outside_corners_and_the_oracle_inside(shape):
    vae = _tiny_image_vae()
    z = torch.randn(1, 8, *shape, generator = torch.Generator().manual_seed(3))
    plan = C.layout(vae, z)
    assert plan is not None
    with torch.no_grad():
        stock = vae._decode(z)
        got = C.canvas_decode(vae, z)
    assert got.shape == stock.shape and got.dtype == stock.dtype and got.is_contiguous()
    corner = _corner_mask(plan[0], plan[1]).expand_as(got)
    assert torch.equal(got[~corner], stock[~corner])
    oracle = _overlap_add(_raw_tiles(vae, z, plan), plan[0], plan[1])
    torch.testing.assert_close(got.double(), oracle, rtol = 1e-6, atol = 1e-6)
    if corner.any():
        assert not torch.equal(got[corner], stock[corner]), "the corners are where stock is wrong"


def test_installed_decode_and_the_kill_switch(monkeypatch):
    vae = _tiny_image_vae()
    ref = copy.deepcopy(vae)
    z = torch.randn(1, 8, 6, 9, generator = torch.Generator().manual_seed(4))
    levers = C.install(vae, "cpu", speed = True, half = True)
    assert levers == (C.LEVER_CANVAS,), "CPU: canvas only, no batching / graphs / fp16"
    assert C.install(vae, "cpu") == (C.LEVER_CANVAS,), "idempotent"
    with torch.no_grad():
        stock = ref.decode(z, return_dict = False)[0]
        got = vae.decode(z, return_dict = False)[0]
        assert torch.equal(got, C.canvas_decode(ref, z))
        assert not torch.equal(got, stock)
        monkeypatch.setenv(C.CANVAS_ENV, "0")
        assert torch.equal(vae.decode(z, return_dict = False)[0], stock)
    assert vae._unsloth_canvas_stats["dtype"] == "fp32"
    C.uninstall(vae)
    assert "_decode" not in vae.__dict__


def test_untiled_and_single_tile_decodes_stay_stock():
    vae = _tiny_image_vae()
    C.install(vae, "cpu")
    z = torch.randn(1, 8, 4, 4)
    ref = copy.deepcopy(vae)
    with torch.no_grad():
        assert torch.equal(vae.decode(z, return_dict = False)[0], ref.decode(z, return_dict = False)[0])
        vae.use_tiling = ref.use_tiling = False
        z = torch.randn(1, 8, 6, 9)
        assert torch.equal(vae.decode(z, return_dict = False)[0], ref.decode(z, return_dict = False)[0])


def test_unknown_stitch_or_layout_keeps_stock(monkeypatch):
    vae = _tiny_image_vae()
    monkeypatch.setattr(vae, "_stitch_tiles", vae._stitch_tiles)
    assert not C.stock_tiling(vae)
    assert C.install(vae, "cpu") == ()
    vae2 = _tiny_image_vae()
    assert C.stock_tiling(vae2)
    # zero overlaps: stock's own crop breaks there, the canvas refuses the layout
    assert not C.axis_ok([0, 16], [16, 16], [0], 32, 4)
    assert C.axis_ok([0, 12, 24], [16, 16, 16], [4, 4], 40, 4)
    assert not C.axis_ok(
        [0, 12, 24], [16, 16, 16], [4, 4], 44, 4
    ), "tiles must end at the canvas edge"
    assert not C.axis_ok([0, 4, 8], [12, 12, 12], [8, 8], 20, 4), "never three tiles over one pixel"
    assert C.axis_ok([0], [24], [], 24, 4)


@pytest.mark.parametrize("batch", ["1", "2", "4", "16"])
def test_tile_batch_is_rounding_level_equal(monkeypatch, batch):
    vae = _tiny_image_vae()
    z = torch.randn(1, 8, 10, 9, generator = torch.Generator().manual_seed(6))
    with torch.no_grad():
        monkeypatch.setenv(C.TILE_BATCH_ENV, "1")
        ref = C.canvas_decode(vae, z)
        monkeypatch.setenv(C.TILE_BATCH_ENV, batch)
        got = C.canvas_decode(vae, z)
    assert vae._unsloth_canvas_stats["batch"] == min(int(batch), vae._unsloth_canvas_stats["tiles"])
    torch.testing.assert_close(got, ref, rtol = 1e-5, atol = 1e-5)


def test_batch_of_several_latents(monkeypatch):
    vae = _tiny_image_vae()
    z = torch.randn(2, 8, 6, 9, generator = torch.Generator().manual_seed(7))
    monkeypatch.setenv(C.TILE_BATCH_ENV, "3")
    with torch.no_grad():
        got = C.canvas_decode(vae, z)
        one = torch.cat([C.canvas_decode(vae, z[k : k + 1]) for k in range(2)])
    torch.testing.assert_close(got, one, rtol = 1e-5, atol = 1e-5)


def test_auto_batch_is_one_off_cuda():
    vae = _tiny_image_vae()
    assert C.tile_batch_size(vae, torch.zeros(1, 8, 4, 4), 12) == 1


def test_tile_batch_groups_are_even(monkeypatch):
    vae = _tiny_image_vae()
    monkeypatch.setattr(C, "tile_batch_setting", lambda: None)
    monkeypatch.setattr(C, "_decoder_tile_bytes", lambda v, t: 1)

    class _Dev:
        type = "cuda"

    tile = types_ns(shape = (1, 8, 4, 4), device = _Dev())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (10**12, 10**12))
    import core.inference.diffusion_memory as M

    monkeypatch.setattr(M, "user_budget_reserve_bytes", lambda d, t: 0)
    assert C.tile_batch_size(vae, tile, 121) == 16  # 8 calls of <= 16
    assert C.tile_batch_size(vae, tile, 17) == 9  # 2 calls, not 16 + 1


def test_speed_off_keeps_one_tile_per_call(monkeypatch):
    """speed=False (UNSLOTH_H3_IMAGE_SPEED=0) must not batch tiles on CUDA: stock memory and numerics, canvas kept."""
    vae = _tiny_image_vae()
    monkeypatch.setattr(C, "tile_batch_setting", lambda: None)
    monkeypatch.setattr(C, "_decoder_tile_bytes", lambda v, t: 1)

    class _Dev:
        type = "cuda"

    tile = types_ns(shape = (1, 8, 4, 4), device = _Dev())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (10**12, 10**12))
    import core.inference.diffusion_memory as M

    monkeypatch.setattr(M, "user_budget_reserve_bytes", lambda d, t: 0)
    vae._unsloth_canvas_batch_ok = False
    assert C.tile_batch_size(vae, tile, 121) == 1
    vae._unsloth_canvas_batch_ok = True
    assert C.tile_batch_size(vae, tile, 121) == 16
    vae2 = _tiny_image_vae()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    C.install(vae2, "cuda", speed = False)
    assert vae2._unsloth_canvas_batch_ok is False


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


def test_half_decoder_falls_back_to_fp32_on_a_bad_decode(monkeypatch):
    vae = _tiny_image_vae()
    C.install(vae, "cpu")
    assert C.half_decoder_ok(vae)
    vae._unsloth_canvas_half = True
    stock_decoder = vae.decoder.forward
    calls = []

    def flaky(x):
        calls.append(1)
        out = stock_decoder(x)
        return out * float("nan") if len(calls) == 1 else out

    monkeypatch.setattr(vae.decoder, "forward", flaky)
    monkeypatch.setenv(C.TILE_BATCH_ENV, "16")
    z = torch.randn(1, 8, 6, 9)
    with torch.no_grad():
        out = vae.decode(z, return_dict = False)[0]
    assert torch.isfinite(out).all()
    assert vae._unsloth_canvas_stats["half_fallback"] == "nonfinite"
    assert vae._unsloth_canvas_half is False
    assert all(m.weight.dtype == torch.float32 for m in C._block_linears(vae))


def test_flat_image_keeps_the_half_decoder_when_fp32_is_flat_too(monkeypatch):
    """A solid-colour image decodes flat in any dtype: the probe re-checks in fp32 once and keeps fp16."""
    vae = _tiny_image_vae()
    C.install(vae, "cpu")
    vae._unsloth_canvas_half = True
    monkeypatch.setattr(
        vae.decoder,
        "forward",
        lambda x: torch.full((x.shape[0], 3, x.shape[-2] * 4, x.shape[-1] * 4), 0.3),
    )
    with torch.no_grad():
        out = vae.decode(torch.randn(1, 8, 6, 9), return_dict = False)[0]
    torch.testing.assert_close(out, torch.full_like(out, 0.3))
    st = vae._unsloth_canvas_stats
    assert st.get("half_flat_check") and "half_fallback" not in st
    assert vae._unsloth_canvas_half is True
    assert all(m.weight.dtype == torch.float16 for m in C._block_linears(vae))


def test_stats_are_per_decode():
    vae = _tiny_image_vae()
    C.install(vae, "cpu")
    with torch.no_grad():
        vae.decode(torch.randn(1, 8, 6, 9), return_dict = False)
        assert vae._unsloth_canvas_stats["canvas"] and vae._unsloth_canvas_stats["tiles"] == 6
        vae.use_tiling = False
        vae.decode(torch.randn(1, 8, 6, 9), return_dict = False)
    assert vae._unsloth_canvas_stats == {"canvas": False, "dtype": "fp32"}


def test_set_decoder_dtype_touches_only_block_gemms():
    vae = _tiny_image_vae()
    n = C.set_decoder_dtype(vae, torch.float16)
    assert n == 2 * (4 + 2)  # per block: q, k, v, out + SwiGLU proj, out
    for name, p in vae.named_parameters():
        half = ".attn.to_" in name or ".ff." in name
        assert (p.dtype == torch.float16) == half, name
    # the block's own casts keep the residual stream fp32
    z = torch.randn(1, 8, 4, 4)
    vae.tile_sample_min_height = vae.tile_sample_min_width = 64
    if torch.backends.mps.is_available() or True:
        try:
            with torch.no_grad():
                out = vae.decode(z, return_dict = False)[0]
        except RuntimeError as exc:  # CPU builds without fp16 GEMM
            pytest.skip(str(exc))
    assert out.dtype == torch.float32


def test_bad_output_probe():
    assert C._bad_output(torch.tensor([0.0, float("nan")]))
    assert C._bad_output(torch.full((3, 4), 0.25))
    assert not C._bad_output(torch.tensor([0.0, 0.5]))


def _timed(race, per_call):
    for arm, ms in per_call.items():
        race.calls[arm] = race.needed(arm)
        race.ms[arm] = ms * race.needed(arm)


def test_race_keeps_eager_on_a_tie_and_picks_fewest_replicas():
    r = C._Race((1, 2, 4))
    _timed(r, {0: 10.0, 1: 9.95, 2: 10.1, 4: 9.9})
    r.settle()
    assert r.decided == 0, "inside the margin: the incumbent stays"
    r = C._Race((1, 2, 4))
    _timed(r, {0: 10.0, 1: 8.0, 2: 6.05, 4: 6.0})
    r.settle()
    assert r.decided == 2, "4 replicas within 2% of 2: the fewer wins"
    r = C._Race((1,))
    _timed(r, {0: 10.0})
    r.calls[1] = 2
    r.settle()
    assert r.decided is None and r.current() == 1
    r = C._Race((1, 2, 4))
    _timed(r, {0: 10.0, 1: 8.0})
    r.skip(2)
    r.skip(4)
    r.settle()
    assert r.decided == 1, "skipped arms (pool over the cap) are never picked"


class _Ev:
    def __init__(self, t):
        self.t = t

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return other.t - self.t


def test_race_times_runs_not_single_calls():
    """Replicas overlap: a run of N calls is timed first start to last end, so per-call intervals that undercount
    the overlapped replays cannot make an arm look faster than it is."""
    r = C._Race((4,))
    r.pending += [(0, _Ev(0), _Ev(10)), (0, _Ev(10), _Ev(20)), (0, _Ev(20), _Ev(30))]
    # 4 replicas: each call's own interval is 1 ms (its replay ran ahead), the run took 36 ms
    r.pending += [
        (4, _Ev(30), _Ev(39)),
        (4, _Ev(48), _Ev(49)),
        (4, _Ev(57), _Ev(58)),
        (4, _Ev(65), _Ev(66)),
    ]
    r.settle()
    assert r.medians[0] == 10.0 and r.medians[4] == 9.0
    assert r.decided == 4


def test_graph_arms_off_cpu_and_by_switch(monkeypatch):
    vae = _tiny_image_vae()
    vae._unsloth_canvas_graphs = True
    assert C._graph_arms(vae, torch.device("cpu")) == ()
    cuda = torch.device("cuda")
    assert C._graph_arms(vae, cuda) == C.GRAPH_REPLICAS
    monkeypatch.setattr(C, "_is_rocm", lambda: True)
    assert C._graph_arms(vae, cuda) == (1,)
    monkeypatch.setattr(C, "_is_rocm", lambda: False)
    monkeypatch.setenv(C.TILE_GRAPHS_ENV, "0")
    assert C._graph_arms(vae, cuda) == ()
    monkeypatch.setenv(C.TILE_GRAPHS_ENV, "3")
    assert C._graph_arms(vae, cuda) == (3,)


@pytest.mark.parametrize(
    "env, value", [("UNSLOTH_DIFFUSION_CUDA_GRAPHS", "0"), ("UNSLOTH_DISABLE_CUDA_GRAPH", "1")]
)
def test_global_cuda_graph_kill_switches_turn_tile_graphs_off(monkeypatch, env, value):
    vae = _tiny_image_vae()
    vae._unsloth_canvas_graphs = True
    monkeypatch.setattr(C, "_is_rocm", lambda: False)
    monkeypatch.delenv(C.TILE_GRAPHS_ENV, raising = False)
    cuda = torch.device("cuda")
    assert C._graph_arms(vae, cuda) == C.GRAPH_REPLICAS
    monkeypatch.setenv(env, value)
    assert C._graph_arms(vae, cuda) == ()
    monkeypatch.setenv(C.TILE_GRAPHS_ENV, "2")
    assert C._graph_arms(vae, cuda) == (), "the global switch wins over a forced tile setting"


def _tiny_video_vae():
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "AutoencoderKLMiniMaxH3", None)
    if cls is None:
        pytest.skip("diffusers without AutoencoderKLMiniMaxH3")
    torch.manual_seed(0)
    vae = cls(
        block_out_channels = (16, 32),
        layers_per_block = 1,
        spatial_downsample_factors = (2, 2),
        temporal_downsample_factors = (2, 1),
        norm_num_groups = 8,
        latent_channels = 8,
        decoder_num_layers = 2,
        decoder_num_attention_heads = 2,
        decoder_attention_head_dim = 16,
        decoder_num_register_tokens = 2,
        decoder_ffn_mult = 2,
        decoder_rope_dim_ratio = 0.75,
        clip_length = 9,
        token_drop = 1,
    ).eval()
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("scale1") or name.endswith("scale2"):
                p.normal_(0, 0.5)
    vae.tile_sample_min_height = vae.tile_sample_min_width = 16
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 4
    return vae


@pytest.mark.parametrize("batch", ["1", "3"])
def test_video_clip_canvas(monkeypatch, batch):
    from core.inference import video_minimax_h3_vae as H

    vae = _tiny_video_vae()
    stock = copy.deepcopy(vae)
    assert H._install_tile_batch(vae)
    monkeypatch.setenv(H.H3_VAE_TILE_BATCH_ENV, batch)
    z = torch.randn(1, 8, 3, 6, 9, generator = torch.Generator().manual_seed(5))
    plan = C.layout(vae, z)
    with torch.no_grad():
        ref = stock._decode_clip(z)
        got = vae._decode_clip(z)
        monkeypatch.setenv(C.CANVAS_ENV, "0")
        off = vae._decode_clip(z)
    if batch == "1":
        assert torch.equal(off, ref), "canvas off: the streamed stock stitch, bit for bit"
    else:
        torch.testing.assert_close(off, ref, rtol = 1e-5, atol = 1e-5)
    corner = _corner_mask(plan[0], plan[1]).expand_as(got)
    assert got.shape == ref.shape and got.is_contiguous()
    if batch == "1":
        assert torch.equal(got[~corner], ref[~corner])
    else:
        torch.testing.assert_close(got[~corner], ref[~corner], rtol = 1e-5, atol = 1e-5)
    oracle = _overlap_add(_raw_tiles(stock, z, plan), plan[0], plan[1])
    torch.testing.assert_close(got.double(), oracle, rtol = 1e-5, atol = 1e-5)


def test_pipeline_release_drops_the_tile_graphs_while_the_vae_is_still_referenced():
    """An unload must not leave the tile graphs' private pools reserved because something outside still holds the
    VAE (``release`` empties the weights in place for exactly that case)."""
    from core.inference import diffusion_minimax_h3_image as H

    vae = _tiny_image_vae()
    assert C.install(vae, "cpu")
    dropped = []

    class _Graphs:
        replicas = [object()]

        def drop(self):
            dropped.append(True)
            self.replicas = []

    race = C._Race((1, 2))
    race.graphs = _Graphs()
    race.reference = (torch.zeros(1), torch.zeros(1))
    vae._unsloth_canvas_races = {"key": race}
    pipe = H.MiniMaxH3ImagePipeline(
        transformer = None,
        vae = vae,
        text_encoder = None,
        tokenizer = None,
        scheduler = H.MiniMaxH3ImageScheduler(sample_sigmas = list(H.H3_IMAGE_SIGMAS)),
        processor = None,
    )
    pipe._unsloth_h3_owned = frozenset({"vae"})
    outside = vae  # a retained frame or cached closure
    pipe.release()
    assert dropped == [True], "release() left the tile graphs (and their pools) alive"
    assert "_unsloth_canvas_races" not in outside.__dict__
    assert race.graphs is None and race.reference is None
