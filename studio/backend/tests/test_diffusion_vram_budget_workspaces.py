# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every runtime workspace Studio sizes from a live free-VRAM reading keeps the user's VRAM Budget reserve free."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from utils import vram_budget_settings

GIB = 1024**3


@pytest.fixture
def fraction(monkeypatch):
    state = {"value": vram_budget_settings.VRAM_FRACTION_DEFAULT}
    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", lambda: state["value"])
    return state


@pytest.fixture
def card(monkeypatch):
    """A fake 24 GiB CUDA card: ``state["free"]`` driver free, 1 GiB idle allocator cache."""
    state = {"free": 20 * GIB, "total": 24 * GIB, "cached": 1 * GIB}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (state["free"], state["total"]))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: state["cached"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    return state


def _cuda_z(*shape):
    return types.SimpleNamespace(
        device = torch.device("cuda", 0), shape = tuple(shape), dim = lambda: len(shape)
    )


def test_reserve_helper_is_zero_at_the_default_and_off_cuda(fraction):
    from core.inference.diffusion_memory import user_budget_reserve_bytes, user_budget_reserve_share

    assert user_budget_reserve_bytes("cuda:0", 24 * GIB) == 0
    assert user_budget_reserve_share() == 0.0
    fraction["value"] = 0.50
    assert user_budget_reserve_bytes(torch.device("cuda", 0), 24 * GIB) == 12 * GIB
    assert user_budget_reserve_share() == pytest.approx(0.50)
    assert user_budget_reserve_bytes("cpu", 24 * GIB) == 0
    assert user_budget_reserve_bytes("mps", 24 * GIB) == 0
    assert user_budget_reserve_bytes("cuda", None) == 0


def test_image_vae_tile_budget_reads_free_less_the_reserve(fraction, card, monkeypatch):
    """#14816 r8: a lowered budget prices the floor tile, so the runtime tile must not grow into the reserve."""
    from core.inference import diffusion_vae_tiling as vt

    from . import test_diffusion_vae_tiling_general as G

    monkeypatch.delenv(vt.MAX_TILE_ENV, raising = False)
    monkeypatch.delenv(vt.WIDE_TILES_ENV, raising = False)
    vae = G._qwen_image()
    assert vt.install(vae)
    z = _cuda_z(1, 4, 1, 64, 64)
    default_mib, scale = vt._free_mib(vae, z)
    default_budget = vt.decode_tile_budget(vae, z)
    assert default_budget > 0
    fraction["value"] = 0.50
    mib, scale2 = vt._free_mib(vae, z)
    assert scale2 == scale
    assert mib == pytest.approx(default_mib - 12 * 1024)
    budget = vt.decode_tile_budget(vae, z)
    per_latent = vt.decode_mib_per_latent(vt._geometry(vae)[0], False, type(vae).__name__) * scale
    assert budget == max(0, int(vt.FREE_FRACTION * mib / per_latent))
    assert budget < default_budget
    card["free"] = 10 * GIB
    assert vt.decode_tile_budget(vae, z) == 0


def test_image_vae_floor_shortfall_charges_the_reserve(fraction, card, monkeypatch):
    from core.inference import diffusion_vae_tiling as vt

    from . import test_diffusion_vae_tiling_general as G

    monkeypatch.delenv(vt.MAX_TILE_ENV, raising = False)
    vae = G._hunyuan_image()
    assert vt.install(vae)
    z = _cuda_z(1, 8, 64, 64)
    card["free"], card["cached"] = 4 * GIB, 0
    assert vt.floor_shortfall(vae, z) is None
    fraction["value"] = 0.85  # 3.6 GiB kept free leaves ~0.4 GiB: the floor tile no longer fits
    assert vt.floor_shortfall(vae, z)


def test_fused_vae_tile_batch_reads_free_less_the_reserve(fraction, card, monkeypatch):
    from core.inference import diffusion_vae_fused as F

    monkeypatch.delenv(F.TILE_BATCH_ENV, raising = False)
    weight = torch.nn.Parameter(torch.zeros(1, dtype = torch.float16))
    decoder = types.SimpleNamespace(
        norm_out = types.SimpleNamespace(gamma = torch.zeros(96)), parameters = lambda: iter([weight])
    )
    vae = types.SimpleNamespace(
        decoder = decoder,
        tile_sample_min_height = 256,
        tile_sample_min_width = 256,
        temporal_compression_ratio = 4,
    )
    per_tile = F._TILE_PEAK_FACTOR * 256 * 256 * 4 * 96 * 2
    z = _cuda_z(1, 16, 4, 32, 32)
    card["free"] = 4 * per_tile + 12 * GIB
    assert F._tile_batch_cap(vae, z) == F._TILE_BATCH_MAX
    fraction["value"] = 0.50  # 12 GiB kept free: 4 tiles of room, half of it is the batch
    assert F._tile_batch_cap(vae, z) == 2
    card["free"] = 2 * per_tile + 12 * GIB
    assert F._tile_batch_cap(vae, z) == 1
    fraction["value"] = vram_budget_settings.VRAM_FRACTION_DEFAULT
    assert F._tile_batch_cap(vae, z) == F._TILE_BATCH_MAX


def test_h3_vae_tile_batch_reads_free_less_the_reserve(fraction, card, monkeypatch):
    from core.inference import video_minimax_h3_vae as H

    monkeypatch.delenv(H.H3_VAE_TILE_BATCH_ENV, raising = False)
    z = _cuda_z(1, 16, 4, 32, 32)
    card["free"] = 12 * GIB + 2 * H._TILE_BATCH_BYTES_PER_TILE
    default = H._tile_batch_size(z)
    assert default == min(H._TILE_BATCH_MAX, card["free"] // H._TILE_BATCH_BYTES_PER_TILE)
    fraction["value"] = 0.50
    assert H._tile_batch_size(z) == min(H._TILE_BATCH_MAX, 2)
    assert H._tile_batch_size(z) < default


def test_video_vae_free_bytes_less_the_reserve(fraction, card):
    from core.inference import video_vae_untiled as U

    assert U._free_bytes(torch.device("cuda", 0)) == 21 * GIB
    fraction["value"] = 0.50
    assert U._free_bytes(torch.device("cuda", 0)) == 9 * GIB
    card["free"] = 4 * GIB  # inside the reserve: no room, never negative
    assert U._free_bytes(torch.device("cuda", 0)) == 0


def test_untiled_video_decode_stays_tiled_when_only_raw_free_fits(fraction, card):
    """Codex #14816 r8: the request guard passed on budgeted room; the untiled gate must not re-read raw free."""
    from core.inference import video_vae_untiled as U

    family = sorted(U._BYTES_PER_LATENT_PIXEL)[0]
    need = U.untiled_decode_bytes(family, (1, 16, 21, 60, 104), itemsize = 2)
    assert need
    gate = int(need * U._MARGIN + U._MARGIN_BYTES)
    card["free"], card["cached"] = gate + GIB, 0
    assert U._free_bytes(torch.device("cuda", 0)) >= gate
    fraction["value"] = 0.50
    assert U._free_bytes(torch.device("cuda", 0)) < gate


_COMMITTED = ["--offload-to-cpu", "--diffusion-fa", "--max-vram", "-1", "--stream-layers"]


class _Rt:
    def __init__(self, free, reserve):
        self.vram_free_probe = lambda: free
        self.vram_reserve_probe = lambda: reserve


def test_h3_tiny_budgeted_room_is_refused_not_handed_to_sd_cpp_raw_reading():
    """Codex #14816 r8: "-1" makes sd.cpp re-read RAW free; 0 GiB left inside the budget must not fall back to it."""
    import core.inference.video as V
    from core.inference.video_minimax_h3 import H3NativeMemoryRefusal

    for free in (0, 1 * GIB, {"cuda0": 2 * GIB}):
        with pytest.raises(H3NativeMemoryRefusal, match = "VRAM Budget"):
            V._h3_budgeted_launch_flags(_Rt(free, 12 * GIB), _COMMITTED)
    assert V._h3_budgeted_launch_flags(_Rt(0, 0), _COMMITTED) == _COMMITTED


def test_h3_small_or_unread_budgeted_room_spares_the_reserve_in_sd_cpp():
    import core.inference.video as V

    out = V._h3_budgeted_launch_flags(_Rt(2 * GIB, 12 * GIB), _COMMITTED)
    assert out[3] == "-13.00" and out[:3] == _COMMITTED[:3] and out[4:] == _COMMITTED[4:]
    assert V._h3_budgeted_launch_flags(_Rt(None, 12 * GIB), _COMMITTED)[3] == "-13.00"
    assert V._h3_budgeted_launch_flags(_Rt(None, int(2.501 * GIB)), _COMMITTED)[3] == "-3.51"
    assert V._h3_budgeted_launch_flags(_Rt(None, None), _COMMITTED) == _COMMITTED
    assert V._h3_budgeted_launch_flags(_Rt(None, 12 * GIB), _COMMITTED, cap = 4.5)[3] == "4.50"
    assert V._h3_budgeted_launch_flags(_Rt(10 * GIB, 12 * GIB), _COMMITTED)[3] == "8.00"


def test_h3_kill_switch_still_spares_the_reserve(monkeypatch):
    import core.inference.video as V
    from core.inference import sd_cpp_args as A

    monkeypatch.setenv(A.GRAPH_CUT_EXPLICIT_BUDGET_ENV, "0")
    assert V._h3_budgeted_launch_flags(_Rt(10 * GIB, 0), _COMMITTED) == _COMMITTED
    assert V._h3_budgeted_launch_flags(_Rt(10 * GIB, 12 * GIB), _COMMITTED)[3] == "-13.00"
    uma = ["--offload-to-cpu", "--max-vram", "-17.05"]
    assert V._h3_budgeted_launch_flags(_Rt(10 * GIB, 12 * GIB), uma) == uma


def test_h3_kill_switch_still_refuses_no_room_inside_the_budget(monkeypatch):
    """Codex #14816 r10: the kill switch keeps "-1" but must still refuse a card (nearly) inside the reserve."""
    import core.inference.video as V
    from core.inference import sd_cpp_args as A
    from core.inference.video_minimax_h3 import H3NativeMemoryRefusal

    monkeypatch.setenv(A.GRAPH_CUT_EXPLICIT_BUDGET_ENV, "0")
    for free in (0, 1 * GIB, {"cuda0": 512 * 1024**2}):
        with pytest.raises(H3NativeMemoryRefusal, match = "VRAM Budget"):
            V._h3_budgeted_launch_flags(_Rt(free, 12 * GIB), _COMMITTED)
    assert V._h3_budgeted_launch_flags(_Rt(2 * GIB, 12 * GIB), _COMMITTED)[3] == "-13.00"
    assert V._h3_budgeted_launch_flags(_Rt(None, 12 * GIB), _COMMITTED)[3] == "-13.00"
    assert V._h3_budgeted_launch_flags(_Rt(0, 0), _COMMITTED) == _COMMITTED


def test_h3_card_reserve_probe(fraction, monkeypatch):
    import contextlib

    import core.inference.video as V
    from utils import hardware
    from utils.hardware import gpu_query

    monkeypatch.setattr(gpu_query, "fresh_reads", contextlib.nullcontext)
    calls = []
    devices = [
        {"visible_ordinal": 0, "vram_total_gb": 24.0, "vram_used_gb": 4.0},
        {"visible_ordinal": 1, "vram_total_gb": 80.0, "vram_used_gb": 4.0},
    ]
    monkeypatch.setattr(
        hardware, "get_visible_gpu_utilization", lambda **k: calls.append(1) or {"devices": devices}
    )
    assert V._h3_card_reserve_bytes("cuda", 0) == 0 and not calls
    fraction["value"] = 0.50
    assert V._h3_card_reserve_bytes("cuda", 0) == 12 * GIB
    assert V._h3_card_reserve_bytes("cuda", 1) == 40 * GIB
    assert V._h3_card_reserve_bytes("cuda", None, single_card_only = True) == 12 * GIB
    assert V._h3_card_reserve_bytes("cpu", 0) == 0
    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", lambda **k: {"devices": []})
    assert V._h3_card_reserve_bytes("cuda", 0) is None


def _two_card_h3_runtime(monkeypatch, devices):
    import contextlib
    import functools

    import core.inference.video as V
    from utils import hardware
    from utils.hardware import gpu_query

    monkeypatch.setattr(gpu_query, "fresh_reads", contextlib.nullcontext)
    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", lambda **k: {"devices": devices})
    return types.SimpleNamespace(
        vram_free_probe = functools.partial(
            V._h3_card_free_bytes, "cuda", None, single_card_only = True
        ),
        vram_reserve_probe = functools.partial(
            V._h3_card_reserve_bytes, "cuda", None, single_card_only = True
        ),
    )


@pytest.mark.parametrize("explicit", [True, False])
def test_unpinned_multi_card_h3_launch_keeps_every_cards_reserve(fraction, monkeypatch, explicit):
    """An unpinned H3 load on a 24 GiB card (22 free) beside a 64 GiB one."""
    import core.inference.video as V
    from core.inference import diffusion_memory as D
    from core.inference import sd_cpp_args

    D._LOAD_BUDGET_FRACTIONS.clear()
    monkeypatch.setattr(sd_cpp_args, "explicit_vram_budget_enabled", lambda: explicit)
    devices = [
        {"visible_ordinal": 0, "vram_total_gb": 24.0, "vram_used_gb": 2.0},
        {"visible_ordinal": 1, "vram_total_gb": 64.0, "vram_used_gb": 2.0},
    ]
    runtime = _two_card_h3_runtime(monkeypatch, devices)
    flags = ["--offload-to-cpu", "--max-vram", "-1", "--stream-layers"]
    assert V._h3_budgeted_launch_flags(runtime, flags) == flags
    assert runtime.vram_free_probe() is None
    fraction["value"] = 0.50
    assert runtime.vram_free_probe() == 10 * GIB
    launch = V._h3_budgeted_launch_flags(runtime, flags)
    value = float(launch[launch.index("--max-vram") + 1])
    for raw_free, budgeted_free in ((22 * GIB, 10 * GIB), (62 * GIB, 30 * GIB)):
        resolved = value * GIB if value >= 0 else raw_free + value * GIB
        assert value > 0, launch  # never a scalar negative spare on an unpinned multi-card load
        assert 0 < resolved <= budgeted_free, (launch, raw_free)
    devices[0]["vram_used_gb"] = 11.5
    from core.inference.video_minimax_h3 import H3NativeMemoryRefusal

    with pytest.raises(H3NativeMemoryRefusal):
        V._h3_budgeted_launch_flags(runtime, flags)


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize(
    "small_used, want",
    [
        (10.0, "1.00"),
        (10.5, "0.50"),
    ],
)
def test_unpinned_mixed_size_h3_small_reading_is_a_positive_cap(
    fraction, monkeypatch, explicit, small_used, want
):
    """A scalar negative spare from the smallest card's reserve must not under-budget the larger card."""
    import core.inference.video as V
    from core.inference import diffusion_memory as D
    from core.inference import sd_cpp_args

    D._LOAD_BUDGET_FRACTIONS.clear()
    monkeypatch.setattr(sd_cpp_args, "explicit_vram_budget_enabled", lambda: explicit)
    devices = [
        {"visible_ordinal": 0, "vram_total_gb": 24.0, "vram_used_gb": small_used},
        {"visible_ordinal": 1, "vram_total_gb": 64.0, "vram_used_gb": 2.0},
    ]
    runtime = _two_card_h3_runtime(monkeypatch, devices)
    flags = ["--offload-to-cpu", "--max-vram", "-1", "--stream-layers"]
    fraction["value"] = 0.50
    launch = V._h3_budgeted_launch_flags(runtime, flags)
    assert launch[2] == want, launch
    small_budgeted = (24.0 - small_used) * GIB - 12 * GIB
    for budgeted_free in (small_budgeted, 30 * GIB):
        assert 0 < float(launch[2]) * GIB <= budgeted_free


@pytest.mark.parametrize("explicit", [True, False])
def test_unpinned_multi_card_h3_refuses_unread_or_no_room(fraction, monkeypatch, explicit):
    """Unpinned multi-card under a lowered Budget."""
    import core.inference.video as V
    from core.inference import diffusion_memory as D
    from core.inference import sd_cpp_args
    from core.inference.video_minimax_h3 import H3NativeMemoryRefusal

    D._LOAD_BUDGET_FRACTIONS.clear()
    monkeypatch.setattr(sd_cpp_args, "explicit_vram_budget_enabled", lambda: explicit)
    devices = [
        {
            "visible_ordinal": 0,
            "vram_total_gb": 24.0,
            "vram_used_gb": 11.0,
        },  # exactly 1 GiB inside the budget
        {"visible_ordinal": 1, "vram_total_gb": 64.0, "vram_used_gb": 2.0},
    ]
    runtime = _two_card_h3_runtime(monkeypatch, devices)
    flags = ["--offload-to-cpu", "--max-vram", "-1", "--stream-layers"]
    fraction["value"] = 0.50
    with pytest.raises(H3NativeMemoryRefusal, match = "VRAM Budget"):
        V._h3_budgeted_launch_flags(runtime, flags)
    devices[0]["vram_used_gb"] = None  # unread usage on one card
    with pytest.raises(H3NativeMemoryRefusal, match = "VRAM Budget"):
        V._h3_budgeted_launch_flags(runtime, flags)
    fraction["value"] = vram_budget_settings.VRAM_FRACTION_DEFAULT
    assert V._h3_budgeted_launch_flags(runtime, flags) == flags


@pytest.mark.parametrize("explicit", [True, False])
def test_h3_single_card_refusal_checks_the_rounded_spare(fraction, monkeypatch, explicit):
    """The reserve rounds UP to the emitted spare, and the refusal checks that spare."""
    import contextlib
    import functools

    import core.inference.video as V
    from core.inference import diffusion_memory as D
    from core.inference import sd_cpp_args
    from core.inference.video_minimax_h3 import H3NativeMemoryRefusal
    from utils import hardware
    from utils.hardware import gpu_query

    MIB = 1 << 20
    D._LOAD_BUDGET_FRACTIONS.clear()
    monkeypatch.setattr(sd_cpp_args, "explicit_vram_budget_enabled", lambda: explicit)
    monkeypatch.setattr(gpu_query, "fresh_reads", contextlib.nullcontext)
    fraction["value"] = 0.899
    reserve = D.user_budget_reserve_bytes("cuda", 24 * GIB)
    devices = [{"visible_ordinal": 0, "vram_total_gb": 24.0, "vram_used_gb": 0.0}]
    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", lambda **k: {"devices": devices})
    runtime = types.SimpleNamespace(
        vram_free_probe = functools.partial(V._h3_card_free_bytes, "cuda", 0),
        vram_reserve_probe = functools.partial(V._h3_card_reserve_bytes, "cuda", 0),
    )
    flags = ["--offload-to-cpu", "--max-vram", "-1", "--stream-layers"]

    def launch_at(raw_free):
        devices[0]["vram_used_gb"] = 24.0 - raw_free / GIB
        return V._h3_budgeted_launch_flags(runtime, flags)

    with pytest.raises(H3NativeMemoryRefusal, match = "VRAM Budget"):
        launch_at(reserve + GIB + MIB)
    raw = reserve + GIB + 64 * MIB
    out = launch_at(raw)
    assert out[2] == "-3.43" and raw + float(out[2]) * GIB > 0


def test_encoder_prefetch_window_reads_free_less_the_reserve(fraction, card):
    from core.inference import diffusion_small_host as S

    card["free"] = 12 * GIB + 2 * GIB
    assert S._prefetch_budget_bytes(torch.device("cuda", 0)) == S._PREFETCH_MAX_BYTES
    fraction["value"] = 0.50
    assert S._prefetch_budget_bytes(torch.device("cuda", 0)) == 2 * GIB // 8
    card["free"] = 12 * GIB + 512 * 1024**2
    assert S._prefetch_budget_bytes(torch.device("cuda", 0)) == S._PREFETCH_MIN_BYTES


def test_encoder_prefetch_is_off_when_the_budget_leaves_under_the_floor(fraction, card):
    """#14816 r9: the 128 MiB floor must not hand the worker a group the budgeted room cannot hold."""
    from core.inference import diffusion_small_host as S

    fraction["value"] = 0.50
    for room in (0, 100 * 1024**2, S._PREFETCH_MIN_BYTES - 1):
        card["free"] = 12 * GIB + room
        assert S._prefetch_budget_bytes(torch.device("cuda", 0)) == 0
    card["free"] = 2 * GIB  # under the reserve: still nothing
    assert S._prefetch_budget_bytes(torch.device("cuda", 0)) == 0
    fraction["value"] = vram_budget_settings.VRAM_FRACTION_DEFAULT
    card["free"] = 64 * 1024**2
    assert S._prefetch_budget_bytes(torch.device("cuda", 0)) == S._PREFETCH_MIN_BYTES


def test_rocm_flash_probe_room_charges_the_reserve(fraction, card):
    from core.inference import diffusion_attention as att

    card["free"], card["cached"] = 4 * GIB, 0
    need = 2 * GIB
    assert att._rocm_flash_probe_fits("cuda:0", need) is True
    fraction["value"] = 0.90  # 2.4 GiB kept free
    assert att._rocm_flash_probe_fits("cuda:0", need) is False


def test_segmenter_card_room_charges_the_reserve(fraction, card):
    from core.inference import diffusion_segment as seg

    # driver free plus the 1 GiB reusable allocator cache, less the reserve
    assert seg._cuda_free_mib(0) == (21 * 1024, 24 * 1024)
    fraction["value"] = 0.50
    assert seg._cuda_free_mib(0) == (9 * 1024, 24 * 1024)
    card["free"] = 4 * GIB
    assert seg._cuda_free_mib(0) == (0, 24 * 1024)


def test_untiled_encode_shortfall_charges_the_reserve(fraction, card, monkeypatch):
    from core.inference import diffusion as D
    from core.inference import diffusion_vae_tiling as vt

    monkeypatch.setattr(vt, "_geometry", lambda vae: (8,))
    monkeypatch.setattr(vt, "_fused", lambda vae: False)
    monkeypatch.setattr(vt, "encode_mib_per_latent", lambda *a, **k: 1.0)
    param = types.SimpleNamespace(element_size = lambda: 2)
    vae = types.SimpleNamespace(encoder = types.SimpleNamespace(parameters = lambda: iter([param])))
    x = _cuda_z(1, 3, 64, 64)
    base = D._untiled_encode_shortfall_mib(vae, x)
    assert base is not None
    fraction["value"] = 0.50
    assert D._untiled_encode_shortfall_mib(vae, x) == base + 12 * 1024


def test_vae_prebuild_child_keeps_the_reserve_free(fraction, card, monkeypatch):
    from core.inference import diffusion_vae_prebuild as P

    spawned = []
    monkeypatch.setattr(P, "enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(P, "plan", lambda pipe: {"device": 0})
    monkeypatch.setattr(P, "_spawn", lambda job, logger: spawned.append(job) or True)
    ctx = types.SimpleNamespace(hit = False)
    card["free"] = P._MIN_FREE_BYTES + 1 * GIB
    assert P.maybe_kick(object(), ctx) is True
    fraction["value"] = 0.90  # 2.4 GiB kept free
    assert P.maybe_kick(object(), ctx) is False
    assert len(spawned) == 1


def test_ltx25_encoder_on_card_check_keeps_the_sent_share_free(card, monkeypatch):
    from core.inference import video_ltx25_te_worker as worker

    monkeypatch.setattr(worker, "_weights_mib", lambda m: 2 * 1024)
    card["free"] = (2 * 1024 + worker.ENCODE_MARGIN_MIB + 1024) << 20
    model = types.SimpleNamespace()
    assert worker._encode_device({"device": "cuda:0", "resident": False}, model) == "cuda:0"
    config = {"device": "cuda:0", "resident": False, "vram_reserve_share": 0.10}  # 2.4 GiB of 24
    assert worker._encode_device(config, model) == "cpu"


def test_ltx25_start_sends_the_reserve_share_only_below_the_default(fraction, monkeypatch):
    from core.inference import video_ltx25 as L

    started = []

    class _Client:
        def __init__(self, **kwargs):
            self.config = {}

        def start(self):
            started.append(dict(self.config))

    monkeypatch.setattr(L, "ensure_text_encoder_sidecar", lambda: "/nonexistent")
    monkeypatch.setattr(L, "Ltx25TextEncoderClient", _Client)
    kw = dict(
        repo = "Lightricks/LTX-2.5",
        revision = "abc",
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cuda:0",
        dtype = "bfloat16",
        resident = False,
    )
    L.start_text_encoder(**kw)
    assert "vram_reserve_share" not in started[-1]
    fraction["value"] = 0.75
    L.start_text_encoder(**kw)
    assert started[-1]["vram_reserve_share"] == pytest.approx(0.25)
