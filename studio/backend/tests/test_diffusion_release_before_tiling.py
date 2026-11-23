# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tile verdict with offload groups still resident streams them at the decode, not before the denoise.

Seen on a 15 GB T4 (Qwen-Image-2.1 fp16 edit, 1024x1024): streaming before the denoise left the decode tiled
(PSNR 52.6 dB vs untiled). CPU-only: decode figures, snapshot and group release are stubbed.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion as diffusion
from core.inference.diffusion_memory import (
    ACTIVATION_RUN,
    ACTIVATION_TILE,
    ImageActivationVerdict,
)

UNTILED_NEED = 9000  # MiB the whole-frame decode needs, past FREE_FRACTION
ENCODE_NEED = (
    3000  # MiB the untiled source encode needs, past FREE_FRACTION (fits the 6000 MiB free)
)


class _Card:
    """Free MiB the decode-time check sees, raised by a release."""

    def __init__(self, free_mib: int, resident_mib: int) -> None:
        self.free = free_mib
        self.resident = resident_mib
        self.releases: list = []
        self.restored = 0


@pytest.fixture
def card(monkeypatch):
    c = _Card(free_mib = 6000, resident_mib = 4283)

    def release(
        pipe,
        need,
        logger = None,
        reason = "",
        device = None,
        **kw,
    ):
        if c.resident <= 0:
            return None
        c.releases.append((need, reason))
        freed = min(need, c.resident)
        c.resident -= freed
        c.free += freed

        def restore():
            c.restored += 1
            c.resident += freed
            c.free -= freed

        return restore

    monkeypatch.setattr(diffusion, "resident_group_mib", lambda pipe: c.resident)
    monkeypatch.setattr(diffusion, "release_resident_groups", release)
    monkeypatch.setattr(
        diffusion, "_untiled_decode_shortfall_mib", lambda vae, z: UNTILED_NEED - c.free
    )
    c.encode_need = ENCODE_NEED
    monkeypatch.setattr(
        diffusion, "_untiled_encode_shortfall_mib", lambda vae, x: c.encode_need - c.free
    )
    monkeypatch.delenv(diffusion.RELEASE_BEFORE_TILING_ENV, raising = False)
    return c


def _decide(card):
    return diffusion._release_for_untiled_decode(object(), None, "z", "cuda:0", None)


def test_the_shortfall_streams_and_the_decode_runs_untiled(card):
    untiled, restore = _decide(card)
    # short 3000 MiB, plus a quarter for the budget's reserve and 256 MiB: 4006 of the 4283 resident
    assert untiled and card.releases == [(4006, "an untiled VAE decode")]
    assert card.resident == 277
    restore()
    assert card.resident == 4283 and card.restored == 1


def test_a_frame_that_already_fits_decodes_untiled_without_streaming(card):
    card.free = 9500
    assert _decide(card) == (True, None)
    assert card.releases == []


def test_a_short_first_release_streams_the_rest_once(card, monkeypatch):
    real = diffusion.release_resident_groups
    monkeypatch.setattr(
        diffusion,
        "release_resident_groups",
        lambda pipe, need, *a, **k: real(pipe, need if card.releases else need // 4, *a, **k),
    )
    untiled, restore = _decide(card)
    assert untiled and len(card.releases) == 2 and card.resident == 0
    restore()
    assert card.resident == 4283 and card.restored == 2


def test_still_short_after_every_group_pins_back_and_tiles(card, monkeypatch):
    monkeypatch.setattr(diffusion, "_untiled_decode_shortfall_mib", lambda vae, z: 1)
    assert _decide(card) == (False, None)
    assert card.resident == 4283 and card.restored >= 1


def test_a_shortfall_past_every_resident_group_streams_nothing(card):
    card.resident = 2000  # short 3000 MiB: even all of it leaves the decode tiled
    assert _decide(card) == (False, None)
    assert card.releases == []


def test_unknown_decode_figures_tile_without_streaming(card, monkeypatch):
    monkeypatch.setattr(diffusion, "_untiled_decode_shortfall_mib", lambda vae, z: None)
    assert _decide(card) == (False, None)
    assert card.releases == []


def test_a_broken_release_tiles_and_pins_back(card, monkeypatch):
    calls = {"n": 0}
    real = diffusion.release_resident_groups

    def flaky(pipe, need, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("copy stream broke")
        return real(pipe, need // 4, *a, **k)

    monkeypatch.setattr(diffusion, "release_resident_groups", flaky)
    assert _decide(card) == (False, None)
    assert card.resident == 4283 and card.restored == 1


def test_a_cancel_mid_release_pins_back_and_propagates(card, monkeypatch):
    real = diffusion._untiled_decode_shortfall_mib
    calls = {"n": 0}

    def cancel(vae, z):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(vae, z)

    monkeypatch.setattr(diffusion, "_untiled_decode_shortfall_mib", cancel)
    with pytest.raises(KeyboardInterrupt):
        _decide(card)
    assert card.resident == 4283 and card.restored == 1


def test_chained_restore_runs_the_later_release_first():
    order: list = []
    chained = diffusion._chain_restores(
        lambda: order.append("later"), lambda: order.append("first")
    )
    chained()
    assert order == ["later", "first"]
    only = lambda: None  # noqa: E731
    assert diffusion._chain_restores(None, only) is only
    assert diffusion._chain_restores(only, None) is only
    assert diffusion._chain_restores(None, None) is None


def test_generate_defers_the_tiling_to_the_decode():
    import inspect

    src = inspect.getsource(diffusion.DiffusionBackend.generate)
    # the request keeps its tiling engagement (and its refusals) and hands the tiling to the decode
    assert src.index("restore_vae = _tile_for_verdict(") < src.index("_defer_tiling_to_decode(")
    assert "release_for_untiled_decode" not in src  # nothing streams before the denoise


def test_the_untiled_decode_shortfall_uses_the_wide_tile_rule(monkeypatch):
    import core.inference.diffusion_vae_tiling as tiling

    monkeypatch.setattr(tiling, "_free_mib", lambda vae, z: (1000.0, 1.0))
    monkeypatch.setattr(tiling, "_geometry", lambda vae: (16, 32, 4))
    monkeypatch.setattr(tiling, "_fused", lambda vae: True)
    z = types.SimpleNamespace(shape = (1, 16, 1, 64, 64), dim = lambda: 5)
    vae = types.SimpleNamespace(use_slicing = True)
    per_latent = tiling.decode_mib_per_latent(16, True, "SimpleNamespace")
    want = 64 * 64 * per_latent / tiling.FREE_FRACTION - 1000
    assert abs(diffusion._untiled_decode_shortfall_mib(vae, z) - want) <= 1
    monkeypatch.setattr(tiling, "_free_mib", lambda vae, z: None)
    assert diffusion._untiled_decode_shortfall_mib(vae, z) is None


class _Vae:
    """A VAE whose decode and encode record the tiling and the resident MiB they ran with."""

    def __init__(
        self,
        card: _Card,
        fail: BaseException | None = None,
        encode_oom: int = 0,
    ) -> None:
        self.card = card
        self.use_tiling = True  # what engage_vae_tiling left
        self.fail = fail
        self.encode_oom = encode_oom
        self.decodes: list = []
        self.encodes: list = []
        self.encode_resident: list = []

    def decode(self, z):
        self.decodes.append((self.use_tiling, self.card.resident))
        if self.fail is not None:
            raise self.fail
        return z

    def encode(self, x):
        self.encodes.append(self.use_tiling)
        self.encode_resident.append(self.card.resident)
        if self.encode_oom:
            self.encode_oom -= 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return x

    def disable_tiling(self):
        self.use_tiling = False


def _priced(free: int) -> ImageActivationVerdict:
    # untiled: 6952 MiB of activations + the 2048 MiB fixed overhead = 9000 MiB
    return ImageActivationVerdict(
        ACTIVATION_TILE, needed_mib = 6952, tiled_needed_mib = 2952, budget_mib = free
    )


TARGET = types.SimpleNamespace(torch_device = "cuda:0")


def _armed(
    card,
    vae,
    verdict = None,
):
    pipe = types.SimpleNamespace(vae = vae)
    undone: list = []

    def restore_vae():
        undone.append(True)
        vae.disable_tiling()

    restore = diffusion._defer_tiling_to_decode(
        pipe, pipe, verdict or _priced(6000), TARGET, {}, restore_vae, None
    )
    return restore, undone


def test_the_denoise_keeps_the_groups_and_the_decode_streams_them_untiled(card):
    vae = _Vae(card)
    restore, undone = _armed(card, vae)
    assert vae.use_tiling is False and card.releases == []  # nothing streamed before the denoise
    assert vae.encode("x") == "x" and vae.encodes == [False]  # the edit's encode runs untiled too
    assert vae.decode("z") == "z"
    assert vae.decodes == [(False, 277)]  # the shortfall streamed, the decode untiled
    assert (
        card.resident == 4283 and card.restored == 1
    )  # pinned back as soon as the decode returned
    stats = vae.__dict__[diffusion.RELEASE_BEFORE_TILING_STATS]
    assert stats["untiled"] == 1 and stats["tiled"] == 0 and stats["released_mib"] == 4006
    restore()
    assert "decode" not in vae.__dict__ and "encode" not in vae.__dict__  # the wrappers are gone
    assert undone == [True] and vae.use_tiling is False


def test_every_decode_of_a_request_streams_and_pins_back_on_its_own(card):
    vae = _Vae(card)
    restore, _ = _armed(card, vae)
    vae.decode("a")
    vae.decode("b")
    assert vae.decodes == [(False, 277), (False, 277)] and card.restored == 2
    restore()


def test_a_decode_that_still_does_not_fit_tiles_as_before(card, monkeypatch):
    vae = _Vae(card)
    restore, _ = _armed(card, vae)
    monkeypatch.setattr(diffusion, "_untiled_decode_shortfall_mib", lambda vae, z: 1)
    vae.decode("z")
    assert vae.decodes == [(True, 4283)] and vae.use_tiling is False
    assert vae.__dict__[diffusion.RELEASE_BEFORE_TILING_STATS]["tiled"] == 1
    restore()


def test_a_failing_or_cancelled_decode_pins_the_groups_back(card):
    for exc in (RuntimeError("decode broke"), KeyboardInterrupt()):
        vae = _Vae(card, fail = exc)
        restore, _ = _armed(card, vae)
        with pytest.raises(type(exc)):
            vae.decode("z")
        assert card.resident == 4283 and vae.use_tiling is False
        restore()


def test_an_untiled_encode_out_of_memory_retries_tiled(card):
    vae = _Vae(card, encode_oom = 1)
    restore, _ = _armed(card, vae)
    assert vae.encode("x") == "x"
    assert vae.encodes == [False, True] and vae.use_tiling is False
    assert vae.__dict__[diffusion.RELEASE_BEFORE_TILING_STATS]["encode_oom"] == 1
    restore()


@pytest.mark.parametrize(
    "case",
    ["kill", "run", "overridden", "nothing_resident", "split", "not_engaged"],
)
def test_cases_that_do_not_stream_keep_the_request_tiling(card, monkeypatch, case):
    verdict = _priced(6000)
    vae = _Vae(card)
    if case == "kill":
        monkeypatch.setenv(diffusion.RELEASE_BEFORE_TILING_ENV, "0")
    elif case == "run":
        verdict = ImageActivationVerdict(ACTIVATION_RUN)
    elif case == "overridden":
        verdict = ImageActivationVerdict(ACTIVATION_TILE, overridden = True)
    elif case == "nothing_resident":
        card.resident = 0
    elif case == "split":
        monkeypatch.setattr(diffusion, "_vae_on_other_card", lambda pipe, target: "cuda:1")
    elif case == "not_engaged":
        vae.use_tiling = False
    pipe = types.SimpleNamespace(vae = vae)
    restore_vae = vae.disable_tiling
    out = diffusion._defer_tiling_to_decode(pipe, pipe, verdict, TARGET, {}, restore_vae, None)
    assert out is restore_vae and "decode" not in vae.__dict__ and "encode" not in vae.__dict__
    assert card.releases == []


@pytest.mark.parametrize("decode_free", [9500, 6000])
def test_a_conditioned_request_short_past_the_groups_still_decodes_untiled(card, decode_free):
    # Verdict short by more than all resident groups, but the frame alone fits at the decode: must decode untiled.
    verdict = ImageActivationVerdict(
        ACTIVATION_TILE, needed_mib = 23952, tiled_needed_mib = 2952, budget_mib = 6000
    )
    vae = _Vae(card)
    restore, _ = _armed(card, vae, verdict)
    card.free = decode_free
    assert vae.decode("z") == "z"
    assert vae.decodes == [(False, 4283 if decode_free >= UNTILED_NEED else 277)]
    assert card.releases == (
        [] if decode_free >= UNTILED_NEED else [(4006, "an untiled VAE decode")]
    )
    assert card.resident == 4283 and vae.use_tiling is False
    restore()


def test_a_frame_short_past_every_group_tiles_at_the_decode_with_nothing_streamed(card):
    card.resident = 2000  # the decode is short 3000 MiB: even every resident group leaves it tiled
    vae = _Vae(card)
    restore, _ = _armed(card, vae)
    assert vae.decode("z") == "z"
    assert vae.decodes == [(True, 2000)] and card.releases == [] and vae.use_tiling is False
    assert vae.__dict__[diffusion.RELEASE_BEFORE_TILING_STATS]["tiled"] == 1
    restore()


def test_an_oversized_encode_keeps_tiling_with_no_untiled_attempt(card):
    card.encode_need = 20000  # short 14000 MiB, past every resident group
    vae = _Vae(card)
    restore, _ = _armed(card, vae)
    assert vae.encode("x") == "x"
    assert vae.encodes == [True] and vae.use_tiling is False  # tiled straight away, as main
    assert card.releases == []
    stats = vae.__dict__[diffusion.RELEASE_BEFORE_TILING_STATS]
    assert stats["encode_tiled"] == 1 and stats["encode_untiled"] == 0 and stats["encode_oom"] == 0
    restore()


def test_a_fitting_encode_runs_untiled_without_streaming(card):
    vae = _Vae(card)
    restore, _ = _armed(card, vae)
    assert vae.encode("x") == "x"
    assert vae.encodes == [False] and vae.encode_resident == [4283] and card.releases == []
    stats = vae.__dict__[diffusion.RELEASE_BEFORE_TILING_STATS]
    assert stats["encode_untiled"] == 1 and stats["encode_tiled"] == 0
    restore()


def test_a_short_encode_streams_runs_untiled_and_pins_back(card):
    card.encode_need = (
        8000  # short 2000 MiB: plus a quarter and 256 MiB, 2756 of the 4283 resident stream
    )
    vae = _Vae(card)
    restore, _ = _armed(card, vae)
    assert vae.encode("x") == "x"
    assert vae.encodes == [False] and vae.encode_resident == [4283 - 2756]
    assert card.releases == [(2756, "an untiled VAE encode")]
    assert (
        card.resident == 4283 and card.restored == 1
    )  # pinned back as soon as the encode returned
    stats = vae.__dict__[diffusion.RELEASE_BEFORE_TILING_STATS]
    assert stats["encode_untiled"] == 1 and stats["encode_released_mib"] == 2756
    assert stats["released_mib"] == 0  # the decode's own counter is untouched
    restore()


def test_unknown_encode_figures_tile(card, monkeypatch):
    monkeypatch.setattr(diffusion, "_untiled_encode_shortfall_mib", lambda vae, x: None)
    vae = _Vae(card)
    restore, _ = _armed(card, vae)
    assert vae.encode("x") == "x"
    assert vae.encodes == [True] and card.releases == []
    restore()


@pytest.mark.parametrize("need", [None, 20000])
def test_on_wddm_no_untiled_encode_without_a_fit(card, monkeypatch, need):
    # WDDM spills an overrun to host RAM instead of raising: only a positive fit check may encode untiled.
    monkeypatch.setattr(diffusion.sys, "platform", "win32")
    monkeypatch.setattr(
        diffusion,
        "_untiled_encode_shortfall_mib",
        lambda vae, x: None if need is None else need - card.free,
    )
    vae = _Vae(card)
    restore, _ = _armed(card, vae)
    assert vae.encode("x") == "x"
    assert vae.encodes == [True] and card.releases == []
    restore()


def test_the_untiled_encode_shortfall_uses_the_measured_encode_figures(monkeypatch):
    import core.inference.diffusion_vae_tiling as tiling

    monkeypatch.setattr(tiling, "_geometry", lambda vae: (16, 32, 4))
    monkeypatch.setattr(tiling, "_fused", lambda vae: False)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (1000 * 2**20, 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 0)
    param = types.SimpleNamespace(element_size = lambda: 2)
    vae = type("AutoencoderKLQwenImage21", (), {})()
    vae.encoder = types.SimpleNamespace(parameters = lambda: iter([param]))
    x = types.SimpleNamespace(
        shape = (1, 4, 1, 1024, 1024), dim = lambda: 5, device = types.SimpleNamespace(type = "cuda")
    )
    want = 64 * 64 * 0.6 / tiling.FREE_FRACTION - 1000
    assert abs(diffusion._untiled_encode_shortfall_mib(vae, x) - want) <= 1
    assert tiling.encode_mib_per_latent(16, True, "AutoencoderKLQwenImage21") == 0.3
    assert tiling.encode_mib_per_latent(8, False, "AutoencoderKL") == tiling.decode_mib_per_latent(
        8, False, "AutoencoderKL"
    )
    assert tiling.encode_mib_per_latent(12, False, "SomeVae") is None  # unknown class and ratio
    x.device = types.SimpleNamespace(type = "cpu")
    assert diffusion._untiled_encode_shortfall_mib(vae, x) is None


def test_the_untiled_decode_check_keeps_the_budget_caps(monkeypatch):
    # The whole-frame check must respect the ROCm Qwen-Image-2.1 floor cap.
    import core.inference.diffusion_vae_tiling as tiling

    monkeypatch.delenv(tiling.MAX_TILE_ENV, raising = False)
    monkeypatch.setattr(tiling, "_free_mib", lambda vae, z: (10**7, 1.0))
    monkeypatch.setattr(tiling, "_geometry", lambda vae: (16, 32, 4))
    monkeypatch.setattr(tiling, "_fused", lambda vae: False)
    vae = type("AutoencoderKLQwenImage21", (), {"use_slicing": True})()
    z = types.SimpleNamespace(
        shape = (1, 16, 1, 64, 64), dim = lambda: 5, device = types.SimpleNamespace(type = "cuda")
    )
    monkeypatch.setattr(tiling, "is_rocm_torch", lambda torch: False)
    assert diffusion._untiled_decode_shortfall_mib(vae, z) <= 0  # CUDA: fits, untiled
    monkeypatch.setattr(tiling, "is_rocm_torch", lambda torch: True)
    assert tiling.decode_tile_budget(vae, z) == tiling.TILE_LATENTS**2
    assert diffusion._untiled_decode_shortfall_mib(vae, z) is None  # ROCm: keep the floor tiles
    small = types.SimpleNamespace(shape = (1, 16, 1, 32, 32), dim = lambda: 5, device = z.device)
    assert diffusion._untiled_decode_shortfall_mib(vae, small) <= 0  # the frame is the floor tile
    monkeypatch.setattr(tiling, "is_rocm_torch", lambda torch: False)
    monkeypatch.setenv(tiling.MAX_TILE_ENV, "40")  # an explicit max tile sizes tiled decodes only
    assert diffusion._untiled_decode_shortfall_mib(vae, z) <= 0


def test_the_untiled_encode_check_prices_the_fp32_encode_of_an_fp16_decode_vae(monkeypatch):
    # fp32_encode upcasts the fp16 SDXL encoder: price the encode at fp32.
    import core.inference.diffusion_vae_tiling as tiling

    monkeypatch.setattr(tiling, "_geometry", lambda vae: (8, 32, 4))
    monkeypatch.setattr(tiling, "_fused", lambda vae: False)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (1000 * 2**20, 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 0)
    param = types.SimpleNamespace(element_size = lambda: 2)
    vae = type("AutoencoderKL", (), {})()
    vae.encoder = types.SimpleNamespace(parameters = lambda: iter([param]))
    x = types.SimpleNamespace(
        shape = (1, 3, 1024, 1024), dim = lambda: 4, device = types.SimpleNamespace(type = "cuda")
    )
    per_latent = tiling.encode_mib_per_latent(8, False, "AutoencoderKL")
    fp16 = 128 * 128 * per_latent / tiling.FREE_FRACTION - 1000
    fp32 = 128 * 128 * per_latent * 2 / tiling.FREE_FRACTION - 1000
    assert abs(diffusion._untiled_encode_shortfall_mib(vae, x) - fp16) <= 1
    vae._unsloth_fp16_decode = True
    assert abs(diffusion._untiled_encode_shortfall_mib(vae, x) - fp32) <= 1


@pytest.fixture
def graph_held_card(monkeypatch):
    """1000 MiB driver free plus 600 MiB cached, 500 MiB of it a CUDA graph pool's free blocks."""
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_memory as dm

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (1000 * 2**20, 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 600 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dm, "user_budget_reserve_bytes", lambda device, total: 0)
    monkeypatch.setattr(dm, "_graph_held_cache_bytes", lambda device = None: 500 * 2**20)
    return types.SimpleNamespace(type = "cuda", index = 0)


def test_the_untiled_decode_check_does_not_count_graph_held_cache(monkeypatch, graph_held_card):
    import core.inference.diffusion_vae_tiling as tiling

    monkeypatch.setattr(tiling, "_geometry", lambda vae: (8, 32, 4))
    param = types.SimpleNamespace(element_size = lambda: 2)
    vae = types.SimpleNamespace(decoder = types.SimpleNamespace(parameters = lambda: iter([param])))
    z = types.SimpleNamespace(shape = (1, 4, 1, 1), dim = lambda: 4, device = graph_held_card)
    mib, _ = tiling._free_mib(vae, z)
    assert abs(mib - 1100) < 1
    mib, _ = tiling._free_mib(vae, z, cached = False)
    assert abs(mib - 1000) < 1


def test_the_untiled_encode_check_does_not_count_graph_held_cache(monkeypatch, graph_held_card):
    import core.inference.diffusion_vae_tiling as tiling

    monkeypatch.setattr(tiling, "_geometry", lambda vae: (16, 32, 4))
    monkeypatch.setattr(tiling, "_fused", lambda vae: False)
    param = types.SimpleNamespace(element_size = lambda: 2)
    vae = type("AutoencoderKLQwenImage21", (), {})()
    vae.encoder = types.SimpleNamespace(parameters = lambda: iter([param]))
    x = types.SimpleNamespace(shape = (1, 4, 1, 1024, 1024), dim = lambda: 5, device = graph_held_card)
    want = 64 * 64 * 0.6 / tiling.FREE_FRACTION - 1100
    assert abs(diffusion._untiled_encode_shortfall_mib(vae, x) - want) <= 1
