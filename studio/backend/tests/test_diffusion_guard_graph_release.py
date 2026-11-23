# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The generate guard drops Studio's own idle CUDA graphs when only their pool stands between a request and a fit.

The guard credits reserved-but-unallocated allocator memory as free, except the CUDA graph pools: only their graphs
can reuse those blocks. On a 24 GB L4, the int8 FLUX.3 Klein 6-image 2048 px edit read 7.60 GB free while the graphs
the previous 4-image render captured held about 3.3 GiB of pool, and was refused for 0.16 GB. A request that fits, a
request that still would not fit, and the kill switch all keep their graphs.
"""

from __future__ import annotations

import inspect
import logging
import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference import diffusion as diff
from core.inference.diffusion_families import _FAMILIES
from core.inference.diffusion_memory import DeviceMemory, calibrated_image_activation

MP = 1024 * 1024
GIB = 1024 * MP
L4_TOTAL = 22_563
_FAM = {f.name: f for f in _FAMILIES}


def _kwargs(
    free,
    refs = 6,
    allow_oversized = False,
    total = L4_TOTAL,
):
    family = "flux.3-klein"
    return dict(
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", free, total),
        width = 2048,
        height = 2048,
        family = family,
        condition_pixels = int(refs * MP * _FAM[family].condition_pixel_weight),
        vae_tile_side = dm.DEFAULT_VAE_TILE_SIDE,
        vae_sliced = True,
        calibrated_placement = True,
        measured_activation = calibrated_image_activation(family, max_speed = False),
        conditioned_denoise_mib = dm.flux3_conditioned_denoise_mib(
            width = 2048,
            height = 2048,
            references = refs,
            batch_size = 1,
            dense = False,
            capability = (8, 9),
            rocm = False,
        ),
        allow_oversized = allow_oversized,
    )


def test_l4_six_image_edit_refused_at_recorded_free_reading():
    assert dm.image_activation_verdict(**_kwargs(7_600)).action == dm.ACTIVATION_REFUSE


def test_release_when_held_pool_makes_it_fit():
    held = int(3.3 * GIB)
    assert dm.graph_release_fits(_kwargs(7_600), lambda: held) == held


def test_allow_oversized_request_still_releases_when_short():
    held = int(3.3 * GIB)
    assert dm.graph_release_fits(_kwargs(7_600, allow_oversized = True), lambda: held) == held


@pytest.mark.parametrize("held", [0, -5, 64 * MP])
def test_no_release_when_held_pool_cannot_make_it_fit(held):
    assert dm.graph_release_fits(_kwargs(7_600), lambda: held) == 0


def test_no_release_and_no_snapshot_when_request_fits():
    def boom():
        raise AssertionError("held bytes read on a request that fits")

    assert dm.graph_release_fits(_kwargs(16_000), boom) == 0
    assert dm.graph_release_fits(_kwargs(16_000, refs = 0), boom) == 0


def test_no_release_when_free_unknown():
    kw = _kwargs(7_600)
    kw["device_memory"] = DeviceMemory("cuda", "cuda", "discrete_vram", None, L4_TOTAL)
    assert dm.graph_release_fits(kw, lambda: 8 * GIB) == 0


def test_credit_capped_at_card_total():
    # a pool reading above the card total never credits past the card
    kw = _kwargs(1_000, refs = 6, total = 4_000)
    assert dm.graph_release_fits(kw, lambda: 64 * GIB) == 0


def test_cuda_graph_held_bytes_reads_current_card(monkeypatch):
    seen = []
    monkeypatch.setattr(dm, "_graph_held_cache_bytes", lambda d: seen.append(d) or 123)
    assert dm.cuda_graph_held_bytes(1) == 123
    assert seen == [1]
    monkeypatch.setattr(dm, "_graph_held_cache_bytes", lambda d: -7)
    assert dm.cuda_graph_held_bytes(0) == 0


class _Harness:
    def __init__(
        self,
        monkeypatch,
        held = int(3.3 * GIB),
        handles = ("g",),
    ):
        self.resets = []
        self.rebuilds = 0
        monkeypatch.setattr(diff.cuda_graph, "reset_all", lambda h: self.resets.append(tuple(h)))
        monkeypatch.setattr(diff, "cuda_graph_held_bytes", lambda: held)
        self.state = types.SimpleNamespace(
            cuda_graphs = tuple(handles),
            pipe = types.SimpleNamespace(_unsloth_cuda_graphs = tuple(handles)),
        )
        self.log = logging.getLogger("test.graph_release")

    def rebuild(self):
        self.rebuilds += 1
        return {"rebuilt": True}

    def run(self, kwargs):
        return diff._release_graphs_for_guard(
            self.state, {**kwargs, "logger": self.log}, self.rebuild, self.log
        )


def test_guard_drops_graphs_then_rereads(monkeypatch):
    monkeypatch.delenv(dm.GUARD_RELEASE_GRAPHS_ENV, raising = False)
    h = _Harness(monkeypatch)
    assert h.run(_kwargs(7_600)) == {"rebuilt": True}
    assert h.resets == [("g",)]  # one handle, deduplicated across state and pipe
    assert h.rebuilds == 1


def test_guard_keeps_graphs_when_request_fits(monkeypatch):
    h = _Harness(monkeypatch)
    kw = _kwargs(16_000)
    assert h.run(kw)["device_memory"] is kw["device_memory"]
    assert h.resets == [] and h.rebuilds == 0


def test_guard_keeps_graphs_when_release_is_not_enough(monkeypatch):
    h = _Harness(monkeypatch, held = 64 * MP)
    h.run(_kwargs(7_600))
    assert h.resets == [] and h.rebuilds == 0


def test_guard_without_graphs_does_nothing(monkeypatch):
    h = _Harness(monkeypatch, handles = ())
    h.run(_kwargs(7_600))
    assert h.resets == [] and h.rebuilds == 0


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_kill_switch_keeps_graphs(monkeypatch, value):
    monkeypatch.setenv(dm.GUARD_RELEASE_GRAPHS_ENV, value)
    h = _Harness(monkeypatch)
    kw = _kwargs(7_600)
    assert h.run(kw)["device_memory"] is kw["device_memory"]
    assert h.resets == [] and h.rebuilds == 0
    # the refusal itself still stands
    with pytest.raises(dm.ImageActivationShortfallError):
        dm.raise_on_image_activation_shortfall(**kw)


def test_failed_release_falls_back_to_the_plain_reading(monkeypatch):
    h = _Harness(monkeypatch)

    def broken(_):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(diff.cuda_graph, "reset_all", broken)
    kw = _kwargs(7_600)
    assert h.run(kw)["device_memory"] is kw["device_memory"]
    assert h.rebuilds == 0


def test_generate_guard_releases_before_the_check():
    src = inspect.getsource(diff)
    call = src.index("guard_kwargs = _release_graphs_for_guard(")
    check = src.index("verdict = raise_on_image_activation_shortfall(**guard_kwargs)")
    assert call < check
    assert src.count("_release_graphs_for_guard(") == 2  # the definition and the generate call
